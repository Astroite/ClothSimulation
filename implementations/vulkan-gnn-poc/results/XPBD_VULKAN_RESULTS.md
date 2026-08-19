# S2' 实测结果：非结构化 Jacobi XPBD 落地 Vulkan

执行 [`plans/gnn/gnn-xpbd-v2.md`](../../../plans/gnn/gnn-xpbd-v2.md) 的 S2'。
测试机 RTX 4060 Ti（锁 2700 MHz），torch 2.10.0+cu128。
Python 侧的质量结论见 [`GATE_G0_RESULTS.md`](GATE_G0_RESULTS.md)。

## 结论

**kernel 正确性通过（与 Python 逐步对拍，10 步 max_abs_error 1.68e-3，mean 8.8e-7，
Khronos 同步校验干净）。但成本比方案的估计高 3.4 倍：实测 9.44 µs/迭代，k=128 = 1.21 ms，
方案估的是 0.36 ms。**

**诊断清楚了：不是 dispatch 受限，也不是带宽受限，是 occupancy 受限**（第三节）。
CH10032 只有 1377 个顶点 = **11 个 workgroup**，而这块卡有 34 个 SM。修法在本仓库里已有先例。

| 量 | v2 §2.3 估计 | 实测 | 差 |
| --- | ---: | ---: | ---: |
| 每迭代 | 2.8 µs | **9.44 µs** | 3.4× |
| k=128 | 0.36 ms | **1.208 ms** | 3.4× |
| k=128 相对整个 GNN（0.924 ms） | 0.39× | **1.31×** | — |
| k=128 折算 GNN block（0.0646 ms/block） | 5.6 个 | **18.7 个** | — |

**这直接否掉了 S7 原本的交易。** "砍 6 个 block 腾出 0.36 ms 买 k=128" 不成立 ——
k=128 要 18.7 个 block，比整个 12 block 的 processor 还多。当前可行的等预算点是
**k=32（0.302 ms ≈ 4.7 个 block）**，而 G0 实测 jacobi k=32 的 score 是 0.775，k=128 是 0.532。

---

## 一、正确性：与 Python 逐步对拍

`tools/run_tinyhood_reference.py --xpbd-asset` 产出带 XPBD 的 golden，
`verify_hood.ps1 -Xpbd` 比对，阈值沿用既有的（step-0 max ≤ 2e-4 / mean ≤ 2e-5，10 步 max ≤ 2e-3）。
**没有为 XPBD 放宽阈值。**

`ch10032_tpose`，`student32x12_r1`，k=128（`xpbd128_verify.json`）：

| 量 | 值 |
| --- | ---: |
| 10 步 max_abs_error | **1.684e-3**（阈值 2e-3） |
| 10 步 mean_abs_error | **8.85e-7** |
| step 0 max / mean | 1.19e-7 / 1.05e-8 |
| first_world_edge_mismatches | 0 |
| Khronos 同步校验 | 干净（无 `SYNC-HAZARD` / `VUID-`） |

逐步误差有一个特征形状，值得解释清楚，否则将来看到 1.68e-3 对纯 GNN 路径的 8.48e-6 会以为
kernel 坏了：

| step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | **8** | 9 | 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| max | 1.2e-7 | 1.4e-7 | 2.6e-7 | 3.2e-7 | 3.8e-7 | 3.9e-7 | 5.7e-7 | **1.7e-3** | 9.3e-4 | 1.2e-3 |

前 7 步稳定在 1e-7 量级 —— 128 次迭代累积下来只有浮点末位差，说明 kernel 与 Python 是同一套
算术。第 8 步跳到 1.7e-3，而 mean 只有 3.5e-6，所以是**极少数顶点**的离散跳变。

原因实测确认（Python 侧统计每步落在分支边界 1e-7 内的数量）：

| step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 约束 \|residual\| < 1e-7 | 81 | 78 | 83 | 88 | 90 | 98 | 99 | 99 | 95 | 90 |
| 接触 \|signed\| < 1e-7 | 301 | 332 | 326 | 327 | 338 | 335 | 326 | 323 | 329 | 333 |

**每一步都有约 90 条约束和约 330 个接触精确坐在分支边界上**，而这是**结构性的、不是巧合**：

- one-sided 约束的 `max(residual, 0)` 把满足的边驱动到 `residual == 0`，那里就是分支边界；
- 接触投影把穿模顶点驱动到 `signed == contactOffset`，同样正好是 `signed < 0` 的边界。

也就是说这两个饱和投影的**不动点就落在自己的判据边界上**。哪一侧被取到取决于浮点末位，
CPU 与 GPU 偶尔取不同侧，那个顶点这一步的修正量就差一整跳，之后被后续步放大。

**这是饱和投影固有的，不是实现缺陷。** 两侧都确定性、逐位可复现（Vulkan 侧确定性，
Python golden 在 CPU 上开了 `use_deterministic_algorithms(True)`），所以 1.684e-3 是一个
固定值而不是随机抽样，测试不会 flake。但**它离 2e-3 阈值只有 16% 余量**，
迭代数或场景一换就可能顶破 —— 届时应该按这一节的机制判断，而不是直接放宽阈值。

### k = 0 必须是逐位无操作

开着 `--hood-xpbd` 但 k=0，对纯 GNN 的 golden：**max_abs_error `8.478760719e-06`**，
与 `STUDENT_STABILITY_ROUND2.md` §6 记录的纯 GNN 路径数字**完全相同**。短路正确。

---

## 二、四场景 300 步稳定性（渲染器内，非 Python）

`benchmark_hood_static.ps1 -Xpbd -XpbdIterations 128`，`student32x12_r1`。
`edge_length_ratio` 与 `triangle_area_ratio` 都是**对 authored rest 网格**的比值。

| 场景 | XPBD | 步数 | 结构 | edge p95 | edge max | flipped | degenerate | max 位移 m |
| --- | --- | ---: | :-: | ---: | ---: | ---: | ---: | ---: |
| `ch10032_tpose` | off | 314 | ✅ | 1.224 | 1.784 | 0.0105 | 0.00233 | 0.247 |
| `ch10032_tpose` | **on** | 314 | ✅ | 1.345 | 4.527 | 0.0506 | 0.00272 | 0.268 |
| `ch10032_sprint` | off | 162 | ❌ | 3.355 | **55.39** | 0.2521 | 0.00350 | 3.130 |
| `ch10032_sprint` | **on** | 162 | ❌ | **2.373** | **7.21** | **0.1300** | 0.01907 | 3.130 |
| `hml_001962` | off | 312 | ✅ | 1.446 | 3.890 | 0.0451 | 0.00350 | 0.511 |
| `hml_001962` | **on** | 312 | ✅ | **1.158** | **3.239** | 0.0595 | **0.00039** | 0.462 |
| `hood_grid64` | off | 314 | ❌ | 3.908 | **328.97** | 0.4893 | 0.00920 | 7.729 |
| `hood_grid64` | **on** | 314 | ✅ | **1.077** | **1.134** | **0.0000** | **0.00000** | 0.360 |

**`hood_grid64` 是完全的挽救**：不接 XPBD 时结构判定失败、最大边长比 **329**、48.9% 三角形翻面、
顶点飞出 7.7 m；接上后 p95 1.077 / max 1.134 / **零翻面** / 位移 0.36 m。
这与 Python 侧 score 1.88 → 0.16 是同一件事，**渲染器独立复现了它**。

三个必须如实说的地方：

1. **`ch10032_sprint` 仍然结构失败。** XPBD 把最大边长比从 55.4 压到 7.2（7.7×）、翻面从
   25.2% 到 13.0%，但 162 步内没有阻止崩坏，而且**退化三角形反而变差 5.4 倍**
   （0.0035 → 0.0191）。原因清楚：距离约束管不住三角形塌成零面积 —— 三条边都可以保持长度而
   三角形被压平。要修这个需要面积约束或二面角约束，本轮的 bend 约束是 2-hop 距离形式，管不了。
2. **`ch10032_tpose` 在这套几何指标上变差了**（edge max 1.784 → 4.527，翻面 0.0105 → 0.0506），
   而 Python 的 teacher-relative score 是变好的（0.2410 → 0.2186）。两者不矛盾：这里的比值是
   **对 rest 的**，而 `GATE_G0_RESULTS.md` §1 实测 tpose 蒙皮初态对 rest 的 p95 已经是 **1.890**。
   XPBD 的目标是 **teacher 标定长度**，把边拉向 teacher 就等于拉离 rest。
   **这套指标在 tpose 上本身就是坏尺子。**
3. 更重要的是这条**内部一致性检查**：XPBD 在 rest-relative 指标上的收益，
   **正好按各场景 rest 的可信度排序** ——

   | 场景 | 蒙皮初态对 rest 的 p95（§1） | XPBD 对 rest-relative 指标 |
   | --- | ---: | --- |
   | `hood_grid64` | **1.000**（完全自洽） | 极大改善 |
   | `hml_001962` | 1.081 | 明显改善 |
   | `ch10032_sprint` | 2.025 | 部分改善 |
   | `ch10032_tpose` | 1.890 | **变差** |

   这是对 §1「`cloth_rest` 不是可用参考」那个结论的独立佐证。

> **顺带发现一个与本轮无关的既有问题**：`ch10032_sprint` 在 `--hood-benchmark-samples 300`
> 下会以 `exitFatal(..., -1)` 失败（"did not collect the requested number of samples"），
> **关掉 XPBD 也一样失败**，150 samples 正常。所以 sprint 那两行是 162 步而非 314 步。
> 这是 benchmark harness 在动画场景上的既有上限，不是本轮引入的。

---

## 三、成本：9.44 µs/迭代，且是 occupancy 受限

`ch10032_tpose`，10 预热 / 60 采样，min_ms：

| k | XPBD min ms | XPBD p95 ms | 每迭代 | 总 min ms | 总 p95 ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.0758 | 0.0778 | **9.472 µs** | 1.024 | 1.040 |
| 32 | 0.3021 | 0.3052 | **9.440 µs** | 1.253 | 1.267 |
| 128 | 1.2080 | 1.2136 | **9.437 µs** | 2.154 | 2.178 |
| 256 | 2.4146 | 2.6235 | **9.432 µs** | 3.349 | 3.568 |

每迭代成本在 8→256 之间平到小数点后两位，所以它是一个干净的固定单价，k 是严格线性的。

### 不是 dispatch 受限：9.44 µs 远高于 2.8 µs 的单价

v2 §2.3 把每迭代成本等同于一次 dispatch 的 2.8 µs。实测高 3.4 倍，所以启动开销只占约 30%，
kernel 本身在做主要的事。

### 也不是带宽/工作量受限：`hood_grid64` 工作量 2.3 倍却更快

| 场景 | 顶点 | slot 宽度 | 总 slot | workgroup | 每迭代 | 每 slot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ch10032` | 1,377 | 18 | 24,786 | **11** | **9.437 µs** | 0.381 ns |
| `hood_grid64` | 4,096 | 14 | 57,344 | **32** | **6.720 µs** | **0.117 ns** |

**grid64 的 slot 工作量是 2.31 倍，每迭代却快 1.40 倍，单 slot 成本低 3.3 倍。**
工作量增加而时间减少，就排除了工作量受限与带宽受限。

**剩下的解释是 occupancy。** CH10032 的 1377 个顶点按 128 线程一组只有 **11 个 workgroup**，
而这块 4060 Ti 有 **34 个 SM** —— 23 个 SM 全程空闲，而那 11 个里每个 lane 要串行走 18 个
slot，每个 slot 是两次分散的 `positionIn[]` 读取，前后依赖。这是延迟暴露，不是吞吐不足。
grid64 的 32 个 workgroup 基本填满了卡，多出来的工作几乎免费。

### 这个病本仓库已经诊断并修过一次

`hood_world_nearest.comp` 的文件注释记的是同一件事，逐字引用：

> The work was never the problem; the occupancy was.

它的原形态是"一个线程一个 cloth 顶点，串行扫 4096 个 proxy"，改成"一个 128-lane workgroup
一个 cloth 顶点，每 lane 只走 `proxyCount/128` 个"，并用组内确定性归约保持逐位一致。

**同样的改法适用于这里**：一个 workgroup 一个顶点，lane 分摊 18 个 slot，组内做确定性归约。
workgroup 数从 11 涨到 1377。按 grid64 实测的单 slot 成本外推：

```
24,786 slot × 0.117 ns ≈ 2.9 µs/迭代  →  k=128 ≈ 0.37 ms
```

**这正好落回 v2 §2.3 原来估的 0.36 ms。** 所以那个估计不是错的，它估的是一个
occupancy 正常的 kernel；当前 kernel 的问题是没做到那一点。

> 这是外推，不是实测。归约需要保持确定性（`hood_world_nearest.comp` 有成例），
> 而每 lane 只处理 1–2 个 slot 时组内归约本身的开销会占比更高，所以 0.37 ms 是乐观下界。

---

## 四、实现要点（读 kernel 之前先看这几条）

### 融合成单 dispatch 的代价与它为什么安全

照搬 `real_scene/xpbd.py::_apply_jacobi` 需要**两次 dispatch/迭代**（先逐约束算全部 Δλ，
再逐顶点 gather），因为顶点看不到邻接约束的结果直到约束 pass 写完。
`_apply_fused` 改成每顶点自己重算它 ~11 条关联约束的 Δλ，于是 **1 dispatch/迭代**。

代价是 **λ 按 (vertex, slot) 冗余存两份**。两份不漂移，靠三件事：

1. 两端都按存储顺序算 `pairs[c].x - pairs[c].y`，不是"我减你"，所以差向量与它的长度在两端是
   同一批浮点数；
2. **`weight_sum` 必须 bake 后读取，不能在 kernel 里现加 `w_a + w_b`** —— 两端的加法顺序相反，
   浮点加法不可交换到逐位，这一条错了两份 λ 就会分开；
3. 端点相关的符号是 `signs`，在 λ 更新之后作用于修正量，不在更新式里面。

`tests/test_xpbd.py::FusedPortTests` 把这三条固定住了 ——
`test_fused_sweep_matches_jacobi` 在真实网格上、两种 compliance 量级、两种残差符号下要求
**与两趟版逐位相同**；`test_fused_lambda_copies_stay_equal_across_endpoints` 迭代 8 次后
逐个约束核对两份 λ 相等。**这两个测试是在写 HLSL 之前先跑通的。**

### v2 §5.2 说要新增速度 pass —— 这条在本路径上不成立

蒸馏路径没有显式速度状态。`hood_integrate.comp:47` 写的
`clothPrevious = firstStep ? predicted : effective`，其中 `effective` 恰好等于 Python 侧
`tools/train_student.py:340` 的 `next_previous = graph.effective_position`。
XPBD 的修正量通过下一步的 `effective_position` 自动传播，**step > 0 完全不需要动 `clothPrevious`**。

只有 settle 步需要处理：那一步 `clothPrevious` 被写成**未修正**的预测，而
`tools/compare_student_stability.py:72-76` 的 `previous = corrected if step == 0` 要求它是修正后的。
runtime 用一次 `vkCmdCopyBuffer(clothPosition → clothPrevious)` 解决，**不是一个新 kernel**。

### 不需要惯性缓冲

G0 胜出的 `standard` 模式从 `x_gnn` 起步，而 `_apply_fused` / `_constraint_step` 全程只读
`current`，**从不引用 `inertial`** —— 只有被 G0 否掉的 `warmstart` / `nowarm` 用它。
所以 x̃ 不必传进 GPU，也不必新增 buffer。

### Jacobi 必须 ping-pong

所有顶点要读同一份位置，就地更新会变成不确定的部分 Gauss-Seidel。runtime 在
`clothPosition` 与 `xpbdScratch` 之间用两个 descriptor set 交替（照 `hood.edgeSets[ping]` 的成例）。
迭代数为奇数时结果落在 scratch 里，补一次 `vkCmdCopyBuffer`；k=128 时不触发，但**没有偷偷把
迭代数取偶**。

### 接触折进同一个 kernel

接触是逐顶点的（runtime 只保留最近代理，每顶点至多一个），写入目标互不相同、不需要累加，
所以它在同一个 dispatch 里做完，**没有额外的 dispatch/迭代**。
复用 `hood_integrate.comp` 已有的 `worldObstacle` / `proxyTarget` / `proxyNormal` 绑定。

开 XPBD 时 `hood_integrate.comp` 自己那次半平面投影会被**强制关掉** ——
否则一步里投影两次，也与 G0 测的配置不符（Python 的 `integrate()` 不做投影）。

### 资产是独立文件

新建 `<motion>.vxpbd`（`VXPBD001`），不动 `.vcloth2`。两个理由：标定长度依赖 teacher rollout
因而可能按动作分（实测可以不分，见 `GATE_G0_RESULTS.md` §11），而给 `.vcloth2` 加段会改它的
payload SHA-256，现有全部 golden 都钉在那个值上。

`slots` / `signs` 直接就是 `ConstraintSet` 的 padded `[V, K]` 表，**没有转成 CSR**：
CH10032 的 K=18 有 38% 是 padding，但既然是 occupancy 受限，那部分 lane 迭代不花钱，
而扁平 stride 让 kernel 与 Python 参考保持可逐行对照。

烘出来的规模：

| 场景 | 约束 | stretch | bend | 顶点 | slot 宽 | 全 pin 的死约束 | 文件 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hml_001962` | 7,682 | 3,947 | 3,735 | 1,377 | 18 | 72 | 360 KB |
| `ch10032_tpose` | 7,682 | 3,947 | 3,735 | 1,377 | 18 | 72 | 360 KB |
| `ch10032_sprint` | 7,682 | 3,947 | 3,735 | 1,377 | 18 | 72 | 360 KB |
| `hood_grid64` | 27,783 | 16,002 | 11,781 | 4,096 | 14 | 63 | 1,039 KB |

compliance **不烘**（`kind` 烘了，两个 compliance 值放 UBO，运行期可调）——
`α̃ = compliance/Δt²` 本来就要在运行期成形，因为 settle 步的 Δt 是 1/3 而其余是 1/30。

---

## 五、本轮没做

- **occupancy 改造**（第三节）。这是现在收益最大的单个改动：1.21 ms → 约 0.37 ms。
- **面积/二面角约束**。`ch10032_sprint` 的退化三角形变差 5.4 倍就是缺它（第二节第 1 点）。
- **swept / 自碰撞 / SDF**（v2 S5）。因此**本文不给任何穿模结论**。
- **trust region / 残差归约 / indirect dispatch**（v2 §7）。`min_edge` 段已经烘进资产备用。
- **训练**。按 v2 修订后的安排推迟到成本已知之后 —— 现在已知了，见第六节。

---

## 六、对 S7 的影响：原来的交易不成立，三条替代路线

v2 §S7 的赔率是"砍 5–6 个 block 腾出 0.36 ms 买 k=128"。实测 k=128 要 1.21 ms ≈ 18.7 个 block，
**比整个 12 block 的 processor 还多**，所以那笔交易作废。三条替代：

| 路线 | 成本 | 已知质量 | 状态 |
| --- | --- | --- | --- |
| **A. 先做 occupancy 改造** | k=128 预计 0.37 ms ≈ 5.7 block | G0 实测 score 0.532（k=128） | **推荐先做这个** |
| B. 降到 k=32 | 0.302 ms ≈ 4.7 block | G0 实测 0.775（对 A 支 0.72） | 可行但收益薄 |
| C. 12 block 全留，XPBD 当纯保险 | +1.21 ms，总 2.15 ms | 第二节的稳定性表 | 已能跑，超预算 39%→133% |

路线 B 的问题值得说清：G0 实测 jacobi k=32 得 0.775，而同场景 A 支是 0.72 —— **k=32 在
`hml_001962` 上根本没有优势**。k=128 的 0.532 才是有意义的收益点。所以"降 k"不是一条好路，
**occupancy 改造是把这件事变得划算的前提**。

至于训练本身，`GATE_G0_RESULTS.md` §9 给了一个独立的约束：**选型目标必须是带 XPBD 的分数**。
按裸闭环分搜出来的 r1，恰好是三个现成模型里最差的 XPBD 搭档。

---

## 复现

烘资产（四场景）：

```bash
cd implementations/vulkan-gnn-poc && for s in hml_001962 ch10032_tpose ch10032_sprint hood_grid64; do .venv/Scripts/python.exe -B tools/bake_xpbd_constraints.py --scene $s --calibration teacher --report results/xpbd_bake_$s.json; done
```

Python golden + Vulkan 对拍：

```bash
cd implementations/vulkan-gnn-poc && .venv/Scripts/python.exe -B tools/run_tinyhood_reference.py --asset-root .work/real_scene/ch10032_tpose --motion ch10032_tpose --steps 10 --model .work/hood_data/student32x12_r1.vhood --xpbd-asset .work/real_scene/ch10032_tpose/ch10032_tpose.vxpbd --xpbd-iterations 128 --golden .work/real_scene/ch10032_tpose/student32x12_r1_xpbd128_rollout.vhgold
```

```bash
cd implementations/vulkan-gnn-poc && pwsh -NoProfile -File verify_hood.ps1 -Motion ch10032_tpose -Solver TinyHood -HoodModel .work/hood_data/student32x12_r1.vhood -Xpbd -XpbdIterations 128 -Golden .work/real_scene/ch10032_tpose/student32x12_r1_xpbd128_rollout.vhgold -Output results/xpbd128_verify.json
```

成本：

```bash
cd implementations/vulkan-gnn-poc && pwsh -NoProfile -File benchmark_hood_static.ps1 -Scene CH10032 -Motion ch10032_tpose -Solver TinyHood -HoodModel .work/hood_data/student32x12_r1.vhood -Xpbd -XpbdIterations 128 -Warmup 10 -Samples 60 -Output results/xpbd_timing_k128.csv
```

交互查看：

```bash
cd implementations/vulkan-gnn-poc && pwsh -NoProfile -File run.ps1 -Scene HoodGrid64 -Solver TinyHood -HoodModel .work\hood_data\student32x12_r1.vhood -Xpbd
```

> 脚本必须用 `pwsh`（PS7 三元语法），`powershell` 5.1 解析不了。overlay 里可以实时开关 XPBD、
> 改迭代数与 compliance。

单元测试（28 项，含 6 项 `FusedPortTests`）：

```bash
cd implementations/vulkan-gnn-poc && .venv/Scripts/python.exe -B -m unittest tests.test_xpbd tests.test_real_formats
```

新增代码：[`hood_xpbd.comp`](../overlay/shaders/hlsl/gnncloth/hood_xpbd.comp)、
[`tools/bake_xpbd_constraints.py`](../tools/bake_xpbd_constraints.py)、
`real_scene/xpbd.py` 的 `_apply_fused` / `bake_tables` / `load_vxpbd` / `sweep="fused"`。
改动：`hood_runtime.inl`（buffer / pipeline / dispatch / 时间戳 / UI / debug dump）、
`gnncloth.cpp`（CLI）、`run.ps1`、`verify_hood.ps1`、`benchmark_hood_static.ps1`、
`tools/run_tinyhood_reference.py`、`tools/gate_g0.py`（`--calibration-source`）。

产出：`xpbd128_verify.json`、`xpbd_timing_k{8,32,128,256}.csv`、
`xpbd_timing_grid64_k128.csv`、`xpbd_bake_*.json`、`xpbd_stability_*_{on,off}.json`。
