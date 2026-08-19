# GNN + XPBD 混合求解器：修订方案

> **执行状态（2026-08-19）：门禁 G0 已跑完，结论见
> [`implementations/vulkan-gnn-poc/results/GATE_G0_RESULTS.md`](../../implementations/vulkan-gnn-poc/results/GATE_G0_RESULTS.md)。**
> **G0 通过 —— 混合架构在四个场景上都胜过 GNN-only 与 XPBD-only。** 本文已按实测修订：
> §2.3 的估计已被证实但结论反转、§4 的 S1/S2a 被删除、§5.1 的方案 C 方向被推翻。
> 修订明细见 §0.1。
>
> **第二轮（2026-08-19，同日）：S7a 与 S2' 已完成**，结论见
> [`GATE_G0_RESULTS.md` §8–11](../../implementations/vulkan-gnn-poc/results/GATE_G0_RESULTS.md)
> 与 [`XPBD_VULKAN_RESULTS.md`](../../implementations/vulkan-gnn-poc/results/XPBD_VULKAN_RESULTS.md)。
> **三条被推翻：**（a）§2.3 的 0.36 ms 实测是 **1.21 ms**，因为 kernel 是 occupancy 受限而非
> dispatch 受限；（b）§5.2「需要新增速度 pass」在这条路径上不成立；（c）§S7 的等预算交易作废。
> **一条新结论：**闭环搜索出来的 r1 是最差的 XPBD 搭档 —— 选型目标必须换成带 XPBD 的分数。
> 修订明细见 §0.2。

本文替代 [`gnn-xpbd.md`](gnn-xpbd.md) 作为执行依据。v1 的物理推理基本正确，被替换的原因是它的
工程前提和收支常数是按"一个已经有 GNN + XPBD 的通用工程"写的，与 `implementations/vulkan-gnn-poc`
的实际状态有三处硬错位。所有数字改为本仓库实测值，并标注了哪些是估计。

**一句话结论（G0 后更新）：方向对，混合架构确实值得做，而且收益集中在学生泛化不了的场景上。
但 v1/v2 都押错了具体路线 —— 该做的是最朴素的"保留网络位置预测 + 无色 Jacobi 修正"，
不是约束空间 warm start，也不需要图着色。**

---

## 0. 与 v1 的差异摘要

| v1 条目 | 处置 | 原因 |
| --- | --- | --- |
| 第一节：不要把 x_gnn 塞进 x₀，维护 g 成本高 | **改写** | `g = M·a_gnn` 在本仓库零成本可得，见 §5 |
| 第二节 方案 A / B / C | ~~保留，但 C 从"研究对照"提为一等变体~~ → **实测后：A 胜，B/C 降级** | 见 §0.1 |
| 第三节：学习型 V-Cycle | 保留为终态 | 层级图已 bake 好，见 §1 |
| 第四节：Node Feature 含 Body SDF | **删除 SDF 项** | 仓库没有 SDF，身体是点代理，见 §1 |
| 第四节：`confidence` 门控 | **保留，且实测支持** | 最优初始化随学生在该状态下是否可信而变，见 §0.1 |
| 第五节：Swept / 交错碰撞 / 自碰撞 | **提为独立阶段 S5，单独预算** | 这是全方案最大的一块新工作，不是"保留现有能力" |
| 第六节：在 K_post 迭代后算 loss | **改方法** | 梯度下降在本架构上主动破坏闭环稳定性；且指标噪声 18%，见 §6 |
| 第七节：四层保险 | 保留，修正 trust region 与 fallback 的实现前提 | 边长非均匀；仓库无 indirect dispatch |
| 第八节：T_N + K_H·T_I + T_C < K_B·T_I，打平约 12 次迭代 | **作废** | T_I 实测 0.034–0.046 ms 且**与布料规模基本无关**（启动开销主导）；打平点 ≈ 27 次。记账单位应换成 dispatch 数与 GNN block 数，见 §2 |
| 第一阶段"复用当前 GNN" | **改写** | 蒸馏模型那条路径完全没有 XPBD，见 §1 |
| — | **新增 §3 门禁 G0** | 仓库已有的消融实验预示了混合架构的核心失效模式 |

### 0.1 G0 实测对本文的修订

| 本文原条目 | 预测 | 实测 | 处置 |
| --- | --- | --- | --- |
| §2.3 非结构化色数 16–20 | 估计 | **18**（stretch-only 10） | ✅ **估对了** |
| §2.3 由此推出"S1 是前置必要条件" | 每迭代 0.07–0.09 ms 会压垮预算 | 色数确实高，**但正确应对是不用着色** | ❌ **结论反转**，见 §2.3 |
| §5.1 方案 C（约束空间 warm start）几乎免费、应提为一等 | 预期低迭代下优于 A | 三场景上"投影到 λ 空间"与"丢掉预测"差 ≤ 0.03 | ❌ **推翻**，见 §5.1 |
| §4 S4 Δλ head（方案 B） | 生产路线 | λ 参数化按构造表达不了网络位移的零空间分量 | ⏸ **降级为存疑** |
| §6.2 确定性改造是硬前置 | 收益可能落在 18% 噪声里 | 效应量 0.27，噪声 0.014，**大 13 倍** | ⏸ **G0 不需要**；仍是 S4 前置 |
| §3.2 G0 判据"B ≤ A 则砍 GNN" | — | 2/4 场景触发，但结论不成立 | ❌ **判据本身有缺陷**，见 §3.3 |
| — | — | **`cloth_rest` 不能作约束目标**（蒙皮 p95 1.89–2.03） | ➕ **新增必做项**，见 §1.4 |
| — | — | one-sided 约束 + `teacher` 标定各值 0.1 分 | ➕ **新增默认配置**，见 §1.4 |
| — | — | compliance 可用量级 ~1e-2，不是 1e-5～1e-6 | ➕ **修正量纲**（第二轮再次更正为 ~15，见 §0.2） |

---

### 0.2 S7a / S2' 实测对本文的修订（第二轮）

| 本文原条目 | 预测 | 实测 | 处置 |
| --- | --- | --- | --- |
| §2.3 Jacobi k=128 ≈ 0.36 ms（按 2.8 µs/dispatch 换算） | dispatch 受限 | **9.44 µs/迭代 → 1.208 ms**，且 `hood_grid64` 工作量 2.3× 却快 1.4× | ❌ **推翻**：是 **occupancy 受限**（11 workgroup / 34 SM）。见 §2.3 |
| §S7 砍 5–6 个 block 买 k=128 | 汇率 5.6 block | k=128 = **18.7 个 block** | ❌ **交易作废**，见 §4 S7 |
| §5.2 XPBD 后需要新增 finalize/速度 pass | 是新 pass | `clothPrevious = effective` 已等于 Python 的 `next_previous`；只有 settle 步要一次 buffer copy | ❌ **不成立**，见 §5.2 |
| §5.1 `standard` 不读 inertial → 需要惯性缓冲？ | 未提 | `standard` 全程不引用 `inertial`，x̃ 不必上 GPU | ➕ **S2' 因此更小** |
| §6.2 指标噪声 18%，由 `--repeats` 估计 | 进程内重复可估噪声 | **进程内重复严重低估**：`grid64` A 支进程内 0.000 / 跨进程 **0.600** | ❌ **估计器无效**，见 §8.2 |
| §1.4 `teacher` 标定可能要按动作烘 | 未知 | **可迁移**，且用最动态动作标定更好（tpose 用 sprint 的标定反而好 0.025） | ➕ **资产按服装烘一份** |
| §6 训练选型目标 | 裸闭环分 | 搜索过的 r1 是**最差**的 XPBD 搭档，tpose 上次序整个反转 | ➕ **改为带 XPBD 的分数**，见 §6.1 |
| §1.4 compliance ~1e-2 | 1e-2 起效 | **1e-1 仍完全无效**；α̃ 追平 weight_sum 需 compliance ≈ **15** | ❌ **再错 3 个数量级**，正确区间 [1, 100] |
| — | — | 距离约束管不住三角形塌成零面积（`sprint` 退化率恶化 5.4×） | ➕ **新增缺口：面积/二面角约束** |
| — | — | one-sided 与接触投影的不动点正好落在自己的分支边界上，CPU/GPU 偶尔取不同侧 | ➕ **对拍误差 1.68e-3 的机制，非缺陷** |

---

## 1. 起点：仓库的实际状态

### 1.1 两条互不相通的路径

| | 玩具 VGNN 路径 | **蒸馏模型路径（本方案的对象）** |
| --- | --- | --- |
| 网格 | 结构化 grid（16/32/64²） | 非结构化服装网格 |
| GNN | 2 层，模仿解析质点弹簧 oracle | tinyhood 32 latent × 12 block / HOOD Fine15 / PostCVPR |
| XPBD | **有**（`gnn_constraints*.comp`） | **完全没有** |
| 碰撞 | 单个球 | 4096 顶点点代理 + 最近点半平面投影 |
| 质量 | 全局 `particleMass` | **per-vertex `clothMass`** |
| 速度状态 | 显式，`finalize` 从修正后位置反推 | **无显式速度**，隐含在 `clothPrevious` |

蒸馏模型路径无 XPBD 有明确证据：`hood_runtime.inl:1114` 的 debug dump 常量写死
`"xpbd": false`，UI 文本是 "XPBD off"（`hood_runtime.inl:1264-1266`）。
（**S2' 已改**：那个字面量现在报告真实状态，UI 也改成可切换。）

**所以"复用现有 XPBD"是不成立的。** 现有 XPBD 的每一处都绑定在 grid 上：

- 着色是解析式的 —— `generateXpbdEdges`（`overlay/examples/gnncloth/gnncloth.cpp:401`）是 grid
  双重循环，`colors[x & 1u]`、`colors[8u + (x & 3u)]`；shader 侧 `colorOf()`
  （`gnn_constraints_tiled.comp:59`）镜像同一套规则；
- rest length 取自 UBO 三个标量 `restDistH/V/D`（`gnn_constraints_tiled.comp:43-50`）；
- 逆质量是 `1/particleMass` 全局常量（`gnn_constraints.comp:23`）。

### 1.2 蒸馏模型路径已经有的东西（可复用）

这些是好消息，决定了工作量的下界：

- **per-vertex 质量**：`clothMass`，M⁻¹ 比 grid 路径还正确；
- **rest 位置**：`clothRestPosition`，per-edge rest length 可现算，无需 bake；
- **三角形 + 顶点→三角形 CSR**：`clothTriangles`、`clothTriangleOffsets/Indices`，二面角对可由此 bake；
- **mesh 边**：`meshSenders` / `meshReceivers`（CH10032 为 7894 有向 = 3947 无向）；
- **已 bake 的 2 级粗层级**（PostCVPR 路径）：资产段 `vertex_level`、`c0_senders/receivers/offsets`、
  `c1_*`（`hood_runtime.inl:378-383`）。**v1 第三节的学习型 V-Cycle 所需的层级结构已经存在**；
- **GPU 侧 CSR 构建与确定性归约的成例**：`hood_world_reverse.comp` 每步在 GPU 上转置
  cloth→proxy 映射，且刻意做成 bit-identical；`hood_world_nearest.comp` 是 128 lane 组内归约。
  自碰撞的空间哈希、以及 fallback 需要的 active list，都可以照这两个的模式写；
- **通用分段资产格式**：`.vhood` / `.vclth` 是带目录的分段容器（`real_scene_format.h`）。
  实际落地时选了**独立的 `.vxpbd` 文件**而不是加段 —— 加段会改 `.vcloth2` 的 payload SHA-256，
  而现有全部 golden 都钉在那个值上；`loadSectioned` 本身通用，所以新文件零格式改动。

### 1.3 缺的东西（必须新建）

| 缺口 | 规模 | 备注 |
| --- | --- | --- |
| bake 期边着色 | 小 | 贪心边着色，~50 行 Python，进 baker |
| 二面角/2-hop 弯曲约束对 | 小 | 由三角形 CSR 生成 |
| lambda 缓冲 + 每步清零 | 极小 | 照 `gnncloth.cpp:949` 的 `vkCmdFillBuffer` |
| ~~**finalize / 速度 pass**~~ | ~~中~~ | ❌ **实测不需要**：`clothPrevious = effective` 已经对齐 Python，只有 settle 步要一次 buffer copy。见 §5.2 |
| 非结构化 XPBD kernel | 中 | 结构可照 `gnn_constraints.comp`，索引与质量改为查表 |
| **Body SDF 及其梯度** | **大** | 不存在。当前只有点代理 + 最近点 |
| **Swept / CCD** | **大** | 不存在 |
| **自碰撞（空间哈希 / BVH / proximity edge）** | **大** | 全仓库 grep 零命中 |
| GPU 残差归约 + indirect dispatch | 小-中 | 仓库现在**没有任何** `vkCmdDispatchIndirect` |

### 1.4 约束目标必须标定 —— `cloth_rest` 不可用（实测后新增）

`tools/audit_rest_lengths.py` 实测各场景**蒙皮 frame 0** 的边长 / authored rest 边长：

| 场景 | 帧数 | p95 | max | >1.5 | >3 | >6 | `rest` 可用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `hml_001962` | 137 | 1.081 | 5.27 | 86 (2.2%) | 27 | 0 | 否 |
| `ch10032_tpose` | 1 | **1.890** | 12.11 | 305 (7.7%) | 69 | 17 | 否 |
| `ch10032_sprint` | 62 | **2.025** | 16.31 | 323 (8.2%) | 111 | 79 | 否 |
| `hood_grid64` | 1 | **1.000** | 1.00 | 0 | 0 | 0 | **是** |

只有合成的 grid64 自洽。真实服装上，光是蒙皮就把 2.2–8.2% 的边拉长到 1.5× 以上。**用
`cloth_rest` 边长做距离约束会把这些边往回缩最多 12–16 倍。** `tools/train_student.py` 的
`edge_penalty` 文档串已经记录过这个教训的训练侧版本。

**默认配置（实测最优）**：

| 参数 | 值 | 依据 |
| --- | --- | --- |
| 约束目标 | **`teacher`** = teacher rollout 稳态边长中位数 | 优于 `bind` 0.10 分。bake 期可得，非运行期信息 |
| 拉伸方向 | **one-sided**（只抗拉伸，不抗压缩） | 优于双向 0.11 分。布料该起皱而不是被撑开 |
| sweep | **jacobi**（无色） | 见 §2.3 |
| compliance | **0（刚性）**；可用区间在 **[1, 100]**，中心约 15 | 实测 0～1e-1 全部完全无效。`α̃ = c/Δt² = c×900` 要追平 `weight_sum` 中位数 13816 需 c ≈ 15。**§0.2 更正了这里原来写的 ~1e-2** |

这也意味着 **bake 管线要多一个段**：per-edge 标定目标长度。它替代了原 S2a 里的着色段。

---

## 2. 成本货币：换掉 v1 的记账单位

### 2.1 实测常数

XPBD（VGNN grid 路径，RTX 4060 Ti，锁 2700 MHz，1000 采样中位数，`results/benchmark_*.csv`）：

| 网格 | 约束数 | 8 迭代 × 16 色 | **每迭代** |
| --- | ---: | ---: | ---: |
| 16×16 | 1,378 | 0.2719 ms | **0.0340 ms** |
| 32×32 | 5,826 | 0.3226 ms | 0.0403 ms |
| 64×64 | 23,938 | 0.3642 ms | **0.0455 ms** |

关键事实（`results/RESULTS.md`）：64×64 每迭代 0.0455 ms 里 **0.0448 ms 是 dispatch + barrier
开销**（16 色 × ~2.8 µs）。顶点数涨 16 倍，XPBD 只涨 1.34 倍。

> **XPBD 的成本由 dispatch 数决定，几乎与布料规模无关。** CH10032 只有 grid64 的 1/6 约束量，
> 但每迭代成本几乎相同。

GNN（tinyhood 32×12，CH10032，`results/student32x12_ch10032_timing.csv`，min_ms）：

| 阶段 | min | 占比 |
| --- | ---: | ---: |
| `skin` | 0.0222 | 2.4% |
| `features_world` | 0.0317 | 3.4% |
| `encoder_total` | 0.0658 | 7.1% |
| **`processor_12_total`** | **0.7956** | **86.1%** |
| `decoder_integrate` | 0.0061 | 0.7% |
| **`total`** | **0.9244** | (p95 **1.1798**) |

processor 内每个 block：`edge` 0.0508 + `node` 0.0137 = **0.0646 ms**。edge update 独占全帧 66%。

### 2.2 正确的记账单位

```
1 个 GNN message-passing block  =  0.0646 ms  ≈  1.9 次 XPBD 迭代
12 个 block（整个 processor）    =  0.775 ms   ≈  22.8 次 XPBD 迭代
整个 GNN（含 skin/encoder/decoder） = 0.924 ms ≈  27 次 XPBD 迭代
```

三个直接推论：

1. **v1 第八节的打平点 12 次是错的，实际 ≈ 27 次。** 按 v1 自己的判据，"GNN 替代 XPBD 迭代"
   在 GPU 时间上是亏的，除非基线真需要 30 次迭代。
2. **但按迭代数记账本身就是错的模型。** 你买卖的是 dispatch。省 8 次迭代 = 省 128 个 dispatch
   ≈ 0.28 ms；而 GNN 用 ~30 个 dispatch 花掉 0.92 ms。
3. **真正的设计空间是"更浅的 GNN + XPBD post-smooth，在同一总预算下"。** 12 block → 6 block
   释放 0.39 ms ≈ 11 次 XPBD 迭代。这是本方案要探索的主轴，v1 完全没提。

### 2.3 色数实测 18 —— 估计对了，但应对方式相反

> ~~⚠️ **估计，非实测**：非结构化网格的边着色色数会**高于** grid 的 16 … 合计每迭代
> ~25–30 个 dispatch … **~0.07–0.09 ms/迭代**，是 grid 的 2 倍。~~
>
> ✅ **实测（`gate_g0_scenes.json`）：18 色**（3947 stretch + 3735 bend = 7682 约束；
> stretch-only 为 10 色）。估计区间 16–20 命中。

但由此推出的"S1 是前置必要条件"是**错的**。正确应对不是把着色的 dispatch 压下去，
而是**根本不用着色**：

| 配置 | dispatch/迭代 | k | 总 dispatch | 估计 ms | score |
| --- | ---: | ---: | ---: | ---: | ---: |
| coloured | 18 | 8 | 144 | 0.40 | 0.812 |
| coloured | 18 | 32 | 576 | 1.61 | 0.465 |
| **jacobi** | **1** | 32 | 32 | 0.09 | 0.775 |
| **jacobi** | **1** | **128** | **128** | **0.36** | **0.532** |

Jacobi 每迭代收敛更差（k=32 时 0.775 对 coloured 的 0.465），但 **18 倍的 dispatch 优势远超
这个劣势**：在约 1.3 ms 的等成本点上 jacobi k=128 得 0.532，coloured k=8 得 0.812。

要 coloured k=32 的 0.465 需要 2.54 ms，是 GNN 自身成本的 2.7 倍 ≈ 25 个 GNN block，不是
合理选择。**所以 S1（修 dispatch 结构）与 S2a（bake 期着色）都从计划中删除。**

> ~~ms 仍是用 dispatch 单价换算的**估计**。Python 计时无意义（teacher 与学生同为 16 ms/step）。
> 真实数字仍需在 Vulkan 上测 —— 这是现在剩下的主要未知量。~~
>
> ✅ **已实测（`xpbd_timing_k*.csv`）：每迭代 9.44 µs，k=128 = 1.208 ms —— 是上表估计的 3.4 倍。**
> **上表整列 "XPBD 估计 ms" 因此全部偏低 3.4 倍**，但列与列之间的比例仍然成立，所以
> "Jacobi 在等成本下胜过 coloured" 这个结论不变（coloured 的 dispatch 也要按同样的倍数放大，
> 而且它的 kernel 更小、occupancy 更差）。
>
> **误差的来源不是换算算错了，是模型选错了**：那个换算假设 XPBD 是 dispatch 受限的。
> 9.44 µs 远高于 2.8 µs 的单价，而 `hood_grid64` 用 2.31 倍的 slot 工作量跑出 1.40 倍的速度，
> 所以它既不是 dispatch 受限也不是带宽受限，**是 occupancy 受限** ——
> CH10032 的 1377 个顶点只有 11 个 workgroup，卡上有 34 个 SM。详见 §4 的 **S8**。

### 2.4 约束目标必须标定（新增，实测后加入）

见 §1.4。这一条不做，实验会静默失效：`ch10032_tpose` 上用错标定时混合架构**比不接 XPBD 更差**
（0.726 对 0.242），换对标定后变成更好（0.217）。**标定决定符号，不是调参。**

---

## 3. 门禁 G0：先证明 GNN 还有存在价值

### 3.1 为什么需要这道门禁

grid 路径上做过一次同构实验（`results/gnn_ablation.json`，600 步确定性场景，下游是近刚性 XPBD）：

| 对比 | 平均顶点距离 | L2 |
| --- | ---: | ---: |
| `analytic` vs `gravity`（**整个图消息传递的贡献**） | 0.0035 | 0.1436 |
| `gnn` vs `analytic`（**网络自身误差**） | 0.0451 | 2.1158 |

**网络自身误差是耦合项贡献的 14.7 倍。** `RESULTS.md` 的原话：

> the near-rigid XPBD distance constraints already enforce what the Laplacian term was
> approximating … the network is a more expensive, less accurate way to evaluate a formula
> that is three lines long.

这正是"把 XPBD 放在网络之后"的核心风险：XPBD 顺手把 GNN 本该提供的低频拉伸行为一起吃掉，
GNN 退化成 0.92 ms 的噪声源。

v1 第三节断言 XPBD 不擅长低频/长程、GNN 擅长 —— 这对**收敛解**成立，但上面的测量说明：
compliance 够硬、迭代够多时 XPBD 自己也到得了。玩具路径不能直接外推到真实服装（那里有
真实剪裁、非均匀质量、真实身体运动），但它把举证责任放在了混合架构这一侧。

### 3.2 G0 的具体形式

在 CH10032 与至少一个真实动作序列上，跑三条曲线，**总 GPU 时间对齐**：

| 配置 | 成本 | 说明 |
| --- | --- | --- |
| **A. GNN only** | 0.924 ms | 当前交付状态（`student32x12_r1`） |
| **B. XPBD only** | 预算 ≤ 0.924 ms | GNN 关掉，纯惯性预测 + 尽可能多的迭代 |
| **C. Hybrid** | 预算 ≤ 0.924 ms | 例：6 block GNN + 6~10 次 XPBD |

按 §8 的指标评分。

### 3.3 G0 结果：通过（2026-08-19）

在 Python 闭环里跑完，未动 Vulkan。最佳配置（jacobi k=128 + `teacher` 标定 + one-sided），
2 次重复：

| 场景 | A：GNN only | B：XPBD only | **C：混合** | C 相对 A |
| --- | ---: | ---: | ---: | ---: |
| `hml_001962`（137 帧运动） | 0.7356 | 2.4123 | **0.4049** | −45% |
| `ch10032_tpose`（静态） | 0.2412 | 2.2539 | **0.2174** | −10% |
| `ch10032_sprint`（62 帧快速） | 2.5297 | 2.1723 | **0.6065** | **−76%** |
| `hood_grid64`（合成压力） | 1.8770 | 1.6950 | **0.1591** | **−92%** |

**C ≪ min(A, B)，四个场景都是。** 噪声底 0.014（3 次重复实测），效应量 0.27–1.72。

**最重要的模式：收益在学生最差的场景上最大。** `ch10032_tpose`（r1 被调优的场景）学生已经
很好（0.241），混合只提升 10%；`ch10032_sprint` 和 `hood_grid64` 上学生崩到 2.53/1.88，
混合拉到 0.61/0.16。这正好补上 §6.1 记录的 r1 过拟合问题 —— **XPBD 补的是学生泛化不了的部分。**

XPBD-only 的输法也值得记：`under` ≈ 0、`over` = 1.44，它不是被约束绷太硬而是**垮掉** ——
没有网络提供的动力学，裙子的垂坠形状与 teacher 不同，第 4 步就越过 edge P95 2.0。

#### G0 判据本身有缺陷（如实记录）

§3.2 原本写的判据是"若 B 的最佳 score ≤ A，则砍掉 GNN"。它在 `ch10032_sprint`（2.17 < 2.53）
和 `hood_grid64`（1.70 < 1.88）上**确实触发了**，但"砍掉 GNN"不成立：判据隐含假设了
"若 XPBD-only 不比 GNN-only 差，则网络无用"，而**网络可以只在组合中有用** —— 这两个场景上
C 比 B 好 3.6 倍和 10.7 倍。

真正被这两个场景证伪的不是"网络有用"，而是"r1 在训练场景之外可靠"：B 支在四场景稳定在
1.70–2.41，A 支从 0.24 摆到 2.53。**学生的方差比 XPBD 大得多。**

> 教训：单阈值判据只在"基线唯一"时成立。以后写门禁要写三方比较，不要写两方阈值。

---

## 4. 阶段划分（G0 后重排）

```
✅ G0  门禁：三方等成本对照（Python）                通过，见 §3.3
❌ S1  修 XPBD 的 dispatch 结构                     删除 —— Jacobi 每迭代已是 1 dispatch
❌ S2a bake 期边着色                                删除 —— Jacobi 不需要着色
✅ S3  方案 A 与 C 对照                             已在 Python 完成，A 胜，见 §5.1
⏸ S4  Δλ head（方案 B）                            降级为存疑，见 §5.1
✅ S7a 现成模型 × XPBD（零训练成本）                 完成，r1 是最差搭档，见 §6.1
✅ S2' Vulkan 落地：非结构化 Jacobi XPBD             完成并对拍通过，但成本是估计的 3.4 倍

── 剩下要做的（按收益排序）──
S8  kernel occupancy 改造：一 workgroup 一顶点        ← 现在收益最大的单个改动
S9  面积或二面角约束                                 距离约束管不住三角形塌陷
S7b 等预算训练（等 S8 定下真实汇率）                  选型目标改为带 XPBD 的分数
S5  碰撞：swept / 交错 / 自碰撞（独立预算）
S6  终态：学习型 V-Cycle
```

### S8 kernel occupancy 改造（新的第一优先）

实测每迭代 9.44 µs，比 §2.3 按 dispatch 单价估的 2.8 µs 高 3.4 倍。诊断在
`XPBD_VULKAN_RESULTS.md` §3：**不是 dispatch 受限也不是带宽受限，是 occupancy 受限。**
判据很干净 —— `hood_grid64` 的 slot 工作量是 CH10032 的 **2.31 倍**，每迭代却**快 1.40 倍**：

| 场景 | 顶点 | 总 slot | workgroup | 每迭代 | 每 slot |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ch10032` | 1,377 | 24,786 | **11** | 9.437 µs | 0.381 ns |
| `hood_grid64` | 4,096 | 57,344 | **32** | 6.720 µs | **0.117 ns** |

CH10032 的 11 个 workgroup 在一块 34 SM 的卡上让 23 个 SM 全程空闲。

**改法本仓库已有先例**：`hood_world_nearest.comp` 的注释就是这句 ——
"The work was never the problem; the occupancy was" —— 它把"一线程一顶点串行扫 4096 proxy"
改成"一个 128-lane workgroup 一个顶点"，并用组内确定性归约保持逐位一致。
照同样的形状改 `hood_xpbd.comp`：一 workgroup 一顶点，lane 分摊 18 个 slot，组内确定性归约，
workgroup 从 11 涨到 1377。

**预期**：`24786 × 0.117 ns ≈ 2.9 µs/迭代 → k=128 ≈ 0.37 ms`，正好落回 §2.3 原来的估计。
这是外推而非实测，且每 lane 只摊 1–2 个 slot 时归约开销占比会上升，所以 0.37 ms 是乐观下界。

**通过条件**：k=128 实测 ≤ 0.5 ms，且 `verify_hood.ps1 -Xpbd` 仍然通过。

### S9 面积或二面角约束（新增）

`ch10032_sprint` 上 XPBD 把最大边长比从 55.4 压到 7.2、翻面从 25.2% 压到 13.0%，
但**退化三角形（面积比 < 0.1）反而从 0.0035 恶化到 0.0191**，且 162 步内没有阻止结构崩坏。
原因是结构性的：**三条边都保持长度，三角形仍可被压成零面积**，而本轮的 bend 约束是 2-hop
距离形式，同样管不住。这是当前约束集的真实缺口，不是调参问题。

### S7b 等预算训练（等 S8）

原 S7 的赔率作废：k=128 实测 1.21 ms ≈ **18.7 个 GNN block**，比整个 12 block 的 processor 还多。
降 k 也不是出路 —— G0 实测 jacobi k=32 在 `hml_001962` 上得 0.775，而 A 支是 0.72，**没有优势**；
有意义的收益点是 k=128 的 0.532。所以 **S8 是让这笔交易重新划算的前提**。

S8 之后汇率变成 0.37 ms ≈ 5.7 个 block，原来的"砍 5–6 个"重新成立，届时再训。

**训练协议必须改一处（§6.1 实测支持）：选型目标是带 XPBD 的分数，不是裸闭环分。**

### ✅ S2' Vulkan 落地（已完成 2026-08-19）

实测报告：[`XPBD_VULKAN_RESULTS.md`](../../implementations/vulkan-gnn-poc/results/XPBD_VULKAN_RESULTS.md)。
比原计划又小了两处（下面标 ❌ 的两条实测发现不需要），交付的是
[`hood_xpbd.comp`](../../implementations/vulkan-gnn-poc/overlay/shaders/hlsl/gnncloth/hood_xpbd.comp)
+ [`tools/bake_xpbd_constraints.py`](../../implementations/vulkan-gnn-poc/tools/bake_xpbd_constraints.py)。

**资产（baker）—— 全部完成，但落在独立的 `.vxpbd` 而非 `.vhood`/`.vclth` 的新段**
- 理由：标定长度依赖 teacher rollout，而 `.vcloth2` 按服装共享；且给它加段会改 payload
  SHA-256，现有全部 golden 都钉在那个值上。新文件不读就不影响任何东西。
- 烘出：`pairs` / `target_len` / `weight_sum` / `kind` / `slots` / `signs` / `incident` /
  `inverse_mass` / `min_edge`。CH10032 = 7682 约束（3947 + 3735）/ 1377 顶点 / slot 宽 18 / 360 KB。
- **`weight_sum` 必须烘、不能在 kernel 里现加** —— 融合 sweep 的两个端点都要算同一个 Δλ，
  `w_a + w_b` 与 `w_b + w_a` 不保证逐位相同，那两份 λ 就会分开。
- padded `[V, K]` 表直接用，**没有转 CSR**：K=18 有 38% padding，但既然是 occupancy 受限，
  那些 lane 不花钱，而扁平 stride 让 kernel 与 Python 参考可逐行对照。
- compliance **不烘**（放 UBO，运行期可调）—— `α̃ = compliance/Δt²` 本来就要在运行期成形，
  settle 步 Δt = 1/3、其余 1/30。

**runtime —— 完成，两处比计划少**
- 一个 Jacobi kernel，1 dispatch/迭代。**接触折进同一个 dispatch**（逐顶点、至多一个、
  写入互不相同，不需要累加），所以真的是 1 而不是 2；
- ❌ **不需要新增 finalize/速度 pass**（§5.2 那条不成立）：`clothPrevious = effective`
  已经等于 Python 的 `next_previous`，只有 settle 步需要一次 `vkCmdCopyBuffer`；
- ❌ **不需要惯性缓冲**：`standard` 模式全程不引用 `inertial`；
- Jacobi 必须 ping-pong（`clothPosition` ↔ `xpbdScratch`，两个 descriptor set）——
  就地更新会变成不确定的部分 Gauss-Seidel；
- 开 XPBD 时强制关掉 `hood_integrate.comp` 自己那次半平面投影，否则一步投影两次，
  也与 G0 测的配置不符。

**S2'c 交付即测 —— 完成，结果是坏消息**
- 每迭代 **9.44 µs**（k=8…256 平到小数点后两位），k=128 = **1.208 ms**，是 §2.3 估计的 **3.4 倍**；
- 诊断为 **occupancy 受限**（`hood_grid64` 工作量 2.3× 却快 1.4×），修法见 **S8**；
- 正确性：10 步对拍 max 1.68e-3 / mean 8.8e-7，Khronos 同步校验干净；k=0 与纯 GNN
  golden 的误差是 `8.478760719e-06`，与既有记录完全相同。

### S5 碰撞（独立预算，不并入 S7 / S2' 的成本核算）

现状必须说清楚：碰撞候选是"半径 0.03 内的最近代理顶点"
（`hood_world_nearest.comp`），碰撞响应是在该顶点上做一次半平面投影
（`hood_integrate.comp:41-44`）。**没有 SDF、没有 swept、没有自碰撞。**

按依赖排序：

1. **swept 候选**：对线段 `[xⁿ, x_gnn]` 生成候选，而不是只测终点。最小改动是把
   `hood_world_nearest.comp` 的点-点距离换成"线段到代理点的最短距离"，代价可控；
2. **交错碰撞**：把 body collision 插进 stretch/bend 之间（v1 第五节 2 的顺序是对的），
   而不是全部结构约束之后做一次；
3. **自碰撞**：空间哈希（照 `hood_world_reverse.comp` 的确定性 CSR 构建模式）+ proximity
   约束。这是最大的一块；
4. **SDF**（可选）：只有在证明点代理精度不够时才做。它是一条新的资产管线，不要顺手引入。

> **S7 / S2' 阶段的验收标准里不得出现穿模指标。** 一个没有自碰撞的系统不能宣称减少了穿模。
> v1 第五节的 Node Feature 里 `Body SDF` / `Body SDF Gradient` 两项**从特征表中删除**，
> 直到 S5-4 真的落地。

### S6 终态：学习型 V-Cycle

```
惯性预测 x̃
→ 1~2 次 XPBD Pre-Smooth
→ 构建约束残差图
→ GNN 预测粗层 Δλ / coarse correction   ← 层级图已 bake（§1.2）
→ 1~3 次 XPBD Post-Smooth
→ 碰撞与摩擦收尾
→ 残差超阈值时 indirect 追加迭代
```

v1 第三节的这个结构保留为终态目标。它比 v1 估计的更接近：`vertex_level` 与 `c0_/c1_` 粗层
边表已经作为资产段存在，PostCVPR 路径已经在用。

---

## 5. 数学修订

### 5.1 `g` 免费成立，但方案 C 的方向被实测推翻

v1 第一节担心：把 x_gnn 当 x₀ 会破坏 XPBD 的 `g(x₀, λ₀) = 0` 假设，而维护 `g` 成本高。

但看 `hood_integrate.comp:39`：

```hlsl
float3 predicted = effective + (effective - (pinMask[vertex] != 0 ? oldPosition : oldPrevious) + acceleration);
```

`effective + (effective - previous)` **就是**惯性预测 x̃。decoder 输出的 `acceleration`
恰好就是偏差 `x_gnn - x̃`，而且**已经逐顶点存在 `accelerationOut[vertex]`**
（`hood_integrate.comp:49`）。所以

```
g = M (x_gnn - x̃) = M · a_gnn
```

零成本可得，不需要重建、不需要额外 pass、不需要额外存储。

于是 v1 第三种方案的更新式

```
Δλ = (-h + J M⁻¹ g) / (J M⁻¹ Jᵀ + α̃)
Δx = M⁻¹ (Jᵀ Δλ - g)
```

在距离约束 kernel 里是**十来行的改动**，不是研究项目。~~因此把 C 从"研究对照"提为 S3 的一等变体。~~

补充一点 v1 没给足的信用：HOOD teacher 用增量势能目标训练，网络本来就在近似 XPBD 所下降的
同一个泛函的 argmin。warm start 的良态性比 v1 估计的好。

（`g` 的分摊仍需注意：多约束共享顶点时，`Δx = M⁻¹(JᵀΔλ - g)` 会让同一个顶点的 `g` 项被
多次减去。实现上应把 `g` 一次性作为独立 pass 消掉，或按顶点的约束计数分摊。**实测确认这是真
bug 来源**：第一版实现每迭代都减一次 `g`，把累积的约束修正全部抹掉。）

#### ❌ 实测推翻：约束空间 warm start 拿不到任何好处

三种初始化只差起点，求解器完全相同（`bind` 标定，coloured k=32；hml 为 3 次重复均值）：

| 场景 | A（无 XPBD） | `standard`（x_gnn） | `warmstart`（x̃+M⁻¹Jᵀλ₀） | `nowarm`（x̃） |
| --- | ---: | ---: | ---: | ---: |
| `hml_001962` | 0.7362 | **0.4678** | 0.8685 | 0.8670 |
| `ch10032_tpose` | **0.2417** | 0.7256 | 0.9660 | 0.9362 |
| `hood_grid64` | 2.0478 | 0.1604 | **0.1172** | **0.1173** |

**稳健结论：`warmstart` 从来没有明显赢过 `nowarm`** —— 三场景差 ≤ 0.03。把网络位移投影到
约束空间，效果和干脆丢掉它没有区别。

机制是结构性的：**网络的位置预测有一部分位于 Jᵀ 的零空间**（刚体平移，以及任何不改变边长的
形变），约束乘子按定义无法表达它。`tests/test_xpbd.py::test_warmstart_discards_a_null_space_displacement`
把这一点固定下来：一个纯平移在 `warmstart` 下被投影成零，落回与 `nowarm` 逐位相同的状态。

**对 v1 方案 B 的含义**：一个只输出 λ 的头，按构造表达不了这部分信号，而实测这部分正是有价值的
（`hml_001962` 上 standard 0.468 对 warmstart 0.869）。保留：`warmstart` 用的是解析对角投影，
不是训练出来的 λ 头；但零空间论证与训练无关。

#### ⏸ 不稳健：`standard` 是否最优取决于学生在该场景上有多好

- `hml_001962`（学生 0.736，尚可）：`standard` 0.468 明显最好 —— 保留网络预测有价值；
- `hood_grid64`（学生 **2.048**，崩了）：`nowarm` 0.117 好于 `standard` 0.160，
  B 支 0.540 也比 A 好 4 倍 —— **学生的预测在这里是负资产。**

所以不是"方案 A 总是最好"，而是：**网络预测该保留多少，取决于它在当前状态下是否可信。**
这恰好是 v1/v2 §四那个 `confidence` 门控要解决的问题，**实测支持保留该设计** —— 它现在有了
一个具体的、可测的用途，而不只是一个安全阀。

### 5.2 ❌ 实测推翻：不需要新的速度 pass

~~蒸馏路径没有显式速度状态：速度隐含在 `clothPrevious` 里，`hood_integrate.comp:47` 直接写
`clothPrevious = effective`。若 XPBD 修正后不重算 `clothPrevious`，修正量会以**伪速度**的形式
漏进下一帧惯性预测 …… 在这条路径上它需要一个**新的 pass**，不是配置项。~~

**前提是对的（没有显式速度状态），推论错了。** 把两侧对齐看：

| | Python（`tools/train_student.py:340`） | Vulkan（`hood_integrate.comp:47`） |
| --- | --- | --- |
| step 0 | `next_previous = predicted` | `clothPrevious = predicted` |
| step > 0 | `next_previous = graph.effective_position` | `clothPrevious = effective` |

两边**已经一致**，而 `effective` 是 XPBD 之前的量、XPBD 不该动它。修正量并不会变成伪速度：
下一步的 `effective_position` 是从**修正后的** `clothPosition` 蒙皮/pin 校正出来的，
所以 `2·effective_next − effective_prev` 已经带上了修正 —— 传播是自动的。

**只有 settle 步需要处理**：那一步 `clothPrevious` 被写成未修正的 `predicted`，而
`tools/compare_student_stability.py:72-76` 的 `previous = corrected if step == 0` 要求修正后的值。
runtime 用一次 `vkCmdCopyBuffer(clothPosition → clothPrevious)` 解决，
**一次 transfer，不是一个新 kernel**。

保留 v1 第七节"只反馈最终物理状态"的原则：GNN 下一帧的历史输入确实必须来自 XPBD 之后的状态，
上面的结构已经保证了这一点。`gnn_finalize.comp:34` 那个显式速度的写法属于 grid 玩具路径，
这条路径上没有对应物、也不需要。

---

## 6. 训练方法：改的不只是常数

v1 第六节"在 K_post 次迭代之后算 loss"原则上对，但忽略了 `results/STUDENT_STABILITY_ROUND2.md`
已经量化的两个障碍。它们直接决定 S4 能不能做成。

### 6.1 梯度下降在本架构上主动破坏闭环稳定性

实测结论，逐字保留其分量：

- 单步保真度与闭环稳定性**对立**：稳态单步方差解释率 `0.357 → 0.928`，闭环稳定性**差了 7 倍**；
- 该轮**没有**通过继续训练拿到更稳定的模型；所有形式的继续训练都让闭环更差，且不是学习率问题；
- 起作用的是**直接在闭环指标上做搜索**：单场景 360 步 score `2.743 → 0.259`，360 步内不再发散；
- 代价是**过拟合被打分的那个场景**（`student32x12_r1` 在合成 grid64 压力场景上比交付版更差）。

**方法修订**：Δλ head 的主要调参工具是**闭环指标搜索**，梯度训练只做初始化。穿过求解器的
梯度路径更长更噪，没有理由指望它比上一轮表现更好。搜索必须在 ≥3 个场景上联合评分以抑制过拟合。

#### ➕ 实测新增：选型目标必须换成"带 XPBD 的分数"

第二轮拿仓库现有三个 32×12 权重（覆盖"忠实 ↔ 稳定"这条轴，零训练成本）× XPBD 实测：

| 场景 | shipped A | v3 A | r1 A | shipped C | v3 C | r1 C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ch10032_tpose` | 0.2874 | 0.2813 | **0.2410** | 0.2120 | **0.1949** | 0.2186 |
| `hood_grid64` | 1.3559 | **0.9125** | 2.3142 | **0.1447** | **0.1446** | 0.1590 |

（tpose / grid64 各 3 次独立进程的均值；两处的三组区间都不重叠，见
`GATE_G0_RESULTS.md` §9。另两个场景带 XPBD 后三者不可分辨。）

**`ch10032_tpose` 上次序整个反转**：不接 XPBD 时搜索过的 r1 最好，接上 XPBD 后 r1 最差。
机制正是 §6.1 那条 —— 搜索是靠**把学生变硬**买稳定性的（`under = 0.537`），而 XPBD 也加硬，
两者叠加过了头。

**所以 S7b 的训练/搜索目标必须是带 XPBD 的闭环分。** 按裸分选出来的权重，恰好是接上 XPBD
之后最差的那一个 —— 这不是一个小偏置，而是选反了方向。

### 6.2 指标噪声 —— 原来 18% 的估计方法本身是无效的

同一份权重连续评 5 次：`0.3946 / 0.3636 / 0.4304 / 0.3866 / 0.4342`，极差 `0.0705`，
相对 `0.40` 的分值是 **18%**。根因是 `index_add_` 在 CUDA 上对 float 没有确定性实现。

原判断是"任何小于 18% 的改进都测不出来，混合架构的收益很可能就落在这个量级"。

**⏸ 实测修正：混合架构的效应量远大于噪声，G0 不需要这道前置。** `gate_g0_confirm.json` 用
3 次重复实测 A 支极差 **0.0141**、XPBD 各支 0.0000–0.0110，而效应量是 0.27–1.72 ——
**大 13 倍以上**。XPBD 自身用 padded-gather 固定顺序累加，逐位可复现，不贡献噪声
（`tests/test_xpbd.py::test_repeated_projection_is_bit_identical`，CPU 与 CUDA 各一次）。

**❌ 第二轮更正：`--repeats`（进程内重复）不是这条管线的有效噪声估计器，它低估 10–30 倍。**
同一配置跨独立进程重跑 4 次：

| 场景 | 进程内 `--repeats` 极差 | **跨进程极差（A / C）** |
| --- | ---: | ---: |
| `hml_001962` | 0.0035 | 0.0025 / 0.0162 |
| `ch10032_tpose` | 0.0000 | 0.0003 / 0.0048 |
| `ch10032_sprint` | 0.0007 | **0.1208** / **0.1436** |
| `hood_grid64` | 0.0000 | **0.5995** / 0.0002 |

机制：同一进程内两次 rollout 背靠背执行，分配器状态与 launch 配置相同，`index_add_` 的原子
累加顺序恰好复现；换进程才变。**`tools/refine_student.py` 的 `--eval-repeats` /
`--confirm-repeats` 是同一个进程内平均，所以它记录的 "0.40 上 ±0.035" 也是低估** ——
那一轮 500 轮搜索可能在比真实噪声更小的差异上做过取舍。这是 S7b 开工前要修的方法学问题。

**另一个独立结果：XPBD 消掉 run-to-run 方差。** `hood_grid64` 上 A 支跨进程极差 0.5995、
C 支 0.0002，相差 3000 倍 —— 学生发散时轨迹混沌，约束投影把它按回稳定吸引子。

**S4 的前置条件仍然成立**：Δλ head 的收益预期本来就小（§5.1 实测它接近零），那才是落在噪声
量级里的东西。若真要做 S4，先把 `real_scene/fine15.py:137,143` 的两个 `index_add_` 换成同样的
padded-gather，**放在开关后、默认关闭**（改了会轻微改变数值，与已发布数字不再逐位对齐）。

### 6.3 保留的部分

v1 关于"不要只做纯 multiplier MSE"、"用混合系统自身 rollout 的状态继续训练"、以及随机化
Δt / 子步数 / pre-post 迭代数 / compliance / 骨骼速度 / 外力 / 初始扰动 / 碰撞厚度的建议全部保留。
冗余约束下 λ 不唯一这一点也成立。

---

## 7. 保险机制（v1 第七节的修订版）

### 7.1 Trust Region

`|Δxᵢ| < η · min L_edge` 的问题：真实服装边长差异很大，全局 min 会让 clamp 被整个网格里最短
的那条边支配。**改为 bake 好的 per-vertex 局部最小边长**（S2' 资产阶段顺手产出）。

### 7.2 Residual Acceptance Test

保留（`scale = 1 → 0.5 → 0` 的极简 line search）。前提：需要 GPU 归约。可照
`hood_world_nearest.comp` 的 128-lane 组内归约模式写，成本很低。

### 7.3 Adaptive Fallback

保留。但前提要说清：**仓库现在没有任何 `vkCmdDispatchIndirect`**（grep 零命中）。需要新建
active-cloth-list + indirect dispatch 管线。不难，但是新东西，要排进 S2' 的工作量。

`activeProxy` 缓冲（`hood_runtime.inl:410`）已经是一个"GPU 侧写标记位"的成例，可以照抄结构。

### 7.4 只反馈最终物理状态

保留，且见 §5.2 —— 在这条路径上它需要一个新 pass。

---

## 8. 验收指标与测量协议

### 8.1 指标（沿用已建立的那套，不要新造）

`STUDENT_STABILITY_ROUND2.md` 建立的 teacher-relative 评分已经踩过"以 rest 为基准"和"只看
末帧"两个坑，直接继承：

| 项 | 含义 | 权重 |
| --- | --- | ---: |
| `over` | 比 teacher 更拉伸 —— 这就是发散 | 1.0 |
| `under` | 比 teacher 更硬 —— 保真度损失，不是发散 | 0.25 |
| `flip` | 超出 teacher 的翻面比例 | 2.0 |
| `drift` | 与 teacher 轨迹的逐步位置 RMS（米） | 2.0 |

加上：`edge P95 首次 > 2.0 的步数`、`与 teacher 轨迹最大位置差`。

参照点（CH10032 T-Pose，360 步）：teacher edge P95 @120 = 1.781；`student32x12_r1` score 0.259、
360 步内 edge P95 从未 > 2.0、最大位置差 0.075 m。**混合架构要赢的是这个 r1，不是最初的交付版。**

### 8.2 性能测量协议

沿用现有基准：锁 2700 MHz、无 validation、10 预热 / 60 采样、**min_ms 为首列**（最小值是跨运行
可复现的统计量），逐阶段独立打时间戳。

必须同时报：

- 总 GPU 时间与 **p95**（当前 GNN 是 0.924 / 1.180 ms，p95 有 28% 的尾部；
  `block_04_edge` p95 0.244 ms vs min 0.051 ms，尾部不可忽略）；
- **dispatch 数**（这是 §2 的真实成本单位）；
- 接触密集帧的 fallback 触发率；
- 达到同一残差阈值所需时间（而非固定迭代数下的时间）。

### 8.3 基线必须包含 small steps

v1 这一条对，保留：基线要加入"更多子步、每子步一次 XPBD"，而不只是"一个大步里很多次迭代"。
在 dispatch 受限的现实下这一点尤其重要 —— 子步会成倍增加 dispatch，需要实测而非推理。

---

## 9. 风险与放弃条件（G0 后更新）

| 风险 | 触发信号 | 处置 | 状态 |
| --- | --- | --- | --- |
| **XPBD 吃掉 GNN 的贡献** | G0 中 B ≈ C | 终止方案 | ✅ **未发生**：C ≪ min(A,B) 四场景 |
| 非结构化色数过高 | 每迭代 > 0.09 ms | 改 Jacobi | ✅ **已解决**：色数 18，改用 Jacobi |
| **Jacobi 迭代的真实 ms 远高于换算估计** | S2'c 实测 ≫ 0.36 ms / k=128 | 重估 S7 赔率；考虑降 k 或 fused kernel | ❌ **已发生**：实测 1.208 ms（3.4×）。诊断为 occupancy 受限，处置是 **S8**，不是降 k |
| 等预算下混合不划算 | S7 中 6 block + XPBD 不优于 12 block | 保留 12 block，把 XPBD 当纯稳定性保险而非省算力手段 | ⚠️ **当前确实不划算**（k=128 = 18.7 block）。S8 之后重估 |
| **闭环搜索把学生变硬，与 XPBD 叠加过头** | 搜索过的权重带 XPBD 后反而更差 | 选型目标换成带 XPBD 的分数 | ❌ **已发生**：tpose 上 r1 次序反转，见 §6.1 |
| **距离约束管不住三角形塌成零面积** | 退化三角形比例上升 | 加面积或二面角约束（**S9**） | ❌ **已发生**：`sprint` 上 0.0035 → 0.0191 |
| 饱和投影的不动点落在自己的分支边界上 | CPU/GPU 对拍误差跳变 | 按机制判断，不要直接放宽阈值 | ⚠️ 10 步 max 1.68e-3 对阈值 2e-3，余量 16% |
| Δλ head 的改进落在噪声里 | — | 不开工 | ⏸ **已基本证实**，见 §5.1 |
| 闭环搜索过拟合单场景 | 换场景后分数崩 | ≥3 场景联合评分 | ⚠️ r1 已经中招；混合架构恰好缓解了它 |
| **标定选错使混合反而变差** | 某场景上 C > A | 检查该场景蒙皮初态 edge ratio，换 `teacher` 标定 | ⚠️ **已发生过**（tpose + `bind`） |
| 碰撞工作量吞掉整个排期 | S5 超出预算 | S5 可独立交付/独立砍 | ⏸ 待做 |
| p95 尾部使 16.67 ms 预算不可控 | p95/min > 1.3 | 先查 `block_*_edge` 的尖峰来源 | ⏸ 当前 1.28，待查 |

**G0 已经给出了正面结局的第一半**：混合架构的价值不是"GNN 变快了"，而是**把学生泛化不了的
部分换成了可控的硬约束** —— 在学生最差的场景上把 score 改善 76–92%。S7 要回答的是这件事能不能
在**不增加总成本**的前提下做到。

---

## 附录 A：实测常数表

| 量 | 值 | 来源 |
| --- | ---: | --- |
| CH10032 cloth 顶点 | 1,377 | `student32x12_ch10032_timing.csv` |
| CH10032 有向 mesh 边 | 7,894（3,947 无向） | 同上 |
| CH10032 代理顶点 | 4,096 | 同上 |
| hood_grid64 顶点 / 有向边 | 4,096 / 32,004 | `student32x12_grid64_timing.csv` |
| student32x12 CH10032 total | **0.924 ms**（p95 1.180） | 同上 |
| student32x12 grid64 total | 2.64 ms | `STUDENT_32X12_RESULTS.md` |
| 1 个 GNN block | 0.0646 ms（edge 0.0508 + node 0.0137） | `student32x12_ch10032_timing.csv` |
| processor_12_total | 0.7956 ms（占 86.1%） | 同上 |
| XPBD/迭代（1,378 约束，16 色） | 0.0340 ms | `benchmark_16.csv` |
| XPBD/迭代（23,938 约束，16 色） | 0.0455 ms，其中 0.0448 为启动开销 | `benchmark_64.csv`、`RESULTS.md` |
| dispatch + barrier 单价 | ~2.8 µs | `RESULTS.md` |
| 消融：图耦合的全部贡献 | 0.0035 平均顶点距离 / 600 步 | `gnn_ablation.json` |
| 消融：网络自身误差 | 0.0451（**14.7×**） | 同上 |
| 指标噪声 | ±0.035 于 ~0.40（**18%**） | `STUDENT_STABILITY_ROUND2.md` |
| 单步 R² vs 闭环稳定性 | 0.357→0.928 使闭环差 **7×** | 同上 |
| teacher edge P95 @120 | 1.781 | `student32x12_longhorizon.json` |
| r1 360 步 score / 最大位置差 | 0.259 / 0.075 m | `STUDENT_STABILITY_ROUND2.md` |
| 60 Hz 预算 | 16.67 ms（GNN 现占 5.5%） | — |

### G0 新增的实测常数（2026-08-19）

| 量 | 值 | 来源 |
| --- | ---: | --- |
| CH10032 stretch 约束（无向边） | 3,947 | `gate_g0_scenes.json` |
| CH10032 bend 约束（内部边去重后） | 3,735（3,763 内部边，28 对重复） | 同上 |
| **贪心边着色色数** | **18**（stretch-only 10） | 同上 |
| 顶点度（mesh 边） | min 3 / median 6 / max 9 | — |
| per-vertex 质量跨度 | 4.3e-6 … 3.0e-4（68×） | `RuntimeScene.cloth_mass` |
| 蒙皮 frame 0 edge ratio p95 | 1.081 / 1.890 / 2.025 / 1.000 | `rest_length_audit.json` |
| **G0 最佳配置 score（四场景）** | 0.405 / 0.217 / 0.607 / 0.159 | `gate_g0_scenes.json` |
| **A（GNN only）score（四场景）** | 0.736 / 0.241 / 2.530 / 1.877 | 同上 |
| B（XPBD only）score（四场景） | 2.412 / 2.254 / 2.172 / 1.695 | 同上 |
| 3 次重复的噪声底 | A 支 0.0141，XPBD 支 0.0000–0.0110 | `gate_g0_confirm.json` |
| compliance 有效量级 | ~~**~1e-2**~~ → 见下表 | `gate_g0_params.json` |
| Python rollout 成本 | 16 ms/step（teacher 与学生相同 → 计时无意义） | — |

### S7a / S2' 新增的实测常数（2026-08-19 第二轮）

| 量 | 值 | 来源 |
| --- | ---: | --- |
| **Jacobi XPBD 每迭代（CH10032，1377 顶点 / 11 workgroup）** | **9.44 µs** | `xpbd_timing_k*.csv` |
| Jacobi XPBD 每迭代（grid64，4096 顶点 / 32 workgroup） | **6.72 µs** | `xpbd_timing_grid64_k128.csv` |
| 每 slot 成本 CH10032 / grid64 | 0.381 / **0.117 ns**（3.3× 差） | 同上 —— **occupancy 受限的判据** |
| k=8 / 32 / 128 / 256 的 XPBD min ms | 0.076 / 0.302 / **1.208** / 2.415 | `xpbd_timing_k*.csv` |
| k=128 折算 GNN block | **18.7 个**（不是估计的 5.6 个） | — |
| CH10032 slot 宽度 / 总 slot / padding | 18 / 24,786 / 38% | `xpbd_bake_*.json` |
| grid64 约束 / slot 宽 / 总 slot | 27,783 / 14 / 57,344 | 同上 |
| `.vxpbd` 大小（CH10032 / grid64） | 360 KB / 1,039 KB | 同上 |
| 全 pin 的死约束（CH10032 / grid64） | 72 / 63 | 同上 |
| **10 步对拍 max / mean abs error（k=128）** | **1.684e-3 / 8.85e-7**（阈值 2e-3 / —） | `xpbd128_verify.json` |
| k=0 对纯 GNN golden 的误差 | `8.478760719e-06`（与既有记录逐位相同） | — |
| 每步落在分支边界 1e-7 内的约束 / 接触 | ~90 / ~330 | 对拍误差的机制 |
| **跨进程 score 极差（A / C）** `hml` | 0.0025 / 0.0162 | `GATE_G0_RESULTS.md` §8 |
| 同上 `tpose` | 0.0003 / 0.0048 | 同上 |
| 同上 `sprint` | **0.1208** / **0.1436** | 同上 |
| 同上 `grid64` | **0.5995** / 0.0002 | 同上 —— XPBD 消掉 3000 倍方差 |
| `weight_sum`（逆质量和，`hml`） | p5 9,092 / 中位 13,816 / p95 154,101 | — |
| **compliance 使 α̃ 追平中位 weight_sum** | **≈ 15**（可用区间 [1, 100]） | `gate_g0_compliance.json` |
| 标定跨动作迁移（tpose→sprint / sprint→tpose） | 0.5343（区间内）/ **0.1936（更好 0.025）** | `gate_g0_calib_*.json` |
| 渲染器 300 步 `grid64` edge max（off / on） | **328.97 / 1.134** | `xpbd_stability_hood_grid64_*.json` |
| 渲染器 `sprint` 退化三角形（off / on） | 0.0035 / **0.0191（恶化 5.4×）** | `xpbd_stability_ch10032_sprint_*.json` |

## 附录 B：v1 中被删除的条目及理由

1. **Node Feature 的 `Body SDF` / `Body SDF Gradient`** —— 不存在，且引入它是一条新资产管线。
   替代：`worldDirectFeatures` 已有的 9 维（相对当前/目标代理位置 + 距离 + Δt），以及
   `proxyNormal` 给出的法线。
2. **第八节的打平公式与"约 12 次迭代"** —— 常数错 2 倍且方向相反；模型（按迭代数记账）也错。
   替代：§2.2 的 block/dispatch 记账。
3. **第一阶段"复用当前 GNN"的措辞** —— 会让人以为只需接线。替代：S2' 的完整清单。

## 附录 C：G0 后从本文删除的条目

1. **S1 修 dispatch 结构（8×8 tile + 精确枚举）** —— Jacobi 每迭代已是 1 dispatch，S1 要解决的
   问题不存在。原 `RESULTS.md` 记录的 tile 优化对 grid 玩具路径仍然有效，但与本方案无关。
2. **S2a bake 期边着色** —— Jacobi 不需要着色。色数 18 这个测量结果的用处，恰恰是证明不该着色。
3. **方案 C 提为一等变体** —— §5.1 实测推翻。`g = M·a_gnn` 免费这一点仍然成立，只是不值得用。
4. **§6.2 确定性改造作为硬前置** —— 效应量比噪声大 13 倍，G0 不需要。仍是 S4 的前置，但 S4 已降级。
5. **"2~4 次 XPBD 是否能替代原本十几次迭代"** —— 蒸馏路径上不存在"原本十几次迭代"这个基线。
   替代：G0 的三支对照。
