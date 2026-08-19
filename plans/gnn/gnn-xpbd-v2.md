# GNN + XPBD 混合求解器：修订方案

> **执行状态（2026-08-19）：门禁 G0 已跑完，结论见
> [`implementations/vulkan-gnn-poc/results/GATE_G0_RESULTS.md`](../../implementations/vulkan-gnn-poc/results/GATE_G0_RESULTS.md)。**
> **G0 通过 —— 混合架构在四个场景上都胜过 GNN-only 与 XPBD-only。** 本文已按实测修订：
> §2.3 的估计已被证实但结论反转、§4 的 S1/S2a 被删除、§5.1 的方案 C 方向被推翻。
> 修订明细见 §0.1。

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
| — | — | compliance 可用量级 ~1e-2，不是 1e-5～1e-6 | ➕ **修正量纲** |

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
- **通用分段资产格式**：`.vhood` / `.vclth` 是带目录的分段容器（`real_scene_format.h`），加一个
  `xpbd_constraints` / `xpbd_colors` 段不动格式、不破坏 hash 校验。

### 1.3 缺的东西（必须新建）

| 缺口 | 规模 | 备注 |
| --- | --- | --- |
| bake 期边着色 | 小 | 贪心边着色，~50 行 Python，进 baker |
| 二面角/2-hop 弯曲约束对 | 小 | 由三角形 CSR 生成 |
| lambda 缓冲 + 每步清零 | 极小 | 照 `gnncloth.cpp:949` 的 `vkCmdFillBuffer` |
| **finalize / 速度 pass** | 中 | 路径上不存在等价物，见 §5.2 |
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
| compliance | 需在 **~1e-2** 附近扫 | `α̃ = c/Δt²`，0～1e-6 比逆质量和小 7 个数量级，完全无效 |

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

> ms 仍是用 dispatch 单价换算的**估计**。Python 计时无意义（teacher 与学生同为 16 ms/step）。
> 真实数字仍需在 Vulkan 上测 —— 这是现在剩下的主要未知量。

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

── 剩下要做的 ──
S7  等预算对比：6 block 学生 + Jacobi k≈128         ← 现在最值得做的单个实验
S2' Vulkan 落地：非结构化 Jacobi XPBD + 标定段 + 速度 pass
S5  碰撞：swept / 交错 / 自碰撞（独立预算）
S6  终态：学习型 V-Cycle
```

### S7 等预算对比（新的第一优先）

G0 的最佳配置比 A 贵约 39%（估计 1.28 对 0.92 ms）。§2.2 的等预算方案是砍 GNN 深度来买迭代：
1 个 block = 0.0646 ms，砍 5–6 个 block 正好腾出 Jacobi k=128 的 0.36 ms。

`train_student.py --blocks` 支持 1–15，所以这是一次训练 + 一次 `gate_g0.py` 复跑。

赔率已经量化：**0.36 ms 的 XPBD 在四场景把 score 从 0.74 / 0.24 / 2.53 / 1.88 买到
0.40 / 0.22 / 0.61 / 0.16。** 问题变成"最后 6 个 block 值不值这些"。考虑到 12 block 的学生在
`ch10032_sprint` 上本身只有 2.53，这笔交易看起来划算，但必须实测。

**通过条件**：6 block + XPBD 在 ≥3 个场景上优于 12 block 无 XPBD，且总估计成本 ≤ 0.924 ms。

### S2' Vulkan 落地（在 S7 通过之后）

比原 S2 显著缩小 —— 不需要着色，不需要 tile 优化：

**资产（baker）**
- 由 `clothTriangles` 生成无向 stretch 边集（去重，CH10032 = 3947）与二面角对（3735，
  注意 28 对重复的对顶点要去重）；
- **per-edge 标定目标长度**（§1.4），由 teacher rollout 稳态中位数烘出；
- per-vertex 局部最小边长（trust region 用，§7.1）；
- per-vertex padded 约束表（`slots` / `signs`，宽度 = 最大关联约束数），这是 Jacobi 确定性
  累加的结构，直接照 `real_scene/xpbd.py` 的 `_gather_tables`；
- 写进 `.vhood` / `.vclth` 的新段，不动既有段布局。

**runtime**
- 一个 Jacobi kernel：per-vertex 一个线程，走 padded 表 gather，`incident` 平均。
  1 dispatch/迭代；
- **新增 finalize/速度 pass**：从修正后位置反推速度并写回 `clothPrevious`（§5.2）。
  Python 侧因为 `advance()` 的结构免费得到这一点，Vulkan 侧不是；
- one-sided 残差 + `teacher` 标定长度 + 最近代理半平面接触。

**S2'c 交付即测（现在最重要的未知量）**
- **实测每次 Jacobi 迭代的真实 ms**。§2.3 的 0.36 ms 是用 2.8 µs/dispatch 换算的估计；
  一次 Jacobi 迭代是一个 7682 约束 + 1377 顶点的 kernel，可能不再是纯启动开销。
- 若实测远高于估计，回头重估 S7 的赔率。

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

### 5.2 速度必须从修正后位置反推 —— 这条在本路径上是新 pass

蒸馏路径没有显式速度状态：速度隐含在 `clothPrevious` 里，`hood_integrate.comp:47` 直接写
`clothPrevious = effective`。若 XPBD 修正后不重算 `clothPrevious`，修正量会以**伪速度**的形式
漏进下一帧惯性预测 —— 这正是 v1 第七节"只反馈最终物理状态"要防的，但在这条路径上它需要
一个**新的 pass**，不是配置项。

模式照 `gnn_finalize.comp:34`：

```hlsl
correctedVelocity = (correctedPosition - previousPosition) / max(deltaT, 1e-4);
```

同时 GNN 下一帧的历史输入必须来自 XPBD 之后的状态，不得保留 `x_gnn`。

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

### 6.2 指标噪声 18% —— G0 未被它挡住，但 S4 仍需先修确定性

同一份权重连续评 5 次：`0.3946 / 0.3636 / 0.4304 / 0.3866 / 0.4342`，极差 `0.0705`，
相对 `0.40` 的分值是 **18%**。根因是 `index_add_` 在 CUDA 上对 float 没有确定性实现。

原判断是"任何小于 18% 的改进都测不出来，混合架构的收益很可能就落在这个量级"。

**⏸ 实测修正：混合架构的效应量远大于噪声，G0 不需要这道前置。** `gate_g0_confirm.json` 用
3 次重复实测 A 支极差 **0.0141**、XPBD 各支 0.0000–0.0110，而效应量是 0.27–1.72 ——
**大 13 倍以上**。XPBD 自身用 padded-gather 固定顺序累加，逐位可复现，不贡献噪声
（`tests/test_xpbd.py::test_repeated_projection_is_bit_identical`，CPU 与 CUDA 各一次）。

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
| **Jacobi 迭代的真实 ms 远高于换算估计** | S2'c 实测 ≫ 0.36 ms / k=128 | 重估 S7 赔率；考虑降 k 或 fused kernel | ⚠️ **现在最大的未知量** |
| 等预算下混合不划算 | S7 中 6 block + XPBD 不优于 12 block | 保留 12 block，把 XPBD 当纯稳定性保险而非省算力手段 | ⏸ 待测 |
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
| compliance 有效量级 | **~1e-2**（0～1e-6 完全无效） | `gate_g0_params.json` |
| Python rollout 成本 | 16 ms/step（teacher 与学生相同 → 计时无意义） | — |

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
4. **"2~4 次 XPBD 是否能替代原本十几次迭代"** —— 蒸馏路径上不存在"原本十几次迭代"这个基线。
   替代：G0 的三支对照。
