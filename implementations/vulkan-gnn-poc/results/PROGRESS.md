# 研究进展总结

> 本文是横向入口，把散在 16 份 result 文档里的结论拉到一处。**数字全部是仓库里 CSV / JSON 的
> 实测值**，每一条都注明来源文档。单项细节请看对应文档，本文不重复推导。
>
> 最后更新：2026-08-19（S2' 完成）。测试机 RTX 4060 Ti，驱动 `596.36.0.0`，
> Vulkan SDK 1.4.309，FP32，torch 2.10.0+cu128。

## 一句话状态

**已经有一个比 HOOD Fine15 快 9.5 倍、在四个场景上都不发散、且与 Python 参考逐步对拍通过的
GPU 混合求解器**（TinyHOOD 32×12 + 非结构化 Jacobi XPBD，CH10032 上 2.15 ms/步）。
主要遗留三件：一个已诊断清楚的 occupancy 问题（可省 0.84 ms）、一个缺失的约束类型
（管三角形塌陷）、以及完全没有自碰撞。

---

## 0. 两条实现路径

`implementations/vulkan-gnn-poc/` 内部有两条**互不相通**的路径，读任何结论前先确认它属于哪条：

| | 玩具 VGNN grid 路径 | **蒸馏模型路径（主线）** |
| --- | --- | --- |
| 网格 | 结构化 grid 16/32/64² | 非结构化服装网格（CH10032，1377 顶点） |
| 网络 | 2 层，模仿解析质点弹簧 oracle | HOOD Fine15 / TinyHOOD 学生 / PostCVPR |
| XPBD | 着色 Gauss-Seidel，早就有 | **2026-08-19 才有**（无色 Jacobi） |
| 碰撞 | 单个球 | 4096 顶点点代理 + 最近点半平面 |
| 质量 | 全局 `particleMass` | per-vertex `clothMass` |

另有 `implementations/vulkan-mlcloth-cpu-poc/`：MNN CPU 推理 → 每帧上传 → Vulkan 点绘制，
一条独立的窄链路验证。

---

## 1. 性能：225 ms → 2.15 ms

CH10032 一个模拟步的 GPU 时间（锁 2700 MHz、无 validation、10 预热 / 60 采样、`min_ms`）：

| 里程碑 | ms | 来源 |
| --- | ---: | --- |
| Fine15 首次移植（正确性优先，一 workgroup 一图元素） | **225.72** | `KERNEL_OPTIMISATION_RESULTS.md` |
| 阶段 1：修 MLP 权重内存布局 | 79.23 | 同上 |
| 阶段 2：processor layer0 协作载入 groupshared | 24.39 | 同上（累计 9.26×） |
| 阶段 3：清掉三处 O(N·M) 暴力循环 | **20.41** | `STUDENT_32X12_RESULTS.md`（累计 **11.1×**） |
| TinyHOOD 64×4 蒸馏 | 1.04–1.65 | `TINYHOOD_64X4_RESULTS.md` —— **但闭环第 1 步发散** |
| **TinyHOOD 32×12 蒸馏** | **0.924**（p95 1.180） | `STUDENT_32X12_RESULTS.md` |
| PostCVPR（官方层级模型） | 39.86–44.30 | `POSTCVPR_RESULTS.md`，比 Fine15 慢 1.82× |
| **32×12 r1 + Jacobi XPBD k=128** | **2.15** | `XPBD_VULKAN_RESULTS.md` |

三轮 kernel 优化**没有改动任何模型、架构、特征或积分方式，数值逐位不变**。

一个反直觉的架构结论，代价是一个失败的模型：**GPU 成本 ∝ `blocks × latent²`，所以深度便宜、
宽度昂贵**。64×4 恰好砍掉了廉价的深度 —— 参数更多（286,275 对 200,227）、更慢、还发散。

记账单位（`gnn-xpbd-v2.md` §2.2）：

```
1 个 GNN message-passing block = 0.0646 ms
整个 GNN（skin/encoder/12 block/decoder） = 0.924 ms
```

---

## 2. 六个已经确立的负面结果

这些是项目最有价值的产出 —— 每一条都省掉了一个方向的投入。

### 2.1 在近刚性 XPBD 下游，GNN 的图耦合项几乎没有贡献

grid 路径 600 步消融（`gnn_ablation.json`、`RESULTS.md`）：

| 对比 | 平均顶点距离 | L2 |
| --- | ---: | ---: |
| 去掉**整个图消息传递**的影响 | 0.0035 | 0.144 |
| **网络自身误差** | 0.0451 | **2.116（14.7×）** |

原文措辞：*the network is a more expensive, less accurate way to evaluate a formula that is
three lines long*。**这条是后来所有混合架构论证的举证责任来源**，它逼出了门禁 G0。

### 2.2 单步保真度与闭环稳定性是对立的

`STUDENT_STABILITY_ROUND2.md`：稳态单步方差解释率 `0.357 → 0.928`，闭环稳定性**差 7 倍**。
所有形式的继续训练都让闭环更差，且不是学习率问题。起作用的是**直接搜索闭环指标**
（360 步 score `2.743 → 0.259`），代价是过拟合被打分的那个场景。

顺带一个方法学教训：该文原报的"单步方差解释率 0.980"是被探针构成误导的 —— step 0 的目标
二阶矩是稳态步的 **28.8 倍**，一个 step-0 样本就能主导整条轨迹的读数。

### 2.3 `cloth_rest` 不能作约束目标，也不能作评价基准

蒙皮 frame 0 的边长 / authored rest 比值（`rest_length_audit.json`）：

| 场景 | p95 | max | >1.5 的边 |
| --- | ---: | ---: | ---: |
| `hood_grid64`（合成） | **1.000** | 1.00 | 0 |
| `hml_001962` | 1.081 | 5.27 | 2.2% |
| `ch10032_tpose` | **1.890** | 12.11 | 7.7% |
| `ch10032_sprint` | **2.025** | 16.31 | 8.2% |

光是蒙皮就把真实服装 2–8% 的边拉过 1.5×。**标定选择会翻转混合架构的结论符号**
（tpose 上 `bind` 标定 0.726 对 `teacher` 标定 0.217）。

这条也解释了一个既有指标的读法：`STUDENT_STABILITY_ROUND2.md` 全部在 tpose 上打分，
而该场景蒙皮初态对 rest 的 p95 已经是 1.890 —— teacher 第 120 步的 1.781 是**低于**初态的，
不是"teacher 把布拉长了"。

### 2.4 预测 λ（方案 B/C）拿不到任何好处 —— 结构性原因

三场景实测"把网络位移投影到约束空间"与"干脆丢掉它"差 ≤ 0.03。机制：
**网络位移有一部分在 Jᵀ 的零空间里**（刚体平移、任何不改变边长的形变），λ 参数化按构造
表达不了。`tests/test_xpbd.py::test_warmstart_discards_a_null_space_displacement` 把它固定成断言。

### 2.5 图着色是错的方向

实测色数 **18**（原估计 16–20，估对了），但正确应对是**根本不着色**：等成本下
Jacobi k=128 得 0.532，coloured k=8 得 0.812。要 coloured k=32 的 0.465 需要 2.54 ms。

### 2.6 没有可直接复用的轻量公开权重

`LIGHTWEIGHT_GNN_CANDIDATES.md` 检索结论：没有一个公开 checkpoint 同时满足"明显轻于
Fine15 / 以动画身体和接触为条件 / 跨拓扑泛化 / 不重训就能用"。最接近的 EUNet 骨干仍是
128 latent × 15 层。**所以自己蒸馏是必要的，不是偷懒。**

---

## 3. 混合架构：门禁 G0 通过

四场景 teacher-relative score（越低越好，teacher = 0；`gate_g0_scenes.json`）：

| 场景 | A：GNN only | B：XPBD only | **C：混合** |
| --- | ---: | ---: | ---: |
| `hml_001962`（137 帧运动） | 0.736 | 2.412 | **0.405** |
| `ch10032_tpose`（静态） | 0.241 | 2.254 | **0.217** |
| `ch10032_sprint`（62 帧快速） | 2.530 | 2.172 | **0.47–0.61** |
| `hood_grid64`（合成压力） | 1.877 | 1.695 | **0.159** |

**C ≪ min(A, B)，四个场景都是。** 关键模式：**收益在学生最差的场景上最大** ——
tpose 只提升 10%，sprint / grid64 提升 76–92%。XPBD 补的正是学生泛化不了的部分。

XPBD 的作用不是"更接近 teacher"，而是**止住发散**：`over` 从 0.49 降到 0.26，
`drift` 基本不变。XPBD-only 的输法也值得记 —— 它不是被绷太硬（`under ≈ 0`）而是**垮掉**：
没有网络提供的动力学，裙子的垂坠形状与 teacher 不同，第 4 步就越过 edge P95 2.0。

**最佳配置**：jacobi k=128 + `teacher` 标定 + one-sided。one-sided 与 `teacher` 标定各值
约 0.1 分；compliance 在 0～1e-1 全部无效（见 §5）。

一个如实记录：G0 预先写死的判据"若 XPBD-only ≤ GNN-only 则砍掉 GNN"在 2/4 场景上触发了，
但结论不成立 —— 判据隐含假设了"网络不能只在组合中有用"。**教训：门禁要写三方比较，
不要写两方阈值。**

---

## 4. XPBD 已落地 Vulkan，但成本是估计的 3.4 倍

### 正确性

10 步与 Python 逐步对拍（`xpbd128_verify.json`）：max `1.684e-3` / mean `8.85e-7`，
Khronos 同步校验干净，`k=0` 逐位复现纯 GNN 的 `8.478760719e-06`。

那个 1.68e-3 **不是缺陷**：one-sided 约束的 `max(residual, 0)` 与接触投影的 `signed < 0`,
它们的**不动点正好坐在自己的分支边界上**（每步约 90 条约束和 330 个接触落在 1e-7 内），
少数顶点在 CPU/GPU 上取了不同侧。两次运行到末位相同，不会 flake，但离 2e-3 阈值只有 16% 余量。

### 成本 —— 这一阶段真正的产出

| | 方案估计 | 实测 |
| --- | ---: | ---: |
| 每迭代 | 2.8 µs | **9.44 µs** |
| k=128 | 0.36 ms | **1.208 ms** |
| 折算 GNN block | 5.6 个 | **18.7 个** |

**诊断判据很干净**：`hood_grid64` 的 slot 工作量是 CH10032 的 **2.31 倍**，每迭代却
**快 1.40 倍**（6.72 对 9.44 µs）。工作量涨而时间降 → 排除工作量受限与带宽受限
→ **是 occupancy 受限**：1377 顶点只有 11 个 workgroup，卡上 34 个 SM。

**修法本仓库已有先例**：`hood_world_nearest.comp` 的注释就是
*"The work was never the problem; the occupancy was"* —— 它把"一线程一顶点"改成
"一个 128-lane workgroup 一顶点 + 组内确定性归约"。照同样形状改，外推 **≈ 0.37 ms**。
所以那个估计没算错，它估的是一个 occupancy 正常的 kernel。

### 渲染器里的 300 步（独立于 Python 的验证）

| 场景 | 结构 off → on | edge max off → on |
| --- | --- | --- |
| `hood_grid64` | ❌ → **✅** | **328.97 → 1.134**（零翻面） |
| `hml_001962` | ✅ → ✅ | 3.890 → 3.239 |
| `ch10032_sprint` | ❌ → **❌** | 55.39 → 7.21 |
| `ch10032_tpose` | ✅ → ✅ | 1.784 → 4.527 |

`grid64` 是完全的挽救，截图对照见
[`tinyhood_grid64_r1_xpbd_off.png`](tinyhood_grid64_r1_xpbd_off.png) 与
[`tinyhood_grid64_r1_xpbd_on.png`](tinyhood_grid64_r1_xpbd_on.png)。

`tpose` 在这套指标上"变差"是尺子问题（比值是对 rest 的，见 §2.3）。有意思的是
**收益正好按各场景 rest 的可信度排序**（grid64 1.000 极大改善 → tpose 1.890 变差），
这是对 §2.3 的独立佐证。

---

## 5. 最新一轮的两条颠覆性发现

### 5.1 闭环搜索出来的模型是**最差**的 XPBD 搭档

`ch10032_tpose` 上次序整个反转（各 3 次独立进程的均值，区间不重叠）：

| 模型 | 不接 XPBD | 接上 XPBD |
| --- | ---: | ---: |
| `student32x12`（未搜索） | 0.2874 | 0.2120 |
| `student32x12_v3`（单步最准，闭环差 7×） | 0.2813 | **0.1949** |
| `student32x12_r1`（搜索 500 轮 / 109 分钟） | **0.2410** | 0.2186 |

机制：搜索是靠**把学生变硬**买稳定性的（`under = 0.537`），XPBD 也加硬，叠加过了头。
**含义：训练/搜索的选型目标必须换成带 XPBD 的分数** —— 按裸分选出来的权重恰好是接上 XPBD
之后最差的那个。这不是小偏置，是选反了方向。

### 5.2 之前用的噪声估计器是无效的

`--repeats` 在同一进程内平均，而 `index_add_` 的原子累加顺序恰好在进程内复现：

| 场景 | 进程内极差 | **跨进程极差（A / C）** |
| --- | ---: | ---: |
| `hml_001962` | 0.0035 | 0.0025 / 0.0162 |
| `ch10032_tpose` | 0.0000 | 0.0003 / 0.0048 |
| `ch10032_sprint` | 0.0007 | **0.1208** / **0.1436** |
| `hood_grid64` | 0.0000 | **0.5995** / 0.0002 |

低估 10–30 倍。G0 的 A-vs-C 结论不受影响（效应量 0.3–2.2），但模型间 0.02–0.14 的比较在
`sprint` / `grid64` 上不可分辨。**`refine_student.py` 是同一个毛病** —— 它记录的 ±0.035
也是低估，那轮 500 轮搜索可能在比真实噪声更小的差异上做过取舍。

**另一个独立结果：XPBD 消掉 run-to-run 方差** —— grid64 上 A 支 0.5995、C 支 0.0002，
差 3000 倍。学生发散时轨迹混沌，约束投影把它按回稳定吸引子。

---

## 6. 还开着的口子（按收益排序）

| # | 缺口 | 已量化的代价 / 收益 |
| --- | --- | --- |
| **S8** | **kernel occupancy 改造** | 1.21 → 约 0.37 ms。总成本 2.15 → 1.29 ms，即比 teacher 快 **15.8 倍** |
| **S9** | **面积或二面角约束** | 距离约束管不住三角形塌成零面积 —— `sprint` 上退化率恶化 5.4 倍且仍崩 |
| S7b | 等预算训练（等 S8 定汇率） | 选型目标必须改成带 XPBD 的分数（§5.1） |
| S5 | swept / 自碰撞 / SDF | **完全不存在**（全仓库 grep 零命中）。因此**现在不能给任何穿模结论** |
| — | compliance 探索 | 实测 0～1e-1 全部无效；α̃ 追平逆质量和需 compliance ≈ **15**，可用区间 [1, 100] |
| S6 | 学习型 V-Cycle（终态） | 2 级粗层级已 bake 好（PostCVPR 路径在用），比原估计更接近 |
| — | 确定性改造 | `fine15.py:137,143` 的两个 `index_add_` 换成 padded-gather，是搜索与 S4 的前置 |
| — | benchmark harness | 动画场景 300 samples 会失败（与 XPBD 无关），`XPBD_VULKAN_RESULTS.md` §2 有记录 |

---

## 7. 文档地图

| 文档 | 内容 |
| --- | --- |
| **`GATE_G0_RESULTS.md`** | 门禁 G0（§1–7）+ 噪声底 / 模型对照 / compliance / 标定迁移（§8–11） |
| **`XPBD_VULKAN_RESULTS.md`** | S2' 落地、对拍、成本与 occupancy 诊断 |
| `STUDENT_STABILITY_ROUND2.md` | 稳定性 vs 保真度的对立、闭环搜索、r1 交付 |
| `STUDENT_32X12_RESULTS.md` | 1 ms 学生、宽/深成本扫描、O(N·M) 清除 |
| `TINYHOOD_64X4_RESULTS.md` | 失败记录：宽/深权衡反了 |
| `KERNEL_OPTIMISATION_RESULTS.md` | 三轮 kernel 优化，数值逐位不变 |
| `RESULTS.md` | grid 路径：XPBD 推导、消融实验、tile 优化 |
| `HOOD_RESULTS.md` / `HOOD_GRID64_RESULTS.md` | Fine15 在真实服装 / 合成网格上的移植验证 |
| `POSTCVPR_RESULTS.md` | 官方层级模型部署链（通，但 44 ms） |
| `HOOD_MODEL_COMPARISON.md` | Fine15 vs Toy2L（优化前的早期测量） |
| `LIGHTWEIGHT_GNN_CANDIDATES.md` | 公开轻量权重检索：没有可直接用的 |
| `plans/gnn/gnn-xpbd-v2.md` | 执行方案，含两轮实测修订明细（§0.1 / §0.2） |

---

## 8. 现在就能看的东西

```bash
cd implementations/vulkan-gnn-poc && pwsh -NoProfile -File run.ps1 -Scene HoodGrid64 -Solver TinyHood -HoodModel .work\hood_data\student32x12_r1.vhood -Xpbd
```

overlay 里可以实时勾掉 `XPBD (Jacobi)` 看对照，也能拖迭代数和 compliance。

> 两个坑：脚本必须用 `pwsh`（PS7 三元语法），`powershell` 5.1 解析不了；
> `-Solver TinyHood` 不传 `-HoodModel` 会静默跑 `tinyhood64x4.vhood`，那是第 1–4 步就发散的
> 第一版学生。
>
> **overlay 里的 ms 不能当性能数据**：交互运行不锁频、有 VSync 和 UI 竞争，实测比锁频
> benchmark 高约 2.6 倍（grid64 k=128 overlay 报 2.25 ms，benchmark 是 0.86 ms）。
> 性能一律用 `benchmark_hood_static.ps1`。
