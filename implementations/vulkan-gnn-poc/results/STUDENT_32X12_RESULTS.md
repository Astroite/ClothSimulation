# 32×12 学生模型：1 ms 预算内的 Fine15 蒸馏

> **本文的「单步方差解释率 0.980」是一个被探针构成误导的数字，见
> [`STUDENT_STABILITY_ROUND2.md`](STUDENT_STABILITY_ROUND2.md) 的修正。**
> 该指标把各样本的平方误差先累加再相除，因此按目标能量加权；而 step 0 的目标二阶矩是稳态步的
> **28.8 倍**（0.4896 对 0.0170），一个 step-0 样本就能主导整条 40 步轨迹的读数。
> 同一份权重按 regime 分开测是：step 0 `0.931`、steps 1–9 `0.540`、**steps 10–39 `0.357`**。
> 稳态区才是 150 步 rollout 几乎全部时间所在的区间，本文其余部分关于性能、架构成本与 Part A
> 的结论不受影响。

测试机：NVIDIA GeForce RTX 4060 Ti，驱动 `596.36.0.0`，Vulkan SDK 1.4.309，FP32。
性能全部为锁 2700 MHz、无 validation、10 预热 / 60 采样的 `min_ms`。

## 结论

| | 64×4（原学生） | **32×12（本轮）** |
| --- | ---: | ---: |
| CH10032 compute | 1.653 ms | **0.924 ms** |
| Grid64 compute | 4.14 ms | **2.64 ms** |
| 参数量 | 286,275 | **200,227** |
| 消息传递轮数 | 4 | **12** |
| 单步方差解释率 | < 0（比预测零更差） | **0.980** |
| 120 步 edge P95 | 55.36 | **2.13**（teacher 为 1.78） |
| edge P95 首次超过 2.0 | 第 **1** 步 | 第 **118** 步 |
| edge P95 首次超过 5.0 | 第 **4** 步 | **120 步内未超过** |

**1 ms 目标达成**（0.924 ms，余量 8%），且参数更少、深度 3 倍，**120 步不再发散**。

### 结构保持时长（CH10032 T-Pose，Python 闭环，edge P95）

teacher 自己也会把裙子拉伸到约 1.8 倍静止边长——这个场景下重力悬挂本来如此，
所以 **teacher 的数值是这个指标能达到的下限，不是 1.0**：

| 模型 | @5 | @10 | @30 | @60 | @120 | 首次 >2.0 | 首次 >5.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fine15 teacher | 1.324 | 1.480 | 1.827 | 1.800 | 1.784 | >120 | >120 |
| TinyHOOD 64×4 | 6.492 | 10.278 | 17.105 | 26.654 | 55.360 | 1 | 4 |
| 32×12（欠训练的中间版本） | 1.485 | 1.521 | 2.586 | 4.677 | 76.690 | 22 | 62 |
| **32×12（交付版本）** | **1.198** | **1.178** | **1.230** | **1.379** | **2.130** | **118** | **>120** |

**剩余差距是保真度而不是稳定性。** 交付模型在 5–60 步区间的 edge P95 比 teacher **更低**
（1.20 对 1.32、1.18 对 1.48、1.23 对 1.83），也就是它比 teacher 更"硬"、褶皱更少；
与 teacher 轨迹的最大位置差在 10 / 60 / 120 步分别为 `0.16 / 0.37 / 0.79 m`。
所以它是一个结构稳定但褶皱形态与 teacher 不同的模拟器，不是 teacher 的忠实复现。
作为对照，零加速度退化基线的位置差是 `0.44 / 7.45 / 28.94 m`、edge P95 `4.07`。

## 1. 为什么 1 ms 不是纯蒸馏问题

阶段 2 之后 CH10032 TinyHOOD 的 `1.653 ms` 里，有 `0.498 ms`（30%）与模型完全无关：

| 阶段 | 阶段 2 | Part A 之后 | 随模型缩小？ |
| --- | ---: | ---: | --- |
| skin | 0.022 | 0.022 | 否 |
| **features_world** | **0.476** | **0.029** | **否** |
| encoder_total | 0.177 | 0.066（32 latent） | 随 latent² |
| processor | 0.953 | — | 随 blocks × latent² |
| decoder_integrate | 0.016 | 0.006 | 随 latent |

即使 processor 归零，原本的地板也是 `0.70 ms`，留给模型 0.3 ms —— 相当于 1.25 个 64-latent block，
不是可用模型。所以先清掉这部分（Part A），预算才存在。

### Part A：三处 O(N·M) 暴力循环

三项都是纯性能改动，**四个 solver 的黄金验证结果逐字节不变**。

| 项 | 原实现 | 改法 | 逐位一致的依据 |
| --- | --- | --- | --- |
| A1 `clothNormal` | 每个 cloth 顶点遍历全部 2570 个三角形（1377×2570） | 加载时从已有 triangle 列表推导顶点→三角形 CSR（拓扑静态，不改二进制格式、不重烘焙） | CSR 内按三角形索引升序，且一个三角形重复引用同一顶点时只存一次——与原 `ids.x != vertex && ...` 的语义相同，累加顺序不变 |
| A2 最近 proxy | 每个 cloth 顶点扫描全部 4096 个 proxy（1377×4096） | 独立 dispatch，一个 128-lane workgroup 负责一个顶点，协作归约 | 原循环用严格 `<`，等距时取最小索引；归约用「距离小者胜，否则索引小者胜」，该规则可结合 |
| A3 proxy→cloth | 每个 proxy 扫描全部 cloth 顶点，**每 block 一次**（15×4096×1377） | 每步反转一次映射成 CSR，block 只读自己的短列表 | 每个 cloth 顶点算自己在 (proxy, cloth) 字典序中的全局 rank，组内 cloth 升序；没有原子决定位置，布局与执行顺序无关 |

A2/A3 的真正病根不是总工作量，而是**占用率**：features dispatch 按
`max(cloth, proxy, meshEdge)` 索引，cloth 顶点的工作只落在前 1377 个线程 = 11 个 workgroup，
而这块卡有 34 个 SM，每条 lane 还要串行 4096 次。改成一个 workgroup 一个顶点后占用率提升约 125 倍。

A3 同时消除了「每加一个 block 都要付一次全量扫描」的固定代价，这是「买深度」成立的前提。

Part A 对所有 solver 都有效：

| 场景 / 模型 | 阶段 2 | Part A | 加速 |
| --- | ---: | ---: | ---: |
| CH10032 Fine15 | 24.39 | **20.41** | 1.20× |
| CH10032 PostCVPR | 44.30 | **39.86** | 1.11× |
| CH10032 TinyHOOD 64×4 | 1.653 | **1.040** | 1.59× |
| Grid64 Fine15 | 71.75 | **66.98** | 1.07× |
| Grid64 TinyHOOD 64×4 | 4.14 | **3.43** | 1.21× |

## 2. 架构选择：深度便宜，宽度昂贵

用**随机初始化权重**实测成本（计时与权重值无关，所以在花时间训练前就能定架构）。
每 block 成本在 32 latent 下完全线性：

| 架构 | 固定开销 | processor | 每 block | 总计 | 参数量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64×4（原） | 0.226 | 0.793 | 0.198 | 1.040 | 286,275 |
| 32×15 | 0.126 | 0.902 | 0.0601 | 1.029 | 247,747 |
| 32×14（推算） | 0.126 | 0.843 | 0.0602 | 0.969 | — |
| **32×12** | 0.126 | 0.722 | 0.0602 | **0.848** | 200,227 |
| 32×10 | 0.125 | 0.603 | 0.0603 | 0.728 | 168,547 |

关键推论：成本 ∝ `blocks × latent²`，所以 `8×32` 与 `2×64` 等价成本但感受野 4 倍。
**原来的 64×4 恰好做了反向权衡**——砍掉廉价的深度，保留昂贵的宽度。

`32×15` 甚至比 `64×4` 更快（1.029 vs 1.040），但差 3% 未进 1 ms。
选 **32×12**（0.848 ms 随机权重 / **0.924 ms 实测交付权重**）留 8% 余量。
训练权重略慢是因为它的布料形态不同，活跃 world edge 更多——计时依赖状态，
所以报告值取实际训练权重。

未采用 48 latent：**48 不是 warp size 32 的倍数**，第二个 warp 会浪费 16 条 lane。
`32×15` 需要修掉串行 LayerNorm 归约（`WaveActiveSum`）才能进 1 ms，本轮未做。

## 3. 训练：修掉「低单步误差 ≠ 闭环稳定」

原 `train_tinyhood.py` 拟合 48 个单步状态，teacher-forced MSE 0.027 看起来不错，
但闭环第 5 步 edge P95 已 6.49。HOOD 用二阶无阻尼位置积分，持续的小加速度偏差按
约 `ε·n²/2` 累积，且学生一旦离开拟合过的状态，误差自行放大。

新 `tools/train_student.py`（保留原脚本不动，因为它复现已记录的 64×4 权重）：

| 缺陷 | 改法 |
| --- | --- |
| 只拟合单步加速度 | **两阶段**：阶段 1 单步拟合（便宜、把网络带进正确盆地），阶段 2 展开 6 步并反传整条链，对齐 teacher 轨迹 |
| 无噪声注入 | 扰动状态后**由 teacher 重新标注**，让学生学会从偏差中恢复 |
| 48 样本 / 单条轨迹 | 四个场景共 **720 样本**（含此前从未使用的 137 帧 `hml_001962`） |
| batch = 1 | 梯度累积，batch 4 |
| 按 teacher-forced MSE 选模型 | 按**闭环结构稳定性**选（`edge_ratio_p95` + 翻面率） |
| 无几何约束 | 边长与翻面惩罚，直接压制被度量的失效模式 |
| CPU-only torch | `torch 2.10.0+cu128`，训练 21 分钟（CPU 估算约 10 小时/轮） |

噪声幅度按边长校准：rest 边长中位数 0.03 m 但 1% 分位只有 0.0065 m，所以
默认 sigma 取 `[0, 0.001, 0.003]`，且**几何惩罚只作用于未扰动样本**——否则会惩罚
学生一步内无法消除的噪声。

### 阶段 2 是决定性的，但先要把阶段 1 训到位

第一次运行只跑了约 **3,600 次优化步**，而 `CosineAnnealingLR(T_max=20)` 在损失仍单调下降时
就把学习率衰减到近 0，等于提前冻住。结果是单步拟合远未收敛：

| | 首次运行（欠训练） | 交付运行 |
| --- | ---: | ---: |
| 优化步数 | 约 3,600 | 约 **37,500** |
| 样本数 | 720 | **1,251** |
| 方差解释率（整体） | 0.617 | **0.980** |
| 方差解释率（step 0） | **0.263** | **0.996** |
| 120 步 edge P95 | 76.69 | **2.13** |

`方差解释率 = 1 − MSE / 目标二阶矩`，即相对「什么都不预测」这个基线解释了多少方差。
它比 loss 有用：loss 的绝对值依赖数据构成，而这个比值可比，也直接说明离可用还有多远。
诊断依据是**单样本过拟合测试**——同一模型对单个样本 600 步就能压到 `1e-4`，
说明梯度与容量都健康，问题是训练量而不是结构。

第二个问题是**前几步是独立 regime 且严重欠采样**：step 0 用 `1/3 s` 的 timestep（其余为
`1/30 s`），MSE 比后续步差 **37 倍**，却只占样本的 1.7%——而每次 rollout 都从它开始。
`--early-step-repeats` 对前 3 步额外采样后，step 0 的方差解释率从 `0.263` 升到 `0.996`，
不再是短板。

阶段 1 收敛后（方差解释率 0.980），阶段 2 把 60 步 edge P95 从 5.98 压到 1.38。
阶段 2 会让单步方差解释率略降（0.980 → 0.93 附近），这是用单步精度换多步稳定的预期取舍。

逐 epoch 记录见 `student32x12_training_log.jsonl`（145 条）。

## 4. Vulkan 数值正确性

| 指标 | 结果 |
| --- | ---: |
| 第一步位置 max abs | `3.874e-07` |
| 第一步加速度 max abs | `3.912e-07` |
| 10 步位置 max abs | `2.080e-06` |
| 10 步位置 mean abs | `5.383e-08` |
| 第一步 world edge mismatch | 0 |

阈值为第一步 max ≤ `2.0e-4`、第一步 mean ≤ `2.0e-5`、10 步 max ≤ `2.0e-3`，全部通过。
原始数据 `student32x12_verify.json`。Khronos validation 与 synchronization validation 无报错。

### 过程中修掉的三个真实缺陷

1. **`tinyhood_integrate.comp` 硬编码 `loadMlp(mlpTable, 15)`**。decoder 的 MLP id 是
   `3 + blocks*3`，随深度移动；15 只对 4 block 正确。12 block 时它把一个 processor MLP
   当成 decoder 用，导致第一步加速度误差 0.1（10 cm）。三个 integrate shader 现在都从
   uniform 读 `decoderMlp`。**这类 bug 只在改变深度时才暴露**。
2. **验证失败时结果 JSON 是 0 字节**。`std::ofstream` 仍在作用域内时抛出未捕获异常，
   MSVC 不做栈展开，流永不 flush——恰好在需要数字诊断失败时把数字丢掉。已改为
   在抛出前离开流的作用域。
3. **四个脚本不绝对化 `-HoodModel` / `-AssetRoot`**（`-Output` 却会）。exe 的工作目录是
   `.work/Vulkan`，相对路径解析到别处，然后 benchmark 模式因 `errorModeSilent` 静默地以
   `0xC0000409` 退出、没有任何提示。

## 5. 架构参数化

换架构现在只需重训 + 换权重文件，C++ 一行不用改：latent 与 block 数**从 VHOOD 的
tensor 形状推断**（`inferTinyArchitecture` 与 Python 侧 `infer_architecture` 镜像实现）。
latent 是 workgroup 大小，必须是编译期常量，所以每个宽度是一个独立 SPIR-V 变体
（`tools/compile_shaders.py` 的 `TINY_LATENT_VARIANTS`）。

## 6. 复现

```powershell
.\build.ps1
.\.venv\Scripts\python.exe .\tools\train_student.py --latent 32 --blocks 12 `
  --phase1-epochs 120 --phase2-epochs 25 --trajectory-steps 100 `
  --early-step-repeats 8 --rollout-samples 240 --rollout-steps 8
.\.venv\Scripts\python.exe .\tools\run_tinyhood_reference.py `
  --asset-root .\.work\real_scene\ch10032_tpose --motion ch10032_tpose --steps 10 `
  --model .\.work\hood_data\student32x12.vhood `
  --golden .\.work\real_scene\ch10032_tpose\student32x12_rollout.vhgold
.\verify_hood.ps1 -Motion ch10032_tpose -Solver TinyHood `
  -HoodModel .work\hood_data\student32x12.vhood `
  -Golden .work\real_scene\ch10032_tpose\student32x12_rollout.vhgold `
  -Output results\student32x12_verify.json
.\benchmark_hood_static.ps1 -Scene CH10032 -Solver TinyHood -Warmup 10 -Samples 60 `
  -HoodModel .work\hood_data\student32x12.vhood -Output results\student32x12_ch10032_timing.csv
.\.venv\Scripts\python.exe .\tools\compare_student_stability.py `
  --models "tinyhood_64x4=.work/hood_data/tinyhood64x4.vhood" `
           "student_32x12=.work/hood_data/student32x12.vhood" --steps 120
```

训练约 2.6 小时（145 epoch，RTX 4060 Ti）。不可逐位复现：`aggregate_sum` / `vertex_normals`
使用 `index_add_`，CUDA 上对 float 没有确定性实现，因此无法启用
`torch.use_deterministic_algorithms(True)`。seed 固定，差异在 float 累加顺序量级。
同一原因也决定了 VHOOD 往返校验必须比较**参数**而不是前向输出——两次 CUDA 前向即使权重
逐位相同也会差约 `1e-6`，用前向输出加 `1e-6` 阈值会误报失败（本轮实际发生过）。

原始数据：`student32x12_verify.json`、`student32x12_ch10032_timing.csv`、
`student32x12_grid64_timing.csv`、`student32x12_training_log.jsonl`、
`student_stability_comparison.json`、`partA_*_timing.csv`。

## 7. 后续

1. **保真度**，而不是稳定性。交付模型比 teacher 更硬（5–60 步的 edge P95 低于 teacher），
   120 步与 teacher 轨迹差 `0.79 m`。要贴近 teacher 的褶皱形态，应在阶段 2 的损失里
   提高位置项相对几何惩罚的权重，或直接降低 `--edge-weight` / `--flip-weight`。
2. **模型选择指标偏向结构保守。** 当前 score 只看 edge P95 与翻面率，不看与 teacher 的
   轨迹差，因此会偏好偏硬的解。应把 teacher 位置误差并入选择指标。
3. **`WaveActiveSum` 替换串行 LayerNorm 归约**。`calculateLayerNorm` / `tinyLayerNorm`
   仍在 lane 0 上串行归约，其余 lane 空转。修掉后 `32×15`（完整 teacher 深度）可进 1 ms。
4. 若要求严格贴合 teacher 的动力学，再考虑 GNN 大形预测 + XPBD 约束恢复的混合路径。
