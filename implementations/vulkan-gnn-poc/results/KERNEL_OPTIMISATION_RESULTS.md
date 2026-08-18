# HOOD Vulkan kernel 优化记录

测试机：NVIDIA GeForce RTX 4060 Ti，驱动 `596.36.0.0`，Vulkan SDK 1.4.309，FP32 HLSL/SPIR-V。

本文记录三轮改动：把性能测量方法修正到可复现（阶段 0），修掉 MLP 权重的内存布局缺陷
（阶段 1），以及把 processor layer0 的激励值改为协作载入 groupshared（阶段 2）。
没有改动任何模型、架构、特征或积分方式，三轮之后数值结果仍然逐位不变。

累计结果（锁 2700 MHz、无 validation、10 预热 / 60 采样、min_ms）：

| 场景 / 模型 | 阶段 0 | 阶段 1 | 阶段 2 | 累计 |
| --- | ---: | ---: | ---: | ---: |
| CH10032 Fine15 | 225.72 ms | 79.23 ms | **24.39 ms** | **9.26×** |
| Grid64 Fine15 | 692.79 ms | 261.80 ms | **71.75 ms** | **9.66×** |
| CH10032 TinyHOOD | 12.62 ms | 2.67 ms | **1.65 ms** | **7.64×** |
| Grid64 TinyHOOD | 50.76 ms | 8.46 ms | **4.14 ms** | **12.26×** |


## 1. 阶段 0：为什么之前的性能数字不可比

### 1.1 两个测量缺陷

`benchmark_hood_static.ps1` 过去在计时的同时启用了 Khronos validation（`-v -vl`）和
`VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT`。同步验证层会逐 dispatch
追踪每个 buffer 的访问区间；一个 HOOD 步有约 60 次 dispatch，由此产生的 CPU 停顿让 GPU
在 dispatch 之间反复回落到低功耗状态。

同时设备时钟未锁定。这块卡的空闲 SM 时钟是 `735 MHz`，最大 `3105 MHz`，而一个延迟受限的
compute 步在时钟调控器看来不像负载，因此实际运行时钟在运行之间漂移。

### 1.2 结果：同一份工作量报出 2.2 倍差异

用与 [`hood_static_timing.csv`](hood_static_timing.csv) 完全相同的场景、模型、边数和采样数
重跑（同一 GPU、同一驱动），得到的 CH10032 Fine15 总时间是 `243.998 ms` 均值，而已提交的记录是
`542.752 ms`。逐阶段对比显示这是一个**全局均匀**的 2.2 倍缩放，30 个阶段全部同比例变化，
与工作量无关：

| 阶段 | 已提交 mean (ms) | 重跑 mean (ms) |
| --- | ---: | ---: |
| block_00_edge | 27.677 | 13.651 |
| block_00_node | 4.648 | 2.434 |
| encoder_total | 18.583 | 8.967 |
| processor_15_total | 521.309 | 233.773 |
| total | 542.752 | 243.998 |

`block_00_edge` 的最小值反而很接近（`10.995` 对 `10.412`，差 5%），说明**最小值可复现、
均值不可复现**：旧记录里有大量被时钟回落污染的样本把均值拉高了。

Grid64 的方向相反：已提交记录是 `551.809 ms`，干净重测是 `733.356 ms`。所以旧数字在两个方向
上都不可靠，不能作为任何加速比的分母。

### 1.3 修正后的测量方法

- `benchmark_hood_static.ps1` 默认不再启用 validation。正确性验证仍然完全由
  `verify_hood.ps1` 负责（它保留 validation 与 synchronization validation）。
  需要为验证过的路径计时时可传 `-Validate`。
- 默认把 SM 时钟锁定在 `2700 MHz`（`-LockClockMHz`，0 表示不锁）。这是本卡在该负载下能
  精确保持的最高值：锁 `2900` 只能达到 `2745`，而 `2700` 稳定达到 `2700`，且与未锁定时
  实际 boost 到的频率一致。脚本在 `finally` 中恢复默认时钟。
- 摘要表以 `min_ms` 为首列。最小值是跨运行可复现的统计量。
- 每个 CSV 旁边生成 `*_environment.json`，记录 validation 开关、锁定时钟、运行前后的
  时钟/温度/功耗/throttle 原因。计时 CSV 只能和用同样方法测出的另一个 CSV 比较。

锁定后的复现性（CH10032 Fine15，连续三次独立运行，锁 2100 MHz 时的验证数据）：

| 运行 | total min (ms) | total mean (ms) |
| --- | ---: | ---: |
| 1 | 296.001 | 322.621 |
| 2 | 302.361 | 326.391 |
| 3 | 298.516 | 317.462 |

最小值跨运行离散度 2.1%，均值 2.8%。`total` 的 max/min 从旧记录的 1.35 降到 1.17。

### 1.4 一个功耗层面的独立佐证

干净运行时 `nvidia-smi` 报告 GPU 利用率 100%，但功耗只有 `47–51 W`，而这是一块 160 W 的卡。
SM 被占满却几乎不做数学运算，这正是延迟受限 kernel 的典型特征，与下面的布局诊断一致。

### 1.5 附带发现：header 改动不触发重新构建

这个问题会让上面所有工作失去意义，所以单独记录。

Ninja 通过匹配 `cl.exe` 的 `/showIncludes` 前缀来学习头文件依赖，而 MSVC 会本地化这个前缀。
本构建树记录的 `msvc_deps_prefix` 与 `cl.exe` 实际输出的字节不匹配（编码不同），因此
`.ninja_deps` 里**一条头文件依赖都没有**。后果是：修改 `fine15_gpu_layout.h`、
`hood_runtime.inl` 或任何 `.hlsli` 之后，`build.ps1` 报告成功但不重新链接，
benchmark 测的是上一个二进制。本轮第一次尝试就命中了这个问题（exe 时间戳停留在改动前）。

处理：
- 重新配置构建树后前缀恢复匹配，头文件依赖开始被记录（已验证：修改 header → 重新链接）。
- `build.ps1` 增加一个不依赖依赖扫描器的显式检查：构建后断言 `gnncloth.exe` 比
  `.work/Vulkan/examples/gnncloth` 下的每一个源文件都新，否则报错并给出恢复命令。
  这样这类问题无法再静默发生。

## 2. 阶段 1：权重布局转置

### 2.1 缺陷

PyTorch 的 `nn.Linear.weight` 是 `[out][in]` row-major，`buildGpuModelFor` 原样打包，
shader 则按 `weights[w + lane * inputCount + input]` 索引，其中每个 lane 拥有一个输出通道。
于是相邻 lane 的地址相差 `inputCount` 个 float：

| 矩阵 | 相邻 lane 步长 | 一个 32-lane warp 单次 load 触及 |
| --- | ---: | --- |
| processor layer0（K=384） | 1536 B | 32 条不同的 cache line |
| layer1 / layer2（K=128） | 512 B | 32 条不同的 cache line |

即一个 warp 取回约 4096 B 只使用 128 B。更糟的是 128-lane workgroup 处理 layer0 时的权重
足迹是 `128 × 384 × 4 B = 192 KB`，超过单 SM 的 128 KB L1，缓存行在被复用前就被逐出。

### 2.2 改动

在 `fine15_gpu_layout.h` 中新增 `appendTransposed`，把每个 `Linear.weight` 以 `[in][out]`
存储；bias、LayerNorm 参数和 `nodetype_embedding`（shader 按 `type * 9 + i` 读取）保持原样。
shader 索引统一改为 `weights[w + input * outputs + lane]`，相邻 lane 读相邻 float。

`hiddenLinear` / `tinyHiddenLinear` 增加一个显式的 `outputCount` 行距参数——它不总是 latent
宽度，decoder 第三层只输出 3 维。涉及 8 个 compute shader 与 2 个 `.hlsli`。

### 2.3 数值等价性

这是纯重排，不改变运算顺序。三个黄金验证的 JSON 与已提交版本**逐字节相同**：

| 验证 | 文件 | 结果 |
| --- | --- | --- |
| Fine15 CH10032 sprint，10 步 | `hood_verify.json` | 与已提交版本逐字节相同 |
| Fine15 CH10032 T-Pose，1 步 | `hood_static_verify.json` | 与已提交版本逐字节相同 |
| TinyHOOD CH10032 T-Pose，10 步 | `tinyhood_verify.json` | 与已提交版本逐字节相同 |

具体数值未变：Fine15 10 步 `max_abs = 1.315e-4`、`mean_abs = 1.929e-7`、world edge mismatch 0；
T-Pose 单步 `max_abs = 1.177e-6`；TinyHOOD 10 步 `max_abs = 9.537e-7`。
Khronos validation 与 synchronization validation 均无报错。

### 2.4 实测加速比

全部数据在锁定 2700 MHz、无 validation、10 预热 / 60 采样下测得。

| 场景 / 模型 | 阶段 | 阶段 0 min (ms) | 阶段 1 min (ms) | 加速 | 阶段 0 mean (ms) | 阶段 1 mean (ms) | 加速 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CH10032 Fine15 | encoders | 6.204 | 0.771 | **8.05×** | 7.994 | 0.965 | 8.29× |
| CH10032 Fine15 | 15 processor blocks | 217.443 | 77.446 | **2.81×** | 228.571 | 82.618 | 2.77× |
| CH10032 Fine15 | decoder + integrate | 0.531 | 0.049 | **10.80×** | 0.655 | 0.066 | 9.93× |
| CH10032 Fine15 | **total** | **225.723** | **79.225** | **2.85×** | 237.920 | 84.427 | 2.82× |
| Grid64 Fine15 | encoders | 18.493 | 2.090 | 8.85× | 23.520 | 2.980 | 7.89× |
| Grid64 Fine15 | 15 processor blocks | 666.677 | 256.656 | 2.60× | 706.524 | 277.000 | 2.55× |
| Grid64 Fine15 | decoder + integrate | 1.560 | 0.139 | 11.25× | 2.078 | 0.179 | 11.60× |
| Grid64 Fine15 | **total** | **692.794** | **261.800** | **2.65×** | 733.356 | 281.594 | 2.60× |
| CH10032 TinyHOOD | encoders | 1.721 | 0.177 | 9.74× | 2.132 | 0.194 | 10.99× |
| CH10032 TinyHOOD | 4 processor blocks | 10.074 | 1.979 | 5.09× | 11.960 | 3.171 | 3.77× |
| CH10032 TinyHOOD | **total** | **12.621** | **2.673** | **4.72×** | 15.052 | 4.188 | 3.59× |
| Grid64 TinyHOOD | encoders | 4.882 | 0.473 | 10.32× | 6.057 | 0.577 | 10.49× |
| Grid64 TinyHOOD | 4 processor blocks | 43.801 | 6.786 | 6.45× | 45.299 | 9.420 | 4.81× |
| Grid64 TinyHOOD | **total** | **50.756** | **8.455** | **6.00×** | 53.174 | 11.351 | 4.68× |

未使用权重的阶段不变，符合预期：GPU 蒙皮和 `features + nearest world edge` 在两侧一致
（CH10032 分别为 `0.046 / 0.482 ms`）。

原始数据：`stage0_*_timing.csv`、`stage1_*_timing.csv` 及各自的 `*_environment.json`。

### 2.5 为什么 encoder 拿到 8–11×，而 processor 只有 2.6–2.8×

encoder 和 decoder 的开销几乎全在 `hiddenLinear` 上，它的激励值来自 groupshared，
所以修掉权重访存后它们就接近干净了。

processor 的 layer0 不同：`hood_edge_update.comp` 通过 `edgeValue(element, input)` 读取
384 个输入，每次都是一次带分支的**全局**load。128 个 lane 读的是同一个地址（广播，不存在
非合并问题），但这意味着每个 workgroup 发出 `128 × 384 = 49,152` 次全局 load，只为取 384 个
不同的 float——384 倍的冗余 load 指令。

`hood_node_update.comp` 已经把聚合结果先写进 groupshared（`aggregateMesh` / `aggregateWorld` /
`oldNodeValue`）再进入内积，这正是它比 edge update 便宜得多的原因。这构成了阶段 2。

## 3. 阶段 2：processor layer0 激励值协作载入 groupshared

### 3.1 改动

`hood_edge_update.comp` 与 `tinyhood_edge_update.comp` 在进入内积之前，先由 128（或
`TINY_LATENT`=64）个 lane 各执行 3 次合并 load，把 384（或 192）个输入写进新的 groupshared
数组，然后内积从 groupshared 读取。

`edgeValue(element, lane)`、`edgeValue(element, lane + 128)`、`edgeValue(element, lane + 256)`
这三次访问本身都是合并的：它们分别读 `nodeLatent[receiver * 128 + lane]`、
`nodeLatent[sender * 128 + lane]` 和 `meshIn[element * 128 + lane]`，相邻 lane 读相邻 float。

early-out 条件只依赖 `element` 和 `worldObstacle[element]`，两者在 workgroup 内均为 uniform，
因此整个 workgroup 一起退出，新增的 barrier 是良构的。

### 3.2 数值等价性

`hood_verify.json` 与 `tinyhood_verify.json` 仍与已提交版本逐字节相同。

### 3.3 实测

| 场景 / 模型 | 阶段 | 阶段 1 min (ms) | 阶段 2 min (ms) | 加速 |
| --- | --- | ---: | ---: | ---: |
| CH10032 Fine15 | 15 processor blocks | 77.45 | 22.20 | **3.49×** |
| CH10032 Fine15 | **total** | **79.23** | **24.39** | **3.25×** |
| Grid64 Fine15 | 15 processor blocks | 256.66 | 68.39 | **3.75×** |
| Grid64 Fine15 | **total** | **261.80** | **71.75** | **3.65×** |
| CH10032 TinyHOOD | 4 processor blocks | 1.98 | 0.95 | 2.08× |
| CH10032 TinyHOOD | **total** | **2.67** | **1.65** | **1.62×** |
| Grid64 TinyHOOD | 4 processor blocks | 6.79 | 2.86 | 2.37× |
| Grid64 TinyHOOD | **total** | **8.46** | **4.14** | **2.04×** |

edge update 在 processor 内的占比随之下降：CH10032 从 52.34 / 6.27 ms（edge / node）变为
14.47 / 6.21 ms，Grid64 从 195.83 / 9.88 ms 变为 51.29 / 9.89 ms。node update 未改动，
时间不变，符合预期。

原始数据：`stage2_*_timing.csv`。

## 5. 当前 roofline 位置

按实际规模统计的算术量（CH10032 约 `28.9 GFLOP/步`，Grid64 约 `92.4 GFLOP/步`），
对比 RTX 4060 Ti 的 `22.06 TFLOPS` FP32 峰值：

| 场景 | 阶段 0 | 阶段 1 | 阶段 2 |
| --- | ---: | ---: | ---: |
| CH10032 Fine15 | 128 GFLOPS（0.58%） | 365 GFLOPS（1.65%） | 1185 GFLOPS（峰值 **5.4%**） |
| Grid64 Fine15 | 133 GFLOPS（0.60%） | 353 GFLOPS（1.60%） | 1288 GFLOPS（峰值 **5.8%**） |

一个写好的分块 GEMM 在这些尺寸上通常能到峰值的 25–50%，所以仍有 5–9 倍空间。
布局和冗余 load 都已解决，剩下的按预期收益排序是：

1. **串行 LayerNorm 归约**：`calculateLayerNorm` 在 lane 0 上串行做 256 次 groupshared 读，
   其余 127 个 lane 在两个 barrier 之间空转。MLP 本身的成本已经降了约 9 倍，这一项现在
   占比明显上升。应改为 `WaveActiveSum` 子组归约。
2. **`[loop]` 抑制的 ILP**：内积是完全串行的相关 FMA 链。应展开 ×4、用 4 个独立累加器，
   并以 `float4` 载入权重。
3. **权重零复用**：仍然是一个 graph element 一个 workgroup，同一个权重矩阵被每个元素
   重新流一遍。改为一个 workgroup 处理 32–64 个元素、权重 tile 进 groupshared，即真正的
   分块 GEMM。
4. **node update 的 `O(clothCount)` 障碍物扫描**和 world edge 的 `O(cloth × proxy)` 暴力
   最近点搜索。前者现在占 CH10032 的 6.21 / 24.39 ms，已经不能忽略。

## 6. 实时性现状

| 场景 / 模型 | 阶段 2 min (ms) | 相当于 |
| --- | ---: | --- |
| CH10032 TinyHOOD | 1.65 | 60 Hz 预算（`16.67 ms`）的 10% |
| Grid64 TinyHOOD | 4.14 | 60 Hz 预算内 |
| CH10032 Fine15 | 24.39 | **41 步/秒，已过 30 Hz** |
| Grid64 Fine15 | 71.75 | 13.9 步/秒 |

以上均不含绘制与 present。

两个结论：

- **学生模型的算力预算问题已经不存在**：TinyHOOD 在 CH10032 上只用 `1.65 ms`。制约它的
  完全是训练与闭环稳定性（单步加速度回归 + 二阶无阻尼积分导致的 `n²` 误差累积），
  而不是速度。
- **完整的 Fine15 teacher 在真实角色场景上已经进入 30 Hz 区间**（`24.39 ms`，41 步/秒），
  而且尚未动用上面列出的 4 项优化。这实质性地改变了蒸馏的定位：在决定学生模型要多小之前，
  应该先用干净数字重新评估“是否还需要学生”。原先“先减少 block 数收益最直接”的判断
  是在错误的性能前提下得出的。

## 7. 复现

```powershell
.\build.ps1
.\verify_hood.ps1 -Motion ch10032_sprint -Solver Fine15
.\verify_hood.ps1 -Motion ch10032_tpose  -Solver TinyHood
.\benchmark_hood_static.ps1 -Scene CH10032    -Solver Fine15   -Warmup 10 -Samples 60
.\benchmark_hood_static.ps1 -Scene HoodGrid64 -Solver Fine15   -Warmup 10 -Samples 60
.\benchmark_hood_static.ps1 -Scene CH10032    -Solver TinyHood -Warmup 10 -Samples 60
.\benchmark_hood_static.ps1 -Scene HoodGrid64 -Solver TinyHood -Warmup 10 -Samples 60
```
