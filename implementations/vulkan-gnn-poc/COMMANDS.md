# 命令参考

本文列出本 PoC 的全部启动命令，参数与默认值均从各脚本的 `param()` 块核对。
所有命令都在本目录（`implementations/vulkan-gnn-poc/`）下执行。

## 先决条件

### 必须用 PowerShell 7（`pwsh`），不能用 `powershell`

脚本使用 `? :` 三元运算符，Windows 自带的 PowerShell 5.1 无法解析，会报一连串
「缺少右“)”」的语法错误。

```powershell
pwsh
```

从 bash / CI 调用时：

```
pwsh -NoProfile -ExecutionPolicy Bypass -File ./build.ps1
```

### Python 环境

`.venv` 实际是 **Python 3.13.7**，安装了 `torch==2.10.0+cpu` 与 `numpy==2.4.4`
（与 `pyproject.toml` 的 pin 一致）。`.python-version` 写的是 `3.11`，`pyproject.toml`
声明 `requires-python >= 3.11`，两者都被满足，但 `uv sync --python 3.11` 会创建**另一个**
环境。现有可用环境的创建方式是：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r model\requirements.txt
```

> **注意：torch 是 CPU-only 构建。** 现在的 TinyHOOD 蒸馏因此只能在 CPU 上训练，这是
> 训练数据预算受限（48 帧 / batch 1 / 960 优化步）的直接原因。要扩大训练规模需要先
> 装 CUDA 版 torch。

### 其他依赖

Vulkan SDK（含 DXC 与 SPIR-V tools）、CMake、Visual Studio 2026 C++ 工具链。
构建包装器 `tools/build-with-vs.cmd` 硬编码了 Visual Studio 18 Community 的
`VsDevCmd.bat` 与 Ninja 路径，安装位置不同需要改这里。

CH10032 资产的离线烘焙额外需要 Unreal Editor、Blender 4.5 和 `uv`。运行时不解析
FBX / UAsset / `.pth` / USD。

---

## 1. 构建

```powershell
.\bootstrap.ps1                          # 拉取锁定 commit + 把 overlay 复制进 .work
.\bootstrap.ps1 -SkipFetch               # 只复制 overlay，不重新拉取上游
.\bootstrap.ps1 -OverwriteWorkEdits      # 丢弃 .work 内比 overlay 更新的改动
.\build.ps1                              # shader 编译 + SPIR-V 校验 + CMake/Ninja + 格式测试
```

`bootstrap.ps1` 严格校验 `upstream.lock.json` 里的 Vulkan / GLM commit。`overlay/` 是样例与
shader 的唯一权威副本，脚本单向复制进 `.work`；若 `.work` 内存在更新的改动，脚本会**报错**
而不是静默覆盖（要覆盖就加 `-OverwriteWorkEdits`）。

`build.ps1` 在构建后断言 `gnncloth.exe` 比 `.work/Vulkan/examples/gnncloth` 下的每个源文件
都新。这是为了防止 ninja 头文件依赖失效造成的假构建：ninja 通过匹配 `cl.exe` 本地化的
`/showIncludes` 前缀学习头依赖，一旦记录的前缀与实际输出字节不符，`.ninja_deps` 会
一条头依赖都不含，改 header 后构建报告成功但不重新链接。恢复方式见报错信息：

```powershell
Remove-Item .work/Vulkan/build-gnn/CMakeCache.txt
Remove-Item -Recurse .work/Vulkan/build-gnn/examples/CMakeFiles/gnncloth.dir
.\build.ps1
```

---

## 2. 资产烘焙（一次性）

```powershell
uv sync --python 3.11
.\tools\fetch_hood_fine15.ps1            # 下载 HOOD 官方 fine15.pth 并转为 VHOOD
.\tools\bake_real_scene.ps1              # CH10032 sprint 动画
.\tools\bake_tpose_scene.ps1             # CH10032 原生 T-Pose（静态）
.\tools\bake_hood_grid_scene.ps1         # 确定性 64x64 规则布片 + 静态球
```

| 脚本 | 参数 |
| --- | --- |
| `fetch_hood_fine15.ps1` | `-Archive <path>`、`-RemoveArchive` |
| `bake_real_scene.ps1` | `-Motion`（默认 `ch10032_sprint`）、`-AnimAsset`、`-Duration`、`-StaticPose`、`-SkipUnrealExport`、`-SkipFine15Golden` |
| `bake_tpose_scene.ps1` | `-SkipUnrealExport`、`-SkipFine15Golden` |
| `bake_hood_grid_scene.ps1` | `-Grid 64`、`-Output` |

`-SkipUnrealExport` / `-SkipFine15Golden` 用于跳过已完成的耗时步骤重跑后半段。

产物全部落在被忽略的 `.work/` 下：`.work/real_scene/<motion>/` 与 `.work/hood_data/`。

---

## 3. 交互运行

```powershell
.\run.ps1 -Grid 32                                              # 规则网格 + Toy GNN + XPBD
.\run.ps1 -Scene CH10032 -Motion ch10032_sprint -Solver Fine15  # 真实角色 + 动画
.\run.ps1 -StaticPose -Solver Fine15                            # CH10032 T-Pose
.\run.ps1 -StaticPose -Solver TinyHood
.\run.ps1 -StaticPose -Solver Toy2L
.\run.ps1 -Scene HoodGrid64                                     # Grid64 + Fine15，无 XPBD
.\run.ps1 -Scene CH10032 -Solver Fine15 -CollisionProjection    # 可选的非黄金后处理
```

| 参数 | 取值 | 默认 |
| --- | --- | --- |
| `-Scene` | `Grid` / `CH10032` / `HoodGrid64` | `Grid` |
| `-Grid` | `16` / `32` / `64` | `32` |
| `-Solver` | `Toy` / `Fine15` / `TinyHood` / `Toy2L` | `Toy` |
| `-Motion` | 任意已烘焙 motion 名 | `ch10032_sprint` |
| `-AssetRoot` / `-HoodModel` | 覆盖默认路径 | 由 Scene/Motion 推导 |
| `-CollisionProjection` | switch | 关 |
| `-StaticPose` | switch | 关 |

两个隐含改写值得注意：

- `-StaticPose` 会强制 `-Scene CH10032`、`-Motion ch10032_tpose`，并把 `Toy` 升级为 `Fine15`。
  所以单独给 `-StaticPose` 就够了。
- `-Scene HoodGrid64` 会强制 `-Motion hood_grid64`、`-Grid 64`，并把 `Toy` 升级为 `Fine15`。

键位：`G` 切换求解器，`R` 重置，`P` 暂停/继续。网格规模在启动时确定，无法在交互中更改。

---

## 4. 正确性验证

这些命令**故意保持 Khronos validation 与 synchronization validation 开启**。性能测量已
拆分到独立路径，见第 5 节。

```powershell
.\verify.ps1                                                # Toy 链路全套
.\verify_hood.ps1                                           # sprint + Fine15（默认）
.\verify_hood.ps1 -Motion ch10032_tpose -Solver Fine15      # T-Pose 单步
.\verify_hood.ps1 -Motion ch10032_tpose -Solver TinyHood    # TinyHOOD 十步
.\ci.ps1                                                    # 不需要 GPU
```

`verify_hood.ps1` 参数：`-Motion`、`-Solver {Fine15|TinyHood}`、`-AssetRoot`、`-HoodModel`。
它只在验证模式回读，严格比对 Python rollout，超出阈值即失败。输出：

| 命令 | 结果文件 |
| --- | --- |
| `-Motion ch10032_sprint -Solver Fine15` | `results/hood_verify.json` |
| `-Motion ch10032_tpose -Solver Fine15` | `results/hood_static_verify.json` |
| `-Solver TinyHood` | `results/tinyhood_verify.json` |

`verify.ps1` 依次跑 Python 参考、C++ `VGNN/VGLD` 负例、Vulkan 黄金样例、1200 帧稳定性、
重置可重复性、双 solver 切换 smoke test 和 synchronization validation。它会调用
`smoke_modes.ps1`，后者**打开一个可见窗口**，因此需要交互式桌面，不适合无头 CI。

`ci.ps1` 不需要 GPU：Python 参考、二进制格式负例、生成的 shader 常量与 `vgnn.py` 一致、
已提交 SPIR-V 与新编译结果一致。

---

## 5. 性能测量

```powershell
.\benchmark_hood_static.ps1 -Scene CH10032    -Solver Fine15   -Warmup 10 -Samples 60
.\benchmark_hood_static.ps1 -Scene HoodGrid64 -Solver Fine15   -Warmup 10 -Samples 60
.\benchmark_hood_static.ps1 -Scene CH10032    -Solver TinyHood -Warmup 10 -Samples 60
.\benchmark_hood_static.ps1 -Scene HoodGrid64 -Solver TinyHood -Warmup 10 -Samples 60
.\benchmark_hood_static.ps1 -Solver Toy2L -Warmup 50 -Samples 200
.\benchmark.ps1                                          # Toy 链路，16/32/64 三档
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `-Scene` | `CH10032` | `CH10032` / `HoodGrid64` |
| `-Solver` | `Fine15` | `Fine15` / `TinyHood` / `Toy2L`（`Toy2L` 不支持 `HoodGrid64`） |
| `-Warmup` | `5` | 丢弃的预热样本数 |
| `-Samples` | `20` | 记录的样本数 |
| `-LockClockMHz` | `2700` | 锁定 SM 时钟；`0` 表示不锁。脚本在 `finally` 中恢复默认 |
| `-Validate` | 关 | 开启后才加 `-v -vl` 与 synchronization validation |
| `-Motion` / `-AssetRoot` / `-HoodModel` / `-Output` | 推导 | 覆盖默认路径 |

### 为什么默认不开 validation、默认锁时钟

同步验证层逐 dispatch 追踪每个 buffer 的访问区间；一个 HOOD 步有约 60 次 dispatch，
由此产生的 CPU 停顿让 GPU 反复回落到低功耗状态。同时这块卡的空闲 SM 时钟是 735 MHz、
最大 3105 MHz，而延迟受限的 compute 步在时钟调控器看来不像负载。

两者叠加会让**同一份工作量在不同运行之间相差 2.2 倍**。仓库里历史记录的
`542.75 ms` 与 `551.81 ms` 就是这样产生的，不能作为任何加速比的分母。

因此：

- 摘要表首列是 `min_ms` —— 最小值跨运行可复现（离散度约 2%），均值会吸收时钟波动。
- 每个 CSV 旁生成 `*_environment.json`，记录 validation 开关、锁定时钟、运行前后的
  时钟 / 温度 / 功耗 / throttle 原因。**一份计时 CSV 只能与用同样方法测出的 CSV 比较。**

`2700 MHz` 是本卡在该负载下能精确保持的最高值（锁 `2900` 只能达到 `2745`），且与未锁定
时实际 boost 到的频率一致。

输出：`results/<stem>_timing.csv`（38 个 timestamp：蒙皮、特征/世界边、4 个 encoder、
每个 block 的 edge/node update、decoder+积分、总计）与 `results/<stem>_stability.json`
（结构指标）。时间区间含阶段间必要的 compute barrier，不含 CPU 提交、绘制和 present。

详见 [`results/KERNEL_OPTIMISATION_RESULTS.md`](results/KERNEL_OPTIMISATION_RESULTS.md)。

---

## 6. 截图与消融

```powershell
.\capture_screenshot.ps1 -Grid 64 -WarmupMilliseconds 15000
.\capture_screenshot.ps1 -Scene CH10032 -Motion ch10032_tpose -Solver Fine15 -SimulationSteps 10 -WarmupMilliseconds 8000
.\capture_screenshot.ps1 -Scene CH10032 -Motion ch10032_tpose -Solver Toy2L  -SimulationSteps 10 -WarmupMilliseconds 1500
.\capture_screenshot.ps1 -Scene HoodGrid64 -SimulationSteps 40 -WarmupMilliseconds 30000
.\ablation.ps1
```

`capture_screenshot.ps1` 参数：`-Scene`、`-Grid`、`-Motion`、`-Solver {Fine15|TinyHood|Toy2L}`、
`-SimulationSteps 0..10000`（0 = 不按步数暂停）、`-WarmupMilliseconds 0..30000`、`-AssetRoot`。
给定 `-SimulationSteps` 可生成跨模型可比较的同步帧。

`ablation.ps1` 在同一个确定性 600 步场景上跑四次，只改加速度来源
（`gnn` / `analytic` / `gravity` / `zero`），比较末帧位置，写入 `results/gnn_ablation.json`。
UI 的 `Acceleration` 下拉可交互切换同样四种模式。

---

## 7. 训练与导出

```powershell
# Toy VGNN（10->16->3）
.\.venv\Scripts\python.exe model\train_export.py
.\.venv\Scripts\python.exe model\verify_export.py

# TinyHOOD 64x4 蒸馏
.\.venv\Scripts\python.exe .\tools\train_tinyhood.py --epochs 20 --train-steps 48 --static-steps 120 --dagger-rounds 0

# Python 参考 rollout（生成黄金数据）
.\.venv\Scripts\python.exe .\tools\run_fine15_reference.py `
  --asset-root .\.work\real_scene\ch10032_tpose --motion ch10032_tpose --steps 10
.\.venv\Scripts\python.exe .\tools\run_tinyhood_reference.py `
  --asset-root .\.work\real_scene\ch10032_tpose --motion ch10032_tpose --steps 10 `
  --golden .\.work\real_scene\ch10032_tpose\tinyhood64x4_rollout.vhgold
```

`train_tinyhood.py` 主要参数：`--epochs`、`--learning-rate`、`--seed`、`--threads`、
`--train-steps`、`--static-steps`、`--resume`，以及 DAgger 相关的 `--dagger-rounds`
（默认 `3`）、`--dagger-steps`、`--dagger-epochs`、`--dagger-learning-rate`。

> **当前交付的权重是用 `--dagger-rounds 0` 训练的**，即完全没有 on-policy 数据。
> `--dagger-rounds 3` 的实验结果保存在 `results/tinyhood64x4_dagger_experiment.json`，
> 闭环表现更差，没有作为运行权重。

其他工具脚本：`tools/export_fine15.py`、`tools/bake_ch10032_scene.py`、
`tools/export_ch10032_animation.py`、`tools/convert_usd_cloth.py`、
`tools/import_motion_samples.py`、`tools/validate_real_assets.py`、
`tools/compare_hood_debug.py`、`tools/compile_shaders.py`、
`model/write_shader_constants.py`、`model/compare_ablation.py`。

---

## 8. 典型工作流

改了 shader、GPU 布局或运行时之后：

```powershell
.\build.ps1
.\verify_hood.ps1 -Motion ch10032_sprint -Solver Fine15
.\verify_hood.ps1 -Motion ch10032_tpose  -Solver TinyHood
.\benchmark_hood_static.ps1 -Scene CH10032 -Solver Fine15 -Warmup 10 -Samples 60
```

只改了 Python 参考或权重打包：

```powershell
.\ci.ps1
.\.venv\Scripts\python.exe model\verify_export.py
```

---

## 附：实测状态

截至本文写作，以下命令在 RTX 4060 Ti / 驱动 `596.36.0.0` 上实际运行并通过：
`bootstrap.ps1`（经 `build.ps1` 间接调用）、`build.ps1`、
`verify_hood.ps1` 的三个变体、`benchmark_hood_static.ps1` 的四个场景/模型组合。

以下命令的参数与行为来自脚本源码核对，**未在本轮实测**：
`run.ps1`、`verify.ps1`、`smoke_modes.ps1`、`benchmark.ps1`、`ablation.ps1`、
`capture_screenshot.ps1`、`ci.ps1`、全部 `tools/bake_*.ps1` 与
`tools/fetch_hood_fine15.ps1`、`train_tinyhood.py`。
