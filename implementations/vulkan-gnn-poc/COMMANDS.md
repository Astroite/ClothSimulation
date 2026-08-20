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

### 2.1 CH10032 动画 / 模型库批量导出

`bake_real_scene.ps1` 一次只处理一条动画。要一次性从 Z2Game 工程取一批资产用于评测，
用下面这条；清单在 `tools/ch10032_export_manifest.json`（31 条动画 + 2 个骨骼网格 +
4 个 physics asset + 4 个 skeleton），产物落在 `.work/ch10032_library/`。

```powershell
.\tools\export_ch10032_assets.ps1                    # 全量
.\tools\export_ch10032_assets.ps1 -Tier skirt        # 只要 6 条美术为裙子调过的
.\tools\export_ch10032_assets.ps1 -Only sprint_skirt,body -Force
```

| 脚本 | 参数 |
| --- | --- |
| `export_ch10032_assets.ps1` | `-Tier all\|skirt\|locomotion`、`-Only <id,...>`、`-Force`、`-OutputRoot`、`-UnrealEditor`、`-Project` |
| `validate_ch10032_exports.py` | `--library-root`、`--min-bones 100`、`--sample-frames 8`（经 Blender 运行） |

导出完成后校验 FBX 真的可用（骨架、动画曲线、帧数、蒙皮、NaN）：

```powershell
& 'C:\Program Files\Blender Foundation\Blender 4.5\blender.exe' --background --factory-startup `
    --python tools\validate_ch10032_exports.py -- --library-root .work\ch10032_library
```

> **调用方式是有讲究的，不要「顺手简化」。** 本工程给引擎打过补丁，
> `FPythonScriptPlugin::StartupModule` 在 `IsRunningCommandlet() || FApp::IsUnattended()`
> 时直接 return，因此 `-run=pythonscript` 和任何带 `-unattended` 的调用都拿不到 Python，
> 且没有命令行开关能救回来（`-ForceEnablePython` 也不行）。可行的方式是
> `-ExecCmds="py <script>"` 且**两者都不传**。另外**不能加 `-nullrhi`**：
> SkeletalMesh 导出会在 `GetCPUSkinnedVertices` 断言失败。
>
> 走 `OBJ EXPORT` 控制台命令则完全是另一套限制：它只认被
> `PackagesToBeFullyLoadedAtStartup` 预载的那一个包，而重复传 `-ini:` 只有最后一个生效，
> 所以那条路要每个资产启动一次编辑器（约 33 次）。Python 路线一次启动全部搞定。

`data/` 下 physics asset 以 `.t3d` 为准（`SkeletalBodySetups` 不是反射属性，
`get_editor_property` 拿不到，只有 `ObjectExporterT3D` 会输出），旁边的 `.json` 是摘要。
skeleton 的 `.json` 同时给 `parent_local_*` 与 `component_*` 两个空间，
前者与 `Assets/Characters/CH10032/SK_JZ_CH_10032_Body.json` 的约定一致。

### 2.2 把导出的库烘成场景资产

导出出来的是 FBX，而求解器与 `tools/recovery_probe.py` 吃的是
`.work/real_scene/<motion>/` 下的 `.vchar` / `.vanim` / `.vcloth2`。`bake_real_scene.ps1`
的 `-AnimationFbx` 直接指向库里任意一条，配 `-SkipUnrealExport` 就不再启动编辑器：

```powershell
foreach ($m in 'sprint_skirt','attack_01_skirt','guard_skirt') {
    .\tools\bake_real_scene.ps1 -Motion $m `
        -AnimationFbx ".work/ch10032_library/animations/$m.fbx" `
        -SkipUnrealExport -SkipFine15Golden
}
```

**约 9.7 秒一条**，31 条全量约 5 分钟。`-SkipFine15Golden` 是有意的：那步要跑 Fine15 的整段
teacher rollout，而 Python 侧的评测工具自己在内存里算 teacher 参考，不读 `.vhgold`。
要给 Vulkan 侧做逐步对拍时才需要去掉这个开关。

> **两条路产出的数据是等价的，已验证。** `sprint_skirt.fbx` 就是 `bake_real_scene.ps1`
> 默认那条 `AS_C10032_ArmedSprint_Skirt`，两边烘出来的 `ch10032.vchar` /
> `ch10032_lower.vcloth2` / `.vanim` 的 **`payload_sha256` 三个全部相同**
> （`7e77d2a4` / `ca67f21d` / `b2f2f21d`）。只有 `file_sha256` 不同，因为文件头里嵌了输入
> FBX 的 `source_sha256`（`47c52d58` 对 `fb445152`）—— 差的是来源记录而不是数据。
> 也就是说 UE Python 导出器与 `OBJ EXPORT` 给出的 FBX 字节不同，但解出的蒙皮矩阵逐位一致，
> **所以新老场景的评测结果可以直接混用比较。**

skirt 档的帧数（262 / 346 / 202 / 189 / 202）都远长于既有的 `ch10032_sprint`（62），
所以 120 步的评测在 1× 下不会把片段播完 —— 对 `--frame-scales` 的速度轴是更干净的输入。

---

## 3. 交互运行

```powershell
.\run.ps1 -Grid 32                                              # 规则网格 + Toy GNN + XPBD
.\run.ps1 -Scene CH10032 -Motion ch10032_sprint -Solver Fine15  # 真实角色 + 动画
.\run.ps1 -StaticPose -Solver Fine15                            # CH10032 T-Pose
.\run.ps1 -StaticPose -Solver PostCvpr                         # 官方层级 HOOD
.\run.ps1 -StaticPose -Solver TinyHood
.\run.ps1 -StaticPose -Solver Toy2L
.\run.ps1 -Scene HoodGrid64                                     # Grid64 + Fine15，无 XPBD
.\run.ps1 -Scene CH10032 -Solver Fine15 -CollisionProjection    # 可选的非黄金后处理
```

| 参数 | 取值 | 默认 |
| --- | --- | --- |
| `-Scene` | `Grid` / `CH10032` / `HoodGrid64` | `Grid` |
| `-Grid` | `16` / `32` / `64` | `32` |
| `-Solver` | `Toy` / `Fine15` / `PostCvpr` / `TinyHood` / `Toy2L` | `Toy` |
| `-Motion` | 任意已烘焙 motion 名 | `ch10032_sprint` |
| `-AssetRoot` / `-HoodModel` | 覆盖默认路径 | 由 Scene/Motion 推导 |
| `-CollisionProjection` | switch | 关 |
| `-StaticPose` | switch | 关 |

两个隐含改写值得注意：

- `-StaticPose` 会强制 `-Scene CH10032`、`-Motion ch10032_tpose`，并把 `Toy` 升级为 `Fine15`。
  所以单独给 `-StaticPose` 就够了。
- `-Scene HoodGrid64` 会强制 `-Motion hood_grid64`、`-Grid 64`，并把 `Toy` 升级为 `Fine15`。

键位：`G` 切换求解器，`R` 重置，`P` 暂停/继续。网格规模在启动时确定，无法在交互中更改。

### 3.1 A/B/C 三方同屏对照

同一套骨骼动画同时驱动三份布料并排显示 —— **A 纯网络（蓝）/ B 纯约束（橙）/ C 混合（绿）**，
动画与播放速度可运行时切换。它回答的是标量指标答不了的问题：`edge_p95` 说不出 12.3 是裙摆炸开、
几个三角形拉长还是整体穿进腿里，而这三种情况的工程含义完全不同。

```powershell
.\run.ps1 -Scene CH10032 -Motion sprint_start -Solver TinyHood `
    -HoodModel .work\hood_data\student32x12_r1.vhood -Compare
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `-Compare` | 关 | 隐含 `-Xpbd`；与 `-Solver Toy`/`Toy2L`/`PostCvpr` 互斥 |
| `-XpbdIterations` | `128` | C 支的 sweep 次数 |
| `-XpbdIterationsB` | `228` | **B 支单独的次数 —— 等 GPU 预算而非等迭代数** |
| `-FrameStep` | `1` | 播放速度 1–4×，即 `--frame-scales` 的交互版 |
| `-CompareSpacing` | `1.2` | 三份之间的间距（米） |
| `-HoldLastFrame` | 关 | 片段播完后**钳在末帧**而不是循环 |

overlay 里可以逐支勾掉、拖间距/相机距离/B 迭代数、从下拉菜单换动画（**列出 `.work/real_scene/`
下所有该角色的动画**，不重启进程）、拖播放速度、切换末帧钳住。

> **`-HoldLastFrame` 是评估恢复力的必需项。** 默认循环播放对单支演示更顺，但它藏掉了对照里
> 信息量最大的一半：C 在难片段上的超冲会在动画停下后完全松回（`sprint_start` 第 60 → 90 步是
> 10.364 → 1.219），而 A 不会 —— **循环播放的片段永远不会停。** 钳住末帧同时也让渲染器与
> `tools/recovery_probe.py` 同口径（后者用 `min(frame_of(step), frame_count - 1)`）。

> **`-XpbdIterationsB` 为什么不是 128。** C 的成本是 GNN 0.924 ms + 128 × 9.437 µs = 2.154 ms，
> 等预算给 B 的是 2.154/0.009437 ≈ **228** 次（`results/RECOVERY_SPEED_RESULTS.md` §0）。两边都给
> 128 会让 B 只拿到 56% 预算 —— 那正是 `results/GATE_G0_RESULTS.md` 已修正过的 G0 缺陷。
> 渲染器里 B 还额外付一次 feature pass，所以 228 是**略微超配** B，方向保守。

> **对照模式不是性能测量路径。** 三支共用一套 timestamp 槽位（一个槽位每帧只能写一次），
> 面板里的 ms 只反映最后一支。性能一律用 `benchmark_hood_static.ps1`（第 5 节）。

> **切动画不重载 `.vchar` / `.vcloth2` / `.vxpbd`** —— 它们按服装共享。这既是正确性保证，
> 也正是"零逐动画配置"这个主张本身的演示。所以对照模式**只认衣服级的
> `.work/real_scene/ch10032_lower.vxpbd`**，不用各动作目录下的那几份逐动作标定（混用会让跨动作
> 比较的标定不一致）。缺这个文件会明确报错，不静默降级。烘法：

```powershell
.\.venv\Scripts\python.exe -B tools\bake_xpbd_constraints.py --scene sprint_start `
    --calibration teacher --output .work\real_scene\ch10032_lower.vxpbd `
    --report results\ch10032_lower_vxpbd.json
```

标定取 `sprint_start` 是因为它是全库根运动 jerk 最大的一条（0.126 m/frame²），而
`GATE_G0_RESULTS.md` §11 实测的结论是**标定可跨动作迁移、且应该用最动态的动作来标定**。

#### 定量对照（无需交互）

`tools/compare_probe.ps1` 跑固定步数后落 `results/*.json`，含逐支的
`edge_length_ratio.p95` —— 与 Python 探针的 `edge_p95` **同参照、同边表**，可直接对数。

```powershell
.\tools\compare_probe.ps1 -Motion sprint_start -Steps 60 -Branches ABC
.\tools\compare_probe.ps1 -Motion ch10032_sprint -Steps 60 -Branches C -Single   # 回归对照
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `-Steps` | `120` | 片段播完后钳在末帧，所以可以超过帧数（见下） |
| `-Branches` | `ABC` | 字母子集；`C` 单支用于回归 |
| `-Single` | 关 | 去掉 `--hood-compare`，走原来的单支路径 |
| `-XpbdAsset` | 衣服级 | 指向逐动作 `.vxpbd` 可单独衡量标定的影响 |
| `-Loop` | 关 | 循环播放而不是钳住末帧（**会让跨语言对数失效**） |

> **口径：默认钳住末帧，因为 Python 探针也钳住。** 如果加 `-Loop`，渲染器会在 62 帧的
> `ch10032_sprint` 上重头播第二遍，而探针的第 120 步是 58 步**静态松弛**之后的状态（C 正是靠这个
> 赢末态）—— 两边测的就不是同一件事了。实测（对探针的同步数值）：
>
> | 动作 | 步 | A | B | C |
> | --- | ---: | --- | --- | --- |
> | `ch10032_sprint` | 60 | 4.604 / 4.978 = **0.92** | 2.558 / 2.545 = **1.01** | 2.965 / 2.636 = **1.12** |
> | `sprint_start` | 60 | 5.453 / 5.253 = **1.04** | 1.908 / 2.491 = 0.77 | **10.430 / 10.364 = 1.01** |
> | `sprint_start` | 120 | 8.197 / 9.472 = 0.87 | 2.719 / 2.754 = **0.99** | **1.291 / 1.287 = 1.00** |
>
> 两条独立实现的差异在已确立的跨进程噪声量级内（`GATE_G0_RESULTS.md` §8），
> 且**C 在 `sprint_start` 上先超冲到 10.4 再完全松回 1.29 这条曲线被 1% 内复现** ——
> 它不是 Python 侧的仪表问题。
>
> **`compare_probe.ps1` 需要交互桌面，窗口必须可见**：上游 sample 基类在最小化时停止渲染，
> 而模拟由渲染循环驱动，最小化跑出来的是一份全零的 JSON。

> **一个踩过的坑：这批 `.ps1` 会静默吞掉拼错的参数。** 它们的 `param()` 只用了
> `[ValidateSet]` / `[ValidateRange]`，没有 `[CmdletBinding()]` 也没有 `[Parameter()]`，所以不是
> advanced function —— 未绑定的实参落进 `$args` 而**不报错**。给 `capture_screenshot.ps1` 传一个它
> 当时还没有的 `-HoldLastFrame` 时，脚本照常成功、截图却是循环播放的。**加了新开关就要确认
> 它真的在 `param()` 里。**

---

## 4. 正确性验证

这些命令**故意保持 Khronos validation 与 synchronization validation 开启**。性能测量已
拆分到独立路径，见第 5 节。

```powershell
.\verify.ps1                                                # Toy 链路全套
.\verify_hood.ps1                                           # sprint + Fine15（默认）
.\verify_hood.ps1 -Motion ch10032_tpose -Solver Fine15      # T-Pose 单步
.\verify_hood.ps1 -Motion ch10032_tpose -Solver PostCvpr    # PostCVPR 十步
.\verify_hood.ps1 -Motion ch10032_tpose -Solver TinyHood    # TinyHOOD 十步
.\ci.ps1                                                    # 不需要 GPU
```

`verify_hood.ps1` 参数：`-Motion`、`-Solver {Fine15|PostCvpr|TinyHood}`、`-AssetRoot`、`-HoodModel`、
`-Golden`、`-Output`。它只在验证模式回读，严格比对 Python rollout，超出阈值即失败。
验证一个重训的学生时用 `-HoodModel` / `-Golden` / `-Output` 三个覆盖，避免覆盖已有结果文件：

```powershell
.\verify_hood.ps1 -Motion ch10032_tpose -Solver TinyHood `
  -HoodModel .work\hood_data\student32x12_r1.vhood `
  -Golden .work\real_scene\ch10032_tpose\student32x12_r1_rollout.vhgold `
  -Output results\student32x12_r1_verify.json
```

输出：

| 命令 | 结果文件 |
| --- | --- |
| `-Motion ch10032_sprint -Solver Fine15` | `results/hood_verify.json` |
| `-Motion ch10032_tpose -Solver Fine15` | `results/hood_static_verify.json` |
| `-Solver PostCvpr` | `results/postcvpr_verify.json` |
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
.\benchmark_hood_static.ps1 -Scene CH10032    -Solver PostCvpr -Warmup 10 -Samples 60
.\benchmark_hood_static.ps1 -Scene HoodGrid64 -Solver Fine15   -Warmup 10 -Samples 60
.\benchmark_hood_static.ps1 -Scene CH10032    -Solver TinyHood -Warmup 10 -Samples 60
.\benchmark_hood_static.ps1 -Scene HoodGrid64 -Solver TinyHood -Warmup 10 -Samples 60
.\benchmark_hood_static.ps1 -Solver Toy2L -Warmup 50 -Samples 200
.\benchmark.ps1                                          # Toy 链路，16/32/64 三档
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `-Scene` | `CH10032` | `CH10032` / `HoodGrid64` |
| `-Solver` | `Fine15` | `Fine15` / `PostCvpr` / `TinyHood` / `Toy2L`（`Toy2L` 不支持 `HoodGrid64`） |
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
.\capture_screenshot.ps1 -Scene CH10032 -Motion ch10032_tpose -Solver PostCvpr -SimulationSteps 30
.\capture_screenshot.ps1 -Scene CH10032 -Motion ch10032_tpose -Solver Toy2L  -SimulationSteps 10 -WarmupMilliseconds 1500
.\capture_screenshot.ps1 -Scene HoodGrid64 -SimulationSteps 40 -WarmupMilliseconds 30000
.\ablation.ps1
```

`capture_screenshot.ps1` 参数：`-Scene`、`-Grid`、`-Motion`、`-Solver {Fine15|PostCvpr|TinyHood|Toy2L}`、
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

# 学生模型蒸馏（32 latent x 12 block，CH10032 为 0.924 ms）
# 约 2.6 小时 / 145 epoch（RTX 4060 Ti）。默认的 20+10 epoch 只够验证管线，会明显欠训练。
#
# 注意：这条命令**不再复现**已交付的 student32x12 权重。第二轮改掉了几何惩罚的参照、
# 损失的样本加权和模型选择指标（见 results/STUDENT_STABILITY_ROUND2.md），其中选择指标
# 是结构性改动，没有开关可以还原。交付权重作为文件保留在 .work/hood_data/ 下，
# 产生它的配方在 git 历史里。
.\.venv\Scripts\python.exe .\tools\train_student.py --latent 32 --blocks 12 `
  --phase1-epochs 120 --phase2-epochs 25 --trajectory-steps 100 `
  --early-step-repeats 8 --rollout-samples 240 --rollout-steps 8

# 直接在闭环指标上搜索（当前推荐权重 student32x12_r1 就是这样产生的）
# 梯度下降在这一步是有害的，见 results/STUDENT_STABILITY_ROUND2.md
.\.venv\Scripts\python.exe .\tools\refine_student.py `
  --resume .work\hood_data\student32x12.pt --tag _r1 `
  --iterations 500 --steps 180 --eval-repeats 1 --confirm-repeats 4 `
  --sigma 1.0e-3 --sigma-max 5.0e-3 --scenes "ch10032_tpose:ch10032:180"

# 只测某个架构的 GPU 成本（随机权重；计时与权重值无关）
.\.venv\Scripts\python.exe .\tools\export_random_student.py --latent 32 --blocks 15

# 比较学生的结构保持时长 + teacher 基准的 score 分解
.\.venv\Scripts\python.exe .\tools\compare_student_stability.py `
  --models "shipped_32x12=.work/hood_data/student32x12.vhood" `
           "refined_32x12=.work/hood_data/student32x12_r1.vhood" `
  --steps 360 --thresholds 2.0 3.0 5.0

# TinyHOOD 64x4 蒸馏（历史配方，复现已记录的 64x4 权重）
.\.venv\Scripts\python.exe .\tools\train_tinyhood.py --epochs 20 --train-steps 48 --static-steps 120 --dagger-rounds 0

# Python 参考 rollout（生成黄金数据）
.\.venv\Scripts\python.exe .\tools\run_fine15_reference.py `
  --asset-root .\.work\real_scene\ch10032_tpose --motion ch10032_tpose --steps 10
.\.venv\Scripts\python.exe .\tools\run_tinyhood_reference.py `
  --asset-root .\.work\real_scene\ch10032_tpose --motion ch10032_tpose --steps 10 `
  --model .\.work\hood_data\student32x12_r1.vhood `
  --golden .\.work\real_scene\ch10032_tpose\student32x12_r1_rollout.vhgold
```

`train_student.py` 主要参数：`--latent {32,64}`、`--blocks 1..15`、`--device`、
`--phase1-epochs`、`--phase2-epochs`、`--rollout-steps`（阶段 2 展开深度）、
`--rollout-samples`、`--trajectory-steps`、`--noise-sigma`、`--edge-weight`
（配 `--edge-lower` / `--edge-upper` 的护栏带）、`--degenerate-weight`、
`--edge-match-weight`、`--dagger-rounds`、`--resume`、`--tag`、
`--eval-steps`、`--stiff-weight`、`--drift-weight`、`--over-cap`、`--no-normalise-fit`。
它取代 `train_tinyhood.py`：两阶段训练（单步拟合后接多步 rollout 反传）、teacher 重标注的
噪声注入、四个场景样本、按闭环结构稳定性而不是 teacher-forced MSE 选模型。

> **想提高长程稳定性的话，这个脚本不是正确的工具。** 实测：从已收敛的权重继续训练，
> 无论 phase 1（单步）还是 phase 2（8–16 步展开）、无论学习率降到多低，都会让闭环变差；
> 而比梯度步大 10 倍的随机权重扰动反而改善 37%。用 `refine_student.py`。
> 详见 [`results/STUDENT_STABILITY_ROUND2.md`](results/STUDENT_STABILITY_ROUND2.md)。
>
> `--no-normalise-fit` 用来复现交付 32×12 的加权方式。默认（归一化）会把稳态单步方差解释率
> 从 `0.357` 提到 `0.93`，但闭环稳定性随之差 7 倍 —— 这两者在本架构上是对立的。

`refine_student.py` 直接在闭环指标上做进化搜索，不用梯度。主要参数：
`--resume`（必填 `.pt`）、`--mode {hill,es}`、`--iterations`、`--steps`、
`--scenes`（`key:asset_stem:steps` 三元组，多个则取均值）、`--eval-repeats`、
`--confirm-repeats`、`--sigma` / `--sigma-min` / `--sigma-max`、
`--var-floor-ratio`（拒绝「预测得更少」的退化解）、`--stiff-weight` / `--drift-weight` / `--over-cap`、
`--tag`。它每次刷新 best 就立刻落盘，所以中途 Ctrl-C 不会丢结果。

> 单场景搜索会**过拟合被打分的场景**：只打分 `ch10032_tpose` 时它的 360 步 score
> 从 `2.743` 降到 `0.259`，同时 `hood_grid64` 从 `3.050` 变差到 `4.066`。
> 多场景均值是正确做法，但本轮在噪声范围内没有再拿到改进。

> **需要 CUDA 版 torch。** 当前 `.venv` 是 `torch 2.10.0+cu128`，一轮训练约 21 分钟；
> CPU-only 构建下同样配置约 10 小时，实际不可迭代。
>
> **训练不可逐位复现。** `aggregate_sum` / `vertex_normals` 使用 `index_add_`，CUDA 上
> 对 float 没有确定性实现，因此无法启用 `torch.use_deterministic_algorithms(True)`。
> seed 已固定，差异在 float 累加顺序量级。

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
