# Vulkan GNN 最小布料验证

这个 PoC 打通以下固定链路：

`纯 PyTorch 训练 → VGNN v1 固定 FP32 权重 → HLSL Compute → SPIR-V → Vulkan 常驻推理 → 顶点缓冲直接绘制`

渲染、ping-pong 缓冲和跨队列同步以 MIT 许可的 Sascha Willems
`computecloth` 为基础。bootstrap 只获取锁定 commit，并把本目录 overlay
复制到生成的 `.work/Vulkan`，不会把上游历史提交到本仓库。布料、球体、
checker 材质和天空均为程序生成，所以不需要 `Vulkan-Assets`。

## 已实现范围

- 16×16、32×32、64×64 固定三角网格；无自环的 8 邻域 CSR。
- 交互场景采用竖直初始布片并固定整条顶边，形成类似晾衣架的悬挂约束。
- 默认启用运动学球体，轨迹为
  `x=1.2·sin(0.7t), z=0.65+0.55·sin(1.1t)`。解析球速一并传入 shader，
  碰撞在球体相对速度空间中执行零恢复法向投影，因此移动球会向布料传递动量。
- graphics path 先绘制无深度写入的全屏程序化天空，包含深蓝天顶、浅色地平线
  和柔和太阳；布料使用双面 key/fill/hemisphere 光照，球体使用提亮后的暖色材质。
- 固定 10→16→3 Gather GNN，两层分别执行自身与邻居均值线性变换；第一层 ReLU。
- 每顶点一个 32-lane workgroup；隐藏状态、位置和速度全程驻留 GPU。
- `GnnLayer0CS` 与 `GnnLayer1IntegrateCS` 之后追加默认 8 次完整 XPBD 迭代：
  每条无向约束边拥有独立累计拉格朗日乘子 λ，并使用
  `α̃ = compliance / Δt²`。每个时间步在 GPU 上清零 λ，在迭代内持续累积。
- XPBD 包含水平/垂直拉伸、两条对角剪切以及两跳水平/垂直弯曲距离约束。
  约束被拆成 16 个内部无共享顶点的颜色批次，shader 可原地同时修正两个端点，
  不需要浮点原子操作。最后按约束后位置重建速度，再执行硬固定点与球体投影。
- 推理、积分和每个颜色批次之间均有 compute barrier；最终位置缓冲直接作为
  vertex buffer 绘制。当前弯曲项是长边距离近似，不是三角形二面角约束；仍未
  实现自碰撞。
- 原始 mass-spring 路径保留，UI 可切换求解器、暂停、重置和风力；`G` 切换，`R` 重置，`P` 暂停。
- `--gnn-verify` 只在验证模式回读：检查黄金单步、1200 帧健康状态，以及 600 帧后的重置回放。
- `--gnn-benchmark` 预热 200 帧、采集 1000 帧 timestamp，正常渲染和 benchmark 均无 host readback。

## 先决条件

当前首轮目标是 Windows x64。需要 Vulkan SDK（含 DXC 和 SPIR-V tools）、
CMake、Visual Studio 2026 C++ 工具链和 Python 3.13。已提交的权重使 PyTorch
不成为构建或运行前置条件；仅在重训时需要。

## 一键 bootstrap 与构建

在本目录的 PowerShell 中执行：

```powershell
.\bootstrap.ps1
.\build.ps1
```

`bootstrap.ps1` 会严格检查 `upstream.lock.json` 中的 Vulkan 与 GLM commit。
`build.ps1` 重新编译并验证所有 HLSL SPIR-V，使用 CMake/Ninja 构建样例和
独立二进制格式测试。当前构建包装器指向 Visual Studio 18 Community；若安装
位置不同，可修改 `tools/build-with-vs.cmd` 的 `VsDevCmd.bat` 与 Ninja 路径。

## 交互运行

```powershell
.\run.ps1 -Grid 32
```

UI 显示当前求解器、节点/有向边/XPBD 约束数量，以及两层和总 GNN GPU 时间。
GNN 模式下可以把 `XPBD iterations` 在 0–16 间调节，并分别调整 stretch/shear
和 bend compliance（界面单位为 `×1e-6`），以及速度阻尼；默认值为 8、1、
10000 和 1.5。迭代数设为 0 时仍执行积分后的固定点、碰撞和速度 finalize，但跳过 XPBD 约束求解。网格
规模在启动时选择，以避免在交互帧中重建全部 Vulkan 资源。`Moving sphere`
可随时关闭；暂停会同时停止布料和球体，重置会把运动相位归零。

## 验证

```powershell
.\verify.ps1
```

脚本依次运行 Python 参考、C++ `VGNN/VGLD` 加载器负例、Vulkan 黄金样例、
1200 帧稳定性、重置可重复性、两种求解器切换/重置 smoke test，以及 Khronos
synchronization validation。交互 smoke test 会先让移动球与 GNN/XPBD 布料接触
2 秒，再切换并重置两种 solver。
结果写入 `results/gnn_verify.json` 与 `results/validation_output.txt`。

当前 RTX 4060 Ti 实测：`max_abs=1.907348633e-06`、
`mean_abs=3.117110078e-07`、健康异常 0、重置回放差 0。1200 帧检查同时报告
布料 AABB、最大拉伸/弯曲应变，以及命中加速度/速度钳位的顶点数——黄金场景
下 402/1024 顶点触发加速度钳位、最大拉伸应变 0.81。完整记录见
[`results/RESULTS.md`](results/RESULTS.md)。

## 消融：网络到底贡献了什么

```powershell
.\ablation.ps1
```

同一个确定性 600 步场景跑四次，只改加速度来源（`gnn`、`analytic` 直接求值训练
目标、`gravity` 去掉邻居耦合、`zero` 完全无加速度），比较末帧位置，结果写入
`results/gnn_ablation.json`。UI 的 `Acceleration` 下拉可交互切换同样四种模式。

实测结论：整个图消息传递带来的差异只有 3.5 mm 平均位移（近刚性 XPBD 已经
把 Laplacian 项要近似的东西约束住了），而网络相对其训练目标的自身误差是该
效应的 **14.7 倍**。因此本 PoC 证明的是部署链路，不是学习到的动力学。

## Benchmark

```powershell
.\benchmark.ps1
```

脚本按 16、32、64 三种网格顺序执行，生成 `results/gnn_benchmark.csv`。
每次运行预热 200 帧、采样 1000 帧，记录 GPU、驱动、节点/边/XPBD 约束数量、
第一层、`第二层 + 积分 + XPBD`、总时长以及 p95。本阶段只检查计时有效且扩展趋势合理，
不设置武断的毫秒门槛。benchmark 保持球体静止，以隔离固定 compute 工作量。

可用 `.\capture_screenshot.ps1 -Grid 64 -WarmupMilliseconds 15000` 在运行约
900 帧后暂停并生成交互窗口截图 `results/gnn_cloth.png`。

## 重新训练与导出

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r model\requirements.txt
.\.venv\Scripts\python.exe model\train_export.py
.\.venv\Scripts\python.exe model\verify_export.py
```

训练使用固定 seed `20260817` 和纯 PyTorch。监督目标为：

`18 * (邻居平均位移 - 自身位移) - 0.9 * 速度 + 外部加速度`

导出物包括严格定长的 `VGNN v1` `model.bin`、元数据 `model.json` 和
`golden.bin`。运行时只解析二进制并检查 magic、版本、维度、长度、保留字段与
CRC32。`model.json` 额外保存训练参数和 SHA-256。

## 目录

- `model/`：训练、参考推理、格式实现和已提交权重/黄金数据。
- `overlay/`：新增 Vulkan 样例和 HLSL/SPIR-V。
- `tools/`：shader 与本机构建包装器。
- `tests/`：不依赖 Vulkan 的 C++ 格式加载器测试。
- `results/`：RTX 4060 Ti 验证、benchmark 和截图。
- `upstream.lock.json`：上游源码锁定。

本 PoC 使用 MIT 许可；上游归属和参考说明见
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。通用 checkpoint 导入、
ONNX、FP16、边 MLP、多层图、动态图、自碰撞和 UE5/RDG 均不在首轮范围内。
