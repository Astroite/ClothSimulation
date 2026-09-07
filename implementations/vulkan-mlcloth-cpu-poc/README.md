# MLCloth 三角色 Demo（重构开发版）

本实现展示三个独立实例：左侧纯 MLCloth、中间纯 XPBD、右侧 MLCloth + XPBD。
右侧可在 Settings → Third character 切换为 **GNN + XPBD**，沿用本 Demo 资源并原地引用隔壁 TinyHOOD 权重；实现及验证见 [GNN_PORT.md](GNN_PORT.md)。
**当前仍在开发和验收中，尚未达到最终布料质量和 1080p / 60 FPS 交付标准。**
完整网格的 CPU / GPU 接触一致性、穿透质量、高频表现和长时间性能仍需改进；不要将当前版本作为已验收演示版。

## 本机启动

双击 `start_demo.cmd`，或在本目录运行：

```powershell
./run.ps1
```

也可以直接运行 `.work/Vulkan/build-mlclothcpu/bin/mlclothcpu.exe`。正常启动不需要参数。
启动器读取 `.work/demo/demo.json`；缺少生成文件时会明确报错。

顶部选择动画、播放 / 暂停、重置、同步状态。底部设置时间轴、速度和循环。
解除同步后，可以选择角色并独立改变动画、暂停和时间。设置面板提供视角、根运动、材质、CPU / Vulkan 后端、约束及碰撞参数、推理线程。
`Space` 暂停，`R` 重置，`H` 隐藏 / 显示 UI。鼠标拖动环绕，滚轮缩放。
设置保存在资源目录的 `user-settings.json`；“Defaults”恢复默认设置。
CPU 标量后端用于正确性参照，完整角色运行较慢，切换和预热可能造成等待。

## 数据与算法

- 原始资源仅从 `F:/ArtWorks/Cloth/CH` 等本地源读取，导出写入 `.work/demo`。
- 保留加密模型、45 根驱动骨骼、5,294 个 ML 输出顶点和 30 Hz `.mldrv` 格式。
- 身体使用完整骨架；独立版本化 `.dma` 保存 60 Hz 完整骨架动画。
- 七个动作：待机、走、跑、冲刺、跳跃循环、往返转身、39 秒复杂动作。
- `C10032_Demo.blend` 保留角色及七个可编辑动作；`.reference.dmp` 保存 Blender 独立求值的骨骼和蒙皮参照。
- 纯 ML 不进行物理修正；纯 XPBD 从静止资产和蒙皮绑定预热，不读取 ML 输出；混合模式持续保存物理状态，用粗层引导跟随 ML。
- CPU / Vulkan 求解包含拉伸、面内剪切、二面角弯曲、扫掠身体接触和顶点—面 / 边—边自碰撞。
- 默认身体碰撞为混合表示：腿部 10 节胶囊，肩臂和臀部 STM，加封闭的腹部过渡。纯 XPBD 与 ML 引导的 XPBD 共用该资产，纯 ML 不接受碰撞修正。
- 设置中的 “Body representation” 可切换混合、仅 STM、仅腿部胶囊；“Collision overlay” 显示所选角色的青色 STM / 橙色胶囊透视轮廓。
- `C10032_HybridCollision.blend` 保存完整骨架绑定的可编辑 STM 和胶囊；独立 Blender 检查验证七动作各三个姿态的蒙皮与端点。
- 默认物理 240 Hz、每步两次结构迭代。网络 30 Hz，同步时共享推理，解除同步后各自维护时序。
- 时间跳转恢复检查点并分帧重放。正常 GPU 播放无 CPU 全网格读取；显式验证、截图和几何诊断会读取结果。
- 身体、布料、法线、镜头和碰撞轮廓按统一显示时间插值；暂停和时间跳转显示确定状态。详见 `docs/PRESENTATION_CLOCK.md`。
- “Limit global stretching” 是默认关闭的实验开关：已减少拉长，但部分动作穿透恶化，结果记录在 `docs/TETHER_EXPERIMENT.md`。

目前身体 / 头部使用找到的原始颜色贴图，衣服仍绘制模拟网格和常量材质。
装饰显示网格的离线映射尚未通过验证，因此未接入默认场景。混合碰撞与精确身体表面绑定已进入开发版默认资源，仍需通过完整质量验收。

## 可重复构建和导出

需要本机 Visual Studio C++、CMake、Vulkan SDK、Blender 4.5，以及既有 AILab 运行库和模型。
`tools/build-with-vs.cmd` 使用本机 Visual Studio 18 Community 路径。

```powershell
./bootstrap.ps1 -SkipFetch
./tools/build-with-vs.cmd
./prepare_demo.ps1
# 已有七动作时，只重建身体绑定、混合碰撞和 Blender 检查
./prepare_demo.ps1 -CollisionOnly
```

首次缺少源动作导出时使用 `./prepare_demo.ps1 -ExportSources`，依赖本机既有 Z2Game UE 工程及源动作。
导出脚本不会覆盖 F 盘原始资源。`tools/bake_demo_blender.py` 是主导出入口。

## 验证与诊断

```powershell
# 构建后运行全部已注册测试
ctest --test-dir tests/build --output-on-failure
# 显式 CPU / GPU 测试和完整资产单步对照
./run.ps1 -Frames 1 -ValidateGpu -ValidateAsset
# 非整数物理子步比下的显示插值检查
./run.ps1 -ValidatePresentation
# 应用自己输出帧图（PPM）
./run.ps1 -Frames 120 -Screenshot .work/demo/frame.ppm
# 固定时间步循环诊断（短动作 20 次，复杂动作 3 次）
./tools/validate_demo_rollouts.ps1
```

验证报告写入资源目录：`gpu-validation.json`、`gpu-asset-validation.json`、`performance.json`。
`self-contact-fixture.json` 可由 `tests/build/demo_contact_probe.exe` 独立重放。
性能面板可开始测量和导出。资源切换、显式跳转和离线固定步长诊断不作为实时 60 FPS 验收数据。
当前报告的 `qualified: false` 表示未完成规定时长和质量验收。
本轮发现 Demo 退出后显卡仍有约 96% 占用；已有帧时间受其他 GPU 工作干扰，需要在显卡空闲时重新验收。
循环脚本保存终态几何诊断和截图；它不等于逐帧穿模审查，也不等于实时性能验收。碰撞资产、区域划分及已知限制见 `docs/HYBRID_COLLISION.md`。
最新接触方向、粗层引导、确定性 gather 与受控对照见 `docs/CONTACT_SOLVER_UPDATES.md`；当前仍有超标穿透和局部拉伸。
默认动画已修复循环末尾的异常减速和模型空间过渡，生成资源有独立验证与历史备份，见 `docs/ANIMATION_TRANSITIONS.md`。

近期验证情况与剩余工作见 `IMPLEMENTATION_STATUS.md`。旧 PoC 文档仅保存在 `LEGACY_POC.md`，其中启动参数、等预算和旧投影器说明不适用于新 Demo。
