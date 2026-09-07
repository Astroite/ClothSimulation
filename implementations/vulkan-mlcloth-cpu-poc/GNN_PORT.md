# GNN + XPBD 算法移植

第三角色现在可以在 Settings → Third character 中切换 **MLCloth + XPBD** 与 **GNN + XPBD**。前两个角色保持纯 MLCloth、纯 XPBD。启动方式仍为双击 `start_demo.cmd`，默认继续使用 MLCloth 混合模式；保存用户偏好后可以记住 GNN 选择。

本轮只移植算法代码，不转移隔壁的角色、动画、裙装网格、碰撞代理或材质，也不重新训练。GNN 直接处理本 Demo 的 5,294 顶点、完整骨架采样、现有身体绑定和胶囊 + STM 碰撞。权重原地读取 `../vulkan-gnn-poc/.work/hood_data/student32x12_r1.vhood`，SHA-256 为 `07345c49ebd50c27e5236239832cc483d55dfdcf9cd204389a7d43c875ce261e`。

可在 Demo manifest 的可选 `gnn_model` 字段填写相对 manifest 的权重路径；默认路径是 `../../../vulkan-gnn-poc/.work/hood_data/student32x12_r1.vhood`。支持 32 通道、1–15 个消息传递块的 TinyHOOD 权重；本机默认是 12 块。模型加载与张量格式校验在切换前完成，失败会保留当前算法和场景。

## 动力学衔接

保留隔壁角色算法的 HOOD `20/12/9` 特征、节点类型 embedding、normalizer、带残差的消息传递和 decoder。没有采用最初规则布片的 10→16→3 toy GNN，也没有导入旧单体渲染器或旧 XPBD 投影器。

- 从现有 XPBD 修正后的位置和速度构建图。速度以 m/s 保存，输入网络时乘 `1/30`；固定点使用下一推理时刻的身体绑定目标。身体 world edge 的两个端点使用当前与 `t + 1/30` 的真实动画采样，禁止跨帧根坐标相减。
- GNN 每秒推理 30 次，按整数物理步计数调度；120 / 240 / 480 Hz 下分别保持 4 / 8 / 16 子步。使用物理预热后的状态，不套用旧 HOOD 首步 `1/3 s` 的特殊积分。
- 原 decoder 反归一化结果是每个 30 Hz 帧的加速度位移 `d`。转换为 `a_net = d / (1/30)^2`，再供高频 XPBD 积分。预测已经包含重力，不能再额外加一份重力。
- 网络加速度在下一次推理前保持。耦合为 `a = g + strength × confidence × (a_net − g)`。confidence 根据预测相对重力的偏差与该顶点最短静止边计算；超过信任范围逐渐退回重力，非有限输出直接退回重力。强度为 0 时得到纯 XPBD。
- GNN 不覆盖粒子位置，不经过 MLCloth 的 PCA 历史，也不使用 MLCloth 的粗层形状引导。拉伸、剪切、二面角弯曲、蒙皮绑定、阻尼、身体碰撞和自碰撞沿用当前 XPBD。
- 加速度缓冲与位置、速度一同存入 GPU 检查点。恢复到两个推理时刻之间时保留正确的加速度，重放重新按物理步生成推理事件。

网络材质特征保留旧 TinyHOOD 的训练参考常量，不等同于本 Demo 的 XPBD 顺应度。此版本不承诺两者经过材料标定；30 Hz 保持加速度也是明确的耦合近似。

## 图计算优化

`include/demo_gnn.*` 封装独立 Vulkan 推理模块，GPU 直接读求解器的位置、速度，结果写入求解器的常驻加速度缓冲。正常播放没有全网格回读。

- 静态 mesh edge 与顶点—三角形 CSR 仅构建一次，沿连通区域原拓扑传递消息，保持稳定累加顺序。
- 身体近邻用 CPU 构建、GPU 遍历的无栈点 BVH，替换布料节点对身体节点的逐点扫描。最近距离相同时按原始点编号决定结果。
- 反向 world edge 使用整数位图、popcount 和按节点编号排序的压缩列表，替换旧的每个布料节点扫描全图计算排名。浮点聚合次序与参考保持一致。
- 网络身体节点由当前 STM 顶点及胶囊表面采样构成，默认 2,119 个节点；它们只用于网络特征，接触仍由现有 STM 三角形和解析胶囊求解。
- 权重、特征、latent 和消息缓冲持久化；复用输入优先的权重布局、协作读取和 ping-pong 消息缓冲。没有复制权重文件。

UI 的 GNN strength 和 Prediction trust radius 可启动后调整，改变后重建物理历史。GNN 当前使用 Vulkan 后端；切换 CPU reference 会将第三角色恢复为 MLCloth 混合模式。性能面板单独显示一次 GNN 推理的 GPU 时间，该数字不是完整帧时间，也不是性能验收结论。

## 验证

```powershell
./tools/build-with-vs.cmd
./run.ps1 -ValidateGnn
./run.ps1 -ValidateGpu -ValidateAsset
../vulkan-gnn-poc/.venv/Scripts/python.exe tools/smoke_demo_gnn.py
```

`-ValidateGnn` 包含 GPU 融合测试和独立 PyTorch 校验。Python 只调用隔壁参考实现的代码与权重，不加载隔壁角色资源。原始 GPU 特征、图和 decoder 数据仅在显式验证模式回读。

本轮记录：

- CTest 13 / 13，通过新增的 120 / 240 / 480 Hz 推理调度与时间轴恢复检查。
- 现有 GPU 合成用例 88 / 88、真实资产 CPU/GPU 用例 4 / 4。
- GNN 集成用例 11 / 11：无 MLCloth 实例、零强度与纯 XPBD 一致、网络贡献非零、子步检查点恢复、重置重放、解除同步、独立换动画、重新同步、双向算法切换及模型加载失败恢复。
- 子步检查点重放与完整重置重放位置误差均为 0；零强度对纯 XPBD 的位置误差为 0。
- 当前 5,294 顶点、30,598 条有向 mesh edge、2,119 个身体节点的独立特征最大绝对误差 `1.91e-6`；原始 decoder 与 PyTorch 最大绝对误差 `8.15e-8 m`，均值 `3.21e-9 m`。BVH 最近点与暴力搜索一致；反向 CSR 顺序完全一致。
- 七动作短程脚本只检查有界播放及末帧有限性：待机一次、五种运动短循环各两次、复杂动作前五秒。不是 20 次循环 / 三次长动画验收，也不是连续穿透或高频质量评估。

汇总见 `results/GNN_PORT_VALIDATION.json`。具体画面及每动作末帧诊断保存在报告中的 smoke 目录。GNN 对部分动作降低 p95 拉伸，但跳跃的 p95 拉伸略差；已有的局部异常拉伸、接触和贴合问题仍需要继续优化。没有进行本轮全量性能验收，不能据此宣称稳定 60 FPS 或最终布料质量合格。

## 来源

移植的算法代码来自同仓库 `vulkan-gnn-poc/overlay/examples/gnncloth/{vgnn_format,real_scene_format,fine15_gpu_layout}.h` 与 `overlay/shaders/hlsl/gnncloth/tinyhood_*`。参照和权重来源说明保留于 `../vulkan-gnn-poc/THIRD_PARTY_NOTICES.md`；验证也复用该实现的 `real_scene/tinyhood.py`、`fine15.py`。本轮新写的特征适配、BVH、反向 CSR、加速度衔接和运行时接线均位于本实现中。
