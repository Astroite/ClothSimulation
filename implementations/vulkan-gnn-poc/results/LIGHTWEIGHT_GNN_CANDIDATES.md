# CH10032 轻量 GNN 权重与架构筛选

## 结论

截至本次检索，没有找到一个公开 checkpoint 同时满足以下四点：

1. 明显轻于 Fine15；
2. 以动画身体和 cloth-to-body contact 为条件；
3. 能泛化到未见过的服装拓扑；
4. 可以不重新训练就用于 CH10032。

HOOD 官方公开包中的 `fine15.pth` 已经是最浅的单层图基线；另外只有更重的 Fine48 和层级模型。最接近可替换权重的是 EUNet 发布的 garment dynamics checkpoint，但其动态骨干依然是 128 latent、15 层 MeshGraphNet，因此可能改善训练目标和泛化，**不会解决当前主要耗时**。

## 候选筛选

| 候选 | 公开权重 | 服装/身体接触 | 跨拓扑 | 复杂度与 CH10032 适用性 |
| --- | --- | --- | --- | --- |
| [HOOD Fine15](https://github.com/dolorousrtur/hood) | 有 | 有 | 有 | 当前基线；128 latent × 15 block，`3,854,164` 个 FP32 数，`15.50 MB` VHOOD。可直接用于 CH10032，但不轻。 |
| [EUNet garment dynamics](https://github.com/ftbabi/EUNet_NeurIPS2024) | 有 | 有 | 有，官方给出 Cloth3D 测试 | 最接近可移植候选，但其 [MGN 配置](https://github.com/ftbabi/EUNet_NeurIPS2024/blob/main/configs/_base_/models/selfsup_hoodmgn_learned_pnet.py) 仍为 128 latent × 15 encoder layers。需要转换 checkpoint/normalizer，预期成本与 Fine15 同级。 |
| [DeepMind MeshGraphNets FlagSimple](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets) | 官方未提供即用 cloth checkpoint | 无动画人体条件 | 规则旗帜可变网格 | 官方发布数据、loader 和训练流程，README 要求用户自行训练；其 flag 输入语义不能直接驱动 CH10032。适合作为重新训练实现参考。 |
| [AdaptiGraph cloth](https://github.com/Boey-li/AdaptiGraph) | 有 | 机器人末端执行器，不是人体 | 粒子布 | checkpoint 面向桌面推动/抓取和材料辨识，动作、节点类型、碰撞语义均不兼容 CH10032；不能作为即插即用权重。 |
| [NeuralClothSim](https://github.com/hbertiche/NeuralClothSim) | 未见通用跨服装即用权重 | 有 | 服装通常绑定特定身体 | 动态角色方向相关，但需要为 body/garment 准备配置和训练，不能直接替换 Fine15。 |
| [PBNS](https://github.com/hbertiche/PBNS) | 方法/代码 | 有 | 否，服装与角色专用 | 不是 GNN，而是极轻的 pose-space deformation。更适合首帧初始化或 LBS coarse pose；值得作为混合管线的旁路方案。 |

没有把 Fine15 截断成前 3–5 个 block 做“轻量权重测试”，因为 decoder 是针对第 15 层 latent 训练的；直接截断得到的失败不能代表浅层架构能力，数值上也不是一个有效 checkpoint。

## 建议的下一版：TinyHOOD-64×4

比继续寻找域不匹配的公开权重更可控的方案，是使用现有 CH10032/Chaos 数据和 Fine15 teacher 训练一个保持输入契约的小模型：

- 保留 HOOD 的 20 维 node、12 维 mesh edge、9 维 world edge、NodeType 和 normalizer，使现有 `VCLTH/VCHAR/VANIM` 烘焙与 Vulkan 前处理可直接复用。
- latent 从 128 降到 64；消息传递从 15 次降到 4 次，优先测试 4 次共享 processor 权重，再与 4 次非共享版本比较。
- 输出仍为 3 维加速度；训练目标同时包含 Fine15 teacher 蒸馏、Chaos/XPBD 下一帧位置、边长/面积、碰撞和多步 rollout 稳定性。
- 训练时混合规则网格、CH10032 裙装和不同细分密度；不要只在 CH10032 单网格上拟合，否则会丢掉这次 Grid64 证明出来的拓扑泛化能力。
- 运行时先以 15–30 Hz 产生大形，XPBD/Chaos 以 60 Hz 补充长度、弯曲、碰撞和自碰撞；首帧 warm-start 可以复用同一个网络的单步模式。

仅按 dense MLP 的宽度平方和 block 数估算，`128×15 → 64×4` 的 processor 算术量约降到原来的 `1/15`；encoder/decoder、世界边搜索和 dispatch 开销不会同比缩小，所以这是设计目标而非实测加速承诺。当前 Grid64 数据显示每个 Fine15 block 的 edge+node update 约 `35.4 ms`，先减少 block 数的收益路径非常直接。

## 推荐执行顺序

1. 先写纯 PyTorch TinyHOOD 配置和 Fine15 teacher cache，在 CH10032 T-Pose、sprint、规则 Grid64 上做 1/10/120 步验证。
2. 比较 `32×4`、`64×4`、`64×6` 三档；以“不拉条、固定点稳定、碰撞可接受”为第一筛选，再看误差。
3. 只移植通过多步 rollout 的最小模型到 Vulkan，并把 edge/node MLP 融合为较少 dispatch。
4. 若目标主要是解决 Chaos 初始帧，则并行验证 PBNS/小型 PSD 网络；它比连续 GNN rollout 更符合一次性 warm-start 的成本结构。

## 实测更新

上述 `TinyHOOD-64×4` 非共享 processor 版本已经完成 PyTorch 训练、VHOOD 导出和 Vulkan 移植。它保留了完整的 `20/12/9` 输入与身体 world edge，参数量为 `286,275`，但本轮仅使用短序列 Fine15 teacher 单步蒸馏，闭环 rollout 在 CH10032 和 Grid64 上都出现严重拉伸。Vulkan 计算速度显著改善，数值实现也通过黄金验证；当前失败属于模型训练与时序稳定性，而不是部署链错误。完整数据见 [TINYHOOD_64X4_RESULTS.md](TINYHOOD_64X4_RESULTS.md)。
