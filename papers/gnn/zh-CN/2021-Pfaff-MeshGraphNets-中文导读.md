# 使用图网络学习基于网格的模拟（MeshGraphNets）

## 书目信息

Tobias Pfaff、Meire Fortunato、Alvaro Sanchez-Gonzalez、Peter W. Battaglia，ICLR 2021；arXiv:2010.03409。原文：[`originals/2021-Pfaff-Learning-Mesh-Based-Simulation-with-Graph-Networks.pdf`](../originals/2021-Pfaff-Learning-Mesh-Based-Simulation-with-Graph-Networks.pdf)。公开来源：[arXiv](https://arxiv.org/abs/2010.03409)、[项目页](https://sites.google.com/view/meshgraphnets)。

## 摘要意译

MeshGraphNets 用图网络学习网格物理模拟：在网格空间传递内部相互作用，在世界空间传递接触/碰撞等外部作用，并可学习分辨率场、在 rollout 中自适应重网格。实验覆盖空气动力学、结构力学和布料，速度比训练所用数值模拟快 1–2 个数量级。

## 逐节导读

### 1 表示与架构

Encode-Process-Decode：编码器把节点、网格空间边和世界空间边转成 128 维嵌入；处理器顺序执行 $L=15$ 个消息传递块，每个块先更新两类边，再分别求和聚合到节点；解码器对布料节点预测加速度。积分器据此更新速度和位置，产生下一时刻网格，并在推理时反复展开为 rollout。网格空间消息用于近似 PDE 中的局部内部作用，世界空间消息用于表达网格拓扑之外的接触与自碰撞。

### 2 图构造

网格边来自三角形或四面体连接，编码参考网格坐标中的相对位移 $u_{ij}$、其范数，以及当前世界坐标中的相对位移 $x_{ij}$、其范数。对拉格朗日系统，还会在世界空间距离小于半径 $r_W$ 且尚无网格边的节点间建立世界空间边。相对坐标使表示不依赖绝对平移，但普通 MLP 并不自动保证旋转等变。自适应版本另行预测 sizing field，再由局部 remesher 执行边拆分、折叠和翻转。

### 3 布料数据与训练

布料实验包含三个由 ArcSim 生成的域：FlagSimple 使用静态规则网格且忽略碰撞；FlagDynamic 模拟风中旗帜并动态改变网格；SphereDynamic 模拟布料与运动球体接触，同样使用动态网格。节点输入包含节点类型和由相邻两帧位置得到的速度；网格边携带参考形状与当前形状的相对几何，世界空间边负责空间邻近与碰撞关系，SphereDynamic 的球体节点也进入图中。

训练对单步节点加速度使用 $L_2$ 监督，并向最新位置加入高斯噪声以提高数百步 rollout 的稳定性。动态网格数据在参考网格空间用重心插值对齐历史量和监督目标。推理时反复调用单步模型；对于动态域，模型同时预测 sizing field，并在每一步用通用局部 remesher 改变网格分辨率。
### 4 实验与结果

作者比较粒子、规则网格和不同消息通道，展示模型在不同初始条件、外力及更高分辨率上的泛化。布料结果说明网格拓扑有助于表达静止形状和材料弹性；世界空间消息对碰撞尤其重要。自适应网格把计算预算集中到褶皱、接触等高梯度区域。

## 关键公式/概念

一次处理块可概括为：

$$e_{ij}^{\prime M}=f^M(e_{ij}^M,v_i,v_j),\qquad e_{ij}^{\prime W}=f^W(e_{ij}^W,v_i,v_j)$$

$$v_i'=f^V\left(v_i,\sum_j e_{ij}^{\prime M},\sum_j e_{ij}^{\prime W}\right)$$

解码器再由最终节点嵌入预测加速度。这不是把完整物理方程硬编码，而是以消息传递学习离散动力学；积分器负责把加速度转成下一状态。

## 布料模拟价值与局限

价值在于：非规则网格可变分辨率；拓扑边表达弹性与静止长度；世界边表达碰撞；一次训练后 rollout 大幅提速。局限包括长时稳定性、强碰撞和自碰撞、拓扑变化、材料参数外推及训练数据成本。模型是数据驱动近似，不能自动保证动量/能量守恒；生产使用应配合约束投影、碰撞修正和守恒监控。

## 术语表

mesh 网格；rollout 迭代展开；message passing 消息传递；mesh-space 网格空间；world-space 世界空间；sizing field 尺寸场；remeshing 重网格；one-step supervision 单步监督；cloth 布料；collision 碰撞。

本文是完整阅读后的中文导读与意译，不是逐字翻译。
