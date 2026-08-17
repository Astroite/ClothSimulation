# Distill：图神经网络入门与图上的卷积（合并中文解读）

## 资料信息

合并阅读：[`interpretations/2021-Distill-A-Gentle-Introduction-to-GNN.html`](../interpretations/2021-Distill-A-Gentle-Introduction-to-GNN.html) 与 [`interpretations/2021-Distill-Understanding-Convolutions-on-Graphs.html`](../interpretations/2021-Distill-Understanding-Convolutions-on-Graphs.html)。公开来源：[A Gentle Introduction to GNNs](https://distill.pub/2021/gnn-intro/)、[Understanding Convolutions on Graphs](https://distill.pub/2021/understanding-gnns/)。

## 学习主线

Distill 先用“节点、边、全局”三类实体解释图网络，再把常见网络统一为消息传递：每条边计算消息，节点聚合邻居消息，更新节点状态，最后按任务读出节点、边或整图。这个视角连接了 GNN、GCN、GraphSAGE、GAT 和物理模拟器。

## 第一篇：从图表示到消息传递

图数据的难点是大小可变、邻居数可变且节点没有天然顺序。可学习函数必须对节点置换保持等变/不变。通用一步更新可写成 
$$
(m_{ij}=\phi_e(h_i,h_j,e_{ij}))
$$

$$
(h_i'=\phi_v(h_i,\square_{j\in N(i)}m_{ij}))
$$

其中 (square) 是 sum/mean/max 等置换不变聚合。多轮传播扩大感受野，但也带来过平滑和过压缩。

图级任务先得到节点嵌入，再用 sum/mean/max 或注意力池化读出整图；节点级任务直接分类；边级任务使用两端节点与边嵌入。边特征在分子键、网格连接和碰撞关系中同样关键。

## 第二篇：理解图上的卷积

规则网格卷积依赖平移对称和固定邻域；一般图没有统一坐标，因此“卷积”通常意味着共享权重的局部聚合。谱方法从拉普拉斯特征向量定义频率，空间方法直接沿邻接边传播。Chebyshev 近似、GCN 的归一化邻居平均、注意力加权都是效率与表达力的折中。

## 与三篇原论文的对应

Scarselli 以固定点递归建立全图状态；Kipf/Welling 将谱域一阶近似化为高效的归一化传播；MeshGraphNets 则把消息传递用于时间推进，并区分网格空间与世界空间边。由此可把布料网格看作带几何特征的图：局部边传弹性，空间邻近边传碰撞，解码器输出加速度。

## 常见误区与实践建议

“层数越深越好”不成立；优先增加有效边、残差和多尺度通道。平均聚合不一定能区分复杂邻域；需要边类型、注意力或更强的集合编码。图卷积不是自动遵守物理定律，模拟任务需明确积分器、边界条件、碰撞和守恒检查。

## 术语表

message passing 消息传递；permutation equivariance 置换等变；readout 读出；pooling 池化；spatial convolution 空间卷积；spectral convolution 谱卷积；receptive field 感受野；oversmoothing 过平滑；oversquashing 过压缩。

## 版权与范围说明

本文为基于两篇公开 Distill 文章的中文学习解读和结构化意译，未复制其整篇正文或大段原文；图示与互动内容请访问原网页。
