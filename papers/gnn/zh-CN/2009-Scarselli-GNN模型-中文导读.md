# 图神经网络模型（The Graph Neural Network Model）

## 书目信息

Franco Scarselli、Marco Gori、Ah Chung Tsoi、Markus Hagenbuchner、Gabriele Monfardini，IEEE Transactions on Neural Networks 20(1), 2009, pp.61–80。DOI: [10.1109/TNN.2008.2005605](https://doi.org/10.1109/TNN.2008.2005605)。原文：[`originals/2009-Scarselli-The-Graph-Neural-Network-Model.pdf`](../originals/2009-Scarselli-The-Graph-Neural-Network-Model.pdf)。

## 摘要意译

许多科学工程数据天然具有图结构。本文提出早期统一的 GNN 模型，能够直接处理有向/无向、含环/无环等实用图，并把图及其节点映射到欧氏向量。作者给出监督学习算法、计算复杂度分析和实验，验证模型的泛化能力。核心思想是让节点状态由自身特征与邻居状态共同决定，并通过递归固定点获得全图上下文。

## 逐节导读

### 1 引言与任务

图可用于分子（原子和化学键）、图像区域邻接图、网页链接和自然语言等。论文区分 graph-focused（输出整图类别）与 node-focused（输出指定节点类别）任务，强调把图“压扁”为向量会损失拓扑依赖。

### 2 GNN 模型

对图 $G=(V,E)$，节点标签 $l_v$、边标签 $l_{(v,w)}$ 和状态 $x_v$ 通过局部转移函数更新：

$$x_v = f_w(l_v,l_{co[v]},x_{ne[v]},l_{ne[v]})$$

输出由 $o_v=g_w(x_v,l_v)$ 计算。这里 $ne[v]$ 是邻居集合，$co[v]$ 是与节点相连的边；参数在所有节点共享，因此重新编号节点只会相应地重新编号节点输出。状态同时编码局部观测和经扩散传来的远处信息。

### 3 收敛性与计算

循环图可能导致递归方程没有唯一解。作者要求转移函数为压缩映射：存在 $0\leq\rho<1$，使状态距离经一次更新至多缩小 $\rho$ 倍。由 Banach 不动点定理，迭代从任意初态出发都收敛到唯一状态。实际训练使用有限次迭代近似固定点，并通过反向传播穿过迭代过程；复杂度随边数、迭代次数和隐藏维数增长。

### 4 学习算法

损失在有监督节点的输出上计算。前向阶段先迭代状态更新直到固定点；反向阶段采用 Almeida-Pineda 方法，在固定点处迭代传播导数，避免保存完整的时间展开，再用梯度下降（实验实现使用 resilient backpropagation）更新参数。训练数据可含不同大小和不同拓扑的图，参数共享使模型具有跨图泛化能力。

### 5 实验与结论

论文实验包含子图匹配、致突变性预测和网页排序；它们分别检验结构匹配、分子图分类以及单个大图上的节点评分。结果表明 GNN 能利用结构并泛化到未见图。作者同时指出，固定点迭代会带来计算开销，收敛速度受压缩映射常数影响。

## 关键概念

- **状态传播**：邻居信息经共享函数聚合并写回节点。
- **固定点**：反复传播直到状态不再变化，代表全图上下文。
- **置换对称性**：重新编号节点时，节点级输出应等变；图级预测应保持不变。
- **图/节点聚焦**：分别对应整图和局部节点输出。

## 局限与今日视角

压缩映射保证稳定但限制表达力；固定点迭代和反向传播开销大；聚合函数对多重邻居结构的区分能力有限。现代 MPNN、GCN 和 Graph Transformer 延续了“共享局部消息传递”思想，但通常采用有限层数、残差/归一化和更丰富的边特征。

## 术语表

Graph neural network 图神经网络；state 状态；transition function 转移函数；output function 输出函数；fixed point 不动点；contractive mapping 压缩映射；node-focused 节点聚焦；graph-focused 图聚焦。

## 来源

原文路径见上；公开来源：[IEEE Xplore](https://ieeexplore.ieee.org/document/4700287)、[DOI](https://doi.org/10.1109/TNN.2008.2005605)。本文是完整阅读后的中文导读与意译，不是逐字翻译。
