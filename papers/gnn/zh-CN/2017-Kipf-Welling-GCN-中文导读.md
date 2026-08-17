# 半监督分类的图卷积网络（GCN）

## 书目信息

Thomas N. Kipf、Max Welling，ICLR 2017；arXiv:1609.02907。原文：[`originals/2017-Kipf-Welling-Semi-Supervised-Classification-with-GCN.pdf`](../originals/2017-Kipf-Welling-Semi-Supervised-Classification-with-GCN.pdf)。公开来源：[arXiv](https://arxiv.org/abs/1609.02907)、[OpenReview](https://openreview.net/forum?id=SJU4ayYgl)。

## 摘要意译

论文提出一种可扩展的图卷积层，用于只有少量节点有标签的半监督学习。模型由谱图卷积的一阶局部近似得到，复杂度与边数线性相关；隐藏表示同时编码节点特征和局部结构，并在引用网络与知识图谱上取得强基线结果。

## 逐节导读

### 1 问题设定

图 $G=(V,E)$ 有邻接矩阵 $A$、节点特征 $X$ 和少量标签。传统拉普拉斯正则化假定相邻节点标签相似，但边也可能表达关系而非相似性。GCN 把 $A$ 直接放进网络，让监督信号通过共享的图传播层影响节点表示。

### 2 图卷积层

核心传播规则为：

$$H^{(l+1)}=\sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l)}W^{(l)})$$

其中 $\tilde{A}=A+I$ 添加自环，$\tilde{D}_{ii}=\sum_j\tilde{A}_{ij}$，$H^{(0)}=X$。对称归一化防止高度节点数值爆炸，并保留节点自己的特征；$\sigma$ 通常取 ReLU。两层模型常写成 $Z=\mathrm{softmax}(\hat{A}\,\mathrm{ReLU}(\hat{A}XW_0)W_1)$。

### 3 谱域动机

在归一化拉普拉斯的特征基中定义卷积，再用 Chebyshev 多项式近似谱滤波器。取一阶近似并约束参数后得到简单的邻居平均规则。该推导解释了局部性和归一化来源，而实际实现只需稀疏矩阵乘法。

### 4 训练与实验

只在标注节点上使用交叉熵，未标注节点通过共享传播参与表示学习。作者在 Cora、Citeseer、Pubmed 引用网络和 NELL 知识图谱上与多种半监督基线比较，GCN 兼具准确率与速度。论文没有提出“过平滑”概念；这是后来用于解释深层消息传递中节点表示趋同的常见视角。

## 关键解释与局限

GCN 本质是“自环 + 归一化邻居聚合 + 线性变换 + 非线性”。原论文明确的限制包括全批训练的内存开销、对无向图的依赖、缺少原生边特征支持，以及固定的局部性和邻居权重。现代实践还需面对过平滑与过压缩，大图通常采用采样或分块。用于布料时，网格边可提供局部材料关系，但碰撞、长程约束需要额外边或专门消息通道。

## 术语表

GCN 图卷积网络；spectral convolution 谱卷积；Laplacian 拉普拉斯；self-loop 自环；symmetric normalization 对称归一化；semi-supervised 半监督；over-smoothing 过平滑。

本文为逐节中文学习导读和公式解释，不是整篇逐字翻译。
