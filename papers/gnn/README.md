# GNN 论文与中文解读

本目录整理图神经网络（Graph Neural Network, GNN）的奠基论文、核心方法和布料模拟相关应用。

## 目录

- `originals/`：英文论文原文
- `interpretations/`：公开解读资料及来源说明
- `zh-CN/`：中文翻译笔记与学习解读

## 中文总览

- [GNN：从图上的局部传播到布料物理推理](zh-CN/GNN-思路与推理过程-详细学习资料.md)：从固定点 GNN、GCN 和现代消息传递统一框架，推导到 MeshGraphNets 的布料构图、单步预测与 rollout，并包含手算示例、表达力/深度局限及工程检查表。

## 材料清单与可靠来源

1. Scarselli 等（2009）奠基模型：[DOI](https://doi.org/10.1109/TNN.2008.2005605)。
2. Kipf & Welling（2017）GCN：[arXiv](https://arxiv.org/abs/1609.02907)、[OpenReview](https://openreview.net/forum?id=SJU4ayYgl)。
3. Pfaff 等（2021）MeshGraphNets：[arXiv](https://arxiv.org/abs/2010.03409)、[项目页](https://sites.google.com/view/meshgraphnets)。
4. Distill 入门：[GNN Introduction](https://distill.pub/2021/gnn-intro/)、[Graph Convolutions](https://distill.pub/2021/understanding-gnns/)。

## 建议阅读顺序

先读 Distill 合并解读建立消息传递直觉，再读 Scarselli 理解固定点与置换不变性，接着读 GCN 掌握归一化传播，最后读 MeshGraphNets 连接到布料模拟。对应中文材料位于 `zh-CN/`。
