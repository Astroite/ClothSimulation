# XPBD / PBD 论文与导读

本目录收录**决定当前 PoC 画面质量的物理方法**的参考资料。

`implementations/vulkan-gnn-poc` 的布料行为主要由 XPBD 约束投影产生，而不是由
神经网络产生（消融实测：整个图消息传递只贡献 3.5 mm 平均位移，见
[`../../implementations/vulkan-gnn-poc/results/RESULTS.md`](../../implementations/vulkan-gnn-poc/results/RESULTS.md)）。
但仓库此前只收录了 GNN 相关论文，XPBD 的 `α̃ = compliance / Δt²` 从何而来
无处可查。本目录补上这一环。

## 论文

出于版权考虑，本目录不放置 PDF 原文，只记录出处：

| 年份 | 作者 | 标题 | 出处 |
| --- | --- | --- | --- |
| 2016 | Macklin, Müller, Chentanez | *XPBD: Position-Based Simulation of Compliant Constrained Dynamics* | ACM MIG 2016 |
| 2007 | Müller, Heidelberger, Hennix, Ratcliff | *Position Based Dynamics* | VRIPHYS 2006 / JVCA 2007 |

建议补充阅读：Macklin 等 2019 *Small Steps in Physics Simulation*（讨论用更多
子步替代更多迭代，与本 PoC「8 次迭代不足以在 32 宽布片上传播张力」的实测
直接相关）。

## 中文导读

- [`zh-CN/XPBD-中文导读.md`](zh-CN/XPBD-中文导读.md)：PBD 的问题、XPBD 的
  推导、`α̃ = compliance / Δt²` 的来历，以及这些结论在本仓库代码中的具体落点。
