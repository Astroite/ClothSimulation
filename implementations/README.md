# 实验与实现

存放布料模拟算法的示例工程、实验代码及运行说明。

- [`vulkan-gnn-poc/`](vulkan-gnn-poc/)：PyTorch → HLSL Compute → SPIR-V → Vulkan 的最小 GNN 布料部署验证。
  横向进展入口：[`results/PROGRESS.md`](vulkan-gnn-poc/results/PROGRESS.md)（性能演进、已确立的负面结果、
  未解决的缺口、文档地图）。
- [`vulkan-mlcloth-cpu-poc/`](vulkan-mlcloth-cpu-poc/)：保留 AILab/MNN CPU 推理，逐帧上传 5,294 个 CH10032 布料顶点，由 Vulkan Compute 完成坐标变换并绘制点云的独立 Win64 PoC；首轮不含 XPBD、碰撞、三角拓扑或 MNN GPU backend。
