# 布料模拟学习资料

本目录用于集中整理布料模拟（Cloth Simulation）相关的论文、学习笔记、教程、实验代码与测试素材。

## 目录结构

- `papers/`：论文原文、论文链接与阅读摘要
  - `gnn/`：GNN 相关（Scarselli / Kipf-Welling / Pfaff，含中文导读）
  - `xpbd/`：XPBD/PBD 出处与中文导读——**当前实现的物理实际依据**
- `plans/`：技术方案与可行性分析
- `implementations/`：算法实现、示例工程和实验代码
  - `vulkan-gnn-poc/`：目前唯一的实质实现。进展总览见
    [`results/PROGRESS.md`](implementations/vulkan-gnn-poc/results/PROGRESS.md)
- `notes/`、`tutorials/`、`assets/`：**尚未填充**，仅有占位 README

当前内容重心明显偏向 `implementations/vulkan-gnn-poc/` 与其相关论文；上面标注
「尚未填充」的三个目录只是预留结构，不要误以为里面有资料。

## 建议的整理方式

- 论文文件可使用 `年份-作者-简称.pdf` 命名，例如 `1998-Baraff-LargeSteps.pdf`。
- 在 `papers/README.md` 中记录论文链接、主题、阅读状态和摘要。
- 实验代码按算法或框架建立独立子目录，并在各自目录中补充运行说明。

