## 结论

**可以，而且固定拓扑的布料 GNN 非常适合放进 UE5 的 Compute Shader。**

更准确地说，可以采用下面的路线：

> **PyTorch / PyG 负责训练 → Cook 阶段固化图结构和模型权重 → UE5 中使用 HLSL Compute Shader + RDG/RHI 完成推理。**

这样可以摆脱 CUDA、cuDNN、TensorRT 对 NVIDIA 的绑定。AMD、NVIDIA、Intel 可以共用一套基础 Shader 代码；但要注意，**代码可移植不等于性能完全一致**，FP16 吞吐、Wave 大小、缓存结构和异步计算能力仍会因 GPU 架构而不同。

严格来说，GNN 也不是传统卷积意义上的“二维模型”，而是在布料曲面对应的图结构上进行消息传递。

------

## 一、GNN 本质上完全可以映射到 Compute Shader

典型的 Message Passing GNN 可以简化成：

[
m_{ij}=\phi_e(h_i,h_j,e_{ij})
]

[
a_i=\sum_{j\in N(i)}m_{ij}
]

[
h'_i=\phi_v(h_i,a_i)
]

其中：

- (h_i)：顶点状态，例如位置、速度、法线、材质参数；
- (e_{ij})：边状态，例如静止长度、相对位置、拉伸率；
- (\phi_e)：边消息网络；
- (a_i)：邻居消息聚合；
- (\phi_v)：顶点更新网络。

这些操作最终无非就是：

- 从 Buffer 中 Gather 邻接顶点；
- 矩阵乘加；
- 激活函数；
- 求和或最大值聚合；
- 写回另一个 Buffer。

Compute Shader 完全能够完成这些操作，并不要求 CUDA。

对于你们的布料场景，甚至比通用 GNN 更有优势，因为绝大多数情况下：

- 布料拓扑在运行时不变；
- 邻接关系可以提前计算；
- 多级图结构可以提前构建；
- 顶点重排、图分块、权重布局都可以在 Cook 阶段完成；
- 不需要一个支持任意动态图的通用 GNN Runtime。

------

## 二、真正的难点是聚合方式，而不是矩阵乘法

最直观的 GNN GPU 实现是：

1. 一个线程处理一条边；
2. 计算边消息；
3. 用原子加法累加到目标顶点。

但这通常不是布料最好的实现，因为会遇到：

- 浮点原子操作竞争；
- 不同厂商原子性能差异；
- 聚合顺序不固定；
- 大量边消息中间 Buffer；
- 多层网络产生大量 Dispatch 和 Barrier。

### 更推荐：顶点中心的 Gather 模式

把邻接关系存成 CSR 或固定度数邻接表：

```text
VertexOffsets:  [0, 5, 11, 17, ...]
NeighborIndices:[3, 8, 12, 15, 21, ...]
```

然后：

- 一个 Workgroup 处理一个或几个顶点；
- Workgroup 内的线程对应隐藏特征通道；
- 遍历该顶点的邻居；
- 在寄存器或 Group Shared Memory 中完成消息累加；
- 最后统一写回。

对于规则三角布料，内部顶点邻居数量通常较小且相对稳定，因此这种 Gather 模式非常合适。它能够避免原子操作，也可以把“消息计算”和“聚合”融合到一个 Kernel 中。

HLSL Shader Model 6 提供了 Wave Reduction、Wave Broadcast、Wave Scan 等操作，可以跨线程完成聚合和数据交换；这些是 HLSL 标准能力，并不是 CUDA 专属。不过实现时不要假定固定 Wave32 或 Wave64，最好准备相应 Shader Permutation，或者编写与 Wave 大小无关的路径。([Microsoft Learn](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/hlsl-shader-model-6-0-features-for-direct3d-12?utm_source=chatgpt.com))

------

## 三、不要直接把论文里的通用 GNN 原样移植

这可能是整个方案最关键的一点。

论文中的通用 Mesh GNN 往往采用：

```text
Edge MLP
→ Edge Message Buffer
→ Aggregate
→ Node MLP
→ 下一层
```

如果每条边都执行一个完整的 MLP，计算量会迅速膨胀。

举一个示意性的规模：

- 4096 个模拟顶点；
- 平均每个顶点 6 个邻居；
- 约 24576 条有向边；
- Hidden Size 为 64；
- FP16 特征。

一个顶点特征 Buffer 大约是：

```text
4096 × 64 × 2 bytes ≈ 0.5 MiB
```

但一个边消息 Buffer 是：

```text
24576 × 64 × 2 bytes ≈ 3 MiB
```

每层写入一次、读取一次，就是约 6 MiB 的额外显存流量。八层消息传递仅边消息中间结果就会产生大约 48 MiB 的读写，还没计算顶点特征、权重、碰撞数据和高分辨率输出。

因此，实际产品模型应当从训练阶段就为 GPU 部署而设计。

### 比较适合实时布料的模型形式

相比每条边执行大型 MLP，更建议采用：

[
h'*i = \sigma(W_s h_i + W_n \sum_j \alpha*{ij}h_j + b)
]

其中 (\alpha_{ij}) 可以由少量边特征控制，例如：

- 当前边长和静止边长之比；
- 当前相对方向；
- 静止空间相对方向；
- 弯曲角；
- 材质参数；
- 边类型：结构边、弯曲边、层级边。

这样可以把主要矩阵乘法放在顶点级，而不是边级：

- 邻接部分主要是 (O(EH))；
- 顶点线性层主要是 (O(VH^2))；
- 避免昂贵的 (O(EH^2)) 边 MLP。

也可以采用小型门控消息：

```text
gate_ij = SmallEdgeNetwork(edge_feature)
message_ij = gate_ij * transformed_neighbor_feature
```

而不是为每条边执行完整的多层网络。

------

## 四、层级图比单纯增加 GNN 层数更重要

如果只沿三角网格的一环邻接传播，信息每经过一层只能传播一个拓扑距离。

长裙、披风这类布料存在明显的长程约束：

- 肩膀运动需要迅速影响裙摆或披风末端；
- 拉伸刚度需要跨越较远顶点传播；
- 只靠局部边可能需要十几层甚至更多消息传递。

层数越多：

- Dispatch 越多；
- 中间 Buffer 越多；
- 延迟越高；
- Autoregressive rollout 也更容易不稳定。

HOOD 这类研究的重要贡献恰恰不是简单使用 GNN，而是使用**多级图和层级消息传递**，让刚性拉伸等长程影响通过粗层级快速传播，同时保留细层级局部细节。([arXiv](https://arxiv.org/abs/2212.07242?utm_source=chatgpt.com))

你们可以在导入布料时预先建立：

```text
Level 0：原始低分辨率模拟网格
Level 1：简化后的粗网格或顶点聚类
Level 2：更粗的衣片级节点
```

运行时执行：

```text
细层 → Restrict 到粗层
粗层消息传递 1～2 次
粗层 → Prolongate 回细层
细层局部更新 1～3 次
```

这通常比在原始网格上连续跑十几层更符合实时渲染预算。

------

## 五、建议不要让 GNN 直接预测高分辨率渲染网格

更现实的结构是：

```text
骨骼姿态 / 速度
        ↓
低分辨率 Cloth Simulation Graph
        ↓
层级 GNN 预测加速度或残差位移
        ↓
显式积分
        ↓
碰撞与约束修正
        ↓
确定性插值或轻量 Decoder
        ↓
高分辨率渲染顶点
```

### 低分辨率图负责

- 大尺度摆动；
- 惯性；
- 拉伸传播；
- 弯曲趋势；
- 身体运动的动态响应。

### 高分辨率阶段负责

- 褶皱；
- 局部细节；
- 法线变化；
- 服装裁片高频形变。

高分辨率阶段可以使用：

- 预计算重心坐标插值；
- Skin Weight 式的低到高映射；
- Wrinkle Basis；
- 一个小型逐顶点 MLP；
- UE Deformer Graph 中的细节修正 Kernel。

这样不需要在几万甚至十几万个渲染顶点上跑完整 GNN。

------

## 六、建议采用“学习动力学 + 传统约束”的混合方案

不建议让网络独自承担所有物理正确性。

更稳健的路径是让网络预测：

- 加速度；
- 速度增量；
- 相对蒙皮基准的位移残差；
- 或 XPBD/PBD 的初始预测位置。

然后用一到数次 GPU 约束投影处理：

- 最大拉伸长度；
- 身体穿透；
- 固定点约束；
- 阻尼；
- 必要的自碰撞。

例如：

```text
GNN Predict
    ↓
Integrate
    ↓
Stretch Limit Projection
    ↓
Body SDF / Capsule Projection
    ↓
Optional Self-Collision
```

这有几个好处：

1. 网络不需要学习“穿模是不允许的”这种硬规则；
2. 极端姿势和训练集外动作更安全；
3. 少量预测误差不会持续累积成布料爆炸；
4. 训练 Loss 可以更专注于动态风格和材质表现。

尤其是自碰撞，固定拓扑 GNN 只能理解拓扑邻居，无法天然理解空间上突然接触的两个非邻接顶点。要让网络处理这类情况，需要运行时构造 proximity edges，而这又涉及空间哈希、BVH 或邻域搜索。实际项目里，把这部分作为独立 GPU 碰撞阶段通常更可控。

------

## 七、在 UE5 中的推荐实现方式

### 生产方案：自定义 Global Shader + RDG

建议建立类似下面的数据结构：

```text
FClothGNNAsset
├── Static Graph
│   ├── VertexOffsets
│   ├── NeighborIndices
│   ├── RestEdgeFeatures
│   ├── HierarchyMapping
│   └── LowToHighMapping
├── Packed Weights
├── Normalization Constants
└── Shader Configuration

FClothGNNInstance
├── CurrentPosition
├── PreviousPosition
├── Velocity
├── HiddenState
└── CollisionFeatures
```

每帧的 RDG Pass 可以是：

```text
1. BuildNodeFeaturesCS
2. EncodeCS
3. HierarchyRestrictCS
4. CoarseMessageUpdateCS × N
5. HierarchyProlongateCS
6. FineMessageUpdateCS × N
7. DecodeIntegrateCS
8. CollisionProjectionCS
9. UpsampleToRenderMeshCS
```

UE 的 RDG 原生支持 Compute Shader Pass、资源依赖和异步计算调度，可以通过 `FComputeShaderUtils::AddPass` 添加计算 Pass；RDG 会根据生产者和消费者关系安排 Barrier 和同步。([Epic Games Developers](https://dev.epicgames.com/documentation/unreal-engine/render-dependency-graph-in-unreal-engine?utm_source=chatgpt.com))

不过异步计算不是免费的。布料结果最终要被同一帧的 Deformer、Skin Cache 或渲染 Pass 消费，因此真正能够与图形管线重叠多少，取决于：

- 骨骼结果何时可用；
- 布料结果何时必须完成；
- 当前 GPU 是否已经 Compute 饱和；
- 是否存在可重叠的 Shadow、Nanite、后处理等任务。

所以应当先按普通 Compute Pass 实现，再用 RDG Insights、PIX 或 RenderDoc 判断是否值得放到 Async Compute。

------

## 八、NNE / DirectML 可以作为原型，但不建议直接作为最终答案

UE5 的 NNE 提供 CPU、GPU 和与 RDG 集成的接口，`IModelRDG` 可以创建在 RDG 中运行的模型实例，因此可以先把模型导成 ONNX，快速验证一版 GPU 推理。([Epic Games Developers](https://dev.epicgames.com/documentation/unreal-engine/API/Runtime/NNE/IModelRDG?utm_source=chatgpt.com))

Windows 上还可以考虑 DirectML。DirectML 面向所有支持 DirectX 12 的硬件，并为 AMD、Intel、NVIDIA 等设备提供统一的厂商无关接口，同时允许后端应用硬件相关优化。([Microsoft Learn](https://learn.microsoft.com/en-us/windows/ai/directml/dml?utm_source=chatgpt.com))

但它有几个问题：

- DirectML 只解决 Windows/DX12 范围内的硬件兼容；
- GNN 导出后通常包含大量 Gather、Scatter、Reduce 和小矩阵操作；
- 这些算子能否被有效融合必须针对具体模型测试；
- NNE 文档也明确说明，并非每个 Runtime 都支持所有文件格式或所有模型。([Epic Games Developers](https://dev.epicgames.com/documentation/unreal-engine/neural-network-engine-overview-in-unreal-engine?utm_source=chatgpt.com))
- 微软当前仍支持 DirectML，但新的功能开发方向已经转移到 Windows ML，因此不适合作为整个跨平台项目唯一的长期抽象层。([Microsoft Learn](https://learn.microsoft.com/ja-jp/windows/ai/directml/dml?utm_source=chatgpt.com))

因此更合理的定位是：

| 运行路径              | 作用                          |
| --------------------- | ----------------------------- |
| NNE + ONNX + DirectML | 快速验证模型和建立性能基线    |
| 自定义 HLSL + RDG     | 正式产品运行时                |
| CUDA / TensorRT       | NVIDIA 性能对照或内部离线工具 |

------

## 九、Tensor Core 并不是这个问题的核心

使用普通 HLSL Compute Shader 时，可以使用 FP16 权重和特征，但不能假定编译器一定会把普通矩阵乘加映射到 NVIDIA Tensor Core 或 AMD Matrix Core。

这意味着：

- 不会绑定 NVIDIA；
- 但也不能指望自动得到 TensorRT 那样的矩阵吞吐；
- 不同 GPU 上可能走普通 FP16 SIMD；
- 应准备 FP16 和 FP32 两套路径；
- 初期建议 FP16 权重与特征、FP32 累加，避免长时间积分误差。

不过对于布料 GNN，实际瓶颈很可能首先是：

1. 不规则邻接读取；
2. 边消息中间 Buffer；
3. 过多 Dispatch；
4. RDG Barrier；
5. 低 Batch 下的 GPU 利用率；
6. 动态碰撞邻居构建。

而不一定是纯矩阵 FLOPS。

因此在追求 Tensor Core 之前，更应该先解决：

- 图结构重排；
- 邻接局部性；
- Kernel Fusion；
- 层级传播；
- 减少边级 MLP；
- 跨角色 Batch；
- 避免 CPU/GPU Readback。

------

## 十、针对你们项目，我认为最合理的最终形态

```text
训练：
PyTorch / PyG
    ↓
固定拓扑与层级图训练
    ↓
导出自定义权重格式

Cook：
顶点重排
图分块
CSR / 固定度邻接生成
层级映射生成
权重转置与 FP16 Packing

运行时：
UE RDG Compute Shader
    ↓
低分辨率层级 GNN
    ↓
显式积分
    ↓
传统碰撞与约束修正
    ↓
高分辨率褶皱 Decoder
    ↓
直接写入 Deformer Buffer
```

其中几个原则尤其重要：

- **不要从 GPU 把顶点结果读回 CPU。**
- **不要在每帧动态构建固定布料拓扑。**
- **不要直接照搬包含大型 Edge MLP 的论文网络。**
- **不要让 GNN 独自负责碰撞和全部物理约束。**
- **不要直接在最终渲染网格上执行完整消息传递。**
- **尽量把多个角色或多件服装合并 Dispatch。**

你们当前 CPU 上约 1ms 的 MLP 推理，迁移到 GPU 后不一定会让单件小布料的“总帧时间”立刻下降，因为低 Batch 时 Dispatch 成本可能占比较高，而且项目也可能本来就是 GPU Bound。但它能够释放 CPU，并且在多角色、多服装场景下获得更好的扩展性。

**最终判断是：Compute Shader 路线不仅可行，而且很可能是跨 AMD/NVIDIA 的正确生产方向；但应该设计一个“为 Shader 部署而生的层级轻量 GNN”，而不是把通用论文模型直接翻译成几十个 Compute Pass。**

下一步最有价值的验证是用同一组权重制作两个版本：一个走 NNE/DirectML，一个走自定义融合 RDG Kernel，并分别测试 1、4、16 件布料在目标 AMD 与 NVIDIA 显卡上的 GPU 时间、Dispatch 数量、显存流量、连续 Rollout 稳定性和穿透率。