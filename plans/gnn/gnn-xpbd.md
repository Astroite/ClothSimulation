## 判断

**可行，而且这很可能比继续强化“纯 GNN 自回归模拟”更适合作为生产架构。**

它本质上是一个 **Learned Predictor / Preconditioner + Physics Corrector**：

- 神经网络负责快速给出接近正确解的全局预测，尤其处理低频、长距离传播；
- XPBD/PBD 负责把结果拉回约束流形，处理拉伸、固定点、碰撞、摩擦等硬条件；
- 每帧将 **XPBD 修正后的状态** 反馈给下一帧，而不是反馈未经修正的神经网络结果。

这种结构正好针对你现在遇到的“短时间效果不错、长时间 rollout 漂移或失稳”问题。已有研究也表明，神经模型与数值求解器结合、并在训练时展开实际求解过程，通常比单步神经预测更适合长期模拟。([arXiv](https://arxiv.org/html/2402.12971v1))

不过这里有一个非常重要、很容易被忽略的数值细节：

> **对标准 XPBD 来说，任意的神经网络顶点位置并不严格等价于普通迭代求解器里的“初值”。**

------

## 一、不要简单地把 GNN 输出直接塞进 `x₀`

XPBD 的经典流程是：

[
\tilde{x}=x^n+\Delta t v^n+\Delta t^2M^{-1}f_{\mathrm{ext}}
]

然后初始化：

[
x_0=\tilde{x},\qquad \lambda_0=0
]

XPBD 实际上在求解：

[
g(x,\lambda)=M(x-\tilde{x})-J^T\lambda=0
]

[
h(x,\lambda)=C(x)+\tilde{\alpha}\lambda=0
]

其中：

- (\tilde{x}) 是惯性预测位置；
- (J=\nabla C) 是约束雅可比；
- (\lambda) 是约束乘子；
- (\tilde{\alpha}=\alpha/\Delta t^2)。

标准 XPBD 的局部更新式之所以简单，是因为它假设最初的 primal residual：

[
g(x_0,\lambda_0)=0
]

当 (x_0=\tilde{x},\lambda_0=0) 时，这个条件天然成立。XPBD 原论文明确以这种方式初始化，并在推导局部更新时使用了这一假设。([Mmacklin](https://mmacklin.com/xpbd.pdf))

如果你改成：

[
x_0=x_{\mathrm{GNN}},\qquad \lambda_0=0
]

那么通常：

[
M(x_{\mathrm{GNN}}-\tilde{x})\neq0
]

也就是 (g\neq0)。

这意味着网络输出不再只是“加快收敛的初值”，而会事实上成为一部分额外的动力学输入。标准 XPBD 后续只修正约束残差，并不会严格消除这部分初始偏置。可能产生的现象包括：

- 网络位移被转换成额外速度；
- 能量无意注入或耗散；
- 低迭代次数时结果对网络预测高度敏感；
- 网络偶尔预测错误后，XPBD 没有足够迭代把它完全拉回；
- 长期仍可能漂移，只是比纯 GNN 慢。

这不代表这种方式不能用，而是应该明确：

> **“GNN 位置预测 + XPBD 修正”更像学习动力学模型，而不是严格意义上的无偏 warm start。**

------

## 二、三种混合方式中，优先推荐预测约束空间修正

### 方案 A：GNN 预测位置，XPBD 做 Corrector

这是最快能验证的版本：

```text
骨骼、历史状态
    ↓
GNN 预测 x_gnn
    ↓
碰撞候选生成
    ↓
2～4 次 XPBD
    ↓
最终 x、v
```

优点是你现有的小 GNN 基本可以直接复用。

但这里建议把网络理解为在生成新的惯性预测：

[
\tilde{x}*{\mathrm{hybrid}}=x*{\mathrm{GNN}}
]

而不是把它宣传为“不改变最终物理解的初值”。网络负责学习部分真实动力学，XPBD 只负责物理安全和约束修正。

这个版本适合先验证：

- 长期稳定性是否显著提高；
- 穿模率是否下降；
- 2～4 次 XPBD 是否能替代原本十几次迭代；
- 网络的 1 ms 成本是否能被节省的求解时间覆盖。

### 方案 B：GNN 预测 (\lambda) 或 (\Delta\lambda)

这是我更推荐的生产路线。

让 GNN 不直接输出每个顶点的最终位置，而是输出：

- Stretch Constraint 的乘子修正；
- Shear Constraint 的乘子修正；
- Bend Constraint 的乘子修正；
- 或者更粗层级上的约束空间系数。

网络预测：

[
\lambda_0=\lambda_{\mathrm{GNN}}
]

然后构造与它近似一致的位置：

[
x_0=\tilde{x}+M^{-1}J^T\lambda_0
]

这样能够近似保持：

[
M(x_0-\tilde{x})-J^T\lambda_0\approx0
]

随后再运行标准 XPBD。

这条路线有几个明显优势：

1. **更接近真正的 Warm Start**

   网络是在估计约束求解器的解，而不是重新定义动力学目标。

2. **内部力结构更自然**

   对于边约束、弯曲约束，(J^T\lambda) 自然生成成对或成组的约束修正，比网络直接输出任意顶点位移更容易保持内部作用关系。

3. **更容易控制**

   可以按约束类型分别 Clamp：

   [
   |\Delta\lambda_{\text{stretch}}|<\tau_s
   ]

   [
   |\Delta\lambda_{\text{bend}}|<\tau_b
   ]

4. **适合你现在的 GNN**

   现有 GNN 已经在 Cloth Graph 上传播消息，只需要增加 Edge Head 或 Constraint Head，不一定需要重新设计完整主干。

需要注意，冗余约束下 (\lambda) 可能不是唯一的，因此训练时不要只做纯粹的 multiplier MSE。更合理的是让网络输出 (\Delta\lambda)，然后以它应用后的顶点结果、约束残差和少量 XPBD 后的结果作为主要监督。

### 方案 C：支持任意位置初值的完整 Primal-Dual Corrector

如果你坚持直接使用 (x_{\mathrm{GNN}}) 作为初值，也可以修改 XPBD 更新，让它同时处理 primal residual (g)。

由 XPBD 的线性化块系统，可以得到：

[
\Delta\lambda=
\frac{-h+JM^{-1}g}
{JM^{-1}J^T+\tilde{\alpha}}
]

[
\Delta x=M^{-1}(J^T\Delta\lambda-g)
]

这样，即使：

[
x_0\neq\tilde{x}
]

求解器也会主动消除网络初值引入的 primal residual。

不过它的实现成本明显更高：

- 需要维护或重新构建 (g)；
- GPU 并行局部更新更复杂；
- 多约束共享顶点时需要认真处理同步；
- 已经逐渐接近完整 primal-dual solver，而不是经典的轻量 XPBD。

所以它更适合作为研究对照，不一定是第一个生产版本。

------

## 三、比“帧开始 Warm Start”更好的结构：学习型 V-Cycle

我认为最值得尝试的结构不是：

```text
GNN
→ XPBD
```

而是：

```text
XPBD Pre-Smooth
→ GNN Global Correction
→ XPBD Post-Smooth
```

具体可以是：

```text
1. 惯性预测 x_tilde
2. 生成 body collision / self-collision 候选
3. 运行 1～2 次 XPBD Pre-Smooth
4. 根据当前约束残差构建 GNN 输入
5. GNN 预测低频 Δλ 或 coarse correction
6. 应用 GNN 修正
7. 运行 1～3 次 XPBD Post-Smooth
8. 最终碰撞修正和速度更新
9. 残差过大时追加 XPBD 迭代
```

原因是局部 PBD/XPBD 迭代通常能够很快消除高频、局部约束误差，但在高分辨率、高刚度条件下，对低频、长距离误差的收敛较慢；这也是多重网格 XPBD 研究重点解决的问题。([arXiv](https://arxiv.org/html/2505.13390v1))

而布料研究中也观察到：

- 拉伸传播等低频误差适合由子空间或全局方法处理；
- 碰撞和自碰撞造成的变形往往更偏高频；
- 低频子空间修正与 GPU 局部迭代可以形成较好的互补。([arXiv](https://arxiv.org/html/2403.19272v4))

所以你的小 GNN 最适合承担的角色可能不是“完整模拟器”，而是：

> **一个学习出来的非线性粗网格求解器，或者学习型预条件器。**

GNN 被用于学习稀疏系统预条件器本身已经是一条明确的研究路线；对 XPBD 来说需要进一步处理非线性约束和动态接触集，但方法论是相通的。([arXiv](https://arxiv.org/abs/2406.00809?utm_source=chatgpt.com))

------

## 四、建议让 GNN 输入“残差”，而不只是姿态和历史状态

如果 GNN 作为求解器加速器，它的输入应该从“预测下一帧”切换为“观察当前还剩哪些误差”。

### Node Feature

可以包括：

```text
x_k - x_tilde
当前速度
逆质量
绑定/自由顶点标记
法线
Body SDF
Body SDF Gradient
相对身体速度
前一帧最终位置与速度
```

### Constraint / Edge Feature

```text
当前 C_j(x)
归一化约束残差
当前 lambda_j
XPBD 经典公式计算出的 delta_lambda_j
Compliance
Rest Length / Rest Angle
当前 Stretch Ratio
约束类型
共享顶点数量或层级信息
```

网络输出可以是：

```text
delta_lambda_stretch
delta_lambda_shear
delta_lambda_bend
confidence
```

其中 `confidence` 用来控制修正幅度：

[
\Delta\lambda_{\mathrm{apply}}
=q\cdot\operatorname{clamp}(\Delta\lambda_{\mathrm{GNN}})
]

这样在分布外状态、极快骨骼运动或复杂接触情况下，网络可以自动退化为较保守的 XPBD。

------

## 五、它可以减少穿模，但不能单独保证不穿模

神经网络给出较好的初始状态，确实可以减少：

- 初始穿透深度；
- 碰撞约束需要传播的距离；
- 局部碰撞求解的迭代数量；
- 布料被身体带入内部后难以恢复的情况。

但不能把“避免穿模”的责任交给网络。

### 1. 必须从上一帧到候选位置做 Swept Collision

假设布料顶点：

```text
上一帧在身体左侧
GNN 预测到身体右侧
最终点本身没有处于身体内部
```

单纯检查最终 SDF 会认为没有碰撞，但顶点实际上已经穿过了身体。

对于薄壳和快速运动，离散碰撞检测会漏掉这类事件；更稳健的处理仍然需要 CCD、Swept SDF 或足够保守的子步。([arXiv](https://arxiv.org/html/2403.19272v4))

至少应当对以下线段生成碰撞候选：

[
[x^n,x_{\mathrm{GNN}}]
]

或者：

[
[x^n,\tilde{x}]
]

而不是只检查 (x_{\mathrm{GNN}})。

### 2. 碰撞约束应穿插在结构约束中

不要只在全部 Stretch/Bend 求解结束后做一次碰撞。

更合理的是：

```text
Stretch / Shear
Bend
Body Collision
Stretch / Shear
Self-Collision
Bend
Body Collision
```

否则结构约束可能不断把已经推出身体的顶点再次拉回碰撞体内部。

### 3. 自碰撞仍需要动态空间邻居

固定布料拓扑图只能看到拓扑邻居，而裙摆正反面、腋下两块布、折叠区域的自碰撞经常发生在拓扑距离很远的顶点之间。已有神经布料工作通过增加基于空间距离的动态边来处理这一问题。([arXiv](https://arxiv.org/html/2407.12479v1))

因此至少需要保留：

- Spatial Hash；
- BVH；
- 动态 Proximity Edge；
- 或传统 Triangle-Triangle / Edge-Edge 候选生成。

GNN 可以使用这些动态 Contact Edges，但不应替代 Broad Phase。

------

## 六、训练时不要只学习高迭代结果的位置

比较合理的数据生成方式是：

```text
高质量 Teacher：
高迭代 XPBD / Chaos Cloth / 其他高精度 Solver
                ↓
记录最终 x_ref、v_ref、constraint residual、lambda
```

训练样本则从混合求解器自己的中间状态产生：

```text
x_tilde
    ↓
运行 K_pre 次 XPBD
    ↓
得到 x_k、lambda_k、residual_k
    ↓
网络预测剩余 correction
```

### 主要 Loss 应放在“少量后处理迭代之后”

例如：

[
x'=\operatorname{XPBD}*{K*{\mathrm{post}}}
\left(x_k,\lambda_k+\Delta\lambda_{\mathrm{GNN}}\right)
]

然后优化：

[
L=
w_x|x'-x_{\mathrm{ref}}|^2
+w_v|v'-v_{\mathrm{ref}}|^2
+w_cL_{\mathrm{constraint}}
+w_pL_{\mathrm{penetration}}
+w_eL_{\mathrm{energy}}
]

比直接监督：

[
|\Delta\lambda_{\mathrm{GNN}}
-(\lambda_{\mathrm{ref}}-\lambda_k)|^2
]

更可靠，因为不同约束排序、冗余约束和不同迭代路径可能得到不同的 (\lambda)，但最终顶点状态依然相近。

训练时还应该随机化：

- (\Delta t)；
- 子步数量；
- Pre/Post XPBD 迭代数；
- 材料 Compliance；
- 骨骼运动速度；
- 外力；
- 初始顶点扰动；
- 碰撞厚度；
- 偶发错误状态。

最重要的是使用混合系统自身 rollout 出来的状态继续训练，而不是永远只喂 Teacher 的理想状态。即使数值求解器不能完整反向传播，非微分的时间展开和数据回灌也能明显缓解训练—推理状态分布偏移。([arXiv](https://arxiv.org/html/2402.12971v1))

------

## 七、长期稳定性的几个硬保险

建议同时加入四层保险。

### Trust Region

限制网络一次能产生的最大修正：

[
|\Delta x_i|
<
\eta\cdot \min L_{\mathrm{edge}}
]

或者直接限制每类 (\Delta\lambda)。

### Residual Acceptance Test

应用网络修正前后分别计算：

```text
Stretch Residual
Bend Residual
Minimum Body SDF
Maximum Penetration
Total Correction Norm
```

如果网络修正后综合残差反而变大：

```text
scale = 0.5
```

仍然变大则：

```text
scale = 0
```

这相当于一个极简 GPU Line Search。

### Adaptive Fallback

正常帧：

```text
1 次 GNN + 2～4 次 XPBD
```

困难帧：

```text
1 次 GNN + 8～12 次 XPBD
```

不要通过 CPU Readback 判断，可以在 GPU 上：

- Reduction 得到每件布料的残差；
- 写入 Active Cloth List；
- 使用 Indirect Dispatch 只对失败实例追加迭代。

### 只反馈最终物理状态

网络下一帧历史必须来自：

```text
x_final
v_final
lambda_final 或归一化后的 constraint state
```

不要保留未经 XPBD 修正的 `x_gnn` 作为循环历史，否则原来的自回归漂移仍然会潜伏在隐藏状态中。

------

## 八、性能是否真的划算，要看这个式子

设：

- (T_N)：GNN 推理时间；
- (T_I)：一次 XPBD 迭代时间；
- (K_B)：原始 XPBD 迭代数；
- (K_H)：混合架构迭代数；
- (T_C)：混合架构增加的残差构建、Apply Correction 等成本。

只有在：

[
T_N+K_HT_I+T_C<K_BT_I
]

时，纯 GPU 时间才真正下降。

例如：

- GNN：0.8 ms；
- 一次 XPBD：0.08 ms；
- 额外 Pass：0.1 ms；

那么至少需要节省：

[
(0.8+0.1)/0.08\approx11.25
]

也就是大约 12 次 XPBD 迭代才能打平。

如果一次 XPBD 是 0.2 ms，只需要节省约 5 次。

因此不要只比较“迭代数”，而应比较：

- 总 GPU 时间；
- P95/P99 时间；
- 接触密集帧的回退率；
- 多角色 Batch 情况；
- 残差达到同一质量阈值所需时间。

基线还应该加入“更多子步、每个子步一次 XPBD”的方案，而不只是“一个大步里做很多次迭代”，因为 XPBD 的 Small Steps 路线在相同计算预算下经常具有更好的稳定性和收敛表现。([Mmacklin](https://mmacklin.com/smallsteps.pdf))

------

## 最终建议

你们可以分两步推进。

### 第一阶段：直接验证混合架构价值

复用当前 GNN：

```text
GNN 预测位置
→ Swept Body Collision
→ 2～4 次 XPBD
→ 最终碰撞修正
→ corrected state 进入下一帧
```

同时加入：

- 位移 Clamp；
- Body SDF Margin；
- GPU Residual；
- 超阈值追加迭代。

这一步能快速判断长期稳定性是否显著改善。

### 第二阶段：把 GNN 改造成真正的 Learned Preconditioner

最终结构建议为：

```text
惯性预测
→ 1～2 次 XPBD Pre-Smooth
→ 构建约束残差图
→ GNN 预测结构约束 Δλ / coarse correction
→ 1～3 次 XPBD Post-Smooth
→ 碰撞与摩擦收尾
→ 自适应追加迭代
```

角色分工是：

> **GNN 解决 XPBD 最不擅长的低频、全局传播问题；XPBD 解决 GNN 最不可靠的硬约束、碰撞和长期稳定性问题。**

这比“神经网络负责全部布料动力学，XPBD 偶尔救场”更容易形成可控、可调试、可跨服装泛化的工程系统。尤其是你目前的小 GNN 已经能够在 1 ms 内运行，下一步最值得做的不是继续压榨纯推理效果，而是把输出从“绝对顶点结果”逐步转成“求解器残差或约束空间修正”。