# XPBD 中文导读：`α̃ = compliance / Δt²` 是怎么来的

面向本仓库读者。目标不是复述论文，而是把**当前 PoC 代码里那几行为什么长这样**
讲清楚。对应实现：
[`gnn_constraints.comp`](../../../implementations/vulkan-gnn-poc/overlay/shaders/hlsl/gnncloth/gnn_constraints.comp)、
[`gnn_constraints_tiled.comp`](../../../implementations/vulkan-gnn-poc/overlay/shaders/hlsl/gnncloth/gnn_constraints_tiled.comp)。

## 一、PBD 的那个老问题：刚度依赖迭代次数

PBD（Müller 2007）不解力，直接改位置。对约束 `C(x) = 0`，沿梯度投影：

```
Δx_i = -w_i · ∇_i C · C / Σ_j w_j |∇_j C|²
```

其中 `w_i = 1/m_i`。乘一个用户刚度 `k ∈ [0,1]` 就得到「软」约束。

问题在于：一次 Gauss-Seidel 扫描只把误差消掉一部分，扫 N 次后残差约
`(1-k)^N`。于是**同一个 k，迭代 5 次和迭代 50 次表现出的软硬完全不同**；改帧率、
改迭代预算，布料手感就变。这对生产是致命的——美术调好的参数换台机器就不对了。

## 二、XPBD 的做法：把拉格朗日乘子显式记下来

XPBD（Macklin 2016）回到隐式积分的视角。约束的弹性势能取

```
U(x) = ½ · C(x)ᵀ · α⁻¹ · C(x)
```

`α` 是**柔度（compliance）**，即刚度的倒数：`α = 1/k`。约束力是

```
f = -∇Uᵀ = -∇Cᵀ · λ ,   λ = -α⁻¹ · C
```

把 `λ` 当作独立未知量，对隐式欧拉做一步牛顿，并令 `α̃ = α / Δt²`，可得每次
迭代对乘子的增量：

```
Δλ = ( -C - α̃ · λ ) / ( Σ w_i |∇_i C|² + α̃ )
Δx_i = w_i · ∇_i C · Δλ
```

**`Δt²` 是从哪冒出来的**：隐式欧拉把力写进位置方程时带了 `Δt²` 因子
（`M(x - x̃)/Δt² = f`）。把它移到约束一侧，柔度就以 `α/Δt²` 的形式出现。
所以 `α̃` 不是经验系数，而是「柔度在位置空间里的等效量」。

两个关键性质，正是 PoC 需要的：

1. **刚度与迭代次数解耦。** `λ` 跨迭代累积，`-α̃·λ` 项会在接近收敛时抵消掉
   继续投影的动力。迭代更多次只是更收敛，而不是更硬。
2. **刚度与 Δt 解耦。** `α̃` 里除了 `Δt²`，恰好抵掉隐式积分引入的 `Δt²`。
   所以变步长在 XPBD 里不是缺陷。这也是本 PoC 用
   `deltaT = min(frameTimer, 0.02)` 变步长仍然站得住的原因。

`α → 0` 时 `α̃ → 0`，退化为标准 PBD 的硬投影。

## 三、对照 PoC 代码

`gnn_constraints.comp` 的核心几行与上式一一对应：

```hlsl
const float alphaTilde = max(complianceMicro, 0.0) * 1.0e-6 / max(deltaT * deltaT, 1.0e-8);
const float lambda = constraintLambdas[edgeIndex];
const float deltaLambda = (-constraintValue - alphaTilde * lambda) / (inverseMassSum + alphaTilde);
```

- `constraintValue` 就是 `C = |p0-p1| - restLength`（距离约束）。
- `inverseMassSum = w0 + w1`。距离约束的 `|∇C| = 1`，所以分母里的
  `Σ w_i |∇_i C|²` 直接退化成 `w0 + w1`。
- UI 单位是 `×1e-6`：`stretchComplianceMicro = 1` 表示 `α = 1e-6`。
- `λ` 每个时间步清零、迭代内累积——这正是性质 1 的前提。**如果每次迭代都重置
  `λ`，XPBD 就退回成 PBD**，刚度重新依赖迭代次数。

### 数量级：什么时候 λ 真的重要

以 `Δt = 1/60`、`particleMass = 0.1`（`w = 10`，`w0+w1 = 20`）为例：

| 项 | compliance | α̃ | 占分母比例 |
| --- | --- | ---: | ---: |
| 拉伸/剪切（默认 1e-6） | `1e-6` | `3.6e-3` | 0.018% |
| 弯曲（默认 1e-2） | `1e-2` | `36` | 64% |

所以：**拉伸约束几乎是刚性投影，`λ` 影响可以忽略；弯曲约束的 `α̃` 反而大于
质量项，`λ` 起主导作用。** 这解释了 tile 化实现里为什么必须把 `λ` 存回全局
缓冲、按 `(type, x, y)` 解析索引跨 pass 保持——若只在 tile 内累积再丢弃，
弯曲行为会明显改变。

## 四、本 PoC 中未做到的部分

- **弯曲不是二面角约束**，而是两跳距离近似（`restLength = 2·restDistH`）。
  真正的弯曲约束作用在两个相邻三角形的二面角上，需要四顶点梯度。新引入的
  `Assets/` 里 VCLOTH 资产已经带 3763 个四顶点二面角约束，是下一步的方向。
- **迭代次数不足。** 实测黄金场景最大拉伸应变 0.81：8 次 Gauss-Seidel 只能把
  张力传播约 8 条边，而布片有 32 格宽，两角悬挂的张力传不到位。按
  *Small Steps in Physics Simulation* 的结论，**增加子步比增加迭代更有效**。
- **无自碰撞。**

## 五、延伸

- Macklin, Müller, Chentanez 2016, *XPBD*（上式出处，含推导细节与稳定性讨论）
- Müller 等 2007, *Position Based Dynamics*（PBD 原始形式）
- Macklin 等 2019, *Small Steps in Physics Simulation*（子步 vs 迭代）
