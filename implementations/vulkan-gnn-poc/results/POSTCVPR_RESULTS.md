# HOOD PostCVPR Vulkan 测试结果

测试日期：2026-08-18。测试机：NVIDIA GeForce RTX 4060 Ti，驱动 `596.36.0.0`，
Vulkan SDK 1.4.309，FP32 HLSL/SPIR-V。

## 结论

官方 `postcvpr.pth` 已经完成下面这条可复现链路：

`checkpoint -> VHOOD -> 确定性层级图 -> 纯 PyTorch -> Vulkan Compute -> 布料顶点直接绘制`

数值实现正确，10 步自回归最大位置误差为 `8.87e-7 m`，Khronos validation 和
synchronization validation 均无错误。但当前 kernel 在 CH10032 T-Pose 上需要
`44.30 ms/模拟步`（min），只能达到约 `22.6 steps/s`，达不到 30 Hz。与同一测量方法下
优化后的 Fine15 `24.39 ms/步` 相比，PostCVPR 慢 `1.82x`。

原始 PostCVPR 在无 XPBD、无身体碰撞投影的 74 步静态闭环中保持有限值和固定腰环，
但会产生明显的局部翻面和身体穿透。因此它证明了官方层级模型的部署链，而不是当前角色上
已经达到可交付视觉质量。

## 模型与资产

- 来源：[HOOD 官方仓库](https://github.com/Dolorousrtur/HOOD)，锁定提交
  `9bc1076195979ac6c027fdd729c6e960cad62f2a`。
- checkpoint SHA-256：`155d2dd25e54756fc04b0d27996ebca3446b2a59d3a715bb1fb73407753ce5ea`。
- `postcvpr.pth` 是官方在 CVPR 后重构和修复后训练的层级模型，不是 Fine15 单层图基线。
- 输入：node/mesh/coarse/world 为 `24/12/12/9` 维；latent 为 128；输出为 3 维加速度。
- 15 个 GraphNet block 按 `fine+c0 -> c0+c1 -> c1 -> c0+c1 -> fine+c0` 分成五组，
  每组 3 个 block。
- CH10032 裙装：1,377 个节点、7,894 条有向 mesh edge；层级图中心为顶点 967，
  c0/c1/c2 分别为 3,182/1,644/804 条有向边。当前官方 schedule 实际计算 c0 与 c1，
  c2 用于 vertex-level embedding，但没有对应 processor block。
- 运行时 `VHOOD` 为 20,697,888 bytes，SHA-256
  `a167ee1607b71bcdfa37ac0a1101ee1e048e718d1ba4e3f3153af0e6729c06e9`。

层级拓扑由不依赖 NetworkX 的确定性 Python 实现烘焙为 `VPHIER01 v1`。节点更新保持官方的
有序聚合输入；world edge 在较粗层只连接当前层级活跃的 cloth vertex。正常帧全部资源常驻
device-local buffer，不做 host readback。

## 正确性

命令：

```powershell
.\tools\fetch_hood_postcvpr.ps1
.\.venv\Scripts\python.exe .\tools\run_postcvpr_reference.py `
  --asset-root .\.work\real_scene\ch10032_tpose --motion ch10032_tpose --steps 10 `
  --golden .\.work\real_scene\ch10032_tpose\postcvpr_rollout.vhgold
.\build.ps1
.\verify_hood.ps1 -Motion ch10032_tpose -Solver PostCvpr
```

10 步 Vulkan 相对纯 PyTorch：

| 指标 | 结果 |
| --- | ---: |
| 最大位置绝对误差 | `8.8662e-7 m` |
| 平均位置绝对误差 | `2.2513e-8 m` |
| 首步加速度最大误差 | `1.2154e-7` |
| 首步加速度平均误差 | `1.1337e-8` |
| 首步 world edge 不匹配 | `0` |
| 最大固定点误差 | `0 m` |
| NaN/Inf | `0` |

checkpoint 与导出后的 `VHOOD` 所有 tensor 逐元素相同，最大误差为 0。

## GPU 时间

命令：

```powershell
.\benchmark_hood_static.ps1 -Scene CH10032 -Solver PostCvpr -Warmup 10 -Samples 60
```

测量时锁定 SM 2700 MHz、关闭 validation，时间包含必要的 compute barrier，不包含绘制和 present：

| 阶段 | min (ms) | mean (ms) | P95 (ms) |
| --- | ---: | ---: | ---: |
| skin | 0.0449 | 0.0548 | 0.0619 |
| features + world | 0.5917 | 0.5972 | 0.6022 |
| encoders | 1.0240 | 1.0539 | 1.2462 |
| hierarchical processor 15 blocks | 42.5779 | 44.1416 | 46.0144 |
| decoder + integrate | 0.0492 | 0.0502 | 0.0512 |
| **total** | **44.3026** | **45.8977** | **47.7350** |

processor 占最小总时间的 96.1%，下一轮优化应继续针对 node/edge MLP dispatch，而不是蒙皮、
world-edge 搜索或积分。Claude 实现的 output-major 权重转置和 cooperative groupshared 输入加载
已经被 PostCVPR 路径复用；没有这两项优化，本次层级模型测试不具备实际可用的迭代速度。

## 74 步结构观测

无 XPBD、无身体碰撞投影：

- invalid vertices：0；固定点最大误差：0。
- edge length ratio：mean `1.0145`，P95 `1.1833`，max `2.5691`。
- 拉伸超过 1.5 倍的边：`1.19%`；小于 0.5 倍的边：`0.30%`。
- 退化三角形：`0.039%`；翻面三角形：`22.72%`。
- 最大位移：`0.5714 m`。

结构没有像未训练好的 TinyHOOD 那样整体拉成条，但局部翻面比例较高，截图也能看到腰部附近
的穿透和开裂观感。后续若评价视觉效果，应分别测试官方 raw output、身体碰撞投影和 XPBD 后处理，
不能用后处理结果替代 raw model 的质量结论。

截图：[`postcvpr_ch10032_tpose.png`](postcvpr_ch10032_tpose.png)。原始数据见
[`postcvpr_verify.json`](postcvpr_verify.json)、
[`postcvpr_ch10032_tpose_timing.csv`](postcvpr_ch10032_tpose_timing.csv) 和
[`postcvpr_ch10032_tpose_stability.json`](postcvpr_ch10032_tpose_stability.json)。
