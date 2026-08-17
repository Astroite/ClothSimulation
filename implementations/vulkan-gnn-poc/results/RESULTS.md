# RTX 4060 Ti validation results

Tested on 2026-08-17 with Vulkan SDK 1.4.309, an NVIDIA GeForce RTX 4060 Ti,
and NVIDIA driver 596.36. HLSL was compiled by the SDK's DXC to Vulkan 1.1
SPIR-V.

The single-step GPU result passed against the Python reference with maximum
absolute error `1.907348633e-06` and mean absolute error `3.117110078e-07`.
The 1200-frame health check reported no non-finite value or pinned-vertex drift.
A reset after 600 frames followed by the same 600 fixed-time-step frames was
bit-identical (`reset_replay_max_abs = 0`). Khronos validation plus
synchronization validation produced no error or warning. A hidden interactive
smoke run switched GNN → mass-spring → GNN and reset both paths without a
validation message. The interactive GNN path used eight colored Gauss-Seidel
XPBD iterations. Each undirected stretch, shear, and two-hop bend-distance
constraint owns an accumulated lambda and applies
`alpha_tilde = compliance / delta_time^2`; lambdas are cleared per time step and
retained across iterations. A 64×64 run still retained an extended hanging
sheet with sphere-contact folds after 15 seconds (about 900 VSync frames)
instead of collapsing into the bundle produced by the unconstrained rollout.
The finalize pass reconstructs velocity from the XPBD-corrected position,
applies exponential damping (1.5 by default), and treats sphere contact as a
zero-restitution normal-velocity projection so collision correction cannot
inject an artificial launch impulse.

The interactive sphere follows `x=1.2*sin(0.7*t)` and
`z=0.65+0.55*sin(1.1*t)`. Its analytic velocity is passed to both solver paths;
contact removes inward cloth velocity relative to the sphere, rather than
treating the moving collider as static. A validation-enabled smoke run exercised
moving contact for two seconds before switching and resetting both solvers with
no validation or synchronization message. The benchmark keeps the sphere still.

The final graphics path draws a procedural full-screen sky before scene
geometry, with a blue zenith, pale horizon, and soft sun. Brighter two-sided
cloth lighting and a lifted warm sphere material improve fold readability.
The sky adds no texture dependency and does not enter the compute timestamps.

| Grid | Nodes | Directed GNN edges | XPBD constraints | Layer 0 median | Layer 1 + integrate + XPBD median | Total median | Total p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16×16 | 256 | 1,860 | 1,378 | 0.007584 ms | 0.276480 ms | 0.283648 ms | 0.287680 ms |
| 32×32 | 1,024 | 7,812 | 5,826 | 0.012576 ms | 0.328416 ms | 0.340800 ms | 0.345408 ms |
| 64×64 | 4,096 | 32,004 | 23,938 | 0.027104 ms | 0.374208 ms | 0.401216 ms | 0.407488 ms |

The timings are measurable and total time increases monotonically with graph
size. Eight iterations over sixteen color batches make fixed dispatch and
barrier overhead visible at the smaller grids; batching several colors in one
shader is the clearest future optimization. This is only a deployment-chain
measurement: the synthetic one-step target plus XPBD fallback is not evidence
of production cloth quality.

## What the network actually contributes

`.\ablation.ps1` runs the same deterministic 600-step scenario four times,
changing only where the acceleration comes from, and compares final positions
(`results/gnn_ablation.json`). `analytic` evaluates the training target
directly, `gravity` is that formula with the neighbor coupling removed, and
`zero` supplies no acceleration at all.

| Comparison | Mean vertex distance | L2 |
| --- | ---: | ---: |
| `analytic` vs `gravity` (all of graph message passing) | 0.0035 | 0.1436 |
| `gnn` vs `analytic` (the network's own error) | 0.0451 | 2.1158 |
| `gnn` vs `zero` (any acceleration at all, gravity included) | 0.2363 | 8.0621 |

Two conclusions, both worth stating plainly:

- The acceleration term matters a great deal, but almost entirely through
  gravity. Removing the whole neighbor coupling moves the cloth by 3.5 mm mean
  over 600 steps, because the near-rigid XPBD distance constraints already
  enforce what the Laplacian term was approximating.
- The network's own approximation error is **14.7x larger than the effect the
  coupling produces**. So on this scenario the graph does not earn its cost, and
  the network is a more expensive, less accurate way to evaluate a formula that
  is three lines long.

This does not invalidate the PoC; it bounds its claim. What is demonstrated is
the deployment chain -- fixed weights, pure compute-shader inference, no host
readback, direct vertex-buffer draw -- and that chain is reproducible and
numerically verified. Learned dynamics are not demonstrated, and would need a
reference simulator for supervision plus a scenario where the coupling is not
already subsumed by the constraint solver.

The 1200-frame verification also reports physical health, not just finiteness:
cloth AABB extent, maximum stretch and bend strain, and how many vertices hit
the hard acceleration and speed clamps. In the golden scenario 402 of 1024
vertices hit the acceleration clamp and maximum stretch strain reaches 0.81.
That scenario hangs a 32-wide sheet from only two corners, where eight
Gauss-Seidel iterations cannot propagate tension across the sheet; the clamp
count also shows the network running well outside the 16x16 two-corner
distribution it was trained on. Both numbers were previously invisible.

![GNN cloth screenshot](gnn_cloth.png)
