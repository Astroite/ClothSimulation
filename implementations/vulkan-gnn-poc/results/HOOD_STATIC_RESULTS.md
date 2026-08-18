# CH10032 T-Pose + HOOD Fine15 static results

Test machine: NVIDIA GeForce RTX 4060 Ti, Vulkan SDK 1.4.309, FP32 HLSL/SPIR-V.

## What is static

The source is CH10032's native `AS_C10032_Tpose`. Its `VANIM v1` contains one frame and 45 retained core/deformation bones. Character skinning, the 4,096-vertex collision proxy and all 72 waist pin targets therefore remain fixed. The 1,377 cloth vertices are not paused: Fine15 takes its first `1/3 s` step and then continues autoregressively at `1/30 s` against the same body pose.

The cloth graph has 7,894 directed mesh edges. The single-frame Python reference produced 870 world edges. The waist binding distance is 4.995 mm median and 9.582 mm maximum.

## Numerical verification

The optional body projection was disabled. Vulkan matched the pure-PyTorch one-step golden result with a `1.177e-6 m` maximum position error, `1.215e-8 m` mean position error, `1.179e-6` maximum acceleration error, and 0/1,377 world-edge mismatches. Khronos validation and synchronization validation reported no errors.

## Detailed GPU timings

> **These timings are superseded.** They were measured with Khronos validation and
> synchronization validation enabled and with the GPU clock unlocked, which makes identical
> work vary by up to 2.2x between runs, so they cannot serve as the denominator of any
> speedup. For the clean reproducible measurements, the method behind them, and the A/B data
> for the weight-layout transpose, see [`KERNEL_OPTIMISATION_RESULTS.md`](KERNEL_OPTIMISATION_RESULTS.md).
> The structural metrics and the numerical verification conclusions are unaffected.


The run discarded 5 warmup samples and recorded 20 samples with Vulkan GPU timestamps. Times include the required compute barriers between stages, but exclude CPU command submission, graphics drawing and present. Full mean/min/p95/max rows are in [`hood_static_timing.csv`](hood_static_timing.csv).

| Stage | Mean (ms) | P95 (ms) | Share of total |
|---|---:|---:|---:|
| GPU skinning | 0.056 | 0.060 | 0.01% |
| Features + nearest world edge | 1.171 | 0.895 | 0.22% |
| Four encoders | 18.583 | 28.254 | 3.42% |
| 15 GraphNet processor blocks | 521.309 | 580.830 | 96.05% |
| Decoder + Verlet integration + pins | 1.633 | 11.202 | 0.30% |
| **Total compute** | **542.752** | **601.109** | **100%** |

Encoder means were 5.994 ms for nodes, 9.725 ms for mesh edges, 1.140 ms for direct world edges and 1.723 ms for inverse world edges. A stage's P95 can be lower than its mean when one of the 20 samples is a large outlier; stage percentiles are intentionally not added together.

| Block | Edge update mean (ms) | Node update mean (ms) | Block total mean (ms) |
|---:|---:|---:|---:|
| 00 | 27.677 | 4.648 | 32.325 |
| 01 | 29.123 | 5.453 | 34.576 |
| 02 | 27.909 | 7.436 | 35.345 |
| 03 | 29.515 | 5.379 | 34.894 |
| 04 | 30.759 | 4.160 | 34.919 |
| 05 | 30.412 | 7.016 | 37.428 |
| 06 | 29.598 | 5.467 | 35.065 |
| 07 | 29.034 | 4.859 | 33.893 |
| 08 | 28.772 | 6.458 | 35.229 |
| 09 | 25.615 | 8.576 | 34.191 |
| 10 | 27.906 | 6.953 | 34.860 |
| 11 | 29.205 | 5.309 | 34.514 |
| 12 | 28.518 | 6.938 | 35.456 |
| 13 | 29.717 | 4.289 | 34.006 |
| 14 | 27.192 | 7.415 | 34.608 |

The total ranged from 481.583 to 651.309 ms. Removing animation confirms that animation playback and LBS are not the performance problem: the direct 128-lane, one-workgroup-per-graph-element Fine15 processor dominates. The next optimization target is therefore the MLP/message-passing implementation and dispatch/barrier structure, not animation sampling.

## Visual check

![CH10032 native T-Pose with continuously simulated Fine15 cloth](hood_ch10032_tpose.png)
