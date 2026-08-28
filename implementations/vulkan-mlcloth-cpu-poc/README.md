# Vulkan MLCloth CPU inference upload PoC

Windows-only validation sample for the deliberately narrow path:

```text
CH10032 driver clip (30 Hz) -> AILab/MNN CPU inference
-> 5,294 Root_M-local UE-cm points -> per-frame Vulkan upload
-> compute root/axis/unit transform -> triangle (or point-list) rendering
```

It has an independent pinned Sascha Willems checkout under `.work/Vulkan` and
builds `mlclothcpu.exe`. Models and vendor DLLs are copied from the local
MLCloth plugin into ignored `.work/runtime`; no vendor binary is committed.

```powershell
pwsh ./prepare_runtime.ps1
pwsh ./bake_driver_clip.ps1
pwsh ./build.ps1
pwsh ./run.ps1 -Verify -Validation
```

The driver bake defaults to `E:\Main\Projects\Z2Game\Z2Game.uproject` and
`E:\Main\Engine\Binaries\Win64\UnrealEditor-Cmd.exe`; all asset paths and the
project/editor path remain CLI-overridable. Z2Game currently contains the
CH10032 mesh and animation but not the locked `.enc` model or vendor runtime
DLLs. Therefore `prepare_runtime.ps1` defaults to the legacy PaperGame MLCloth
plugin as the artifact source. Override `-MLClothRoot` after those artifacts
are synchronized into another workspace.

Use `run.ps1 -Benchmark -Threads 1` for a 200-frame warm-up and 1,000-frame
capture. `P` pauses, `R` deterministically resets and `Esc` exits.

## Geometry

Without `-Mesh` the sample renders a point cloud, exactly as it did before the
topology existed, so the verified coordinates and the benchmark numbers are
unaffected by anything below. With a `.mlmesh` it draws shaded triangles;
`-Points` forces the point path back on.

`run.ps1` picks up `.work/mesh/ch10032_cloth2607.mlmesh` automatically when it
is present.

```powershell
pwsh ./bake_cloth_topology.ps1 -ExportRoot <MLCloth training export directory>
pwsh ./bake_mlcloth_drivers.ps1 -ExportRoot <MLCloth training export directory>
py -3 ./tools/bake_mlcloth_capsules.py
```

### Driver clips without Unreal

`bake_driver_clip.ps1` evaluates `AnimPose` inside the editor. The training export
already carries the same information: `bones_local.bin` and `bones_component.bin`
hold a full transform per bone per frame, and that bone order is byte-identical to
the model's own `driverNames` (checked element by element, not assumed). The 6D
rotation feature is the quaternion applied to two basis vectors, so
`tools/bake_mlcloth_drivers.py` reconstructs all three features offline.

That is verified rather than argued. Cross-checked against the editor's own bake of
`AS_C10032_ArmedSprint_Skirt`, aligning export frame 34 -- its
`leadingStaticPaddingFrames` -- with `.mldrv` frame 0:

| feature | max abs error | tolerance |
| --- | --- | --- |
| component position (cm) | 0.0 | **0** -- copied, not recomputed |
| local 6D rotation | 1.52e-06 | 3e-06 |
| component 6D rotation | 1.46e-06 | 3e-06 |

Bit-exact positions also confirm the frame alignment: a one-frame slip in a sprint
would show centimetre-scale error. `tests/test_mlcloth_drivers.py` keeps that
comparison running, and the wrapper additionally feeds one baked clip to the real
AILab runtime, because a container Python considers well-formed can still be
refused by the C++ `parse_clip`.

Clips are emitted at exactly the training frame count, one frame per
`cloth_sim.bin` frame, so a `.mldrv` frame and a ChaosCloth reference frame are the
same instant. The exporter's static padding is therefore **kept**, not trimmed --
trimming would silently desynchronise the two -- and the padding ranges go into the
sidecar so a consumer excludes them deliberately.

### Where the topology comes from, and why not from Unreal Python
The 5,294 model outputs belong to the LOD0 sim mesh of the ChaosClothAsset
`CCA_JZ_CH10032_2607_ML1`, and `MLClothSimulationProxy.cpp` shows both that the
inference output is already in that mesh's Root_M-local centimetre space and
that the topology is `GetClothSimulationModel(0)->GetIndices(0)`. That mesh is
**not reachable from Unreal Python**: on `UChaosClothAsset` the sim model is a
bare `TSharedPtr` rather than a `UPROPERTY`, `GetClothSimulationModel` is not a
`UFUNCTION`, and `ClothCollections` is unreflected too. `FClothPatternToDynamicMesh`
with `Sim3D` + `PatternIndex = INDEX_NONE` would give exactly the welded ordering
needed, but it is unreflected C++ as well, and the one reflected converter that
exists (`SkeletalMeshConverter`) asks for `EClothPatternVertexType::Render`,
which is a different mesh.

The MLCloth plugin already exports this for its own training pipeline:
`AIClothTrainingAnimationBuilder` writes `cloth_topology.bin` (`AICCTOP1`) beside
`cloth_sim.bin` (`AICCLTH1`/`AICCLTH2`), with `manifest.json` recording the
guarantee the conversion rests on --

> `cloth_sim.bin` positions use zero-based array order; `cloth_topology.bin`
> indices address that same array order

-- and that same order is what the inference output uses. So
`tools/bake_cloth_topology.py` reads a training export offline, with no Unreal
dependency at all, and refuses the pairing if the manifest ever stops stating
that guarantee. `cloth_sim.bin` also carries ChaosCloth's own per-frame
positions, which is a reference the constraint calibration would otherwise not
have.

`tests/mesh_validate.cpp` re-derives the edge set, both CSRs and the boundary
structure in C++ and rejects a file whose Python-side derivation disagrees, so
the two implementations are pinned to each other rather than each trusted to be
self-consistent. `bake_cloth_topology.ps1` runs it automatically.

### Two conventions in this data that are silently wrong if guessed

**Up is the reference bone's local X, not Z.** The per-frame `transform` stored
in `cloth_sim.bin` is the reference bone's own component transform, and its
rotation maps local X to world up (alignment 0.9899), local Y to -Y and local Z
to +X. The converter therefore *derives* the up axis from that transform;
`--up-axis` is an assertion that fails the bake on disagreement, not a setting.
The plausible-looking `--up-axis z` produced a perfectly well-formed mesh whose
pin set was an arbitrary 21-vertex slice, with nothing reporting a problem.

**Pins come from measured attachment, not rest-pose height.** This garment is
four disconnected pieces with nine boundary loops, so "the highest boundary
loop" is wrong; per-component height is wrong too. Scoring each loop by how
rigidly it rides its best-matching driver bone -- the standard deviation of each
loop vertex in that bone's local frame, normalised per clip -- separates six
attached loops (432 vertices) from three free ones with a 7.2x gap that holds
across run, dodge, jump and turn. Height instead pins one of the *free* cuffs,
which measurement shows tracks `Wrist_R`. Pass `-AttachmentClip` (or let the
wrapper pick and print a deterministic spread); without any, the bake falls back
to height and says so in `pin_rule_warning`.

Absolute residuals scale with how violent the motion is -- a dodge moves about
three times a run -- so the threshold is on the per-clip normalised value, and a
loop landing between `--attached-below` and `--free-above` is a hard error rather
than a coin flip.

### The rest pose is a simulation result, not an asset

Two exports of the same `TPose_Reference` under the same settings gave surface
areas 6.9% apart and maximum edge lengths 49% apart, while the median edge agreed
to 0.3% and the attached loops did not move at all. It is not a convergence
problem: across one 90-frame settle the area varies 0.09% from frame 0 onward,
yet the largest per-vertex frame-to-frame drift never falls below ~2 cm -- the
garment is metrically converged and still fluttering, and the two exports landed
in different equilibria.

So the report records `sim_sha256` and `topology_sha256`, and the wide rest-edge
spread is reported as ratios to the median (26.8% of edges over 1.5x, 10.0% over
2x, one over 4x) because that spread is the tessellation rather than damage. The
consequence for calibration is concrete: a `target_length` taken from a high
quantile of the ChaosCloth reference, or a gate built on its p95 or max, measures
which equilibrium that export happened to reach.

### Collision

`tools/bake_mlcloth_capsules.py` bakes the 14 capsules of the physics asset the
character actually references at runtime into `MLCAP001`. All 14 of its bones are
among the model's 45 drivers, so the `.mldrv` clip already carries their per-frame
transforms: collision needs no new per-frame data, no vertex proxy and no runtime
skinning. A capsule's signed distance is closed-form and unambiguous about inside,
which is a deliberate correction to the sibling PoC's nearest-proxy half-plane --
`vulkan-gnn-poc/results/PROGRESS.md` section 6 records that criterion reporting
28.5% penetration at 0.12 m and only 1% at 0.25 m, because past the far surface
its signed distance turns positive again.

## XPBD post-constraint, on the CPU

`include/mlcloth_xpbd.{h,cpp}` is a port of `vulkan-gnn-poc/real_scene/xpbd.py::project`
with `sweep="jacobi"`: structural sweep, then area floor, then guide, then contacts, that
order every iteration. It runs in **metres** so the sibling PoC's tuned compliance,
contact-offset and area constants transfer unchanged.

### Why the CPU and not a compute shader

Measured, not assumed. On this machine MNN CPU inference is 2.574 ms median against
0.0169 ms for the entire GPU pass, so the CPU is the bottleneck and the GPU is 99.3% idle.
A compute-shader solve would therefore be close to free, while a CPU solve is additive:

| configuration | total step (median of 1,000) | solve share |
| --- | --- | --- |
| pure network | 2.600 ms | -- |
| + XPBD k=4 | 4.067 ms | ~1.47 ms |
| + XPBD k=8 | 4.924 ms | ~2.32 ms |
| + XPBD k=8, area floor | 6.035 ms | ~3.44 ms |

So the shader port is worth doing, and that is now a measurement rather than a guess. It is
sequenced after the CPU solver for three reasons: the open question here is calibration
rather than throughput, and sweeping calibration sources across 86 clips is a loop on the
CPU and a windowed Vulkan session on the GPU; the CPU version becomes the oracle the port
is validated against; and only the CPU version can be held to a tight comparison at all --
`hood_xpbd.comp` records that its 1.68e-3 gap against the same Python reference is not a
summation artifact but the one-sided `max(residual, 0)` and the contact's `signed < 0`
landing on a branch boundary.

### The comparison that validates it

`tests/test_mlcloth_xpbd.py` hands the same scenario to the Python reference and to
`tests/build/xpbd_reference.exe` and compares. Agreement, over displacements of 1.4 to
2.3 cm:

| pass | gap, as a fraction of displacement |
| --- | --- |
| stretch only | 8.7e-08 |
| stretch + bend | 7.6e-07 |
| two-sided under compression | 1.0e-06 |
| area floor | 6.5e-07 |
| guide | 3.9e-07 |
| guide + trust gate | 3.9e-07 |
| all four | 6.6e-07 |

Writing that harness paid for itself before it ran: reading the reference closely enough to
build it surfaced two errors in the guide pass -- confidence multiplied onto the correction
instead of dividing into alpha-tilde, and measured against the evolving position instead of
the Verlet extrapolation `2 * x_now - x_previous`. Both would have produced a plausible
guide that was the wrong function.

### The no-regression gate

`--xpbd-iterations 0` must reproduce the pure-network output exactly, and
`mlcloth_verify.json` reports `first_solved_hash` so that is scriptable:

| configuration | `first_solved_hash` |
| --- | --- |
| pure network | `0xd36ec99cd356ec23` |
| `--xpbd --xpbd-iterations 0` | `0xd36ec99cd356ec23` |
| `--xpbd --xpbd-iterations 1` | `0x85b948c817e3282a` |
| `--xpbd --xpbd-iterations 8` | `0x8340620d091a3df8` |

The hash is over the positions that reach the GPU, not the network's raw output. Hashing the
raw output made the gate vacuous -- it is unaffected by the solver by construction, so k=0
and k=8 both reproduced it while k=8 was visibly moving vertices 2.5 cm. And zero iterations
skips the centimetre-to-metre round trip rather than performing and undoing it, because
`x * 0.01f * 100.0f` differs from `x` by an ulp on most values, which was by itself enough
to break the equality.

### Compliance magnitudes are not intuitive here

`alpha-tilde = compliance / dt^2`, so at 30 Hz a compliance of 1.0 gives 900 against an
inverse-mass sum near 90: the constraint is then so compliant it barely acts. The useful
band is roughly 0 to 0.1 for the structural constraints, and `SolverConfig`'s own note
records the sibling PoC hitting the same trap ("gate G0 swept stretch compliance over
0..1e-1 and found the entire range inert"). The defaults here are rigid stretch and
0.05 bend for that reason.

### Not yet wired

Constraint targets currently come from the mesh's own stored rest configuration. For this
asset that is ChaosCloth's settled T-pose rather than an authored flat pattern, so it is a
defensible control arm -- but calibrating over a trajectory is the point of the S13 sweep,
and this is the arm it will be measured against.

Collision is complete in the solver (analytic capsule signed distance, deepest-capsule
resolution, tested against hand-computed geometry) and in the data (14 baked capsules), but
the runtime does not yet place the capsules per frame from the clip's bone transforms, so
`--xpbd` runs without contacts. The solver's contact test also cannot move a pinned vertex,
which is correct -- pins are kinematic -- and means a network that puts a *pinned* vertex
inside the body is not something anything downstream will fix.

## Picking an animation at runtime

The overlay carries an `Animation` combo box over every `.mldrv` beside the loaded clip --
86 of them after `bake_mlcloth_drivers.ps1` -- plus an integer playback-speed slider and a
hold-last-frame toggle. `-ClipDir` points the scan somewhere else.

This follows the sibling PoC's `hoodCollectMotions` / `hoodLoadAnimation` split, including
the reason it is split that way: switching motions replaces the clip and **nothing else**.
The model, the AILab runtime, the topology, the constraint set and the pin set are shared by
every motion of this garment. That is the correctness argument, and it is also the
zero-per-asset-authoring claim the whole comparison rests on -- if changing animation needed
anything re-authored, the claim would be false.

A clip that fails `parse_clip`'s model-hash check leaves the current one playing and reports
why in the overlay, rather than tearing down a working state. A directory of bakes is a place
where a clip built against another model is a real thing to meet.

Frame decimation is the speed axis, and the **timestep is deliberately not scaled with it**:
the question is what happens when the body moves further per solver step, and scaling the
timestep would cancel exactly that. Integer only, so a setting reproduces a column rather
than landing between two.

### What switching had to fix first

`resetSequence` reset the recurrent network state but not the solver's two-frame Verlet
history, and the clip loop-back reset neither. Both are wrong in the same way: frame 0 of the
second pass would take the *first* pass's last frames as its inertial reference, so a looped
or switched-to clip would not reproduce the same clip launched directly.

Nothing caught it, because the existing reset-determinism assertion hashed the network's raw
output -- which the solver cannot influence. `runCpuVerification` now checks both hashes, and
the solved one is what makes it a check of the whole step rather than of the recurrent state
alone. It throws with "solver state leaked across the clip loop" if they diverge.

## Still excluded

The 5,294-to-1,377 mapping, self-collision, materials, MNN GPU execution, and the XPBD
compute-shader port.

The 5,294-to-1,377 mapping was originally excluded on the grounds that it would be a lossy
resample. That premise turned out to be false: the skirt piece of this garment has exactly
1,377 vertices and 2,570 triangles, identical to the sibling PoC's `ch10032_lower.vcloth2`,
so it is the *same surface* rather than a resampling of it. What differs is only the vertex
ordering -- gnn-poc's `source_vertex` runs 3732..5820, indexing a render or unwelded pattern
array rather than the 3404..5293 sim range, and its positions are metres with y down and a
recentring applied. Recovering the correspondence needs topological matching seeded by the
boundary loops, not nearest-neighbour. So the exclusion still stands, but as a cost decision
rather than a correctness one, and the evaluation does not need it: the axes are intra-mesh
(edge ratio, collapsed fraction, triangle area, flips), so the two front ends can be
compared without sharing a mesh.
