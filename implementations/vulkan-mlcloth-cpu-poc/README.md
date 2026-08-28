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

The three-way view, which is what most of this README is about:

```powershell
pwsh ./bake_cloth_topology.ps1 -ExportRoot data/AIClothTraining_10032_Test
pwsh ./bake_mlcloth_drivers.ps1
py -3 tools/bake_mlcloth_body.py
pwsh ./run.ps1 -Compare -XpbdAreaFloor 1.0 -XpbdIterationsB 11
```

The body bake reads the sibling GNN PoC's character export; without it the sample falls
back to drawing the collision capsules and says so.

## Geometry

Without `-Mesh` the sample renders a point cloud, exactly as it did before the
topology existed, so the verified coordinates and the benchmark numbers are
unaffected by anything below. With a `.mlmesh` it draws shaded triangles;
`-Points` forces the point path back on. Both paths get the backdrop and the
same tone curve; a point has no normal to light, so its colour is treated as
emitted radiance and gained up to stay clear of the horizon behind it.

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

The same measurement yields more than a boolean. The winning bone is the bone the
loop rides, and that bone's mean offset over the motion is a least-squares rigid
bind, so the bake also writes `pin_driver` and `pin_local_cm` into the mesh. Those
two sections are what lets a solver-only branch keep its anchor on the moving body
without consulting the network at all -- see *Three architectures side by side*.
They are optional: a bake with no reference clip has nothing to measure them from
and writes neither, and `parse_mesh` refuses a file carrying one without the other.

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

### Collision, and what gets drawn

`tools/bake_mlcloth_capsules.py` bakes the 14 capsules of the physics asset the
character actually references at runtime into `MLCAP001`. All 14 of its bones are
among the model's 45 drivers, so the `.mldrv` clip already carries their per-frame
transforms: collision needs no new per-frame data, no vertex proxy and no runtime
skinning. A capsule's signed distance is closed-form and unambiguous about inside,
which is a deliberate correction to the sibling PoC's nearest-proxy half-plane --
`vulkan-gnn-poc/results/PROGRESS.md` section 6 records that criterion reporting
28.5% penetration at 0.12 m and only 1% at 0.25 m, because past the far surface
its signed distance turns positive again.

The capsules are what the contacts resolve against and what the reported penetration
measures. They are **not** what gets drawn. A ragdoll envelope is several centimetres
wider than the skin, so judging a garment against it by eye is misleading in both
directions: cloth resting on the chest looks like it is floating, and cloth a
centimetre inside the skin looks fine. `--body mesh`, the default, draws the
character's own render mesh instead; `--body capsules` puts the envelope back, and the
overlay switches between them live.

### The character body

`tools/bake_mlcloth_body.py` produces `MLBDY001` from the sibling PoC's existing
character export -- 67,857 vertices, 128,988 triangles, rest normals and twelve bone
influences per vertex -- plus that character's `skeleton.json`. Nothing has to be
re-exported from Unreal, and the runtime's whole per-frame job is 45 matrices of
`pose * inverseBind` on the host followed by one compute dispatch.

Three things make it exact rather than approximate, and each is checked rather than
assumed:

**The spaces are the same up to a constant.** The character export's
`coordinate_mapping` is `x = body_x - 0.00705, y = body_z - 1.05863, z = -body_y +
0.02257` over a y-up metre space, and that axis permutation is exactly the one
`buildPointData` applies to reach world space. Inverting it lands in Unreal component
centimetres, which is where the driver clip's bone transforms already are. The check is
anatomical and unforgiving: inverted, the rest body must stand on z = 0 and be
symmetric about x = 0. It comes out at −0.035 cm and ±50.96 cm, 164.87 cm tall.

**The bind pose is recoverable exactly.** `skeleton.json` gives each of the 1,016 bones
a parent-local translation and rotator plus a component translation read off the mesh.
Composing the hierarchy reproduces the recorded translations to **1.2e-4 cm**, and that
is the check -- a real one: the first attempt read the rotator triple as (yaw, roll,
pitch) instead of (roll, pitch, yaw) and missed by 189 cm while still producing a
perfectly well-formed skeleton. The bake refuses above 1e-3 cm.

**41 of the character's 45 bones are model drivers.** The four that are not --
`Chest1_L`, `Chest1_R`, `Toes_L`, `Toes_R` -- each have a *driven direct parent*
(`Chest_M`, `Chest_M`, `Ankle_L`, `Ankle_R`), so their influences fold onto that parent
and merge with it. A bone whose parent is *also* undriven is refused rather than walked
further up the hierarchy.

That fold is the one approximation, and it is measured rather than waved at. Against
the sibling PoC's own per-frame skin matrices over five clips it moves the affected
vertices by up to **2.2 cm** on the breast patch (`Chest1_L/R`, 0.34% of all skin weight
each) and **5.2 cm** at the toes (3.4% each). The toes are 60 cm from the nearest garment
vertex and do not matter; the breast patch is under the bodice, so **chest clearance has
to be read off the numbers, not off the picture**. Adding those four bones to the
training export's driver list would remove the approximation entirely. The overlay says
so on screen whenever a body carries folded bones.

Merging also pays for itself: folding onto parents that were frequently already
influencing the same vertex takes this body from the source's twelve influences to
**five**, and that width is what the skinning loop costs. It is stored in the asset
rather than assumed.

#### Two checks that a picture cannot make

`-Verify` measures both, and fails on either:

| check | measured | gate |
| --- | --- | --- |
| compute skin vs the same arithmetic on the host | 3.6e-07 m | 1e-5 m |
| the garment to the nearest skin vertex, median | 2.32 cm | 0.2 .. 8 cm |

The second is the one that decides whether the body is worth drawing. The character
mesh, its skeleton and the driver clip come from three different exports; if the bind
pose, the coordinate mapping or the bone order were wrong the result would still be a
body, just not this character's body in this pose. Skinning it with a clip's own driver
transforms and measuring the distance from the garment for the same frame gives p05
0.4 cm and p50 2.2–2.7 cm across run, sprint, lunge and death poses -- and the
sibling PoC independently measured its own cloth-to-body distance at a 2.41 cm median. A
wrong chain does not produce a stable 2.4 cm across poses; it produces a body whose
height changes with the animation, which is exactly what the first attempt did (138 cm
on one frame, 188 cm on another) when it indexed the export's 1,016-bone T-pose file in
`skeleton.json`'s order. Those two orders agree for the first three bones and then
diverge.

The garment side of that measurement is the network branch's prediction, not ChaosCloth's
own output -- an earlier version of this section said otherwise. It matters in two ways.
It is the arm no constraint and no capsule touches, so the number does not move when the
solver is retuned, which is why every comparison run above reports the same 2.32 cm. And
what makes it evidence at all is that the network was trained without ever seeing the body
mesh, the skeleton or the bind pose, so the two sides of the comparison share nothing but
the character.

The first check is there because in this PoC a correct shader with a wrong dispatch
around it has now happened twice.

## Lighting, and a floor to stand on

Up to this point the scene was a garment and a body floating in a near-black field, lit
by one lambert term plus a vertical colour ramp standing in for ambient. With only the
cloth on screen that was enough; with a body in frame it stopped being readable at all --
a flat backdrop gives no cue for how high a hem sits, how far a limb has swung, or, in the
three-way view, whether the branches stand at the same height.

What is there now is a gradient sky over a gridded floor, and a three-light rig that
shares its constants with that sky:

| piece | what it is | why |
| --- | --- | --- |
| `sky.vert` / `sky.frag` | one fullscreen triangle, no vertex buffer, no depth | the backdrop, evaluated along the view ray |
| `scene_lighting.hlsli` | sky, three lights, floor, tone curve | shared by the backdrop and the surface shading so the two cannot drift |
| `scene_camera.hlsli` | the one declaration of the camera block | it grew an inverse view-projection and an eye position; four stages read it |

Three of those choices are load-bearing rather than cosmetic.

**The floor is at y = 0 because that is the character's floor.** The body's rest pose puts
its feet on component z = 0 (`body_validate` measures -0.035 cm) and world y is component
z in metres, so the grid is the plane the animation was authored against. That is what
turns it from decoration into a ruler: 0.25 m minor lines, 1 m major, analytically
antialiased so the far field fades out instead of aliasing.

**The backdrop writes no depth and tests none.** It is drawn first and covered by whatever
geometry follows, which means a hem or a foot that has gone *below* the floor stays
visible. A depth-writing ground plane would hide it -- and hiding the thing being judged
is worse than having no floor.

**Everything ends with the same tone curve.** The swapchain is `B8G8R8A8_UNORM`
(`VulkanSwapChain.cpp` prefers UNORM over any sRGB format), so nothing downstream encodes
for us: linear radiance written straight to an 8-bit surface is what made adding light not
help. Reinhard then gamma 2.2, in `tonemapAndEncode`, in every stage including the point
cloud's.

The tone curve compresses hard at this exposure, and that has one consequence worth
recording because it is not a matter of taste: a lit body and the sky behind it land within
a few percent of each other in *value* whatever either is set to. The separation that
survives is hue, which is why the sky is distinctly cool, the key light warm, and the body
a warm grey. A silhouette against the horizon reads on colour, not brightness.

Two diagnostics survive the rewrite unchanged, and constrain the shading. The back face
still gets a darker albedo, because a patch that speckles is inside out in places and that
is the `flipped_fraction` seen directly rather than as a scalar. And the branch tint still
*replaces* albedo rather than multiplying it -- no multiplier turns blue into orange. Both
are why the specular is a single weak white lobe: a strong highlight washes out a hue that
is carrying information.

`--no-sky` (`-NoSky`, or the overlay checkbox) drops the backdrop. The lighting is
identical either way, so what changes is only what a surface is being read against, which
is the honest way to settle whether a dark patch is shading or geometry.

### One more check a picture cannot make

The backdrop is the only thing on screen not drawn from a vertex buffer: it turns a pixel
back into a world ray through the inverse view-projection. A sign or convention error there
does not fail -- it draws the sky where the floor belongs, or puts the backdrop behind the
camera, and both look like a camera problem. So `-Verify` checks the reconstruction against
the forward transform the geometry uses, at the four screen corners and the centre:

| check | measured | gate |
| --- | --- | --- |
| re-projecting a point along the reconstructed ray returns its own pixel | 6.4e-05 NDC | 1e-3 NDC |
| the ray points into the scene (`dot` with the camera's forward) | 0.686 | > 0 |

The round trip is what catches a *convention* error rather than an arithmetic one: feeding
the NDC y with the wrong sign inverts it here while every length and angle stays plausible.
The limit is loose on purpose -- unprojecting at the far plane is badly conditioned, since a
hyperbolic depth range with near 0.01 and far 1000 puts nearly the whole NDC z range in the
first few centimetres. A convention error is off by order 1, three orders clear of the
gate, so tightening it would only produce a float-precision tripwire that fires when
someone changes the far plane.

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
| pure network | `0x543787fd06ca3c5d` |
| `--xpbd --xpbd-iterations 0` | `0x543787fd06ca3c5d` |
| `--xpbd --xpbd-iterations 1` | `0x5e7121cdf84be323` |
| `--xpbd --xpbd-iterations 8` | `0x4b2ea34202feab15` |

The hash is over the positions that reach the GPU -- every drawn branch's block -- not the
network's raw output. Hashing the raw output made the gate vacuous: it is unaffected by the
solver by construction, so k=0 and k=8 both reproduced it while k=8 was visibly moving
vertices 2.8 cm. Covering every block rather than the primary one matters for the same
reason once there are three: a branch the hash skipped could carry state across a reset
unnoticed. The hashes above are therefore comparable only between runs with the same branch
count, which is the caveat the upload size already carries.

Zero iterations skips the centimetre-to-metre round trip rather than performing and undoing
it, because `x * 0.01f * 100.0f` differs from `x` by an ulp on most values, which was by
itself enough to break the equality.

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

`--collision-pieces 2` is a hand value. The rule behind it is principled -- exclude a piece
whose *reference* solution already sits inside the capsules -- but the piece index is not
derived, it is typed. `bake_cloth_topology.py` already reads reference clips to measure pin
attachment, so the same pass could bake a per-vertex collision mask; until it does, moving to
another garment means measuring the pieces again by hand.

The solver's contact test cannot move a pinned vertex, which is correct -- pins are
kinematic -- and means a network that puts a *pinned* vertex inside the body is not something
anything downstream will fix.

## Three architectures side by side

`-Compare` draws the three arms the sibling PoC's `--hood-compare` draws, in the same
colours, so a screenshot from either reads the same way:

| | branch | inference | constraints | pin target |
| --- | --- | --- | --- | --- |
| blue | A network only | yes | none | network's own prediction |
| orange | B constraints only | **no** | yes | measured rigid bind to a driver bone |
| green | C hybrid | yes | yes | network's own prediction |

Everything else -- garment, topology, calibration, animation, frame, camera, capsules -- is
shared, so the only difference on screen is the solver. A and C share one inference by
definition, so the mode costs one extra solve for C and one for B.

**Branch B is the arm that has to be built carefully, because it is the one that could kill
the whole idea.** If constraints alone match the hybrid, the honest conclusion is to drop the
network. Two things make it a fair test rather than a straw man:

*Its prediction is ballistic, not zero.* Verlet extrapolation of its own last two frames plus
one step of gravity, with gravity rotated into the root's frame per frame rather than assumed
to be an axis of it -- in this asset the root's local **X** is what points up, so a hardcoded
`-Z` would have pushed the garment sideways. Its first frame is seeded from the network and
takes no gravity: the branch needs *some* initial configuration, the network's frame-0
prediction is the one the other two also start from, and `gate_g0.py::BallisticGravity`
reasons the same way about its settle step.

*Its pins follow the body.* A network-free branch has no prediction to take a pin target from,
and a static rest-pose anchor would make B lose for a reason that is not the solver. So
`bake_cloth_topology.py` now writes the bind it was already measuring: for each pinned
boundary loop it expresses every vertex in each of the 45 driver frames over reference motion
and keeps the frame with the least variance, which gives both the bone and -- as that frame's
mean position -- a least-squares rigid bind over the motion. On CH10032 the six pinned loops
come out as Chest_M (106, 104 and 110 vertices), Shoulder_R (20), ShoulderPart2_L (20) and
Spine1_M (72). The runtime places them with the same arithmetic `placeCapsules` uses, so the
anchor and the collision geometry cannot drift apart. `--compare` refuses to start against a
mesh without those sections rather than falling back.

That bind is also the one piece of this that a screenshot cannot validate, so `-Verify`
measures it: the network and the bind are independent descriptions of where the garment is
sewn to the body, so they have to agree. Over 15 clips spanning run, sprint, dodge, turn,
fall, attack and death, the worst single pinned vertex per clip lands between 6.3 and 14.5 cm
and the per-frame mean between 1.1 and 1.7 cm -- which is what a rigid approximation of smooth
skinning costs. Verification fails above 35 cm, well under the fraction-of-a-garment error a
bind taken in the wrong bone frame or with a guessed up axis would give.

*The budget is equal CPU, not equal iterations.* Measured here: C is 2.558 ms of MNN inference
plus 2.54 ms for 8 iterations; B runs at 0.34 ms per iteration and buys 15 of them for the
same 5.1 ms. Comparing at equal iterations would hand B a third of C's compute and then report
the result as an architecture difference. Both numbers are in `mlcloth_verify.json` per branch,
so the default is re-derivable rather than folklore.

That default is a *measurement in one configuration*, and it does not transfer. Turning the
area floor on costs the solver about 40% per iteration, which lands on B fifteen times and on C
eight, so the same flag that is supposed to equalise them pulls them apart: B ends up costing
1.31x C. So the achieved ratio is now computed and shown -- `compare_budget_ratio` in the
report, and a line in the overlay that says `<-- retune B iterations` outside 0.85..1.15. It is
reported rather than corrected automatically, because retuning per frame would make the picture
wobble as the machine's load changes. At `-XpbdAreaFloor 1.0` the equal-budget count on this
machine is `-XpbdIterationsB 11`, which reads 1.04.

One frame of one clip, skirt piece, so read it as a smoke test and not as the result. The
solver defaults, area floor off:

| branch | solve ms | whole garment inside | skirt inside |
| --- | --- | --- | --- |
| A network only | 0 | 71.5% | 27.2% |
| B constraints only, k=15 | 4.87 | 67.4% | 11.6% |
| C hybrid, k=8 | 2.53 | 67.0% | **9.9%** |

And with the area floor on at its equal-budget count, which is the configuration the command at
the top of this README asks for:

| branch | solve ms | skirt inside |
| --- | --- | --- |
| B constraints only, k=11 | 5.35 | 10.3% |
| C hybrid, k=8 | 3.75 | **10.0%** |

The whole-garment column is there to be ignored, which is why it is shown: the capsules are a
ragdoll envelope and two of the four pieces are *supposed* to sit inside it. Only the skirt's
number is one a solver should be reducing. Branch B is also one step from its network seed at
frame 1 and has not yet diverged, so its column says nothing yet about a network-free rollout
-- that is what the side-by-side view and, later, `results/MLCLOTH_S13_RESULTS.md` are for.

The two tables also disagree by less than the gap between B and C in either, which is the more
useful reading: at one frame, at equal budget, the hybrid is ahead of constraints-alone on the
one piece where the number means anything, and the area floor does not change that verdict.

### What the comparison mode had to fix first

The two-branch version of this shipped broken, and the way it broke is worth recording: both
compute passes dispatched `(kVertexCount + 127) / 128` workgroups -- one branch's worth --
while the buffers, the index list and the uniforms all covered every branch. The shaders were
already correct, because each bounds-checks against a count from its own uniform, so nothing
failed loudly. The second garment was simply drawn from whatever was in device-local memory.
The GPU coordinate check did not catch it either, because it compared only the first block.
Both are fixed, and the check now covers every block, which is what makes the max error of
3.6e-07 m mean what it says.

Three sliders in the XPBD panel were also inert: area floor, guide compliance and guide trust
ratio wrote the runtime's copy of the configuration, and the solver reads the copy that only
`configure` installs. Dragging them did nothing.

And `run.ps1 -Compare` forwarded none of the `--xpbd-*` knobs: the solver is switched on inside
the exe by `--compare`, but the block that passes its configuration was gated on `-Xpbd` alone.
So `-Compare -XpbdAreaFloor 1.0` -- the command this README recommends -- ran with the exe's
default of 0, and the first table above is what that produced. Nothing could have revealed it,
because the report carried the iteration counts and not the compliances. It now carries every
knob the solver was configured with, which is the part of this fix that generalises.

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

The 5,294-to-1,377 mapping, self-collision, textured materials, shadows, MNN GPU execution,
and the XPBD compute-shader port.

Shadows are worth a word, since the lighting is otherwise complete: the rig has none, and a
contact shadow is exactly what would make a hem's height read best. It is left out because
the cheap versions are all a form of invented contrast, and this scene is used to judge
whether cloth is touching a body -- a blob under the feet or a screen-space occlusion term
would put a dark band at precisely the contact being assessed.

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
