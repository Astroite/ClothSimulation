"""Deterministic XPBD projection for the distilled-model (HOOD/TinyHOOD) path.

Why this module exists
---------------------
The distilled path has no constraint solver at all: `hood_integrate.comp` writes a Verlet-style
position update and a single nearest-proxy half-plane projection, and the debug dump records
`"xpbd": false`. The only XPBD in the repository lives on the toy VGNN grid path and is bound
to a structured grid in every respect -- analytic edge colouring (`x & 1`), rest lengths from
three UBO scalars, a single global `particleMass`. None of it transfers.

This is the Python half of `plans/gnn/gnn-xpbd-v2.md` gate G0: enough of an unstructured-mesh
XPBD to answer "is the hybrid worth building" before any Vulkan work. It is a reference
implementation, not a port target, but the data layout is chosen to match what a GPU kernel
would want.

Two design decisions carry most of the weight:

**Rest lengths must be calibrated, not read from `cloth_rest`.** Skinning the authored rest mesh
into the scene's frame 0 already stretches edges well past rest: measured p95 / max edge ratio is
1.890 / 12.11 on `ch10032_tpose`, 2.025 / 16.31 on `ch10032_sprint`, 1.081 / 5.27 on
`hml_001962`, and exactly 1.000 / 1.00 on the synthetic `hood_grid64`. On the CH10032 scenes
7.7-8.2% of edges start above 1.5x. A distance constraint aimed at `cloth_rest` would contract
those edges by up to 12x and produce a garment far stiffer than the teacher -- the same failure
`edge_penalty` in `tools/train_student.py` already documents on the training side. So the target
length is a parameter (`build_constraints(reference_position=...)`), and edges whose calibration
is untrustworthy are flagged rather than silently included.

**Accumulation is deterministic.** `index_add_` has no deterministic float CUDA kernel, and it is
already the reason the existing selection metric carries +/-0.035 of irreproducible noise on a
~0.40 score. Putting it inside the solver would add solver noise on top of metric noise. Instead
each vertex owns a fixed-width, fixed-order list of the constraints touching it (measured max
mesh-edge degree is 9, so the table is small) and corrections are gathered and summed in that
order. Bit-identical on CPU and CUDA, and it is also the structure a compute shader would use.

What the three modes mean
-------------------------
The modes differ only in where the solve starts -- same constraints, same sweeps -- so a
difference in the result cannot be blamed on a different solver:

* `standard` -- start from x_gnn. The network's displacement is kept in full and acts as a new
  inertial prediction. This is scheme A.
* `warmstart` -- convert the displacement into multipliers first and start from
  `x_tilde + M^-1 J^T lambda_0`. Only the part expressible as a constraint force survives. This is
  scheme C, and it is also scheme B with the multiplier head replaced by an analytic projection, so
  it says whether predicting lambda is worth training before anything is trained.
* `nowarm` -- start from x_tilde and drop the displacement. Bounds what the position prediction is
  worth at all.

Because `M^-1 g = x - x_tilde` and the decoder output *is* that displacement, none of this needs a
reconstruction pass -- which is the concrete form of the plan's claim that the primal residual is
free on this path.

A consequence worth stating before reading any result: `warmstart` collapsing towards `nowarm` is
not a bug. A displacement in the null space of J -- a rigid translation, say -- projects to zero
multipliers and is discarded by construction. Distinguishing "the network had nothing a constraint
force could express" from "the solver threw away something useful" is what gate G0 is for.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Matches COLLISION_RADIUS in real_scene/fine15.py, which is also the radius the runtime's
# nearest-proxy search uses. Kept as a separate name because the solver's contact offset and the
# graph's neighbour-search radius are conceptually different knobs that happen to agree today.
DEFAULT_CONTACT_OFFSET = 0.005

# The radius `contacts_from_search` accepts a proxy within. This has to stay equal to
# real_scene/fine15.py::COLLISION_RADIUS -- a substepped solve searches for its own contacts instead
# of reading the network's graph, and if the two radii diverged the first substep would see a
# different contact set than the network did. Duplicated rather than imported so this module keeps
# taking its graph duck-typed; tests/test_xpbd.py asserts the two agree.
DEFAULT_SEARCH_RADIUS = 0.03

STRETCH = 0
BEND = 1


@dataclass(frozen=True)
class ConstraintSet:
    """Distance constraints plus the per-vertex gather tables needed to apply them.

    `slots` and `signs` are the deterministic accumulation structure: row `v` lists the
    constraints touching vertex `v`, padded to a fixed width with the sentinel index
    `count` (one past the last real constraint), whose gradient and multiplier rows are
    kept zero so padded lanes contribute nothing.
    """

    pairs: torch.Tensor          # [C, 2] long
    target_length: torch.Tensor  # [C] float -- calibrated, not necessarily the rest length
    kind: torch.Tensor           # [C] long -- STRETCH or BEND
    suspect: torch.Tensor        # [C] bool -- calibration outside the trusted band
    slots: torch.Tensor          # [V, K] long -- constraint index, padded with `count`
    signs: torch.Tensor          # [V, K] float -- +1 for pairs[:, 0], -1 for pairs[:, 1], 0 pad
    incident: torch.Tensor       # [V, 1] float -- real constraints per vertex, for Jacobi averaging
    inverse_mass: torch.Tensor   # [V, 1] float -- 0 on pinned vertices
    colour: torch.Tensor         # [C] long -- vertex-disjoint groups for Gauss-Seidel

    @property
    def count(self) -> int:
        return int(self.pairs.shape[0])

    @property
    def vertex_count(self) -> int:
        return int(self.slots.shape[0])

    @property
    def colour_count(self) -> int:
        return 0 if self.count == 0 else int(self.colour.max().item()) + 1

    def colour_groups(self) -> list[torch.Tensor]:
        """Constraint indices per colour, in ascending colour order."""
        return [torch.nonzero(self.colour == c, as_tuple=False).reshape(-1) for c in range(self.colour_count)]


def undirected_edges(senders: torch.Tensor, receivers: torch.Tensor) -> torch.Tensor:
    """Deduplicate the scene's directed CSR neighbour lists into undirected pairs.

    `RuntimeScene` stores each mesh edge twice (once per direction); keeping only sender <
    receiver halves it. On CH10032 that is 7894 -> 3947, which equals the number of unique
    triangle edges, so the CSR really is the triangle-edge graph and no edge is missed.
    """
    keep = senders < receivers
    return torch.stack((senders[keep], receivers[keep]), dim=-1)


def bend_pairs(triangles: torch.Tensor) -> torch.Tensor:
    """Two-hop distance constraints across each interior edge, one per adjacent triangle pair.

    This is the same cheap stand-in for bending stiffness the grid path uses (`colors[8..15]`
    hold two-hop horizontal/vertical distance constraints) rather than a dihedral-angle
    constraint. Measured on CH10032: 2570 triangles, 3947 unique edges, 3763 of them shared by
    exactly two triangles, 184 boundary, no non-manifold edges -- so 3763 bend constraints.
    """
    corners = triangles.tolist()
    owners: dict[tuple[int, int], list[int]] = {}
    for index, (i, j, k) in enumerate(corners):
        for a, b in ((i, j), (j, k), (k, i)):
            owners.setdefault((min(a, b), max(a, b)), []).append(index)
    pairs: list[tuple[int, int]] = []
    for (a, b), faces in sorted(owners.items()):
        if len(faces) != 2:
            continue
        left, right = (set(corners[face]) - {a, b} for face in faces)
        if len(left) != 1 or len(right) != 1:
            continue
        first, second = left.pop(), right.pop()
        if first == second:
            continue
        pairs.append((min(first, second), max(first, second)))
    if not pairs:
        return torch.zeros((0, 2), dtype=triangles.dtype, device=triangles.device)
    # Sorted and deduplicated so the constraint order is a property of the mesh, not of dict
    # iteration -- the whole solve is supposed to be reproducible.
    unique = sorted(set(pairs))
    return torch.tensor(unique, dtype=triangles.dtype, device=triangles.device)


def greedy_colouring(pairs: torch.Tensor, vertex_count: int) -> torch.Tensor:
    """Assign each constraint a colour such that no two constraints in a colour share a vertex.

    This is the unstructured-mesh replacement for the grid path's analytic `colorOf()`
    (`x & 1`, `8 + (x & 3)`, ...), and it is what makes Gauss-Seidel possible: within a colour the
    constraints are vertex-disjoint, so both endpoints can be written in place with no atomics and
    no ordering dependence. Jacobi needs no colouring but converges far too slowly to stand in for
    a real solver here -- measured on CH10032, 128 Jacobi sweeps only cut the stretch residual 3.2x.

    Constraints are visited in index order, which is a property of the mesh, so the colouring is
    reproducible. The resulting colour count is the quantity `plans/gnn/gnn-xpbd-v2.md` section 2.3
    flags as an estimate: it drives the per-iteration dispatch count on the GPU.
    """
    used: list[set[int]] = [set() for _ in range(vertex_count)]
    colours: list[int] = []
    for first, second in pairs.tolist():
        blocked = used[first] | used[second]
        colour = 0
        while colour in blocked:
            colour += 1
        colours.append(colour)
        used[first].add(colour)
        used[second].add(colour)
    return torch.tensor(colours, dtype=torch.long, device=pairs.device)


def _gather_tables(pairs: torch.Tensor, vertex_count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the padded per-vertex constraint tables.

    Every step here is either a stable sort or a scatter to distinct destinations, so the
    resulting tables -- and therefore the accumulation order they impose -- do not depend on
    execution order.
    """
    device = pairs.device
    count = int(pairs.shape[0])
    if count == 0:
        empty = torch.full((vertex_count, 1), 0, dtype=torch.long, device=device)
        return empty, torch.zeros((vertex_count, 1), device=device), torch.zeros((vertex_count, 1), device=device)

    flat_vertex = torch.cat((pairs[:, 0], pairs[:, 1]))
    flat_constraint = torch.cat((torch.arange(count, device=device), torch.arange(count, device=device)))
    flat_sign = torch.cat((torch.ones(count, device=device), -torch.ones(count, device=device)))

    order = torch.argsort(flat_vertex, stable=True)
    sorted_vertex = flat_vertex[order]
    incident = torch.bincount(sorted_vertex, minlength=vertex_count)
    width = int(incident.max().item())

    # Rank of each entry inside its vertex's run, so (vertex, rank) is a unique slot.
    offsets = torch.zeros(vertex_count + 1, dtype=torch.long, device=device)
    offsets[1:] = torch.cumsum(incident, dim=0)
    rank = torch.arange(sorted_vertex.shape[0], device=device) - offsets[sorted_vertex]

    slots = torch.full((vertex_count, width), count, dtype=torch.long, device=device)
    signs = torch.zeros((vertex_count, width), device=device)
    slots[sorted_vertex, rank] = flat_constraint[order]
    signs[sorted_vertex, rank] = flat_sign[order]
    return slots, signs, incident.to(signs.dtype).reshape(-1, 1)


def build_constraints(
    scene,
    reference_position: torch.Tensor,
    *,
    include_bend: bool = True,
    suspect_above: float = 3.0,
    mass: torch.Tensor | None = None,
    pin_mask: torch.Tensor | None = None,
) -> ConstraintSet:
    """Assemble stretch and bend constraints with target lengths taken from `reference_position`.

    `reference_position` is the configuration the solver treats as unstretched. Passing
    `scene.cloth_rest` reproduces a textbook rest-length solver and is kept available as a
    control, but see the module docstring for why it is the wrong choice on the CH10032 scenes.
    `suspect_above` flags constraints whose reference length differs from the authored rest
    length by more than this factor -- they are still solved, but the count is reported so a
    result is never read as if the whole mesh were trustworthy.
    """
    stretch = undirected_edges(scene.cloth_senders, scene.cloth_receivers)
    pieces = [stretch]
    kinds = [torch.full((stretch.shape[0],), STRETCH, dtype=torch.long, device=stretch.device)]
    if include_bend:
        bend = bend_pairs(scene.cloth_triangles)
        pieces.append(bend)
        kinds.append(torch.full((bend.shape[0],), BEND, dtype=torch.long, device=bend.device))
    pairs = torch.cat(pieces, dim=0)
    kind = torch.cat(kinds, dim=0)

    reference = reference_position.to(dtype=torch.float32)
    target = torch.linalg.vector_norm(reference[pairs[:, 0]] - reference[pairs[:, 1]], dim=-1)
    authored = torch.linalg.vector_norm(
        scene.cloth_rest[pairs[:, 0]] - scene.cloth_rest[pairs[:, 1]], dim=-1
    ).clamp_min(1.0e-12)
    ratio = target / authored
    suspect = (ratio > suspect_above) | (ratio < 1.0 / suspect_above)

    vertex_count = int(scene.cloth_rest.shape[0])
    slots, signs, incident = _gather_tables(pairs, vertex_count)

    mass = (scene.cloth_mass if mass is None else mass).reshape(-1, 1).to(dtype=torch.float32)
    pin_mask = (scene.cloth_pins if pin_mask is None else pin_mask).reshape(-1)
    inverse_mass = torch.where(
        pin_mask.reshape(-1, 1), torch.zeros_like(mass), 1.0 / mass.clamp_min(1.0e-12)
    )
    return ConstraintSet(
        pairs=pairs,
        target_length=target.clamp_min(1.0e-9),
        kind=kind,
        suspect=suspect,
        slots=slots,
        signs=signs,
        incident=incident,
        inverse_mass=inverse_mass,
        colour=greedy_colouring(pairs, vertex_count),
    )


def calibrate_from_trajectory(pairs: torch.Tensor, positions: list[torch.Tensor], skip: int = 0) -> torch.Tensor:
    """Median edge length over a trajectory, for target lengths taken from the teacher's own solve.

    The teacher is what the student imitates, so its settled edge lengths are the most defensible
    definition of "unstretched" for this comparison. `skip` drops the settling transient; the
    reference runtime's first step is a 1/3 s settle, so the first few steps are not steady state.
    """
    if not positions:
        raise ValueError("calibrate_from_trajectory needs at least one position frame")
    frames = positions[skip:] or positions
    lengths = torch.stack([
        torch.linalg.vector_norm(frame[pairs[:, 0]] - frame[pairs[:, 1]], dim=-1) for frame in frames
    ])
    return lengths.median(dim=0).values


def triangle_areas(position: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    """Area of each triangle, formed in the mesh's stored winding order.

    The winding order is load-bearing, not cosmetic. In the fused sweep all three vertices of a
    triangle recompute the same multiplier update, and they only agree bit for bit if they all form
    `cross(b - a, c - a)` with a, b, c in the stored order rather than "mine first". Same rule as the
    pair sweep's `pairs[c, 0] - pairs[c, 1]`.
    """
    corners = position[triangles]
    normal = torch.linalg.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    return 0.5 * torch.linalg.vector_norm(normal, dim=-1)


def calibrate_area_from_trajectory(
    triangles: torch.Tensor, positions: list[torch.Tensor], skip: int = 0
) -> torch.Tensor:
    """Median triangle area over a trajectory. The area analogue of `calibrate_from_trajectory`.

    Measured, and it is worth recording because the obvious prediction was wrong. Reference *lengths*
    cannot come from `cloth_rest` -- skinning the authored mesh into frame 0 puts the CH10032 edge
    ratio p95 at 1.890 / 2.025 -- so the expectation was that reference *areas* would be off by
    roughly the square of that. They are not. On `ch10032_lower` the calibrated area over the rest
    area runs 0.744 to 1.171, p5 to p95 of 0.929 to 1.051, with nothing outside [0.5, 2.0].

    The reason is that skinning stretches a triangle anisotropically: one edge goes long while
    another goes short, and the area barely moves. A per-edge upper tail of 1.89 is compatible with an
    area within 5%.

    So a rest-area floor would in fact have been usable on this garment, and the honest statement is
    that this function is the right default for two weaker reasons rather than one strong one: it
    costs nothing on top of the length calibration (same rollout -- see
    `tools/bake_xpbd_constraints.py::teacher_calibration`), and it does not assume the anisotropic
    cancellation holds on the next garment, which is untested at N=1. `build_area_constraints` keeps
    `reference_position` available so the rest-area version stays a one-line control.
    """
    if not positions:
        raise ValueError("calibrate_area_from_trajectory needs at least one position frame")
    frames = positions[skip:] or positions
    areas = torch.stack([triangle_areas(frame, triangles) for frame in frames])
    return areas.median(dim=0).values


@dataclass(frozen=True)
class AreaConstraints:
    """Per-triangle area floors plus the per-vertex gather tables needed to apply them.

    Separate from `ConstraintSet` rather than folded into it because an area constraint is ternary and
    the pair tables are binary: `signs` picks an endpoint, `corner` picks one of three gradients. Also
    separate so that `area_floor = 0` leaves the pair sweep untouched -- with the area pass skipped the
    solve is byte-for-byte the one gate G0 measured.

    There is deliberately no baked `weight_sum` here, unlike `SolverTables`. That trick exists because
    `w_a + w_b` and `w_b + w_a` are not guaranteed to give the same float on two different threads, and
    for a distance constraint the sum is state-independent so it can be baked once. An area
    constraint's denominator is `sum_i w_i |grad_i A|^2`, which changes every iteration and cannot be
    baked at all. Agreement between the three threads is bought a different way: they sum the three
    terms in corner order 0, 1, 2 -- an order fixed by the mesh, not by which vertex is asking.
    """

    triangles: torch.Tensor     # [T, 3] long -- stored winding order
    target_area: torch.Tensor   # [T] float -- calibrated; see calibrate_area_from_trajectory
    slots: torch.Tensor         # [V, K] long -- incident triangle index, padded with `count`
    corner: torch.Tensor        # [V, K] long -- 0/1/2, which corner this vertex is; 0 in the padding
    incident: torch.Tensor      # [V, 1] float -- real incident triangles, for Jacobi averaging

    @property
    def count(self) -> int:
        return int(self.triangles.shape[0])

    @property
    def vertex_count(self) -> int:
        return int(self.slots.shape[0])


def _gather_triangle_tables(
    triangles: torch.Tensor, vertex_count: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`_gather_tables` for triangles: three columns, and the corner index replaces the sign."""
    device = triangles.device
    count = int(triangles.shape[0])
    if count == 0:
        empty_slot = torch.zeros((vertex_count, 1), dtype=torch.long, device=device)
        return empty_slot, empty_slot.clone(), torch.zeros((vertex_count, 1), device=device)

    flat_vertex = torch.cat((triangles[:, 0], triangles[:, 1], triangles[:, 2]))
    index = torch.arange(count, device=device)
    flat_triangle = torch.cat((index, index, index))
    flat_corner = torch.cat((
        torch.zeros(count, dtype=torch.long, device=device),
        torch.ones(count, dtype=torch.long, device=device),
        torch.full((count,), 2, dtype=torch.long, device=device),
    ))

    order = torch.argsort(flat_vertex, stable=True)
    sorted_vertex = flat_vertex[order]
    incident = torch.bincount(sorted_vertex, minlength=vertex_count)
    width = max(int(incident.max().item()), 1)

    offsets = torch.zeros(vertex_count + 1, dtype=torch.long, device=device)
    offsets[1:] = torch.cumsum(incident, dim=0)
    rank = torch.arange(sorted_vertex.shape[0], device=device) - offsets[sorted_vertex]

    slots = torch.full((vertex_count, width), count, dtype=torch.long, device=device)
    corner = torch.zeros((vertex_count, width), dtype=torch.long, device=device)
    slots[sorted_vertex, rank] = flat_triangle[order]
    corner[sorted_vertex, rank] = flat_corner[order]
    return slots, corner, incident.to(torch.float32).reshape(-1, 1)


def build_area_constraints(
    scene, target_area: torch.Tensor | None = None, *, reference_position: torch.Tensor | None = None
) -> AreaConstraints:
    """Assemble the area floor tables. Supply `target_area` from `calibrate_area_from_trajectory`.

    `reference_position` is the control path: passing `scene.cloth_rest` gives a textbook rest-area
    floor, which on `ch10032_lower` turns out to be within 5% of the calibrated one at the median (see
    `calibrate_area_from_trajectory` for the measurement and why it is still not the default). One of
    the two arguments must be given.
    """
    triangles = scene.cloth_triangles
    if target_area is None:
        if reference_position is None:
            raise ValueError("build_area_constraints needs either target_area or reference_position")
        target_area = triangle_areas(reference_position.to(dtype=torch.float32), triangles)
    vertex_count = int(scene.cloth_rest.shape[0])
    slots, corner, incident = _gather_triangle_tables(triangles, vertex_count)
    return AreaConstraints(
        triangles=triangles,
        target_area=target_area.to(dtype=torch.float32).clamp_min(0.0),
        slots=slots,
        corner=corner,
        incident=incident,
    )


@dataclass
class SolverConfig:
    """One point in the gate's configuration matrix."""

    iterations: int = 4
    mode: str = "standard"                 # "standard" | "warmstart" | "nowarm" | "guide"
    sweep: str = "coloured"                # "coloured" (Gauss-Seidel) | "jacobi" | "fused"
    stretch_compliance: float = 0.0        # metres^2 / newton; 0 = rigid
    bend_compliance: float = 1.0e-5
    one_sided: bool = False                # resist stretch only, never compression
    relaxation: float = 1.0                # Jacobi under/over-relaxation
    warmstart_scale: float = 1.0           # scales lambda_0 in "warmstart" mode
    collision: bool = True
    contact_offset: float = DEFAULT_CONTACT_OFFSET
    # `mode="guide"` only. Compliance of the per-vertex pull towards the network's prediction, in the
    # same metres^2/newton as the structural compliances. Magnitude matters more than it looks: gate
    # G0 swept stretch compliance over 0..1e-1 and found the entire range inert, because alpha-tilde
    # = compliance / dt^2 only reaches the inverse-mass sum at compliance around 15 for a two-endpoint
    # constraint. The guide is a *one*-vertex constraint, so its denominator is w rather than
    # w_a + w_b and the crossover sits near half that. Anything below ~0.5 is a hard guide in
    # disguise; the useful band is measured, not assumed. See `_resolve_guide`.
    guide_compliance: float = 0.0
    # Trust region on the guide, as a multiple of the vertex's shortest constraint. 0 disables it.
    # A vertex whose guide sits further away than `guide_trust_ratio * min_edge` has its confidence
    # scaled down, which raises its alpha-tilde and lets XPBD win locally. See `guide_confidence`.
    guide_trust_ratio: float = 0.0
    # Triangle area floor, as a fraction of the *calibrated* area. 0 disables it. The calibrated area
    # is not the rest area -- see `calibrate_area_from_trajectory`.
    area_floor: float = 0.0
    area_compliance: float = 0.0

    def compliance_per_constraint(self, kind: torch.Tensor) -> torch.Tensor:
        return torch.where(
            kind == BEND,
            torch.full_like(kind, 0.0, dtype=torch.float32) + self.bend_compliance,
            torch.full_like(kind, 0.0, dtype=torch.float32) + self.stretch_compliance,
        )


def load_vxpbd(path, *, device=None) -> ConstraintSet:
    """Rebuild a `ConstraintSet` from a baked `.vxpbd`, the way hood_xpbd.comp reads it.

    Used by `tools/run_tinyhood_reference.py` so the Python golden and the Vulkan run drive the
    same constraint data down to the last bit. Recomputing the calibration on both sides instead
    would inject a difference of its own: the teacher rollout the targets are measured on is not
    reproducible (`index_add_` has no deterministic CUDA float kernel), and the measured spread in
    the stretch target p95 across rollouts is around 1e-5 m. A comparison meant to detect a wrong
    kernel must not carry that.

    Two fields the asset does not store, because the kernel does not use them:

    * `suspect` -- a bake-time diagnostic only, so it comes back all false;
    * `colour` -- the runtime sweeps Jacobi, so no colouring is baked. It is filled with one colour
      per constraint, which makes `sweep="coloured"` degrade to fully sequential Gauss-Seidel:
      slow, but correct. A single shared colour would have been silently wrong.

    The area floor's `area_target` section lives in the same file but is not part of a
    `ConstraintSet` -- see `load_vxpbd_area_target`, which is optional so assets baked before the
    area constraint existed still load.
    """
    from .formats import load_sectioned

    asset = load_sectioned(path, expected_magic=b"VXPBD001", expected_version=1)

    def read(name: str, dtype: torch.dtype) -> torch.Tensor:
        view = asset.require(name)
        return torch.frombuffer(bytearray(view.data), dtype=dtype).to(device=device)

    count, vertices, width, _stretch = read("info", torch.int32).tolist()
    return ConstraintSet(
        pairs=read("pairs", torch.int32).reshape(count, 2).to(torch.long),
        target_length=read("target_len", torch.float32),
        kind=read("kind", torch.int32).to(torch.long),
        suspect=torch.zeros(count, dtype=torch.bool, device=device),
        slots=read("slots", torch.int32).reshape(vertices, width).to(torch.long),
        signs=read("signs", torch.float32).reshape(vertices, width),
        incident=read("incident", torch.float32).reshape(vertices, 1),
        inverse_mass=read("inverse_mass", torch.float32).reshape(vertices, 1),
        colour=torch.arange(count, dtype=torch.long, device=device),
    )


def load_vxpbd_section(path, name: str, *, device=None) -> torch.Tensor | None:
    """Read one optional float section out of a `.vxpbd`, or `None` if the asset predates it.

    `area_target` and `min_edge` are both in this category: the sectioned format looks sections up by
    name, so adding one does not invalidate an older reader, but a newer reader must not require one
    from an older file. Returning `None` lets the caller fall back rather than fail on an asset that
    is otherwise perfectly current.
    """
    from .formats import load_sectioned

    asset = load_sectioned(path, expected_magic=b"VXPBD001", expected_version=1)
    if name not in asset.sections:
        return None
    return torch.frombuffer(bytearray(asset.require(name).data), dtype=torch.float32).to(device=device)


@dataclass(frozen=True)
class SolverTables:
    """Per-constraint scalars that depend on the configuration but not on the state.

    These are exactly what the Vulkan `.vxpbd` asset has to carry, which is why they are a named
    type rather than four locals inside `project`. `weight_sum` in particular *must* be baked and
    read, not recomputed in the kernel: the fused sweep has both endpoints of a constraint evaluate
    the same multiplier update, and `w_a + w_b` on one thread against `w_b + w_a` on the other is
    not guaranteed to give the same float. A stored value removes the question.
    """

    weight_sum: torch.Tensor   # [C] float -- w_a + w_b, summed once here
    alpha: torch.Tensor        # [C] float -- compliance / dt^2
    denominator: torch.Tensor  # [C] float -- weight_sum + alpha, clamped away from zero
    alive: torch.Tensor        # [C] bool  -- false when both endpoints are pinned


def bake_tables(constraints: ConstraintSet, config: SolverConfig, timestep: float) -> SolverTables:
    """Everything a sweep needs beyond the state, computed once."""
    first, second = constraints.pairs[:, 0], constraints.pairs[:, 1]
    weight_sum = constraints.inverse_mass[first].reshape(-1) + constraints.inverse_mass[second].reshape(-1)
    alpha = config.compliance_per_constraint(constraints.kind) / max(timestep * timestep, 1.0e-12)
    return SolverTables(
        weight_sum=weight_sum,
        alpha=alpha,
        denominator=(weight_sum + alpha).clamp_min(1.0e-20),
        alive=weight_sum > 0.0,
    )


@dataclass(frozen=True)
class Contacts:
    """Per-vertex one-sided plane contacts, as produced by the runtime's nearest-proxy search."""

    vertex: torch.Tensor  # [N] long
    point: torch.Tensor   # [N, 3] float -- a point on the plane
    normal: torch.Tensor  # [N, 3] float -- unit, pointing out of the body


def contacts_from_graph(graph, obstacle_target: torch.Tensor, obstacle_normals: torch.Tensor) -> Contacts:
    """Reuse `Fine15.prepare_graph`'s nearest-proxy pairing as the contact set.

    `prepare_graph` already ran the same `cdist`/`min` search the runtime's
    `hood_world_nearest.comp` performs, and stored the result as world edges. Reading it back
    costs nothing and guarantees the solver sees exactly the contacts the network saw.
    """
    proxy = graph.active_obstacle[graph.world_obstacle]
    normal = torch.nn.functional.normalize(obstacle_normals[proxy], dim=-1)
    return Contacts(vertex=graph.world_cloth, point=obstacle_target[proxy], normal=normal)


def contacts_from_search(
    position: torch.Tensor,
    proxy_position: torch.Tensor,
    proxy_target: torch.Tensor,
    proxy_normals: torch.Tensor,
    *,
    radius: float = DEFAULT_SEARCH_RADIUS,
) -> Contacts:
    """Nearest-proxy contact set searched fresh, for substeps that have no graph of their own.

    `contacts_from_graph` reuses the pairing the network's graph already made, which is exactly right
    when the solver runs once per network step -- the constraints then see the contacts the network
    saw. A substepped solve cannot use it. The network runs once per visual frame, so after a substep
    has moved a vertex the nearest body surface may be a different proxy entirely, and every
    remaining substep would be converging accurately against an expired half-plane. That is the
    concrete form of "the collision constraint is stale for the whole solve": adding iterations
    cannot fix a plane that is in the wrong place.

    The search is `real_scene/fine15.py::_world_edges` verbatim -- `cdist`, `min` over dim 1, strict
    `< radius`, lowest index winning ties through `min`'s own tie rule -- so at one substep this
    reproduces the pairing that function produced.

    `proxy_position` is the body at the start of the substep (what the vertex is near) and
    `proxy_target` the body at its end (where the plane goes). That lead is not an accident: it is the
    same asymmetry `make_hook` and `hood_world_nearest.comp` already use, and it is what stops the
    contact set from lagging the motion by a frame.
    """
    distance = torch.cdist(position, proxy_position)
    minimum, nearest = distance.min(dim=1)
    valid = minimum < radius
    vertex = torch.arange(position.shape[0], device=position.device, dtype=torch.long)[valid]
    proxy = nearest[valid]
    return Contacts(
        vertex=vertex,
        point=proxy_target[proxy],
        normal=torch.nn.functional.normalize(proxy_normals[proxy], dim=-1),
    )


def inertial_prediction(graph) -> torch.Tensor:
    """x_tilde, the prediction the network's output is a displacement from.

    `integrate()` in tools/train_student.py forms
    `effective_position + (effective_position - effective_previous + acceleration)`, so the
    network-free part is `2 * effective_position - effective_previous` and `acceleration` is
    exactly `x_gnn - x_tilde`. That is what makes the primal residual free here: M^-1 g is the
    decoder output, with no reconstruction pass.
    """
    return 2.0 * graph.effective_position - graph.effective_previous


def project(
    constraints: ConstraintSet,
    config: SolverConfig,
    *,
    position: torch.Tensor,
    inertial: torch.Tensor,
    pin_mask: torch.Tensor,
    pin_target: torch.Tensor,
    timestep: float,
    contacts: Contacts | None = None,
    guide: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
    area: AreaConstraints | None = None,
) -> torch.Tensor:
    """Run `config.iterations` deterministic XPBD sweeps and return the corrected position.

    Pinned vertices are placed on their target before the first sweep, not after the last. They
    carry zero inverse mass so they never move again, which makes them infinite-mass anchors the
    solve can propagate from. Snapping them at the end instead would leave the pin violation
    entirely unsolved and inject it fresh every step -- measured, that alone drove edge P95 to 28
    over 137 steps. `integrate()` already returns pins on target, so in the normal pipeline this is
    a no-op; it is here so the function cannot be misused.

    Three sweep schedules:

    * `coloured` -- Gauss-Seidel over vertex-disjoint colour groups. Within a colour both endpoints
      are written in place; the indices are distinct by construction so the writes are
      deterministic. Converges fastest per iteration, but needs one GPU dispatch per colour and
      the measured colour count on CH10032 is 18, so it loses badly at equal cost.
    * `jacobi` -- every constraint reads the same `x` and per-vertex corrections are averaged over
      the constraints touching that vertex. Needs no colouring, but converges much more slowly:
      128 sweeps cut the CH10032 stretch residual only 3.2x, versus a few sweeps coloured. This is
      the schedule gate G0 measured, and at equal cost it wins.
    * `fused` -- the same mathematics as `jacobi`, restructured into the shape the Vulkan kernel
      needs: one thread per vertex, each recomputing the multiplier update of every constraint it
      touches instead of reading a per-constraint result another pass produced. See
      `_apply_fused`. It exists so the port target can be validated in Python before any HLSL is
      written, and `--sweep fused` must reproduce `--sweep jacobi`'s scores.

    Each iteration runs up to four passes, in this order:

        structural sweep -> area floor -> guide -> contacts

    The last three are skipped unless enabled, so with `area=None`, `guide=None` and the historical
    modes the loop is byte-for-byte the one gate G0 measured. The order encodes who gets the last
    word: the guide is a soft target and sits *after* the structural sweep so it can shape the
    result, but *before* the contacts, so non-penetration is never traded away for staying near the
    network's prediction.
    """
    if config.iterations <= 0:
        return position
    count = constraints.count
    if count == 0:
        return position

    pinned = pin_mask.reshape(-1, 1)
    tables = bake_tables(constraints, config, timestep)
    denominator, alpha, alive = tables.denominator, tables.alpha, tables.alive

    current, multiplier = _initialise(
        constraints, config, position=position, inertial=inertial, pinned=pinned,
        pin_target=pin_target, denominator=denominator,
    )
    averaging = constraints.incident.clamp_min(1.0)
    groups = constraints.colour_groups() if config.sweep == "coloured" else None
    if config.sweep == "fused":
        if config.mode == "warmstart":
            # lambda_0 is per constraint and would have to be scattered to the per-slot layout.
            # Gate G0 measured warmstart as worth nothing, so rather than ship an untested
            # scatter the combination is refused.
            raise ValueError("sweep='fused' does not implement mode='warmstart'")
        multiplier = torch.zeros_like(constraints.slots, dtype=current.dtype)

    # Auxiliary multipliers, fresh per call exactly as the structural ones are. `project` is one
    # substep, so "per call" and "per substep" are the same rule.
    guide_multiplier = torch.zeros_like(constraints.inverse_mass)
    area_multiplier = (
        None if area is None else torch.zeros_like(area.slots, dtype=current.dtype)
    )

    for _ in range(config.iterations):
        if groups is not None:
            for group in groups:
                current, multiplier = _apply_group(
                    current, multiplier, constraints, config, group, denominator, alpha, alive
                )
        elif config.sweep == "fused":
            current, multiplier = _apply_fused(
                current, multiplier, constraints, config, denominator, alpha, alive, averaging, pinned
            )
        else:
            current, multiplier = _apply_jacobi(
                current, multiplier, constraints, config, denominator, alpha, alive, averaging, pinned
            )

        if area is not None and config.area_floor > 0.0:
            current, area_multiplier = _apply_area(
                current, area_multiplier, area, config, constraints.inverse_mass, pinned, timestep
            )

        if guide is not None:
            current, guide_multiplier = _resolve_guide(
                current, guide, guide_multiplier, constraints.inverse_mass, pinned,
                confidence, config.guide_compliance, timestep,
            )

        if config.collision and contacts is not None and contacts.vertex.numel() > 0:
            current = _resolve_contacts(current, contacts, constraints.inverse_mass, config.contact_offset)

    return torch.where(pinned, pin_target, current)


def _scatter_constraint_forces(
    constraints: ConstraintSet, gradient: torch.Tensor, magnitude: torch.Tensor
) -> torch.Tensor:
    """M^-1 J^T lambda, accumulated through the deterministic padded gather."""
    padded_gradient = torch.cat((gradient, torch.zeros_like(gradient[:1])), dim=0)
    padded_magnitude = torch.cat((magnitude, torch.zeros_like(magnitude[:1])), dim=0)
    contribution = (
        constraints.signs.unsqueeze(-1)
        * padded_gradient[constraints.slots]
        * padded_magnitude[constraints.slots].unsqueeze(-1)
    ).sum(dim=1)
    return constraints.inverse_mass * contribution


def _initialise(
    constraints: ConstraintSet,
    config: SolverConfig,
    *,
    position: torch.Tensor,
    inertial: torch.Tensor,
    pinned: torch.Tensor,
    pin_target: torch.Tensor,
    denominator: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick the starting state. This one choice is the whole difference between the schemes.

    * `standard` (plan scheme A) -- start from the network's position. Its displacement is kept in
      full and acts as a new inertial prediction, so the network is free to inject dynamics the
      constraints never asked for.
    * `warmstart` (plan scheme C, and scheme B without training anything) -- convert the network's
      displacement into multipliers first, then start from `x_tilde + M^-1 J^T lambda_0`. Only the
      part of the prediction expressible as a constraint force survives; the rest is discarded.
      The fit is one diagonal Jacobi step of the normal equations rather than a real least-squares
      solve, which is the cheap approximation a GPU kernel could afford.
    * `nowarm` -- discard the displacement entirely and start from `x_tilde`. The network still
      shapes nothing but its own graph inputs, so this bounds what the position prediction is worth.
    * `guide` -- start from `x_tilde`, same as `nowarm`, and let the network back in through
      `_resolve_guide` as a compliant per-vertex target instead of as the starting state. This is the
      one mode that also changes the *sweep* and not only the initialiser, which is worth saying out
      loud because the paragraph above is otherwise a promise this module keeps. It is spelled that way
      because the network's authority has to be a dial rather than a switch, and the initialiser alone
      only has two settings.

      Note what `guide` does *not* reduce to. There is no compliance that makes it reproduce
      `standard`: `standard` applies the network once with infinite stiffness and then never again,
      while `guide` applies it every iteration with finite stiffness. What it does reduce to exactly is
      pure XPBD -- confidence 0 everywhere leaves the guide masked off and the solve is `nowarm` with
      whatever external displacement the caller folded into `x_tilde`.

    Writing it this way is the point of the exercise: the schemes differ by their initialiser, not
    by their solver, so a difference in the result cannot be blamed on a different solve.
    """
    count = constraints.count
    zero = torch.zeros(count, device=position.device, dtype=position.dtype)
    if config.mode == "standard":
        return torch.where(pinned, pin_target, position), zero

    base = torch.where(pinned, pin_target, inertial)
    if config.mode in ("nowarm", "guide"):
        return base, zero
    if config.mode != "warmstart":
        raise ValueError(f"unknown solver mode {config.mode!r}")

    # lambda_0 from the network's displacement, measured at x_tilde so the gradient is the one the
    # first sweep will use.
    first, second = constraints.pairs[:, 0], constraints.pairs[:, 1]
    delta = base[first] - base[second]
    distance = torch.linalg.vector_norm(delta, dim=-1)
    safe = distance > 1.0e-9
    gradient = torch.where(
        safe.unsqueeze(-1), delta / distance.clamp_min(1.0e-9).unsqueeze(-1), torch.zeros_like(delta)
    )
    displacement = position - inertial
    projected = (gradient * (displacement[first] - displacement[second])).sum(dim=-1)
    multiplier = torch.where(safe, projected / denominator, zero) * config.warmstart_scale
    # Averaged over the constraints meeting at each vertex, for the same reason the Jacobi sweep is:
    # a vertex carries ~11 constraints on this mesh, and summing all of their forces without
    # averaging overshoots by roughly that factor. Measured, the unaveraged version scored worse
    # than discarding the prediction outright, which is a property of the scaling and not of the
    # scheme.
    forces = _scatter_constraint_forces(constraints, gradient, multiplier) / constraints.incident.clamp_min(1.0)
    return base + forces, multiplier


def _constraint_step(
    current: torch.Tensor,
    multiplier: torch.Tensor,
    constraints: ConstraintSet,
    config: SolverConfig,
    index: torch.Tensor | slice,
    denominator: torch.Tensor,
    alpha: torch.Tensor,
    alive: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gradient and multiplier increment for a subset of constraints."""
    pairs = constraints.pairs[index]
    delta = current[pairs[:, 0]] - current[pairs[:, 1]]
    distance = torch.linalg.vector_norm(delta, dim=-1)
    safe = distance > 1.0e-9
    gradient = torch.where(
        safe.unsqueeze(-1), delta / distance.clamp_min(1.0e-9).unsqueeze(-1), torch.zeros_like(delta)
    )
    residual = distance - constraints.target_length[index]
    if config.one_sided:
        residual = residual.clamp_min(0.0)
    numerator = -residual - alpha[index] * multiplier[index]
    step = torch.where(safe & alive[index], numerator / denominator[index], torch.zeros_like(numerator))
    return pairs, gradient, step


def _apply_group(
    current: torch.Tensor,
    multiplier: torch.Tensor,
    constraints: ConstraintSet,
    config: SolverConfig,
    group: torch.Tensor,
    denominator: torch.Tensor,
    alpha: torch.Tensor,
    alive: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One Gauss-Seidel colour: vertex-disjoint, so both endpoints are written in place."""
    pairs, gradient, step = _constraint_step(
        current, multiplier, constraints, config, group, denominator, alpha, alive
    )
    multiplier = multiplier.clone()
    multiplier[group] = multiplier[group] + step
    first, second = pairs[:, 0], pairs[:, 1]
    correction = gradient * step.unsqueeze(-1)
    updated = current.clone()
    updated[first] = current[first] + constraints.inverse_mass[first] * correction
    updated[second] = current[second] - constraints.inverse_mass[second] * correction
    return updated, multiplier


def _apply_jacobi(
    current: torch.Tensor,
    multiplier: torch.Tensor,
    constraints: ConstraintSet,
    config: SolverConfig,
    denominator: torch.Tensor,
    alpha: torch.Tensor,
    alive: torch.Tensor,
    averaging: torch.Tensor,
    pinned: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All constraints from the same `x`, accumulated through the deterministic padded gather."""
    _, gradient, step = _constraint_step(
        current, multiplier, constraints, config, slice(None), denominator, alpha, alive
    )
    multiplier = multiplier + step
    correction = config.relaxation * _scatter_constraint_forces(constraints, gradient, step) / averaging
    return current + torch.where(pinned, torch.zeros_like(correction), correction), multiplier


def _apply_fused(
    current: torch.Tensor,
    multiplier: torch.Tensor,
    constraints: ConstraintSet,
    config: SolverConfig,
    denominator: torch.Tensor,
    alpha: torch.Tensor,
    alive: torch.Tensor,
    averaging: torch.Tensor,
    pinned: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`_apply_jacobi` restructured as one thread per vertex, which is the Vulkan port target.

    `_apply_jacobi` needs two GPU dispatches per iteration: one over constraints to produce every
    `Delta lambda`, then one over vertices to gather them, because a vertex cannot see a neighbour
    constraint's result until the constraint pass has finished writing. At 2.8 us per dispatch and
    128 iterations that is 0.72 ms, about eleven GNN blocks, which is more than the whole hybrid is
    worth. Here each vertex recomputes the update of every constraint it touches, so the iteration
    is a single dispatch: 0.36 ms, and the redundant arithmetic (1377 vertices x ~11 constraints)
    is far below what a dispatch costs anyway.

    The price is that `lambda` becomes per (vertex, slot) rather than per constraint, so the two
    endpoints of a constraint each keep their own copy. Those copies do not drift:

    * both threads form `x[pairs[c, 0]] - x[pairs[c, 1]]` in that stored order, not "mine minus
      theirs", so the difference and its norm are the same floats on both;
    * `target_length`, `alpha` and `denominator` are read from the baked tables, so in particular
      `weight_sum` is not re-added in opposite orders (see `SolverTables`);
    * the endpoint-dependent sign is `signs`, applied after the multiplier update, not inside it.

    Given the same inputs and the same instructions the two updates are therefore bit-identical,
    and the copies stay equal for every iteration. `test_fused_sweep_matches_jacobi` pins this
    down against `_apply_jacobi` itself.
    """
    count = constraints.count
    slots = constraints.slots
    pairs = torch.cat((constraints.pairs, torch.zeros_like(constraints.pairs[:1])), dim=0)[slots]
    first, second = pairs[..., 0], pairs[..., 1]
    assert int(slots.max()) <= count, "slot table must pad with the sentinel index `count`"

    delta = current[first] - current[second]
    distance = torch.linalg.vector_norm(delta, dim=-1)
    safe = distance > 1.0e-9
    gradient = torch.where(
        safe.unsqueeze(-1), delta / distance.clamp_min(1.0e-9).unsqueeze(-1), torch.zeros_like(delta)
    )

    def gather(values: torch.Tensor) -> torch.Tensor:
        return torch.cat((values, torch.zeros_like(values[:1])), dim=0)[slots]

    residual = distance - gather(constraints.target_length)
    if config.one_sided:
        residual = residual.clamp_min(0.0)
    numerator = -residual - gather(alpha) * multiplier
    step = torch.where(
        safe & gather(alive.to(current.dtype)).bool(),
        numerator / gather(denominator).clamp_min(1.0e-20),
        torch.zeros_like(numerator),
    )
    # Padded lanes point at the sentinel constraint, whose endpoints are both vertex 0, so their
    # distance is 0, `safe` is false and `step` is 0. `signs` is 0 there as well.
    correction = (constraints.signs.unsqueeze(-1) * gradient * step.unsqueeze(-1)).sum(dim=1)
    correction = config.relaxation * constraints.inverse_mass * correction / averaging
    return (
        current + torch.where(pinned, torch.zeros_like(correction), correction),
        multiplier + step,
    )


def _resolve_contacts(
    position: torch.Tensor, contacts: Contacts, inverse_mass: torch.Tensor, offset: float
) -> torch.Tensor:
    """One-sided plane projection against the body proxy.

    Each cloth vertex has at most one contact (the runtime's search keeps only the nearest
    proxy), so this is a scatter to distinct destinations and needs no accumulation. The proxy
    is treated as infinite mass, matching `hood_integrate.comp`.
    """
    vertex = contacts.vertex
    movable = inverse_mass[vertex].reshape(-1) > 0.0
    signed = ((position[vertex] - contacts.point) * contacts.normal).sum(dim=-1) - offset
    penetrating = movable & (signed < 0.0)
    if not bool(penetrating.any()):
        return position
    updated = position.clone()
    correction = (-signed).clamp_min(0.0).unsqueeze(-1) * contacts.normal
    updated[vertex[penetrating]] = position[vertex[penetrating]] + correction[penetrating]
    return updated


def guide_confidence(
    guide: torch.Tensor,
    position: torch.Tensor,
    min_edge: torch.Tensor | None,
    ratio: float,
) -> torch.Tensor | None:
    """Per-vertex trust in the network's prediction, from how far it wants to move the vertex.

    Returns `None` when the gate is off, which the caller reads as "confidence 1 everywhere".

    The one signal used here is displacement against the vertex's own shortest constraint. That is
    deliberately the only gate wired up, for two reasons.

    First, its input already exists: `min_edge` has been baked into every `.vxpbd` since
    `tools/bake_xpbd_constraints.py::per_vertex_min_edge`, and no kernel has ever read it. A per-vertex
    shortest edge is also the right scale -- the plan this came from originally clamped against the
    global minimum edge, which on a real garment lets the single shortest edge in the mesh dictate the
    clamp everywhere.

    Second, the obvious alternative gate is unusable. Keying confidence on post-prediction penetration
    depth looks natural and is backwards: the nearest-proxy half-plane is blind to tunnelling, and
    measured, pushing the hem 0.12 m into the body reports 28.5% penetration while pushing it 0.25 m
    reports 1%. A gate on that column would trust the network *more* the further through the body it
    threw a vertex. Until there is an inside-outside test, penetration cannot drive this.

    The ramp is linear in the excess and clamped to [0, 1], so a vertex the network wants to move one
    trust radius stays fully trusted and one it wants to move twice that is fully distrusted.
    """
    if ratio <= 0.0 or min_edge is None:
        return None
    displacement = torch.linalg.vector_norm(guide - position, dim=-1, keepdim=True)
    allowed = (ratio * min_edge.reshape(-1, 1)).clamp_min(1.0e-12)
    return (2.0 - displacement / allowed).clamp(0.0, 1.0)


def _resolve_guide(
    position: torch.Tensor,
    guide: torch.Tensor,
    multiplier: torch.Tensor,
    inverse_mass: torch.Tensor,
    pinned: torch.Tensor,
    confidence: torch.Tensor | None,
    compliance: float,
    timestep: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compliant per-vertex pull towards the network's prediction.

    This is the whole point of `mode="guide"`. In `mode="standard"` the network's output *is* the state
    the solve starts from, so the network decides the entire step and XPBD can only project afterwards
    from whatever initial value it was handed. Measured on `sprint_start`, that lets one prediction
    throw the hem far out of the feasible set: the hybrid peaks at edge p95 10.4 at step 60 where pure
    XPBD sits at 2.5, and only comes back (1.2) once the animation stops. Here the network instead
    contributes a force-like term the constraints can outvote.

    `C(x) = |x - g|`, one constraint per vertex, so `grad C` is a unit vector and
    `grad C M^-1 grad C^T` is just `w`. With `alpha-tilde = compliance / h^2`:

        Delta lambda = (-C - alpha-tilde * lambda) / (w + alpha-tilde)
        Delta x      = w * n * Delta lambda,     n = (x - g) / |x - g|

    `n` points from the guide towards the vertex and `Delta lambda` is negative, so the correction
    moves the vertex towards the guide. On the first iteration (`lambda = 0`) it closes the fraction
    `w / (w + alpha-tilde)` of the gap, which is the dial: `alpha-tilde = 0` snaps onto the guide,
    `alpha-tilde -> infinity` leaves the vertex where the physics put it. Accumulating `lambda` across
    iterations is what stops repeated application from stiffening into a hard constraint -- the
    `-alpha-tilde * lambda` term is exactly the correction that makes the converged state the compliant
    equilibrium rather than `x = g`.

    Two things this is NOT:

    * It is not the multiplier warm start gate G0 measured as worthless. That failed structurally: a
      displacement in the null space of `J^T` -- a rigid translation, any deformation that leaves every
      edge length alone -- cannot be written as `J^T lambda` and was discarded by construction, which
      `test_warmstart_discards_a_null_space_displacement` pins down. This constraint's Jacobian is the
      identity, so it spans that null space and can carry the skirt swing and the low-frequency fold
      placement the network is actually good at.
    * It is not applied inside the incident-averaged structural correction. `_apply_fused` divides a
      vertex's accumulated correction by the number of constraints touching it (~18 on CH10032), so
      adding the guide there would both dilute every structural constraint at that vertex by ~5% and
      put a stiff guide in a tug of war with stiff edges under Jacobi averaging, which converges badly.
      It goes here, next to the contacts, outside that average.

    Pinned vertices are untouched: they carry zero inverse mass, and the caller overwrites them with
    their target anyway.
    """
    delta = position - guide
    distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    safe = distance > 1.0e-9
    normal = torch.where(safe, delta / distance.clamp_min(1.0e-9), torch.zeros_like(delta))

    alpha = compliance / max(timestep * timestep, 1.0e-12)
    if confidence is not None:
        # Zero confidence has to mean "no guide at all", so it maps to an infinite alpha-tilde rather
        # than a large one. Dividing would produce inf and then nan through 0 * inf, so the step is
        # masked instead.
        trusted = confidence > 0.0
        alpha_tilde = torch.where(
            trusted, alpha / confidence.clamp_min(1.0e-6), torch.zeros_like(confidence)
        )
    else:
        trusted = torch.ones_like(distance, dtype=torch.bool)
        alpha_tilde = torch.full_like(distance, alpha)

    movable = inverse_mass > 0.0
    numerator = -distance - alpha_tilde * multiplier
    step = torch.where(
        safe & movable & trusted,
        numerator / (inverse_mass + alpha_tilde).clamp_min(1.0e-20),
        torch.zeros_like(numerator),
    )
    correction = inverse_mass * normal * step
    return (
        position + torch.where(pinned, torch.zeros_like(correction), correction),
        multiplier + step,
    )


def _apply_area(
    current: torch.Tensor,
    multiplier: torch.Tensor,
    area: AreaConstraints,
    config: SolverConfig,
    inverse_mass: torch.Tensor,
    pinned: torch.Tensor,
    timestep: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-sided triangle area floor, `C = max(rho * A_0 - A(x), 0)`.

    This is the constraint the existing set is missing. Today's structural constraints are mesh-edge
    distances plus a two-hop distance standing in for bending, run one-sided so only stretch is
    resisted. Nothing in that set objects to a triangle whose area goes to zero: an edge that is too
    short is legal, and a triangle squashed flat against the body can make the headline metric
    *improve*, because `edge_ratio_p95` is an upper-tail statistic. Measured on `sprint_start` step
    120, the hybrid holds the best p95 of the three branches (1.291) together with the most collapsed
    edges (13.3%), the smallest triangles (area median 0.478) and the most flipped ones (0.435).

    A floor rather than an equality on purpose: it forbids a triangle approaching zero area without
    requiring it to keep any particular size or orientation, so normal folding is still free.

    The gradients, with `n` the unit normal of `cross(b - a, c - a)`:

        grad_a A = 0.5 * (b - c) x n        grad_b A = 0.5 * (c - a) x n
        grad_c A = 0.5 * (a - b) x n

    which sum to zero, as a constraint that cannot move the centre of mass must. `grad C = -grad A`
    while the floor is violated, so the correction inflates the triangle.

    Unlike a distance constraint the gradients are not unit vectors, so the denominator is
    `sum_i w_i |grad_i A|^2` and not `sum_i w_i`. It is summed in corner order 0, 1, 2 -- fixed by the
    mesh -- so all three vertices of a triangle produce the same multiplier update, the same guarantee
    `SolverTables` buys for the pair sweep by baking `weight_sum`.
    """
    count = area.count
    if count == 0 or config.area_floor <= 0.0:
        return current, multiplier

    slots = area.slots
    padded = torch.cat((area.triangles, torch.zeros_like(area.triangles[:1])), dim=0)[slots]
    corners = current[padded]                                   # [V, K, 3, 3]
    first, second, third = corners[..., 0, :], corners[..., 1, :], corners[..., 2, :]
    normal = torch.linalg.cross(second - first, third - first)
    twice_area = torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
    safe = twice_area > 1.0e-12
    unit = torch.where(safe, normal / twice_area.clamp_min(1.0e-12), torch.zeros_like(normal))

    def gather(values: torch.Tensor) -> torch.Tensor:
        return torch.cat((values, torch.zeros_like(values[:1])), dim=0)[slots]

    target = config.area_floor * gather(area.target_area).unsqueeze(-1)
    residual = (target - 0.5 * twice_area).clamp_min(0.0)

    gradient = torch.stack((
        0.5 * torch.linalg.cross(second - third, unit),
        0.5 * torch.linalg.cross(third - first, unit),
        0.5 * torch.linalg.cross(first - second, unit),
    ), dim=-2)                                                  # [V, K, 3, 3]
    weight = inverse_mass.reshape(-1)[padded].unsqueeze(-1)      # [V, K, 3, 1]
    # Corner order 0, 1, 2 -- a property of the mesh, so every vertex of the triangle sums the same
    # three floats in the same sequence.
    denominator = (weight * gradient.square().sum(dim=-1, keepdim=True)).squeeze(-1)
    denominator = denominator[..., 0] + denominator[..., 1] + denominator[..., 2]

    alpha = config.area_compliance / max(timestep * timestep, 1.0e-12)
    live = (slots < count) & safe.squeeze(-1) & (denominator > 0.0)
    numerator = -residual.squeeze(-1) - alpha * multiplier
    step = torch.where(live, numerator / (denominator + alpha).clamp_min(1.0e-20),
                       torch.zeros_like(numerator))

    # Each vertex applies only the gradient of the corner it occupies.
    own = torch.gather(
        gradient, -2, area.corner[..., None, None].expand(-1, -1, 1, 3)
    ).squeeze(-2)                                               # [V, K, 3]
    correction = (-own * step.unsqueeze(-1)).sum(dim=1)
    correction = config.relaxation * inverse_mass * correction / area.incident.clamp_min(1.0)
    return (
        current + torch.where(pinned, torch.zeros_like(correction), correction),
        multiplier + step,
    )


def _blend(start: torch.Tensor, end: torch.Tensor, fraction: float) -> torch.Tensor:
    """Linear blend that is *exact* at the endpoints.

    `torch.lerp(a, b, 1.0)` computes `a + 1.0 * (b - a)`, which is not bit-identical to `b` -- when
    `a` is large and `b - a` small the subtraction loses the low bits and the addition does not put
    them back. On `sprint_start` the root travels to z = 10 m while a substep moves a vertex by
    millimetres, which is exactly that regime. Returning the endpoint itself is what lets
    `substeps=1` reproduce the un-substepped path bit for bit, and that equality is the regression
    anchor for this whole change.
    """
    if fraction >= 1.0:
        return end
    if fraction <= 0.0:
        return start
    return torch.lerp(start, end, fraction)


def step_substepped(
    constraints: ConstraintSet,
    config: SolverConfig,
    *,
    scene,
    graph,
    guide: torch.Tensor | None,
    timestep: float,
    frame: float,
    frame_advance: float,
    substeps: int = 1,
    area: AreaConstraints | None = None,
    min_edge: torch.Tensor | None = None,
    gravity: torch.Tensor | None = None,
) -> torch.Tensor:
    """One visual frame, solved as `substeps` physics substeps. The network runs once, outside.

    Why this exists. Every XPBD iteration in this repository has so far happened inside a single
    `dt = 1/30` step, after one prediction: 128 sweeps all linearising around the same predicted
    state, with the same body pose, the same contact set and no velocity update in between. Small
    Steps' result is that at equal solver cost, `N` steps of one iteration usually beats one step of
    `N` iterations, because each small step re-forms the inertial prediction, the constraint
    gradients, the contacts and the velocity instead of polishing one bad linearisation. The
    repository's own numbers say the same thing from the other side: branch B's score is 81-86%
    `over` -- edges stretched and never pulled back, a pure convergence failure -- and it was
    measured at one step by 128 Jacobi sweeps. `plans/gnn/gnn-xpbd-v2.md` section 8.3 asked for this
    comparison and it was never run.

    What each substep re-does, and why every item is load-bearing:

    1. the body pose is re-skinned at a fractional frame, so the cloth is not chasing a pose 1/30 s
       stale (on `sprint_start` the root covers 0.32 m per frame);
    2. the inertial prediction is re-formed from the substep's own state;
    3. the contact set is searched fresh, because a vertex that moved may be nearest a different
       proxy -- see `contacts_from_search`;
    4. `alpha-tilde = compliance / h^2` is recomputed with `h = dt / substeps`;
    5. the guide target advances only `1/substeps` of the way towards the network's prediction.

    Item 5 is the one that is easy to get wrong and worth being explicit about. Substepping the
    solver while still jumping straight to `x_gnn` first would change nothing that matters: the
    overshoot has already happened by the time the first substep starts, and the remaining substeps
    only clean up after it. The guide has to arrive in `substeps` instalments for the split to mean
    anything.

    `gravity` is a per-second-squared acceleration folded into the inertial prediction as `h^2 * g`
    each substep, and it is only used by the modes that start from `x_tilde` (`nowarm`, `guide`).
    `standard` takes its start from the guide, which already carries whatever acceleration produced
    it. Two consequences worth stating: a `guide` arm with confidence driven to zero degrades to
    *ballistic plus constraints* rather than to inertial coasting, which is the meaningful pure-XPBD
    fallback; and a substepped ballistic arm accumulates less gravity displacement over the frame
    than a single step does (`dt^2 g (N+1) / 2N` against `dt^2 g`), which is the ordinary Verlet
    small-step correction and not a bug -- the substepped figure is the more accurate one.

    `frame` is the animation frame the step starts on and `frame_advance` how many frames it covers,
    both as floats so a scaled clip works: `frame_of` gives integers only because the un-substepped
    path has nowhere to put a fraction.

    The first substep reads its inertial prediction, pin target and contacts from `graph`, which the
    network already built at the frame boundary; later substeps have no graph and recompute them.
    That is not a shortcut, it is what keeps `substeps=1` exactly equal to the previous behaviour.
    """
    if substeps < 1:
        raise ValueError("substeps must be at least 1")

    step_size = timestep / substeps
    external = None if gravity is None else (step_size * step_size) * gravity.reshape(1, 3)

    position = graph.effective_position
    previous = graph.effective_previous
    entry = position

    # Confidence is decided once for the whole frame, not per substep. The question the trust region
    # asks is "does the network want to move this vertex further than its own geometry can absorb
    # *this frame*", and the quantity that answers it is the network's displacement away from physics
    # -- `guide - x_tilde`, which is exactly the decoder's output. Recomputing it per substep against
    # the instalment would measure 1/N of the same displacement and, at a fixed radius, would quietly
    # stop firing as the substep count rose: the gate would appear to work at N=1 and be inert at N=8.
    # hood_integrate.comp computes the same thing from `acceleration`, which IS that difference.
    confidence = None
    if guide is not None:
        confidence = guide_confidence(guide, inertial_prediction(graph), min_edge, config.guide_trust_ratio)

    for index in range(substeps):
        fraction = float(index + 1) / substeps
        search_time = frame + frame_advance * (float(index) / substeps)
        target_time = frame + frame_advance * fraction

        if index == 0:
            # The state entering the frame is a full visual frame old, so `position - previous` is
            # `dt * v`, not `h * v`. Handing that to a substep of length `h = dt / N` would predict a
            # whole frame of motion in a fraction of the time -- N times the real velocity, every
            # substep, which diverges immediately rather than subtly (measured: `sprint_start` goes
            # non-finite inside a few steps at N=4).
            #
            # At N=1 the scale is 1 and `inertial_prediction`'s own expression is used verbatim
            # instead: it forms `2a - b` where the general branch forms `a + (a - b) / N`, and those
            # two are not the same float. Keeping the exact expression is what makes `substeps=1`
            # bit-identical to the un-substepped path.
            if substeps == 1:
                inertial = inertial_prediction(graph)
            else:
                inertial = position + (position - previous) / substeps
            pin_target = graph.pin_target
        else:
            # From here on `previous` is one substep back, so the difference is already `h * v`.
            inertial = position + (position - previous)
            pin_target = scene.cloth_target_at(target_time)
        if external is not None and config.mode != "standard":
            inertial = inertial + external

        contacts = None
        if config.collision:
            if index == 0 and substeps == 1:
                # The graph's own pairing, so a single substep is the historical path exactly.
                proxy_next, _ = scene.proxy_at(target_time)
                _, normals = scene.proxy_at(search_time)
                contacts = contacts_from_graph(graph, proxy_next, normals)
            else:
                proxy_now, normals = scene.proxy_at(search_time)
                proxy_next, _ = scene.proxy_at(target_time)
                contacts = contacts_from_search(position, proxy_now, proxy_next, normals)

        target = None if guide is None else _blend(entry, guide, fraction)
        start = inertial
        if config.mode == "standard":
            # `standard` *is* "start from the network", so the instalment becomes the starting state
            # rather than a constraint, and there is no guide pass.
            if target is not None:
                start = target
            target = None

        corrected = project(
            constraints, config,
            position=start,
            inertial=inertial,
            pin_mask=graph.pin_mask,
            pin_target=pin_target,
            timestep=step_size,
            contacts=contacts,
            guide=target,
            confidence=confidence if target is not None else None,
            area=area,
        )
        # Velocity is implicit in (position, previous), so carrying the pre-solve state forward as
        # `previous` is the position-based velocity update: the solver's correction becomes part of
        # the velocity the next substep predicts from.
        previous, position = position, corrected

    return position


def stretch_residual(constraints: ConstraintSet, position: torch.Tensor) -> torch.Tensor:
    """RMS violation of the stretch constraints, for the residual-monotonicity check."""
    mask = constraints.kind == STRETCH
    pairs = constraints.pairs[mask]
    target = constraints.target_length[mask]
    distance = torch.linalg.vector_norm(position[pairs[:, 0]] - position[pairs[:, 1]], dim=-1)
    return (distance - target).square().mean().sqrt()


def primal_residual(
    constraints: ConstraintSet, position: torch.Tensor, inertial: torch.Tensor, pin_mask: torch.Tensor
) -> torch.Tensor:
    """RMS of M^-1 g = x - x_tilde over free vertices.

    How far the state sits from the inertial prediction, i.e. how much of the step the solver and
    the network together attribute to something other than inertia.
    """
    free = (~pin_mask.reshape(-1)) & (constraints.inverse_mass.reshape(-1) > 0.0)
    if not bool(free.any()):
        return torch.zeros((), device=position.device, dtype=position.dtype)
    return (position - inertial)[free].square().mean().sqrt()
