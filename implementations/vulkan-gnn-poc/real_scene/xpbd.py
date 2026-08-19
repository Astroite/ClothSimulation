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


@dataclass
class SolverConfig:
    """One point in the gate's configuration matrix."""

    iterations: int = 4
    mode: str = "standard"                 # "standard" | "warmstart" | "nowarm"
    sweep: str = "coloured"                # "coloured" (Gauss-Seidel) | "jacobi"
    stretch_compliance: float = 0.0        # metres^2 / newton; 0 = rigid
    bend_compliance: float = 1.0e-5
    one_sided: bool = False                # resist stretch only, never compression
    relaxation: float = 1.0                # Jacobi under/over-relaxation
    warmstart_scale: float = 1.0           # scales lambda_0 in "warmstart" mode
    collision: bool = True
    contact_offset: float = DEFAULT_CONTACT_OFFSET

    def compliance_per_constraint(self, kind: torch.Tensor) -> torch.Tensor:
        return torch.where(
            kind == BEND,
            torch.full_like(kind, 0.0, dtype=torch.float32) + self.bend_compliance,
            torch.full_like(kind, 0.0, dtype=torch.float32) + self.stretch_compliance,
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
) -> torch.Tensor:
    """Run `config.iterations` deterministic XPBD sweeps and return the corrected position.

    Pinned vertices are placed on their target before the first sweep, not after the last. They
    carry zero inverse mass so they never move again, which makes them infinite-mass anchors the
    solve can propagate from. Snapping them at the end instead would leave the pin violation
    entirely unsolved and inject it fresh every step -- measured, that alone drove edge P95 to 28
    over 137 steps. `integrate()` already returns pins on target, so in the normal pipeline this is
    a no-op; it is here so the function cannot be misused.

    Two sweep schedules:

    * `coloured` -- Gauss-Seidel over vertex-disjoint colour groups, the schedule a GPU kernel
      would use. Within a colour both endpoints are written in place; the indices are distinct by
      construction so the writes are deterministic.
    * `jacobi` -- every constraint reads the same `x` and per-vertex corrections are averaged over
      the constraints touching that vertex. Needs no colouring, but converges much more slowly:
      128 sweeps cut the CH10032 stretch residual only 3.2x, versus a few sweeps coloured.
    """
    if config.iterations <= 0:
        return position
    count = constraints.count
    if count == 0:
        return position

    pinned = pin_mask.reshape(-1, 1)
    first, second = constraints.pairs[:, 0], constraints.pairs[:, 1]
    weight_first = constraints.inverse_mass[first].reshape(-1)
    weight_second = constraints.inverse_mass[second].reshape(-1)
    weight_sum = weight_first + weight_second
    alive = weight_sum > 0.0

    compliance = config.compliance_per_constraint(constraints.kind)
    alpha = compliance / max(timestep * timestep, 1.0e-12)
    denominator = (weight_sum + alpha).clamp_min(1.0e-20)

    current, multiplier = _initialise(
        constraints, config, position=position, inertial=inertial, pinned=pinned,
        pin_target=pin_target, denominator=denominator,
    )
    averaging = constraints.incident.clamp_min(1.0)
    groups = constraints.colour_groups() if config.sweep == "coloured" else None

    for _ in range(config.iterations):
        if groups is not None:
            for group in groups:
                current, multiplier = _apply_group(
                    current, multiplier, constraints, config, group, denominator, alpha, alive
                )
        else:
            current, multiplier = _apply_jacobi(
                current, multiplier, constraints, config, denominator, alpha, alive, averaging, pinned
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

    Writing it this way is the point of the exercise: the schemes differ by their initialiser, not
    by their solver, so a difference in the result cannot be blamed on a different solve.
    """
    count = constraints.count
    zero = torch.zeros(count, device=position.device, dtype=position.dtype)
    if config.mode == "standard":
        return torch.where(pinned, pin_target, position), zero

    base = torch.where(pinned, pin_target, inertial)
    if config.mode == "nowarm":
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
