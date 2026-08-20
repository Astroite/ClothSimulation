#!/usr/bin/env python3
"""S10: does the solver come back from a bad pose, and how fast must the motion be to break it?

Gate G0 scored every branch by its distance to the HOOD teacher. That answers "can we reproduce a
44 ms solver cheaply" and nothing else. The reasons this project exists instead of shipping
ChaosCloth are different ones: authoring is laborious, fast motion breaks it, and once the cloth
has penetrated or tangled it never returns to a good pose. A teacher-relative score is not a proxy
for any of the three. This tool measures the third and the second directly, and it references no
teacher at all.

The metric is the solver's own clean rollout:

    forget(t) = RMS || x_corrupted(t) - x_clean(t) ||

Both rollouts use the same arm, the same weights and the same scene, and differ only in that one
had its state deliberately damaged at step `--warmup`. If forget decays, the solver forgot the
damage and returned to the trajectory it would have been on. If it stays flat or grows, it did not.
No ground truth is needed, so `cloth_rest` -- which results/GATE_G0_RESULTS.md section 1 showed is
not a usable reference on real garments -- never enters the measurement.

Three corruptions, chosen so the prediction is falsifiable rather than decorative:

  stretch    -- scale the free vertices about their centroid. Breaks every edge length.
  fold       -- rotate a connected patch 180 degrees about its own widest axis. A proper rigid
                motion, so every distance inside the patch is preserved *exactly* and only the
                patch's boundary edges are violated -- measured, about 10% of the edges it touches.
  penetrate  -- translate a connected patch rigidly into the body. Also exactly isometric.

The prediction comes from `tests/test_xpbd.py::test_warmstart_discards_a_null_space_displacement`:
a displacement that preserves every edge length lies in the null space of J^T, so no multiplier can
express it and no number of sweeps can see it. `fold` and `penetrate` are built to sit in that null
space up to a thin boundary ring. So branch B should recover from `stretch` and fail to recover from
`fold`; branch A and C should recover from both, because a network trained on plausible states is a
learned map back onto them. If B recovers from `fold` too, that prediction is wrong and the
architecture's main claim over an authored solver loses its mechanism.

Two design choices worth stating because they change what the number means:

  * The corruption is applied to `position` and `previous` **identically**, so it injects a bad pose
    and zero velocity. Damaging only `position` would also inject a large velocity and the run would
    be measuring two things at once.
  * Substepping is not implemented here, so branch B runs one step of k sweeps. For `stretch` that
    under-powers it and `--b-iterations` is provided to give it an equal-cost budget instead. For
    `fold` and `penetrate` substeps cannot help by construction -- they change the schedule, not the
    constraint's null space -- so the central prediction is testable without them.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import torch

POC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC_ROOT))
sys.path.insert(0, str(POC_ROOT / "tools"))

from real_scene.fine15 import Fine15, Fine15Weights  # noqa: E402
from real_scene.tinyhood import load_tinyhood  # noqa: E402
from real_scene.xpbd import (  # noqa: E402
    SolverConfig,
    build_constraints,
    calibrate_from_trajectory,
    contacts_from_graph,
    inertial_prediction,
    project,
)
from gate_g0 import BallisticGravity, load_scene  # noqa: E402
from train_student import advance, curve_point, frame_of  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", nargs="+", default=["ch10032_tpose", "hml_001962"])
    parser.add_argument("--scene-root", type=Path, default=POC_ROOT / ".work/real_scene")
    parser.add_argument("--fine15", type=Path, default=POC_ROOT / ".work/hood_data/fine15.vhood")
    parser.add_argument("--student", type=Path, default=POC_ROOT / ".work/hood_data/student32x12_r1.vhood")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=30, help="step at which the state is damaged")
    parser.add_argument("--frame-scales", nargs="+", type=float, default=[1.0],
                        help="motion playback multipliers; the timestep is deliberately not scaled")
    parser.add_argument("--corruptions", nargs="+", default=["none", "stretch", "fold", "penetrate"],
                        choices=("none", "stretch", "fold", "penetrate"))
    parser.add_argument("--branches", nargs="+", default=["A", "B", "C"], choices=("A", "B", "C"))
    parser.add_argument("--patch", type=int, default=200, help="vertices in the fold/penetrate patch")
    parser.add_argument("--stretch-amount", type=float, default=0.30)
    parser.add_argument("--penetrate-depth", type=float, default=0.12,
                        help="metres to translate the patch inward. The measured penetration this "
                             "produces is NOT monotone in depth: on ch10032_tpose with --patch 200 "
                             "it peaks at 28.5%% of the patch behind its nearest proxy plane at "
                             "0.12 m and falls to 1%% at 0.25 m, because by then the patch has "
                             "passed clean through the leg and its nearest proxy is the far surface. "
                             "See `penetration` -- the same blindness applies to the solver.")
    # Gate G0's winning configuration, so a difference here is the corruption and not a retune.
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--b-iterations", type=int, default=228,
                        help="branch B's sweep count. The default spends the hybrid's whole 2.15 ms "
                             "budget on XPBD alone (2.154 / 0.009437 ms per sweep) so B is not "
                             "beaten merely by being given less compute.")
    parser.add_argument("--sweep", default="jacobi", choices=("coloured", "jacobi", "fused"))
    parser.add_argument("--stretch-compliance", type=float, default=0.0)
    parser.add_argument("--bend-compliance", type=float, default=1.0e-5)
    parser.add_argument("--one-sided", type=int, default=1, choices=(0, 1))
    parser.add_argument("--relaxation", type=float, default=1.0)
    parser.add_argument("--gravity", nargs=3, type=float, default=[0.0, -9.81, 0.0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=POC_ROOT / "results/recovery_probe.json")
    return parser.parse_args()


# --------------------------------------------------------------------------------------------
# Corruptions. Each returns a new position for the whole mesh and is applied to `previous` too.
# --------------------------------------------------------------------------------------------


def geodesic_patch(scene, position: torch.Tensor, size: int) -> torch.Tensor:
    """The `size` free vertices closest to the hem seed *along the mesh*, not through space.

    A Euclidean ball fails here and the failure is not subtle. Measured on `ch10032_tpose`, a ball
    of 200 vertices around a hem seed has ratio-of-thinnest-to-widest principal axis 0.72 and 170
    boundary edges against 1040 interior ones, because the ball reaches across the leg and picks up
    the far side of the garment. Growing the patch over mesh edges instead gives 0.19 and 106
    against 1088. That matters twice over: the patch stays a single connected sheet, so moving it
    rigidly is a fold rather than two pieces sliding past each other, and only ~10% of the edges it
    touches end up carrying any violation at all. Routing is blocked through the pinned band so the
    patch cannot creep around the waistband into the other leg.
    """
    free = ~scene.cloth_pins
    senders, receivers = scene.cloth_senders, scene.cloth_receivers
    count = position.shape[0]
    index = free.nonzero(as_tuple=True)[0]
    pin_centre = position[scene.cloth_pins].mean(dim=0, keepdim=True)
    seed = index[torch.linalg.vector_norm(position[index] - pin_centre, dim=-1).argmax()]

    unreached = count + 1
    hops = torch.full((count,), unreached, dtype=torch.long, device=position.device)
    hops[seed] = 0
    for _ in range(count):
        relaxed = hops.clone()
        relaxed.scatter_reduce_(0, receivers, (hops[senders] + 1).clamp_max(unreached), reduce="amin")
        relaxed.scatter_reduce_(0, senders, (hops[receivers] + 1).clamp_max(unreached), reduce="amin")
        relaxed[~free] = unreached
        relaxed[seed] = 0
        if torch.equal(relaxed, hops):
            break
        hops = relaxed

    reachable = int((hops < unreached).sum())
    if reachable < size:
        raise SystemExit(f"only {reachable} free vertices are reachable from the seed; --patch {size} "
                         f"would silently fall back to unreachable vertices")
    # Hop count first, then straight-line distance to break ties. Both are deterministic, so a rerun
    # damages exactly the same vertices; the 1e6 factor is an exact lexicographic key in float64
    # because hops <= count + 1 and the distances are metres.
    distance = torch.linalg.vector_norm(position - position[seed], dim=-1)
    key = hops.to(torch.float64) * 1.0e6 + distance.to(torch.float64)
    key[~free] = float("inf")
    return key.argsort()[:size]


def corrupt_stretch(scene, position, proxy, args) -> torch.Tensor:
    """Scale the free vertices away from their centroid. Violates every incident edge."""
    free = (~scene.cloth_pins).nonzero(as_tuple=True)[0]
    centre = position[free].mean(dim=0, keepdim=True)
    updated = position.clone()
    updated[free] = centre + (position[free] - centre) * (1.0 + args.stretch_amount)
    return updated


def corrupt_fold(scene, position, proxy, args) -> torch.Tensor:
    """Rotate a patch 180 degrees about its own widest principal axis.

    A half-turn is a proper rigid motion -- `R(v) = 2 (v.u) u - v`, orthogonal with determinant +1 --
    so every distance inside the patch is preserved *exactly*, for any patch shape. That is why this
    replaced an earlier reflection through the patch's thin axis: the reflection needed the patch to
    be planar to be a fold at all, and measured flatness was 0.72, so it was really a scramble that
    violated edges harder than `stretch` did. A half-turn needs no planarity assumption.

    The result is the "cloth ended up on the wrong side" pose: the patch swings through the garment,
    its triangles face backwards, and the only constraints that notice are the ones crossing the
    patch boundary.
    """
    patch = geodesic_patch(scene, position, args.patch)
    centre = position[patch].mean(dim=0, keepdim=True)
    local = position[patch] - centre
    _, _, basis = torch.linalg.svd(local.double(), full_matrices=False)
    axis = basis[0].to(position.dtype).reshape(1, 3)
    updated = position.clone()
    updated[patch] = centre + 2.0 * (local * axis).sum(dim=-1, keepdim=True) * axis - local
    return updated


def corrupt_penetrate(scene, position, proxy, args) -> torch.Tensor:
    """Translate a patch rigidly inward along the body normal nearest its centroid.

    A single translation is exactly isometric, so like `fold` this is invisible to the distance
    constraints. Unlike `fold` it *is* visible to the contact projection, which makes the pair a
    clean discriminator: contacts are a local per-vertex push, so if the hybrid recovers here and
    branch B does not, the difference is the network's global information rather than the contacts.

    The direction is the outward normal of the single proxy vertex nearest the patch centroid,
    negated. Averaging the patch's own per-vertex nearest-proxy directions was tried first and is
    wrong: around a leg those point radially outward in every direction, the mean is near zero, and
    normalising it sent the patch *away* from the body.
    """
    points, normals = proxy
    patch = geodesic_patch(scene, position, args.patch)
    centre = position[patch].mean(dim=0, keepdim=True)
    nearest = int(torch.cdist(centre, points).argmin())
    direction = -torch.nn.functional.normalize(normals[nearest], dim=-1).reshape(1, 3)
    updated = position.clone()
    updated[patch] = position[patch] + args.penetrate_depth * direction
    return updated


CORRUPTIONS = {"stretch": corrupt_stretch, "fold": corrupt_fold, "penetrate": corrupt_penetrate}


def isometry_error(position: torch.Tensor, corrupted: torch.Tensor, patch: torch.Tensor) -> float:
    """Largest change in any intra-patch pairwise distance, which `fold`/`penetrate` claim is zero.

    Computed in float64 from explicit differences rather than via `torch.cdist`. cdist expands
    `||a-b||^2` into `||a||^2 + ||b||^2 - 2 a.b`, which on metre-scale coordinates in float32 loses
    about 5e-4 m -- larger than the quantity being measured, so it reported a genuine isometry as a
    4.9e-4 violation.

    Reported per run and not only asserted in tests, because the claim that the damage lies in the
    null space of J^T is the whole argument and a reader should be able to check it against the same
    numbers the recovery curves came from.
    """
    before = position[patch].double()
    after = corrupted[patch].double()
    before = (before.unsqueeze(1) - before.unsqueeze(0)).norm(dim=-1)
    after = (after.unsqueeze(1) - after.unsqueeze(0)).norm(dim=-1)
    return float((after - before).abs().max().item())


# --------------------------------------------------------------------------------------------
# Rollout
# --------------------------------------------------------------------------------------------


def penetration(scene, graph, position, step, frame_scale, offset) -> tuple[int, float]:
    """Vertices behind their nearest proxy plane, and the deepest such violation in metres.

    Two undercounts, both of which have to be read with the number:

    * Only vertices carrying a world edge are covered, because those are the ones the solver itself
      can see. That makes this the right set for "did it fix what it could see" and the wrong set for
      "how bad is it really".
    * A single nearest-proxy half-plane cannot detect a vertex that has passed all the way through a
      thin limb: once it is out the far side, its nearest proxy is the far surface and the signed
      distance is positive again. Measured while calibrating `corrupt_penetrate`, translating the hem
      patch 0.25 m inward reports 1% penetration where 0.12 m reports 28.5%. This is not a quirk of
      the metric -- `_resolve_contacts` in real_scene/xpbd.py and hood_xpbd.comp use exactly this
      test, so a tunnelled vertex looks legal to the solver too.

    A figure that survives both caveats needs the self-collision and inside-outside work (v2 S5),
    which does not exist yet, so no absolute penetration claim can be made from this column.
    """
    if graph.world_cloth.numel() == 0:
        return 0, 0.0
    _, normals = scene.proxy(min(frame_of(step, frame_scale), scene.frame_count - 1))
    target, _ = scene.proxy(min(frame_of(step + 1, frame_scale), scene.frame_count - 1))
    contacts = contacts_from_graph(graph, target, normals)
    signed = ((position[contacts.vertex] - contacts.point) * contacts.normal).sum(dim=-1) - offset
    depth = (-signed).clamp_min(0.0)
    return int((depth > 0.0).sum().item()), float(depth.max().item())


def rollout(predictor, builder, scene, args, *, frame_scale, hook, damage=None):
    """Closed-loop rollout that can have its state damaged once, recording HOOD-free curves.

    The settle-step rule is the one in tools/compare_student_stability.py:75 -- `advance` carries the
    prediction itself forward as `previous` on step 0, so a correction there has to replace both or
    step 1 builds a velocity from an uncorrected position. It is repeated rather than reused because
    `trace` cannot inject damage or report penetration.
    """
    position = scene.cloth_target(0)
    previous = position.clone()
    positions: list[torch.Tensor] = []
    edge: list[float] = []
    flipped: list[float] = []
    pierced: list[int] = []
    depth: list[float] = []
    injected = None
    with torch.no_grad():
        for step in range(args.steps):
            if damage is not None and step == args.warmup:
                # Same transform on both, so this is a bad pose with no velocity kick.
                damaged = damage(position)
                injected = float((damaged - position).square().mean().sqrt().item())
                previous = previous + (damaged - position)
                position = damaged
            position, previous, graph, _ = advance(
                predictor, builder, scene, position, previous, step, args.mean, args.std, frame_scale
            )
            if hook is not None:
                corrected = hook(position, graph, step)
                previous = corrected if step == 0 else previous
                position = corrected
            if not torch.isfinite(position).all():
                edge.append(float("inf"))
                break
            point = curve_point(scene, position)
            edge.append(point[0])
            flipped.append(point[1])
            count, deepest = penetration(scene, graph, position, step, frame_scale, args.contact_offset)
            pierced.append(count)
            depth.append(deepest)
            positions.append(position.clone())
    return {"positions": positions, "edge": edge, "flipped": flipped,
            "pierced": pierced, "depth": depth, "injected_rms": injected}


def make_hook(scene, constraints, config, *, network, frame_scale):
    """The `(position, graph, step) -> position` projection, matching tools/gate_g0.py's."""

    def hook(position, graph, step):
        contacts = None
        if config.collision and graph.world_cloth.numel() > 0:
            _, normals = scene.proxy(min(frame_of(step, frame_scale), scene.frame_count - 1))
            target, _ = scene.proxy(min(frame_of(step + 1, frame_scale), scene.frame_count - 1))
            contacts = contacts_from_graph(graph, target, normals)
        return project(
            constraints, config,
            position=position,
            inertial=inertial_prediction(graph) if network else position,
            pin_mask=graph.pin_mask, pin_target=graph.pin_target,
            timestep=1.0 / 3.0 if step == 0 else 1.0 / 30.0,
            contacts=contacts,
        )

    return hook


def forgetting(clean: dict, dirty: dict, warmup: int, noise: list[float] | None = None) -> dict:
    """How much of the damage survives, step by step, measured against the arm's own clean run.

    `noise` is the same arm's control curve: an uncorrupted rerun of the identical configuration,
    whose separation from `clean` is pure run-to-run nondeterminism. It has to be subtracted
    pointwise rather than compared against as a scalar, because that separation is itself amplified
    by the chaos in the rollout -- measured on branch A over 90 steps it grows from 3.8e-4 to 7.4e-3,
    a factor of 19. So the honest definition of recovery is:

        excess(t) = forget(t) - noise(t)      how much of the state difference the damage explains
        recovered_at = first t with forget(t) <= 2 * noise(t)

    i.e. the corrupted run has become as close to the clean run as an undamaged rerun is. Comparing
    `forget`'s own start against its own end instead answers a different and misleading question: two
    rollouts of a chaotic system separate even from identical states, so `ratio > 1` does not imply
    the damage persisted. That is also why the caller reports the dirty run's own `edge_p95_end` and
    `flipped_max` against the clean run's -- those compare pose quality rather than trajectory
    identity, and survive the chaos that this metric cannot.
    """
    count = min(len(clean["positions"]), len(dirty["positions"]))
    curve = [float((dirty["positions"][t] - clean["positions"][t]).square().mean().sqrt().item())
             for t in range(warmup, count)]
    result = {"curve": [round(value, 8) for value in curve], "observed_steps": len(curve)}
    if not curve:
        return {**result, "initial": None, "final": None, "curve_max": None,
                "excess_initial": None, "excess_final": None, "excess_ratio": None,
                "half_life": None, "recovered_at": None}
    result.update({"initial": curve[0], "final": curve[-1], "curve_max": max(curve)})
    if noise is None:
        # This *is* the control row: its own curve is the floor, so there is nothing to subtract.
        return {**result, "excess_initial": None, "excess_final": None, "excess_ratio": None,
                "half_life": None, "recovered_at": None}
    floor = noise[: len(curve)] + [noise[-1]] * max(0, len(curve) - len(noise))
    excess = [max(0.0, value - floor[index]) for index, value in enumerate(curve)]
    return {
        **result,
        "excess_curve": [round(value, 8) for value in excess],
        "excess_initial": excess[0],
        "excess_final": excess[-1],
        "excess_ratio": (excess[-1] / excess[0]) if excess[0] > 0.0 else None,
        "half_life": next((index for index, value in enumerate(excess)
                           if value <= 0.5 * excess[0]), None) if excess[0] > 0.0 else None,
        "recovered_at": next((index for index, value in enumerate(curve)
                              if value <= 2.0 * floor[index]), None),
    }


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    weights = Fine15Weights.from_vhood(args.fine15.resolve(), device=device)
    teacher = Fine15(weights)
    args.mean, args.std = weights.normalizer("output")
    student = load_tinyhood(args.student.resolve(), device=device).eval()
    args.contact_offset = SolverConfig().contact_offset

    report = {"student": args.student.name, "steps": args.steps, "warmup": args.warmup,
              "frame_scales": args.frame_scales, "iterations": args.iterations,
              "b_iterations": args.b_iterations, "patch": args.patch,
              "stretch_amount": args.stretch_amount, "penetrate_depth": args.penetrate_depth,
              "scenes": {}}

    for scene_name in args.scenes:
        if not (args.scene_root / scene_name).is_dir():
            print(f"{scene_name}: not baked, skipped")
            continue
        scene = load_scene(scene_name, args)
        base = build_constraints(scene, scene.cloth_target(0))
        # Gate G0's winning `teacher` calibration, measured on this scene's own teacher rollout.
        reference = rollout(teacher.predict_graph, teacher, scene, args, frame_scale=1.0, hook=None)
        lengths = calibrate_from_trajectory(base.pairs, reference["positions"],
                                           skip=min(5, args.steps - 1)).clamp_min(1.0e-9)
        base = dataclasses.replace(base, target_length=lengths)

        entry: dict = {"constraints": base.count, "speeds": {}}
        for frame_scale in args.frame_scales:
            probe = position_probe(scene, base, student, teacher, args, frame_scale)
            entry["speeds"][f"{frame_scale:g}x"] = probe
        report["scenes"][scene_name] = entry
        print()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


def window_metrics(run: dict, warmup: int) -> dict:
    """Pose-quality summary over the observed window, which needs the max and not only the end.

    `edge_p95_end` alone is confounded as soon as `--frame-scales` exceeds 1: a 137-frame clip played
    at 3x is exhausted by step 46, after which `make_graph` clamps to the last frame and the rest of
    the rollout is a static settle in which the constraints relax everything. Measured on
    `hml_001962`, branch C's end-of-run edge P95 *improves* from 1.344 at 1x to 1.181 at 2x purely
    because more of the window is settle. The max over the window is not fooled by that, because a
    blow-up does not heal -- branch A reads 16.664 at 2x at both the peak and the end.
    """
    window = slice(warmup, len(run["edge"]))
    edge = run["edge"][window]
    return {
        "edge_p95_end": run["edge"][-1] if run["edge"] else None,
        "edge_p95_max": max(edge, default=None),
        "edge_p95_at": {str(at): (run["edge"][at - 1] if at <= len(run["edge"]) else None)
                        for at in (30, 60, 90, 120)},
        "flipped_max": max(run["flipped"][window], default=0.0),
        "pierced_max": max(run["pierced"][window], default=0),
        "depth_max": max(run["depth"][window], default=0.0),
        "completed_steps": len(run["positions"]),
    }


def position_probe(scene, constraints, student, teacher, args, frame_scale: float) -> dict:
    """Every branch x corruption at one playback speed."""
    proxy = scene.proxy(min(frame_of(args.warmup, frame_scale), scene.frame_count - 1))
    # After this step the clip is exhausted and make_graph clamps to the last frame, so the remainder
    # of the rollout is a static settle. Recorded because it decides how the speed rows may be read.
    exhausted = next((step for step in range(args.steps)
                      if frame_of(step, frame_scale) >= scene.frame_count - 1), None)
    # "none" is an uncorrupted rerun. It has to go first because its curve is the noise floor every
    # other row is judged against -- without it a ratio of 0.9 cannot be told from a rounding artifact.
    order = (["none"] if "none" in args.corruptions else []) + \
            [name for name in args.corruptions if name != "none"]
    result: dict = {"clip_exhausted_at": exhausted}
    print(f"\n  speed={frame_scale:g}x (clip ends step {exhausted})  {'arm':3s} {'corruption':10s} "
          f"{'inject':>8s} {'xs0':>8s} {'xsEnd':>8s} {'xsRatio':>7s} {'half':>4s} {'back':>4s} "
          f"{'iso':>7s} {'edgeMAX':>8s} {'(clean)':>8s} {'edgeEnd':>8s} {'flip':>6s} {'(clean)':>7s} "
          f"{'pierce':>6s} {'(clean)':>7s}")
    for branch in args.branches:
        config = SolverConfig(
            iterations=args.b_iterations if branch == "B" else args.iterations,
            mode="standard", sweep=args.sweep,
            stretch_compliance=args.stretch_compliance, bend_compliance=args.bend_compliance,
            one_sided=bool(args.one_sided), relaxation=args.relaxation, collision=True,
        )
        gravity = torch.tensor(args.gravity, dtype=torch.float32,
                               device=scene.cloth_rest.device).reshape(1, 3)

        def arm():
            # A fresh BallisticGravity per rollout: it carries a step counter that must not leak.
            predictor = BallisticGravity(gravity, args.mean, args.std) if branch == "B" else student
            hook = None if branch == "A" else make_hook(
                scene, constraints, config, network=branch != "B", frame_scale=frame_scale)
            return predictor, hook

        predictor, hook = arm()
        clean = rollout(predictor, teacher, scene, args, frame_scale=frame_scale, hook=hook)
        reference = window_metrics(clean, args.warmup)
        branch_result: dict = {"iterations": config.iterations, "clean": reference, "corruptions": {}}
        noise = None
        for name in order:
            damage = None if name == "none" else \
                (lambda position, fn=CORRUPTIONS[name]: fn(scene, position, proxy, args))
            predictor, hook = arm()
            dirty = rollout(predictor, teacher, scene, args, frame_scale=frame_scale,
                            hook=hook, damage=damage)
            forget = forgetting(clean, dirty, args.warmup, noise)
            if name == "none":
                noise = forget["curve"]
            iso = None
            if name in ("fold", "penetrate"):
                start = scene.cloth_target(0) if args.warmup == 0 else clean["positions"][args.warmup - 1]
                iso = isometry_error(start, damage(start), geodesic_patch(scene, start, args.patch))
            forget.update(window_metrics(dirty, args.warmup))
            forget["isometry_error"] = iso
            forget["injected_rms"] = dirty["injected_rms"]
            branch_result["corruptions"][name] = forget
            print(f"  {'':24s}  {branch:3s} {name:10s} "
                  f"{_fmt(dirty['injected_rms'])} "
                  f"{_fmt(forget['excess_initial'])} {_fmt(forget['excess_final'])} "
                  f"{_fmt(forget['excess_ratio'], 7, 2)} {str(forget['half_life']):>4s} "
                  f"{str(forget['recovered_at']):>4s} {_fmt(iso, 7, 0, exp=True)} "
                  f"{_fmt(forget['edge_p95_max'], 8, 3)} {_fmt(reference['edge_p95_max'], 8, 3)} "
                  f"{_fmt(forget['edge_p95_end'], 8, 3)} "
                  f"{_fmt(forget['flipped_max'], 6, 3)} {_fmt(reference['flipped_max'], 7, 3)} "
                  f"{forget['pierced_max']:6d} {reference['pierced_max']:7d}", flush=True)
        result[branch] = branch_result
    return result


def _fmt(value, width: int = 8, digits: int = 5, *, exp: bool = False) -> str:
    if value is None:
        return "n/a".rjust(width)
    return f"{value:{width}.{digits}{'e' if exp else 'f'}}"


if __name__ == "__main__":
    raise SystemExit(main())
