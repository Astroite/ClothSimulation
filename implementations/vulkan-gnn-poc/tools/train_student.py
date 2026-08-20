#!/usr/bin/env python3
"""Distil a HOOD-compatible student that stays stable in closed loop.

This supersedes tools/train_tinyhood.py, which is kept as-is because it reproduces the
shipped 64x4 weights documented in results/TINYHOOD_64X4_RESULTS.md. That recipe fit a
single-step acceleration regression on 48 states from one trajectory, and the resulting
student diverged within five steps (edge ratio P95 6.49). Low teacher-forced error does not
imply closed-loop stability: HOOD integrates positions to second order with no damping, so a
small persistent acceleration bias accumulates roughly as eps * n^2 / 2, and once the student
leaves the states it was fitted on the error grows on its own.

What is different here:

* Two phases. Phase 1 fits normalised acceleration on many states -- cheap, one forward and
  backward per sample, and it gets the network into the right basin. Phase 2 unrolls the
  student for several steps and backpropagates through the whole chain against the teacher's
  trajectory, which is what actually penalises accumulation.
* Noise injection with teacher relabelling. States are perturbed and the teacher is asked
  what it would do from the perturbed state, so the student learns to recover instead of only
  seeing the teacher's own clean trajectory.
* All four baked scenes, including hml_001962 (137 frames), which no previous run used.
* Geometry penalties as divergence guards only, plus a degenerate-area guard.
* Model selection on a closed-loop rollout compared against the teacher's own rollout.

Everything above produced the 32x12 weights documented in results/STUDENT_32X12_RESULTS.md,
which hold teacher-level structure for about 119 steps and then drift: edge P95 crosses 3.0 at
step 180. Three things in that recipe caused it, all fixed here:

* The reference for both geometry penalties was the REST mesh, but the teacher's own trajectory
  sits at edge P95 1.8 and has 12.4% of triangles flipped relative to rest, because the skirt
  genuinely folds and hangs. `edge_penalty(tolerance=0.1)` therefore punished any edge past
  1.1x rest and `flip_penalty` punished any fold -- both fought the supervision signal and are
  the direct cause of the "stiffer than the teacher" gap. Edge length is now a wide guard band
  (`--edge-lower/--edge-upper`) that only fires on real divergence, flip-versus-rest is off by
  default, and the genuine orientation-free failure -- triangles collapsing to zero area --
  gets its own term.
* Training states came only from the teacher's trajectory perturbed by at most 3 mm, while the
  student is 0.79 m away from the teacher by step 120. It was asked to be accurate 260x closer
  to the teacher than where it actually runs. `--dagger-rounds` now rolls the CURRENT student
  out, relabels its own states with the teacher, and trains on those. The earlier attempt at
  this in train_tinyhood.py made things worse, because that student was already broken within
  five steps so its states were garbage the teacher could not sensibly label; states whose
  edge P95 already exceeds `--dagger-max-edge-p95` are dropped for exactly that reason.
* Model selection scored `|edge_p95 - 1| + 2 * flipped` at the final step, i.e. against the
  rest mesh again, so a stiff student outscored a faithful one -- and it looked only at the end
  state, which says nothing about when the curve turns up. Selection now compares the whole
  curve against the teacher's curve and adds the position error against the teacher's
  trajectory, so being stiffer than the teacher costs something instead of paying.

Determinism note: aggregate_sum and vertex_normals use index_add_, which has no deterministic
CUDA implementation for float, so torch.use_deterministic_algorithms(True) cannot be enabled
here. Runs are seeded and reproducible up to float accumulation order.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

POC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC_ROOT))

from real_scene.fine15 import Fine15, Fine15Graph, Fine15Weights  # noqa: E402
from real_scene.formats import sha256_file  # noqa: E402
from real_scene.runtime_scene import RuntimeScene  # noqa: E402
from real_scene.tinyhood import TinyHood, export_tinyhood, load_tinyhood  # noqa: E402


@dataclass(frozen=True)
class SceneSpec:
    key: str
    root: Path
    motion: str
    asset_stem: str = "ch10032"
    # Static scenes have a single animation frame but still evolve the cloth autoregressively,
    # so they can supply as many states as asked for.
    steps: int = 0


@dataclass
class LoadedScene:
    spec: SceneSpec
    scene: RuntimeScene


@dataclass
class StudentSample:
    scene_key: str
    step: int
    position: torch.Tensor
    previous: torch.Tensor
    target_normalized: torch.Tensor
    noise_sigma: float
    # Teacher positions for the `rollout_steps` steps that follow this state. Only populated
    # for the subset used by phase 2.
    teacher_rollout: torch.Tensor | None = None
    # True for states visited by the student itself rather than by the teacher. These are the
    # states the student actually has to be accurate on, so they are tracked separately in the
    # dataset statistics.
    on_policy: bool = False


@dataclass
class EpochRecord:
    phase: str
    epoch: int
    loss: float
    seconds: float
    evaluation: dict = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--latent", type=int, default=32, choices=(32, 64))
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260818)

    parser.add_argument("--phase1-epochs", type=int, default=15)
    parser.add_argument("--phase1-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--phase2-epochs", type=int, default=8)
    parser.add_argument("--phase2-learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--batch", type=int, default=4, help="samples accumulated per optimiser step")
    parser.add_argument(
        "--no-normalise-fit",
        dest="normalise_fit",
        action="store_false",
        help="weight the acceleration loss by raw magnitude instead of relative error. This "
             "reproduces the weighting that produced the shipped 32x12 weights, where a single "
             "step-0 sample carried 28.8x the target energy of a steady-state step and the steady "
             "regime ended up at 0.357 variance explained",
    )
    parser.set_defaults(normalise_fit=True)

    parser.add_argument("--rollout-steps", type=int, default=6, help="phase 2 unroll depth")
    parser.add_argument("--rollout-samples", type=int, default=120, help="states given a teacher continuation")
    parser.add_argument("--trajectory-steps", type=int, default=60, help="teacher steps taken per scene")
    parser.add_argument(
        "--noise-sigma",
        type=float,
        nargs="*",
        default=[0.0, 0.001, 0.003],
        help="per-vertex position noise in metres; 0 keeps the clean state",
    )
    parser.add_argument(
        "--early-steps",
        type=int,
        default=3,
        help="how many leading steps count as the transient regime",
    )
    parser.add_argument(
        "--early-step-repeats",
        type=int,
        default=8,
        help="extra noisy draws per leading step; they are a different regime and rare",
    )

    parser.add_argument("--edge-weight", type=float, default=0.1, help="edge-length guard weight")
    parser.add_argument(
        "--edge-lower",
        type=float,
        default=0.4,
        help="edges shorter than this fraction of rest length are penalised",
    )
    parser.add_argument(
        "--edge-upper",
        type=float,
        default=2.5,
        help="edges longer than this multiple of rest length are penalised. The teacher's own "
             "trajectory reaches edge P95 1.8, so anything near 1.1 penalises correct behaviour",
    )
    parser.add_argument(
        "--flip-weight",
        type=float,
        default=0.0,
        help="weight of the flip-versus-REST-orientation penalty. Off by default: the teacher "
             "flips 12.4%% of triangles relative to rest at 120 steps because the skirt folds",
    )
    parser.add_argument(
        "--degenerate-weight",
        type=float,
        default=0.05,
        help="weight of the collapsed-triangle-area penalty, which is orientation-free",
    )
    parser.add_argument(
        "--degenerate-floor",
        type=float,
        default=0.1,
        help="triangle area below this fraction of its rest area is penalised",
    )
    parser.add_argument("--accel-weight", type=float, default=1.0, help="phase 2 single-step anchor weight")
    parser.add_argument(
        "--edge-match-weight",
        type=float,
        default=0.0,
        help="phase 2 weight on matching the teacher's edge lengths. Targets the accumulating "
             "stretch directly, which plain position MSE buries under bulk motion",
    )

    parser.add_argument("--eval-steps", type=int, default=150, help="closed-loop steps used for model selection")
    parser.add_argument("--final-eval-steps", type=int, default=240)
    parser.add_argument(
        "--stiff-weight",
        type=float,
        default=0.25,
        help="selection weight on being LESS stretched than the teacher. Non-zero so an "
             "over-damped student cannot win, lower than 1.0 because it is a fidelity loss "
             "rather than a divergence",
    )
    parser.add_argument(
        "--drift-weight",
        type=float,
        default=2.0,
        help="selection weight on RMS position error against the teacher's trajectory, in metres",
    )
    parser.add_argument(
        "--over-cap",
        type=float,
        default=2.0,
        help="cap each step's contribution to the selection score. Once a rollout has diverged "
             "its tail is chaotic, and index_add_ is nondeterministic on CUDA, so the same "
             "weights scored 0.377 and 0.436 on two runs. Capping makes the score read as 'how "
             "long until divergence' instead of 'how violent was the divergence'",
    )

    parser.add_argument(
        "--dagger-rounds",
        type=int,
        default=0,
        help="on-policy rounds after phase 2: roll the student out, relabel with the teacher, "
             "retrain. This is the fix for the 260x gap between the training noise and the "
             "distance the student actually drifts",
    )
    parser.add_argument("--dagger-steps", type=int, default=200, help="student rollout horizon per round")
    parser.add_argument("--dagger-stride", type=int, default=4, help="keep every Nth state of that rollout")
    parser.add_argument(
        "--dagger-max-edge-p95",
        type=float,
        default=2.6,
        help="drop on-policy states already stretched beyond this. The teacher tops out near "
             "1.8; past this the state is one the student must never reach and the teacher's "
             "label for it is itself out of distribution",
    )
    parser.add_argument("--dagger-epochs", type=int, default=6, help="single-step epochs per round")
    parser.add_argument("--dagger-rollout-epochs", type=int, default=6, help="unrolled epochs per round")
    parser.add_argument("--dagger-learning-rate", type=float, default=1.0e-4)
    parser.add_argument(
        "--dagger-rollout-fraction",
        type=float,
        default=0.5,
        help="fraction of on-policy states given a teacher continuation for the unrolled loss",
    )

    parser.add_argument("--fine15", type=Path, default=POC_ROOT / ".work/hood_data/fine15.vhood")
    parser.add_argument("--scene-root", type=Path, default=POC_ROOT / ".work/real_scene")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="start from this .pt checkpoint instead of a fresh initialisation. Phase 1 on a "
             "converged model is wasted time; pass --phase1-epochs 0 with this",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="suffix for the derived checkpoint/output/report names, so a new run does not "
             "overwrite weights that are already verified and benchmarked",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def scene_specs(root: Path, trajectory_steps: int) -> list[SceneSpec]:
    return [
        SceneSpec("ch10032_sprint", root / "ch10032_sprint", "ch10032_sprint"),
        SceneSpec("ch10032_tpose", root / "ch10032_tpose", "ch10032_tpose", steps=trajectory_steps),
        SceneSpec("hml_001962", root / "hml_001962", "hml_001962"),
        SceneSpec("hood_grid64", root / "hood_grid64", "hood_grid64", "hood_grid64", steps=trajectory_steps),
    ]


def frame_of(step: int, frame_scale: float = 1.0) -> int:
    """Animation frame driven by simulation step.

    `frame_scale > 1` plays the motion faster without changing the timestep, so the body covers
    more ground per solver step. That is the stress axis tools/recovery_probe.py sweeps: the
    complaint against an authored solver is that it fails on fast motion, and a single clip gives
    only a yes/no answer. Scaling the clip instead of the timestep is the honest form of the test --
    scaling dt would hand every solver proportionally more budget and measure nothing.

    `frame_scale == 1.0` returns `step` itself, so the default path is bit-identical to the
    original expression and every existing golden still reproduces.
    """
    return step if frame_scale == 1.0 else int(step * frame_scale)


def make_graph(
    builder: Fine15,
    scene: RuntimeScene,
    position: torch.Tensor,
    previous: torch.Tensor,
    step: int,
    frame_scale: float = 1.0,
) -> Fine15Graph:
    """Build the HOOD graph for one step, matching run_fine15_reference exactly."""
    # The obstacle sits on this step's frame and the target on the *next step's* frame, which under
    # a scaled clip is `frame_scale` frames later rather than one -- otherwise the body would be
    # asked to reach a half-step-ahead pose and the contact set would lag the motion.
    target_frame = min(frame_of(step + 1, frame_scale), scene.frame_count - 1)
    obstacle_frame = min(frame_of(step, frame_scale), scene.frame_count - 1)
    obstacle_position, obstacle_normals = scene.proxy(obstacle_frame)
    obstacle_target, _ = scene.proxy(target_frame)
    return builder.prepare_graph(
        position=position,
        previous=previous,
        rest_position=scene.cloth_rest,
        triangles=scene.cloth_triangles,
        mesh_senders=scene.cloth_senders,
        mesh_receivers=scene.cloth_receivers,
        mass=scene.cloth_mass,
        pin_mask=scene.cloth_pins,
        pin_target=scene.cloth_target(target_frame),
        obstacle_position=obstacle_position,
        obstacle_target=obstacle_target,
        obstacle_normals=obstacle_normals,
        timestep=1.0 / 3.0 if step == 0 else 1.0 / 30.0,
    )


def integrate(graph: Fine15Graph, normalized: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Second-order HOOD position update. Kept out-of-place so it is safe to backprop through."""
    acceleration = normalized * std + mean
    velocity = graph.effective_position - graph.effective_previous + acceleration
    predicted = graph.effective_position + velocity
    return torch.where(graph.pin_mask.unsqueeze(-1), graph.pin_target, predicted)


def advance(
    predictor,
    builder: Fine15,
    scene: RuntimeScene,
    position: torch.Tensor,
    previous: torch.Tensor,
    step: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    frame_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, Fine15Graph, torch.Tensor]:
    """One simulation step. Returns (next position, next previous, graph, normalised output)."""
    graph = make_graph(builder, scene, position, previous, step, frame_scale)
    normalized = predictor(graph)
    predicted = integrate(graph, normalized, mean, std)
    # The reference runtime carries the effective (pin-corrected) position forward, and the
    # very first step is a 1/3 s settle whose "previous" is the new position itself.
    next_previous = predicted if step == 0 else graph.effective_position
    return predicted, next_previous, graph, normalized


def teacher_trajectory(
    teacher: Fine15,
    scene: RuntimeScene,
    steps: int,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
    """Autoregressive teacher states: the distribution the student must be accurate on."""
    position = scene.cloth_target(0)
    previous = position.clone()
    states: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        for step in range(steps):
            states.append((step, position.clone(), previous.clone()))
            position, previous, _, _ = advance(
                teacher.predict_graph, teacher, scene, position, previous, step, mean, std
            )
            if not torch.isfinite(position).all():
                break
    return states


def perturb(
    position: torch.Tensor,
    previous: torch.Tensor,
    pin_mask: torch.Tensor,
    sigma: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Jitter a state's free vertices. Pinned vertices are overwritten by the pin target anyway."""
    if sigma <= 0.0:
        return position.clone(), previous.clone()
    free = (~pin_mask).unsqueeze(-1)
    noise = torch.randn(position.shape, generator=generator, device=position.device, dtype=position.dtype) * sigma
    # Perturbing position alone would also perturb the implied velocity by the same amount.
    # Applying an independent, smaller jitter to `previous` decorrelates the two so the student
    # sees both displaced positions and displaced velocities.
    velocity_noise = torch.randn(position.shape, generator=generator, device=position.device, dtype=position.dtype) * (sigma * 0.5)
    return position + noise * free, previous + (noise + velocity_noise) * free


def build_dataset(
    teacher: Fine15,
    scenes: list[LoadedScene],
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    sigmas: list[float],
    rollout_steps: int,
    rollout_samples: int,
    trajectory_steps: int,
    early_steps: int,
    early_step_repeats: int,
    generator: torch.Generator,
) -> tuple[list[StudentSample], dict]:
    samples: list[StudentSample] = []
    statistics: dict = {}
    started = time.perf_counter()
    for loaded in scenes:
        scene, spec = loaded.scene, loaded.spec
        steps = spec.steps or min(trajectory_steps, max(scene.frame_count - 1, 1))
        states = teacher_trajectory(teacher, scene, steps, mean, std)
        made = 0
        with torch.no_grad():
            for step, position, previous in states:
                # The leading steps are a different regime and are rare: step 0 runs a 1/3 s
                # settle against 1/30 s afterwards, and step 1 sits right after that jump. At
                # one sample per sigma they are under 2% of the set, yet every rollout starts
                # there, so their error is injected into everything downstream. Draw extra
                # noisy variants of them.
                repeats = early_step_repeats if step < early_steps else 1
                for sigma in sigmas:
                    # Repeating a clean state would just duplicate an identical sample.
                    draws = 1 if sigma == 0.0 else repeats
                    for _ in range(draws):
                        noisy_position, noisy_previous = perturb(position, previous, scene.cloth_pins, sigma, generator)
                        graph = make_graph(teacher, scene, noisy_position, noisy_previous, step)
                        target = teacher.predict_graph(graph)
                        if not torch.isfinite(target).all():
                            continue
                        samples.append(StudentSample(spec.key, step, noisy_position, noisy_previous, target, sigma))
                        made += 1
        statistics[spec.key] = {"teacher_states": len(states), "samples": made, "cloth_vertices": int(scene.cloth_rest.shape[0])}

    # Give a subset a teacher continuation for phase 2. Sampling across the whole set keeps
    # every scene and noise level represented in the unrolled objective.
    scene_by_key = {loaded.spec.key: loaded.scene for loaded in scenes}
    chosen = list(range(len(samples)))
    random.shuffle(chosen)
    chosen = chosen[: min(rollout_samples, len(chosen))]
    with torch.no_grad():
        for index in chosen:
            sample = samples[index]
            scene = scene_by_key[sample.scene_key]
            position, previous = sample.position, sample.previous
            future = []
            for offset in range(rollout_steps):
                position, previous, _, _ = advance(
                    teacher.predict_graph, teacher, scene, position, previous, sample.step + offset, mean, std
                )
                if not torch.isfinite(position).all():
                    break
                future.append(position)
            if len(future) == rollout_steps:
                sample.teacher_rollout = torch.stack(future)
    statistics["rollout_samples"] = sum(1 for sample in samples if sample.teacher_rollout is not None)
    statistics["total_samples"] = len(samples)
    statistics["generation_seconds"] = time.perf_counter() - started
    return samples, statistics


def edge_ratios(scene: RuntimeScene, position: torch.Tensor) -> torch.Tensor:
    rest = scene.cloth_rest[scene.cloth_senders] - scene.cloth_rest[scene.cloth_receivers]
    current = position[scene.cloth_senders] - position[scene.cloth_receivers]
    rest_length = torch.linalg.vector_norm(rest, dim=-1).clamp_min(1.0e-12)
    return torch.linalg.vector_norm(current, dim=-1) / rest_length


def edge_penalty(scene: RuntimeScene, position: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    """Penalise mesh edges outside a wide guard band around their rest length.

    This is deliberately NOT a shape prior. The band used to be rest length +/- 10%, but the
    teacher's own trajectory reaches edge P95 1.8 -- a hanging skirt is genuinely stretched --
    so that band penalised the network for reproducing its target and is why the 32x12 student
    came out stiffer than the teacher. The band now only fires on stretch the teacher never
    produces, leaving the shape entirely to the position loss.
    """
    ratio = edge_ratios(scene, position)
    excess = torch.relu(ratio - upper) + torch.relu(lower - ratio)
    return excess.square().mean()


def triangle_normals(scene: RuntimeScene, position: torch.Tensor) -> torch.Tensor:
    corners = position[scene.cloth_triangles]
    return torch.linalg.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])


def flip_penalty(scene: RuntimeScene, position: torch.Tensor) -> torch.Tensor:
    """Penalise triangles whose normal has turned away from its rest orientation.

    Kept for ablation but disabled by default (`--flip-weight 0`). The teacher has 12.4% of
    triangles flipped relative to rest after 120 steps because the skirt folds over itself, so
    this term penalises correct behaviour along with incorrect behaviour and cannot tell them
    apart. `degenerate_penalty` is the orientation-free half of what this was trying to catch.
    """
    rest_normal = torch.nn.functional.normalize(triangle_normals(scene, scene.cloth_rest), dim=-1)
    current_normal = torch.nn.functional.normalize(triangle_normals(scene, position), dim=-1)
    return torch.relu(-(rest_normal * current_normal).sum(dim=-1)).mean()


def degenerate_penalty(scene: RuntimeScene, position: torch.Tensor, floor: float) -> torch.Tensor:
    """Penalise triangles collapsing towards zero area.

    Unlike a flip, a collapsed triangle is a failure no matter how the cloth is folded: it means
    the three vertices became colinear, which the teacher never does. The cross product length
    is twice the area, so the ratio against the rest triangle is scale-free.
    """
    rest_area = torch.linalg.vector_norm(triangle_normals(scene, scene.cloth_rest), dim=-1).clamp_min(1.0e-12)
    current_area = torch.linalg.vector_norm(triangle_normals(scene, position), dim=-1)
    return torch.relu(floor - current_area / rest_area).square().mean()


def geometry_penalty(
    scene: RuntimeScene,
    position: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    """The guard terms, summed with their weights. Zero when all weights are zero."""
    total = position.new_zeros(())
    if args.edge_weight > 0.0:
        total = total + args.edge_weight * edge_penalty(scene, position, args.edge_lower, args.edge_upper)
    if args.flip_weight > 0.0:
        total = total + args.flip_weight * flip_penalty(scene, position)
    if args.degenerate_weight > 0.0:
        total = total + args.degenerate_weight * degenerate_penalty(scene, position, args.degenerate_floor)
    return total


def edge_match_loss(scene: RuntimeScene, position: torch.Tensor, teacher_position: torch.Tensor) -> torch.Tensor:
    """Match the teacher's edge LENGTHS, not just its vertex positions.

    Plain position MSE is dominated by bulk motion -- the whole skirt swinging a few centimetres
    differently -- which is not the failure mode. The failure is stretch accumulating under
    HOOD's undamped second-order integration, and stretch is exactly what the selection score's
    `over` term measures. Comparing edge lengths against the teacher's own edge lengths isolates
    that one degree of freedom and, unlike a rest-referenced penalty, it cannot bias the student
    towards being stiffer than the teacher: the target IS the teacher's stretch.
    """
    rest_length = torch.linalg.vector_norm(
        scene.cloth_rest[scene.cloth_senders] - scene.cloth_rest[scene.cloth_receivers], dim=-1
    ).clamp_min(1.0e-12)
    student = torch.linalg.vector_norm(position[scene.cloth_senders] - position[scene.cloth_receivers], dim=-1)
    target = torch.linalg.vector_norm(
        teacher_position[scene.cloth_senders] - teacher_position[scene.cloth_receivers], dim=-1
    )
    # Divide by rest length so a 1 mm error on a 6.5 mm edge counts like a 1 mm error on a
    # 50 mm edge would not; the short edges are where the collapse starts.
    return ((student - target) / rest_length).square().mean()


def teacher_forced_fit(
    model: TinyHood,
    samples: list[StudentSample],
    scene_by_key: dict[str, RuntimeScene],
    builder: Fine15,
    steady_from: int = 10,
) -> dict:
    """Fraction of the teacher's normalised acceleration variance the student reproduces.

    Loss alone hides how far there is left to go: the target's own second moment is the score a
    model that predicts nothing would get, so the ratio is the interpretable number.

    Reported three ways, because the aggregate is genuinely misleading on its own. `variance_
    explained` pools the squared errors before dividing, so it is weighted by target energy -- and
    step 0's target carries 28.8x the energy of a steady-state step, letting the transient dominate
    a number that reads like a whole-trajectory score. The shipped 32x12 student scored 0.980 that
    way while explaining only 0.357 of the variance over steps 10-39. `variance_explained_mean`
    weights every state equally, and the two regime breakdowns show where the error actually is.
    """
    pooled_residual = pooled_total = 0.0
    per_sample: list[float] = []
    regimes: dict[str, list[float]] = {"first_step": [], "early": [], "steady": []}
    with torch.no_grad():
        for sample in samples:
            graph = make_graph(builder, scene_by_key[sample.scene_key], sample.position, sample.previous, sample.step)
            free = ~graph.pin_mask
            target = sample.target_normalized[free]
            difference = float((model(graph)[free] - target).square().mean())
            baseline = float(target.square().mean())
            pooled_residual += difference
            pooled_total += baseline
            explained = 1.0 - difference / max(baseline, 1.0e-12)
            per_sample.append(explained)
            key = "first_step" if sample.step == 0 else ("steady" if sample.step >= steady_from else "early")
            regimes[key].append(explained)

    def average(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    return {
        "variance_explained": 1.0 - pooled_residual / max(pooled_total, 1.0e-12),
        "variance_explained_mean": average(per_sample),
        "first_step_variance_explained": average(regimes["first_step"]),
        "early_variance_explained": average(regimes["early"]),
        "steady_variance_explained": average(regimes["steady"]),
        "steady_from_step": steady_from,
        "samples": len(samples),
    }


def structure_metrics(scene: RuntimeScene, position: torch.Tensor, pin_target: torch.Tensor) -> dict:
    finite = torch.isfinite(position).all(dim=1)
    safe = torch.nan_to_num(position)
    ratios = edge_ratios(scene, safe)
    rest_normal = triangle_normals(scene, scene.cloth_rest)
    current_normal = triangle_normals(scene, safe)
    areas = torch.linalg.vector_norm(current_normal, dim=-1) / torch.linalg.vector_norm(rest_normal, dim=-1).clamp_min(1.0e-12)
    pin_error = (safe[scene.cloth_pins] - pin_target[scene.cloth_pins]).abs()
    return {
        "invalid_vertices": int((~finite).sum().item()),
        "pin_max_abs": float(pin_error.max().item()) if pin_error.numel() else 0.0,
        "edge_ratio_mean": float(ratios.mean().item()),
        "edge_ratio_p95": float(torch.quantile(ratios, 0.95).item()),
        "edge_ratio_max": float(ratios.max().item()),
        "collapsed_fraction_lt_0_5": float((ratios < 0.5).float().mean().item()),
        "stretched_fraction_gt_1_5": float((ratios > 1.5).float().mean().item()),
        "area_ratio_mean": float(areas.mean().item()),
        "flipped_fraction": float(((rest_normal * current_normal).sum(dim=-1) < 0.0).float().mean().item()),
    }


def curve_point(scene: RuntimeScene, position: torch.Tensor) -> tuple[float, float]:
    """The two per-step quantities the selection score needs. Cheaper than structure_metrics."""
    ratios = edge_ratios(scene, position)
    rest_normal = triangle_normals(scene, scene.cloth_rest)
    current_normal = triangle_normals(scene, position)
    flipped = ((rest_normal * current_normal).sum(dim=-1) < 0.0).float().mean()
    return float(torch.quantile(ratios, 0.95).item()), float(flipped.item())


def rollout_curve(
    model,
    builder: Fine15,
    scene: RuntimeScene,
    steps: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    reference: torch.Tensor | None = None,
    store_positions: bool = False,
) -> dict:
    """Closed-loop rollout, recording the whole curve rather than only the end state.

    `model` is any callable taking a graph, so the teacher's bound predict_graph can be passed
    in to produce the reference curve this metric is measured against. The end state alone hides
    when the curve turns up, which is the only thing that distinguishes a student that survives
    120 steps from one that survives 360.
    """
    if hasattr(model, "eval"):
        model.eval()
    position = scene.cloth_target(0)
    previous = position.clone()
    edge_p95: list[float] = []
    flipped: list[float] = []
    reference_rms: list[float] = []
    reference_max: list[float] = []
    positions: list[torch.Tensor] = []
    completed = 0
    with torch.no_grad():
        for step in range(steps):
            position, previous, _, _ = advance(model, builder, scene, position, previous, step, mean, std)
            if not torch.isfinite(position).all():
                break
            completed += 1
            point = curve_point(scene, position)
            edge_p95.append(point[0])
            flipped.append(point[1])
            if store_positions:
                positions.append(position.clone())
            if reference is not None and step < reference.shape[0]:
                delta = position - reference[step]
                reference_rms.append(float(delta.square().mean().sqrt().item()))
                reference_max.append(float(delta.abs().max().item()))
    frame = min(completed, scene.frame_count - 1)
    result = {
        "requested_steps": steps,
        "completed_steps": completed,
        "edge_p95_curve": edge_p95,
        "flipped_curve": flipped,
    }
    result.update(structure_metrics(scene, position, scene.cloth_target(frame)))
    if reference is not None:
        result["teacher_position_rms_curve"] = reference_rms
        result["teacher_position_rms_mean"] = sum(reference_rms) / max(len(reference_rms), 1)
        result["teacher_position_max_abs"] = max(reference_max, default=0.0)
    if store_positions:
        result["positions"] = torch.stack(positions) if positions else None
    return result


def evaluate_rollout(
    model,
    builder: Fine15,
    scene: RuntimeScene,
    steps: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    teacher_positions: torch.Tensor | None = None,
) -> dict:
    """rollout_curve without the per-step curves, for the report's final table."""
    result = rollout_curve(model, builder, scene, steps, mean, std, reference=teacher_positions)
    return {key: value for key, value in result.items() if not key.endswith("_curve")}


def stability_score(trace: dict, teacher: dict, args: argparse.Namespace) -> float:
    """Lower is better. Measured against the TEACHER's curve, not against the rest mesh.

    The previous score was `|edge_p95 - 1| + 2 * flipped` at the final step. Both references
    were wrong: the teacher itself scores 0.986 on it (edge P95 1.8, 12.4% flipped) because a
    hanging skirt is stretched and folded, so a student stiffer than the teacher scored BETTER
    than the teacher. That is precisely the solution the 32x12 run converged to.

    Here, exceeding the teacher's stretch and falling short of it are both charged, asymmetric
    because over-stretch is the divergence and under-stretch is only a fidelity loss, and the
    RMS position error against the teacher's own trajectory is added so a smooth-but-wrong
    trajectory cannot win. A run that fails to complete is worse than any run that finishes.

    Every per-step term is capped: past divergence the rollout is chaotic and, because index_add_
    has no deterministic CUDA kernel, an uncapped score is not even repeatable for fixed weights
    (0.377 vs 0.436 on two evaluations of the same checkpoint). Capped, the score is dominated by
    HOW MANY steps stayed close to the teacher rather than by how violently the tail blew up,
    which is both the quantity that matters and a repeatable one.
    """
    if trace["invalid_vertices"] > 0 or trace["completed_steps"] < trace["requested_steps"]:
        return 1.0e6 + float(trace["requested_steps"] - trace["completed_steps"])
    count = min(len(trace["edge_p95_curve"]), len(teacher["edge_p95_curve"]))
    if count == 0:
        return 1.0e6
    cap = args.over_cap
    over = under = flip = drift = 0.0
    drift_curve = trace.get("teacher_position_rms_curve", [])
    for index in range(count):
        difference = trace["edge_p95_curve"][index] - teacher["edge_p95_curve"][index]
        over += min(max(difference, 0.0), cap)
        under += min(max(-difference, 0.0), cap)
        flip += max(trace["flipped_curve"][index] - teacher["flipped_curve"][index], 0.0)
        if index < len(drift_curve):
            drift += min(drift_curve[index], cap)
    return over / count + args.stiff_weight * (under / count) + 2.0 * (flip / count) + args.drift_weight * (drift / count)


def normalised_fit_loss(prediction: torch.Tensor, target: torch.Tensor, normalise: bool) -> torch.Tensor:
    """Squared error, optionally divided by the target's own second moment.

    This is the single most consequential weighting choice in the recipe. HOOD's first step is a
    1/3 s settle against 1/30 s afterwards, so the teacher's normalised acceleration at step 0
    carries 28.8x the energy of a steady-state step (0.4896 against 0.0170 on the tpose
    trajectory). An unnormalised mean-square loss therefore lets ONE step-0 sample contribute 37%
    of the total over a 40-state trajectory, and --early-step-repeats multiplied that by another
    eight. The 32x12 student trained that way reached 0.931 variance explained at step 0 and only
    0.357 over steps 10-39 -- the regime a 150-step rollout spends essentially all of its time in,
    and the regime whose error is what accumulates.

    Dividing by the target's second moment makes every sample contribute its RELATIVE error, so
    the transient and the steady state get equal weight regardless of scale, and the quantity being
    minimised matches the variance-explained metric the run is judged by.
    """
    error = (prediction - target).square().mean()
    if not normalise:
        return error
    return error / target.square().mean().clamp_min(1.0e-8)


def phase1_loss(
    model: TinyHood,
    sample: StudentSample,
    scene: RuntimeScene,
    builder: Fine15,
    mean: torch.Tensor,
    std: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    graph = make_graph(builder, scene, sample.position, sample.previous, sample.step)
    normalized = model(graph)
    free = ~graph.pin_mask
    loss = normalised_fit_loss(normalized[free], sample.target_normalized[free], args.normalise_fit)
    # Guard terms only make sense on an unperturbed state. Rest edges run from 0.0065 m
    # (1st percentile) to 0.05 m, so an injected jitter of a few millimetres already distorts
    # the shortest edges by tens of percent -- charging that would punish the student for noise
    # no single step can remove. Noisy samples are supervised purely by the teacher's relabelled
    # acceleration, and phase 2 supervises their geometry against the teacher's own recovery.
    if sample.noise_sigma == 0.0:
        loss = loss + geometry_penalty(scene, integrate(graph, normalized, mean, std), args)
    return loss


def phase2_loss(
    model: TinyHood,
    sample: StudentSample,
    scene: RuntimeScene,
    builder: Fine15,
    mean: torch.Tensor,
    std: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    """Unroll the student and compare its trajectory to the teacher's from the same state.

    Later steps are weighted more heavily than earlier ones: a single-step fit is already what
    phase 1 optimised, and the whole point here is the drift that only shows up further out.
    """
    position, previous = sample.position, sample.previous
    total = position.new_zeros(())
    weight_sum = 0.0
    for offset in range(sample.teacher_rollout.shape[0]):
        position, previous, graph, normalized = advance(
            model, builder, scene, position, previous, sample.step + offset, mean, std
        )
        if offset == 0 and args.accel_weight > 0.0:
            free = ~graph.pin_mask
            total = total + args.accel_weight * normalised_fit_loss(
                normalized[free], sample.target_normalized[free], args.normalise_fit
            )
        weight = float(offset + 1)
        total = total + weight * (position - sample.teacher_rollout[offset]).square().mean()
        if args.edge_match_weight > 0.0:
            total = total + weight * args.edge_match_weight * edge_match_loss(
                scene, position, sample.teacher_rollout[offset]
            )
        total = total + weight * geometry_penalty(scene, position, args)
        weight_sum += weight
    return total / max(weight_sum, 1.0)


def collect_on_policy(
    model: TinyHood,
    teacher: Fine15,
    scenes: list[LoadedScene],
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    steps: int,
    stride: int,
    max_edge_p95: float,
    rollout_steps: int,
    rollout_fraction: float,
) -> tuple[list[StudentSample], dict]:
    """Roll the CURRENT student out and relabel the states it actually visits with the teacher.

    This closes the gap the noise injection cannot: sigma tops out at 3 mm while the student is
    0.79 m from the teacher by step 120, so every state it is scored on late in a rollout is one
    it was never trained on. train_tinyhood.py tried this and made things worse, but that
    student was already broken by step 5, so the states it contributed were degenerate cloth the
    teacher has never seen either and cannot label meaningfully. The `max_edge_p95` filter is
    what makes the idea safe: past the teacher's own envelope, the label is not trustworthy and
    the state is one the student must never reach anyway, so capacity spent on it is wasted.
    """
    model.eval()
    samples: list[StudentSample] = []
    statistics: dict = {"kept": 0, "rejected_stretched": 0, "diverged_scenes": []}
    for loaded in scenes:
        scene, spec = loaded.scene, loaded.spec
        horizon = steps if scene.frame_count == 1 else min(steps, scene.frame_count - 1)
        position = scene.cloth_target(0)
        previous = position.clone()
        kept = 0
        with torch.no_grad():
            for step in range(horizon):
                if step % stride == 0:
                    stretch, _ = curve_point(scene, position)
                    if stretch <= max_edge_p95:
                        graph = make_graph(teacher, scene, position, previous, step)
                        target = teacher.predict_graph(graph)
                        if torch.isfinite(target).all():
                            samples.append(StudentSample(
                                spec.key, step, position.clone(), previous.clone(), target, 0.0, on_policy=True
                            ))
                            kept += 1
                    else:
                        statistics["rejected_stretched"] += 1
                position, previous, _, _ = advance(model, teacher, scene, position, previous, step, mean, std)
                if not torch.isfinite(position).all():
                    statistics["diverged_scenes"].append({"scene": spec.key, "step": step})
                    break
        statistics[spec.key] = {"horizon": horizon, "kept": kept}
        statistics["kept"] += kept

    # A teacher continuation from a state the student reached on its own is the recovery signal:
    # it says where the teacher would have taken THIS state, not where the teacher's own clean
    # trajectory happens to be.
    scene_by_key = {loaded.spec.key: loaded.scene for loaded in scenes}
    chosen = list(range(len(samples)))
    random.shuffle(chosen)
    chosen = chosen[: int(len(chosen) * rollout_fraction)]
    with torch.no_grad():
        for index in chosen:
            sample = samples[index]
            scene = scene_by_key[sample.scene_key]
            position, previous = sample.position, sample.previous
            future = []
            for offset in range(rollout_steps):
                position, previous, _, _ = advance(
                    teacher.predict_graph, teacher, scene, position, previous, sample.step + offset, mean, std
                )
                if not torch.isfinite(position).all():
                    break
                future.append(position)
            if len(future) == rollout_steps:
                sample.teacher_rollout = torch.stack(future)
    statistics["with_continuation"] = sum(1 for sample in samples if sample.teacher_rollout is not None)
    return samples, statistics


def run_phase(
    name: str,
    model: TinyHood,
    samples: list[StudentSample],
    scene_by_key: dict[str, RuntimeScene],
    builder: Fine15,
    loss_fn,
    *,
    epochs: int,
    learning_rate: float,
    batch: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    args: argparse.Namespace,
    evaluate,
    history: list[EpochRecord],
    best: dict,
) -> None:
    if epochs <= 0 or not samples:
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    order = list(range(len(samples)))
    for epoch in range(epochs):
        started = time.perf_counter()
        random.shuffle(order)
        model.train()
        total = 0.0
        counted = 0
        optimizer.zero_grad(set_to_none=True)
        for position_in_epoch, index in enumerate(order):
            sample = samples[index]
            loss = loss_fn(model, sample, scene_by_key[sample.scene_key], builder, mean, std, args)
            if not torch.isfinite(loss):
                continue
            (loss / batch).backward()
            total += float(loss.item())
            counted += 1
            if (position_in_epoch + 1) % batch == 0 or position_in_epoch + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        evaluation = evaluate(model)
        record = EpochRecord(name, epoch + 1, total / max(counted, 1), time.perf_counter() - started, evaluation)
        history.append(record)
        score = evaluation["score"]
        if score < best["score"]:
            best["score"] = score
            best["state"] = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best["phase"] = name
            best["epoch"] = epoch + 1
        selected = evaluation["ch10032_tpose"]
        print(json.dumps({
            "phase": name,
            "epoch": epoch + 1,
            "loss": record.loss,
            "seconds": round(record.seconds, 1),
            "var_pooled": round(evaluation["fit"]["variance_explained"], 4),
            "var_mean": round(evaluation["fit"]["variance_explained_mean"], 4)
                if evaluation["fit"]["variance_explained_mean"] is not None else None,
            "var_step0": round(evaluation["fit"]["first_step_variance_explained"], 4)
                if evaluation["fit"]["first_step_variance_explained"] is not None else None,
            "var_steady": round(evaluation["fit"]["steady_variance_explained"], 4)
                if evaluation["fit"]["steady_variance_explained"] is not None else None,
            "tpose_edge_p95": round(selected["edge_ratio_p95"], 4),
            "tpose_flipped": round(selected["flipped_fraction"], 4),
            "drift_rms": round(selected.get("teacher_position_rms_mean", 0.0), 4),
            "drift_max": round(selected.get("teacher_position_max_abs", 0.0), 4),
            "completed": selected["completed_steps"],
            "score": round(score, 5),
            "best_score": round(best["score"], 5),
        }, sort_keys=True), flush=True)


def main() -> int:
    args = parse_args()
    if args.blocks < 1 or args.blocks > 15:
        raise ValueError("blocks must be between 1 and 15 to fit the Vulkan timestamp schedule")
    label = f"{args.latent}x{args.blocks}{args.tag}"
    args.checkpoint = args.checkpoint or POC_ROOT / f".work/hood_data/student{label}.pt"
    args.output = args.output or POC_ROOT / f".work/hood_data/student{label}.vhood"
    args.report = args.report or POC_ROOT / f"results/student{label}_python.json"

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    weights = Fine15Weights.from_vhood(args.fine15.resolve(), device=device)
    teacher = Fine15(weights)
    mean, std = weights.normalizer("output")

    scenes: list[LoadedScene] = []
    for spec in scene_specs(args.scene_root.resolve(), args.trajectory_steps):
        scenes.append(LoadedScene(spec, RuntimeScene.load(spec.root, spec.motion, device=device, asset_stem=spec.asset_stem)))
    scene_by_key = {loaded.spec.key: loaded.scene for loaded in scenes}

    samples, data_statistics = build_dataset(
        teacher,
        scenes,
        mean=mean,
        std=std,
        sigmas=list(args.noise_sigma),
        rollout_steps=args.rollout_steps,
        rollout_samples=args.rollout_samples,
        trajectory_steps=args.trajectory_steps,
        early_steps=args.early_steps,
        early_step_repeats=args.early_step_repeats,
        generator=generator,
    )
    print(json.dumps({"dataset": data_statistics}, indent=2, sort_keys=True), flush=True)

    model = TinyHood(latent=args.latent, blocks=args.blocks).to(device)
    resumed = None
    if args.resume is not None:
        state = torch.load(args.resume.resolve(), map_location=device, weights_only=False)
        architecture = state.get("architecture", {})
        if architecture.get("latent") != args.latent or architecture.get("blocks") != args.blocks:
            raise ValueError(
                f"--resume checkpoint is {architecture.get('latent')}x{architecture.get('blocks')}, "
                f"not {args.latent}x{args.blocks}"
            )
        model.load_state_dict(state["model"], strict=True)
        resumed = {"path": str(args.resume), "selected": state.get("selected")}
        print(json.dumps({"resumed": resumed}, sort_keys=True), flush=True)

    history: list[EpochRecord] = []
    best: dict = {"score": float("inf"), "state": None, "phase": None, "epoch": None}

    # The reference the selection score is measured against. Computed once: the teacher is 22x
    # the student's cost, so recomputing this per epoch would dominate the run.
    evaluation_scene = scene_by_key["ch10032_tpose"]
    teacher_reference = rollout_curve(
        teacher.predict_graph, teacher, evaluation_scene, args.eval_steps, mean, std, store_positions=True
    )
    teacher_positions = teacher_reference.pop("positions")
    print(json.dumps({"teacher_reference": {
        "edge_p95_at": {str(at): round(teacher_reference["edge_p95_curve"][at - 1], 4)
                        for at in (5, 30, 60, 120, args.eval_steps) if at <= len(teacher_reference["edge_p95_curve"])},
        "flipped_final": round(teacher_reference["flipped_fraction"], 4),
        "completed_steps": teacher_reference["completed_steps"],
    }}, sort_keys=True), flush=True)

    def evaluate(current: TinyHood) -> dict:
        trace = rollout_curve(
            current, teacher, evaluation_scene, args.eval_steps, mean, std, reference=teacher_positions
        )
        return {
            "ch10032_tpose": trace,
            "score": stability_score(trace, teacher_reference, args),
            "fit": teacher_forced_fit(current, fit_probe, scene_by_key, teacher),
        }

    # A small fixed probe of clean states, biased to include the leading steps, so the fit
    # metric is comparable across epochs without costing a full pass.
    clean = [sample for sample in samples if sample.noise_sigma == 0.0]
    early = [sample for sample in clean if sample.step < args.early_steps]
    # Strided across the clean states so the steady regime is represented; the earlier version
    # took `early` plus a stride and so reported a number dominated by the transient.
    fit_probe = early + clean[:: max(len(clean) // 60, 1)]

    if resumed is not None:
        baseline = evaluate(model)
        best.update({
            "score": baseline["score"],
            "state": {key: value.detach().clone() for key, value in model.state_dict().items()},
            "phase": "resumed",
            "epoch": 0,
        })
        # Seed `best` with the resumed model's own score so a round of retraining can only ship
        # weights that beat what was already there.
        print(json.dumps({"resumed_score": round(baseline["score"], 5),
                          "resumed_edge_p95": round(baseline["ch10032_tpose"]["edge_ratio_p95"], 4),
                          "resumed_drift_rms": round(baseline["ch10032_tpose"]["teacher_position_rms_mean"], 4)},
                         sort_keys=True), flush=True)

    training_started = time.perf_counter()
    run_phase(
        "phase1_single_step", model, samples, scene_by_key, teacher, phase1_loss,
        epochs=args.phase1_epochs, learning_rate=args.phase1_learning_rate, batch=args.batch,
        mean=mean, std=std, args=args, evaluate=evaluate, history=history, best=best,
    )
    rollout_samples = [sample for sample in samples if sample.teacher_rollout is not None]
    run_phase(
        "phase2_rollout", model, rollout_samples, scene_by_key, teacher, phase2_loss,
        epochs=args.phase2_epochs, learning_rate=args.phase2_learning_rate, batch=args.batch,
        mean=mean, std=std, args=args, evaluate=evaluate, history=history, best=best,
    )

    dagger_statistics = []
    for round_index in range(args.dagger_rounds):
        # Collect from the best weights so far, not from whatever the last epoch left behind: a
        # round that overshot would otherwise poison the next round's state distribution.
        if best["state"] is not None:
            model.load_state_dict(best["state"], strict=True)
        fresh, statistics = collect_on_policy(
            model, teacher, scenes,
            mean=mean, std=std, steps=args.dagger_steps, stride=args.dagger_stride,
            max_edge_p95=args.dagger_max_edge_p95, rollout_steps=args.rollout_steps,
            rollout_fraction=args.dagger_rollout_fraction,
        )
        statistics["round"] = round_index + 1
        dagger_statistics.append(statistics)
        print(json.dumps({"dagger": statistics}, indent=2, sort_keys=True), flush=True)
        if not fresh:
            print(json.dumps({"dagger_stopped": "no usable on-policy states"}), flush=True)
            break
        samples.extend(fresh)
        run_phase(
            f"dagger{round_index + 1}_single_step", model, samples, scene_by_key, teacher, phase1_loss,
            epochs=args.dagger_epochs, learning_rate=args.dagger_learning_rate, batch=args.batch,
            mean=mean, std=std, args=args, evaluate=evaluate, history=history, best=best,
        )
        unrolled = [sample for sample in samples if sample.teacher_rollout is not None]
        run_phase(
            f"dagger{round_index + 1}_rollout", model, unrolled, scene_by_key, teacher, phase2_loss,
            epochs=args.dagger_rollout_epochs, learning_rate=args.dagger_learning_rate, batch=args.batch,
            mean=mean, std=std, args=args, evaluate=evaluate, history=history, best=best,
        )
    training_seconds = time.perf_counter() - training_started

    if best["state"] is not None:
        model.load_state_dict(best["state"], strict=True)
    model.eval()

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "architecture": {"node": 20, "mesh_edge": 12, "world_edge": 9, "latent": args.latent, "blocks": args.blocks},
        "seed": args.seed,
        "selected": {"phase": best["phase"], "epoch": best["epoch"], "score": best["score"]},
        "model": model.state_dict(),
    }, args.checkpoint.resolve())
    checkpoint_hash = sha256_file(args.checkpoint.resolve())
    export_info = export_tinyhood(model, weights, args.output.resolve(), checkpoint_sha256=checkpoint_hash)

    reloaded = load_tinyhood(args.output.resolve(), device=device)
    reloaded.eval()
    # Compare the parameters, not a forward pass. A VHOOD round-trip is exact for FP32, so the
    # tensors must match bit for bit -- a stricter test than any tolerance. Comparing forward
    # outputs instead would measure CUDA's nondeterministic index_add_ reduction order, which
    # drifts by ~1e-6 between two runs of identical weights and made this check fail spuriously.
    saved = model.state_dict()
    restored = reloaded.state_dict()
    if saved.keys() != restored.keys():
        raise ValueError("VHOOD reload produced a different parameter set")
    mismatched = {
        name: float((saved[name] - restored[name]).abs().max().item())
        for name in saved
        if not torch.equal(saved[name], restored[name])
    }
    if mismatched:
        raise ValueError(f"VHOOD reload is not bit-identical: {mismatched}")
    with torch.no_grad():
        probe = make_graph(teacher, scene_by_key[samples[0].scene_key], samples[0].position, samples[0].previous, samples[0].step)
        reload_max_abs = float((model(probe) - reloaded(probe)).abs().max().item())

    final = {}
    for key in ("ch10032_tpose", "hood_grid64", "ch10032_sprint", "hml_001962"):
        scene = scene_by_key[key]
        steps = args.final_eval_steps if scene.frame_count == 1 else min(args.final_eval_steps, scene.frame_count - 1)
        reference = teacher_positions if key == "ch10032_tpose" else None
        final[key] = evaluate_rollout(model, teacher, scene, steps, mean, std, reference)

    report = {
        "architecture": {"node": 20, "mesh_edge": 12, "world_edge": 9, "latent": args.latent, "blocks": args.blocks},
        "parameter_count": model.parameter_count,
        "fine15_packed_float_count": 3_854_164,
        "parameter_ratio_vs_fine15": model.parameter_count / 3_854_164,
        "device": str(device),
        "torch_version": torch.__version__,
        "deterministic_algorithms": False,
        "deterministic_note": "index_add_ in aggregate_sum/vertex_normals has no deterministic CUDA float kernel",
        "seed": args.seed,
        "resumed_from": resumed,
        "noise_sigma": list(args.noise_sigma),
        "rollout_steps": args.rollout_steps,
        "loss_weights": {
            "edge": args.edge_weight, "edge_lower": args.edge_lower, "edge_upper": args.edge_upper,
            "flip": args.flip_weight, "degenerate": args.degenerate_weight,
            "degenerate_floor": args.degenerate_floor, "accel": args.accel_weight,
            "edge_match": args.edge_match_weight,
        },
        "normalise_fit": args.normalise_fit,
        "selection": {
            "eval_steps": args.eval_steps,
            "stiff_weight": args.stiff_weight,
            "drift_weight": args.drift_weight,
            "over_cap": args.over_cap,
            "reference": "fine15 teacher rollout on ch10032_tpose",
            "teacher_edge_p95_curve": teacher_reference["edge_p95_curve"],
            "teacher_flipped_curve": teacher_reference["flipped_curve"],
        },
        "dagger": {
            "rounds": args.dagger_rounds,
            "steps": args.dagger_steps,
            "stride": args.dagger_stride,
            "max_edge_p95": args.dagger_max_edge_p95,
            "rounds_detail": dagger_statistics,
        },
        "dataset": data_statistics,
        "training_seconds": training_seconds,
        "selected": {"phase": best["phase"], "epoch": best["epoch"], "score": best["score"]},
        "checkpoint_sha256": checkpoint_hash,
        "vhood": {key: value for key, value in export_info.items() if key != "tensors"},
        "vhood_parameters_bit_identical": True,
        "vhood_reload_forward_max_abs": reload_max_abs,
        "vhood_reload_forward_note": "two CUDA forward passes of identical weights differ by ~1e-6 because index_add_ is nondeterministic; the bit-identical parameter check above is the real test",
        # The per-epoch curves are `eval_steps` floats each and would dominate the file, so the
        # history keeps the scalars and the curves live only in the selection reference above.
        "history": [
            {"phase": record.phase, "epoch": record.epoch, "loss": record.loss, "seconds": record.seconds,
             "score": record.evaluation["score"],
             "fit": record.evaluation["fit"],
             "rollout": {key: value for key, value in record.evaluation["ch10032_tpose"].items()
                         if not key.endswith("_curve")}}
            for record in history
        ],
        "final_rollouts": final,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("selected", "final_rollouts", "training_seconds")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
