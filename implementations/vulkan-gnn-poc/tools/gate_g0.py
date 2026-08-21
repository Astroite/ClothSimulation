#!/usr/bin/env python3
"""Gate G0 from plans/gnn/gnn-xpbd-v2.md: is the GNN + XPBD hybrid worth building?

The gate compares three branches on the same closed-loop rollout and the same teacher-relative
score the student work already uses:

  A  GNN only            -- the delivered student, no constraint solve. The status quo.
  B  XPBD only           -- no network at all: ballistic Verlet plus gravity, then XPBD.
  C  Hybrid              -- the student's prediction, then XPBD.

Two pre-registered decisions come out of it:

  G0-a  If B's best score is at least as good as A's, the honest conclusion is to ship XPBD and
        drop the network. The repository already contains a precedent: on the grid path, removing
        the entire graph message-passing term moved the cloth 3.5 mm over 600 steps while the
        network's own error was 14.7x larger, because stiff XPBD distance constraints already
        enforced what the network was approximating.
  G0-b  If no C configuration beats A by more than the measurement noise, the hybrid adds nothing
        and the plan stops regardless of what B did.

What this does NOT measure is GPU cost. A Python step is ~16 ms for both teacher and student
because interpreter overhead dominates, so cost has to come from the Vulkan constants in the plan's
appendix A. This is the quality half of the gate only.

Branch B deliberately supplies explicit gravity rather than a zero network output. The output
normalizer's mean is [0.0022, 0.0040, -0.0015] (magnitude 0.0048, mostly +Y), which is a
dataset artifact rather than gravity, so predicting the mean would give branch B no downward force
and make it lose for a reason unrelated to XPBD. The -Y down axis is confirmed from the data: the
render mesh is 1.649 m tall along Y, the 72 pinned vertices are the waistband sitting above the
free ones, and the teacher's own 60-step sag direction is [0.34, -0.88, 0.35].
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
from pathlib import Path

import torch

POC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC_ROOT))
sys.path.insert(0, str(POC_ROOT / "tools"))

from real_scene.fine15 import Fine15, Fine15Weights  # noqa: E402
from real_scene.runtime_scene import RuntimeScene  # noqa: E402
from real_scene.tinyhood import load_tinyhood  # noqa: E402
from real_scene.xpbd import (  # noqa: E402
    STRETCH,
    SolverConfig,
    build_area_constraints,
    build_constraints,
    calibrate_area_from_trajectory,
    calibrate_from_trajectory,
    step_substepped,
)
from compare_student_stability import crossings, decompose, trace  # noqa: E402

ASSET_STEMS = {"hood_grid64": "hood_grid64"}
DEFAULT_SCENES = ("hml_001962", "hood_grid64", "ch10032_tpose")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES))
    parser.add_argument("--scene-root", type=Path, default=POC_ROOT / ".work/real_scene")
    parser.add_argument("--fine15", type=Path, default=POC_ROOT / ".work/hood_data/fine15.vhood")
    parser.add_argument("--student", type=Path, default=POC_ROOT / ".work/hood_data/student32x12_r1.vhood")
    parser.add_argument("--steps", type=int, default=0, help="0 = the scene's frame count, min 120")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--calibrations", nargs="+", default=["bind"], choices=("rest", "bind", "teacher"))
    parser.add_argument(
        "--calibration-source",
        default=None,
        help="Scene whose teacher rollout supplies the `teacher` target lengths for every scene. "
             "Default: each scene calibrates against its own rollout. Set it to ask whether one "
             "baked calibration can serve a garment across motions, which decides whether the "
             "Vulkan .vxpbd asset is per-garment or per-motion.",
    )
    parser.add_argument("--iterations", nargs="+", type=int, default=[2, 4, 8, 16],
                        help="sweeps per SUBSTEP, so the total per visual frame is this x --substeps")
    parser.add_argument("--modes", nargs="+", default=["standard", "warmstart", "nowarm"],
                        choices=("standard", "warmstart", "nowarm", "guide"))
    parser.add_argument("--sweep", default="coloured", choices=("coloured", "jacobi", "fused"))
    parser.add_argument("--stretch-compliance", nargs="+", type=float, default=[0.0])
    parser.add_argument("--bend-compliance", nargs="+", type=float, default=[1.0e-5])
    parser.add_argument("--one-sided", nargs="+", type=int, default=[0], choices=(0, 1))
    parser.add_argument("--relaxation", type=float, default=1.0)
    parser.add_argument("--substeps", type=int, default=1,
                        help="physics substeps per visual frame. The network still runs once per "
                             "frame; its guide arrives in equal instalments. Compare at equal "
                             "--substeps x --iterations, not at equal --iterations.")
    parser.add_argument("--guide-compliance", nargs="+", type=float, default=[0.0],
                        help="mode=guide only. m^2/N. Gate G0 measured 0..1e-1 as inert for the "
                             "two-endpoint constraints; the one-vertex guide crosses over near 1.")
    parser.add_argument("--guide-trust-ratio", type=float, default=0.0,
                        help="mode=guide only. Distrust a vertex whose guide sits further than this "
                             "many of its own shortest constraints away. 0 disables the gate.")
    parser.add_argument("--area-floor", nargs="+", type=float, default=[0.0],
                        help="minimum triangle area as a fraction of the CALIBRATED area (not the "
                             "rest area). 0 disables the constraint.")
    parser.add_argument("--area-compliance", type=float, default=0.0)
    parser.add_argument("--gravity", nargs=3, type=float, default=[0.0, -9.81, 0.0])
    parser.add_argument("--no-collision", action="store_true")
    parser.add_argument("--branches", nargs="+", default=["A", "B", "C"], choices=("A", "B", "C"))
    parser.add_argument("--thresholds", type=float, nargs="+", default=[1.2, 1.5, 2.0, 5.0])
    parser.add_argument("--stiff-weight", type=float, default=0.25)
    parser.add_argument("--drift-weight", type=float, default=2.0)
    parser.add_argument("--over-cap", type=float, default=2.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=POC_ROOT / "results/gate_g0.json")
    return parser.parse_args()


class BallisticGravity:
    """Stands in for the network in branch B: the step displacement is dt^2 * g.

    `trace` calls the predictor once per step in order, so a step counter reproduces
    `make_graph`'s own timestep rule. A fresh instance is built per rollout so the counter cannot
    leak between configurations.

    Step 0 gets no gravity. The reference runtime labels it a 1/3 s settle, which is a HOOD
    convention rather than a physical substep -- at dt = 1/3 a full gravity step is 1.09 m of
    displacement, about a hundred times the whole garment's per-step motion, and no number of
    constraint sweeps recovers from it. The teacher's own step-0 output is 0.008 m, i.e. it also
    treats the settle step as "barely move", so starting branch B at rest is the faithful reading.
    """

    def __init__(self, gravity: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        self.gravity = gravity
        self.mean = mean
        self.std = std
        self.step = 0

    def __call__(self, graph) -> torch.Tensor:
        settling = self.step == 0
        self.step += 1
        timestep = 1.0 / 30.0
        displacement = torch.zeros_like(self.gravity) if settling else self.gravity * timestep * timestep
        return ((displacement - self.mean) / self.std).expand(graph.effective_position.shape[0], 3)


def make_projector(scene, constraints, config: SolverConfig, *, network: bool,
                   substeps: int = 1, area=None, min_edge=None, gravity: torch.Tensor | None = None):
    """Build the `(position, graph, step) -> position` hook `trace` applies after each step.

    `position` is whatever the predictor produced for the whole visual frame, so under
    `mode="guide"` it is the guide target rather than the state the solve starts from. The substep
    loop itself lives in `real_scene/xpbd.py::step_substepped` -- it is solver logic, and
    `tools/recovery_probe.py::make_hook` has to reach exactly the same arithmetic.

    Contacts are gated by `config.collision` alone. The old signature also took a `collision` flag,
    but the only call site passed `not args.no_collision`, which is the same value `solver_configs`
    already put into the config.
    """

    def hook(position: torch.Tensor, graph, step: int) -> torch.Tensor:
        # See tools/recovery_probe.py::make_hook for why gravity and the guide are keyed off the mode:
        # `standard` starts from the predictor's output so that output already carries its own
        # acceleration, while `guide`/`nowarm` start from the inertial prediction and need `h^2 g` per
        # substep. Branch B in a x_tilde-starting mode drops the guide, which is what makes a
        # substepped pure-XPBD row ballistic-plus-constraints rather than a pull towards a ballistic
        # target.
        analytic_gravity = config.mode != "standard"
        obstacle_frame = min(step, scene.frame_count - 1)
        target_frame = min(step + 1, scene.frame_count - 1)
        return step_substepped(
            constraints,
            config,
            scene=scene,
            graph=graph,
            guide=position if (network or not analytic_gravity) else None,
            timestep=1.0 / 3.0 if step == 0 else 1.0 / 30.0,
            frame=float(obstacle_frame),
            frame_advance=float(target_frame - obstacle_frame),
            substeps=substeps,
            area=area,
            min_edge=min_edge,
            gravity=gravity if analytic_gravity else None,
        )

    return hook


def solver_configs(args: argparse.Namespace, branch: str) -> list[SolverConfig]:
    if branch == "A":
        return [SolverConfig(iterations=0)]
    # Branch B has no network displacement, so every mode initialiser coincides there -- except
    # `guide`, which is how branch B gets substepped honestly: it starts from x_tilde and takes its
    # gravity as h^2 g per substep instead of one dt^2 g jump. Row 0 of the substep matrix is exactly
    # that, so `guide` has to survive this filter even for B, where it carries no guide target.
    modes = ["standard"] if branch == "B" and "guide" not in args.modes else \
        (["guide"] if branch == "B" else args.modes)
    configs = []
    for iterations in args.iterations:
        for mode in modes:
            for stretch in args.stretch_compliance:
                for bend in args.bend_compliance:
                    for one_sided in args.one_sided:
                        for guide_compliance in (args.guide_compliance if mode == "guide" else [0.0]):
                            for area_floor in args.area_floor:
                                configs.append(SolverConfig(
                                    iterations=iterations, mode=mode, sweep=args.sweep,
                                    stretch_compliance=stretch, bend_compliance=bend,
                                    one_sided=bool(one_sided), relaxation=args.relaxation,
                                    collision=not args.no_collision,
                                    guide_compliance=guide_compliance,
                                    guide_trust_ratio=args.guide_trust_ratio,
                                    area_floor=area_floor,
                                    area_compliance=args.area_compliance,
                                ))
    return configs


def label_for(branch: str, config: SolverConfig, calibration: str, substeps: int = 1) -> str:
    if branch == "A":
        return "A_gnn_only"
    stem = f"{branch}_{'xpbd' if branch == 'B' else 'hybrid'}_k{config.iterations}"
    if substeps > 1:
        # `k` is per substep, so a label without this would collide across the substep matrix rows
        # that are deliberately equal-budget (4x32 and 8x16 both total 128).
        stem += f"x{substeps}"
    if branch == "C" or config.mode != "standard":
        stem += f"_{config.mode}"
    stem += f"_{calibration}_sc{config.stretch_compliance:g}_bc{config.bend_compliance:g}"
    if config.mode == "guide":
        stem += f"_gc{config.guide_compliance:g}"
        if config.guide_trust_ratio > 0.0:
            stem += f"_tr{config.guide_trust_ratio:g}"
    if config.area_floor > 0.0:
        stem += f"_area{config.area_floor:g}"
    if config.one_sided:
        stem += "_onesided"
    return stem


def load_scene(scene_name: str, args: argparse.Namespace) -> RuntimeScene:
    stem = ASSET_STEMS.get(scene_name, "ch10032")
    return RuntimeScene.load(
        args.scene_root / scene_name, scene_name, device=torch.device(args.device), asset_stem=stem
    )


def calibration_donor(args: argparse.Namespace, teacher: Fine15, mean, std) -> tuple[torch.Tensor, torch.Tensor]:
    """Target lengths measured on `--calibration-source`, with the pair table they came from.

    The pairs travel with the lengths so a scene borrowing them can assert the two really share a
    garment. `ch10032_tpose`, `ch10032_sprint` and `hml_001962` all load `ch10032_lower.vcloth2`,
    so they do; `hood_grid64` is a different mesh and will trip the assertion.
    """
    scene = load_scene(args.calibration_source, args)
    steps = args.steps or max(scene.frame_count, 120)
    reference = trace(teacher.predict_graph, teacher, scene, steps, mean, std)
    base = build_constraints(scene, scene.cloth_target(0))
    lengths = calibrate_from_trajectory(base.pairs, reference["positions"], skip=min(5, steps - 1))
    return base.pairs, lengths.clamp_min(1.0e-9)


def run_scene(
    scene_name: str,
    args: argparse.Namespace,
    teacher: Fine15,
    student,
    mean,
    std,
    donor: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict:
    device = torch.device(args.device)
    scene = load_scene(scene_name, args)
    steps = args.steps or max(scene.frame_count, 120)
    gravity = torch.tensor(args.gravity, dtype=torch.float32, device=device).reshape(1, 3)

    reference = trace(teacher.predict_graph, teacher, scene, steps, mean, std)
    entry: dict = {
        "steps": steps,
        "frames": scene.frame_count,
        "teacher": summarise(reference, reference, args, steps),
        "configs": {},
    }

    references = {"rest": scene.cloth_rest, "bind": scene.cloth_target(0)}
    for calibration in args.calibrations:
        base = build_constraints(scene, references.get(calibration, scene.cloth_target(0)))
        if calibration == "teacher":
            if donor is None:
                # Its own rollout, which is already in hand -- reusing it rather than taking a
                # second one keeps `--calibration-source <this scene>` bit-identical to the
                # default despite index_add_ making rollouts non-reproducible.
                lengths = calibrate_from_trajectory(base.pairs, reference["positions"], skip=min(5, steps - 1))
                lengths = lengths.clamp_min(1.0e-9)
            else:
                donor_pairs, lengths = donor
                if donor_pairs.shape != base.pairs.shape or not bool(torch.equal(donor_pairs, base.pairs)):
                    raise SystemExit(
                        f"--calibration-source {args.calibration_source} has a different constraint "
                        f"topology than {scene_name}; lengths cannot be transplanted between them"
                    )
            base = dataclasses.replace(base, target_length=lengths)
        stretch_mask = base.kind == STRETCH
        # The area floor's reference is calibrated from the same trajectory the lengths are, for the
        # same reason: rest areas on these scenes are wrong by up to the square of the 1.890-2.025
        # edge ratio skinning already introduces, and one global rho cannot absorb that. Built per
        # calibration block so a `rest`-calibrated run gets a rest-calibrated floor and the control
        # stays a real control.
        area = None
        if any(config.area_floor > 0.0 for config in solver_configs(args, "C")):
            area_source = reference["positions"] if calibration == "teacher" else None
            target_area = (
                calibrate_area_from_trajectory(scene.cloth_triangles, area_source, skip=min(5, steps - 1))
                if area_source is not None else None
            )
            area = build_area_constraints(
                scene, target_area,
                reference_position=references.get(calibration, scene.cloth_target(0)),
            )
        # Shortest constraint at each vertex, the trust radius `guide_confidence` measures against.
        # Same quantity `tools/bake_xpbd_constraints.py::per_vertex_min_edge` bakes into every .vxpbd
        # and that no kernel has read yet.
        min_edge = None
        if args.guide_trust_ratio > 0.0:
            min_edge = torch.full((base.vertex_count,), float("inf"), device=device)
            for column in (0, 1):
                min_edge = min_edge.scatter_reduce(
                    0, base.pairs[:, column], base.target_length, reduce="amin"
                )
            min_edge = torch.where(torch.isfinite(min_edge), min_edge, torch.zeros_like(min_edge))
        entry.setdefault("calibration", {})[calibration] = {
            "source": (args.calibration_source if calibration == "teacher" and donor is not None else scene_name),
            "suspect_constraints": int(base.suspect.sum()),
            "target_length_p95": round(float(torch.quantile(base.target_length[stretch_mask], 0.95)), 6),
            # The per-iteration GPU dispatch count, which plans/gnn/gnn-xpbd-v2.md section 2.3 could
            # only estimate. Recorded per scene because it is a property of the mesh.
            "constraints": base.count,
            "colours": base.colour_count,
        }

        for branch in args.branches:
            if branch == "A" and calibration != args.calibrations[0]:
                continue  # branch A has no constraints, so it does not vary with calibration
            for config in solver_configs(args, branch):
                label = label_for(branch, config, calibration, args.substeps)
                scores, curves = [], None
                for _ in range(args.repeats):
                    if branch == "B":
                        predictor = BallisticGravity(gravity, mean, std)
                    else:
                        predictor = student
                    hook = None if branch == "A" else make_projector(
                        scene, base, config, network=branch != "B",
                        substeps=args.substeps, area=area, min_edge=min_edge, gravity=gravity,
                    )
                    current = trace(predictor, teacher, scene, steps, mean, std, reference["positions"], hook)
                    summary = summarise(current, reference, args, steps)
                    scores.append(summary["score"]["score"])
                    curves = curves or summary
                assert curves is not None
                curves["score_repeats"] = scores
                curves["score_mean"] = statistics.fmean(scores)
                curves["score_spread"] = max(scores) - min(scores)
                entry["configs"][label] = curves
                print(f"  {scene_name:14s} {label:52s} score={curves['score_mean']:8.4f} "
                      f"spread={curves['score_spread']:7.4f} steps={curves['completed_steps']}", flush=True)
    return entry


def summarise(current: dict, reference: dict, args: argparse.Namespace, steps: int) -> dict:
    curve = current["edge"]
    return {
        "completed_steps": len(curve),
        "edge_p95_at": {str(at): (curve[at - 1] if at <= len(curve) else None) for at in (5, 10, 30, 60, 120)},
        "score": decompose(current, reference, args.stiff_weight, args.drift_weight, args.over_cap),
        **crossings(curve, args.thresholds, steps),
    }


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    weights = Fine15Weights.from_vhood(args.fine15.resolve(), device=device)
    teacher = Fine15(weights)
    mean, std = weights.normalizer("output")
    student = load_tinyhood(args.student.resolve(), device=device).eval()

    report = {
        "student": args.student.name,
        "gravity": args.gravity,
        "score_weights": {"stiff": args.stiff_weight, "drift": args.drift_weight, "over_cap": args.over_cap},
        "collision": not args.no_collision,
        "relaxation": args.relaxation,
        "calibration_source": args.calibration_source,
        "scenes": {},
    }
    donor = None
    for scene_name in args.scenes:
        if not (args.scene_root / scene_name).is_dir():
            print(f"{scene_name}: not baked, skipped")
            continue
        # A scene asked to borrow its own calibration takes the default path instead, so the
        # identity case stays bit-identical to leaving the flag off.
        borrowing = args.calibration_source is not None and args.calibration_source != scene_name
        if borrowing and donor is None and "teacher" in args.calibrations:
            print(f"calibrating from {args.calibration_source}:")
            donor = calibration_donor(args, teacher, mean, std)
        print(f"{scene_name}:")
        report["scenes"][scene_name] = run_scene(
            scene_name, args, teacher, student, mean, std, donor if borrowing else None
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")
    verdict(report)
    return 0


def verdict(report: dict) -> None:
    """Apply the pre-registered G0-a / G0-b rules to whatever configurations were run."""
    print()
    for scene_name, entry in report["scenes"].items():
        configs = entry["configs"]
        baseline = configs.get("A_gnn_only")
        if baseline is None:
            continue
        noise = max((item["score_spread"] for item in configs.values()), default=0.0)
        best_b = min((item["score_mean"] for label, item in configs.items() if label.startswith("B_")), default=None)
        best_c = min((item["score_mean"] for label, item in configs.items() if label.startswith("C_")), default=None)
        print(f"{scene_name}: A={baseline['score_mean']:.4f}  "
              f"best B={'n/a' if best_b is None else f'{best_b:.4f}'}  "
              f"best C={'n/a' if best_c is None else f'{best_c:.4f}'}  noise={noise:.4f}")
        if best_b is not None and best_b <= baseline["score_mean"]:
            print("  G0-a FAILS: XPBD-only matches or beats the network -- drop the GNN.")
        if best_c is not None and best_c >= baseline["score_mean"] - noise:
            print("  G0-b FAILS: no hybrid configuration beats the network beyond the noise floor.")
        if best_b is not None and best_c is not None and best_b > baseline["score_mean"] > best_c + noise:
            print("  G0 PASSES: the network is needed and XPBD improves it.")


if __name__ == "__main__":
    raise SystemExit(main())
