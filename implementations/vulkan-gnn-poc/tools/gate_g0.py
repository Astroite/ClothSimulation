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
    build_constraints,
    calibrate_from_trajectory,
    contacts_from_graph,
    inertial_prediction,
    project,
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
    parser.add_argument("--iterations", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--modes", nargs="+", default=["standard", "warmstart", "nowarm"],
                        choices=("standard", "warmstart", "nowarm"))
    parser.add_argument("--sweep", default="coloured", choices=("coloured", "jacobi"))
    parser.add_argument("--stretch-compliance", nargs="+", type=float, default=[0.0])
    parser.add_argument("--bend-compliance", nargs="+", type=float, default=[1.0e-5])
    parser.add_argument("--one-sided", nargs="+", type=int, default=[0], choices=(0, 1))
    parser.add_argument("--relaxation", type=float, default=1.0)
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


def make_projector(scene, constraints, config: SolverConfig, *, network: bool, collision: bool):
    """Build the `(position, graph, step) -> position` hook `trace` applies after each step."""

    def hook(position: torch.Tensor, graph, step: int) -> torch.Tensor:
        contacts = None
        if collision and graph.world_cloth.numel() > 0:
            obstacle_frame = min(step, scene.frame_count - 1)
            target_frame = min(step + 1, scene.frame_count - 1)
            _, normals = scene.proxy(obstacle_frame)
            target, _ = scene.proxy(target_frame)
            contacts = contacts_from_graph(graph, target, normals)
        # Branch B's displacement from the Verlet prediction *is* gravity -- real dynamics, not a
        # network artifact -- so the inertial reference has to include it. With `inertial = position`
        # there is nothing for `warmstart`/`nowarm` to discard and all three initialisers coincide,
        # which is the correct reading of a branch that has no network.
        inertial = inertial_prediction(graph) if network else position
        return project(
            constraints,
            config,
            position=position,
            inertial=inertial,
            pin_mask=graph.pin_mask,
            pin_target=graph.pin_target,
            timestep=1.0 / 3.0 if step == 0 else 1.0 / 30.0,
            contacts=contacts,
        )

    return hook


def solver_configs(args: argparse.Namespace, branch: str) -> list[SolverConfig]:
    if branch == "A":
        return [SolverConfig(iterations=0)]
    # Branch B has no network displacement, so every mode initialiser coincides there.
    modes = ["standard"] if branch == "B" else args.modes
    configs = []
    for iterations in args.iterations:
        for mode in modes:
            for stretch in args.stretch_compliance:
                for bend in args.bend_compliance:
                    for one_sided in args.one_sided:
                        configs.append(SolverConfig(
                            iterations=iterations, mode=mode, sweep=args.sweep,
                            stretch_compliance=stretch, bend_compliance=bend,
                            one_sided=bool(one_sided), relaxation=args.relaxation,
                            collision=not args.no_collision,
                        ))
    return configs


def label_for(branch: str, config: SolverConfig, calibration: str) -> str:
    if branch == "A":
        return "A_gnn_only"
    stem = f"{branch}_{'xpbd' if branch == 'B' else 'hybrid'}_k{config.iterations}"
    if branch == "C":
        stem += f"_{config.mode}"
    stem += f"_{calibration}_sc{config.stretch_compliance:g}_bc{config.bend_compliance:g}"
    if config.one_sided:
        stem += "_onesided"
    return stem


def run_scene(scene_name: str, args: argparse.Namespace, teacher: Fine15, student, mean, std) -> dict:
    device = torch.device(args.device)
    stem = ASSET_STEMS.get(scene_name, "ch10032")
    scene = RuntimeScene.load(args.scene_root / scene_name, scene_name, device=device, asset_stem=stem)
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
            lengths = calibrate_from_trajectory(base.pairs, reference["positions"], skip=min(5, steps - 1))
            base = dataclasses.replace(base, target_length=lengths.clamp_min(1.0e-9))
        stretch_mask = base.kind == STRETCH
        entry.setdefault("calibration", {})[calibration] = {
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
                label = label_for(branch, config, calibration)
                scores, curves = [], None
                for _ in range(args.repeats):
                    if branch == "B":
                        predictor = BallisticGravity(gravity, mean, std)
                    else:
                        predictor = student
                    hook = None if branch == "A" else make_projector(
                        scene, base, config, network=branch != "B", collision=not args.no_collision
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
        "scenes": {},
    }
    for scene_name in args.scenes:
        if not (args.scene_root / scene_name).is_dir():
            print(f"{scene_name}: not baked, skipped")
            continue
        print(f"{scene_name}:")
        report["scenes"][scene_name] = run_scene(scene_name, args, teacher, student, mean, std)

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
