#!/usr/bin/env python3
"""Compare students by how long they hold cloth structure, not by a single end-state metric.

Comparing edge_ratio_p95 at a fixed step between two already-diverged students says little:
both numbers are large and their ordering is close to arbitrary. What matters for a simulator
is how many steps it survives before the structure degrades past a usable threshold, so this
reports the first step at which the 95th-percentile edge ratio crosses each threshold.

It also decomposes the training script's selection score against the teacher's own rollout, so
the same number that picked the weights can be read off here. The decomposition matters because
the total hides which way a model is wrong: `over` is divergence, `under` is over-damping, and
`drift` is a trajectory that stays smooth while going somewhere else.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

POC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC_ROOT))

from real_scene.fine15 import Fine15, Fine15Weights  # noqa: E402
from real_scene.runtime_scene import RuntimeScene  # noqa: E402
from real_scene.tinyhood import load_tinyhood  # noqa: E402

sys.path.insert(0, str(POC_ROOT / "tools"))
from train_student import advance, curve_point, make_graph  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, help="label=path pairs")
    parser.add_argument("--scene-root", type=Path, default=POC_ROOT / ".work/real_scene")
    parser.add_argument("--scene", default="ch10032_tpose")
    parser.add_argument("--asset-stem", default="ch10032")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[1.2, 1.5, 2.0, 5.0])
    parser.add_argument("--stiff-weight", type=float, default=0.25)
    parser.add_argument("--drift-weight", type=float, default=2.0)
    parser.add_argument("--over-cap", type=float, default=2.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fine15", type=Path, default=POC_ROOT / ".work/hood_data/fine15.vhood")
    parser.add_argument("--output", type=Path, default=POC_ROOT / "results/student_stability_comparison.json")
    return parser.parse_args()


def trace(predictor, builder, scene, steps, mean, std, reference=None) -> dict:
    position = scene.cloth_target(0)
    previous = position.clone()
    edge: list[float] = []
    flipped: list[float] = []
    drift: list[float] = []
    positions: list[torch.Tensor] = []
    with torch.no_grad():
        for step in range(steps):
            position, previous, _, _ = advance(predictor, builder, scene, position, previous, step, mean, std)
            if not torch.isfinite(position).all():
                edge.append(float("inf"))
                break
            point = curve_point(scene, position)
            edge.append(point[0])
            flipped.append(point[1])
            positions.append(position.clone())
            if reference is not None and step < len(reference):
                drift.append(float((position - reference[step]).square().mean().sqrt().item()))
    return {"edge": edge, "flipped": flipped, "drift": drift, "positions": positions}


def decompose(student: dict, teacher: dict, stiff_weight: float, drift_weight: float, cap: float) -> dict:
    """Split the selection score into the three ways a student can be wrong.

    Per-step terms are capped exactly as the training script caps them: past divergence the tail
    is chaotic and, because index_add_ has no deterministic CUDA kernel, an uncapped score is not
    repeatable for fixed weights. `score_uncapped` is reported alongside so the raw magnitude of
    a blow-up is still visible.
    """
    count = min(len(student["flipped"]), len(teacher["flipped"]))
    if count == 0:
        return {"score": float("inf"), "over": None, "under": None, "flip": None,
                "drift_rms": None, "drift_rms_max": None, "score_uncapped": None, "compared_steps": 0}
    over = under = flip = drift = 0.0
    raw_over = raw_drift = 0.0
    for index in range(count):
        difference = student["edge"][index] - teacher["edge"][index]
        over += min(max(difference, 0.0), cap)
        raw_over += max(difference, 0.0)
        under += min(max(-difference, 0.0), cap)
        flip += max(student["flipped"][index] - teacher["flipped"][index], 0.0)
        if index < len(student["drift"]):
            drift += min(student["drift"][index], cap)
            raw_drift += student["drift"][index]
    over, under, flip, drift = over / count, under / count, flip / count, drift / count
    return {
        "compared_steps": count,
        "over": over,
        "under": under,
        "flip": flip,
        "drift_rms": drift,
        "drift_rms_max": max(student["drift"][:count], default=0.0),
        "score": over + stiff_weight * under + 2.0 * flip + drift_weight * drift,
        "score_uncapped": raw_over / count + stiff_weight * under + 2.0 * flip + drift_weight * (raw_drift / count),
    }


def crossings(curve: list[float], thresholds: list[float], steps: int) -> dict:
    result = {}
    for threshold in thresholds:
        first = next((index + 1 for index, value in enumerate(curve) if value > threshold), None)
        # None means it never crossed within the horizon; report it as such rather than as a
        # number, so a survivor is never confused with a late failure.
        result[f"first_step_above_{threshold}"] = first if first is not None else f">{steps}"
    return result


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    weights = Fine15Weights.from_vhood(args.fine15.resolve(), device=device)
    teacher = Fine15(weights)
    mean, std = weights.normalizer("output")
    scene = RuntimeScene.load(args.scene_root / args.scene, args.scene, device=device, asset_stem=args.asset_stem)

    report = {
        "scene": args.scene,
        "steps": args.steps,
        "thresholds": args.thresholds,
        "score_weights": {"stiff": args.stiff_weight, "drift": args.drift_weight, "over_cap": args.over_cap},
        "models": {},
    }
    reference = trace(teacher.predict_graph, teacher, scene, args.steps, mean, std)
    entries = [("fine15_teacher", None)] + [tuple(pair.split("=", 1)) for pair in args.models]
    for label, path in entries:
        if path is None:
            current = reference
        else:
            predictor = load_tinyhood(Path(path).resolve(), device=device).eval()
            current = trace(predictor, teacher, scene, args.steps, mean, std, reference["positions"])
        curve = current["edge"]
        report["models"][label] = {
            "completed_steps": len(curve),
            "edge_p95_curve": curve,
            "edge_p95_at": {str(at): (curve[at - 1] if at <= len(curve) else None) for at in (5, 10, 30, 60, 120)},
            "score": decompose(current, reference, args.stiff_weight, args.drift_weight, args.over_cap),
            **crossings(curve, args.thresholds, args.steps),
        }
        print(f"{label:22} " + "  ".join(
            f"@{at}={curve[at-1]:8.3f}" if at <= len(curve) else f"@{at}=     n/a" for at in (5, 10, 30, 60, 120)
        ), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {key: value for key, value in report.items()}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print()
    print(f"{'model':22} " + "  ".join(f">{t}" .rjust(9) for t in args.thresholds))
    for label, entry in report["models"].items():
        print(f"{label:22} " + "  ".join(str(entry[f'first_step_above_{t}']).rjust(9) for t in args.thresholds))
    print()
    columns = ("score", "score_uncapped", "over", "under", "flip", "drift_rms", "drift_rms_max")
    print(f"{'model':22} " + "  ".join(name.rjust(8) for name in
                                       ("score", "uncapped", "over", "under", "flip", "driftRMS", "driftMax")))
    for label, entry in report["models"].items():
        score = entry["score"]
        print(f"{label:22} " + "  ".join(
            f"{score[key]:8.4f}" if score.get(key) is not None else "     n/a" for key in columns
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
