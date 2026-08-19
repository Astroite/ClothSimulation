#!/usr/bin/env python3
"""Audit whether `cloth_rest` is a usable constraint target for each baked scene.

Gate G0 in `plans/gnn/gnn-xpbd-v2.md` compares XPBD-only against a GNN, which presumes the XPBD
knows what "unstretched" means. It does not: skinning the authored rest mesh into a scene's frame
0 already stretches a substantial fraction of edges, so a distance constraint aimed at
`cloth_rest` would contract them hard and produce a garment far stiffer than the teacher. That is
the same failure `edge_penalty` in tools/train_student.py documents from the training side, where
a rest-length +/-10% band is what made the 32x12 student stiffer than its teacher.

This script quantifies the problem per scene and reports the three candidate calibrations the
gate sweeps, so the choice is a recorded measurement rather than an implicit default.

It also reports a consequence for the existing metric: `edge_p95` is a ratio against
`cloth_rest`, so on a scene whose skinned frame 0 already sits at p95 1.89, the published
thresholds (1.2 / 1.5 / 2.0 / 5.0) are being read against that baseline rather than against 1.0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

POC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC_ROOT))

from real_scene.runtime_scene import RuntimeScene  # noqa: E402
from real_scene.xpbd import BEND, STRETCH, build_constraints  # noqa: E402

# (scene directory, asset stem). ch10032_lower is the garment for all three character scenes.
SCENES = (("hml_001962", "ch10032"), ("ch10032_tpose", "ch10032"), ("ch10032_sprint", "ch10032"),
          ("hood_grid64", "hood_grid64"))
QUANTILES = (0.0, 0.05, 0.5, 0.95, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-root", type=Path, default=POC_ROOT / ".work/real_scene")
    parser.add_argument("--suspect-above", type=float, default=3.0,
                        help="flag edges whose calibrated target differs from rest by more than this factor")
    parser.add_argument("--output", type=Path, default=POC_ROOT / "results/rest_length_audit.json")
    return parser.parse_args()


def quantiles(values: torch.Tensor) -> dict[str, float]:
    return {f"q{int(q * 100)}": round(float(torch.quantile(values, q)), 6) for q in QUANTILES}


def audit_scene(root: Path, scene: str, stem: str, suspect_above: float) -> dict:
    loaded = RuntimeScene.load(root / scene, scene, device="cpu", asset_stem=stem)
    rest_constraints = build_constraints(loaded, loaded.cloth_rest, suspect_above=suspect_above)
    bind_constraints = build_constraints(loaded, loaded.cloth_target(0), suspect_above=suspect_above)

    stretch = rest_constraints.kind == STRETCH
    ratio = bind_constraints.target_length[stretch] / rest_constraints.target_length[stretch]
    return {
        "cloth_vertices": rest_constraints.vertex_count,
        "stretch_constraints": int(stretch.sum()),
        "bend_constraints": int((rest_constraints.kind == BEND).sum()),
        "pinned_vertices": int(loaded.cloth_pins.sum()),
        "frames": loaded.frame_count,
        "rest_edge_length": quantiles(rest_constraints.target_length[stretch]),
        # The ratio of skinned frame-0 edge length to authored rest length. 1.0 everywhere means
        # the two configurations agree and `cloth_rest` is a valid constraint target.
        "skinned_over_rest": quantiles(ratio),
        "edges_above_1.5": int((ratio > 1.5).sum()),
        "edges_above_3": int((ratio > 3.0).sum()),
        "edges_above_6": int((ratio > 6.0).sum()),
        "fraction_above_1.5": round(float((ratio > 1.5).float().mean()), 5),
        "suspect_under_bind_calibration": int(bind_constraints.suspect.sum()),
        "rest_is_usable_target": bool(float(ratio.max()) < 1.5),
    }


def main() -> int:
    args = parse_args()
    report = {"suspect_above": args.suspect_above, "scenes": {}}
    for scene, stem in SCENES:
        directory = args.scene_root / scene
        if not directory.is_dir():
            print(f"{scene:16s} skipped (not baked)")
            continue
        report["scenes"][scene] = audit_scene(args.scene_root, scene, stem, args.suspect_above)

    print(f"{'scene':16s} {'frames':>7s} {'p95':>7s} {'max':>8s} {'>1.5':>7s} {'>3':>5s} {'>6':>5s} "
          f"{'rest usable':>12s}")
    for scene, entry in report["scenes"].items():
        print(f"{scene:16s} {entry['frames']:7d} {entry['skinned_over_rest']['q95']:7.3f} "
              f"{entry['skinned_over_rest']['q100']:8.2f} "
              f"{entry['edges_above_1.5']:7d} {entry['edges_above_3']:5d} {entry['edges_above_6']:5d} "
              f"{str(entry['rest_is_usable_target']):>12s}")

    usable = [scene for scene, entry in report["scenes"].items() if entry["rest_is_usable_target"]]
    print()
    print("Constraint target recommendation per scene:")
    for scene, entry in report["scenes"].items():
        target = "rest (self-consistent)" if entry["rest_is_usable_target"] else "bind or teacher (rest unusable)"
        print(f"  {scene:16s} -> {target}")
    if usable:
        print(f"\n`cloth_rest` is only a valid target on: {', '.join(usable)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
