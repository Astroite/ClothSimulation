#!/usr/bin/env python3
"""Bake the XPBD constraint set for a garment into a `.vxpbd` asset the Vulkan runtime can load.

Why a separate file rather than new sections in `.vcloth2`. Two reasons, both measured:

* The target lengths are calibrated against a teacher rollout, and `ch10032_lower.vcloth2` is
  shared by `ch10032_tpose`, `ch10032_sprint` and `hml_001962`. Whether one calibration serves all
  three is an empirical question (`tools/gate_g0.py --calibration-source`), so the file that
  carries it has to be free to be per-motion if the answer is no.
* Adding sections to `.vcloth2` changes its payload SHA-256, which every existing golden and
  validation record is pinned to. A new file cannot break anything that does not read it.

The constraint definition itself is NOT reimplemented here: this calls `real_scene.xpbd`'s
`build_constraints` / `calibrate_from_trajectory` / `bake_tables`, the same functions gate G0 ran.
That shared definition is what makes the Python-versus-Vulkan comparison in
`tools/run_tinyhood_reference.py` meaningful -- if the baker had its own idea of which pairs are
constraints, a mismatch would show up as a numerical difference with no way to tell the two causes
apart.

Layout notes for the kernel:

* `slots` / `signs` are the padded [V, K] gather tables straight out of `ConstraintSet`, with the
  sentinel index `constraint_count` in unused lanes. Measured K is 18 on CH10032 (38% of lanes
  padding) and 14 on hood_grid64 (3%). A CSR layout would remove that padding, but the sweep is
  dispatch-bound rather than work-bound, so the padding costs nothing and a flat stride keeps the
  kernel and this baker trivially comparable to the Python reference.
* `weight_sum` is baked rather than recomputed. The fused sweep has both endpoints of a constraint
  evaluate the same multiplier update, and `w_a + w_b` against `w_b + w_a` is not guaranteed to
  give the same float, which would let the two per-slot copies of lambda drift apart. See
  `real_scene/xpbd.py::_apply_fused` and `tests/test_xpbd.py::FusedPortTests`.
* Compliance is NOT baked. `kind` is, and the two compliance values live in the runtime's uniform
  buffer, so they stay tunable from the UI. `alpha = compliance / dt^2` has to be formed at
  runtime anyway because the reference pipeline's first step is a 1/3 s settle and every later
  step is 1/30 s.
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
from real_scene.formats import Section, sha256_file, write_sectioned  # noqa: E402
from real_scene.runtime_scene import RuntimeScene  # noqa: E402
from real_scene.xpbd import (  # noqa: E402
    STRETCH,
    SolverConfig,
    bake_tables,
    build_constraints,
    calibrate_from_trajectory,
)

sys.path.insert(0, str(POC_ROOT / "tools"))

from compare_student_stability import trace  # noqa: E402

MAGIC = b"VXPBD001"
VERSION = 1
ASSET_STEMS = {"hood_grid64": "hood_grid64"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, help="scene directory under --scene-root, e.g. ch10032_tpose")
    parser.add_argument("--scene-root", type=Path, default=POC_ROOT / ".work/real_scene")
    parser.add_argument("--asset-stem", default=None, help="defaults to the scene's usual stem")
    parser.add_argument(
        "--calibration",
        default="teacher",
        choices=("teacher", "bind", "rest"),
        help="teacher = median edge length over the teacher's own rollout (gate G0's winner); "
             "bind = the skinned frame 0; rest = the authored mesh, which gate G0 measured as "
             "unusable on real garments (p95 edge ratio 1.89-2.03)",
    )
    parser.add_argument(
        "--calibration-scene",
        default=None,
        help="take the teacher rollout from this scene instead, to bake one calibration for a "
             "garment shared by several motions. Must have the same constraint topology.",
    )
    parser.add_argument("--fine15", type=Path, default=POC_ROOT / ".work/hood_data/fine15.vhood")
    parser.add_argument("--steps", type=int, default=0, help="0 = the scene's frame count, min 120")
    parser.add_argument("--no-bend", action="store_true", help="stretch constraints only")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def load_scene(name: str, args: argparse.Namespace) -> RuntimeScene:
    stem = args.asset_stem or ASSET_STEMS.get(name, "ch10032")
    return RuntimeScene.load(
        args.scene_root / name, name, device=torch.device(args.device), asset_stem=stem
    )


def teacher_lengths(name: str, args: argparse.Namespace, pairs: torch.Tensor) -> torch.Tensor:
    """Median edge length over `name`'s teacher rollout, measured on `pairs`."""
    weights = Fine15Weights.from_vhood(args.fine15.resolve(), device=torch.device(args.device))
    teacher = Fine15(weights)
    mean, std = weights.normalizer("output")
    scene = load_scene(name, args)
    steps = args.steps or max(scene.frame_count, 120)
    if not torch.equal(build_constraints(scene, scene.cloth_target(0)).pairs, pairs):
        raise SystemExit(f"{name} does not share a constraint topology with the scene being baked")
    reference = trace(teacher.predict_graph, teacher, scene, steps, mean, std)
    return calibrate_from_trajectory(pairs, reference["positions"], skip=min(5, steps - 1)).clamp_min(1.0e-9)


def per_vertex_min_edge(pairs: torch.Tensor, lengths: torch.Tensor, vertex_count: int) -> torch.Tensor:
    """Shortest constraint at each vertex, for the trust region in gnn-xpbd-v2.md section 7.1.

    Baked now and unused by the first kernel. The plan's original form clamped against the global
    minimum edge length, which on a real garment lets the single shortest edge in the mesh dictate
    the clamp everywhere; a per-vertex value is the fix and it costs nothing to produce here.
    """
    result = torch.full((vertex_count,), float("inf"), dtype=torch.float32, device=lengths.device)
    for column in (0, 1):
        result = result.scatter_reduce(0, pairs[:, column], lengths, reduce="amin")
    return torch.where(torch.isfinite(result), result, torch.zeros_like(result))


def little(tensor: torch.Tensor, dtype: str) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().astype(dtype).tobytes()


def main() -> int:
    args = parse_args()
    scene = load_scene(args.scene, args)
    constraints = build_constraints(
        scene, scene.cloth_target(0), include_bend=not args.no_bend
    )
    count, vertices = constraints.count, constraints.vertex_count
    width = int(constraints.slots.shape[1])

    calibration_scene = args.calibration_scene or args.scene
    if args.calibration == "teacher":
        target = teacher_lengths(calibration_scene, args, constraints.pairs)
    elif args.calibration == "bind":
        target = constraints.target_length
    else:
        target = build_constraints(
            scene, scene.cloth_rest, include_bend=not args.no_bend
        ).target_length

    # `bake_tables` needs a config only for the compliance, and compliance is a runtime uniform
    # here, so the timestep and compliance passed in do not reach the asset -- only `weight_sum`
    # does, and that depends on neither.
    tables = bake_tables(constraints, SolverConfig(), 1.0 / 30.0)
    stretch = int((constraints.kind == STRETCH).sum())

    sections = [
        Section("info", 4, 4, little(torch.tensor([count, vertices, width, stretch]), "<u4")),
        Section("pairs", count, 8, little(constraints.pairs, "<u4")),
        Section("target_len", count, 4, little(target, "<f4")),
        Section("weight_sum", count, 4, little(tables.weight_sum, "<f4")),
        Section("kind", count, 4, little(constraints.kind, "<u4")),
        Section("slots", vertices * width, 4, little(constraints.slots.reshape(-1), "<u4")),
        Section("signs", vertices * width, 4, little(constraints.signs.reshape(-1), "<f4")),
        Section("incident", vertices, 4, little(constraints.incident.reshape(-1).clamp_min(1.0), "<f4")),
        Section("inverse_mass", vertices, 4, little(constraints.inverse_mass.reshape(-1), "<f4")),
        Section("min_edge", vertices, 4, little(
            per_vertex_min_edge(constraints.pairs, target, vertices), "<f4"
        )),
    ]
    output = args.output or (args.scene_root / args.scene / f"{args.scene}.vxpbd")
    written = write_sectioned(output.resolve(), MAGIC, VERSION, sections,
                              source_sha256=sha256_file(args.fine15.resolve()))

    stretch_target = target[constraints.kind == STRETCH]
    report = {
        "output": str(output.resolve()),
        "scene": args.scene,
        "calibration": args.calibration,
        "calibration_scene": calibration_scene,
        "constraints": count,
        "stretch": stretch,
        "bend": count - stretch,
        "vertices": vertices,
        "slot_width": width,
        "live_slots": int((constraints.slots < count).sum()),
        "suspect_constraints": int(constraints.suspect.sum()),
        "pinned_vertices": int((constraints.inverse_mass.reshape(-1) == 0.0).sum()),
        "dead_constraints": int((~tables.alive).sum()),
        "target_length_p50": round(float(stretch_target.median()), 6),
        "target_length_p95": round(float(torch.quantile(stretch_target, 0.95)), 6),
        "payload_sha256": written["payload_sha256"],
        "file_bytes": written["file_bytes"],
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
