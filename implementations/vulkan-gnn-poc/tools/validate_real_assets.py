#!/usr/bin/env python3
"""Strict validation for generated CH10032 runtime assets."""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

POC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC_ROOT))

from real_scene.formats import FormatError, load_sectioned  # noqa: E402


def unpack(section, code: str) -> tuple:
    component_size = struct.calcsize(code)
    return struct.unpack(f"<{len(section.data) // component_size}{code}", section.data)


def finite(section) -> None:
    values = unpack(section, "f")
    if not all(math.isfinite(value) for value in values):
        raise FormatError(f"section {section.name} contains NaN/Inf")


def validate(asset_root: Path, motion: str) -> dict:
    character = load_sectioned(
        asset_root / "ch10032.vchar",
        expected_magic=b"VCHAR001",
        expected_version=1,
        required_sections=(
            "info", "render_pos", "render_nrm", "render_uv", "render_tri", "tri_material",
            "bone_idx", "bone_weight", "proxy_pos", "proxy_nrm", "proxy_bone_idx", "proxy_weight",
        ),
    )
    cloth = load_sectioned(
        asset_root / "ch10032_lower.vcloth2",
        expected_magic=b"VCLTH002",
        expected_version=2,
        required_sections=(
            "positions", "triangles", "csr_offsets", "csr_neighbors", "pin_mask", "mass",
            "bone_idx", "bone_weight", "coord_params",
        ),
    )
    animation = load_sectioned(
        asset_root / f"{motion}.vanim",
        expected_magic=b"VANIM001",
        expected_version=1,
        required_sections=("info", "skin_matrices", "root_pos"),
    )

    render_vertices, render_triangles, bone_count, proxy_vertices, material_count, influences = unpack(
        character.require("info", count=6, stride=4), "I"
    )
    if influences != 12 or not 1 <= bone_count <= 1024 or proxy_vertices > 4096:
        raise FormatError("unsupported character bone/influence/proxy declaration")
    character.require("render_pos", count=render_vertices, stride=12)
    character.require("render_nrm", count=render_vertices, stride=12)
    character.require("render_uv", count=render_vertices, stride=8)
    character.require("render_tri", count=render_triangles, stride=12)
    character.require("tri_material", count=render_triangles, stride=4)
    character.require("bone_idx", count=render_vertices, stride=48)
    character.require("bone_weight", count=render_vertices, stride=48)
    character.require("proxy_pos", count=proxy_vertices, stride=12)
    character.require("proxy_nrm", count=proxy_vertices, stride=12)
    character.require("proxy_bone_idx", count=proxy_vertices, stride=48)
    character.require("proxy_weight", count=proxy_vertices, stride=48)

    for name in ("render_pos", "render_nrm", "render_uv", "bone_weight", "proxy_pos", "proxy_nrm", "proxy_weight"):
        finite(character.require(name))
    for name in ("positions", "mass", "bone_weight", "coord_params"):
        finite(cloth.require(name))

    cloth_vertices = cloth.require("positions", stride=12).count
    cloth_triangles = cloth.require("triangles", stride=12).count
    cloth.require("pin_mask", count=cloth_vertices, stride=4)
    cloth.require("mass", count=cloth_vertices, stride=4)
    cloth.require("bone_idx", count=cloth_vertices, stride=48)
    cloth.require("bone_weight", count=cloth_vertices, stride=48)
    offsets = unpack(cloth.require("csr_offsets", count=cloth_vertices + 1, stride=4), "I")
    neighbors = unpack(cloth.require("csr_neighbors", stride=4), "I")
    if offsets[0] != 0 or offsets[-1] != len(neighbors) or any(a > b for a, b in zip(offsets, offsets[1:])):
        raise FormatError("cloth CSR offsets are invalid")
    if any(value >= cloth_vertices for value in neighbors):
        raise FormatError("cloth CSR neighbor is out of range")

    def check_weight_sums(section, count: int, label: str) -> float:
        values = unpack(section, "f")
        maximum_error = 0.0
        for index in range(count):
            total = sum(values[index * 12 : (index + 1) * 12])
            maximum_error = max(maximum_error, abs(total - 1.0))
        if maximum_error > 2.0e-6:
            raise FormatError(f"{label} LBS weights are not normalized: max error {maximum_error}")
        return maximum_error

    render_weight_error = check_weight_sums(character.require("bone_weight"), render_vertices, "render")
    proxy_weight_error = check_weight_sums(character.require("proxy_weight"), proxy_vertices, "proxy")
    cloth_weight_error = check_weight_sums(cloth.require("bone_weight"), cloth_vertices, "cloth")

    frame_count, animation_bones, fps, root_bone = unpack(animation.require("info", count=4, stride=4), "I")
    if animation_bones != bone_count or fps != 30 or frame_count == 0 or root_bone >= bone_count:
        raise FormatError("animation declaration does not match the character")
    finite(animation.require("skin_matrices", count=frame_count * bone_count, stride=48))
    finite(animation.require("root_pos", count=frame_count, stride=12))
    roots = unpack(animation.require("root_pos"), "f")
    maximum_root_step = 0.0
    for frame in range(1, frame_count):
        previous = roots[(frame - 1) * 3 : frame * 3]
        current = roots[frame * 3 : (frame + 1) * 3]
        maximum_root_step = max(maximum_root_step, math.dist(previous, current))
    if maximum_root_step > 0.25:
        raise FormatError(f"root motion contains a discontinuity: {maximum_root_step:.6f} m/frame")

    metadata = json.loads((asset_root / "scene.json").read_text(encoding="utf-8"))
    if metadata.get("character") != "CH10032" or metadata.get("motion") != motion:
        raise FormatError("scene.json identity does not match requested scene")
    pin_mask = unpack(cloth.require("pin_mask"), "I")
    return {
        "character": "CH10032",
        "motion": motion,
        "render_vertices": render_vertices,
        "render_triangles": render_triangles,
        "proxy_vertices": proxy_vertices,
        "cloth_vertices": cloth_vertices,
        "cloth_triangles": cloth_triangles,
        "directed_mesh_edges": len(neighbors),
        "bones": bone_count,
        "frames": frame_count,
        "fps": fps,
        "pinned_vertices": sum(bool(value) for value in pin_mask),
        "max_root_step_m": maximum_root_step,
        "max_weight_sum_error": max(render_weight_error, proxy_weight_error, cloth_weight_error),
        "material_slots": material_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--motion", default="ch10032_sprint")
    args = parser.parse_args()
    report = validate(args.asset_root.resolve(), args.motion)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
