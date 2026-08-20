"""Validate the CH10032 FBX exports by importing each one in Blender.

A non-empty FBX is not necessarily a usable one, so check the properties the
downstream bake actually depends on: an armature, an animation action with
keyframes, a plausible frame range, a root bone, and finite transforms.

Run through Blender's Python runtime:

    blender --background --factory-startup \
        --python tools/validate_ch10032_exports.py -- \
        --library-root .work/ch10032_library
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-root", required=True, type=Path)
    parser.add_argument(
        "--min-bones",
        type=int,
        default=100,
        help="a CH10032 rig has ~1000 bones; anything near zero means a broken export",
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=8,
        help="evenly spaced frames to check for non-finite bone transforms",
    )
    return parser.parse_args(argv)


ARGS = parse_args()


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def armatures() -> list[bpy.types.Object]:
    return [o for o in bpy.data.objects if o.type == "ARMATURE"]


def finite(values) -> bool:
    return all(math.isfinite(v) for v in values)


def check_animation(path: Path) -> dict:
    """Import one animation FBX and report what the bake would find in it."""
    reset_scene()
    result: dict = {"file": path.name, "checks": {}, "status": "passed", "errors": []}
    checks = result["checks"]

    try:
        bpy.ops.import_scene.fbx(filepath=str(path))
    except Exception as exc:  # noqa: BLE001 - report, do not abort the batch
        result["status"] = "failed"
        result["errors"].append(f"import raised {type(exc).__name__}: {exc}")
        return result

    checks["blender_can_import"] = True

    rigs = armatures()
    checks["has_armature"] = bool(rigs)
    if not rigs:
        result["status"] = "failed"
        result["errors"].append("no armature in file")
        return result

    rig = rigs[0]
    bones = list(rig.pose.bones)
    result["bone_count"] = len(bones)
    checks["bone_count_plausible"] = len(bones) >= ARGS.min_bones
    checks["has_root_bone"] = any(b.parent is None for b in bones)

    action = rig.animation_data.action if rig.animation_data else None
    checks["has_animation_curve"] = action is not None and len(action.fcurves) > 0
    if action is None:
        result["status"] = "failed"
        result["errors"].append("armature carries no action")
        return result

    result["fcurve_count"] = len(action.fcurves)
    start, end = (int(round(v)) for v in action.frame_range)
    result["frame_range"] = [start, end]
    result["frame_count"] = end - start + 1
    checks["multiple_frames"] = result["frame_count"] > 1

    keyframes = sum(len(fc.keyframe_points) for fc in action.fcurves)
    result["keyframe_count"] = keyframes
    checks["has_keyframes"] = keyframes > 0

    # Sample across the action; a broken export tends to show NaN once the
    # curves are actually evaluated rather than at import time.
    step = max(1, result["frame_count"] // max(1, ARGS.sample_frames))
    non_finite: list[str] = []
    for frame in range(start, end + 1, step):
        bpy.context.scene.frame_set(frame)
        for bone in bones:
            matrix = bone.matrix
            if not finite([c for row in matrix for c in row]):
                non_finite.append(f"{bone.name}@{frame}")
                break
        if non_finite:
            break
    checks["no_nan_inf_transforms"] = not non_finite
    if non_finite:
        result["errors"].append(f"non-finite transforms: {non_finite[:5]}")

    if not all(checks.values()):
        result["status"] = "failed"
        result["errors"].extend(f"check failed: {k}" for k, v in checks.items() if not v)
    return result


def check_model(path: Path) -> dict:
    """Import one skeletal mesh FBX and report geometry plus rig integrity."""
    reset_scene()
    result: dict = {"file": path.name, "checks": {}, "status": "passed", "errors": []}
    checks = result["checks"]

    try:
        bpy.ops.import_scene.fbx(filepath=str(path))
    except Exception as exc:  # noqa: BLE001
        result["status"] = "failed"
        result["errors"].append(f"import raised {type(exc).__name__}: {exc}")
        return result

    checks["blender_can_import"] = True

    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    checks["has_mesh"] = bool(meshes)
    rigs = armatures()
    checks["has_armature"] = bool(rigs)
    if not meshes or not rigs:
        result["status"] = "failed"
        result["errors"].append("missing mesh or armature")
        return result

    result["mesh_count"] = len(meshes)
    result["vertex_count"] = sum(len(m.data.vertices) for m in meshes)
    result["triangle_count"] = sum(len(m.data.loop_triangles) for m in meshes)
    result["bone_count"] = len(rigs[0].pose.bones)
    checks["bone_count_plausible"] = result["bone_count"] >= ARGS.min_bones
    checks["has_vertices"] = result["vertex_count"] > 0

    # Skinning is what the runtime consumes, so an unweighted mesh is a failure.
    skinned = [m for m in meshes if any(mod.type == "ARMATURE" for mod in m.modifiers)]
    checks["mesh_is_skinned"] = bool(skinned)
    result["skinned_mesh_count"] = len(skinned)

    non_finite = [
        m.name for m in meshes
        for v in m.data.vertices
        if not finite(v.co)
    ][:5]
    checks["no_nan_inf_vertices"] = not non_finite
    if non_finite:
        result["errors"].append(f"non-finite vertices in {non_finite}")

    if not all(checks.values()):
        result["status"] = "failed"
        result["errors"].extend(f"check failed: {k}" for k, v in checks.items() if not v)
    return result


def main() -> int:
    root: Path = ARGS.library_root
    anim_dir = root / "animations"
    model_dir = root / "models"

    animations = sorted(anim_dir.glob("*.fbx"))
    models = sorted(model_dir.glob("*.fbx"))
    if not animations and not models:
        print(f"VALIDATION ERROR: no FBX files under {root}", file=sys.stderr)
        return 2

    report = {
        "schema_version": 1,
        "library_root": str(root),
        "min_bones": ARGS.min_bones,
        "animations": [check_animation(p) for p in animations],
        "models": [check_model(p) for p in models],
    }

    entries = report["animations"] + report["models"]
    failed = [e for e in entries if e["status"] == "failed"]
    report["counts"] = {
        "checked": len(entries),
        "passed": len(entries) - len(failed),
        "failed": len(failed),
    }

    report_path = root / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for entry in entries:
        detail = (
            f"bones={entry.get('bone_count')} frames={entry.get('frame_count')}"
            if "frame_count" in entry
            else f"bones={entry.get('bone_count')} verts={entry.get('vertex_count')}"
        )
        print(f"[{entry['status']}] {entry['file']} {detail}")
    for entry in failed:
        print(f"FAILED {entry['file']}: {'; '.join(entry['errors'])}", file=sys.stderr)

    print(f"checked={len(entries)} passed={report['counts']['passed']} failed={len(failed)}")
    print(f"report: {report_path}")
    return 1 if failed else 0


sys.exit(main())
