#!/usr/bin/env python3
"""Read-only UE AnimPose bake for the MLDRV001 interchange format."""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import sys
from pathlib import Path
from typing import Iterable

import unreal  # type: ignore


MAGIC = b"MLDRV001"
HEADER = struct.Struct("<8s10I32s32s32s")
EXPECTED_MODEL_SHA256 = "6c5165eae13d3b23888ad74ab0204bc528ea88bf20af4c26771cd9217813b65b"


def fail(message: str) -> "NoReturn":
    unreal.log_error(f"[MLClothBake] {message}")
    raise RuntimeError(message)


def env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        fail(f"missing environment input {name}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_model(path: Path) -> tuple[list[str], str]:
    data = path.read_bytes()
    if len(data) < 5:
        fail("encoded model is truncated")
    json_bytes = struct.unpack_from("<I", data, 0)[0]
    if json_bytes <= 1 or 4 + json_bytes >= len(data):
        fail("encoded model JSON header length is invalid")
    try:
        config = json.loads(data[4 : 4 + json_bytes].decode("utf-8"))
    except Exception as exc:
        fail(f"encoded model JSON is invalid: {exc}")
    required = {
        "modelType": 2,
        "driverFeatureLen": 1969,
        "drivenFeatureLen": 16394,
        "pcaDim": 512,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            fail(f"encoded model {key}: expected {expected}, got {config.get(key)!r}")
    names = config.get("driverNames")
    if not isinstance(names, list) or len(names) != 45 or any(not isinstance(x, str) or not x for x in names):
        fail("encoded model must provide exactly 45 non-empty driverNames")
    if names[0] != "Root_M" or len(set(names)) != len(names):
        fail("driverNames must be unique and begin with Root_M")
    model_hash = hashlib.sha256(data).hexdigest()
    if model_hash != EXPECTED_MODEL_SHA256:
        fail(f"model SHA256 mismatch: {model_hash}")
    return names, model_hash


def asset_path_to_file(asset_path: str) -> Path | None:
    package = asset_path.split(".", 1)[0]
    if not package.startswith("/Game/"):
        return None
    relative = package[len("/Game/") :].replace("/", os.sep) + ".uasset"
    return Path(unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_content_dir())) / relative


def finite(values: Iterable[float], label: str) -> list[float]:
    result = [float(v) for v in values]
    if not all(math.isfinite(v) for v in result):
        fail(f"non-finite values in {label}")
    return result


def quat_rotate(q, xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    try:
        out = q.rotate_vector(unreal.Vector(*xyz))
        return float(out.x), float(out.y), float(out.z)
    except Exception:
        x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
        vx, vy, vz = xyz
        tx, ty, tz = 2.0 * (y * vz - z * vy), 2.0 * (z * vx - x * vz), 2.0 * (x * vy - y * vx)
        return (
            vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx),
        )


def rotation_6d(transform) -> list[float]:
    up = quat_rotate(transform.rotation, (0.0, 0.0, 1.0))
    right = quat_rotate(transform.rotation, (0.0, 1.0, 0.0))
    return finite((*up, *right), "rotation feature")


def make_options(mesh):
    options = unreal.AnimPoseEvaluationOptions()
    properties = {
        "evaluation_type": unreal.AnimDataEvalType.RAW,
        "should_retarget": False,
        "evaluate_curves": False,
        "optional_skeletal_mesh": mesh,
    }
    for name, value in properties.items():
        try:
            options.set_editor_property(name, value)
        except Exception as exc:
            if name == "optional_skeletal_mesh":
                fail(f"cannot bind preview SkeletalMesh to pose evaluation: {exc}")
            fail(f"cannot configure AnimPoseEvaluationOptions.{name}: {exc}")
    return options


def pose_bone_names(pose) -> set[str]:
    try:
        return {str(name) for name in unreal.AnimPoseExtensions.get_bone_names(pose)}
    except Exception as exc:
        fail(f"cannot enumerate evaluated pose bones: {exc}")


def bake() -> int:
    model_path = Path(env("MLCLOTH_BAKE_MODEL")).resolve()
    output_path = Path(env("MLCLOTH_BAKE_OUTPUT")).resolve()
    mesh_path, anim_path = env("MLCLOTH_BAKE_MESH"), env("MLCLOTH_BAKE_ANIM")
    fps = int(env("MLCLOTH_BAKE_FPS"))
    if fps != 30:
        fail("this format version requires exactly 30 Hz")
    driver_names, model_hash = read_model(model_path)
    mesh = unreal.load_asset(mesh_path)
    animation = unreal.load_asset(anim_path)
    if mesh is None or not isinstance(mesh, unreal.SkeletalMesh):
        fail(f"SkeletalMesh asset cannot be loaded: {mesh_path}")
    if animation is None or not isinstance(animation, unreal.AnimSequence):
        fail(f"AnimSequence asset cannot be loaded: {anim_path}")
    try:
        duration = float(animation.get_play_length())
    except Exception:
        duration = float(animation.get_editor_property("sequence_length"))
    if not math.isfinite(duration) or duration <= 0.0:
        fail(f"animation duration is invalid: {duration}")
    frame_count = max(2, int(math.floor(duration * fps)) + 1)
    options = make_options(mesh)
    local_values: list[float] = []
    component_values: list[float] = []
    positions: list[float] = []
    first_signature: list[float] | None = None
    changed = False
    for frame in range(frame_count):
        sample_time = min(frame / float(fps), duration)
        # UE 5.8 exposes AnimPose as the return value (older examples used an
        # explicit output parameter).
        pose = unreal.AnimPoseExtensions.get_anim_pose_at_time(animation, sample_time, options)
        try:
            valid = bool(unreal.AnimPoseExtensions.is_valid(pose))
        except Exception:
            valid = True
        if not valid:
            fail(f"AnimPose evaluation failed at frame {frame} time {sample_time}")
        available = pose_bone_names(pose)
        missing = [name for name in driver_names if name not in available]
        if missing:
            fail(f"pose is missing model drivers: {missing}")
        signature: list[float] = []
        for name in driver_names:
            bone = unreal.Name(name)
            local = unreal.AnimPoseExtensions.get_bone_pose(pose, bone, unreal.AnimPoseSpaces.LOCAL)
            component = unreal.AnimPoseExtensions.get_bone_pose(pose, bone, unreal.AnimPoseSpaces.WORLD)
            local6 = rotation_6d(local)
            component6 = rotation_6d(component)
            pos = finite((component.translation.x, component.translation.y, component.translation.z), "component position")
            local_values.extend(local6)
            component_values.extend(component6)
            positions.extend(pos)
            signature.extend((*local6, *component6, *pos))
        if first_signature is None:
            first_signature = signature
        elif not changed and max(abs(a - b) for a, b in zip(first_signature, signature)) > 1.0e-5:
            changed = True
    if not changed:
        fail("animation produced no material driver change")

    expected_local = frame_count * 45 * 6
    expected_pos = frame_count * 45 * 3
    if len(local_values) != expected_local or len(component_values) != expected_local or len(positions) != expected_pos:
        fail("internal feature array count mismatch")
    payload = struct.pack(f"<{len(local_values)}f", *local_values)
    payload += struct.pack(f"<{len(component_values)}f", *component_values)
    payload += struct.pack(f"<{len(positions)}f", *positions)
    driver_hash = hashlib.sha256("\n".join(driver_names).encode("utf-8")).digest()
    payload_hash = hashlib.sha256(payload).digest()
    header = HEADER.pack(
        MAGIC, 1, HEADER.size, frame_count, fps, 1, 45, 0,
        len(local_values), len(component_values), len(positions),
        bytes.fromhex(model_hash), driver_hash, payload_hash,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(header + payload)

    anim_file, mesh_file = asset_path_to_file(anim_path), asset_path_to_file(mesh_path)
    sidecar = {
        "schema": "MLDRV001-provenance",
        "binary": str(output_path),
        "frames": frame_count,
        "fps": 30,
        "driverCount": 45,
        "rootDriverIndex": 0,
        "model": str(model_path),
        "modelSha256": model_hash,
        "driverNameListSha256": driver_hash.hex(),
        "payloadSha256": payload_hash.hex(),
        "meshAsset": mesh_path,
        "meshSourceSha256": sha256_file(mesh_file) if mesh_file and mesh_file.is_file() else None,
        "animationAsset": anim_path,
        "animationSourceSha256": sha256_file(anim_file) if anim_file and anim_file.is_file() else None,
        "durationSeconds": duration,
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
    unreal.log(f"[MLClothBake] wrote {frame_count} frames, {len(header) + len(payload)} bytes -> {output_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(bake())
    except Exception as exc:
        unreal.log_error(f"[MLClothBake] FAILED: {exc}")
        raise
