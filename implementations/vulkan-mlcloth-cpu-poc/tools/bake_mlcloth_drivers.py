#!/usr/bin/env python3
"""Bake MLDRV001 driver clips from an MLCloth training export, with no Unreal.

Why this can exist at all
-------------------------
`tools/bake_driver_clip_unreal.py` evaluates `AnimPose` inside the editor to build
the three driver features the model wants. The training export already contains the
same information: `bones_local.bin` and `bones_component.bin` carry, per frame, a
full transform for each of 45 bones, and that bone order is **byte-identical to the
model's own `driverNames`** (checked element by element, not assumed). The 6D
rotation feature is just the quaternion applied to two basis vectors, so all three
features are recoverable offline.

That claim is not taken on faith. `--cross-check` compares a bake against an
Unreal-produced `.mldrv` for the same animation, and on
`AS_C10032_ArmedSprint_Skirt` the component positions come out **bit-exact** while
the two 6D arrays differ by 1.5e-06 -- a dozen float32 ulps from recomputing axes
out of a stored quaternion. `tests/test_mlcloth_drivers.py` keeps that comparison
running.

Frame alignment
---------------
Clips are emitted at **exactly the training frame count**, one frame per
`cloth_sim.bin` frame, so frame *i* of a `.mldrv` and frame *i* of the ChaosCloth
reference are the same instant. That is what makes the reference usable for
calibration, and it is why the leading and trailing static padding the exporter
adds is kept rather than trimmed: trimming would silently shift the two apart. The
padding ranges go into the sidecar so a consumer can exclude them deliberately.

Formats, decoded from AIClothTrainingAnimationBuilder.cpp (little-endian; magic is
8 bytes with no terminator):

    bones_local.bin     -- "AICBONE2", spaceFlag 0 (parent-local)
    bones_component.bin -- "AICBCMP2", spaceFlag 1 (component space)
        i32 version=2, i32 fps, i32 frameCount, i32 boneCount, i32 spaceFlag,
        then frameCount * boneCount transforms of ten floats:
        translation xyz, rotation xyzw, scale xyz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

POC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC_ROOT / "tools"))

from bake_cloth_topology import read_model as read_model_strict  # noqa: E402

MAGIC = b"MLDRV001"
HEADER = struct.Struct("<8s10I32s32s32s")
DRIVER_COUNT = 45
ROOT_DRIVER_INDEX = 0
FPS = 30
BONE_STRIDE = 40
BONE_HEADER_BYTES = 28

# The residual between this bake and Unreal's is float32 rounding in the quaternion to
# axis step, so the tolerance is a small multiple of the epsilon of a unit-magnitude
# float32 rather than a number tuned until the test passed. Positions must be exact:
# they are copied, not recomputed, so any difference at all means a layout or frame
# alignment error rather than arithmetic.
ROTATION_TOLERANCE = 3.0e-06
POSITION_TOLERANCE = 0.0


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"bake_mlcloth_drivers: {message}")


def read_model_drivers(path: Path) -> tuple[list[str], str]:
    """Driver names and the model digest, after the shared strict field check.

    The field validation lives in `bake_cloth_topology.read_model` and is reused rather
    than repeated, so there is one definition of "a model this PoC accepts".
    """
    read_model_strict(path)
    data = path.read_bytes()
    json_bytes = struct.unpack_from("<I", data, 0)[0]
    config = json.loads(data[4 : 4 + json_bytes].decode("utf-8"))
    names = config.get("driverNames")
    if not isinstance(names, list) or len(names) != DRIVER_COUNT:
        fail(f"encoded model lists {names if not isinstance(names, list) else len(names)} drivers, expected {DRIVER_COUNT}")
    if names[ROOT_DRIVER_INDEX] != "Root_M":
        fail(f"driver {ROOT_DRIVER_INDEX} is {names[ROOT_DRIVER_INDEX]!r}, expected 'Root_M'")
    return [str(name) for name in names], hashlib.sha256(data).hexdigest()


def read_bone_clip(path: Path, magic: bytes, space_flag: int) -> tuple[int, int, bytes]:
    """Frame count, bone count and raw blob of a bone clip, with the header checked."""
    if not path.is_file():
        fail(f"missing bone clip: {path}")
    blob = path.read_bytes()
    if len(blob) < BONE_HEADER_BYTES:
        fail(f"{path.name} is truncated at {len(blob)} bytes")
    if blob[:8] != magic:
        fail(f"{path.name} magic is {blob[:8]!r}, expected {magic!r}")
    version, fps, frames, bones, space = struct.unpack_from("<5i", blob, 8)
    if version != 2:
        fail(f"{path.name} declares version {version}, expected 2")
    if fps != FPS:
        fail(f"{path.name} is {fps} Hz; this format version requires exactly {FPS}")
    if space != space_flag:
        fail(f"{path.name} declares spaceFlag {space}, expected {space_flag}")
    if bones != DRIVER_COUNT:
        fail(f"{path.name} carries {bones} bones, expected {DRIVER_COUNT}")
    if frames <= 0:
        fail(f"{path.name} declares {frames} frames")
    expected = BONE_HEADER_BYTES + BONE_STRIDE * bones * frames
    if len(blob) != expected:
        fail(f"{path.name} is {len(blob)} bytes, expected {expected} for {frames} frames of {bones} bones")
    return frames, bones, blob


def bone_transform(blob: bytes, bones: int, frame: int, index: int):
    offset = BONE_HEADER_BYTES + BONE_STRIDE * (bones * frame + index)
    values = struct.unpack_from("<10f", blob, offset)
    return values[0:3], values[3:7], values[7:10]


def quaternion_rotate(q, v):
    x, y, z, w = q
    vx, vy, vz = v
    tx, ty, tz = 2.0 * (y * vz - z * vy), 2.0 * (z * vx - x * vz), 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def rotation_6d(q) -> tuple[float, ...]:
    """The model's rotation feature: the quaternion applied to local Z then local Y.

    Same two vectors, in the same order, as `bake_driver_clip_unreal.py::rotation_6d`.
    Two axes rather than a full matrix because the third is their cross product, so this
    is a minimal non-degenerate rotation encoding -- and because it is what the model
    was trained on, which is not negotiable.
    """
    return (*quaternion_rotate(q, (0.0, 0.0, 1.0)), *quaternion_rotate(q, (0.0, 1.0, 0.0)))


def read_clip_manifest(clip_dir: Path) -> dict:
    path = clip_dir / "manifest.json"
    if not path.is_file():
        fail(f"missing manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def check_driver_order(manifest: dict, driver_names: list[str], label: str) -> None:
    """The export's bone order must equal the model's driver order, element by element.

    A set comparison would pass on a permutation, and a permuted driver feature vector is
    exactly the kind of input that produces plausible-looking garbage rather than an
    error, so the check is on order.
    """
    bones = manifest.get("bones")
    if not isinstance(bones, list):
        fail(f"{label}: manifest has no bones array")
    ordered = [str(entry["name"]) for entry in sorted(bones, key=lambda entry: entry["index"])]
    if ordered != driver_names:
        if sorted(ordered) == sorted(driver_names):
            first = next(i for i, (a, b) in enumerate(zip(ordered, driver_names)) if a != b)
            fail(
                f"{label}: the export's bones are a permutation of the model's drivers, first differing at "
                f"index {first} ({ordered[first]!r} vs {driver_names[first]!r}). Refusing: a permuted driver "
                "vector would infer without error."
            )
        missing = [name for name in driver_names if name not in ordered]
        fail(f"{label}: the export is missing model drivers {missing}")


def bake_clip(clip_dir: Path, driver_names: list[str], model_sha256: str, output: Path) -> dict:
    manifest = read_clip_manifest(clip_dir)
    label = manifest.get("clip", {}).get("label") or clip_dir.name
    check_driver_order(manifest, driver_names, label)
    if manifest.get("fps") != FPS:
        fail(f"{label}: the export is {manifest.get('fps')} Hz; this format version requires exactly {FPS}")

    local_frames, bones, local_blob = read_bone_clip(clip_dir / "bones_local.bin", b"AICBONE2", 0)
    comp_frames, _bones, comp_blob = read_bone_clip(clip_dir / "bones_component.bin", b"AICBCMP2", 1)
    if local_frames != comp_frames:
        fail(f"{label}: bones_local has {local_frames} frames but bones_component has {comp_frames}")
    declared = manifest.get("totalFrames")
    if declared is not None and declared != local_frames:
        fail(f"{label}: manifest declares {declared} frames but the bone clips carry {local_frames}")

    local_values: list[float] = []
    component_values: list[float] = []
    positions: list[float] = []
    first: list[float] | None = None
    changed = False
    for frame in range(local_frames):
        signature: list[float] = []
        for index in range(bones):
            _lt, lq, _ls = bone_transform(local_blob, bones, frame, index)
            ct, cq, _cs = bone_transform(comp_blob, bones, frame, index)
            local6 = rotation_6d(lq)
            component6 = rotation_6d(cq)
            for value in (*local6, *component6, *ct):
                if not math.isfinite(value):
                    fail(f"{label}: frame {frame} driver {driver_names[index]} has a non-finite feature")
            local_values.extend(local6)
            component_values.extend(component6)
            positions.extend(ct)
            signature.extend((*local6, *component6, *ct))
        if first is None:
            first = signature
        elif not changed and max(abs(a - b) for a, b in zip(first, signature)) > 1.0e-5:
            changed = True
    if not changed:
        # A clip whose drivers never move drives the network with a constant input, so it
        # cannot say anything about the front end. Better to refuse than to contribute a
        # flat row to a results table.
        fail(f"{label}: the drivers never move, so this clip carries no motion")

    payload = struct.pack(f"<{len(local_values)}f", *local_values)
    payload += struct.pack(f"<{len(component_values)}f", *component_values)
    payload += struct.pack(f"<{len(positions)}f", *positions)
    driver_hash = hashlib.sha256("\n".join(driver_names).encode("utf-8")).digest()
    payload_hash = hashlib.sha256(payload).digest()
    header = HEADER.pack(
        MAGIC, 1, HEADER.size, local_frames, FPS, 1, DRIVER_COUNT, ROOT_DRIVER_INDEX,
        len(local_values), len(component_values), len(positions),
        bytes.fromhex(model_sha256), driver_hash, payload_hash,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(header + payload)

    clip = manifest.get("clip", {})
    sidecar = {
        "schema": "MLDRV001-provenance",
        "binary": str(output),
        "source": "MLCloth training export bones_local.bin + bones_component.bin (no Unreal)",
        "frames": local_frames,
        "fps": FPS,
        "driverCount": DRIVER_COUNT,
        "rootDriverIndex": ROOT_DRIVER_INDEX,
        "modelSha256": model_sha256,
        "driverNameListSha256": driver_hash.hex(),
        "payloadSha256": payload_hash.hex(),
        "clipLabel": label,
        "animationAsset": clip.get("source"),
        "exportDirectory": str(clip_dir),
        "bonesLocalSha256": hashlib.sha256(local_blob).hexdigest(),
        "bonesComponentSha256": hashlib.sha256(comp_blob).hexdigest(),
        # Frames are 1:1 with this clip's cloth_sim.bin, padding included, so the
        # ChaosCloth reference lines up frame for frame. A consumer that wants only the
        # authored motion excludes these ranges itself rather than guessing.
        "clothReferenceAligned": True,
        "leadingStaticPaddingFrames": clip.get("leadingStaticPaddingFrames"),
        "trailingStaticPaddingFrames": clip.get("trailingStaticPaddingFrames"),
        "realFrameCount": clip.get("realFrameCount"),
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return sidecar


def read_mldrv(path: Path) -> dict:
    blob = path.read_bytes()
    if len(blob) < HEADER.size or blob[:8] != MAGIC:
        fail(f"{path.name} is not an MLDRV001 clip")
    fields = HEADER.unpack_from(blob)
    frames, local_count, component_count, position_count = fields[3], fields[8], fields[9], fields[10]
    offset = HEADER.size
    local = struct.unpack_from(f"<{local_count}f", blob, offset)
    offset += 4 * local_count
    component = struct.unpack_from(f"<{component_count}f", blob, offset)
    offset += 4 * component_count
    position = struct.unpack_from(f"<{position_count}f", blob, offset)
    return {"frames": frames, "local": local, "component": component, "position": position}


def cross_check(mine: Path, reference: Path, offset: int) -> int:
    """Compare this bake against an Unreal-produced clip, starting at frame `offset`.

    The offset exists because the exporter pads a clip with static frames while the
    Unreal bake covers only the authored animation, so the two start at different
    instants. `--cross-check-offset -1` derives it from the sidecar's
    `leadingStaticPaddingFrames`, which is the value that made positions match exactly.
    """
    a, b = read_mldrv(mine), read_mldrv(reference)
    if offset < 0:
        sidecar = mine.with_suffix(mine.suffix + ".json")
        if not sidecar.is_file():
            fail(f"cannot derive the frame offset: no sidecar beside {mine.name}")
        offset = json.loads(sidecar.read_text(encoding="utf-8")).get("leadingStaticPaddingFrames") or 0
    if offset + b["frames"] > a["frames"]:
        fail(
            f"the reference has {b['frames']} frames and the offset is {offset}, which needs "
            f"{offset + b['frames']} frames but this bake has {a['frames']}"
        )

    failures = []
    for name, per_frame, tolerance in (
        ("local 6D", DRIVER_COUNT * 6, ROTATION_TOLERANCE),
        ("component 6D", DRIVER_COUNT * 6, ROTATION_TOLERANCE),
        ("component position cm", DRIVER_COUNT * 3, POSITION_TOLERANCE),
    ):
        key = {"local 6D": "local", "component 6D": "component", "component position cm": "position"}[name]
        start = offset * per_frame
        length = b["frames"] * per_frame
        mine_slice = a[key][start : start + length]
        worst = max(abs(x - y) for x, y in zip(mine_slice, b[key]))
        squares = sum((x - y) ** 2 for x, y in zip(mine_slice, b[key])) / length
        status = "ok" if worst <= tolerance else "FAILED"
        print(f"  {name:<22} max_abs={worst:.3e}  rms={math.sqrt(squares):.3e}  tol={tolerance:.1e}  {status}")
        if worst > tolerance:
            failures.append(f"{name} differs by {worst:.3e} (tolerance {tolerance:.1e})")
    if failures:
        print("cross-check FAILED: " + "; ".join(failures), file=sys.stderr)
        return 1
    print(f"  cross-check passed at frame offset {offset}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="the encoded .enc model to lock clips to")
    parser.add_argument("--clip", type=Path, action="append", default=None, help="a single export clip directory")
    parser.add_argument("--export-root", type=Path, default=None, help="a dataset root; every clip under it is baked")
    parser.add_argument("--output-dir", type=Path, required=True, help="where the .mldrv files go")
    parser.add_argument("--report", type=Path, default=None, help="JSON summary of everything baked")
    parser.add_argument(
        "--cross-check",
        type=Path,
        default=None,
        help="an Unreal-produced .mldrv to compare the matching bake against",
    )
    parser.add_argument(
        "--cross-check-clip",
        type=str,
        default=None,
        help="which baked clip label the --cross-check reference corresponds to",
    )
    parser.add_argument(
        "--cross-check-offset",
        type=int,
        default=-1,
        help="frame offset into this bake; -1 (default) takes the clip's leading static padding",
    )
    return parser.parse_args()


def discover_clips(export_root: Path) -> list[Path]:
    """Clip directories under a dataset root, excluding the reference pose clip.

    The T-pose reference holds a single pose, so it has no driver motion and would be
    refused by the no-motion check; skipping it here keeps a whole-dataset bake from
    failing on a directory nobody wanted baked.
    """
    found = []
    for candidate in sorted(export_root.iterdir()):
        if not candidate.is_dir() or "TPose" in candidate.name:
            continue
        if (candidate / "bones_local.bin").is_file() and (candidate / "bones_component.bin").is_file():
            found.append(candidate)
    if not found and (export_root / "bones_local.bin").is_file():
        found = [export_root]
    return found


def main() -> int:
    args = parse_args()
    driver_names, model_sha256 = read_model_drivers(args.model)

    clips: list[Path] = list(args.clip or [])
    if args.export_root is not None:
        clips.extend(discover_clips(args.export_root))
    if not clips:
        fail("no clip directory given; pass --clip or --export-root")

    baked = []
    for clip_dir in clips:
        output = args.output_dir / f"{clip_dir.name}.mldrv"
        sidecar = bake_clip(clip_dir, driver_names, model_sha256, output)
        baked.append(sidecar)
        print(f"  {sidecar['clipLabel']:<46} {sidecar['frames']:>4} frames -> {output.name}")
    print(f"baked {len(baked)} clip(s), {sum(entry['frames'] for entry in baked)} frames total")

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "schema": "MLDRV001-batch",
                    "model": str(args.model.resolve()),
                    "modelSha256": model_sha256,
                    "driverNames": driver_names,
                    "clips": baked,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    if args.cross_check is not None:
        if args.cross_check_clip is None:
            fail("--cross-check needs --cross-check-clip to say which baked clip it corresponds to")
        matches = [entry for entry in baked if args.cross_check_clip in (entry["clipLabel"], Path(entry["binary"]).stem)]
        if not matches:
            fail(f"--cross-check-clip {args.cross_check_clip!r} was not among the clips baked")
        print(f"cross-checking {matches[0]['clipLabel']} against {args.cross_check.name}:")
        return cross_check(Path(matches[0]["binary"]), args.cross_check, args.cross_check_offset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
