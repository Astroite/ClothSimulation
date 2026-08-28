#!/usr/bin/env python3
"""Pin down the offline driver bake: the 6D feature, the refusals, and the Unreal match.

The assertion that matters most here is `test_matches_the_unreal_bake`. Everything else
checks the baker against a Python re-derivation that shares its assumptions; that one
compares it against a clip Unreal produced by evaluating `AnimPose`, which shares
nothing. Component positions are required to match **bit-exactly** because they are
copied rather than recomputed, so any difference at all means a layout or frame
alignment error rather than arithmetic. It is skipped with a printed note when the real
assets are absent, never silently.

Run directly: `py -3 tests/test_mlcloth_drivers.py`
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

POC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC_ROOT / "tools"))

from bake_mlcloth_drivers import (  # noqa: E402
    DRIVER_COUNT,
    HEADER,
    MAGIC,
    read_mldrv,
    rotation_6d,
)

BAKER = POC_ROOT / "tools" / "bake_mlcloth_drivers.py"
REAL_MODEL = POC_ROOT / ".work/runtime/model_NeuralRes4_NeuralRes4_final.enc"
REAL_UE_CLIP = POC_ROOT / ".work/clips/AS_C10032_ArmedSprint_Skirt.mldrv"
REAL_EXPORT_CLIP = POC_ROOT / "data/AIClothTraining_10032_Test/0000_AS_C10032_ArmedSprint_Skirt"

_failures: list[str] = []

# The 45 driver names, in the order the real model lists them. Hard-coded rather than
# read from the model so the fixture tests run on a checkout with no vendor assets.
DRIVER_NAMES = [
    "Root_M", "Shoulder_L", "Shoulder_R", "Elbow_L", "Elbow_R", "Hip_L", "Hip_R",
    "Knee_L", "Knee_R", "Chest_M", "HipPart0_L", "KneePart2_L", "Ankle_L", "KneePart1_L",
    "HipPart2_L", "ThighTwist1_L", "HipPart1_L", "HipPart0_R", "KneePart2_R", "Ankle_R",
    "KneePart1_R", "HipPart2_R", "ThighTwist1_R", "HipPart1_R", "ShoulderPart0_L",
    "ThumbFinger1_L", "Wrist_L", "ElbowPart2_L", "ElbowPart1_L", "ShoulderPart2_L",
    "ShoulderPart1_L", "Scapula_L", "Head_M", "Neck1_M", "Neck_M", "ShoulderPart0_R",
    "ThumbFinger1_R", "Wrist_R", "ElbowPart2_R", "ElbowPart1_R", "ShoulderPart2_R",
    "ShoulderPart1_R", "Scapula_R", "Spine2_M", "Spine1_M",
]


def check(condition: bool, message: str) -> None:
    if not condition:
        _failures.append(message)
        print(f"  FAIL: {message}")


def check_equal(actual, expected, message: str) -> None:
    check(actual == expected, f"{message}: got {actual!r}, expected {expected!r}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def write_model(path: Path, names: list[str] | None = None) -> str:
    config = {
        "modelType": 2,
        "driverFeatureLen": 1969,
        "drivenFeatureLen": 16394,
        "pcaDim": 512,
        "driverNames": names if names is not None else DRIVER_NAMES,
    }
    blob = json.dumps(config).encode("utf-8")
    data = struct.pack("<I", len(blob)) + blob + b"\x00" * 64
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def axis_quaternion(axis: tuple[float, float, float], degrees: float) -> tuple[float, ...]:
    half = math.radians(degrees) * 0.5
    s = math.sin(half)
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half))


def write_bone_clip(path: Path, magic: bytes, space: int, frames: int, rotate: bool = True, fps: int = 30) -> None:
    payload = magic + struct.pack("<5i", 2, fps, frames, DRIVER_COUNT, space)
    for frame in range(frames):
        angle = (90.0 * frame / max(1, frames - 1)) if rotate else 0.0
        q = axis_quaternion((1.0, 0.0, 0.0), angle)
        # When `rotate` is off the translation is held constant too, so the clip is
        # genuinely motionless. Varying only the translation would be *motion* -- a
        # character sliding without rotating still drives the cloth -- so a fixture that
        # left z ticking would not exercise the no-motion refusal at all.
        z = float(frame) if rotate else 0.0
        for index in range(DRIVER_COUNT):
            payload += struct.pack(
                "<10f",
                float(index), 0.0, z,
                q[0], q[1], q[2], q[3],
                1.0, 1.0, 1.0,
            )
    path.write_bytes(payload)


def write_clip_manifest(path: Path, frames: int, names: list[str] | None = None, fps: int = 30) -> None:
    bones = names if names is not None else DRIVER_NAMES
    path.write_text(
        json.dumps(
            {
                "fps": fps,
                "totalFrames": frames,
                "bones": [{"index": i, "name": name} for i, name in enumerate(bones)],
                "clip": {
                    "label": "Fixture_Clip",
                    "source": "/Game/Test/AS_Fixture",
                    "leadingStaticPaddingFrames": 3,
                    "trailingStaticPaddingFrames": 2,
                    "realFrameCount": frames - 5,
                },
            }
        ),
        encoding="utf-8",
    )


def build_clip(work: Path, frames: int = 8, **kwargs) -> tuple[Path, str]:
    clip = work / "0000_Fixture_Clip"
    clip.mkdir(parents=True, exist_ok=True)
    model_sha = write_model(work / "model.enc", kwargs.get("model_names"))
    write_bone_clip(clip / "bones_local.bin", b"AICBONE2", 0, frames, rotate=kwargs.get("rotate", True))
    write_bone_clip(clip / "bones_component.bin", b"AICBCMP2", 1, frames, rotate=kwargs.get("rotate", True))
    write_clip_manifest(clip / "manifest.json", frames, kwargs.get("manifest_names"))
    return clip, model_sha


def run_baker(work: Path, clip: Path, *extra: str, expect_failure: bool = False) -> subprocess.CompletedProcess:
    command = [
        sys.executable, str(BAKER),
        "--model", str(work / "model.enc"),
        "--clip", str(clip),
        "--output-dir", str(work / "out"),
        *extra,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if expect_failure:
        check(result.returncode != 0, f"baker should have refused: {' '.join(extra) or '(defaults)'}")
    elif result.returncode != 0:
        check(False, f"baker failed: {result.stderr.strip()[:400]}")
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_rotation_6d_is_the_two_expected_axes(work: Path) -> None:
    """Hand cases, because a snapshot would lock in a wrong basis choice.

    The feature is the quaternion applied to local Z then local Y, in that order. The
    order is not a free choice -- the model was trained on it -- so an implementation
    that emitted (Y, Z) would be silently wrong on every frame.
    """
    identity = (0.0, 0.0, 0.0, 1.0)
    check(
        all(abs(a - b) < 1e-12 for a, b in zip(rotation_6d(identity), (0, 0, 1, 0, 1, 0))),
        f"identity gives (Z, Y), got {rotation_6d(identity)}",
    )
    # 90 degrees about X takes Z to -Y and Y to +Z (right-handed quaternion algebra;
    # this is the maths, not Unreal's rotator convention).
    result = rotation_6d(axis_quaternion((1.0, 0.0, 0.0), 90.0))
    check(
        all(abs(a - b) < 1e-9 for a, b in zip(result, (0, -1, 0, 0, 0, 1))),
        f"90 deg about X gives (-Y, +Z), got {tuple(round(v, 6) for v in result)}",
    )
    # 180 degrees about Z flips both axes in the XY plane.
    result = rotation_6d(axis_quaternion((0.0, 0.0, 1.0), 180.0))
    check(
        all(abs(a - b) < 1e-9 for a, b in zip(result, (0, 0, 1, 0, -1, 0))),
        f"180 deg about Z leaves Z and flips Y, got {tuple(round(v, 6) for v in result)}",
    )


def test_full_bake_round_trips(work: Path) -> None:
    frames = 8
    clip, model_sha = build_clip(work, frames)
    run_baker(work, clip)
    output = work / "out" / f"{clip.name}.mldrv"
    check(output.is_file(), "the baker wrote a .mldrv")
    if not output.is_file():
        return

    blob = output.read_bytes()
    fields = HEADER.unpack_from(blob)
    check_equal(fields[0], MAGIC, "magic")
    check_equal(fields[1], 1, "version")
    check_equal(fields[2], HEADER.size, "headerBytes")
    check_equal(fields[3], frames, "frameCount is the export's training frame count")
    check_equal(fields[4], 30, "fps numerator")
    check_equal(fields[5], 1, "fps denominator")
    check_equal(fields[6], DRIVER_COUNT, "driverCount")
    check_equal(fields[7], 0, "rootDriverIndex")
    check_equal(fields[8], frames * DRIVER_COUNT * 6, "localFloatCount")
    check_equal(fields[9], frames * DRIVER_COUNT * 6, "componentFloatCount")
    check_equal(fields[10], frames * DRIVER_COUNT * 3, "positionFloatCount")
    check_equal(fields[11].hex(), model_sha, "the clip is locked to the model digest")
    check_equal(
        fields[12].hex(),
        hashlib.sha256("\n".join(DRIVER_NAMES).encode("utf-8")).hexdigest(),
        "driver name list digest",
    )
    payload = blob[HEADER.size :]
    check_equal(fields[13].hex(), hashlib.sha256(payload).hexdigest(), "payload digest covers the payload")
    check_equal(len(payload), 4 * (fields[8] + fields[9] + fields[10]), "payload length matches the declared counts")

    # Positions are copied straight from bones_component.bin, so they are checkable
    # against the fixture's own construction rather than merely self-consistent.
    parsed = read_mldrv(output)
    for frame in range(frames):
        for index in range(DRIVER_COUNT):
            base = (frame * DRIVER_COUNT + index) * 3
            check_equal(parsed["position"][base], float(index), f"frame {frame} driver {index} position x")
            check_equal(parsed["position"][base + 2], float(frame), f"frame {frame} driver {index} position z")

    sidecar = json.loads((work / "out" / f"{clip.name}.mldrv.json").read_text(encoding="utf-8"))
    check_equal(sidecar["frames"], frames, "sidecar frame count")
    check(sidecar["clothReferenceAligned"] is True, "the sidecar states the cloth reference alignment")
    check_equal(sidecar["leadingStaticPaddingFrames"], 3, "sidecar carries the padding so a consumer can trim")
    check_equal(sidecar["trailingStaticPaddingFrames"], 2, "sidecar carries the trailing padding")
    check("no Unreal" in sidecar["source"], "the sidecar records where the data came from")


def test_frames_are_one_to_one_with_the_export(work: Path) -> None:
    """Padding is kept, not trimmed, so a .mldrv frame is a cloth_sim.bin frame."""
    frames = 12
    clip, _ = build_clip(work, frames)
    run_baker(work, clip)
    parsed = read_mldrv(work / "out" / f"{clip.name}.mldrv")
    check_equal(parsed["frames"], frames, "all training frames are emitted, padding included")
    # realFrameCount is 7 here; a baker that trimmed to the authored motion would emit
    # that instead, and would silently desynchronise from the ChaosCloth reference.
    check(parsed["frames"] != frames - 5, "the bake is not trimmed to the authored motion")


def test_permuted_driver_order_is_refused(work: Path) -> None:
    """A permutation passes a set comparison and then infers without error."""
    swapped = list(DRIVER_NAMES)
    swapped[5], swapped[6] = swapped[6], swapped[5]
    clip, _ = build_clip(work, 6, manifest_names=swapped)
    result = run_baker(work, clip, expect_failure=True)
    check(
        "permutation" in result.stderr,
        f"the refusal names the problem, got: {result.stderr.strip()[:200]}",
    )
    check("index 5" in result.stderr, f"the refusal locates the first difference, got: {result.stderr.strip()[:200]}")


def test_missing_driver_is_refused(work: Path) -> None:
    renamed = list(DRIVER_NAMES)
    renamed[20] = "Not_A_Driver"
    clip, _ = build_clip(work, 6, manifest_names=renamed)
    result = run_baker(work, clip, expect_failure=True)
    check("missing model drivers" in result.stderr, f"got: {result.stderr.strip()[:200]}")


def test_motionless_clip_is_refused(work: Path) -> None:
    """A clip with constant drivers contributes a flat row to a results table."""
    clip, _ = build_clip(work, 6, rotate=False)
    result = run_baker(work, clip, expect_failure=True)
    check("never move" in result.stderr, f"got: {result.stderr.strip()[:200]}")


def test_translation_alone_counts_as_motion(work: Path) -> None:
    """The converse of the check above, so it cannot be satisfied by refusing everything.

    A character that slides without rotating still drives the cloth, so a clip whose
    rotations are constant but whose translations move must be accepted. Without this the
    no-motion refusal could be implemented as "reject unless rotations vary" and still
    pass its own test.
    """
    frames = 6
    clip = work / "0000_Fixture_Clip"
    clip.mkdir(parents=True, exist_ok=True)
    write_model(work / "model.enc")
    identity = axis_quaternion((1.0, 0.0, 0.0), 0.0)
    for name, magic, space in (("bones_local.bin", b"AICBONE2", 0), ("bones_component.bin", b"AICBCMP2", 1)):
        payload = magic + struct.pack("<5i", 2, 30, frames, DRIVER_COUNT, space)
        for frame in range(frames):
            for index in range(DRIVER_COUNT):
                payload += struct.pack(
                    "<10f", float(index), 0.0, 4.0 * frame, *identity, 1.0, 1.0, 1.0
                )
        (clip / name).write_bytes(payload)
    write_clip_manifest(clip / "manifest.json", frames)
    run_baker(work, clip)
    parsed = read_mldrv(work / "out" / f"{clip.name}.mldrv")
    check_equal(parsed["frames"], frames, "a translating-only clip is accepted")
    check_equal(parsed["position"][2], 0.0, "frame 0 z")
    check_equal(parsed["position"][(frames - 1) * DRIVER_COUNT * 3 + 2], 4.0 * (frames - 1), "last frame z")


def test_bone_clip_header_is_validated(work: Path) -> None:
    frames = 6
    clip, _ = build_clip(work, frames)

    # Wrong magic: bones_local must not be a component-space clip.
    write_bone_clip(clip / "bones_local.bin", b"AICBCMP2", 0, frames)
    result = run_baker(work, clip, expect_failure=True)
    check("magic is" in result.stderr, f"wrong magic is refused, got: {result.stderr.strip()[:200]}")

    # Right magic, wrong space flag.
    write_bone_clip(clip / "bones_local.bin", b"AICBONE2", 1, frames)
    result = run_baker(work, clip, expect_failure=True)
    check("spaceFlag" in result.stderr, f"wrong space flag is refused, got: {result.stderr.strip()[:200]}")

    # Frame counts that disagree between the two files.
    write_bone_clip(clip / "bones_local.bin", b"AICBONE2", 0, frames)
    write_bone_clip(clip / "bones_component.bin", b"AICBCMP2", 1, frames + 1)
    result = run_baker(work, clip, expect_failure=True)
    check("frames but" in result.stderr, f"a frame count disagreement is refused, got: {result.stderr.strip()[:200]}")

    # A frame rate other than 30 is refused rather than resampled.
    write_bone_clip(clip / "bones_component.bin", b"AICBCMP2", 1, frames)
    write_bone_clip(clip / "bones_local.bin", b"AICBONE2", 0, frames, fps=60)
    result = run_baker(work, clip, expect_failure=True)
    check("requires exactly 30" in result.stderr, f"60 Hz is refused, got: {result.stderr.strip()[:200]}")

    # Truncation.
    write_bone_clip(clip / "bones_local.bin", b"AICBONE2", 0, frames)
    blob = (clip / "bones_local.bin").read_bytes()
    (clip / "bones_local.bin").write_bytes(blob[: len(blob) - 40])
    result = run_baker(work, clip, expect_failure=True)
    check("bytes, expected" in result.stderr, f"truncation is refused, got: {result.stderr.strip()[:200]}")


def test_manifest_frame_count_must_agree(work: Path) -> None:
    frames = 6
    clip, _ = build_clip(work, frames)
    write_clip_manifest(clip / "manifest.json", frames + 4)
    result = run_baker(work, clip, expect_failure=True)
    check("manifest declares" in result.stderr, f"got: {result.stderr.strip()[:200]}")


def test_matches_the_unreal_bake(work: Path) -> None:
    """The assertion this whole tool rests on: agreement with an Unreal-produced clip.

    Positions are compared at zero tolerance. The 6D arrays get a few float32 ulps,
    because they are recomputed from a stored quaternion rather than copied.
    """
    missing = [path for path in (REAL_MODEL, REAL_UE_CLIP, REAL_EXPORT_CLIP) if not path.exists()]
    if missing:
        print(f"    (skipped: {', '.join(path.name for path in missing)} not present)")
        return
    result = subprocess.run(
        [
            sys.executable, str(BAKER),
            "--model", str(REAL_MODEL),
            "--clip", str(REAL_EXPORT_CLIP),
            "--output-dir", str(work / "real"),
            "--cross-check", str(REAL_UE_CLIP),
            "--cross-check-clip", "AS_C10032_ArmedSprint_Skirt",
        ],
        capture_output=True,
        text=True,
    )
    check(result.returncode == 0, f"cross-check against the Unreal bake failed: {result.stdout}{result.stderr}")
    check(
        "cross-check passed at frame offset 34" in result.stdout,
        f"the offset came from the clip's leading padding, got: {result.stdout.strip()[-300:]}",
    )
    for line in result.stdout.splitlines():
        if "position" in line or "6D" in line:
            print(f"    {line.strip()}")


def main() -> int:
    tests = [
        test_rotation_6d_is_the_two_expected_axes,
        test_full_bake_round_trips,
        test_frames_are_one_to_one_with_the_export,
        test_permuted_driver_order_is_refused,
        test_missing_driver_is_refused,
        test_motionless_clip_is_refused,
        test_translation_alone_counts_as_motion,
        test_bone_clip_header_is_validated,
        test_manifest_frame_count_must_agree,
        test_matches_the_unreal_bake,
    ]
    for test in tests:
        print(f"  {test.__name__}...")
        before = len(_failures)
        with tempfile.TemporaryDirectory() as directory:
            test(Path(directory))
        if len(_failures) == before:
            print("    ok")
    print()
    if _failures:
        print(f"{len(_failures)} failed")
        return 1
    print(f"{len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
