#!/usr/bin/env python3
"""Pin down the capsule bake: Unreal's rotator convention, the signed distance, the real asset.

The signed-distance function here is the reference the HLSL contact block has to
match, so its cases are hand-computed rather than snapshotted -- a snapshot would
happily lock in a wrong axis convention.

The rotator cases matter more than they look. Unreal's `FRotator` is not a plain XYZ
Euler triple, and a composition order that reads as reasonable gives a *different*
capsule axis; on the real asset that would swing the leg capsules from along-the-limb
to across it, which is exactly the kind of error that produces a plausible-looking
but wrong penetration column. So two of these cases are checked against the real
physics asset's own numbers, worked out by hand from the matrix in
`Core/Public/Math/RotationTranslationMatrix.h`.

Run directly: `py -3 tests/test_mlcloth_capsules.py`
"""

from __future__ import annotations

import math
import struct
import sys
import tempfile
from pathlib import Path

POC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC_ROOT / "tools"))

from bake_mlcloth_capsules import (  # noqa: E402
    DEFAULT_T3D,
    capsule_signed_distance,
    parse_t3d,
    rotator_axis,
)

_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        _failures.append(message)
        print(f"  FAIL: {message}")


def check_close(actual: float, expected: float, tolerance: float, message: str) -> None:
    check(abs(actual - expected) <= tolerance, f"{message}: got {actual!r}, expected {expected!r} +/- {tolerance}")


def check_vector_close(actual, expected, tolerance: float, message: str) -> None:
    for index, (a, e) in enumerate(zip(actual, expected)):
        check_close(a, e, tolerance, f"{message}[{index}]")


def test_identity_rotator_gives_local_z() -> None:
    check_vector_close(rotator_axis(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1e-9, "identity rotator axis")


def test_yaw_alone_does_not_move_the_axis() -> None:
    # Yaw is a rotation about Z, so it cannot change the Z axis. This is the case that
    # catches an implementation that mixed up which angle goes with which axis.
    for yaw in (-180.0, -90.0, 37.0, 90.0, 180.0):
        check_vector_close(rotator_axis(0.0, yaw, 0.0), (0.0, 0.0, 1.0), 1e-9, f"yaw={yaw} axis")


def test_roll_and_pitch_swing_the_axis() -> None:
    # Unreal is left-handed (X forward, Y right, Z up), and these two signs are the
    # consequence people get wrong. Roll is about X and positive roll takes Z toward
    # *+Y*, not -Y as a right-handed rotation would. Pitch is about Y and positive
    # pitch takes Z toward -X. Both read off the third row of FRotationMatrix.
    check_vector_close(rotator_axis(0.0, 0.0, 90.0), (0.0, 1.0, 0.0), 1e-9, "roll=+90 axis")
    check_vector_close(rotator_axis(0.0, 0.0, -90.0), (0.0, -1.0, 0.0), 1e-9, "roll=-90 axis")
    check_vector_close(rotator_axis(90.0, 0.0, 0.0), (-1.0, 0.0, 0.0), 1e-9, "pitch=+90 axis")
    check_vector_close(rotator_axis(-90.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1e-9, "pitch=-90 axis")


def test_real_asset_rotators_by_hand() -> None:
    """Two capsules from the real asset, with their axes derived by hand.

    Hip_L: Pitch 0, Yaw 86.865260, Roll -88.657538. With Pitch = 0 the third row of
    FRotationMatrix reduces to (-SR*SY, CY*SR, CR), so the axis is
    (-sin(R)*sin(Y), cos(Y)*sin(R), cos(R)) -- which lands on bone-local +X, i.e.
    along the thigh, which is what a hip capsule has to be.
    """
    yaw, roll = 86.865260, -88.657538
    sr, cr = math.sin(math.radians(roll)), math.cos(math.radians(roll))
    sy, cy = math.sin(math.radians(yaw)), math.cos(math.radians(yaw))
    expected = (-sr * sy, cy * sr, cr)
    norm = math.sqrt(sum(component**2 for component in expected))
    expected = tuple(component / norm for component in expected)
    check_vector_close(rotator_axis(0.0, yaw, roll), expected, 1e-9, "Hip_L axis")
    check(expected[0] > 0.99, "the Hip_L capsule runs along the limb (local +X), not across it")

    # Root_M: Pitch -0.213091, Yaw -90, Roll -0.146725. Near-identity roll and pitch
    # with a pure yaw, so the axis must stay essentially local +Z.
    axis = rotator_axis(-0.213091, -90.0, -0.146725)
    check(axis[2] > 0.9999, "the Root_M capsule stays essentially along local +Z")


def test_signed_distance_is_exact_for_hand_cases() -> None:
    center = (0.0, 0.0, 0.0)
    axis = (0.0, 0.0, 1.0)
    radius, half = 2.0, 5.0

    # Radially outside the cylinder: distance is the radial distance minus the radius.
    check_close(capsule_signed_distance((5.0, 0.0, 0.0), center, axis, radius, half), 3.0, 1e-12, "radial outside")
    # On the surface.
    check_close(capsule_signed_distance((2.0, 0.0, 3.0), center, axis, radius, half), 0.0, 1e-12, "on the surface")
    # Radially inside: negative, and the magnitude is how far in.
    check_close(capsule_signed_distance((0.5, 0.0, 0.0), center, axis, radius, half), -1.5, 1e-12, "radially inside")
    # Dead centre is the deepest point, at exactly -radius.
    check_close(capsule_signed_distance(center, center, axis, radius, half), -radius, 1e-12, "at the centre")
    # Beyond the cap: the clamp makes this the distance to the cap's sphere centre.
    check_close(capsule_signed_distance((0.0, 0.0, 9.0), center, axis, radius, half), 2.0, 1e-12, "beyond the cap")
    # Diagonally beyond the cap: 3-4-5 from the cap centre, minus the radius.
    check_close(capsule_signed_distance((3.0, 0.0, 9.0), center, axis, radius, half), 3.0, 1e-12, "diagonal past cap")


def test_signed_distance_sees_all_the_way_through() -> None:
    """The property the nearest-proxy half-plane lacks.

    `results/PROGRESS.md` section 6 records that the sibling PoC's criterion reports
    *less* penetration the deeper a patch is pushed -- 28.5% at 0.12 m against 1% at
    0.25 m -- because past the far surface the signed distance turns positive again.
    Here the distance is monotone from the surface to the axis and symmetric out the
    far side, so a vertex driven through a limb is never mistaken for a resolved one.
    """
    center, axis, radius, half = (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 10.0, 20.0
    depths = [capsule_signed_distance((x, 0.0, 0.0), center, axis, radius, half) for x in (15.0, 10.0, 5.0, 0.0)]
    check(depths == sorted(depths, reverse=True), f"distance decreases monotonically inward: {depths}")
    check_close(depths[3], -radius, 1e-12, "deepest at the axis")
    # Out the far side, and still reported as a violation of the same magnitude.
    near = capsule_signed_distance((-5.0, 0.0, 0.0), center, axis, radius, half)
    far = capsule_signed_distance((5.0, 0.0, 0.0), center, axis, radius, half)
    check_close(near, far, 1e-12, "symmetric through the axis")
    check(far < 0.0, "a vertex past the axis is still inside")


def test_real_physics_asset() -> None:
    if not DEFAULT_T3D.is_file():
        print(f"    (skipped: {DEFAULT_T3D} is not present)")
        return
    capsules, skipped = parse_t3d(DEFAULT_T3D)
    check(len(capsules) == 14, f"the runtime physics asset has 14 capsules, got {len(capsules)}")
    check(skipped == [], f"no non-capsule geometry is silently dropped, got {skipped}")
    bones = {capsule["bone"] for capsule in capsules}
    # The ones a skirt actually needs. If a future asset drops one of these the bake
    # would still succeed, so it is asserted rather than assumed.
    for bone in ("Root_M", "Hip_L", "Hip_R", "Knee_L", "Knee_R", "KneePart2_L", "KneePart2_R"):
        check(bone in bones, f"the skirt-relevant bone {bone} has a collision capsule")
    for capsule in capsules:
        radius = capsule["radius_cm"]
        length = capsule["segment_length_cm"]
        check(1.0 < radius < 30.0, f"{capsule['bone']} radius {radius} cm is a plausible body dimension")
        check(0.0 <= length < 60.0, f"{capsule['bone']} segment {length} cm is a plausible body dimension")
        axis = rotator_axis(*capsule["rotation_pyr_deg"])
        check_close(math.sqrt(sum(c * c for c in axis)), 1.0, 1e-9, f"{capsule['bone']} axis is unit length")

    # A point at the centre of the Root_M capsule must read as inside, and one a metre
    # away must not. This is the end-to-end sanity check on the parsed numbers.
    root = next(capsule for capsule in capsules if capsule["bone"] == "Root_M")
    axis = rotator_axis(*root["rotation_pyr_deg"])
    inside = capsule_signed_distance(
        root["center_cm"], root["center_cm"], axis, root["radius_cm"], 0.5 * root["segment_length_cm"]
    )
    check_close(inside, -root["radius_cm"], 1e-9, "the Root_M centre is radius-deep inside")
    far_point = tuple(component + 100.0 for component in root["center_cm"])
    outside = capsule_signed_distance(
        far_point, root["center_cm"], axis, root["radius_cm"], 0.5 * root["segment_length_cm"]
    )
    check(outside > 50.0, f"a point a metre away is well outside, got {outside}")


def test_baked_container_round_trips() -> None:
    """Run the baker and read its container back, checking the section shapes."""
    import subprocess

    model = POC_ROOT / ".work/runtime/model_NeuralRes4_NeuralRes4_final.enc"
    if not model.is_file() or not DEFAULT_T3D.is_file():
        print("    (skipped: the runtime model or the physics export is not present)")
        return
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "caps.mlcap"
        result = subprocess.run(
            [sys.executable, str(POC_ROOT / "tools/bake_mlcloth_capsules.py"), "--output", str(output)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            check(False, f"capsule bake failed: {result.stderr.strip()[:400]}")
            return
        blob = output.read_bytes()
        header = struct.Struct("<8sIIQQ32s32s")
        entry = struct.Struct("<16sQII")
        magic, version, section_count, file_bytes, payload_offset, _payload_sha, _source_sha = header.unpack_from(blob)
        check(magic == b"MLCAP001", f"capsule magic, got {magic!r}")
        check(version == 1, "capsule version")
        check(file_bytes == len(blob), "capsule fileBytes")
        sections = {}
        for index in range(section_count):
            name, offset, count, stride = entry.unpack_from(blob, header.size + entry.size * index)
            sections[name.rstrip(b"\x00").decode("ascii")] = (offset, count, stride)
        check(set(sections) == {"info", "driver", "center", "axis", "size"}, f"capsule sections, got {sorted(sections)}")
        info_offset = sections["info"][0]
        count, drivers, _r0, _r1 = struct.unpack_from("<4I", blob, info_offset)
        check(count == 14, f"capsule count in info, got {count}")
        check(drivers == 45, f"driver count in info, got {drivers}")
        for label, expected_stride in (("driver", 4), ("center", 12), ("axis", 12), ("size", 8)):
            offset, section_count_value, stride = sections[label]
            check(section_count_value == count, f"section {label} count matches the capsule count")
            check(stride == expected_stride, f"section {label} stride")
            check(offset % 16 == 0, f"section {label} is 16-byte aligned")
        # Every driver index must address the 45-entry table.
        driver_offset, _c, _s = sections["driver"]
        indices = struct.unpack_from(f"<{count}I", blob, driver_offset)
        check(all(index < drivers for index in indices), f"driver indices are in range, got {indices}")
        check(len(set(indices)) == count, "each capsule sits on a distinct driver bone")
        # Axes must arrive unit length after the round trip through float32.
        axis_offset, _c, _s = sections["axis"]
        axes = struct.unpack_from(f"<{count * 3}f", blob, axis_offset)
        for i in range(count):
            norm = math.sqrt(sum(axes[3 * i + k] ** 2 for k in range(3)))
            check_close(norm, 1.0, 1e-6, f"baked axis {i} is unit length")


def main() -> int:
    tests = [
        test_identity_rotator_gives_local_z,
        test_yaw_alone_does_not_move_the_axis,
        test_roll_and_pitch_swing_the_axis,
        test_real_asset_rotators_by_hand,
        test_signed_distance_is_exact_for_hand_cases,
        test_signed_distance_sees_all_the_way_through,
        test_real_physics_asset,
        test_baked_container_round_trips,
    ]
    for test in tests:
        print(f"  {test.__name__}...")
        before = len(_failures)
        test()
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
