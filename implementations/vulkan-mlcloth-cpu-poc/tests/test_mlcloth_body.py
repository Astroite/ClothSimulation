#!/usr/bin/env python3
"""Tests for `tools/bake_mlcloth_body.py`, on a synthesised character.

The fixture is built here rather than borrowed from the sibling PoC's export, for the same
reason the mesh tests synthesise their training export: the bake's job is to turn one set of
conventions into another, and a test that can only run where the real 11 MB character asset
happens to be checked out is a test that stops running.

What each test is actually for:

* `test_full_bake` -- the happy path, end to end, cross-checked by the C++ `parse_body`. The
  fixture's rest body is placed so that inverting the coordinate mapping must put its feet on
  z = 0 and centre it on x = 0; if the inversion were wrong the bake refuses.
* `test_fold_merges_onto_the_parent` -- a bone the model does not drive must end up folded onto
  its driven parent, and a vertex weighted to both must come out with *one* merged influence.
  Getting this wrong is not a crash: it is paying for the same bone twice and reporting an
  influence count that is a lie.
* `test_two_level_fold_is_refused` -- a bone whose parent is *also* undriven is a much larger
  approximation than folding one level, so it has to be an error rather than a silent walk up
  the hierarchy.
* `test_bad_rotator_convention_is_caught` -- the bind pose self-check. The first version of the
  bake read the rotator triple in the wrong order and missed by 189 cm while still producing a
  perfectly well-formed skeleton, so this corrupts the recorded component translations and
  asserts the bake notices.
* `test_wrong_coordinate_mapping_is_refused` -- a mapping form the bake cannot invert, and a
  mapping whose constant is wrong. The second is the dangerous one: it produces a body that is
  a body, just not standing on the floor.
"""

from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

POC_ROOT = Path(__file__).resolve().parents[1]
BAKER = POC_ROOT / "tools" / "bake_mlcloth_body.py"

SECTION_HEADER = struct.Struct("<8sIIQQ32s32s")
SECTION_ENTRY = struct.Struct("<16sQII")

DRIVERS = [
    "Root_M", "Shoulder_L", "Shoulder_R", "Elbow_L", "Elbow_R",
    "Hip_L", "Hip_R", "Knee_L", "Knee_R", "Chest_M",
    "HipPart0_L", "KneePart2_L", "Ankle_L", "KneePart1_L", "HipPart2_L",
    "ThighTwist1_L", "HipPart1_L", "HipPart0_R", "KneePart2_R", "Ankle_R",
    "KneePart1_R", "HipPart2_R", "ThighTwist1_R", "HipPart1_R", "ShoulderPart0_L",
    "ThumbFinger1_L", "Wrist_L", "ElbowPart2_L", "ElbowPart1_L", "ShoulderPart2_L",
    "ShoulderPart1_L", "Scapula_L", "Head_M", "Neck1_M", "Neck_M",
    "ShoulderPart0_R", "ThumbFinger1_R", "Wrist_R", "ElbowPart2_R", "ElbowPart1_R",
    "ShoulderPart2_R", "ShoulderPart1_R", "Scapula_R", "Spine2_M", "Spine1_M",
]
OFFSETS = (0.0070480118, 1.05862854, -0.0225656815)

_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        _failures.append(message)
        print(f"  FAIL: {message}")


def check_equal(got, want, message: str) -> None:
    check(got == want, f"{message}: got {got!r}, want {want!r}")


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def quat_from_rotator(rpy_deg):
    roll, pitch, yaw = (math.radians(v) for v in rpy_deg)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        cr * sp * sy - sr * cp * cy,
        -cr * sp * cy - sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def quat_multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_rotate(q, v):
    x, y, z, w = q
    tx, ty, tz = 2.0 * (y * v[2] - z * v[1]), 2.0 * (z * v[0] - x * v[2]), 2.0 * (x * v[1] - y * v[0])
    return (
        v[0] + w * tx + (y * tz - z * ty),
        v[1] + w * ty + (z * tx - x * tz),
        v[2] + w * tz + (x * ty - y * tx),
    )


def write_sectioned(path: Path, magic: bytes, version: int, sections: list[tuple], source_sha256: str) -> None:
    import hashlib

    directory = SECTION_HEADER.size + SECTION_ENTRY.size * len(sections)
    payload_offset = (directory + 15) // 16 * 16
    entries = []
    payload = bytearray()
    cursor = payload_offset
    for name, count, stride, blob in sections:
        pad = (-cursor) % 16
        payload.extend(b"\x00" * pad)
        cursor += pad
        entries.append((name.encode("ascii"), cursor, count, stride))
        payload.extend(blob)
        cursor += len(blob)
    body = bytes(payload)
    file_bytes = payload_offset + len(body)
    header = SECTION_HEADER.pack(
        magic, version, len(sections), file_bytes, payload_offset,
        hashlib.sha256(body).digest(), bytes.fromhex(source_sha256),
    )
    out = bytearray(header)
    for name, offset, count, stride in entries:
        out.extend(SECTION_ENTRY.pack(name, offset, count, stride))
    out.extend(b"\x00" * (payload_offset - len(out)))
    out.extend(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))


def build_skeleton(extra: list[tuple[str, str]]) -> dict:
    """A three-deep skeleton holding every driver plus `extra` (name, parent) children.

    The chain gives the bind-pose composition something to compose: every bone sits 10 cm
    above its parent with a 15-degree yaw, so a wrong rotator order accumulates.
    """
    bones = []
    index_of: dict[str, int] = {}
    order = ["Root"] + DRIVERS + [name for name, _ in extra]
    parents = {"Root": None, "Root_M": "Root"}
    for name in DRIVERS[1:]:
        parents[name] = "Root_M"
    for name, parent in extra:
        parents[name] = parent
    component: dict[str, tuple] = {}
    for index, name in enumerate(order):
        index_of[name] = index
        parent = parents[name]
        local_translation = [0.0, 0.0, 0.0] if parent is None else [1.0, 2.0, 10.0]
        local_rotation = [0.0, 0.0, 0.0] if parent is None else [0.0, 0.0, 15.0]
        rotation = quat_from_rotator(local_rotation)
        if parent is None:
            component[name] = (tuple(local_translation), rotation)
        else:
            pt, pq = component[parent]
            r = quat_rotate(pq, local_translation)
            component[name] = (
                (pt[0] + r[0], pt[1] + r[1], pt[2] + r[2]),
                quat_multiply(pq, rotation),
            )
        bones.append({
            "index": index,
            "name": name,
            "parent": parent,
            "parent_index": -1 if parent is None else index_of[parent],
            "parent_local_translation": local_translation,
            "parent_local_rotation_rpy_deg": local_rotation,
            "parent_local_scale": [1.0, 1.0, 1.0],
            "component_translation": list(component[name][0]),
        })
    return {"bone_count": len(bones), "bones": bones}


def build_character(work: Path, extra: list[tuple[str, str]], *,
                    mapping: dict | None = None,
                    shift_cm: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> tuple[str, list[str]]:
    """A four-vertex character plus its scene and skeleton sidecars.

    The rest vertices are placed in component centimetres first and then pushed *out* through
    the coordinate mapping, so the bake's inversion has a known right answer: feet on z = 0
    and symmetric about x = 0. `shift_cm` breaks that on purpose.
    """
    model = work / "model.enc"
    config = {
        "modelType": 2, "driverFeatureLen": 1969, "drivenFeatureLen": 16394,
        "pcaDim": 512, "driverNames": DRIVERS,
    }
    payload = json.dumps(config).encode("utf-8")
    model.write_bytes(struct.pack("<I", len(payload)) + payload + b"\x00" * 64)
    import hashlib
    model_sha = hashlib.sha256(model.read_bytes()).hexdigest()

    bone_names = DRIVERS + [name for name, _ in extra]
    # Component cm: two feet on the floor, two at shoulder height, mirrored in x.
    component = [
        (-20.0, 0.0, 0.0 + shift_cm[2]),
        (20.0, 0.0, 0.0 + shift_cm[2]),
        (-20.0 + shift_cm[0], 3.0, 140.0),
        (20.0 + shift_cm[0], 3.0, 140.0),
    ]
    positions = []
    for cx, cy, cz in component:
        # component -> world metres -> gnn space (subtract the offsets the mapping applies)
        wx, wy, wz = cx * 0.01, cz * 0.01, -cy * 0.01
        positions.extend((wx - OFFSETS[0], wy - OFFSETS[1], wz - OFFSETS[2]))
    normals = [0.0, 0.0, 1.0] * len(component)
    influences = 12
    bone_idx: list[int] = []
    bone_weight: list[float] = []
    for vertex in range(len(component)):
        entries: list[tuple[int, float]] = []
        if extra and vertex >= 2:
            # A vertex sharing weight between an undriven bone and its driven parent, which
            # is what the merge has to collapse.
            parent = extra[0][1]
            entries = [(bone_names.index(extra[0][0]), 0.4), (bone_names.index(parent), 0.6)]
        else:
            entries = [(0, 0.5), (bone_names.index("Chest_M"), 0.5)]
        for slot in range(influences):
            if slot < len(entries):
                bone_idx.append(entries[slot][0])
                bone_weight.append(entries[slot][1])
            else:
                bone_idx.append(0)
                bone_weight.append(0.0)
    triangles = [0, 1, 2, 1, 3, 2]

    def u32(values):
        values = list(values)
        return struct.pack(f"<{len(values)}I", *values)

    def f32(values):
        values = list(values)
        return struct.pack(f"<{len(values)}f", *values)

    write_sectioned(
        work / "ch.vchar", b"VCHAR001", 1,
        [
            ("info", 6, 4, u32([len(component), len(triangles) // 3, 0, 0, 0, 0])),
            ("render_pos", len(component), 12, f32(positions)),
            ("render_nrm", len(component), 12, f32(normals)),
            ("render_uv", len(component), 8, f32([0.0] * (2 * len(component)))),
            ("render_tri", len(triangles) // 3, 12, u32(triangles)),
            ("bone_idx", len(component), influences * 4, u32(bone_idx)),
            ("bone_weight", len(component), influences * 4, f32(bone_weight)),
        ],
        source_sha256=model_sha,
    )
    scene = {
        "coordinate_system": "right_handed_y_up_meters_waist_origin",
        "coordinate_mapping": mapping if mapping is not None else {
            "x": f"body_x - {OFFSETS[0]}",
            "y": f"body_z - {OFFSETS[1]}",
            "z": f"-body_y + {-OFFSETS[2]}",
        },
        "skinning": {"bone_names": bone_names, "maximum_runtime_influences": influences},
    }
    (work / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
    (work / "skeleton.json").write_text(json.dumps(build_skeleton(extra)), encoding="utf-8")
    return model_sha, bone_names


def run_baker(work: Path, expect_failure: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, "-B", str(BAKER),
         "--character", str(work / "ch.vchar"),
         "--scene", str(work / "scene.json"),
         "--skeleton", str(work / "skeleton.json"),
         "--model", str(work / "model.enc"),
         "--output", str(work / "out.mlbody"),
         "--report", str(work / "report.json")],
        capture_output=True, text=True,
    )
    if expect_failure:
        check(result.returncode != 0, f"baker should have refused this input\n{result.stdout}")
    else:
        check(result.returncode == 0, f"baker failed: {result.stderr or result.stdout}")
    return result


def read_body(path: Path) -> dict:
    blob = path.read_bytes()
    magic, version, count, file_bytes, payload_offset, payload_sha, source_sha = SECTION_HEADER.unpack_from(blob)
    check_equal(magic, b"MLBDY001", "body magic")
    check_equal(version, 1, "body version")
    check_equal(file_bytes, len(blob), "body fileBytes")
    import hashlib
    check_equal(hashlib.sha256(blob[payload_offset:]).digest(), payload_sha, "body payload digest")
    sections = {}
    for index in range(count):
        name, offset, entries, stride = SECTION_ENTRY.unpack_from(blob, SECTION_HEADER.size + SECTION_ENTRY.size * index)
        sections[name.rstrip(b"\x00").decode()] = (offset, entries, stride)

    def read(label, fmt, stride):
        offset, entries, found = sections[label]
        check_equal(found, stride, f"section {label} stride")
        return struct.unpack_from(f"<{entries * stride // 4}{fmt}", blob, offset)

    info = read("info", "I", 4)
    vertices, triangles, drivers, influences, folded, reserved = info
    check_equal(reserved, 0, "reserved info word")
    return {
        "vertices": vertices, "triangles": triangles, "drivers": drivers,
        "influences": influences, "folded": folded,
        "rest_pos": read("rest_pos", "f", 12),
        "bone_idx": read("bone_idx", "I", influences * 4),
        "bone_weight": read("bone_weight", "f", influences * 4),
        "inv_bind": read("inv_bind", "f", 48),
        "source_sha256": source_sha.hex(),
    }


def cross_check_with_cpp(work: Path) -> None:
    validator = os.environ.get("MLCLOTH_BODY_VALIDATE")
    if not validator or not Path(validator).is_file():
        print("    (C++ body_validate not available; skipping the cross-check)")
        return
    result = subprocess.run([validator, str(work / "model.enc"), str(work / "out.mlbody")],
                            capture_output=True, text=True)
    if result.returncode != 0:
        check(False, f"C++ parse_body rejected the Python-written body: {result.stdout}{result.stderr}")
    else:
        print("    C++ cross-check: " + result.stdout.strip().replace("\n", "\n      "))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_full_bake(work: Path) -> None:
    model_sha, _ = build_character(work, [])
    run_baker(work)
    body = read_body(work / "out.mlbody")
    check_equal(body["vertices"], 4, "vertex count")
    check_equal(body["triangles"], 2, "triangle count")
    check_equal(body["drivers"], 45, "driver count")
    check_equal(body["folded"], 0, "nothing to fold without extra bones")
    check_equal(body["source_sha256"], model_sha, "the body is sealed to the model digest")
    # The inversion, which is the point: the fixture placed the feet on the floor in component
    # centimetres and pushed them out through the mapping, so they have to come back.
    floor = min(body["rest_pos"][3 * v + 2] for v in range(4))
    span = [body["rest_pos"][3 * v] for v in range(4)]
    check(abs(floor) < 1.0e-3, f"the inverted rest body stands on z = 0 (got {floor:.6f} cm)")
    check(abs(min(span) + max(span)) < 1.0e-3, f"and is symmetric about x = 0 (got {min(span)}..{max(span)})")
    report = json.loads((work / "report.json").read_text(encoding="utf-8"))
    check(report["bind_pose_residual_cm"] < 1.0e-3,
          f"the composed bind pose matches the recorded translations ({report['bind_pose_residual_cm']})")
    cross_check_with_cpp(work)


def test_fold_merges_onto_the_parent(work: Path) -> None:
    build_character(work, [("Chest1_L", "Chest_M")])
    run_baker(work)
    body = read_body(work / "out.mlbody")
    check_equal(body["folded"], 1, "one bone folded")
    chest = DRIVERS.index("Chest_M")
    # Vertex 2 was 0.4 Chest1_L + 0.6 Chest_M. Folded and merged that is one influence of 1.0
    # on Chest_M -- not two influences of 0.4 and 0.6 on the same slot.
    width = body["influences"]
    entries = [(body["bone_idx"][2 * width + k], body["bone_weight"][2 * width + k]) for k in range(width)]
    positive = [(bone, weight) for bone, weight in entries if weight > 0.0]
    check_equal(len(positive), 1, f"the folded pair merged into one influence (got {positive})")
    check_equal(positive[0][0], chest, "onto the driven parent")
    check(abs(positive[0][1] - 1.0) < 1.0e-6, f"with the summed weight (got {positive[0][1]})")
    report = json.loads((work / "report.json").read_text(encoding="utf-8"))
    check_equal([entry["bone"] for entry in report["folded_bones"]], ["Chest1_L"], "the report names the folded bone")
    check_equal(report["folded_bones"][0]["folded_onto"], "Chest_M", "and its parent")
    cross_check_with_cpp(work)


def test_two_level_fold_is_refused(work: Path) -> None:
    # Toes_X's parent is Ankle_X, which is a driver; Toes_Y's parent is Toes_X, which is not.
    build_character(work, [("Toes_X", "Ankle_L"), ("Toes_Y", "Toes_X")])
    result = run_baker(work, expect_failure=True)
    check("Toes_Y" in result.stderr or "Toes_Y" in result.stdout,
          f"the refusal names the bone it could not place: {result.stderr}")


def test_bad_rotator_convention_is_caught(work: Path) -> None:
    build_character(work, [])
    skeleton = json.loads((work / "skeleton.json").read_text(encoding="utf-8"))
    # Stand in for a wrong rotator convention by moving the recorded translations the
    # composition is checked against. Either way the two disagree, which is the assertion.
    for bone in skeleton["bones"]:
        if bone["parent_index"] >= 0:
            bone["component_translation"][0] += 5.0
    (work / "skeleton.json").write_text(json.dumps(skeleton), encoding="utf-8")
    result = run_baker(work, expect_failure=True)
    check("bind pose" in (result.stderr + result.stdout),
          f"the refusal names the bind pose: {result.stderr}")


def test_wrong_coordinate_mapping_is_refused(work: Path) -> None:
    build_character(work, [], mapping={"x": "body_x * 2", "y": "body_z - 1.0", "z": "-body_y + 0.0"})
    result = run_baker(work, expect_failure=True)
    check("coordinate_mapping" in (result.stderr + result.stdout),
          f"a mapping form the bake cannot invert is named: {result.stderr}")


def test_offset_constant_is_checked_anatomically(work: Path) -> None:
    # The dangerous case: a well-formed mapping whose constant is stale. The body is still a
    # body, just 30 cm off the floor, and only the anatomical check catches it.
    build_character(work, [], shift_cm=(0.0, 0.0, 30.0))
    result = run_baker(work, expect_failure=True)
    check("floor" in (result.stderr + result.stdout),
          f"the refusal says the body is not on the floor: {result.stderr}")


def main() -> int:
    tests = [
        test_full_bake,
        test_fold_merges_onto_the_parent,
        test_two_level_fold_is_refused,
        test_bad_rotator_convention_is_caught,
        test_wrong_coordinate_mapping_is_refused,
        test_offset_constant_is_checked_anatomically,
    ]
    if not BAKER.is_file():
        print(f"missing baker: {BAKER}")
        return 1
    for test in tests:
        print(f"  {test.__name__}...")
        with tempfile.TemporaryDirectory() as directory:
            test(Path(directory))
        if not _failures:
            print("    ok")
    print()
    if _failures:
        print(f"{len(_failures)} failure(s)")
        return 1
    print(f"{len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
