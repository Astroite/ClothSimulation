#!/usr/bin/env python3
"""Bake the character's render body into `MLBDY001`, skinnable from a `.mldrv` clip.

Why a real body and not the capsules
------------------------------------
The capsules are what the solver collides against, and they are the right thing to
*measure* penetration with, but they are a ragdoll envelope several centimetres wider
than the skin. Judging a garment by eye against them is misleading in both directions:
cloth resting on the chest looks like it is floating, and cloth a centimetre inside the
skin looks fine. So the capsules stay as the collision geometry and this is what gets
drawn.

Where the body comes from, and why nothing has to be re-exported
----------------------------------------------------------------
The sibling GNN PoC already exports this character's render mesh with skinning:
`ch10032.vchar` (VCHAR001) carries 67,857 vertices, 128,988 triangles, rest normals,
and twelve bone influences per vertex over a 45-bone list collapsed from the rig's
705 referenced bones. What it does *not* carry is a way to pose it from MLCloth data,
because its skin matrices are baked per animation frame in its own space.

This bake supplies that. Three facts make it exact rather than approximate, and each
one is checked here rather than assumed:

1. **The spaces are the same up to a constant.** `scene.json` records
   `coordinate_mapping` as `x = body_x - 0.00705, y = body_z - 1.05863, z = -body_y +
   0.02257` over `right_handed_y_up_meters_waist_origin`. That axis permutation is
   exactly the one `mlclothcpu.cpp::buildPointData` applies to reach world space
   (`world = (c.x, c.z, -c.y) * 0.01`), so inverting it lands in Unreal component
   centimetres -- the space the driver clip's bone transforms are already in. The check
   is anatomical and unforgiving: inverted, the rest body must stand with its feet on
   z = 0 and be symmetric about x = 0.

2. **The bind pose is recoverable exactly.** `skeleton.json` gives every one of the
   1,016 bones its parent-local translation and rotator plus its component translation.
   Composing the hierarchy reproduces the recorded component translations to 1.2e-4 cm,
   which is the check -- and it is a real one: the first attempt read the rotator triple
   as (yaw, roll, pitch) instead of (roll, pitch, yaw) and missed by 189 cm while still
   producing a perfectly well-formed skeleton.

3. **41 of the character's 45 bones are MLCloth drivers.** The four that are not --
   `Chest1_L`, `Chest1_R`, `Toes_L`, `Toes_R` -- each have a *driven direct parent*
   (`Chest_M`, `Chest_M`, `Ankle_L`, `Ankle_R`), so their influences are folded onto
   that parent. This is the one approximation in the bake and it is not free: measured
   against the sibling PoC's own per-frame skin matrices over five clips, the fold moves
   the affected vertices by up to 2.2 cm on the breast patch (0.34% of all skin weight
   each side) and up to 5.2 cm at the toes (3.4% each). The toes are 60 cm from the
   nearest garment vertex; the breast patch is under the bodice, so chest clearance has
   to be read off the numbers rather than the picture. Adding those four bones to the
   training export's driver list would remove the approximation entirely.

The registration is validated end to end, which is what the numbers in the report are:
skinning the body with a clip's own driver transforms and measuring the distance from
ChaosCloth's own garment for the same frame to the nearest skin vertex gives p05 0.4 cm
and p50 2.2-2.7 cm across run, sprint, lunge and death poses -- and the sibling PoC
independently measured its own cloth-to-body distance at a 2.41 cm median. A wrong bind
pose or bone order does not produce a stable 2.4 cm across poses; it produces a body
that changes height with the animation.

Output layout (`MLBDY001`, the sectioned container shared with the other assets):

    info        6 x 4   u32: vertices, triangles, drivers, influences, folded, reserved
    rest_pos    V x 12  float3, component cm at the bind pose
    rest_nrm    V x 12  float3, unit, same frame
    bone_idx    V x 4*I  I x u32, already remapped to driver slots 0..44
    bone_weight V x 4*I  I x f32, normalised, descending
    tri         T x 12  uint3
    inv_bind    45 x 48 12 x f32, row-major 3x4 inverse bind matrix per driver slot

`I` is the measured maximum influence count, not a constant: merging the folded bones onto
parents that were frequently already influencing the same vertex takes this body from the
source's twelve to five, and that width is what the per-frame skinning pass loops over.

`sourceSha256` carries the model digest, the same lock the mesh and the capsules use, so
a body can never be paired with a different `.enc`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path

POC_ROOT = Path(__file__).resolve().parents[1]
GNN_POC = POC_ROOT.parent / "vulkan-gnn-poc"
DEFAULT_CHAR = GNN_POC / ".work/real_scene/ch10032_tpose/ch10032.vchar"
DEFAULT_SCENE = GNN_POC / ".work/real_scene/ch10032_tpose/scene.json"
DEFAULT_SKELETON = GNN_POC / ".work/ch10032_library/data/skeleton.json"

MAGIC = b"MLBDY001"
VERSION = 1
DRIVER_COUNT = 45
INFLUENCES = 12

SECTION_HEADER = struct.Struct("<8sIIQQ32s32s")
SECTION_ENTRY = struct.Struct("<16sQII")


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"bake_mlcloth_body: {message}")


def load_write_sectioned():
    if not (GNN_POC / "real_scene" / "formats.py").is_file():
        fail(
            f"the sibling GNN PoC is required for the container writer but was not found at {GNN_POC}. "
            "It supplies real_scene/formats.py::write_sectioned."
        )
    sys.path.insert(0, str(GNN_POC))
    from real_scene.formats import Section, write_sectioned  # noqa: E402

    return Section, write_sectioned


# ---------------------------------------------------------------------------
# Quaternion helpers. Deliberately the same expressions the runtime uses, so a
# disagreement is a disagreement about data and not about arithmetic.
# ---------------------------------------------------------------------------


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


def quat_conjugate(q):
    return (-q[0], -q[1], -q[2], q[3])


def quat_from_rotator(rpy_deg):
    """Unreal's `FRotator::Quaternion()`, from a (roll, pitch, yaw) triple in degrees.

    Written out rather than composed from three axis rotations for the reason the
    capsule bake gives about the same conversion: Unreal's rotator is not a plain XYZ
    Euler triple, and an independently reasonable composition order silently produces a
    different rotation. The composed hierarchy is checked against the recorded component
    translations, which is what caught the first version of this.
    """
    roll, pitch, yaw = (math.radians(value) for value in rpy_deg)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        cr * sp * sy - sr * cp * cy,
        -cr * sp * cy - sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def read_sectioned(path: Path, magic: bytes) -> tuple[bytes, dict[str, tuple[int, int, int]]]:
    blob = path.read_bytes()
    if len(blob) < SECTION_HEADER.size:
        fail(f"{path.name} is too small to be a sectioned container")
    found, version, count, file_bytes, payload_offset, payload_sha, _source = SECTION_HEADER.unpack_from(blob)
    if found != magic:
        fail(f"{path.name} magic is {found!r}, expected {magic!r}")
    if file_bytes != len(blob):
        fail(f"{path.name} declares {file_bytes} bytes but is {len(blob)}")
    if hashlib.sha256(blob[payload_offset:]).digest() != payload_sha:
        fail(f"{path.name} payload digest does not match its header")
    sections: dict[str, tuple[int, int, int]] = {}
    for index in range(count):
        name, offset, entries, stride = SECTION_ENTRY.unpack_from(
            blob, SECTION_HEADER.size + SECTION_ENTRY.size * index
        )
        label = name.rstrip(b"\x00").decode("ascii")
        if label in sections:
            fail(f"{path.name} has a duplicate section {label}")
        if offset < payload_offset or offset + entries * stride > len(blob):
            fail(f"{path.name} section {label} is out of bounds")
        sections[label] = (offset, entries, stride)
    return blob, sections


def section_values(blob, sections, label, fmt, stride) -> tuple:
    if label not in sections:
        fail(f"the character asset has no {label} section")
    offset, count, found = sections[label]
    if found != stride:
        fail(f"section {label} stride is {found}, expected {stride}")
    return struct.unpack_from(f"<{count * stride // 4}{fmt}", blob, offset), count


def read_driver_names(path: Path) -> list[str]:
    data = path.read_bytes()
    if len(data) < 5:
        fail(f"encoded model is truncated: {path}")
    json_bytes = struct.unpack_from("<I", data, 0)[0]
    if json_bytes <= 1 or 4 + json_bytes >= len(data):
        fail(f"encoded model JSON length is invalid: {path}")
    config = json.loads(data[4 : 4 + json_bytes].decode("utf-8"))
    names = config.get("driverNames")
    if not isinstance(names, list) or len(names) != DRIVER_COUNT:
        fail(f"encoded model must declare exactly {DRIVER_COUNT} driverNames")
    if names[0] != "Root_M":
        fail("the first driver must be Root_M")
    return [str(name) for name in names]


def compose_bind_pose(skeleton: list[dict]) -> tuple[dict[str, tuple], float]:
    """Component-space bind transform per bone, plus the worst self-check residual.

    The residual is against the `component_translation` the same file records, which is
    an independent quantity: it was read off the mesh, while this is composed from the
    parent-local chain. They have to agree.
    """
    ordered = sorted(skeleton, key=lambda bone: bone["index"])
    if [bone["index"] for bone in ordered] != list(range(len(ordered))):
        fail("skeleton bone indices are not a dense 0..n-1 range")
    name_of = [bone["name"] for bone in ordered]
    pose: dict[str, tuple] = {}
    for bone in ordered:
        rotation = quat_from_rotator(bone["parent_local_rotation_rpy_deg"])
        translation = tuple(bone["parent_local_translation"])
        scale = bone.get("parent_local_scale", [1.0, 1.0, 1.0])
        if any(abs(value - 1.0) > 1.0e-4 for value in scale):
            fail(f"bone {bone['name']} has non-unit bind scale {scale}, which this bake does not carry")
        parent = bone["parent_index"]
        if parent < 0:
            pose[bone["name"]] = (translation, rotation)
        else:
            parent_translation, parent_rotation = pose[name_of[parent]]
            offset = quat_rotate(parent_rotation, translation)
            pose[bone["name"]] = (
                tuple(parent_translation[axis] + offset[axis] for axis in range(3)),
                quat_multiply(parent_rotation, rotation),
            )
    worst = 0.0
    for bone in ordered:
        got = pose[bone["name"]][0]
        want = bone["component_translation"]
        worst = max(worst, math.dist(got, want))
    return pose, worst


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--character", type=Path, default=DEFAULT_CHAR, help="sibling PoC's ch10032.vchar")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="its scene.json, for the bone list and the space")
    parser.add_argument("--skeleton", type=Path, default=DEFAULT_SKELETON, help="skeleton.json, for the bind pose")
    parser.add_argument("--model", type=Path, default=POC_ROOT / ".work/runtime/model_NeuralRes4_NeuralRes4_final.enc")
    parser.add_argument("--output", type=Path, default=POC_ROOT / ".work/body/ch10032_body.mlbody")
    parser.add_argument("--report", type=Path, default=POC_ROOT / ".work/body/ch10032_body.report.json")
    # The anatomical check on the inverted coordinate mapping. Generous, because it is
    # there to catch a wrong axis or a missing offset -- which are metres out -- not to
    # assert the rig is perfectly symmetric.
    parser.add_argument("--floor-tolerance-cm", type=float, default=2.0)
    parser.add_argument("--symmetry-tolerance-cm", type=float, default=2.0)
    parser.add_argument("--bind-tolerance-cm", type=float, default=1.0e-3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    Section, write_sectioned = load_write_sectioned()
    for required in (args.character, args.scene, args.skeleton, args.model):
        if not required.is_file():
            fail(f"missing input: {required}")

    drivers = read_driver_names(args.model)
    driver_slot = {name: index for index, name in enumerate(drivers)}
    model_sha256 = hashlib.sha256(args.model.read_bytes()).hexdigest()

    scene = json.loads(args.scene.read_text(encoding="utf-8"))
    skinning = scene.get("skinning") or {}
    character_bones = skinning.get("bone_names")
    if not isinstance(character_bones, list) or not character_bones:
        fail(f"{args.scene.name} has no skinning.bone_names")
    if skinning.get("maximum_runtime_influences") != INFLUENCES:
        fail(
            f"{args.scene.name} declares {skinning.get('maximum_runtime_influences')} runtime influences; "
            f"this bake is written for {INFLUENCES}"
        )
    mapping = scene.get("coordinate_mapping") or {}
    system = scene.get("coordinate_system")
    if system != "right_handed_y_up_meters_waist_origin":
        fail(f"{args.scene.name} declares coordinate_system {system!r}, which this bake cannot invert")
    offsets = parse_coordinate_mapping(mapping)

    skeleton = json.loads(args.skeleton.read_text(encoding="utf-8"))["bones"]
    bind, bind_residual_cm = compose_bind_pose(skeleton)
    if bind_residual_cm > args.bind_tolerance_cm:
        fail(
            f"the composed bind pose disagrees with the recorded component translations by "
            f"{bind_residual_cm:.6g} cm, over the {args.bind_tolerance_cm} cm tolerance. The rotator "
            "convention or the hierarchy order is wrong; a wrong one still yields a well-formed skeleton."
        )
    parent_of = {bone["name"]: bone["parent"] for bone in skeleton}
    for name in character_bones:
        if name not in bind:
            fail(f"character bone {name} is absent from {args.skeleton.name}")

    # Character bone -> driver slot. A bone that is not a driver folds onto its parent,
    # which must itself be a driver: folding two levels would be a different and much
    # larger approximation, so it is refused rather than done quietly.
    slot_of_character_bone: list[int] = []
    folded: list[dict] = []
    for name in character_bones:
        if name in driver_slot:
            slot_of_character_bone.append(driver_slot[name])
            continue
        parent = parent_of.get(name)
        if parent is None or parent not in driver_slot:
            fail(
                f"character bone {name} is not one of the model's {DRIVER_COUNT} drivers and its parent "
                f"({parent!r}) is not either, so its motion cannot be recovered from a .mldrv clip"
            )
        slot_of_character_bone.append(driver_slot[parent])
        folded.append({"bone": name, "folded_onto": parent})

    blob, sections = read_sectioned(args.character, b"VCHAR001")
    rest_pos, vertices = section_values(blob, sections, "render_pos", "f", 12)
    rest_nrm, normal_count = section_values(blob, sections, "render_nrm", "f", 12)
    bone_idx_raw, idx_count = section_values(blob, sections, "bone_idx", "I", INFLUENCES * 4)
    weight_raw, weight_count = section_values(blob, sections, "bone_weight", "f", INFLUENCES * 4)
    triangles_raw, triangle_count = section_values(blob, sections, "render_tri", "I", 12)
    if not (vertices == normal_count == idx_count == weight_count):
        fail("the character asset's per-vertex sections disagree about the vertex count")
    for index in triangles_raw:
        if index >= vertices:
            fail("a character triangle indexes past the end of the vertex array")

    # Vertices into component centimetres, then the anatomical check on the inversion.
    rest_component: list[tuple[float, float, float]] = []
    for vertex in range(vertices):
        gx, gy, gz = rest_pos[3 * vertex : 3 * vertex + 3]
        wx, wy, wz = gx + offsets[0], gy + offsets[1], gz + offsets[2]
        rest_component.append((wx * 100.0, -wz * 100.0, wy * 100.0))
    normals_component: list[tuple[float, float, float]] = []
    for vertex in range(vertices):
        nx, ny, nz = rest_nrm[3 * vertex : 3 * vertex + 3]
        # A direction, so only the axis permutation applies -- no offset and no scale.
        direction = (nx, -nz, ny)
        norm = math.sqrt(sum(component * component for component in direction))
        if norm < 1.0e-6:
            fail(f"character vertex {vertex} has a degenerate rest normal")
        normals_component.append(tuple(component / norm for component in direction))

    floor = min(point[2] for point in rest_component)
    ceiling = max(point[2] for point in rest_component)
    span_x = (min(point[0] for point in rest_component), max(point[0] for point in rest_component))
    if abs(floor) > args.floor_tolerance_cm:
        fail(
            f"the inverted rest body's lowest vertex is at z = {floor:.3f} cm, not on the floor. The "
            "coordinate mapping did not invert; a wrong axis or a dropped offset lands metres away."
        )
    if abs(span_x[0] + span_x[1]) > args.symmetry_tolerance_cm:
        fail(
            f"the inverted rest body spans x = {span_x[0]:.2f}..{span_x[1]:.2f} cm, which is not symmetric "
            "about the component origin; the axis permutation is wrong"
        )

    # Remap influences onto driver slots, merging the duplicates the fold creates and
    # renormalising. Merging matters twice over: a vertex weighted to both Chest_M and
    # Chest1_L ends up with two influences on the same slot, and leaving them separate
    # would both make the influence count a lie and pay for the same bone twice in the
    # per-frame skinning pass.
    merged_per_vertex: list[list[tuple[int, float]]] = []
    dropped_weight = 0.0
    for vertex in range(vertices):
        merged: dict[int, float] = {}
        for slot in range(INFLUENCES):
            weight = weight_raw[vertex * INFLUENCES + slot]
            if weight <= 0.0:
                continue
            character_bone = bone_idx_raw[vertex * INFLUENCES + slot]
            if character_bone >= len(character_bones):
                fail(f"character vertex {vertex} names bone {character_bone}, past the {len(character_bones)}-bone list")
            driver = slot_of_character_bone[character_bone]
            merged[driver] = merged.get(driver, 0.0) + weight
        if not merged:
            fail(f"character vertex {vertex} has no positive skin weight")
        total = sum(merged.values())
        dropped_weight = max(dropped_weight, abs(total - 1.0))
        if len(merged) > INFLUENCES:
            # Cannot happen: merging only ever reduces the count. Asserted anyway,
            # because the alternative is silently truncating a vertex's skinning.
            fail(f"character vertex {vertex} has {len(merged)} influences after merging")
        merged_per_vertex.append(
            [(driver, weight / total) for driver, weight in sorted(merged.items(), key=lambda item: (-item[1], item[0]))]
        )

    # The stored width is the measured maximum, not the source's declared 12. Folding four
    # bones onto parents that were often already influencing the same vertex takes this
    # garment's body from 12 to 5, and the width is what the per-frame skinning pass loops
    # over, so padding to 12 would spend 2.4x the work on zero weights. It goes in `info`
    # rather than being assumed, so an asset that needs more is read correctly.
    influence_width = max(len(entry) for entry in merged_per_vertex)
    bone_idx: list[int] = []
    bone_weight: list[float] = []
    for entry in merged_per_vertex:
        for slot in range(influence_width):
            if slot < len(entry):
                bone_idx.append(entry[slot][0])
                bone_weight.append(entry[slot][1])
            else:
                bone_idx.append(0)
                bone_weight.append(0.0)

    # Inverse bind per driver slot. Rotation is a unit quaternion and scale is unit, so
    # the inverse is the conjugate rotation and the negated, rotated translation.
    inv_bind: list[float] = []
    for slot, name in enumerate(drivers):
        if name not in bind:
            fail(f"driver {name} is absent from {args.skeleton.name}, so its inverse bind cannot be formed")
        translation, rotation = bind[name]
        inverse = quat_conjugate(rotation)
        shifted = quat_rotate(inverse, (-translation[0], -translation[1], -translation[2]))
        basis = [quat_rotate(inverse, axis) for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))]
        for row in range(3):
            inv_bind.extend((basis[0][row], basis[1][row], basis[2][row], shifted[row]))

    def u32(values) -> bytes:
        values = list(values)
        return struct.pack(f"<{len(values)}I", *values)

    def f32(values) -> bytes:
        values = list(values)
        return struct.pack(f"<{len(values)}f", *values)

    info = [vertices, triangle_count, DRIVER_COUNT, influence_width, len(folded), 0]
    flat_positions = [component for point in rest_component for component in point]
    flat_normals = [component for point in normals_component for component in point]
    sections_out = [
        Section("info", 6, 4, u32(info)),
        Section("rest_pos", vertices, 12, f32(flat_positions)),
        Section("rest_nrm", vertices, 12, f32(flat_normals)),
        Section("bone_idx", vertices, influence_width * 4, u32(bone_idx)),
        Section("bone_weight", vertices, influence_width * 4, f32(bone_weight)),
        Section("tri", triangle_count, 12, u32(triangles_raw)),
        # 48 bytes is twelve floats here for a different reason than above: a row-major
        # 3x4 matrix, not twelve influences.
        Section("inv_bind", DRIVER_COUNT, 48, f32(inv_bind)),
    ]
    written = write_sectioned(args.output.resolve(), MAGIC, VERSION, sections_out, source_sha256=model_sha256)

    report = {
        "output": str(args.output.resolve()),
        "bytes": written,
        "model": str(args.model.resolve()),
        "model_sha256": model_sha256,
        "character": str(args.character.resolve()),
        "character_sha256": hashlib.sha256(args.character.read_bytes()).hexdigest(),
        "skeleton": str(args.skeleton.resolve()),
        "vertices": vertices,
        "triangles": triangle_count,
        "influences_stored": influence_width,
        "influences_source_declared": INFLUENCES,
        "coordinate_offsets_m": list(offsets),
        "rest_floor_cm": round(floor, 6),
        "rest_height_cm": round(ceiling - floor, 4),
        "rest_x_span_cm": [round(span_x[0], 4), round(span_x[1], 4)],
        # The self-check that caught a wrong rotator convention. Kept in the report
        # because a silent regression here produces a body that looks like a body.
        "bind_pose_residual_cm": round(bind_residual_cm, 9),
        "weight_sum_worst_deviation": round(dropped_weight, 9),
        "folded_bones": folded,
        "folded_note": (
            "Bones the model does not drive, folded onto their driven parent. Measured against the "
            "sibling PoC's own per-frame skin matrices over five clips this moves the affected "
            "vertices by up to 2.2 cm (Chest1_L/R, 0.34% of skin weight each) and 5.2 cm "
            "(Toes_L/R, 3.4% each). Adding these four to the training export's driver list removes "
            "the approximation."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Baked body: {args.output.resolve()}")
    return 0


def parse_coordinate_mapping(mapping: dict) -> tuple[float, float, float]:
    """The three constants in `scene.json`'s coordinate_mapping, read rather than assumed.

    The strings look like `body_x - 0.0070480118`. Parsing them instead of hardcoding the
    numbers means a re-export with a different waist origin is either picked up or refused,
    not silently applied with last week's offset.
    """
    expected = {"x": "body_x", "y": "body_z", "z": "-body_y"}
    out = []
    for axis in ("x", "y", "z"):
        text = str(mapping.get(axis, ""))
        match = re.fullmatch(rf"\s*{re.escape(expected[axis])}\s*([+-])\s*([0-9.eE+-]+)\s*", text)
        if not match:
            fail(
                f"coordinate_mapping.{axis} is {text!r}, which is not the "
                f"'{expected[axis]} +/- constant' form this bake inverts"
            )
        value = float(match.group(2))
        out.append(value if match.group(1) == "+" else -value)
    # The mapping subtracts the constant, so inverting adds it back.
    return (-out[0], -out[1], -out[2])


if __name__ == "__main__":
    raise SystemExit(main())
