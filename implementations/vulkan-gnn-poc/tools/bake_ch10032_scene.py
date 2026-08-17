"""Blender offline baker for the CH10032 Vulkan runtime assets.

The body and animation FBX files are input-only. The generated VCHAR/VANIM and
VCLTH v2 files contain only tightly packed arrays required by the Vulkan sample.
Run with Blender's Python runtime so no FBX dependency enters the executable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body", required=True, type=Path)
    parser.add_argument("--animation", type=Path)
    parser.add_argument("--cloth", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--motion", default="ch10032_sprint")
    parser.add_argument("--duration", type=float, default=0.0, help="0 derives the exact duration from the exported FBX")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--static-pose", action="store_true", help="collapse the imported action to its first sampled pose")
    parser.add_argument("--max-proxy-vertices", type=int, default=4096)
    parser.add_argument("--proxy-top-y", type=float, default=0.15)
    parser.add_argument("--density", type=float, default=0.20022)
    return parser.parse_args(argv)


ARGS = parse_args()
POC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC_ROOT))
from real_scene.formats import (  # noqa: E402
    Section,
    load_vcloth_v1,
    pack_f32,
    pack_u32,
    sha256_file,
    write_sectioned,
)


COORD_CENTER_X = 0.0070480118
COORD_WAIST_Z = 1.0586285400
COORD_CENTER_Z = -0.0225656815
MAX_INFLUENCES = 12


CORE_BONE_PATTERN = re.compile(
    r"^(?:Root|Root_M|Spine[12]_M|Chest_M|Chest1_(?:L|R)|Neck(?:1)?_M|Head_M|"
    r"Scapula_(?:L|R)|Shoulder(?:Part[0-2])?_(?:L|R)|Elbow(?:Part[0-2])?_(?:L|R)|"
    r"Wrist_(?:L|R)|Hip(?:Part[0-2])?_(?:L|R)|Knee(?:Part[0-2])?_(?:L|R)|"
    r"Ankle_(?:L|R)|Toes_(?:L|R))$"
)


def clear_scene() -> None:
    bpy.ops.object.mode_set(mode="OBJECT") if bpy.context.object and bpy.context.object.mode != "OBJECT" else None
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.armatures, bpy.data.actions):
        for item in list(block):
            if item.users == 0:
                block.remove(item)


def import_fbx(path: Path) -> list[bpy.types.Object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(path), use_anim=True, automatic_bone_orientation=False)
    imported = [obj for obj in bpy.data.objects if obj not in before]
    if not imported:
        raise RuntimeError(f"FBX import created no objects: {path}")
    return imported


def runtime_transform() -> Matrix:
    return Matrix(
        (
            (1.0, 0.0, 0.0, -COORD_CENTER_X),
            (0.0, 0.0, 1.0, -COORD_WAIST_Z),
            (0.0, -1.0, 0.0, -COORD_CENTER_Z),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


def find_body(imported: Sequence[bpy.types.Object]) -> tuple[bpy.types.Object, bpy.types.Object]:
    armatures = [obj for obj in imported if obj.type == "ARMATURE"]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not armatures or not meshes:
        raise RuntimeError("body FBX must contain an armature and a mesh")
    body = max(meshes, key=lambda obj: len(obj.data.vertices))
    modifiers = [modifier for modifier in body.modifiers if modifier.type == "ARMATURE" and modifier.object]
    armature = modifiers[0].object if modifiers else max(armatures, key=lambda obj: len(obj.data.bones))
    if armature.type != "ARMATURE":
        raise RuntimeError("could not associate body mesh with an armature")
    return body, armature


def vertex_influences(
    body: bpy.types.Object,
    armature: bpy.types.Object,
) -> tuple[list[str], list[list[int]], list[list[float]], dict]:
    bone_name_set = {bone.name for bone in armature.data.bones}
    source_referenced_names = {
        body.vertex_groups[group.group].name
        for vertex in body.data.vertices
        for group in vertex.groups
        if body.vertex_groups[group.group].name in bone_name_set and group.weight > 0.0
    }
    if not source_referenced_names:
        raise RuntimeError("body mesh contains no weights that match armature bones")

    # Keep only the deformation chain needed to render the body silhouette and
    # drive the waist/lower-body collider. Facial, hair, finger, socket and
    # secondary bones are rigidly folded into their nearest retained ancestor.
    # Bone order follows the source skeleton so animation mapping is stable.
    retained_set = {bone.name for bone in armature.data.bones if CORE_BONE_PATTERN.fullmatch(bone.name)}
    retained_set.update(name for name in ("Root", "Root_M") if name in bone_name_set)
    referenced_names = [bone.name for bone in armature.data.bones if bone.name in retained_set]
    if "Root_M" not in referenced_names:
        raise RuntimeError("CH10032 core skeleton has no Root_M")
    bone_index = {name: index for index, name in enumerate(referenced_names)}

    remap: dict[str, int] = {}
    for source_name in source_referenced_names:
        bone = armature.data.bones[source_name]
        while bone is not None and bone.name not in bone_index:
            bone = bone.parent
        fallback = "Root_M" if "Root_M" in bone_index else "Root"
        remap[source_name] = bone_index[bone.name if bone is not None else fallback]

    indices: list[list[int]] = []
    weights: list[list[float]] = []
    maximum_source = 0
    minimum_retained = 1.0
    collapsed_weight = 0.0
    total_weight = 0.0
    for vertex in body.data.vertices:
        merged: defaultdict[int, float] = defaultdict(float)
        source_count = 0
        for group in vertex.groups:
            name = body.vertex_groups[group.group].name
            if name in remap and group.weight > 0.0:
                weight = float(group.weight)
                merged[remap[name]] += weight
                source_count += 1
                total_weight += weight
                if name not in retained_set:
                    collapsed_weight += weight
        raw = sorted(merged.items(), key=lambda item: (-item[1], item[0]))
        maximum_source = max(maximum_source, source_count)
        total = sum(weight for _, weight in raw)
        retained = sum(weight for _, weight in raw[:MAX_INFLUENCES])
        if total <= 1.0e-12:
            raise RuntimeError(f"unweighted render vertex {vertex.index}")
        minimum_retained = min(minimum_retained, retained / total)
        selected = raw[:MAX_INFLUENCES]
        selected_total = sum(weight for _, weight in selected)
        idx = [item[0] for item in selected]
        wgt = [item[1] / selected_total for item in selected]
        idx.extend([0] * (MAX_INFLUENCES - len(idx)))
        wgt.extend([0.0] * (MAX_INFLUENCES - len(wgt)))
        indices.append(idx)
        weights.append(wgt)
    return referenced_names, indices, weights, {
        "source_referenced_bones": len(source_referenced_names),
        "runtime_core_bones": len(referenced_names),
        "collapsed_source_bones": len(source_referenced_names - retained_set),
        "collapsed_weight_fraction": collapsed_weight / total_weight if total_weight else 0.0,
        "maximum_source_influences": maximum_source,
        "maximum_runtime_influences": MAX_INFLUENCES,
        "minimum_retained_weight_fraction": minimum_retained,
    }


def triangulated_render_mesh(
    body: bpy.types.Object,
    coordinate: Matrix,
    source_indices: Sequence[Sequence[int]],
    source_weights: Sequence[Sequence[float]],
) -> dict:
    mesh = body.data
    mesh.calc_loop_triangles()
    world = body.matrix_world
    normal_world = world.to_3x3().inverted().transposed()
    normal_runtime = coordinate.to_3x3()
    uv_layer = mesh.uv_layers.active.data if mesh.uv_layers.active else None
    dedup: dict[tuple, int] = {}
    positions: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    bone_indices: list[Sequence[int]] = []
    bone_weights: list[Sequence[float]] = []
    triangles: list[tuple[int, int, int]] = []
    source_triangles: list[tuple[int, int, int]] = []
    materials: list[int] = []
    source_positions = [tuple((coordinate @ (world @ vertex.co).to_4d()).to_3d()) for vertex in mesh.vertices]
    source_normals = []
    for vertex in mesh.vertices:
        normal = normal_runtime @ (normal_world @ vertex.normal)
        normal.normalize()
        source_normals.append(tuple(normal))
    for triangle in mesh.loop_triangles:
        out_triangle = []
        source_triangle = []
        for loop_index in triangle.loops:
            loop = mesh.loops[loop_index]
            vertex_index = loop.vertex_index
            uv = tuple(uv_layer[loop_index].uv) if uv_layer else (0.0, 0.0)
            key = (vertex_index, round(uv[0], 7), round(uv[1], 7))
            if key not in dedup:
                dedup[key] = len(positions)
                positions.append(source_positions[vertex_index])
                normals.append(source_normals[vertex_index])
                uvs.append(uv)
                bone_indices.append(source_indices[vertex_index])
                bone_weights.append(source_weights[vertex_index])
            out_triangle.append(dedup[key])
            source_triangle.append(vertex_index)
        triangles.append(tuple(out_triangle))
        source_triangles.append(tuple(source_triangle))
        materials.append(int(triangle.material_index))
    return {
        "positions": positions,
        "normals": normals,
        "uvs": uvs,
        "triangles": triangles,
        "source_triangles": source_triangles,
        "materials": materials,
        "bone_indices": bone_indices,
        "bone_weights": bone_weights,
        "source_positions": source_positions,
        "source_normals": source_normals,
        "material_names": [material.name if material else f"material_{index}" for index, material in enumerate(mesh.materials)],
    }


def select_proxy_indices(positions: Sequence[Sequence[float]], top_y: float, maximum: int) -> list[int]:
    candidates = [index for index, position in enumerate(positions) if position[1] <= top_y]
    if not candidates:
        raise RuntimeError("lower-body proxy selection is empty")
    if len(candidates) <= maximum:
        return candidates

    def voxelize(size: float) -> list[int]:
        chosen: dict[tuple[int, int, int], int] = {}
        for index in candidates:
            position = positions[index]
            key = tuple(math.floor(component / size) for component in position)
            chosen.setdefault(key, index)
        return sorted(chosen.values())

    low, high = 1.0e-4, 0.25
    best = candidates
    for _ in range(32):
        middle = (low + high) * 0.5
        current = voxelize(middle)
        if len(current) > maximum:
            low = middle
        else:
            best = current
            high = middle
    if len(best) > maximum:
        step = len(best) / maximum
        best = [best[min(int(i * step), len(best) - 1)] for i in range(maximum)]
    return best


def barycentric(point: Vector, a: Vector, b: Vector, c: Vector) -> tuple[float, float, float]:
    v0, v1, v2 = b - a, c - a, point - a
    d00, d01, d11 = v0.dot(v0), v0.dot(v1), v1.dot(v1)
    d20, d21 = v2.dot(v0), v2.dot(v1)
    denominator = d00 * d11 - d01 * d01
    if abs(denominator) <= 1.0e-20:
        return (1.0, 0.0, 0.0)
    v = (d11 * d20 - d01 * d21) / denominator
    w = (d00 * d21 - d01 * d20) / denominator
    u = 1.0 - v - w
    values = [max(0.0, min(1.0, value)) for value in (u, v, w)]
    total = sum(values)
    return tuple(value / total for value in values) if total > 1.0e-12 else (1.0, 0.0, 0.0)


def transfer_cloth_weights(
    cloth_positions: Sequence[Sequence[float]],
    body_positions: Sequence[Sequence[float]],
    body_triangles: Sequence[Sequence[int]],
    body_indices: Sequence[Sequence[int]],
    body_weights: Sequence[Sequence[float]],
) -> tuple[list[list[int]], list[list[float]], list[float]]:
    bvh = BVHTree.FromPolygons([Vector(value) for value in body_positions], body_triangles, all_triangles=True)
    output_indices: list[list[int]] = []
    output_weights: list[list[float]] = []
    distances: list[float] = []
    for cloth_position in cloth_positions:
        location, _, face_index, distance = bvh.find_nearest(Vector(cloth_position))
        if location is None or face_index is None:
            raise RuntimeError("body BVH could not bind a cloth vertex")
        triangle = body_triangles[face_index]
        factors = barycentric(
            location,
            Vector(body_positions[triangle[0]]),
            Vector(body_positions[triangle[1]]),
            Vector(body_positions[triangle[2]]),
        )
        merged: defaultdict[int, float] = defaultdict(float)
        for source_vertex, factor in zip(triangle, factors):
            for index, weight in zip(body_indices[source_vertex], body_weights[source_vertex]):
                if weight > 0.0:
                    merged[index] += factor * weight
        selected = sorted(merged.items(), key=lambda item: (-item[1], item[0]))[:MAX_INFLUENCES]
        total = sum(weight for _, weight in selected)
        if total <= 1.0e-12:
            raise RuntimeError("cloth binding produced no bone weights")
        indices = [index for index, _ in selected]
        weights = [weight / total for _, weight in selected]
        indices.extend([0] * (MAX_INFLUENCES - len(indices)))
        weights.extend([0.0] * (MAX_INFLUENCES - len(weights)))
        output_indices.append(indices)
        output_weights.append(weights)
        distances.append(float(distance))
    return output_indices, output_weights, distances


def unpack_rows(data: memoryview, count: int, components: int, code: str) -> list[tuple]:
    values = struct.unpack(f"<{count * components}{code}", data)
    return [tuple(values[index * components : (index + 1) * components]) for index in range(count)]


def vertex_masses(positions: Sequence[Sequence[float]], triangles: Sequence[Sequence[int]], density: float) -> list[float]:
    masses = [0.0] * len(positions)
    for triangle in triangles:
        a, b, c = (Vector(positions[index]) for index in triangle)
        triangle_mass = density * (b - a).cross(c - a).length * 0.5
        for index in triangle:
            masses[index] += triangle_mass / 3.0
    if any(value <= 0.0 or not math.isfinite(value) for value in masses):
        raise RuntimeError("cloth contains a zero/invalid vertex mass")
    return masses


def attach_animation(
    body_armature: bpy.types.Object, animation_path: Path | None
) -> tuple[bpy.types.Object, tuple[float, float]]:
    if animation_path is None:
        body_armature.data.pose_position = "REST"
        return body_armature, (0.0, 0.0)
    before = set(bpy.data.objects)
    imported = import_fbx(animation_path)
    animation_armatures = [obj for obj in imported if obj.type == "ARMATURE" and obj not in before]
    if not animation_armatures:
        animation_armatures = [obj for obj in imported if obj.type == "ARMATURE"]
    with_action = [obj for obj in animation_armatures if obj.animation_data and obj.animation_data.action]
    if not with_action:
        raise RuntimeError(f"animation FBX contains no armature action: {animation_path}")
    source_armature = max(with_action, key=lambda obj: len(obj.animation_data.action.fcurves))
    action = source_armature.animation_data.action
    missing = sorted({bone.name for bone in body_armature.data.bones} - {bone.name for bone in source_armature.data.bones})
    # Extra facial/deformation bones are allowed to remain in bind pose, but all
    # bones explicitly animated by the exported target sequence must exist.
    animated_names = {
        curve.data_path.split('pose.bones["', 1)[1].split('"]', 1)[0]
        for curve in action.fcurves
        if 'pose.bones["' in curve.data_path
    }
    body_bones = {bone.name for bone in body_armature.data.bones}
    missing_animated = sorted(animated_names - body_bones)
    # Unreal's AnimSequence FBX exporter includes curves for a few auxiliary
    # end/toe bones which the separately exported render FBX omits because they
    # have no skin influence. Applying the action to the target armature is
    # still well-defined: Blender ignores those unresolved data paths.
    required_motion_bones = {"Root_M", "Spine1_M", "Hip_L", "Hip_R", "Knee_L", "Knee_R", "Ankle_L", "Ankle_R"}
    missing_required = sorted(required_motion_bones - body_bones)
    if missing_required:
        raise RuntimeError(f"target character is missing required motion bones: {missing_required}")
    missing_required_curves = sorted(required_motion_bones - animated_names)
    if missing_required_curves:
        raise RuntimeError(f"animation is missing required target-bone curves: {missing_required_curves}")
    if missing_animated:
        print(f"Ignoring {len(missing_animated)} auxiliary animation curves absent from the render FBX: {missing_animated[:12]}")
    # Sample the exported animation armature itself. Blender 4.4+ Actions carry
    # source-ID slots; assigning one to a separately imported armature can
    # silently evaluate as bind pose despite matching fcurve paths. Both FBXs
    # originate from the same CH10032 skeleton, so skin transforms map by name.
    source_armature.data.pose_position = "POSE"
    return source_armature, tuple(float(value) for value in action.frame_range)


def bake_animation(
    armature: bpy.types.Object,
    bone_names: Sequence[str],
    coordinate: Matrix,
    frame_range: tuple[float, float],
    duration: float,
    fps: int,
) -> tuple[list[tuple[float, ...]], list[tuple[float, float, float]], int, int]:
    scene = bpy.context.scene
    frame_count = (
        max(1, int(round(duration * fps)) + 1)
        if duration > 0.0
        else max(1, int(round(frame_range[1] - frame_range[0])) + 1)
    )
    if frame_range[0] == frame_range[1]:
        frame_count = 1
    compact_root = bone_names.index("Root_M") if "Root_M" in bone_names else 0
    coordinate_inverse = coordinate.inverted()
    armature_world = armature.matrix_world
    armature_world_inverse = armature_world.inverted()
    matrices: list[tuple[float, ...]] = []
    root_positions: list[tuple[float, float, float]] = []
    for output_frame in range(frame_count):
        alpha = 0.0 if frame_count == 1 else output_frame / (frame_count - 1)
        source_frame = frame_range[0] + alpha * (frame_range[1] - frame_range[0])
        integer = math.floor(source_frame)
        scene.frame_set(integer, subframe=source_frame - integer)
        for bone_name in bone_names:
            pose_bone = armature.pose.bones.get(bone_name)
            rest_bone = armature.data.bones.get(bone_name)
            if pose_bone is None or rest_bone is None:
                raise RuntimeError(f"referenced skin bone is absent: {bone_name}")
            skin_world = armature_world @ pose_bone.matrix @ rest_bone.matrix_local.inverted() @ armature_world_inverse
            skin_runtime = coordinate @ skin_world @ coordinate_inverse
            matrices.append(tuple(float(skin_runtime[row][column]) for row in range(3) for column in range(4)))
        root_bone = armature.pose.bones.get("Root_M") or armature.pose.bones[bone_names[compact_root]]
        root_world = armature_world @ root_bone.matrix
        root_positions.append(tuple((coordinate @ root_world.translation.to_4d()).to_3d()))
    return matrices, root_positions, frame_count, compact_root


def main() -> None:
    if ARGS.fps <= 0 or ARGS.duration < 0.0 or ARGS.max_proxy_vertices <= 0:
        raise ValueError("fps/max proxy vertices must be positive and duration must be non-negative")
    clear_scene()
    body_objects = import_fbx(ARGS.body.resolve())
    body, armature = find_body(body_objects)
    coordinate = runtime_transform()
    bone_names, source_bone_indices, source_bone_weights, influence_stats = vertex_influences(body, armature)
    render = triangulated_render_mesh(body, coordinate, source_bone_indices, source_bone_weights)
    proxy_source = select_proxy_indices(render["source_positions"], ARGS.proxy_top_y, ARGS.max_proxy_vertices)

    cloth_v1 = load_vcloth_v1(ARGS.cloth.resolve())
    positions_section = cloth_v1.require("positions", stride=12)
    triangles_section = cloth_v1.require("triangles", stride=12)
    pin_section = cloth_v1.require("pin_mask", count=positions_section.count, stride=4)
    cloth_positions = unpack_rows(positions_section.data, positions_section.count, 3, "f")
    cloth_triangles = unpack_rows(triangles_section.data, triangles_section.count, 3, "I")
    pin_mask = [row[0] for row in unpack_rows(pin_section.data, pin_section.count, 1, "I")]
    cloth_bone_indices, cloth_bone_weights, binding_distances = transfer_cloth_weights(
        cloth_positions,
        render["source_positions"],
        render["source_triangles"],
        source_bone_indices,
        source_bone_weights,
    )
    masses = vertex_masses(cloth_positions, cloth_triangles, ARGS.density)

    animation_armature, frame_range = attach_animation(armature, ARGS.animation.resolve() if ARGS.animation else None)
    skin_matrices, root_positions, frame_count, root_bone_index = bake_animation(
        animation_armature, bone_names, coordinate, frame_range, ARGS.duration, ARGS.fps
    )
    if ARGS.static_pose:
        skin_matrices = skin_matrices[: len(bone_names)]
        root_positions = root_positions[:1]
        frame_count = 1
    elif frame_count > 1:
        bind_frame = skin_matrices[: len(bone_names)]
        changed = any(
            any(abs(a - b) > 1.0e-8 for a, b in zip(matrix, bind_frame[index % len(bind_frame)]))
            for index, matrix in enumerate(skin_matrices[len(bind_frame) :], start=len(bind_frame))
        )
        if not changed:
            raise RuntimeError("animation bake is static across all sampled frames")

    source_hasher = hashlib.sha256()
    for source in (ARGS.body, ARGS.cloth, ARGS.animation):
        if source:
            source_hasher.update(Path(source).read_bytes())
    source_hash = source_hasher.digest()
    output = ARGS.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    character_sections = [
        Section("info", 6, 4, pack_u32([len(render["positions"]), len(render["triangles"]), len(bone_names), len(proxy_source), len(render["material_names"]), MAX_INFLUENCES])),
        Section("render_pos", len(render["positions"]), 12, pack_f32(render["positions"])),
        Section("render_nrm", len(render["normals"]), 12, pack_f32(render["normals"])),
        Section("render_uv", len(render["uvs"]), 8, pack_f32(render["uvs"])),
        Section("render_tri", len(render["triangles"]), 12, pack_u32(render["triangles"])),
        Section("tri_material", len(render["materials"]), 4, pack_u32(render["materials"])),
        Section("bone_idx", len(render["bone_indices"]), 48, pack_u32(render["bone_indices"])),
        Section("bone_weight", len(render["bone_weights"]), 48, pack_f32(render["bone_weights"])),
        Section("proxy_pos", len(proxy_source), 12, pack_f32(render["source_positions"][index] for index in proxy_source)),
        Section("proxy_nrm", len(proxy_source), 12, pack_f32(render["source_normals"][index] for index in proxy_source)),
        Section("proxy_bone_idx", len(proxy_source), 48, pack_u32(source_bone_indices[index] for index in proxy_source)),
        Section("proxy_weight", len(proxy_source), 48, pack_f32(source_bone_weights[index] for index in proxy_source)),
    ]
    character_meta = write_sectioned(output / "ch10032.vchar", b"VCHAR001", 1, character_sections, source_sha256=source_hash)

    animation_sections = [
        Section("info", 4, 4, pack_u32([frame_count, len(bone_names), ARGS.fps, root_bone_index])),
        Section("skin_matrices", frame_count * len(bone_names), 48, pack_f32(skin_matrices)),
        Section("root_pos", frame_count, 12, pack_f32(root_positions)),
    ]
    animation_meta = write_sectioned(output / f"{ARGS.motion}.vanim", b"VANIM001", 1, animation_sections, source_sha256=source_hash)

    cloth_sections = [
        Section(name, section.count, section.stride, bytes(section.data)) for name, section in cloth_v1.sections.items()
    ]
    cloth_sections.extend(
        [
            Section("mass", len(masses), 4, pack_f32(masses)),
            Section("bone_idx", len(cloth_bone_indices), 48, pack_u32(cloth_bone_indices)),
            Section("bone_weight", len(cloth_bone_weights), 48, pack_f32(cloth_bone_weights)),
            Section("coord_params", 4, 4, pack_f32([COORD_CENTER_X, COORD_WAIST_Z, COORD_CENTER_Z, ARGS.density])),
        ]
    )
    cloth_meta = write_sectioned(output / "ch10032_lower.vcloth2", b"VCLTH002", 2, cloth_sections, source_sha256=source_hash)

    pinned_distances = [distance for distance, pinned in zip(binding_distances, pin_mask) if pinned]
    pinned_sorted = sorted(pinned_distances)
    metadata = {
        "schema_version": 1,
        "character": "CH10032",
        "motion": ARGS.motion,
        "coordinate_system": "right_handed_y_up_meters_waist_origin",
        "coordinate_mapping": {
            "x": f"body_x - {COORD_CENTER_X}",
            "y": f"body_z - {COORD_WAIST_Z}",
            "z": f"-body_y + {-COORD_CENTER_Z}",
        },
        "body_source": {"path": str(ARGS.body.resolve()), "sha256": sha256_file(ARGS.body)},
        "cloth_source": {"path": str(ARGS.cloth.resolve()), "sha256": sha256_file(ARGS.cloth)},
        "animation_source": None if not ARGS.animation else {"path": str(ARGS.animation.resolve()), "sha256": sha256_file(ARGS.animation)},
        "animation": {
            "fps": ARGS.fps,
            "frame_count": frame_count,
            "duration_s": (frame_count - 1) / ARGS.fps,
            "source_frame_range": list(frame_range),
            "root_bone": bone_names[root_bone_index],
        },
        "mesh": {
            "render_vertices": len(render["positions"]),
            "render_triangles": len(render["triangles"]),
            "proxy_vertices": len(proxy_source),
            "cloth_vertices": len(cloth_positions),
            "cloth_triangles": len(cloth_triangles),
            "pinned_vertices": len(pinned_distances),
        },
        "skinning": {**influence_stats, "referenced_bones": len(bone_names), "bone_names": bone_names},
        "materials": render["material_names"],
        "cloth_binding_distance_m": {
            "minimum": min(binding_distances),
            "median": sorted(binding_distances)[len(binding_distances) // 2],
            "maximum": max(binding_distances),
            "pinned_median": pinned_sorted[len(pinned_sorted) // 2],
            "pinned_maximum": max(pinned_distances),
        },
        "files": {
            "ch10032.vchar": character_meta,
            f"{ARGS.motion}.vanim": animation_meta,
            "ch10032_lower.vcloth2": cloth_meta,
        },
    }
    (output / "scene.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output), "frames": frame_count, "bones": len(bone_names), "proxy": len(proxy_source), "pin_max_m": max(pinned_distances)}, indent=2))


if __name__ == "__main__":
    main()
