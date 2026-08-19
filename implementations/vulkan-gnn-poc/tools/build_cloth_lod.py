"""Build an isolated CH10032 VCLTH v2 LOD with Blender's quadric decimator.

Run this file through Blender, not the training virtual environment. The source
runtime cloth is read-only and the output name is required to differ, so an LOD
bake cannot overwrite the topology used by student distillation or its goldens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree


POC_ROOT = Path(__file__).resolve().parents[1]
if str(POC_ROOT) not in sys.path:
    sys.path.insert(0, str(POC_ROOT))

from real_scene.formats import Section, load_sectioned, sha256_file, write_sectioned  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a protected-boundary CH10032 cloth LOD.")
    parser.add_argument(
        "--input",
        type=Path,
        default=POC_ROOT / ".work/real_scene/ch10032_tpose/ch10032_lower.vcloth2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=POC_ROOT / "Assets/Meshes/CH10032_lower_sim_lod1.vcloth2",
    )
    parser.add_argument("--target-triangles", type=int, default=1280)
    parser.add_argument("--aggressiveness", type=float, default=1.0)
    raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(raw)


def unpack_rows(data: memoryview, count: int, components: int, code: str) -> list[tuple]:
    values = struct.unpack(f"<{count * components}{code}", data)
    return [tuple(values[index * components : (index + 1) * components]) for index in range(count)]


def flatten(values: Iterable[Iterable]) -> list:
    return [item for group in values for item in group]


def pack_f32(values: Iterable) -> bytes:
    flat = list(values)
    if flat and isinstance(flat[0], (tuple, list)):
        flat = flatten(flat)
    return struct.pack(f"<{len(flat)}f", *flat)


def pack_u32(values: Iterable) -> bytes:
    flat = list(values)
    if flat and isinstance(flat[0], (tuple, list)):
        flat = flatten(flat)
    return struct.pack(f"<{len(flat)}I", *flat)


def edge_counts_for(triangles: Sequence[Sequence[int]]) -> Counter:
    result: Counter = Counter()
    for triangle in triangles:
        for index in range(3):
            result[tuple(sorted((triangle[index], triangle[(index + 1) % 3])))] += 1
    return result


def boundary_components(edge_counts: Counter) -> list[list[int]]:
    adjacency: dict[int, set[int]] = defaultdict(set)
    for (first, second), count in edge_counts.items():
        if count == 1:
            adjacency[first].add(second)
            adjacency[second].add(first)
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise ValueError("cloth boundary is not a collection of closed loops")
    components: list[list[int]] = []
    visited: set[int] = set()
    for start in sorted(adjacency):
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        component: list[int] = []
        while stack:
            vertex = stack.pop()
            component.append(vertex)
            for neighbor in adjacency[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def color_constraints(constraints: Sequence[tuple], vertices_of: Callable) -> tuple[list[tuple], list[int]]:
    vertex_colors: dict[int, set[int]] = defaultdict(set)
    buckets: list[list[tuple]] = []
    for constraint in constraints:
        forbidden: set[int] = set()
        vertices = tuple(vertices_of(constraint))
        for vertex in vertices:
            forbidden.update(vertex_colors[vertex])
        color = 0
        while color in forbidden:
            color += 1
        while len(buckets) <= color:
            buckets.append([])
        buckets[color].append(constraint)
        for vertex in vertices:
            vertex_colors[vertex].add(color)
    result: list[tuple] = []
    offsets = [0]
    for bucket in buckets:
        result.extend(bucket)
        offsets.append(len(result))
    return result, offsets


def barycentric(point: Vector, a: Vector, b: Vector, c: Vector) -> tuple[float, float, float]:
    v0, v1, v2 = b - a, c - a, point - a
    d00, d01, d11 = v0.dot(v0), v0.dot(v1), v1.dot(v1)
    d20, d21 = v2.dot(v0), v2.dot(v1)
    denominator = d00 * d11 - d01 * d01
    if abs(denominator) <= 1.0e-20:
        return (1.0, 0.0, 0.0)
    values = [
        1.0 - (d11 * d20 - d01 * d21) / denominator - (d00 * d21 - d01 * d20) / denominator,
        (d11 * d20 - d01 * d21) / denominator,
        (d00 * d21 - d01 * d20) / denominator,
    ]
    values = [max(0.0, min(1.0, value)) for value in values]
    total = sum(values)
    return tuple(value / total for value in values) if total > 1.0e-12 else (1.0, 0.0, 0.0)


def interpolate_attributes(
    output_positions: Sequence[Sequence[float]],
    source_positions: Sequence[Sequence[float]],
    source_triangles: Sequence[Sequence[int]],
    source_uvs: Sequence[Sequence[float]],
    source_bones: Sequence[Sequence[int]],
    source_weights: Sequence[Sequence[float]],
    source_vertices: Sequence[int],
) -> tuple[list[tuple], list[tuple], list[tuple], list[int], list[float]]:
    vectors = [Vector(position) for position in source_positions]
    tree = BVHTree.FromPolygons(vectors, source_triangles, all_triangles=True)
    output_uvs: list[tuple] = []
    output_bones: list[tuple] = []
    output_weights: list[tuple] = []
    output_source: list[int] = []
    distances: list[float] = []
    for position in output_positions:
        location, _normal, face_index, distance = tree.find_nearest(Vector(position))
        if location is None or face_index is None:
            raise RuntimeError("could not project a decimated cloth vertex to the source mesh")
        triangle = source_triangles[face_index]
        factors = barycentric(location, *(vectors[index] for index in triangle))
        output_uvs.append(
            tuple(sum(factor * source_uvs[index][axis] for factor, index in zip(factors, triangle)) for axis in range(2))
        )
        influences: dict[int, float] = defaultdict(float)
        for factor, vertex in zip(factors, triangle):
            for bone, weight in zip(source_bones[vertex], source_weights[vertex]):
                if weight > 1.0e-8:
                    influences[int(bone)] += factor * float(weight)
        ordered = sorted(influences.items(), key=lambda item: (-item[1], item[0]))[:12]
        total = sum(weight for _, weight in ordered)
        if total <= 1.0e-12:
            raise RuntimeError("decimated cloth vertex has no transferred LBS influence")
        bones = [bone for bone, _ in ordered]
        weights = [weight / total for _, weight in ordered]
        output_bones.append(tuple(bones + [0] * (12 - len(bones))))
        output_weights.append(tuple(weights + [0.0] * (12 - len(weights))))
        strongest = triangle[max(range(3), key=lambda index: factors[index])]
        output_source.append(int(source_vertices[strongest]))
        distances.append(float(distance))
    return output_uvs, output_bones, output_weights, output_source, distances


def make_constraints(
    positions: Sequence[Sequence[float]], triangles: Sequence[Sequence[int]]
) -> dict[str, list]:
    edge_counts = edge_counts_for(triangles)
    if any(count > 2 for count in edge_counts.values()):
        raise ValueError("decimated cloth contains non-manifold edges")
    unique_edges = sorted(edge_counts)
    adjacency: dict[int, set[int]] = defaultdict(set)
    for first, second in unique_edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    offsets = [0]
    neighbors: list[int] = []
    for vertex in range(len(positions)):
        neighbors.extend(sorted(adjacency[vertex]))
        offsets.append(len(neighbors))
    stretch_edges, stretch_colors = color_constraints(unique_edges, lambda edge: edge)

    edge_faces: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for triangle in triangles:
        for index in range(3):
            a, b, opposite = triangle[index], triangle[(index + 1) % 3], triangle[(index + 2) % 3]
            edge_faces[tuple(sorted((a, b)))].append((a, b, opposite))
    bend_records: list[tuple[tuple[int, int, int, int], float]] = []
    vectors = [Vector(position) for position in positions]
    for edge in sorted(edge_faces):
        adjacent = edge_faces[edge]
        if len(adjacent) != 2:
            continue
        first, second = adjacent
        a, b, c = first
        d = second[2]
        edge_direction = (vectors[b] - vectors[a]).normalized()
        normal_first = (vectors[b] - vectors[a]).cross(vectors[c] - vectors[a]).normalized()
        normal_second = (vectors[second[1]] - vectors[second[0]]).cross(
            vectors[d] - vectors[second[0]]
        ).normalized()
        cosine = max(-1.0, min(1.0, normal_first.dot(normal_second)))
        sine = normal_first.cross(normal_second).dot(edge_direction)
        bend_records.append(((a, b, c, d), math.atan2(sine, cosine)))
    bend_records, bend_colors = color_constraints(bend_records, lambda record: record[0])
    return {
        "edge_counts": edge_counts,
        "csr_offsets": offsets,
        "csr_neighbors": neighbors,
        "stretch_edges": stretch_edges,
        "stretch_colors": stretch_colors,
        "bend_quads": [record[0] for record in bend_records],
        "bend_rest": [record[1] for record in bend_records],
        "bend_colors": bend_colors,
    }


def vertex_masses(
    positions: Sequence[Sequence[float]], triangles: Sequence[Sequence[int]], density: float
) -> tuple[list[float], float]:
    masses = [0.0] * len(positions)
    area = 0.0
    vectors = [Vector(position) for position in positions]
    for triangle in triangles:
        a, b, c = (vectors[index] for index in triangle)
        triangle_area = (b - a).cross(c - a).length * 0.5
        area += triangle_area
        for index in triangle:
            masses[index] += density * triangle_area / 3.0
    if any(value <= 0.0 or not math.isfinite(value) for value in masses):
        raise ValueError("decimated cloth contains zero or invalid vertex mass")
    return masses, area


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(int(math.ceil(fraction * len(ordered))) - 1, len(ordered) - 1)]


def write_obj(
    path: Path,
    positions: Sequence[Sequence[float]],
    uvs: Sequence[Sequence[float]],
    triangles: Sequence[Sequence[int]],
    pins: set[int],
) -> None:
    lines = [
        "# CH10032 protected-boundary simulation LOD",
        "# Right-handed Y-up meters; waist is near Y=0",
        f"# pinned waist vertices: {','.join(str(value) for value in sorted(pins))}",
        "o CH10032_lower_sim_lod1",
    ]
    lines.extend(f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in positions)
    lines.extend(f"vt {u:.9g} {v:.9g}" for u, v in uvs)
    lines.append("g lower_skirt_lod1")
    lines.extend(
        "f " + " ".join(f"{vertex + 1}/{vertex + 1}" for vertex in triangle)
        for triangle in triangles
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    args = parse_args()
    source_path = args.input.resolve()
    output_path = args.output.resolve()
    if source_path == output_path:
        raise ValueError("LOD output must not overwrite its source asset")
    source = load_sectioned(source_path, expected_magic=b"VCLTH002", expected_version=2)
    source_positions = unpack_rows(
        source.require("positions", stride=12).data, source.require("positions").count, 3, "f"
    )
    source_triangles = unpack_rows(
        source.require("triangles", stride=12).data, source.require("triangles").count, 3, "I"
    )
    source_uvs = unpack_rows(source.require("uv", stride=8).data, len(source_positions), 2, "f")
    source_pins = [value[0] for value in unpack_rows(source.require("pin_mask").data, len(source_positions), 1, "I")]
    source_vertices = [
        value[0] for value in unpack_rows(source.require("source_vertex").data, len(source_positions), 1, "I")
    ]
    source_bones = unpack_rows(source.require("bone_idx", stride=48).data, len(source_positions), 12, "I")
    source_weights = unpack_rows(source.require("bone_weight", stride=48).data, len(source_positions), 12, "f")
    coord_params = unpack_rows(source.require("coord_params", count=4, stride=4).data, 4, 1, "f")
    density = float(coord_params[3][0])
    if not 0 < args.target_triangles < len(source_triangles):
        raise ValueError("target triangle count must be positive and below the source count")

    source_edges = edge_counts_for(source_triangles)
    protected = {vertex for edge, count in source_edges.items() if count == 1 for vertex in edge}
    protected.update(index for index, pinned in enumerate(source_pins) if pinned)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    mesh = bpy.data.meshes.new("CH10032_lower_sim_source")
    mesh.from_pydata(source_positions, [], source_triangles)
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new("CH10032_lower_sim_lod1", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    decimate_vertices = [index for index in range(len(source_positions)) if index not in protected]
    group = obj.vertex_groups.new(name="DecimateInterior")
    group.add(decimate_vertices, 1.0, "REPLACE")
    modifier = obj.modifiers.new(name="ProtectedBoundaryQuadricLOD", type="DECIMATE")
    modifier.decimate_type = "COLLAPSE"
    modifier.ratio = args.target_triangles / len(source_triangles)
    modifier.use_collapse_triangulate = True
    modifier.vertex_group = group.name
    modifier.vertex_group_factor = args.aggressiveness
    modifier.invert_vertex_group = False
    bpy.ops.object.modifier_apply(modifier=modifier.name)

    output_positions = [tuple(float(value) for value in vertex.co) for vertex in mesh.vertices]
    output_triangles = [tuple(int(value) for value in polygon.vertices) for polygon in mesh.polygons]
    if any(len(triangle) != 3 or len(set(triangle)) != 3 for triangle in output_triangles):
        raise ValueError("Blender decimation did not produce a strict triangle mesh")
    constraints = make_constraints(output_positions, output_triangles)
    output_loops = boundary_components(constraints["edge_counts"])
    if not output_loops:
        raise ValueError("decimated cloth lost all boundary loops")
    waist_loop = max(
        output_loops,
        key=lambda loop: sum(output_positions[vertex][1] for vertex in loop) / len(loop),
    )
    pins = set(waist_loop)
    source_pin_count = sum(bool(value) for value in source_pins)
    if len(pins) != source_pin_count:
        raise ValueError(f"protected waist loop changed from {source_pin_count} to {len(pins)} vertices")

    output_uvs, output_bones, output_weights, output_source, projection_distances = interpolate_attributes(
        output_positions,
        source_positions,
        source_triangles,
        source_uvs,
        source_bones,
        source_weights,
        source_vertices,
    )
    masses, output_area = vertex_masses(output_positions, output_triangles, density)
    source_masses, source_area = vertex_masses(source_positions, source_triangles, density)
    pin_mask = [1 if vertex in pins else 0 for vertex in range(len(output_positions))]

    sections = [
        Section("positions", len(output_positions), 12, pack_f32(output_positions)),
        Section("uv", len(output_uvs), 8, pack_f32(output_uvs)),
        Section("triangles", len(output_triangles), 12, pack_u32(output_triangles)),
        Section("csr_offsets", len(constraints["csr_offsets"]), 4, pack_u32(constraints["csr_offsets"])),
        Section("csr_neighbors", len(constraints["csr_neighbors"]), 4, pack_u32(constraints["csr_neighbors"])),
        Section("stretch_edges", len(constraints["stretch_edges"]), 8, pack_u32(constraints["stretch_edges"])),
        Section("stretch_colors", len(constraints["stretch_colors"]), 4, pack_u32(constraints["stretch_colors"])),
        Section("bend_quads", len(constraints["bend_quads"]), 16, pack_u32(constraints["bend_quads"])),
        Section("bend_rest", len(constraints["bend_rest"]), 4, pack_f32(constraints["bend_rest"])),
        Section("bend_colors", len(constraints["bend_colors"]), 4, pack_u32(constraints["bend_colors"])),
        Section("pin_mask", len(pin_mask), 4, pack_u32(pin_mask)),
        Section("source_vertex", len(output_source), 4, pack_u32(output_source)),
        Section("mass", len(masses), 4, pack_f32(masses)),
        Section("bone_idx", len(output_bones), 48, pack_u32(output_bones)),
        Section("bone_weight", len(output_weights), 48, pack_f32(output_weights)),
        Section("coord_params", 4, 4, bytes(source.require("coord_params").data)),
    ]
    metadata = write_sectioned(
        output_path,
        b"VCLTH002",
        2,
        sections,
        source_sha256=hashlib.sha256(source_path.read_bytes()).digest(),
    )
    reloaded = load_sectioned(output_path, expected_magic=b"VCLTH002", expected_version=2)
    if reloaded.require("positions", stride=12).count != len(output_positions):
        raise ValueError("reloaded LOD vertex count differs")

    output_vectors = [Vector(position) for position in output_positions]
    output_tree = BVHTree.FromPolygons(output_vectors, output_triangles, all_triangles=True)
    source_tree = BVHTree.FromPolygons(
        [Vector(position) for position in source_positions], source_triangles, all_triangles=True
    )
    source_to_lod = []
    for position in source_positions:
        _location, _normal, _face, distance = output_tree.find_nearest(Vector(position))
        source_to_lod.append(float(distance))
    orientation_mismatches = 0
    triangle_areas: list[float] = []
    for triangle in output_triangles:
        a, b, c = (output_vectors[index] for index in triangle)
        cross = (b - a).cross(c - a)
        triangle_areas.append(cross.length * 0.5)
        center = (a + b + c) / 3.0
        _location, source_normal, _face, _distance = source_tree.find_nearest(center)
        if source_normal is None or cross.length <= 1.0e-20:
            raise ValueError("decimated cloth contains an invalid triangle")
        if cross.normalized().dot(source_normal) < 0.0:
            orientation_mismatches += 1
    source_pin_positions = [
        Vector(position) for position, pinned in zip(source_positions, source_pins) if pinned
    ]
    pin_drifts = [
        min((output_vectors[index] - source_pin).length for source_pin in source_pin_positions)
        for index in pins
    ]
    output_edges = constraints["edge_counts"]
    degrees = [0] * len(output_positions)
    for first, second in output_edges:
        degrees[first] += 1
        degrees[second] += 1
    weight_errors = [abs(sum(weights) - 1.0) for weights in output_weights]
    report = {
        "schema_version": 1,
        "asset": "CH10032_lower_sim_lod1",
        "purpose": "standalone simulation LOD; does not replace distillation topology",
        "source": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "vertices": len(source_positions),
            "triangles": len(source_triangles),
        },
        "output": {
            "path": str(output_path),
            "sha256": metadata["file_sha256"],
            "bytes": metadata["file_bytes"],
            "vertices": len(output_positions),
            "triangles": len(output_triangles),
            "vertex_reduction": 1.0 - len(output_positions) / len(source_positions),
            "triangle_reduction": 1.0 - len(output_triangles) / len(source_triangles),
        },
        "topology": {
            "undirected_edges": len(output_edges),
            "directed_csr_edges": len(constraints["csr_neighbors"]),
            "boundary_edges": sum(count == 1 for count in output_edges.values()),
            "boundary_loops": [len(loop) for loop in sorted(output_loops, key=len, reverse=True)],
            "non_manifold_edges": sum(count > 2 for count in output_edges.values()),
            "degree": {"min": min(degrees), "mean": sum(degrees) / len(degrees), "max": max(degrees)},
        },
        "geometry": {
            "source_area_m2": source_area,
            "output_area_m2": output_area,
            "area_ratio": output_area / source_area,
            "lod_to_source_projection_m": {
                "mean": sum(projection_distances) / len(projection_distances),
                "p95": percentile(projection_distances, 0.95),
                "max": max(projection_distances),
            },
            "source_to_lod_distance_m": {
                "mean": sum(source_to_lod) / len(source_to_lod),
                "p95": percentile(source_to_lod, 0.95),
                "max": max(source_to_lod),
            },
            "minimum_triangle_area_m2": min(triangle_areas),
            "orientation_mismatches": orientation_mismatches,
            "maximum_pin_position_drift_m": max(pin_drifts),
        },
        "constraints": {
            "pins": len(pins),
            "stretch": len(constraints["stretch_edges"]),
            "stretch_colors": len(constraints["stretch_colors"]) - 1,
            "bend": len(constraints["bend_quads"]),
            "bend_colors": len(constraints["bend_colors"]) - 1,
        },
        "binding": {
            "slots": 12,
            "referenced_bones": len(
                {bone for bones, weights in zip(output_bones, output_weights) for bone, weight in zip(bones, weights) if weight > 1.0e-8}
            ),
            "max_weight_sum_error": max(weight_errors),
        },
        "mass": {"density_kg_m2": density, "total_kg": sum(masses), "source_total_kg": sum(source_masses)},
        "decimator": {
            "implementation": f"Blender {bpy.app.version_string} Decimate/Collapse",
            "target_triangles": args.target_triangles,
            "protected_vertices": len(protected),
            "rule": "all source boundary vertices and waist pins are excluded from collapse",
        },
    }
    report_path = output_path.with_suffix(".json")
    obj_path = output_path.with_suffix(".obj")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_obj(obj_path, output_positions, output_uvs, output_triangles, pins)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
