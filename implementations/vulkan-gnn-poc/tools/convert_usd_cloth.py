#!/usr/bin/env python3
"""Convert selected USD garment patterns into a compact Vulkan cloth asset.

The script intentionally depends only on Python's standard library and Pixar
USD's ``pxr`` module. On Windows it can be run with Blender's bundled USD
runtime:

    blender --background --factory-startup \
      --python tools/convert_usd_cloth.py -- \
      --input garment.usd --output-dir meshs \
      --name CH10032_lower_sim --patterns 20-23,26-29

VCLOTH v1 stores a section directory followed by aligned little-endian arrays.
The JSON sidecar is the authoritative description of the section layout and
the source/conversion provenance.
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
from typing import Iterable, Sequence

from pxr import Gf, Usd, UsdGeom


MAGIC = b"VCLTH001"
VERSION = 1
HEADER = struct.Struct("<8sIIQ32s8s")
SECTION_ENTRY = struct.Struct("<16sQII")
ALIGNMENT = 16


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--mesh-prim", default="/Garment/SimMesh")
    parser.add_argument(
        "--patterns",
        default="20-23,26-29",
        help="Comma separated pattern indices/ranges, for example 20-23,26-29",
    )
    parser.add_argument(
        "--sewing-tolerance",
        type=float,
        default=1.0e-6,
        help="Maximum sewing-pair separation in meters before conversion fails",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def selected_pattern_names(specification: str) -> list[str]:
    indices: set[int] = set()
    for token in specification.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            first, last = (int(value) for value in token.split("-", 1))
            if first > last:
                raise ValueError(f"Invalid descending pattern range: {token}")
            indices.update(range(first, last + 1))
        else:
            indices.add(int(token))
    if not indices:
        raise ValueError("At least one pattern must be selected")
    return [f"pattern{index}" for index in sorted(indices)]


def cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def subtract(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def length(vector: Sequence[float]) -> float:
    return math.sqrt(dot(vector, vector))


def normalize(vector: Sequence[float]) -> tuple[float, float, float]:
    magnitude = length(vector)
    if magnitude <= 1.0e-12:
        raise ValueError("Cannot normalize a zero-length vector")
    return tuple(component / magnitude for component in vector)  # type: ignore[return-value]


class DisjointSet:
    def __init__(self, values: Iterable[int]):
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def boundary_loops(triangles: Sequence[Sequence[int]]) -> tuple[list[list[int]], Counter]:
    edge_counts: Counter = Counter()
    for triangle in triangles:
        for index in range(3):
            edge_counts[tuple(sorted((triangle[index], triangle[(index + 1) % 3])))] += 1

    adjacency: dict[int, set[int]] = defaultdict(set)
    for (first, second), count in edge_counts.items():
        if count == 1:
            adjacency[first].add(second)
            adjacency[second].add(first)

    loops: list[list[int]] = []
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
        loops.append(sorted(component))
    return loops, edge_counts


def color_constraints(
    constraints: Sequence[tuple], vertices_of,
) -> tuple[list[tuple], list[int]]:
    """Greedily group constraints so each color has disjoint vertices."""

    vertex_colors: dict[int, set[int]] = defaultdict(set)
    buckets: list[list[tuple]] = []
    for constraint in constraints:
        vertices = tuple(vertices_of(constraint))
        forbidden: set[int] = set()
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

    reordered: list[tuple] = []
    offsets = [0]
    for bucket in buckets:
        reordered.extend(bucket)
        offsets.append(len(reordered))
    return reordered, offsets


def pack_floats(values: Iterable[float]) -> bytes:
    flattened = list(values)
    return struct.pack(f"<{len(flattened)}f", *flattened)


def pack_uints(values: Iterable[int]) -> bytes:
    flattened = list(values)
    return struct.pack(f"<{len(flattened)}I", *flattened)


def flatten(values: Iterable[Iterable]) -> list:
    return [item for group in values for item in group]


def write_vcloth(path: Path, sections: list[dict]) -> dict:
    directory_end = HEADER.size + SECTION_ENTRY.size * len(sections)
    payload_offset = align_up(directory_end)
    output = bytearray(payload_offset)

    for section in sections:
        aligned_offset = align_up(len(output))
        output.extend(b"\0" * (aligned_offset - len(output)))
        section["offset"] = aligned_offset
        output.extend(section["data"])

    payload_hash = hashlib.sha256(output[payload_offset:]).digest()
    output[: HEADER.size] = HEADER.pack(
        MAGIC,
        VERSION,
        len(sections),
        len(output),
        payload_hash,
        b"\0" * 8,
    )
    cursor = HEADER.size
    for section in sections:
        tag = section["name"].encode("ascii")
        if len(tag) >= 16:
            raise ValueError(f"VCLOTH section name is too long: {section['name']}")
        output[cursor : cursor + SECTION_ENTRY.size] = SECTION_ENTRY.pack(
            tag + b"\0" * (16 - len(tag)),
            section["offset"],
            section["count"],
            section["stride"],
        )
        cursor += SECTION_ENTRY.size

    path.write_bytes(output)
    return {
        "magic": MAGIC.decode("ascii"),
        "version": VERSION,
        "endianness": "little",
        "alignment": ALIGNMENT,
        "header_bytes": HEADER.size,
        "section_entry_bytes": SECTION_ENTRY.size,
        "payload_offset": payload_offset,
        "payload_sha256": payload_hash.hex(),
        "file_sha256": hashlib.sha256(output).hexdigest(),
        "file_bytes": len(output),
        "sections": [
            {
                "name": section["name"],
                "offset": section["offset"],
                "count": section["count"],
                "stride": section["stride"],
                "encoding": section["encoding"],
            }
            for section in sections
        ],
    }


def validate_vcloth(path: Path) -> None:
    data = path.read_bytes()
    magic, version, section_count, file_size, payload_hash, reserved = HEADER.unpack_from(data)
    if magic != MAGIC or version != VERSION or file_size != len(data) or any(reserved):
        raise ValueError("Generated VCLOTH header failed validation")
    payload_offset = align_up(HEADER.size + section_count * SECTION_ENTRY.size)
    if hashlib.sha256(data[payload_offset:]).digest() != payload_hash:
        raise ValueError("Generated VCLOTH payload hash failed validation")
    for index in range(section_count):
        tag, offset, count, stride = SECTION_ENTRY.unpack_from(
            data, HEADER.size + index * SECTION_ENTRY.size
        )
        if not tag.rstrip(b"\0") or stride == 0 or offset + count * stride > len(data):
            raise ValueError("Generated VCLOTH section directory failed validation")


def write_obj(
    path: Path,
    positions: Sequence[Sequence[float]],
    uvs: Sequence[Sequence[float]],
    triangles: Sequence[Sequence[int]],
    pinned: set[int],
) -> None:
    lines = [
        "# Vulkan GNN cloth PoC processed simulation mesh",
        "# Coordinates: right-handed, Y-up, meters; waist is near Y=0",
        f"# Pinned waist-loop vertices (zero-based): {','.join(str(v) for v in sorted(pinned))}",
        "o CH10032_lower_sim",
    ]
    lines.extend(f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in positions)
    lines.extend(f"vt {u:.9g} {v:.9g}" for u, v in uvs)
    lines.append("g lower_skirt")
    lines.extend(
        "f " + " ".join(f"{vertex + 1}/{vertex + 1}" for vertex in triangle)
        for triangle in triangles
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    args = parse_args()
    source_path = args.input.resolve()
    output_directory = args.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    pattern_names = selected_pattern_names(args.patterns)
    selected_patterns = set(pattern_names)

    stage = Usd.Stage.Open(str(source_path))
    if stage is None:
        raise ValueError(f"Unable to open USD stage: {source_path}")
    mesh_prim = stage.GetPrimAtPath(args.mesh_prim)
    if not mesh_prim or not mesh_prim.IsA(UsdGeom.Mesh):
        raise ValueError(f"USD mesh prim was not found: {args.mesh_prim}")
    mesh = UsdGeom.Mesh(mesh_prim)

    raw_points = list(mesh.GetPointsAttr().Get() or [])
    face_counts = [int(value) for value in (mesh.GetFaceVertexCountsAttr().Get() or [])]
    face_indices = [int(value) for value in (mesh.GetFaceVertexIndicesAttr().Get() or [])]
    if not face_counts or set(face_counts) != {3}:
        raise ValueError("The converter currently requires a non-empty all-triangle SimMesh")

    transform = UsdGeom.XformCache().GetLocalToWorldTransform(mesh_prim)
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    source_positions: list[tuple[float, float, float]] = []
    for point in raw_points:
        transformed = transform.Transform(Gf.Vec3d(*(float(value) for value in point)))
        source_positions.append(
            tuple(float(transformed[axis]) * meters_per_unit for axis in range(3))
        )

    faces: list[tuple[int, int, int]] = []
    cursor = 0
    for count in face_counts:
        face = tuple(face_indices[cursor : cursor + count])
        cursor += count
        faces.append(face)  # type: ignore[arg-type]
    if cursor != len(face_indices):
        raise ValueError("USD face-index payload has an unexpected length")

    pattern_faces: dict[str, list[int]] = {}
    internal_sewing: list[dict] = []
    cross_sewing: list[dict] = []
    available_patterns: set[str] = set()
    subsets = UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(mesh_prim))
    for subset in subsets:
        subset_prim = subset.GetPrim()
        name = subset_prim.GetName()
        values = [int(value) for value in (subset.GetIndicesAttr().Get() or [])]
        if name.startswith("pattern"):
            available_patterns.add(name)
            if subset.GetElementTypeAttr().Get() != "face":
                raise ValueError(f"Pattern subset is not face-based: {name}")
            pattern_faces[name] = values
        if not name.startswith("sewing"):
            continue
        pattern_a_targets = subset_prim.GetRelationship("patternA").GetTargets()
        pattern_b_targets = subset_prim.GetRelationship("patternB").GetTargets()
        if len(pattern_a_targets) != 1 or len(pattern_b_targets) != 1 or len(values) % 2:
            raise ValueError(f"Malformed SewingAPI subset: {name}")
        record = {
            "name": name,
            "pattern_a": pattern_a_targets[0].name,
            "pattern_b": pattern_b_targets[0].name,
            "pairs": list(zip(values[::2], values[1::2])),
        }
        a_selected = record["pattern_a"] in selected_patterns
        b_selected = record["pattern_b"] in selected_patterns
        if a_selected and b_selected:
            internal_sewing.append(record)
        elif a_selected != b_selected:
            cross_sewing.append(record)

    missing_patterns = sorted(selected_patterns - available_patterns)
    if missing_patterns:
        raise ValueError(f"Requested USD pattern subsets are missing: {missing_patterns}")
    if cross_sewing:
        names = ", ".join(record["name"] for record in cross_sewing)
        raise ValueError(f"Selected garment is sewn to excluded patterns: {names}")

    selected_face_indices = sorted(
        {face for pattern in pattern_names for face in pattern_faces[pattern]}
    )
    if any(face < 0 or face >= len(faces) for face in selected_face_indices):
        raise ValueError("A selected pattern contains an invalid face index")
    selected_source_vertices = sorted(
        {vertex for face in selected_face_indices for vertex in faces[face]}
    )
    selected_source_vertex_set = set(selected_source_vertices)
    disjoint_set = DisjointSet(selected_source_vertices)
    sewing_pair_count = 0
    maximum_sewing_distance = 0.0
    for sewing in internal_sewing:
        for first, second in sewing["pairs"]:
            if first not in selected_source_vertex_set or second not in selected_source_vertex_set:
                raise ValueError(f"Sewing subset references a vertex outside selected faces: {sewing['name']}")
            distance = length(subtract(source_positions[first], source_positions[second]))
            maximum_sewing_distance = max(maximum_sewing_distance, distance)
            if distance > args.sewing_tolerance:
                raise ValueError(
                    f"Sewing pair {first}/{second} is {distance:g} m apart, beyond tolerance"
                )
            disjoint_set.union(first, second)
            sewing_pair_count += 1

    groups: dict[int, list[int]] = defaultdict(list)
    for vertex in selected_source_vertices:
        groups[disjoint_set.find(vertex)].append(vertex)
    roots = sorted(groups)
    welded_index = {root: index for index, root in enumerate(roots)}
    source_to_welded = {
        source: welded_index[disjoint_set.find(source)] for source in selected_source_vertices
    }
    welded_source_positions = [
        tuple(
            sum(source_positions[source][axis] for source in groups[root]) / len(groups[root])
            for axis in range(3)
        )
        for root in roots
    ]
    triangles = [
        tuple(source_to_welded[vertex] for vertex in faces[face_index])
        for face_index in selected_face_indices
    ]
    if any(len(set(triangle)) != 3 for triangle in triangles):
        raise ValueError("Sewing weld generated a degenerate triangle")
    if len({tuple(sorted(triangle)) for triangle in triangles}) != len(triangles):
        raise ValueError("Sewing weld generated duplicate triangles")

    loops, edge_counts = boundary_loops(triangles)
    non_manifold_edges = [edge for edge, count in edge_counts.items() if count > 2]
    if non_manifold_edges:
        raise ValueError(f"Sewing weld generated {len(non_manifold_edges)} non-manifold edges")
    if not loops:
        raise ValueError("Selected mesh has no boundary loop to pin")
    waist_loop = max(
        loops,
        key=lambda loop: sum(welded_source_positions[v][1] for v in loop) / len(loop),
    )
    pinned = set(waist_loop)

    center_x = (min(point[0] for point in welded_source_positions) + max(point[0] for point in welded_source_positions)) * 0.5
    center_z = (min(point[2] for point in welded_source_positions) + max(point[2] for point in welded_source_positions)) * 0.5
    waist_top = max(welded_source_positions[vertex][1] for vertex in pinned)
    positions = [
        (point[0] - center_x, point[1] - waist_top, point[2] - center_z)
        for point in welded_source_positions
    ]

    minimum_x = min(point[0] for point in positions)
    maximum_x = max(point[0] for point in positions)
    minimum_y = min(point[1] for point in positions)
    maximum_y = max(point[1] for point in positions)
    width = max(maximum_x - minimum_x, 1.0e-9)
    height = max(maximum_y - minimum_y, 1.0e-9)
    uvs = [
        ((point[0] - minimum_x) / width, (maximum_y - point[1]) / height)
        for point in positions
    ]

    unique_edges = sorted(edge_counts)
    adjacency: dict[int, set[int]] = defaultdict(set)
    for first, second in unique_edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    csr_offsets = [0]
    csr_neighbors: list[int] = []
    for vertex in range(len(positions)):
        csr_neighbors.extend(sorted(adjacency[vertex]))
        csr_offsets.append(len(csr_neighbors))

    stretch_edges, stretch_color_offsets = color_constraints(
        unique_edges, lambda edge: edge
    )

    edge_faces: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for triangle in triangles:
        for index in range(3):
            first = triangle[index]
            second = triangle[(index + 1) % 3]
            opposite = triangle[(index + 2) % 3]
            edge_faces[tuple(sorted((first, second)))].append((first, second, opposite))
    bend_records: list[tuple[tuple[int, int, int, int], float]] = []
    for edge in sorted(edge_faces):
        adjacent = edge_faces[edge]
        if len(adjacent) != 2:
            continue
        first, second = adjacent
        a, b, c = first
        d = second[2]
        edge_direction = normalize(subtract(positions[b], positions[a]))
        normal_first = normalize(cross(subtract(positions[b], positions[a]), subtract(positions[c], positions[a])))
        normal_second = normalize(
            cross(
                subtract(positions[second[1]], positions[second[0]]),
                subtract(positions[d], positions[second[0]]),
            )
        )
        cosine = max(-1.0, min(1.0, dot(normal_first, normal_second)))
        sine = dot(cross(normal_first, normal_second), edge_direction)
        rest_angle = math.atan2(sine, cosine)
        bend_records.append(((a, b, c, d), rest_angle))
    bend_records, bend_color_offsets = color_constraints(
        bend_records, lambda record: record[0]
    )
    bend_quads = [record[0] for record in bend_records]
    bend_rest_angles = [record[1] for record in bend_records]

    pin_mask = [1 if vertex in pinned else 0 for vertex in range(len(positions))]
    source_vertex = [min(groups[root]) for root in roots]

    sections = [
        {"name": "positions", "count": len(positions), "stride": 12, "encoding": "float32x3", "data": pack_floats(flatten(positions))},
        {"name": "uv", "count": len(uvs), "stride": 8, "encoding": "float32x2", "data": pack_floats(flatten(uvs))},
        {"name": "triangles", "count": len(triangles), "stride": 12, "encoding": "uint32x3", "data": pack_uints(flatten(triangles))},
        {"name": "csr_offsets", "count": len(csr_offsets), "stride": 4, "encoding": "uint32", "data": pack_uints(csr_offsets)},
        {"name": "csr_neighbors", "count": len(csr_neighbors), "stride": 4, "encoding": "uint32", "data": pack_uints(csr_neighbors)},
        {"name": "stretch_edges", "count": len(stretch_edges), "stride": 8, "encoding": "uint32x2", "data": pack_uints(flatten(stretch_edges))},
        {"name": "stretch_colors", "count": len(stretch_color_offsets), "stride": 4, "encoding": "uint32", "data": pack_uints(stretch_color_offsets)},
        {"name": "bend_quads", "count": len(bend_quads), "stride": 16, "encoding": "uint32x4", "data": pack_uints(flatten(bend_quads))},
        {"name": "bend_rest", "count": len(bend_rest_angles), "stride": 4, "encoding": "float32_radians", "data": pack_floats(bend_rest_angles)},
        {"name": "bend_colors", "count": len(bend_color_offsets), "stride": 4, "encoding": "uint32", "data": pack_uints(bend_color_offsets)},
        {"name": "pin_mask", "count": len(pin_mask), "stride": 4, "encoding": "uint32_bool", "data": pack_uints(pin_mask)},
        {"name": "source_vertex", "count": len(source_vertex), "stride": 4, "encoding": "uint32", "data": pack_uints(source_vertex)},
    ]

    binary_path = output_directory / f"{args.name}.vcloth"
    obj_path = output_directory / f"{args.name}.obj"
    metadata_path = output_directory / f"{args.name}.json"
    binary_metadata = write_vcloth(binary_path, sections)
    validate_vcloth(binary_path)
    write_obj(obj_path, positions, uvs, triangles, pinned)

    degrees = [len(adjacency[vertex]) for vertex in range(len(positions))]
    metadata = {
        "asset": args.name,
        "source": {
            "file": source_path.name,
            "sha256": sha256_file(source_path),
            "mesh_prim": args.mesh_prim,
            "stage_up_axis": str(UsdGeom.GetStageUpAxis(stage)),
            "stage_meters_per_unit": meters_per_unit,
            "selected_patterns": pattern_names,
        },
        "conversion": {
            "coordinate_system": "right_handed_y_up_meters",
            "origin": "waist top at Y=0; X/Z centered on selected source bounds",
            "sewing_tolerance_m": args.sewing_tolerance,
            "sewing_groups": [record["name"] for record in internal_sewing],
            "sewing_pairs": sewing_pair_count,
            "maximum_sewing_pair_distance_m": maximum_sewing_distance,
            "source_vertices": len(selected_source_vertices),
            "welded_vertices": len(positions),
        },
        "topology": {
            "vertices": len(positions),
            "triangles": len(triangles),
            "undirected_edges": len(unique_edges),
            "directed_csr_edges": len(csr_neighbors),
            "boundary_edges": sum(1 for count in edge_counts.values() if count == 1),
            "boundary_loops": [
                {
                    "vertices": len(loop),
                    "minimum_y_m": min(positions[vertex][1] for vertex in loop),
                    "maximum_y_m": max(positions[vertex][1] for vertex in loop),
                }
                for loop in sorted(loops, key=len, reverse=True)
            ],
            "non_manifold_edges": 0,
            "minimum_degree": min(degrees),
            "maximum_degree": max(degrees),
            "mean_degree": sum(degrees) / len(degrees),
        },
        "constraints": {
            "stretch_edges": len(stretch_edges),
            "stretch_colors": len(stretch_color_offsets) - 1,
            "dihedral_bend_constraints": len(bend_quads),
            "bend_colors": len(bend_color_offsets) - 1,
            "pinned_vertices": len(pinned),
            "pin_rule": "highest-average-Y boundary loop after sewing weld",
            "pin_indices": sorted(pinned),
        },
        "bounds_m": {
            "minimum": [min(point[axis] for point in positions) for axis in range(3)],
            "maximum": [max(point[axis] for point in positions) for axis in range(3)],
        },
        "files": {
            binary_path.name: binary_metadata,
            obj_path.name: {
                "format": "Wavefront OBJ",
                "sha256": sha256_file(obj_path),
                "purpose": "human inspection/debug visualization; simulation uses .vcloth",
            },
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"Wrote {binary_path}")
    print(f"Wrote {obj_path}")
    print(f"Wrote {metadata_path}")
    print(
        f"vertices={len(positions)} triangles={len(triangles)} "
        f"csr_edges={len(csr_neighbors)} stretch={len(stretch_edges)} "
        f"bend={len(bend_quads)} pinned={len(pinned)}"
    )


if __name__ == "__main__":
    main()
