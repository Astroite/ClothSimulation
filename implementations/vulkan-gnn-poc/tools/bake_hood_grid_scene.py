"""Bake a deterministic 64x64 hanging sheet and sphere for Fine15.

The synthetic scene deliberately uses the same VCHAR/VANIM/VCLTH containers as
the character path.  This keeps the Vulkan inference and synchronization path
identical while replacing the character/garment geometry with the original PoC
topology: an eight-neighbour regular grid, a fully pinned top row, and a sphere.
No XPBD data is consumed by the Fine15 runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from real_scene.formats import Section, pack_f32, pack_u32, write_sectioned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--width", type=float, default=1.2)
    parser.add_argument("--height", type=float, default=1.2)
    parser.add_argument("--density", type=float, default=0.2)
    parser.add_argument("--sphere-radius", type=float, default=0.3)
    parser.add_argument("--sphere-center", type=float, nargs=3, default=(0.0, -0.15, 0.28))
    parser.add_argument("--sphere-latitudes", type=int, default=32)
    parser.add_argument("--sphere-longitudes", type=int, default=64)
    return parser.parse_args()


def identity_skin(count: int) -> tuple[list[list[int]], list[list[float]]]:
    indices = [[0] * 12 for _ in range(count)]
    weights = [[1.0, *([0.0] * 11)] for _ in range(count)]
    return indices, weights


def make_uv_sphere(
    center: tuple[float, float, float], radius: float, latitudes: int, longitudes: int
) -> tuple[list[list[float]], list[list[float]], list[list[float]], list[list[int]]]:
    if latitudes < 4 or longitudes < 8:
        raise ValueError("sphere tessellation is too small")
    positions: list[list[float]] = []
    normals: list[list[float]] = []
    uvs: list[list[float]] = []
    triangles: list[list[int]] = []

    def append(normal: tuple[float, float, float], uv: tuple[float, float]) -> int:
        normals.append(list(normal))
        positions.append([center[i] + radius * normal[i] for i in range(3)])
        uvs.append(list(uv))
        return len(positions) - 1

    top = append((0.0, 1.0, 0.0), (0.5, 0.0))
    rings: list[list[int]] = []
    for latitude in range(1, latitudes):
        theta = math.pi * latitude / latitudes
        ring: list[int] = []
        for longitude in range(longitudes):
            phi = 2.0 * math.pi * longitude / longitudes
            normal = (math.sin(theta) * math.cos(phi), math.cos(theta), math.sin(theta) * math.sin(phi))
            ring.append(append(normal, (longitude / longitudes, latitude / latitudes)))
        rings.append(ring)
    bottom = append((0.0, -1.0, 0.0), (0.5, 1.0))

    for longitude in range(longitudes):
        following = (longitude + 1) % longitudes
        triangles.append([top, rings[0][following], rings[0][longitude]])
    for upper, lower in zip(rings, rings[1:]):
        for longitude in range(longitudes):
            following = (longitude + 1) % longitudes
            triangles.append([upper[longitude], upper[following], lower[longitude]])
            triangles.append([upper[following], lower[following], lower[longitude]])
    for longitude in range(longitudes):
        following = (longitude + 1) % longitudes
        triangles.append([rings[-1][longitude], rings[-1][following], bottom])
    return positions, normals, uvs, triangles


def make_grid(
    size: int, width: float, height: float, density: float
) -> tuple[list[list[float]], list[list[float]], list[list[int]], list[int], list[int], list[float]]:
    if size < 2 or width <= 0.0 or height <= 0.0 or density <= 0.0:
        raise ValueError("grid dimensions and density must be positive")
    positions: list[list[float]] = []
    uvs: list[list[float]] = []
    triangles: list[list[int]] = []
    offsets = [0]
    neighbors: list[int] = []
    top_y = height * 0.5
    dx = width / (size - 1)
    dy = height / (size - 1)
    for row in range(size):
        for column in range(size):
            positions.append([-width * 0.5 + column * dx, top_y - row * dy, 0.0])
            uvs.append([column / (size - 1), row / (size - 1)])
            for oy in (-1, 0, 1):
                for ox in (-1, 0, 1):
                    nx, ny = column + ox, row + oy
                    if (ox or oy) and 0 <= nx < size and 0 <= ny < size:
                        neighbors.append(ny * size + nx)
            offsets.append(len(neighbors))
    for row in range(size - 1):
        for column in range(size - 1):
            upper_left = row * size + column
            upper_right = upper_left + 1
            lower_left = upper_left + size
            lower_right = lower_left + 1
            triangles.append([upper_left, upper_right, lower_left])
            triangles.append([upper_right, lower_right, lower_left])

    masses = [0.0] * len(positions)
    triangle_mass = density * dx * dy * 0.5
    for triangle in triangles:
        for vertex in triangle:
            masses[vertex] += triangle_mass / 3.0
    return positions, uvs, triangles, offsets, neighbors, masses


def main() -> None:
    args = parse_args()
    sphere_center = tuple(float(value) for value in args.sphere_center)
    sphere_positions, sphere_normals, sphere_uvs, sphere_triangles = make_uv_sphere(
        sphere_center, args.sphere_radius, args.sphere_latitudes, args.sphere_longitudes
    )
    cloth_positions, cloth_uvs, cloth_triangles, offsets, neighbors, masses = make_grid(
        args.grid, args.width, args.height, args.density
    )
    sphere_indices, sphere_weights = identity_skin(len(sphere_positions))
    cloth_indices, cloth_weights = identity_skin(len(cloth_positions))
    pin_mask = [1 if vertex < args.grid else 0 for vertex in range(len(cloth_positions))]
    source_parameters = {
        "grid": args.grid,
        "width_m": args.width,
        "height_m": args.height,
        "density_kg_m2": args.density,
        "sphere_center_m": sphere_center,
        "sphere_radius_m": args.sphere_radius,
        "sphere_latitudes": args.sphere_latitudes,
        "sphere_longitudes": args.sphere_longitudes,
        "graph": "directed eight-neighbour CSR without self edges",
        "constraint_mode": "top row pinned; no XPBD",
    }
    source_blob = json.dumps(source_parameters, sort_keys=True, separators=(",", ":")).encode("utf-8")
    source_hash = hashlib.sha256(source_blob).digest()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    character_sections = [
        Section("info", 6, 4, pack_u32([len(sphere_positions), len(sphere_triangles), 1, len(sphere_positions), 1, 12])),
        Section("render_pos", len(sphere_positions), 12, pack_f32(sphere_positions)),
        Section("render_nrm", len(sphere_normals), 12, pack_f32(sphere_normals)),
        Section("render_uv", len(sphere_uvs), 8, pack_f32(sphere_uvs)),
        Section("render_tri", len(sphere_triangles), 12, pack_u32(sphere_triangles)),
        Section("tri_material", len(sphere_triangles), 4, pack_u32([0] * len(sphere_triangles))),
        Section("bone_idx", len(sphere_indices), 48, pack_u32(sphere_indices)),
        Section("bone_weight", len(sphere_weights), 48, pack_f32(sphere_weights)),
        Section("proxy_pos", len(sphere_positions), 12, pack_f32(sphere_positions)),
        Section("proxy_nrm", len(sphere_normals), 12, pack_f32(sphere_normals)),
        Section("proxy_bone_idx", len(sphere_indices), 48, pack_u32(sphere_indices)),
        Section("proxy_weight", len(sphere_weights), 48, pack_f32(sphere_weights)),
    ]
    character_meta = write_sectioned(
        output / "hood_grid64.vchar", b"VCHAR001", 1, character_sections, source_sha256=source_hash
    )

    identity = [[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]
    animation_sections = [
        Section("info", 4, 4, pack_u32([1, 1, 30, 0])),
        Section("skin_matrices", 1, 48, pack_f32(identity)),
        Section("root_pos", 1, 12, pack_f32([[0.0, 0.0, 0.0]])),
    ]
    animation_meta = write_sectioned(
        output / "hood_grid64.vanim", b"VANIM001", 1, animation_sections, source_sha256=source_hash
    )

    cloth_sections = [
        Section("positions", len(cloth_positions), 12, pack_f32(cloth_positions)),
        Section("uv", len(cloth_uvs), 8, pack_f32(cloth_uvs)),
        Section("triangles", len(cloth_triangles), 12, pack_u32(cloth_triangles)),
        Section("csr_offsets", len(offsets), 4, pack_u32(offsets)),
        Section("csr_neighbors", len(neighbors), 4, pack_u32(neighbors)),
        Section("pin_mask", len(pin_mask), 4, pack_u32(pin_mask)),
        Section("source_vertex", len(cloth_positions), 4, pack_u32(range(len(cloth_positions)))),
        Section("mass", len(masses), 4, pack_f32(masses)),
        Section("bone_idx", len(cloth_indices), 48, pack_u32(cloth_indices)),
        Section("bone_weight", len(cloth_weights), 48, pack_f32(cloth_weights)),
        Section("coord_params", 4, 4, pack_f32([args.width, args.height, args.sphere_radius, args.density])),
    ]
    cloth_meta = write_sectioned(
        output / "hood_grid64.vcloth2", b"VCLTH002", 2, cloth_sections, source_sha256=source_hash
    )

    metadata = {
        "scene": "hood_grid64",
        "description": "Fine15 on the original regular-grid/sphere topology without XPBD",
        "parameters": source_parameters,
        "cloth_vertices": len(cloth_positions),
        "cloth_triangles": len(cloth_triangles),
        "directed_edges": len(neighbors),
        "pinned_vertices": sum(pin_mask),
        "sphere_vertices": len(sphere_positions),
        "sphere_triangles": len(sphere_triangles),
        "assets": {"character": character_meta, "animation": animation_meta, "cloth": cloth_meta},
    }
    (output / "scene.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"HOOD grid scene: {len(cloth_positions)} cloth vertices, {len(neighbors)} directed edges, "
        f"{sum(pin_mask)} pins, {len(sphere_positions)} sphere proxy vertices"
    )


if __name__ == "__main__":
    main()
