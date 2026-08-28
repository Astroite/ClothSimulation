#!/usr/bin/env python3
"""Turn the MLCloth training export's cloth topology into the PoC's `.mlmesh` asset.

Why this reads a training export rather than talking to Unreal
-------------------------------------------------------------
The 5,294 outputs of the model belong to the LOD0 sim mesh of the ChaosClothAsset
(`CCA_JZ_CH10032_2607_ML1`). That mesh is *not reachable from UE Python*: on
`UChaosClothAsset` the sim model is a bare `TSharedPtr<FChaosClothSimulationModel>`
rather than a `UPROPERTY`, `GetClothSimulationModel` is not a `UFUNCTION`, and
`ClothCollections` is not reflected either (checked against the 5.8 headers in
`Engine/Plugins/ChaosClothAsset/.../ClothAsset.h`). `FClothPatternToDynamicMesh`
with `Sim3D` + `PatternIndex = INDEX_NONE` would give exactly the welded ordering
we need, but it is unreflected C++ too, and the one reflected converter that does
exist (`SkeletalMeshConverter`) asks for `EClothPatternVertexType::Render`, which
is the wrong mesh.

The MLCloth plugin already solved this for its own training pipeline. It reads
`GetClothSimulationModel(assetIndex)->GetIndices(LODIndex)` and writes it to
`cloth_topology.bin`, with `manifest.json` recording the guarantee this whole
converter rests on:

    "cloth_sim.bin positions use zero-based array order;
     cloth_topology.bin indices address that same array order"

and that same array order is what the inference output uses, because
`MLClothSimulationProxy::TickGameThread` memcpys the network output straight into
the buffer whose layout came from `CacheSimModel`. So a training export is a
first-class source for this asset, not a workaround -- it is the plugin's own
documented topology export, consumed offline with no Unreal dependency at all.

Formats, decoded from AIClothTrainingAnimationBuilder.cpp (magic is 8 bytes with
no terminator; every scalar is little-endian):

    cloth_topology.bin -- "AICCTOP1", i32 version=1, i32 actorCount, then per actor
        i32 assetIndex, i32 lodIndex, i32 vertexCount, i32 indexCount,
        u32 topologyHash, u32 indices[indexCount]

    cloth_sim.bin -- "AICCLTH1", i32 version=1, i32 fps, i32 frameCount,
        i32 actorCount, then per actor { i32 assetIndex, i32 vertexCount },
        then per frame per actor:
            i32 lodIndex, transform, componentRelativeTransform,
            f32 positions[vertexCount * 3], f32 normals[vertexCount * 3]
        "AICCLTH2" is the compact variant: version=2, one extra i32 flag word
        after actorCount (bit 0 = normals present), and normals are omitted when
        that bit is clear.
        A transform is 10 floats: translation xyz, rotation xyzw, scale xyz.

`topologyHash` is UE's `FCrc::MemCrc32`, which is not a standard CRC-32; it is
carried through into the report for traceability but deliberately NOT recomputed
here, because a reimplementation that silently disagreed would be worse than no
check. The payload SHA-256 in the container is the integrity check that matters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from collections import defaultdict
from pathlib import Path

POC_ROOT = Path(__file__).resolve().parents[1]
GNN_POC = POC_ROOT.parent / "vulkan-gnn-poc"

MAGIC = b"MLMSH001"
VERSION = 1
CLOTH_VERTEX_COUNT = 5294

# Measured on the same garment's lower-body sim mesh in the sibling PoC
# (`Assets/Meshes/CH10032_lower_sim_lod1.json`: density_kg_m2 0.2002200037240982).
# It is a *global* constant, not a per-asset authored value -- which is the point
# of the zero-configuration claim, so it stays a CLI default rather than becoming
# something a new garment has to have tuned.
DEFAULT_DENSITY_KG_PER_M2 = 0.2002200037240982


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"bake_cloth_topology: {message}")


def load_write_sectioned():
    """Borrow the sibling PoC's container writer instead of writing a third one.

    Two implementations of this layout already exist -- `real_scene/formats.py` and
    `mlcloth_formats.cpp::pack_sections` -- and the strict C++ `parse_mesh` is the
    arbiter that would catch a disagreement. Adding a third here would only add a
    way for them to drift. `formats.py` is standard-library only, so the import
    costs nothing beyond the path.
    """
    if not (GNN_POC / "real_scene" / "formats.py").is_file():
        fail(
            f"the sibling GNN PoC is required for the container writer but was not found at {GNN_POC}. "
            "It supplies real_scene/formats.py::write_sectioned."
        )
    sys.path.insert(0, str(GNN_POC))
    from real_scene.formats import Section, write_sectioned  # noqa: E402

    return Section, write_sectioned


class Reader:
    """Little-endian cursor with bounds checking, so a truncated file fails loudly."""

    def __init__(self, data: bytes, label: str) -> None:
        self.data = data
        self.label = label
        self.offset = 0

    def take(self, count: int) -> bytes:
        if count < 0 or self.offset + count > len(self.data):
            fail(f"{self.label} is truncated at byte {self.offset} (wanted {count} more)")
        chunk = self.data[self.offset : self.offset + count]
        self.offset += count
        return chunk

    def magic(self, expected: bytes) -> None:
        found = self.take(8)
        if found != expected:
            fail(f"{self.label} magic is {found!r}, expected {expected!r}")

    def i32(self) -> int:
        return struct.unpack("<i", self.take(4))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u32_array(self, count: int) -> tuple[int, ...]:
        return struct.unpack(f"<{count}I", self.take(4 * count))

    def f32_array(self, count: int) -> tuple[float, ...]:
        return struct.unpack(f"<{count}f", self.take(4 * count))

    def transform(self) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
        """Translation xyz, rotation xyzw, scale xyz -- 40 bytes."""
        values = self.f32_array(10)
        return values[0:3], values[3:7], values[7:10]

    def skip_transform(self) -> None:
        self.transform()

    def expect_end(self) -> None:
        if self.offset != len(self.data):
            fail(f"{self.label} has {len(self.data) - self.offset} trailing bytes")


def read_model(path: Path) -> tuple[int, str]:
    """Derived vertex count and SHA-256 of the encoded model.

    Same strict field set the C++ `parse_model` requires, so a model this converter
    accepts is one the runtime will also accept.
    """
    data = path.read_bytes()
    if len(data) < 5:
        fail(f"encoded model is truncated: {path}")
    json_bytes = struct.unpack_from("<I", data, 0)[0]
    if json_bytes <= 1 or 4 + json_bytes >= len(data):
        fail(f"encoded model JSON length is invalid: {path}")
    try:
        config = json.loads(data[4 : 4 + json_bytes].decode("utf-8"))
    except Exception as exc:
        fail(f"encoded model JSON is invalid: {exc}")
    for key, expected in (("modelType", 2), ("driverFeatureLen", 1969), ("drivenFeatureLen", 16394), ("pcaDim", 512)):
        if config.get(key) != expected:
            fail(f"encoded model {key}: expected {expected}, got {config.get(key)!r}")
    vertices = (config["drivenFeatureLen"] - config["pcaDim"]) // 3
    if vertices != CLOTH_VERTEX_COUNT:
        fail(f"encoded model derives {vertices} vertices, expected {CLOTH_VERTEX_COUNT}")
    return vertices, hashlib.sha256(data).hexdigest()


def read_topology(path: Path, asset_index: int, lod_index: int) -> tuple[int, list[int], int]:
    reader = Reader(path.read_bytes(), path.name)
    reader.magic(b"AICCTOP1")
    if reader.i32() != 1:
        fail(f"{path.name} version is not 1")
    actor_count = reader.i32()
    if actor_count <= 0:
        fail(f"{path.name} declares no actors")
    found: dict[tuple[int, int], tuple[int, list[int], int]] = {}
    for _ in range(actor_count):
        actor_asset = reader.i32()
        actor_lod = reader.i32()
        vertex_count = reader.i32()
        index_count = reader.i32()
        topology_hash = reader.u32()
        if index_count <= 0 or index_count % 3 != 0:
            fail(f"{path.name} actor {actor_asset} has {index_count} indices, not a positive multiple of three")
        indices = list(reader.u32_array(index_count))
        found[(actor_asset, actor_lod)] = (vertex_count, indices, topology_hash)
    reader.expect_end()
    key = (asset_index, lod_index)
    if key not in found:
        fail(
            f"{path.name} has no actor for assetIndex={asset_index} lodIndex={lod_index}; "
            f"available: {sorted(found)}"
        )
    return found[key]


def read_sim_frame(
    path: Path, asset_index: int, lod_index: int, frame: int
) -> tuple[list[float], int, int, tuple[float, ...]]:
    """Positions of one actor in one frame, plus fps, frame count and reference rotation.

    The fourth return value is the quaternion of the frame's `transform`, which is the
    reference bone's own component transform (verified on the real export: `transform`
    and `componentRelativeTransform` are equal, and the translation is Root_M's
    component position). `reference_up_axis` turns it into the local axis that points
    up, which is the only reliable way to know that -- see the note there.
    """
    reader = Reader(path.read_bytes(), path.name)
    magic = reader.take(8)
    if magic == b"AICCLTH1":
        compact = False
    elif magic == b"AICCLTH2":
        compact = True
    else:
        fail(f"{path.name} magic is {magic!r}, expected AICCLTH1 or AICCLTH2")
    version = reader.i32()
    if version != (2 if compact else 1):
        fail(f"{path.name} version {version} does not match its magic")
    fps = reader.i32()
    frame_count = reader.i32()
    actor_count = reader.i32()
    normals_present = True
    if compact:
        normals_present = bool(reader.i32() & 1)
    if frame_count <= 0 or actor_count <= 0:
        fail(f"{path.name} declares {frame_count} frames and {actor_count} actors")
    layout = [(reader.i32(), reader.i32()) for _ in range(actor_count)]
    # Negative indices are resolved by the caller, deliberately not here: resolving in
    # both places turned --rest-frame -4 on a 3-frame clip into frame 2 instead of an
    # error, which is the exact class of silent index bug this converter exists to refuse.
    if frame < 0 or frame >= frame_count:
        fail(f"rest frame {frame} is outside {path.name}'s {frame_count} frames")

    selected: list[float] | None = None
    reference_rotation: tuple[float, ...] | None = None
    # Every frame is walked even though only one is wanted: reaching the exact end of
    # the file is what proves the layout was parsed correctly rather than merely
    # plausibly, and `expect_end` is the assertion that says so.
    for frame_index in range(frame_count):
        for actor_asset, vertex_count in layout:
            actor_lod = reader.i32()
            _translation, rotation, _scale = reader.transform()
            reader.skip_transform()
            positions = reader.f32_array(vertex_count * 3)
            if normals_present:
                reader.take(vertex_count * 3 * 4)
            if frame_index == frame and actor_asset == asset_index and actor_lod == lod_index:
                selected = list(positions)
                reference_rotation = rotation
    reader.expect_end()
    if selected is None or reference_rotation is None:
        fail(
            f"{path.name} frame {frame} has no actor with assetIndex={asset_index} lodIndex={lod_index}; "
            f"layout={layout}"
        )
    if not all(math.isfinite(value) for value in selected):
        fail(f"{path.name} frame {frame} contains a non-finite position")
    return selected, fps, frame_count, reference_rotation


def reference_up_axis(rotation: tuple[float, ...]) -> tuple[str, float]:
    """Which reference-bone-local axis points up in component space, and how squarely.

    Root_M-local space is *not* axis-aligned with the component space it sits in. On
    CH10032 the reference bone's rotation maps local X to component +Z (world up), local
    Y to -Y and local Z to +X, so "the highest boundary loop" has to be measured along
    local **X**. Taking the up axis as a caller-supplied flag was a mistake: the default
    `z` produced a perfectly well-formed mesh whose pin set was an arbitrary 21-vertex
    slice through a horizontal-in-the-wrong-sense direction, with nothing anywhere
    reporting a problem. The rotation that answers the question is in the file, so it is
    read from the file.

    Returns the axis name and the absolute dot product with component +Z, so a rig whose
    bone axes are not near-aligned with anything shows up as a low score instead of an
    arbitrary pick.
    """
    x, y, z, w = rotation
    # Columns of the rotation matrix: where each local axis lands in component space.
    columns = {
        "x": (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w)),
        "y": (2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w)),
        "z": (2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y)),
    }
    # Component +Z is up: the export drives a PoseableMeshComponent, and the reference
    # transform's own translation is the pelvis height in that space.
    axis, alignment = max(((name, abs(col[2])) for name, col in columns.items()), key=lambda item: item[1])
    return axis, alignment


def quaternion_rotate(q, v):
    x, y, z, w = q
    vx, vy, vz = v
    tx, ty, tz = 2.0 * (y * vz - z * vy), 2.0 * (z * vx - x * vz), 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def read_bone_components(path: Path) -> tuple[int, int, bytes]:
    """Frame count, bone count and the raw blob of a bones_component.bin.

    Header is 28 bytes: magic[8], i32 version=2, i32 fps, i32 frameCount, i32 boneCount,
    i32 spaceFlag (0 local, 1 component). Then frameCount * boneCount transforms of ten
    floats: translation xyz, rotation xyzw, scale xyz. The bone order is the model's
    `driverNames` order, verified element by element against the real export.
    """
    blob = path.read_bytes()
    if len(blob) < 28 or blob[:8] != b"AICBCMP2":
        fail(f"{path.name} is not an AICBCMP2 component-space bone clip")
    version, _fps, frames, bones, space = struct.unpack_from("<5i", blob, 8)
    if version != 2:
        fail(f"{path.name} declares version {version}, expected 2")
    if space != 1:
        fail(f"{path.name} declares spaceFlag {space}; component space (1) is required")
    if len(blob) != 28 + 40 * bones * frames:
        fail(f"{path.name} is {len(blob)} bytes, expected {28 + 40 * bones * frames}")
    return frames, bones, blob


def read_sim_trajectory(path: Path, asset_index: int, lod_index: int, step: int):
    """Every `step`-th frame of one actor as (reference translation, rotation, positions).

    Unlike `read_sim_frame` this keeps many frames, so it walks the file once instead of
    once per frame. It still reaches the exact end of the file, for the same reason.
    """
    reader = Reader(path.read_bytes(), path.name)
    magic = reader.take(8)
    if magic == b"AICCLTH1":
        compact = False
    elif magic == b"AICCLTH2":
        compact = True
    else:
        fail(f"{path.name} magic is {magic!r}, expected AICCLTH1 or AICCLTH2")
    if reader.i32() != (2 if compact else 1):
        fail(f"{path.name} version does not match its magic")
    _fps = reader.i32()
    frame_count = reader.i32()
    actor_count = reader.i32()
    normals_present = bool(reader.i32() & 1) if compact else True
    layout = [(reader.i32(), reader.i32()) for _ in range(actor_count)]

    samples = []
    for frame_index in range(frame_count):
        for actor_asset, vertex_count in layout:
            actor_lod = reader.i32()
            translation, rotation, _scale = reader.transform()
            reader.skip_transform()
            positions = reader.f32_array(vertex_count * 3)
            if normals_present:
                reader.take(vertex_count * 3 * 4)
            wanted = frame_index % step == 0 and actor_asset == asset_index and actor_lod == lod_index
            if wanted:
                samples.append((frame_index, translation, rotation, positions))
    reader.expect_end()
    if not samples:
        fail(f"{path.name} has no frames for assetIndex={asset_index} lodIndex={lod_index}")
    return samples, frame_count


def measure_loop_attachment(
    loops: list[list[int]], clip_dirs: list[Path], asset_index: int, lod_index: int, step: int
) -> tuple[list[dict], list[dict]]:
    """Per boundary loop, how rigidly it rides a driver bone across reference motion.

    This replaces "pin the highest boundary loop", which is wrong on a multi-piece
    garment in a way no amount of care about the geometry would catch: on CH10032 the
    height rule pinned a 21-vertex ring that measurement shows to be a *free cuff*
    tracking Wrist_R, and pinned only one of the three attached loops on another piece.
    Attachment is a property of the motion, not of the rest pose, so it is measured on
    the motion.

    For each loop and each of the 45 driver bones, every loop vertex is expressed in that
    bone's local frame and its positional standard deviation over the sampled frames is
    taken; the score is the mean over the loop's vertices, minimised over bones. A loop
    sewn to the body scores near zero; a hem swings.

    The score is then normalised by the loosest loop in the same clip, because the raw
    magnitude tracks how violent the motion is -- a dodge moves roughly three times as
    much as a run -- and a fixed centimetre threshold flips classification between clips
    while the normalised ordering does not.

    The winning bone's per-vertex mean position in its own frame comes back as well. That
    is the rigid bind the solver-only branch of the comparison drives its pins with, and it
    falls out of the same computation: a least-squares bind over the reference motion rather
    than a single frame's offset.
    """
    if not clip_dirs:
        return [], []
    # Starts below zero so the first clip always populates the bone and the bind. At 0.0 a
    # perfectly rigid loop -- normalised residual exactly 0 -- failed the `>` test on every
    # clip and came out with bone index -1, which then read as a loop with no attachment.
    per_loop_worst = [-1.0] * len(loops)
    best_bone = [-1] * len(loops)
    best_local: list[list[list[float]]] = [[] for _ in loops]
    per_clip = []

    for clip_dir in clip_dirs:
        sim_path, bones_path = clip_dir / "cloth_sim.bin", clip_dir / "bones_component.bin"
        for required in (sim_path, bones_path):
            if not required.is_file():
                fail(f"attachment clip {clip_dir} is missing {required.name}")
        samples, _frames = read_sim_trajectory(sim_path, asset_index, lod_index, step)
        bone_frames, bone_count, bone_blob = read_bone_components(bones_path)

        # Loop vertices in component space, so they can be compared against bone frames.
        world = []
        for frame_index, translation, rotation, positions in samples:
            if frame_index >= bone_frames:
                fail(
                    f"{clip_dir.name}: cloth frame {frame_index} has no matching bone frame "
                    f"({bone_frames} available); the two files disagree about the clip length"
                )
            frame_points = []
            for vertices in loops:
                points = []
                for v in vertices:
                    rotated = quaternion_rotate(rotation, positions[3 * v : 3 * v + 3])
                    points.append(
                        (rotated[0] + translation[0], rotated[1] + translation[1], rotated[2] + translation[2])
                    )
                frame_points.append(points)
            world.append((frame_index, frame_points))

        scores = []
        for loop_index, vertices in enumerate(loops):
            best = (math.inf, -1, [])
            for bone in range(bone_count):
                local = []
                for frame_index, frame_points in world:
                    offset = 28 + 40 * (bone_count * frame_index + bone)
                    bt = struct.unpack_from("<3f", bone_blob, offset)
                    bq = struct.unpack_from("<4f", bone_blob, offset + 12)
                    inverse = (-bq[0], -bq[1], -bq[2], bq[3])
                    local.append(
                        [
                            quaternion_rotate(inverse, (p[0] - bt[0], p[1] - bt[1], p[2] - bt[2]))
                            for p in frame_points[loop_index]
                        ]
                    )
                total = 0.0
                count = len(local)
                bone_means = []
                for j in range(len(vertices)):
                    means = [sum(local[k][j][c] for k in range(count)) / count for c in range(3)]
                    variance = (
                        sum(sum((local[k][j][c] - means[c]) ** 2 for c in range(3)) for k in range(count)) / count
                    )
                    total += math.sqrt(variance)
                    bone_means.append(means)
                score = total / len(vertices)
                if score < best[0]:
                    # The means come along because they are the rigid bind: the vertex's mean
                    # position in the frame that explains its motion best. Recomputing them
                    # later from a single frame would give a different answer, and a worse one.
                    best = (score, bone, bone_means)
            scores.append(best)

        loosest = max(score for score, _bone, _means in scores)
        if loosest <= 0.0:
            fail(f"{clip_dir.name}: every boundary loop is perfectly rigid, so this clip carries no cloth motion")
        normalised = [score / loosest for score, _bone, _means in scores]
        per_clip.append({"clip": clip_dir.name, "normalised": [round(value, 6) for value in normalised]})
        for loop_index, value in enumerate(normalised):
            if value > per_loop_worst[loop_index]:
                per_loop_worst[loop_index] = value
                best_bone[loop_index] = scores[loop_index][1]
                best_local[loop_index] = scores[loop_index][2]

    return (
        [
            {
                "worst_normalised_residual": round(per_loop_worst[i], 6),
                "best_bone_index": best_bone[i],
                "bone_local_cm": best_local[i],
            }
            for i in range(len(loops))
        ],
        per_clip,
    )


def derive_topology(vertex_count: int, triangles: list[tuple[int, int, int]]):
    """Edges, both CSRs, boundary structure -- the same derivation `write_mesh` runs.

    Deliberately duplicated rather than shared: the C++ `parse_mesh` re-derives the
    edge set from the triangles and refuses a file whose edge list is not exactly
    that set, so if this and the C++ side ever disagree the asset fails to load
    instead of loading with a subtly wrong constraint graph.
    """
    use: defaultdict[tuple[int, int], int] = defaultdict(int)
    for corners in triangles:
        for k in range(3):
            a, b = corners[k], corners[(k + 1) % 3]
            if a == b:
                fail("a triangle has a repeated corner")
            use[(min(a, b), max(a, b))] += 1
    edges = sorted(use)
    for (a, b), count in use.items():
        if count > 2:
            fail(f"edge ({a}, {b}) is shared by {count} triangles; the sim mesh is not manifold")

    directed = sorted((a, b) for edge in edges for a, b in (edge, edge[::-1]))
    edge_offsets = [0] * (vertex_count + 1)
    for a, _ in directed:
        edge_offsets[a + 1] += 1
    for v in range(vertex_count):
        edge_offsets[v + 1] += edge_offsets[v]
    edge_neighbours = [b for _, b in directed]

    incident = sorted((corner, index) for index, corners in enumerate(triangles) for corner in corners)
    tri_offsets = [0] * (vertex_count + 1)
    for v, _ in incident:
        tri_offsets[v + 1] += 1
    for v in range(vertex_count):
        tri_offsets[v + 1] += tri_offsets[v]
    tri_indices = [t for _, t in incident]
    max_valence = max(tri_offsets[v + 1] - tri_offsets[v] for v in range(vertex_count))

    boundary = [edge for edge in edges if use[edge] == 1]
    boundary_degree = [0] * vertex_count
    parent = list(range(vertex_count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in boundary:
        boundary_degree[a] += 1
        boundary_degree[b] += 1
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for v in range(vertex_count):
        if boundary_degree[v] not in (0, 2):
            fail(f"vertex {v} sits on {boundary_degree[v]} boundary edges; the boundary is not manifold")
    loops: defaultdict[int, list[int]] = defaultdict(list)
    for v in range(vertex_count):
        if boundary_degree[v]:
            loops[find(v)].append(v)

    return {
        "edges": edges,
        "edge_offsets": edge_offsets,
        "edge_neighbours": edge_neighbours,
        "tri_offsets": tri_offsets,
        "tri_indices": tri_indices,
        "max_valence": max_valence,
        "boundary_edges": len(boundary),
        "loops": list(loops.values()),
    }


def area_weighted_mass(
    positions_cm: list[float], triangles, vertex_count: int, density: float
) -> tuple[list[float], float]:
    """Per-vertex mass from a third of each incident triangle's area, times density.

    Positions are centimetres and the density is kg/m^2, so areas are scaled by
    1e-4. A vertex with no incident area would get zero mass, which `parse_mesh`
    rejects, so it is floored -- an isolated vertex is a bake bug, but a zero
    divide in the solver would be a much less legible symptom of it.
    """
    mass = [0.0] * vertex_count
    total_area_m2 = 0.0
    for a, b, c in triangles:
        ax, ay, az = positions_cm[3 * a : 3 * a + 3]
        bx, by, bz = positions_cm[3 * b : 3 * b + 3]
        cx, cy, cz = positions_cm[3 * c : 3 * c + 3]
        ux, uy, uz = bx - ax, by - ay, bz - az
        vx, vy, vz = cx - ax, cy - ay, cz - az
        nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        area_m2 = 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz) * 1.0e-4
        total_area_m2 += area_m2
        share = area_m2 * density / 3.0
        mass[a] += share
        mass[b] += share
        mass[c] += share
    floor = 1.0e-9
    return [value if value > floor else floor for value in mass], total_area_m2


def surface_components(vertex_count: int, triangles) -> list[list[int]]:
    """Connected components of the triangle graph, largest first."""
    parent = list(range(vertex_count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, c in triangles:
        for u, v in ((a, b), (b, c)):
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
    groups: defaultdict[int, list[int]] = defaultdict(list)
    for v in range(vertex_count):
        groups[find(v)].append(v)
    return sorted(groups.values(), key=len, reverse=True)


def choose_pin_loops(
    loops: list[list[int]], components: list[list[int]], positions_cm: list[float], axis: str
) -> tuple[list[int], list[dict]]:
    """Pin the highest boundary loop of **each** connected component.

    The sibling PoC pinned "the highest boundary loop" full stop, which is correct for
    a single-piece skirt and wrong here: this garment is four disconnected pieces with
    nine boundary loops between them, so a single global winner leaves three pieces with
    no kinematic attachment at all. Under a position-based solver those pieces do not
    misbehave subtly -- they fall off the character -- but the *bake* looks entirely
    healthy, which is why the per-component structure is reported rather than reduced to
    a pin count.

    A component with no boundary loop cannot be pinned by this rule and is reported as
    such instead of being silently skipped, because a closed piece needs a different
    rule rather than no rule.
    """
    component_of = {}
    for index, vertices in enumerate(components):
        for v in vertices:
            component_of[v] = index

    offset = {"x": 0, "y": 1, "z": 2}[axis]
    summary = []
    for vertices in loops:
        values = [positions_cm[3 * v + offset] for v in vertices]
        summary.append(
            {
                "vertices": len(vertices),
                "component": component_of[vertices[0]],
                "mean_cm": round(sum(values) / len(values), 4),
                "minimum_cm": round(min(values), 4),
                "maximum_cm": round(max(values), 4),
            }
        )

    order = sorted(range(len(loops)), key=lambda i: summary[i]["mean_cm"], reverse=True)
    winner_of_component: dict[int, int] = {}
    for rank, index in enumerate(order):
        summary[index]["rank_by_mean"] = rank
        component = summary[index]["component"]
        summary[index]["pinned"] = component not in winner_of_component
        winner_of_component.setdefault(component, index)

    unpinnable = [index for index in range(len(components)) if index not in winner_of_component]
    if unpinnable:
        fail(
            "these surface components have no boundary loop to pin, so the "
            f"highest-loop rule cannot attach them: {unpinnable}"
        )

    # How decisively each winner won. "Highest loop" is only a meaningful rule when the
    # candidates are well separated; on this garment two components pick their loop by
    # under half a centimetre of mean height, and a rule decided by rounding is a rule
    # that should be read by a human rather than trusted. Reported, not resolved.
    for component, winner in winner_of_component.items():
        rivals = [
            summary[i]["mean_cm"] for i in range(len(loops)) if summary[i]["component"] == component and i != winner
        ]
        summary[winner]["margin_cm"] = round(summary[winner]["mean_cm"] - max(rivals), 4) if rivals else None

    pinned = sorted(v for index in winner_of_component.values() for v in loops[index])
    return pinned, summary


def choose_pin_loops_by_attachment(
    loops: list[list[int]],
    components: list[list[int]],
    positions_cm: list[float],
    axis: str,
    attachment: list[dict],
    attached_below: float,
    free_above: float,
) -> tuple[list[int], list[dict]]:
    """Pin every boundary loop the reference motion shows to be body-locked.

    Two thresholds rather than one, with the band between them a hard error: a loop whose
    normalised residual lands in the gap is a loop this rule cannot classify, and the
    honest response is to say so. On CH10032 the six attached loops top out at 0.036 and
    the next loosest is 0.158, a 7.2x gap, so the band is empty by a wide margin -- but
    the point of checking is that a different garment need not be so obliging.
    """
    _pinned_by_height, summary = choose_pin_loops(loops, components, positions_cm, axis)
    for index, record in enumerate(attachment):
        summary[index]["attachment_residual"] = record["worst_normalised_residual"]
        summary[index]["attachment_bone_index"] = record["best_bone_index"]
        summary[index]["pinned_by_height_rule"] = summary[index]["pinned"]

    ambiguous = [
        index
        for index, record in enumerate(attachment)
        if attached_below <= record["worst_normalised_residual"] <= free_above
    ]
    if ambiguous:
        detail = ", ".join(
            f"loop {i} (component {summary[i]['component']}, {summary[i]['vertices']} vertices, "
            f"residual {attachment[i]['worst_normalised_residual']:.4f})"
            for i in ambiguous
        )
        fail(
            f"these boundary loops fall in the undecidable band [{attached_below}, {free_above}] "
            f"between attached and free, so the pin set cannot be measured: {detail}"
        )

    for index, record in enumerate(attachment):
        summary[index]["pinned"] = record["worst_normalised_residual"] < attached_below

    component_of_loop = {index: summary[index]["component"] for index in range(len(loops))}
    attached_components = {component_of_loop[i] for i in range(len(loops)) if summary[i]["pinned"]}
    orphans = sorted(set(range(len(components))) - attached_components)
    if orphans:
        fail(
            "these surface components have no loop the reference motion shows to be attached, "
            f"so they would be unconstrained: {orphans}"
        )

    pinned = sorted(v for index in range(len(loops)) if summary[index]["pinned"] for v in loops[index])
    return pinned, summary


def sim_frame_count(path: Path) -> int:
    """Just the frame count, so a negative --rest-frame can be resolved before use."""
    reader = Reader(path.read_bytes()[:32], path.name)
    magic = reader.take(8)
    if magic not in (b"AICCLTH1", b"AICCLTH2"):
        fail(f"{path.name} magic is {magic!r}, expected AICCLTH1 or AICCLTH2")
    reader.i32()
    reader.i32()
    return reader.i32()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topology", required=True, type=Path, help="cloth_topology.bin from the training export")
    parser.add_argument("--sim", required=True, type=Path, help="cloth_sim.bin, for the rest configuration")
    parser.add_argument("--model", required=True, type=Path, help="the encoded .enc model this mesh must be locked to")
    parser.add_argument("--manifest", type=Path, default=None, help="manifest.json, checked and recorded as provenance")
    parser.add_argument("--asset-index", type=int, default=0)
    parser.add_argument("--lod", type=int, default=0)
    parser.add_argument(
        "--rest-frame",
        type=int,
        default=-1,
        help="which cloth_sim.bin frame is the rest configuration; -1 (default) is the last "
             "frame. A T-pose reference clip is a settle, so its final frame is the rested "
             "garment. A mid-motion frame is not a rest pose.",
    )
    parser.add_argument("--density", type=float, default=DEFAULT_DENSITY_KG_PER_M2, help="kg/m^2, a global constant")
    parser.add_argument(
        "--attachment-clip",
        type=Path,
        action="append",
        default=None,
        metavar="DIR",
        help="a training-export clip directory (needs cloth_sim.bin and bones_component.bin) used to "
             "measure which boundary loops ride the body. Repeatable; pass several dissimilar motions. "
             "Without any, the bake falls back to the per-component height rule, which is known to "
             "misclassify this garment -- see pin_rule_warning in the report.",
    )
    parser.add_argument("--attachment-step", type=int, default=4, help="frame stride when sampling those clips")
    parser.add_argument(
        "--attached-below",
        type=float,
        default=0.08,
        help="normalised residual under which a loop counts as body-locked",
    )
    parser.add_argument(
        "--free-above",
        type=float,
        default=0.12,
        help="normalised residual over which a loop counts as free. Loops between this and "
             "--attached-below are an error, not a coin flip.",
    )
    parser.add_argument(
        "--up-axis",
        choices=("x", "y", "z"),
        default=None,
        help="optional assertion, not a setting. The up axis is derived from the reference "
             "transform stored in cloth_sim.bin; passing this makes the bake fail if the "
             "derived axis is not the one you expected.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    Section, write_sectioned = load_write_sectioned()

    vertex_count, model_sha256 = read_model(args.model)
    topology_vertices, indices, topology_hash = read_topology(args.topology, args.asset_index, args.lod)
    if topology_vertices != vertex_count:
        fail(
            f"{args.topology.name} declares {topology_vertices} vertices but the model derives {vertex_count}. "
            "This pairing would be silently wrong, so it is refused."
        )
    if max(indices) >= vertex_count:
        fail(f"{args.topology.name} indexes vertex {max(indices)} with only {vertex_count} vertices")
    triangles = [tuple(indices[i : i + 3]) for i in range(0, len(indices), 3)]

    # Resolved before use so the report records the frame that was actually read rather
    # than the -1 that was asked for. -1 is the last frame: a T-pose reference clip is a
    # settle under gravity, so its final frame is the rested garment while frame 0 is the
    # instant after a hard reset -- a pose nobody chose that still looks plausible.
    rest_frame = args.rest_frame
    if rest_frame < 0:
        rest_frame += sim_frame_count(args.sim)
    positions_cm, fps, frame_count, reference_rotation = read_sim_frame(
        args.sim, args.asset_index, args.lod, rest_frame
    )
    if len(positions_cm) != vertex_count * 3:
        fail(f"{args.sim.name} frame {rest_frame} has {len(positions_cm) // 3} vertices, expected {vertex_count}")

    up_axis, up_alignment = reference_up_axis(reference_rotation)
    if up_alignment < 0.9:
        fail(
            f"no reference-bone-local axis is within 26 degrees of component up (best is {up_axis} at "
            f"{up_alignment:.4f}). The highest-loop pin rule needs a well-defined up direction, so "
            "rather than pick one arbitrarily this refuses and asks for an explicit rule."
        )
    if args.up_axis is not None and args.up_axis != up_axis:
        fail(
            f"--up-axis {args.up_axis} contradicts the reference transform in {args.sim.name}, which puts "
            f"up along local {up_axis} (alignment {up_alignment:.4f}). Refusing rather than trusting the flag: "
            "the wrong axis produces a well-formed mesh with a meaningless pin set."
        )

    manifest: dict = {}
    if args.manifest is not None:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        # The one guarantee this converter rests on. If a future export changes the
        # convention, that has to surface here rather than as a scrambled garment.
        convention = manifest.get("clothVertexIndexConvention", "")
        if "same array order" not in convention:
            fail(
                "manifest.json does not state that cloth_sim.bin positions and cloth_topology.bin "
                f"indices share one array order; got {convention!r}"
            )

    derived = derive_topology(vertex_count, triangles)
    mass, total_area_m2 = area_weighted_mass(positions_cm, triangles, vertex_count, args.density)
    if not derived["loops"]:
        fail("the sim mesh is closed (no boundary loop), so there is no waist loop to pin")
    components = surface_components(vertex_count, triangles)
    attachment_clips = args.attachment_clip or []
    attachment, attachment_per_clip = measure_loop_attachment(
        derived["loops"], attachment_clips, args.asset_index, args.lod, max(1, args.attachment_step)
    )
    if attachment:
        pinned, loop_summary = choose_pin_loops_by_attachment(
            derived["loops"],
            components,
            positions_cm,
            up_axis,
            attachment,
            args.attached_below,
            args.free_above,
        )
        pin_rule = (
            "every boundary loop whose worst normalised residual against its best-matching driver bone "
            f"is under {args.attached_below}, measured over {len(attachment_clips)} reference clip(s)"
        )
        pin_rule_warning = None
    else:
        pinned, loop_summary = choose_pin_loops(derived["loops"], components, positions_cm, up_axis)
        pin_rule = (
            f"highest boundary loop of each surface component, by mean {up_axis.upper()} "
            "in reference-bone-local centimetres"
        )
        pin_rule_warning = (
            "No --attachment-clip was given, so pins come from rest-pose height. On CH10032 that rule "
            "is measurably wrong: it pins a free cuff that tracks Wrist_R and under-pins a piece with "
            "three attached loops. Pass reference clips."
        )
    pin_mask = [0] * vertex_count
    for v in pinned:
        pin_mask[v] = 1

    # Per pinned vertex, the driver bone it rides and its mean position in that bone's frame.
    # This is what lets a branch with no network keep its anchor on the body: without it the
    # only pin target available is the rest pose, which does not move, and a solver-only arm
    # would be judged on an anchor error that has nothing to do with the solver. Written only
    # when attachment was measured -- there is nothing to guess it from otherwise.
    NO_DRIVER = 0xFFFFFFFF
    pin_driver = [NO_DRIVER] * vertex_count
    pin_local_cm = [0.0] * (vertex_count * 3)
    if attachment:
        for loop_index, vertices in enumerate(derived["loops"]):
            if not loop_summary[loop_index]["pinned"]:
                continue
            record = attachment[loop_index]
            bone = record["best_bone_index"]
            local = record["bone_local_cm"]
            if bone < 0 or len(local) != len(vertices):
                fail(f"loop {loop_index} is pinned but carries no measured bind")
            for slot, v in enumerate(vertices):
                # Two loops sharing a vertex would be a pinch point, and the second write
                # would silently overwrite the first bind with a different bone.
                if pin_driver[v] != NO_DRIVER and pin_driver[v] != bone:
                    fail(f"vertex {v} is on two pinned loops with different attachment bones")
                pin_driver[v] = bone
                pin_local_cm[3 * v : 3 * v + 3] = [float(c) for c in local[slot]]
        bound = sum(1 for v in range(vertex_count) if pin_driver[v] != NO_DRIVER)
        if bound != len(pinned):
            fail(f"{bound} vertices carry a bind but {len(pinned)} are pinned")

    def u32(values) -> bytes:
        values = list(values)
        return struct.pack(f"<{len(values)}I", *values)

    def f32(values) -> bytes:
        values = list(values)
        return struct.pack(f"<{len(values)}f", *values)

    edge_pairs = [component for edge in derived["edges"] for component in edge]
    info = [
        vertex_count,
        len(triangles),
        len(derived["edges"]),
        derived["boundary_edges"],
        len(derived["loops"]),
        derived["max_valence"],
        len(pinned),
        0,
    ]
    sections = [
        Section("info", 8, 4, u32(info)),
        Section("positions", vertex_count, 12, f32(positions_cm)),
        Section("triangles", len(triangles), 12, u32(indices)),
        Section("edges", len(derived["edges"]), 8, u32(edge_pairs)),
        Section("edge_csr_offs", vertex_count + 1, 4, u32(derived["edge_offsets"])),
        Section("edge_csr_nbr", 2 * len(derived["edges"]), 4, u32(derived["edge_neighbours"])),
        Section("tri_csr_offs", vertex_count + 1, 4, u32(derived["tri_offsets"])),
        Section("tri_csr_idx", 3 * len(triangles), 4, u32(derived["tri_indices"])),
        Section("vertex_mass", vertex_count, 4, f32(mass)),
        Section("pin_mask", vertex_count, 4, u32(pin_mask)),
    ]
    if attachment:
        sections.append(Section("pin_driver", vertex_count, 4, u32(pin_driver)))
        sections.append(Section("pin_local_cm", vertex_count, 12, f32(pin_local_cm)))
    written = write_sectioned(args.output.resolve(), MAGIC, VERSION, sections, source_sha256=model_sha256)

    edge_lengths = sorted(
        math.dist(positions_cm[3 * a : 3 * a + 3], positions_cm[3 * b : 3 * b + 3])
        for a, b in derived["edges"]
    )

    def quantile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        return values[min(len(values) - 1, max(0, math.ceil(fraction * len(values)) - 1))]

    report = {
        "output": str(args.output.resolve()),
        "model": str(args.model.resolve()),
        "model_sha256": model_sha256,
        "topology": str(args.topology.resolve()),
        "topology_source": "MLCloth training export cloth_topology.bin (AICCTOP1)",
        # UE's FCrc::MemCrc32, carried for traceability and deliberately not recomputed.
        "topology_hash_ue_crc32": topology_hash,
        "sim": str(args.sim.resolve()),
        "sim_fps": fps,
        "sim_frames": frame_count,
        "rest_frame": rest_frame,
        "rest_frame_requested": args.rest_frame,
        # The rest pose is a Chaos simulation result, not a fixed asset: two exports of the
        # same T-pose reference produced areas 7% apart and upper-tail edge lengths 49%
        # apart while the median agreed to 0.3%. So the inputs are hashed. A calibration
        # that silently changed because the export was re-run would be indistinguishable
        # from a calibration that changed because the rule changed.
        "sim_sha256": hashlib.sha256(args.sim.read_bytes()).hexdigest(),
        "topology_sha256": hashlib.sha256(args.topology.read_bytes()).hexdigest(),
        "vertices": vertex_count,
        "triangles": len(triangles),
        "edges": len(derived["edges"]),
        "boundary_edges": derived["boundary_edges"],
        "boundary_loops": len(derived["loops"]),
        "surface_components": [
            {
                "index": index,
                "vertices": len(vertices),
                "up_min_cm": round(min(positions_cm[3 * v + {"x": 0, "y": 1, "z": 2}[up_axis]] for v in vertices), 4),
                "up_max_cm": round(max(positions_cm[3 * v + {"x": 0, "y": 1, "z": 2}[up_axis]] for v in vertices), 4),
            }
            for index, vertices in enumerate(components)
        ],
        "max_triangle_valence": derived["max_valence"],
        "pinned_vertices": len(pinned),
        "up_axis": up_axis,
        "up_axis_source": "reference transform in cloth_sim.bin, not a CLI setting",
        "up_axis_alignment_with_component_z": round(up_alignment, 6),
        "pin_rule": pin_rule,
        "pin_rule_warning": pin_rule_warning,
        "attachment_clips": [str(path) for path in attachment_clips],
        "attachment_step": args.attachment_step,
        "attachment_thresholds": {"attached_below": args.attached_below, "free_above": args.free_above},
        "attachment_per_clip": attachment_per_clip,
        # The rigid bind written into pin_driver / pin_local_cm, which is what a solver-only
        # branch uses as its kinematic anchor. Absent when no reference clip was given.
        "pin_bind": {
            "written": bool(attachment),
            "loops": [
                {
                    "loop": index,
                    "bone_index": attachment[index]["best_bone_index"],
                    "vertices": len(derived["loops"][index]),
                }
                for index in range(len(derived["loops"]))
                if attachment and loop_summary[index]["pinned"]
            ],
        },
        "boundary_loop_summary": loop_summary,
        # Euler characteristic. A wrong edge derivation shows up here before it can
        # show up as a wrong constraint graph.
        "euler_characteristic": vertex_count - len(derived["edges"]) + len(triangles),
        "surface_area_m2": round(total_area_m2, 8),
        "density_kg_m2": args.density,
        "total_mass_kg": round(sum(mass), 8),
        "rest_edge_length_cm": {
            "p05": round(quantile(edge_lengths, 0.05), 5),
            "p50": round(quantile(edge_lengths, 0.50), 5),
            "p95": round(quantile(edge_lengths, 0.95), 5),
            "p99": round(quantile(edge_lengths, 0.99), 5),
            "max": round(edge_lengths[-1], 5),
        },
        # Ratios to the median, because that is the statistic that survives a re-export.
        # A broad spread here is the tessellation, not damage -- 27% of this garment's
        # edges are over 1.5x the median by construction -- so the count is reported at
        # several thresholds rather than reduced to one pass/fail number that would
        # invite reading normal panel density as a defect.
        "rest_edge_ratio_to_median": {
            f"over_{int(k * 10)}x_tenths": sum(1 for value in edge_lengths if value > k * quantile(edge_lengths, 0.50))
            for k in (1.5, 2.0, 3.0, 4.0)
        },
        "manifest": str(args.manifest.resolve()) if args.manifest else None,
        "cloth_position_space": manifest.get("clothPositionSpace"),
        "cloth_vertex_index_convention": manifest.get("clothVertexIndexConvention"),
        "chaos_cloth_asset": manifest.get("chaosClothAsset"),
        "payload_sha256": written["payload_sha256"],
        "file_bytes": written["file_bytes"],
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
