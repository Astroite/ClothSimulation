"""VGNN v1 model, graph utilities, serialization, and reference inference.

The runtime format is deliberately fixed and small.  It is not a generic neural
network container.  All integers and floats are little-endian.
"""

from __future__ import annotations

import binascii
import dataclasses
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


MAGIC = b"VGNN"
VERSION = 1
SCALAR_FP32 = 1
INPUT_DIM = 10
HIDDEN_DIM = 16
OUTPUT_DIM = 3
HEADER = struct.Struct("<4s11I")

GOLDEN_MAGIC = b"VGLD"
GOLDEN_VERSION = 1
GOLDEN_HEADER = struct.Struct("<4s9I3f")

INPUT_SCALE = np.asarray(
    [4.0, 4.0, 4.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 1.0],
    dtype=np.float32,
)
OUTPUT_SCALE = np.asarray([10.0, 10.0, 10.0], dtype=np.float32)


@dataclasses.dataclass(frozen=True)
class Graph:
    grid_size: int
    offsets: np.ndarray
    neighbors: np.ndarray

    @property
    def vertex_count(self) -> int:
        return self.grid_size * self.grid_size


@dataclasses.dataclass
class Weights:
    input_scale: np.ndarray
    output_scale: np.ndarray
    self0: np.ndarray
    neighbor0: np.ndarray
    bias0: np.ndarray
    self1: np.ndarray
    neighbor1: np.ndarray
    bias1: np.ndarray

    def arrays_in_file_order(self) -> tuple[np.ndarray, ...]:
        return (
            self.input_scale,
            self.output_scale,
            self.self0,
            self.neighbor0,
            self.bias0,
            self.self1,
            self.neighbor1,
            self.bias1,
        )


@dataclasses.dataclass(frozen=True)
class GoldenCase:
    graph: Graph
    rest_position: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    pinned: np.ndarray
    external_acceleration: np.ndarray
    expected_acceleration: np.ndarray


def make_grid_graph(grid_size: int) -> Graph:
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    offsets = [0]
    neighbors: list[int] = []
    for y in range(grid_size):
        for x in range(grid_size):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < grid_size and 0 <= ny < grid_size:
                        neighbors.append(ny * grid_size + nx)
            offsets.append(len(neighbors))
    return Graph(
        grid_size=grid_size,
        offsets=np.asarray(offsets, dtype=np.uint32),
        neighbors=np.asarray(neighbors, dtype=np.uint32),
    )


def neighbor_mean_numpy(values: np.ndarray, graph: Graph) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-2] != graph.vertex_count:
        raise ValueError("values vertex dimension does not match graph")
    output = np.zeros_like(values)
    for vertex in range(graph.vertex_count):
        begin = int(graph.offsets[vertex])
        end = int(graph.offsets[vertex + 1])
        output[..., vertex, :] = values[..., graph.neighbors[begin:end], :].mean(axis=-2)
    return output


def neighbor_mean_torch(values: "torch.Tensor", graph: Graph) -> "torch.Tensor":
    import torch

    result = torch.empty_like(values)
    for vertex in range(graph.vertex_count):
        begin = int(graph.offsets[vertex])
        end = int(graph.offsets[vertex + 1])
        indices = torch.as_tensor(
            graph.neighbors[begin:end].astype(np.int64),
            dtype=torch.long,
            device=values.device,
        )
        result[..., vertex, :] = values.index_select(-2, indices).mean(dim=-2)
    return result


def make_rest_positions(grid_size: int, cloth_size: float = 5.0) -> np.ndarray:
    axis = np.linspace(-cloth_size * 0.5, cloth_size * 0.5, grid_size, dtype=np.float32)
    xx, zz = np.meshgrid(axis, axis)
    yy = np.full_like(xx, -2.0)
    return np.stack((xx, yy, zz), axis=-1).reshape(-1, 3).astype(np.float32)


def make_smooth_state(
    graph: Graph,
    rng: np.random.Generator,
    mode_count: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    size = graph.grid_size
    uv = np.linspace(0.0, 1.0, size, dtype=np.float32)
    uu, vv = np.meshgrid(uv, uv)
    displacement = np.zeros((size, size, 3), dtype=np.float32)
    velocity = np.zeros_like(displacement)
    for channel in range(3):
        for _ in range(mode_count):
            kx = int(rng.integers(1, 4))
            ky = int(rng.integers(1, 4))
            phase = float(rng.uniform(-math.pi, math.pi))
            basis = np.sin(math.pi * kx * uu + phase) * np.sin(math.pi * ky * vv)
            displacement[..., channel] += float(rng.uniform(-0.08, 0.08)) * basis
            velocity[..., channel] += float(rng.uniform(-0.35, 0.35)) * basis
    external = np.asarray(
        [rng.uniform(-2.0, 2.0), rng.uniform(7.0, 11.0), rng.uniform(-2.0, 2.0)],
        dtype=np.float32,
    )
    pinned = np.zeros((graph.vertex_count, 1), dtype=np.float32)
    pinned[0, 0] = 1.0
    pinned[size - 1, 0] = 1.0
    return (
        displacement.reshape(-1, 3),
        velocity.reshape(-1, 3),
        np.broadcast_to(external, (graph.vertex_count, 3)).copy(),
        pinned,
    )


def make_features(
    displacement: np.ndarray,
    velocity: np.ndarray,
    external: np.ndarray,
    pinned: np.ndarray,
) -> np.ndarray:
    features = np.concatenate((displacement, velocity, external, pinned), axis=-1)
    if features.shape[-1] != INPUT_DIM:
        raise AssertionError("feature layout no longer matches VGNN v1")
    return np.asarray(features, dtype=np.float32)


def oracle_acceleration(
    displacement: np.ndarray,
    velocity: np.ndarray,
    external: np.ndarray,
    pinned: np.ndarray,
    graph: Graph,
    stiffness: float = 18.0,
    damping: float = 0.9,
) -> np.ndarray:
    laplacian = neighbor_mean_numpy(displacement, graph) - displacement
    acceleration = stiffness * laplacian - damping * velocity + external
    return np.asarray(acceleration * (1.0 - pinned), dtype=np.float32)


def infer_numpy(features: np.ndarray, graph: Graph, weights: Weights) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32) * weights.input_scale
    neighbor_x = neighbor_mean_numpy(x, graph)
    hidden = np.maximum(
        x @ weights.self0.T + neighbor_x @ weights.neighbor0.T + weights.bias0,
        np.float32(0.0),
    ).astype(np.float32)
    neighbor_hidden = neighbor_mean_numpy(hidden, graph)
    output = (
        hidden @ weights.self1.T
        + neighbor_hidden @ weights.neighbor1.T
        + weights.bias1
    ) * weights.output_scale
    output *= 1.0 - features[..., 9:10]
    return np.asarray(output, dtype=np.float32)


def _flatten_payload(weights: Weights) -> np.ndarray:
    arrays = [np.asarray(value, dtype="<f4").reshape(-1) for value in weights.arrays_in_file_order()]
    return np.concatenate(arrays).astype("<f4", copy=False)


def write_model(path: Path, weights: Weights) -> dict[str, object]:
    payload_array = _flatten_payload(weights)
    payload = payload_array.tobytes(order="C")
    crc32 = binascii.crc32(payload) & 0xFFFFFFFF
    header = HEADER.pack(
        MAGIC,
        VERSION,
        HEADER.size,
        SCALAR_FP32,
        INPUT_DIM,
        HIDDEN_DIM,
        OUTPUT_DIM,
        int(payload_array.size),
        crc32,
        0,
        0,
        0,
    )
    data = header + payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "bytes": len(data),
        "payload_float_count": int(payload_array.size),
        "payload_crc32": f"{crc32:08x}",
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def read_model(path: Path) -> Weights:
    data = path.read_bytes()
    if len(data) < HEADER.size:
        raise ValueError("model is shorter than the VGNN header")
    unpacked = HEADER.unpack_from(data)
    magic = unpacked[0]
    (
        version,
        header_size,
        scalar_type,
        input_dim,
        hidden_dim,
        output_dim,
        payload_float_count,
        payload_crc32,
        reserved0,
        reserved1,
        reserved2,
    ) = unpacked[1:]
    if magic != MAGIC:
        raise ValueError("invalid VGNN magic")
    if version != VERSION or header_size != HEADER.size or scalar_type != SCALAR_FP32:
        raise ValueError("unsupported VGNN version, header, or scalar type")
    if (input_dim, hidden_dim, output_dim) != (INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM):
        raise ValueError("VGNN dimensions do not match the fixed runtime")
    if reserved0 or reserved1 or reserved2:
        raise ValueError("VGNN reserved header fields must be zero")
    payload = data[header_size:]
    if len(payload) != payload_float_count * 4:
        raise ValueError("VGNN payload length mismatch")
    if (binascii.crc32(payload) & 0xFFFFFFFF) != payload_crc32:
        raise ValueError("VGNN payload checksum mismatch")
    floats = np.frombuffer(payload, dtype="<f4")
    cursor = 0

    def take(count: int, shape: tuple[int, ...]) -> np.ndarray:
        nonlocal cursor
        result = floats[cursor : cursor + count].reshape(shape).astype(np.float32, copy=True)
        cursor += count
        return result

    result = Weights(
        input_scale=take(INPUT_DIM, (INPUT_DIM,)),
        output_scale=take(OUTPUT_DIM, (OUTPUT_DIM,)),
        self0=take(HIDDEN_DIM * INPUT_DIM, (HIDDEN_DIM, INPUT_DIM)),
        neighbor0=take(HIDDEN_DIM * INPUT_DIM, (HIDDEN_DIM, INPUT_DIM)),
        bias0=take(HIDDEN_DIM, (HIDDEN_DIM,)),
        self1=take(OUTPUT_DIM * HIDDEN_DIM, (OUTPUT_DIM, HIDDEN_DIM)),
        neighbor1=take(OUTPUT_DIM * HIDDEN_DIM, (OUTPUT_DIM, HIDDEN_DIM)),
        bias1=take(OUTPUT_DIM, (OUTPUT_DIM,)),
    )
    if cursor != payload_float_count:
        raise ValueError("VGNN contains unexpected trailing floats")
    return result


def write_golden(path: Path, case: GoldenCase) -> dict[str, object]:
    arrays = (
        np.asarray(case.graph.offsets, dtype="<u4"),
        np.asarray(case.graph.neighbors, dtype="<u4"),
        np.asarray(case.rest_position, dtype="<f4").reshape(-1),
        np.asarray(case.position, dtype="<f4").reshape(-1),
        np.asarray(case.velocity, dtype="<f4").reshape(-1),
        np.asarray(case.pinned, dtype="<f4").reshape(-1),
        np.asarray(case.expected_acceleration, dtype="<f4").reshape(-1),
    )
    payload = b"".join(array.tobytes(order="C") for array in arrays)
    crc32 = binascii.crc32(payload) & 0xFFFFFFFF
    external = np.asarray(case.external_acceleration, dtype=np.float32).reshape(3)
    header = GOLDEN_HEADER.pack(
        GOLDEN_MAGIC,
        GOLDEN_VERSION,
        GOLDEN_HEADER.size,
        case.graph.grid_size,
        case.graph.vertex_count,
        int(case.graph.neighbors.size),
        int(len(payload)),
        crc32,
        0,
        0,
        *map(float, external),
    )
    data = header + payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "bytes": len(data),
        "payload_crc32": f"{crc32:08x}",
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def read_golden(path: Path) -> GoldenCase:
    data = path.read_bytes()
    if len(data) < GOLDEN_HEADER.size:
        raise ValueError("golden case is shorter than its header")
    unpacked = GOLDEN_HEADER.unpack_from(data)
    magic = unpacked[0]
    version, header_size, grid_size, vertex_count, edge_count, payload_bytes, crc32, r0, r1 = unpacked[1:10]
    external = np.asarray(unpacked[10:13], dtype=np.float32)
    if magic != GOLDEN_MAGIC or version != GOLDEN_VERSION or header_size != GOLDEN_HEADER.size:
        raise ValueError("unsupported golden case header")
    if r0 or r1 or vertex_count != grid_size * grid_size:
        raise ValueError("invalid golden case dimensions")
    payload = data[header_size:]
    if len(payload) != payload_bytes or (binascii.crc32(payload) & 0xFFFFFFFF) != crc32:
        raise ValueError("golden case length or checksum mismatch")
    cursor = 0

    def take(dtype: str, count: int, shape: tuple[int, ...]) -> np.ndarray:
        nonlocal cursor
        item_size = np.dtype(dtype).itemsize
        result = np.frombuffer(payload, dtype=dtype, count=count, offset=cursor).reshape(shape).copy()
        cursor += count * item_size
        return result

    offsets = take("<u4", vertex_count + 1, (vertex_count + 1,))
    neighbors = take("<u4", edge_count, (edge_count,))
    rest = take("<f4", vertex_count * 3, (vertex_count, 3))
    position = take("<f4", vertex_count * 3, (vertex_count, 3))
    velocity = take("<f4", vertex_count * 3, (vertex_count, 3))
    pinned = take("<f4", vertex_count, (vertex_count, 1))
    expected = take("<f4", vertex_count * 3, (vertex_count, 3))
    if cursor != len(payload):
        raise ValueError("golden case has trailing bytes")
    return GoldenCase(
        graph=Graph(grid_size, offsets, neighbors),
        rest_position=rest,
        position=position,
        velocity=velocity,
        pinned=pinned,
        external_acceleration=external,
        expected_acceleration=expected,
    )


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
