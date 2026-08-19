"""Deterministic long-range graph construction for HOOD PostCVPR.

The implementation mirrors ``utils/coarse.py`` from the official HOOD
repository without depending on NetworkX.  The selected graph centre and the
edge ordering are deterministic so Python and Vulkan consume identical CSR
graphs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .formats import Section, load_sectioned, pack_u32, write_sectioned


MAGIC = b"VPHIER01"
VERSION = 1
LEVEL_COUNT = 3


@dataclass(frozen=True)
class PostCvprHierarchy:
    center: int
    vertex_level: torch.Tensor
    senders: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    receivers: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    offsets: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    def to(self, device: torch.device | str) -> "PostCvprHierarchy":
        return PostCvprHierarchy(
            center=self.center,
            vertex_level=self.vertex_level.to(device=device),
            senders=tuple(value.to(device=device) for value in self.senders),
            receivers=tuple(value.to(device=device) for value in self.receivers),
            offsets=tuple(value.to(device=device) for value in self.offsets),
        )


def _add_edge(graph: list[set[int]], a: int, b: int) -> None:
    if a == b:
        return
    graph[a].add(b)
    graph[b].add(a)


def _graph_from_triangles(triangles: np.ndarray, vertex_count: int) -> list[set[int]]:
    graph = [set() for _ in range(vertex_count)]
    for a, b, c in np.asarray(triangles, dtype=np.int64):
        _add_edge(graph, int(a), int(b))
        _add_edge(graph, int(b), int(c))
        _add_edge(graph, int(c), int(a))
    if not vertex_count or any(not neighbors for neighbors in graph):
        raise ValueError("PostCVPR hierarchy requires one connected, non-isolated cloth mesh")
    return graph


def _distances(graph: list[set[int]], source: int) -> list[int]:
    # The official helper initializes unreachable entries to zero.  Coarser
    # graphs intentionally retain isolated fine vertices, so preserve that
    # detail instead of returning -1 for them.
    distance = [0] * len(graph)
    visited = [False] * len(graph)
    visited[source] = True
    queue: deque[int] = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in sorted(graph[node]):
            if visited[neighbor]:
                continue
            visited[neighbor] = True
            distance[neighbor] = distance[node] + 1
            queue.append(neighbor)
    return distance


def _shortest_distance(graph: list[set[int]], source: int, target: int) -> int | None:
    if source == target:
        return 0
    visited = {source}
    queue: deque[tuple[int, int]] = deque([(source, 0)])
    while queue:
        node, distance = queue.popleft()
        for neighbor in sorted(graph[node]):
            if neighbor == target:
                return distance + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
    return None


def _graph_center(graph: list[set[int]]) -> int:
    eccentricities = []
    for source in range(len(graph)):
        distance = _distances(graph, source)
        eccentricities.append(max(distance))
    minimum = min(eccentricities)
    return next(index for index, value in enumerate(eccentricities) if value == minimum)


def _subsample(graph: list[set[int]], center: int) -> list[set[int]]:
    distances = _distances(graph, center)
    level2 = [set() for _ in graph]
    for node in range(len(graph)):
        if distances[node] % 2 == 0:
            continue
        neighbors = sorted(graph[node])
        closer = [value for value in neighbors if distances[value] == distances[node] - 1]
        farther = [value for value in neighbors if distances[value] == distances[node] + 1]
        for a in closer:
            for b in farther:
                if a != b and _shortest_distance(graph, a, b) == 2:
                    _add_edge(level2, a, b)

    level3 = [set(values) for values in level2]
    for node in range(len(graph)):
        if distances[node] % 2 == 0:
            continue
        neighbors = sorted(graph[node])
        closer = [value for value in neighbors if distances[value] == distances[node] - 1]
        farther = [value for value in neighbors if distances[value] == distances[node] + 1]
        if farther:
            continue
        for a in closer:
            for b in closer:
                if a == b:
                    continue
                distance = _shortest_distance(level2, a, b)
                if distance is None or distance > 2:
                    _add_edge(level3, a, b)

    for node in range(len(graph)):
        if distances[node] % 2 != 0:
            continue
        same_parity = [value for value in sorted(graph[node]) if distances[value] % 2 == 0]
        for neighbor in same_parity:
            distance = _shortest_distance(level3, neighbor, node)
            if distance is None or distance > 2:
                _add_edge(level3, neighbor, node)
    return level3


def _undirected_edges(graph: list[set[int]]) -> np.ndarray:
    edges = [(a, b) for a, neighbors in enumerate(graph) for b in neighbors if a < b]
    edges.sort()
    return np.asarray(edges, dtype=np.uint32).reshape(-1, 2)


def _directed_csr(edges: np.ndarray, vertex_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    directed = np.concatenate((edges, edges[:, ::-1]), axis=0) if len(edges) else np.empty((0, 2), dtype=np.uint32)
    # HOOD edge_index stores sender first, receiver second.  Sorting by receiver
    # makes the aggregation order explicit and gives Vulkan a compact CSR.
    if len(directed):
        order = np.lexsort((directed[:, 0], directed[:, 1]))
        directed = directed[order]
    senders = directed[:, 0].astype(np.uint32, copy=False)
    receivers = directed[:, 1].astype(np.uint32, copy=False)
    counts = np.bincount(receivers, minlength=vertex_count)
    offsets = np.empty(vertex_count + 1, dtype=np.uint32)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return senders, receivers, offsets


def build_hierarchy(triangles: np.ndarray, vertex_count: int) -> tuple[int, np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    graph = _graph_from_triangles(triangles, vertex_count)
    center = _graph_center(graph)
    vertex_level = np.zeros(vertex_count, dtype=np.uint32)
    levels: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for level in range(LEVEL_COUNT):
        graph = _subsample(graph, center)
        edges = _undirected_edges(graph)
        if not len(edges):
            raise ValueError(f"PostCVPR hierarchy level {level} has no edges")
        vertex_level[np.unique(edges)] = level + 1
        levels.append(_directed_csr(edges, vertex_count))
    return center, vertex_level, levels


def write_hierarchy(path: Path | str, triangles: np.ndarray, vertex_count: int) -> dict:
    center, vertex_level, levels = build_hierarchy(triangles, vertex_count)
    counts = [len(level[0]) for level in levels]
    sections = [
        Section("info", 6, 4, pack_u32([vertex_count, center, *counts, LEVEL_COUNT])),
        Section("vertex_level", vertex_count, 4, vertex_level.astype("<u4", copy=False).tobytes()),
    ]
    for index, (senders, receivers, offsets) in enumerate(levels):
        sections.extend(
            (
                Section(f"c{index}_senders", len(senders), 4, senders.astype("<u4", copy=False).tobytes()),
                Section(f"c{index}_receivers", len(receivers), 4, receivers.astype("<u4", copy=False).tobytes()),
                Section(f"c{index}_offsets", len(offsets), 4, offsets.astype("<u4", copy=False).tobytes()),
            )
        )
    metadata = write_sectioned(path, MAGIC, VERSION, sections)
    metadata.update(center=center, vertex_count=vertex_count, directed_edge_counts=counts)
    return metadata


def _u32(section) -> torch.Tensor:
    return torch.frombuffer(bytearray(section.data), dtype=torch.uint32).to(torch.long)


def load_hierarchy(path: Path | str, *, vertex_count: int | None = None, device: torch.device | str = "cpu") -> PostCvprHierarchy:
    asset = load_sectioned(path, expected_magic=MAGIC, expected_version=VERSION)
    info = _u32(asset.require("info", count=6, stride=4))
    count, center, c0, c1, c2, level_count = (int(value) for value in info)
    if level_count != LEVEL_COUNT or (vertex_count is not None and count != vertex_count):
        raise ValueError("PostCVPR hierarchy dimensions do not match the cloth")
    senders = []
    receivers = []
    offsets = []
    for index, edge_count in enumerate((c0, c1, c2)):
        senders.append(_u32(asset.require(f"c{index}_senders", count=edge_count, stride=4)))
        receivers.append(_u32(asset.require(f"c{index}_receivers", count=edge_count, stride=4)))
        offsets.append(_u32(asset.require(f"c{index}_offsets", count=count + 1, stride=4)))
    result = PostCvprHierarchy(
        center=center,
        vertex_level=_u32(asset.require("vertex_level", count=count, stride=4)),
        senders=tuple(senders),
        receivers=tuple(receivers),
        offsets=tuple(offsets),
    )
    return result.to(device)
