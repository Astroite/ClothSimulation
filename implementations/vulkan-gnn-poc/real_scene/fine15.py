"""Dependency-light, pure-PyTorch implementation of HOOD's Fine15 baseline.

This module intentionally does not import PyG or PyTorch3D. It mirrors the
official baseline's feature construction, add aggregation, residual updates,
normalizers, MLP layout, and Verlet-style integration using ordinary PyTorch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from .formats import TensorAsset, load_tensor_asset


NODE_NORMAL = 0
NODE_OBSTACLE = 1
NODE_HANDLE = 3
COLLISION_RADIUS = 0.03
LATENT = 128


@dataclass
class Fine15Output:
    position: torch.Tensor
    velocity: torch.Tensor
    acceleration: torch.Tensor
    effective_position: torch.Tensor
    effective_previous: torch.Tensor
    world_cloth: torch.Tensor
    world_obstacle: torch.Tensor


@dataclass
class Fine15Graph:
    effective_position: torch.Tensor
    effective_previous: torch.Tensor
    pin_mask: torch.Tensor
    pin_target: torch.Tensor
    cloth_nodes: torch.Tensor
    obstacle_nodes: torch.Tensor
    mesh_edges: torch.Tensor
    direct_world: torch.Tensor
    inverse_world: torch.Tensor
    mesh_senders: torch.Tensor
    mesh_receivers: torch.Tensor
    world_cloth: torch.Tensor
    world_obstacle: torch.Tensor
    active_obstacle: torch.Tensor


class Fine15Weights:
    def __init__(self, tensors: Mapping[str, torch.Tensor], device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        self.tensors = {name: value.detach().to(device=self.device, dtype=torch.float32).contiguous() for name, value in tensors.items()}

    @classmethod
    def from_checkpoint(cls, path: Path | str, device: torch.device | str = "cpu") -> "Fine15Weights":
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint.get("training_module") if isinstance(checkpoint, dict) else None
        if not isinstance(state, dict) or not all(isinstance(value, torch.Tensor) for value in state.values()):
            raise ValueError("checkpoint does not contain a tensor-only training_module state")
        return cls(state, device)

    @classmethod
    def from_vhood(cls, path: Path | str, device: torch.device | str = "cpu") -> "Fine15Weights":
        asset: TensorAsset = load_tensor_asset(path)
        tensors = {}
        for name, view in asset.tensors.items():
            tensors[name] = torch.frombuffer(bytearray(view.data), dtype=torch.float32).reshape(view.shape).clone()
        return cls(tensors, device)

    def require(self, name: str, shape: Sequence[int] | None = None) -> torch.Tensor:
        if name not in self.tensors:
            raise KeyError(f"Fine15 tensor is missing: {name}")
        tensor = self.tensors[name]
        if shape is not None and tuple(tensor.shape) != tuple(shape):
            raise ValueError(f"Fine15 tensor {name} is {tuple(tensor.shape)}, expected {tuple(shape)}")
        return tensor

    def normalizer(self, label: str) -> tuple[torch.Tensor, torch.Tensor]:
        prefix = f"model._{label}_normalizer"
        count = torch.clamp(self.require(f"{prefix}._acc_count"), min=1.0)
        mean = self.require(f"{prefix}._acc_sum") / count
        variance = torch.clamp(self.require(f"{prefix}._acc_sum_squared") / count - mean.square(), min=0.0)
        std = torch.clamp(torch.sqrt(variance), min=1.0e-8)
        return mean, std

    def normalize(self, label: str, value: torch.Tensor) -> torch.Tensor:
        mean, std = self.normalizer(label)
        return (value - mean) / std

    def inverse(self, label: str, value: torch.Tensor) -> torch.Tensor:
        mean, std = self.normalizer(label)
        return value * std + mean

    def mlp(self, prefix: str, value: torch.Tensor, *, layer_norm: bool = True) -> torch.Tensor:
        if f"{prefix}.0.layers.0.weight" in self.tensors:
            network = f"{prefix}.0"
            norm = f"{prefix}.1"
        else:
            network = prefix
            norm = None
        for layer in (0, 2, 4):
            value = F.linear(
                value,
                self.require(f"{network}.layers.{layer}.weight"),
                self.require(f"{network}.layers.{layer}.bias"),
            )
            if layer != 4:
                value = F.relu(value)
        if layer_norm:
            if norm is None:
                raise ValueError(f"MLP {prefix} has no LayerNorm")
            value = F.layer_norm(
                value,
                (value.shape[-1],),
                self.require(f"{norm}.weight"),
                self.require(f"{norm}.bias"),
                eps=1.0e-5,
            )
        return value


def vertex_normals(positions: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    vertices = positions[triangles]
    v0, v1, v2 = vertices.unbind(1)
    e0, e1, e2 = v1 - v0, v2 - v1, v0 - v2
    face = torch.linalg.cross(e0, e1) + torch.linalg.cross(e1, e2) + torch.linalg.cross(e2, e0)
    result = torch.zeros_like(positions)
    for corner in range(3):
        result.index_add_(0, triangles[:, corner], face)
    return F.normalize(result, dim=-1)


def aggregate_sum(values: torch.Tensor, receivers: torch.Tensor, count: int) -> torch.Tensor:
    output = torch.zeros((count, values.shape[-1]), dtype=values.dtype, device=values.device)
    output.index_add_(0, receivers, values)
    return output


def _relative_between(value: float, minimum: float, maximum: float) -> float:
    return (value - minimum) / (maximum - minimum)


def _relative_between_log(value: float, minimum: float, maximum: float) -> float:
    return (math.log(value) - math.log(minimum)) / (math.log(maximum) - math.log(minimum))


MATERIAL_FEATURES = (
    _relative_between_log(3.9625778333333325e-5, 6.370782056371576e-8, 0.0013139737991266374),
    _relative_between_log(23600.0, 15909.0, 63636.0),
    _relative_between(44400.0, 3535.414406069427, 93333.73508005822),
)


class Fine15:
    def __init__(self, weights: Fine15Weights):
        self.weights = weights
        self.device = weights.device
        self.embedding = weights.require("model.nodetype_embedding.weight", (9, 9))

    def _world_edges(self, cloth: torch.Tensor, obstacle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distances = torch.cdist(cloth, obstacle)
        minimum, receiver = distances.min(dim=1)
        sender = torch.arange(cloth.shape[0], device=self.device, dtype=torch.long)
        valid = minimum < COLLISION_RADIUS
        return sender[valid], receiver[valid]

    def _node_features(
        self,
        velocity: torch.Tensor,
        node_type: torch.Tensor,
        normals: torch.Tensor,
        timestep: float,
        log_mass: torch.Tensor,
        material: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.embedding[node_type]
        ts = torch.full((velocity.shape[0], 1), timestep, dtype=torch.float32, device=self.device)
        return torch.cat((velocity, embedding, normals, ts, log_mass, material), dim=-1)

    def prepare_graph(
        self,
        *,
        position: torch.Tensor,
        previous: torch.Tensor,
        rest_position: torch.Tensor,
        triangles: torch.Tensor,
        mesh_senders: torch.Tensor,
        mesh_receivers: torch.Tensor,
        mass: torch.Tensor,
        pin_mask: torch.Tensor,
        pin_target: torch.Tensor,
        obstacle_position: torch.Tensor,
        obstacle_target: torch.Tensor,
        obstacle_normals: torch.Tensor,
        timestep: float,
    ) -> Fine15Graph:
        w = self.weights
        position = position.to(self.device, dtype=torch.float32)
        previous = previous.to(self.device, dtype=torch.float32)
        rest_position = rest_position.to(self.device, dtype=torch.float32)
        triangles = triangles.to(self.device, dtype=torch.long)
        mesh_senders = mesh_senders.to(self.device, dtype=torch.long)
        mesh_receivers = mesh_receivers.to(self.device, dtype=torch.long)
        mass = mass.to(self.device, dtype=torch.float32).reshape(-1, 1)
        pin_mask = pin_mask.to(self.device, dtype=torch.bool).reshape(-1)
        pin_target = pin_target.to(self.device, dtype=torch.float32)
        obstacle_position = obstacle_position.to(self.device, dtype=torch.float32)
        obstacle_target = obstacle_target.to(self.device, dtype=torch.float32)
        obstacle_normals = F.normalize(obstacle_normals.to(self.device, dtype=torch.float32), dim=-1)

        effective_previous = previous.clone()
        effective_position = position.clone()
        effective_previous[pin_mask] = position[pin_mask]
        effective_position[pin_mask] = pin_target[pin_mask]

        world_cloth, world_obstacle_full = self._world_edges(effective_position, obstacle_position)
        active_obstacle, world_obstacle = torch.unique(world_obstacle_full, sorted=True, return_inverse=True)
        active_position = obstacle_position[active_obstacle]
        active_target = obstacle_target[active_obstacle]
        active_normals = obstacle_normals[active_obstacle]

        cloth_velocity = effective_position - effective_previous
        obstacle_velocity = active_target - active_position
        cloth_types = torch.where(pin_mask, NODE_HANDLE, NODE_NORMAL).long()
        obstacle_types = torch.full((len(active_obstacle),), NODE_OBSTACLE, dtype=torch.long, device=self.device)
        cloth_material = torch.tensor(MATERIAL_FEATURES, dtype=torch.float32, device=self.device).reshape(1, 3).expand(len(position), 3)
        obstacle_material = torch.full((len(active_obstacle), 3), -1.0, dtype=torch.float32, device=self.device)
        cloth_nodes = self._node_features(
            cloth_velocity,
            cloth_types,
            vertex_normals(effective_position, triangles),
            timestep,
            torch.log(mass),
            cloth_material,
        )
        obstacle_nodes = self._node_features(
            obstacle_velocity,
            obstacle_types,
            active_normals,
            timestep,
            torch.full((len(active_obstacle), 1), -1.0, dtype=torch.float32, device=self.device),
            obstacle_material,
        )
        cloth_nodes = torch.cat((w.normalize("node", cloth_nodes[:, :-3]), cloth_nodes[:, -3:]), dim=-1)
        obstacle_nodes = torch.cat((w.normalize("node", obstacle_nodes[:, :-3]), obstacle_nodes[:, -3:]), dim=-1)

        relative = effective_position[mesh_senders] - effective_position[mesh_receivers]
        relative_rest = rest_position[mesh_senders] - rest_position[mesh_receivers]
        mesh_to_normalize = torch.cat(
            (
                relative,
                torch.linalg.vector_norm(relative, dim=-1, keepdim=True),
                relative_rest,
                torch.linalg.vector_norm(relative_rest, dim=-1, keepdim=True),
                torch.full((len(relative), 1), timestep, dtype=torch.float32, device=self.device),
            ),
            dim=-1,
        )
        mesh_material = torch.tensor(MATERIAL_FEATURES, dtype=torch.float32, device=self.device).reshape(1, 3).expand(len(relative), 3)
        mesh_edges = torch.cat((w.normalize("mesh_edge", mesh_to_normalize), mesh_material), dim=-1)

        cloth_edge_pos = effective_position[world_cloth]
        obstacle_edge_current = active_position[world_obstacle]
        obstacle_edge_target = active_target[world_obstacle]
        relative_current = cloth_edge_pos - obstacle_edge_current
        relative_target = cloth_edge_pos - obstacle_edge_target
        ts_world = torch.full((len(world_cloth), 1), timestep, dtype=torch.float32, device=self.device)
        direct_world = torch.cat(
            (
                relative_current,
                torch.linalg.vector_norm(relative_current, dim=-1, keepdim=True),
                relative_target,
                torch.linalg.vector_norm(relative_target, dim=-1, keepdim=True),
                ts_world,
            ),
            dim=-1,
        )
        inverse_world = torch.cat(
            (
                -relative_current,
                torch.linalg.vector_norm(relative_current, dim=-1, keepdim=True),
                -relative_target,
                torch.linalg.vector_norm(relative_target, dim=-1, keepdim=True),
                ts_world,
            ),
            dim=-1,
        )
        return Fine15Graph(
            effective_position=effective_position,
            effective_previous=effective_previous,
            pin_mask=pin_mask,
            pin_target=pin_target,
            cloth_nodes=cloth_nodes,
            obstacle_nodes=obstacle_nodes,
            mesh_edges=mesh_edges,
            direct_world=w.normalize("world_edge", direct_world),
            inverse_world=w.normalize("world_edge", inverse_world),
            mesh_senders=mesh_senders,
            mesh_receivers=mesh_receivers,
            world_cloth=world_cloth,
            world_obstacle=world_obstacle,
            active_obstacle=active_obstacle,
        )

    def predict_graph(
        self,
        graph: Fine15Graph,
        trace: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Return normalized acceleration for an already constructed graph."""

        w = self.weights
        cloth_latent = w.mlp("model._learned_model.node_encoder", graph.cloth_nodes)
        obstacle_latent = w.mlp("model._learned_model.node_encoder", graph.obstacle_nodes)
        mesh_latent = w.mlp("model._learned_model.edgeset_encoders.mesh", graph.mesh_edges)
        combined_world = w.mlp(
            "model._learned_model.edgeset_encoders.world",
            torch.cat((graph.direct_world, graph.inverse_world), dim=0),
        )
        if len(graph.world_cloth):
            direct_latent, inverse_latent = combined_world.split(len(graph.world_cloth), dim=0)
        else:
            direct_latent = combined_world
            inverse_latent = combined_world

        for block in range(15):
            prefix = f"model._learned_model.processor_steps.{block}"
            mesh_update = w.mlp(
                f"{prefix}.mesh_edge_processor",
                torch.cat(
                    (
                        cloth_latent[graph.mesh_receivers],
                        cloth_latent[graph.mesh_senders],
                        mesh_latent,
                    ),
                    dim=-1,
                ),
            )
            direct_update = w.mlp(
                f"{prefix}.world_edge_processor",
                torch.cat(
                    (
                        obstacle_latent[graph.world_obstacle],
                        cloth_latent[graph.world_cloth],
                        direct_latent,
                    ),
                    dim=-1,
                ),
            )
            inverse_update = w.mlp(
                f"{prefix}.world_edge_processor",
                torch.cat(
                    (
                        cloth_latent[graph.world_cloth],
                        obstacle_latent[graph.world_obstacle],
                        inverse_latent,
                    ),
                    dim=-1,
                ),
            )
            mesh_aggregate = aggregate_sum(mesh_update, graph.mesh_receivers, len(graph.cloth_nodes))
            cloth_world_aggregate = aggregate_sum(inverse_update, graph.world_cloth, len(graph.cloth_nodes))
            obstacle_world_aggregate = aggregate_sum(
                direct_update, graph.world_obstacle, len(graph.obstacle_nodes)
            )
            cloth_update = w.mlp(
                f"{prefix}.node_processor",
                torch.cat((mesh_aggregate, cloth_world_aggregate, cloth_latent), dim=-1),
            )
            obstacle_update = w.mlp(
                f"{prefix}.node_processor",
                torch.cat(
                    (
                        torch.zeros_like(obstacle_world_aggregate),
                        obstacle_world_aggregate,
                        obstacle_latent,
                    ),
                    dim=-1,
                ),
            )
            cloth_latent = cloth_latent + cloth_update
            obstacle_latent = obstacle_latent + obstacle_update
            mesh_latent = mesh_latent + mesh_update
            direct_latent = direct_latent + direct_update
            inverse_latent = inverse_latent + inverse_update

        normalized_acceleration = w.mlp("model._learned_model.decoder", cloth_latent, layer_norm=False)
        if trace is not None:
            trace["cloth_node_latent"] = cloth_latent.detach()
        return normalized_acceleration

    def step(
        self,
        *,
        position: torch.Tensor,
        previous: torch.Tensor,
        rest_position: torch.Tensor,
        triangles: torch.Tensor,
        mesh_senders: torch.Tensor,
        mesh_receivers: torch.Tensor,
        mass: torch.Tensor,
        pin_mask: torch.Tensor,
        pin_target: torch.Tensor,
        obstacle_position: torch.Tensor,
        obstacle_target: torch.Tensor,
        obstacle_normals: torch.Tensor,
        timestep: float,
        trace: dict[str, torch.Tensor] | None = None,
    ) -> Fine15Output:
        w = self.weights
        graph = self.prepare_graph(
            position=position,
            previous=previous,
            rest_position=rest_position,
            triangles=triangles,
            mesh_senders=mesh_senders,
            mesh_receivers=mesh_receivers,
            mass=mass,
            pin_mask=pin_mask,
            pin_target=pin_target,
            obstacle_position=obstacle_position,
            obstacle_target=obstacle_target,
            obstacle_normals=obstacle_normals,
            timestep=timestep,
        )
        effective_position, effective_previous = graph.effective_position, graph.effective_previous
        pin_mask, pin_target = graph.pin_mask, graph.pin_target
        cloth_nodes, obstacle_nodes = graph.cloth_nodes, graph.obstacle_nodes
        mesh_edges, direct_world, inverse_world = graph.mesh_edges, graph.direct_world, graph.inverse_world
        mesh_senders, mesh_receivers = graph.mesh_senders, graph.mesh_receivers
        world_cloth, world_obstacle, active_obstacle = graph.world_cloth, graph.world_obstacle, graph.active_obstacle

        if trace is not None:
            trace.update(
                effective_position=effective_position.detach(),
                cloth_node_features=cloth_nodes.detach(),
                obstacle_node_features=obstacle_nodes.detach(),
                mesh_features=mesh_edges.detach(),
                world_direct_features=direct_world.detach(),
                world_inverse_features=inverse_world.detach(),
                world_cloth=world_cloth.detach(),
                active_obstacle=active_obstacle.detach(),
            )

        normalized_acceleration = self.predict_graph(graph, trace)
        acceleration = w.inverse("output", normalized_acceleration)
        if trace is not None:
            trace["acceleration"] = acceleration.detach()
        velocity = effective_position - effective_previous + acceleration
        predicted = effective_position + velocity
        velocity[pin_mask] = pin_target[pin_mask] - effective_position[pin_mask]
        predicted[pin_mask] = pin_target[pin_mask]
        return Fine15Output(
            position=predicted,
            velocity=velocity,
            acceleration=acceleration,
            effective_position=effective_position,
            effective_previous=effective_previous,
            world_cloth=world_cloth,
            world_obstacle=active_obstacle[world_obstacle],
        )
