"""Pure-PyTorch reference for the hierarchical HOOD PostCVPR checkpoint.

This implementation deliberately avoids PyG and PyTorch3D.  It follows the
official five-level encode/process/decode schedule and consumes the exact same
deterministic coarse graphs that are uploaded to Vulkan.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .fine15 import (
    COLLISION_RADIUS,
    MATERIAL_FEATURES,
    NODE_HANDLE,
    NODE_NORMAL,
    NODE_OBSTACLE,
    Fine15Output,
    Fine15Weights,
    aggregate_sum,
    vertex_normals,
)
from .postcvpr_hierarchy import PostCvprHierarchy


LATENT = 128
ARCHITECTURE = (
    (0, ("mesh_edge", "coarse_edge0")),
    (0, ("mesh_edge", "coarse_edge0")),
    (0, ("mesh_edge", "coarse_edge0")),
    (1, ("coarse_edge0", "coarse_edge1")),
    (1, ("coarse_edge0", "coarse_edge1")),
    (1, ("coarse_edge0", "coarse_edge1")),
    (2, ("coarse_edge1",)),
    (2, ("coarse_edge1",)),
    (2, ("coarse_edge1",)),
    (3, ("coarse_edge0", "coarse_edge1")),
    (3, ("coarse_edge0", "coarse_edge1")),
    (3, ("coarse_edge0", "coarse_edge1")),
    (4, ("mesh_edge", "coarse_edge0")),
    (4, ("mesh_edge", "coarse_edge0")),
    (4, ("mesh_edge", "coarse_edge0")),
)
ACTIVE_LEVEL = (0, 0, 0, 1, 1, 1, 2, 2, 2, 1, 1, 1, 0, 0, 0)


class PostCvprWeights(Fine15Weights):
    """Checkpoint/VHOOD loader with the common strict tensor representation."""


@dataclass
class PostCvprGraph:
    effective_position: torch.Tensor
    effective_previous: torch.Tensor
    pin_mask: torch.Tensor
    pin_target: torch.Tensor
    cloth_nodes: torch.Tensor
    obstacle_nodes: torch.Tensor
    edges: dict[str, torch.Tensor]
    senders: dict[str, torch.Tensor]
    receivers: dict[str, torch.Tensor]
    direct_world: torch.Tensor
    inverse_world: torch.Tensor
    world_cloth: torch.Tensor
    world_obstacle: torch.Tensor
    active_obstacle: torch.Tensor
    vertex_level: torch.Tensor


class PostCvpr:
    def __init__(self, weights: PostCvprWeights):
        self.weights = weights
        self.device = weights.device
        self.type_embedding = weights.require("model.nodetype_embedding.weight", (9, 9))
        self.level_embedding = weights.require("model.vertexlevel_embedding.weight", (4, 4))

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
        vertex_level: torch.Tensor,
        normals: torch.Tensor,
        timestep: float,
        log_mass: torch.Tensor,
        material: torch.Tensor,
    ) -> torch.Tensor:
        ts = torch.full((velocity.shape[0], 1), timestep, dtype=torch.float32, device=self.device)
        return torch.cat(
            (
                velocity,
                self.type_embedding[node_type],
                self.level_embedding[vertex_level],
                normals,
                ts,
                log_mass,
                material,
            ),
            dim=-1,
        )

    def _cloth_edge_features(
        self,
        position: torch.Tensor,
        rest_position: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        timestep: float,
    ) -> torch.Tensor:
        relative = position[senders] - position[receivers]
        relative_rest = rest_position[senders] - rest_position[receivers]
        raw = torch.cat(
            (
                relative,
                torch.linalg.vector_norm(relative, dim=-1, keepdim=True),
                relative_rest,
                torch.linalg.vector_norm(relative_rest, dim=-1, keepdim=True),
                torch.full((len(relative), 1), timestep, dtype=torch.float32, device=self.device),
            ),
            dim=-1,
        )
        material = torch.tensor(MATERIAL_FEATURES, dtype=torch.float32, device=self.device).reshape(1, 3).expand(len(raw), 3)
        return torch.cat((self.weights.normalize("mesh_edge", raw), material), dim=-1)

    def prepare_graph(
        self,
        *,
        position: torch.Tensor,
        previous: torch.Tensor,
        rest_position: torch.Tensor,
        triangles: torch.Tensor,
        mesh_senders: torch.Tensor,
        mesh_receivers: torch.Tensor,
        hierarchy: PostCvprHierarchy,
        mass: torch.Tensor,
        pin_mask: torch.Tensor,
        pin_target: torch.Tensor,
        obstacle_position: torch.Tensor,
        obstacle_target: torch.Tensor,
        obstacle_normals: torch.Tensor,
        timestep: float,
    ) -> PostCvprGraph:
        w = self.weights
        position = position.to(self.device, dtype=torch.float32)
        previous = previous.to(self.device, dtype=torch.float32)
        rest_position = rest_position.to(self.device, dtype=torch.float32)
        triangles = triangles.to(self.device, dtype=torch.long)
        mesh_senders = mesh_senders.to(self.device, dtype=torch.long)
        mesh_receivers = mesh_receivers.to(self.device, dtype=torch.long)
        hierarchy = hierarchy.to(self.device)
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

        world_cloth, world_obstacle = self._world_edges(effective_position, obstacle_position)
        active_obstacle = torch.zeros(len(obstacle_position), dtype=torch.bool, device=self.device)
        active_obstacle[world_obstacle] = True

        cloth_type = torch.where(pin_mask, NODE_HANDLE, NODE_NORMAL).long()
        obstacle_type = torch.full((len(obstacle_position),), NODE_OBSTACLE, dtype=torch.long, device=self.device)
        cloth_material = torch.tensor(MATERIAL_FEATURES, dtype=torch.float32, device=self.device).reshape(1, 3).expand(len(position), 3)
        obstacle_material = torch.full((len(obstacle_position), 3), -1.0, dtype=torch.float32, device=self.device)
        cloth_nodes = self._node_features(
            effective_position - effective_previous,
            cloth_type,
            hierarchy.vertex_level,
            vertex_normals(effective_position, triangles),
            timestep,
            torch.log(mass),
            cloth_material,
        )
        obstacle_nodes = self._node_features(
            obstacle_target - obstacle_position,
            obstacle_type,
            torch.zeros(len(obstacle_position), dtype=torch.long, device=self.device),
            obstacle_normals,
            timestep,
            torch.full((len(obstacle_position), 1), -1.0, dtype=torch.float32, device=self.device),
            obstacle_material,
        )
        cloth_nodes = torch.cat((w.normalize("node", cloth_nodes[:, :-3]), cloth_nodes[:, -3:]), dim=-1)
        obstacle_nodes = torch.cat((w.normalize("node", obstacle_nodes[:, :-3]), obstacle_nodes[:, -3:]), dim=-1)

        senders = {"mesh_edge": mesh_senders}
        receivers = {"mesh_edge": mesh_receivers}
        for level in range(3):
            senders[f"coarse_edge{level}"] = hierarchy.senders[level]
            receivers[f"coarse_edge{level}"] = hierarchy.receivers[level]
        edges = {
            name: self._cloth_edge_features(effective_position, rest_position, senders[name], receivers[name], timestep)
            for name in senders
        }

        cloth_edge_position = effective_position[world_cloth]
        relative_current = cloth_edge_position - obstacle_position[world_obstacle]
        relative_target = cloth_edge_position - obstacle_target[world_obstacle]
        ts_world = torch.full((len(world_cloth), 1), timestep, dtype=torch.float32, device=self.device)
        direct_world = torch.cat(
            (relative_current, torch.linalg.vector_norm(relative_current, dim=-1, keepdim=True),
             relative_target, torch.linalg.vector_norm(relative_target, dim=-1, keepdim=True), ts_world), dim=-1
        )
        inverse_world = torch.cat(
            (-relative_current, torch.linalg.vector_norm(relative_current, dim=-1, keepdim=True),
             -relative_target, torch.linalg.vector_norm(relative_target, dim=-1, keepdim=True), ts_world), dim=-1
        )
        return PostCvprGraph(
            effective_position=effective_position,
            effective_previous=effective_previous,
            pin_mask=pin_mask,
            pin_target=pin_target,
            cloth_nodes=cloth_nodes,
            obstacle_nodes=obstacle_nodes,
            edges=edges,
            senders=senders,
            receivers=receivers,
            direct_world=w.normalize("world_edge", direct_world),
            inverse_world=w.normalize("world_edge", inverse_world),
            world_cloth=world_cloth,
            world_obstacle=world_obstacle,
            active_obstacle=active_obstacle,
            vertex_level=hierarchy.vertex_level,
        )

    def predict_graph(self, graph: PostCvprGraph, trace: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        w = self.weights
        cloth_latent = w.mlp("model._learned_model.node_encoder", graph.cloth_nodes)
        obstacle_latent = torch.zeros((len(graph.obstacle_nodes), LATENT), dtype=torch.float32, device=self.device)
        if graph.active_obstacle.any():
            obstacle_latent[graph.active_obstacle] = w.mlp(
                "model._learned_model.node_encoder", graph.obstacle_nodes[graph.active_obstacle]
            )
        edge_latent = {
            "mesh_edge": w.mlp("model._learned_model.edgeset_encoders.mesh", graph.edges["mesh_edge"]),
            "coarse_edge0": w.mlp("model._learned_model.edgeset_encoders.coarse0", graph.edges["coarse_edge0"]),
            "coarse_edge1": w.mlp("model._learned_model.edgeset_encoders.coarse1", graph.edges["coarse_edge1"]),
            "coarse_edge2": w.mlp("model._learned_model.edgeset_encoders.coarse2", graph.edges["coarse_edge2"]),
        }
        combined_world = w.mlp(
            "model._learned_model.edgeset_encoders.world", torch.cat((graph.direct_world, graph.inverse_world), dim=0)
        )
        if len(graph.world_cloth):
            direct_latent, inverse_latent = combined_world.split(len(graph.world_cloth), dim=0)
        else:
            direct_latent = combined_world
            inverse_latent = combined_world

        for block, (level_index, edge_names) in enumerate(ARCHITECTURE):
            step_index = block % 3
            prefix = f"model._learned_model.levels.{level_index}.{step_index}"
            edge_updates: dict[str, torch.Tensor] = {}
            for edge_name in edge_names:
                edge_updates[edge_name] = w.mlp(
                    f"{prefix}.edge_processor_dict.{edge_name}",
                    torch.cat(
                        (
                            cloth_latent[graph.receivers[edge_name]],
                            cloth_latent[graph.senders[edge_name]],
                            edge_latent[edge_name],
                        ),
                        dim=-1,
                    ),
                )

            active_world = graph.vertex_level[graph.world_cloth] >= ACTIVE_LEVEL[block]
            direct_update = torch.zeros_like(direct_latent)
            inverse_update = torch.zeros_like(inverse_latent)
            if active_world.any():
                cloth_ids = graph.world_cloth[active_world]
                obstacle_ids = graph.world_obstacle[active_world]
                direct_update[active_world] = w.mlp(
                    f"{prefix}.edge_processor_dict.world_edge",
                    torch.cat((obstacle_latent[obstacle_ids], cloth_latent[cloth_ids], direct_latent[active_world]), dim=-1),
                )
                inverse_update[active_world] = w.mlp(
                    f"{prefix}.edge_processor_dict.world_edge",
                    torch.cat((cloth_latent[cloth_ids], obstacle_latent[obstacle_ids], inverse_latent[active_world]), dim=-1),
                )

            cloth_aggregates: dict[str, torch.Tensor] = {}
            for edge_name, update in edge_updates.items():
                cloth_aggregates[edge_name] = aggregate_sum(update, graph.receivers[edge_name], len(cloth_latent))
            cloth_aggregates["world_edge"] = aggregate_sum(
                inverse_update[active_world], graph.world_cloth[active_world], len(cloth_latent)
            )
            obstacle_world = aggregate_sum(
                direct_update[active_world], graph.world_obstacle[active_world], len(obstacle_latent)
            )
            edge_keys = sorted((*edge_names, "world_edge"))
            cloth_input = torch.cat((cloth_latent, *(cloth_aggregates[name] for name in edge_keys)), dim=-1)
            obstacle_parts = [obstacle_latent]
            for name in edge_keys:
                obstacle_parts.append(obstacle_world if name == "world_edge" else torch.zeros_like(obstacle_world))
            cloth_update = w.mlp(f"{prefix}.node_processor_dict.node", cloth_input)
            obstacle_update = w.mlp(f"{prefix}.node_processor_dict.node", torch.cat(obstacle_parts, dim=-1))
            cloth_latent = cloth_latent + cloth_update
            obstacle_latent[graph.active_obstacle] += obstacle_update[graph.active_obstacle]
            for edge_name, update in edge_updates.items():
                edge_latent[edge_name] = edge_latent[edge_name] + update
            direct_latent = direct_latent + direct_update
            inverse_latent = inverse_latent + inverse_update

        if trace is not None:
            trace["cloth_node_latent"] = cloth_latent.detach()
        return w.mlp("model._learned_model.decoder", cloth_latent, layer_norm=False)

    def step(
        self,
        *,
        position: torch.Tensor,
        previous: torch.Tensor,
        rest_position: torch.Tensor,
        triangles: torch.Tensor,
        mesh_senders: torch.Tensor,
        mesh_receivers: torch.Tensor,
        hierarchy: PostCvprHierarchy,
        mass: torch.Tensor,
        pin_mask: torch.Tensor,
        pin_target: torch.Tensor,
        obstacle_position: torch.Tensor,
        obstacle_target: torch.Tensor,
        obstacle_normals: torch.Tensor,
        timestep: float,
        trace: dict[str, torch.Tensor] | None = None,
    ) -> Fine15Output:
        graph = self.prepare_graph(
            position=position,
            previous=previous,
            rest_position=rest_position,
            triangles=triangles,
            mesh_senders=mesh_senders,
            mesh_receivers=mesh_receivers,
            hierarchy=hierarchy,
            mass=mass,
            pin_mask=pin_mask,
            pin_target=pin_target,
            obstacle_position=obstacle_position,
            obstacle_target=obstacle_target,
            obstacle_normals=obstacle_normals,
            timestep=timestep,
        )
        if trace is not None:
            trace.update(
                effective_position=graph.effective_position.detach(),
                cloth_node_features=graph.cloth_nodes.detach(),
                obstacle_node_features=graph.obstacle_nodes.detach(),
                mesh_features=graph.edges["mesh_edge"].detach(),
                coarse0_features=graph.edges["coarse_edge0"].detach(),
                coarse1_features=graph.edges["coarse_edge1"].detach(),
                world_direct_features=graph.direct_world.detach(),
                world_inverse_features=graph.inverse_world.detach(),
                world_cloth=graph.world_cloth.detach(),
            )
        normalized_acceleration = self.predict_graph(graph, trace)
        acceleration = self.weights.inverse("output", normalized_acceleration)
        velocity = graph.effective_position - graph.effective_previous + acceleration
        predicted = graph.effective_position + velocity
        velocity[graph.pin_mask] = graph.pin_target[graph.pin_mask] - graph.effective_position[graph.pin_mask]
        predicted[graph.pin_mask] = graph.pin_target[graph.pin_mask]
        if trace is not None:
            trace["acceleration"] = acceleration.detach()
        return Fine15Output(
            position=predicted,
            velocity=velocity,
            acceleration=acceleration,
            effective_position=graph.effective_position,
            effective_previous=graph.effective_previous,
            world_cloth=graph.world_cloth,
            world_obstacle=graph.world_obstacle,
        )
