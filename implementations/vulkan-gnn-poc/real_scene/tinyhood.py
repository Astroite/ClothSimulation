"""Trainable 64-channel, four-block HOOD-compatible student network."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn

from .fine15 import Fine15Graph, Fine15Weights, aggregate_sum
from .formats import TensorAsset, load_tensor_asset, write_tensor_asset


LATENT = 64
PROCESSOR_BLOCKS = 4


class TinyMlp(nn.Module):
    def __init__(self, input_dimension: int, output_dimension: int, *, layer_norm: bool = True):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dimension, LATENT),
            nn.ReLU(),
            nn.Linear(LATENT, LATENT),
            nn.ReLU(),
            nn.Linear(LATENT, output_dimension),
        )
        self.norm = nn.LayerNorm(output_dimension, eps=1.0e-5) if layer_norm else None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.layers(value)
        return self.norm(value) if self.norm is not None else value


class TinyProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mesh_edge_processor = TinyMlp(LATENT * 3, LATENT)
        self.world_edge_processor = TinyMlp(LATENT * 3, LATENT)
        self.node_processor = TinyMlp(LATENT * 3, LATENT)


class TinyHood(nn.Module):
    """HOOD input contract with a 64-wide, four-step processor."""

    def __init__(self):
        super().__init__()
        self.node_encoder = TinyMlp(20, LATENT)
        self.mesh_encoder = TinyMlp(12, LATENT)
        self.world_encoder = TinyMlp(9, LATENT)
        self.processor_steps = nn.ModuleList(TinyProcessor() for _ in range(PROCESSOR_BLOCKS))
        self.decoder = TinyMlp(LATENT, 3, layer_norm=False)

    def forward(self, graph: Fine15Graph) -> torch.Tensor:
        cloth_count = graph.cloth_nodes.shape[0]
        obstacle_count = graph.obstacle_nodes.shape[0]
        cloth_latent = self.node_encoder(graph.cloth_nodes)
        obstacle_latent = self.node_encoder(graph.obstacle_nodes)
        mesh_latent = self.mesh_encoder(graph.mesh_edges)
        direct_latent = self.world_encoder(graph.direct_world)
        inverse_latent = self.world_encoder(graph.inverse_world)

        for block in self.processor_steps:
            mesh_update = block.mesh_edge_processor(
                torch.cat(
                    (
                        cloth_latent[graph.mesh_receivers],
                        cloth_latent[graph.mesh_senders],
                        mesh_latent,
                    ),
                    dim=-1,
                )
            )
            direct_update = block.world_edge_processor(
                torch.cat(
                    (
                        obstacle_latent[graph.world_obstacle],
                        cloth_latent[graph.world_cloth],
                        direct_latent,
                    ),
                    dim=-1,
                )
            )
            inverse_update = block.world_edge_processor(
                torch.cat(
                    (
                        cloth_latent[graph.world_cloth],
                        obstacle_latent[graph.world_obstacle],
                        inverse_latent,
                    ),
                    dim=-1,
                )
            )
            mesh_aggregate = aggregate_sum(mesh_update, graph.mesh_receivers, cloth_count)
            cloth_world_aggregate = aggregate_sum(inverse_update, graph.world_cloth, cloth_count)
            obstacle_world_aggregate = aggregate_sum(direct_update, graph.world_obstacle, obstacle_count)
            cloth_update = block.node_processor(
                torch.cat((mesh_aggregate, cloth_world_aggregate, cloth_latent), dim=-1)
            )
            obstacle_update = block.node_processor(
                torch.cat(
                    (
                        torch.zeros_like(obstacle_world_aggregate),
                        obstacle_world_aggregate,
                        obstacle_latent,
                    ),
                    dim=-1,
                )
            )
            cloth_latent = cloth_latent + cloth_update
            obstacle_latent = obstacle_latent + obstacle_update
            mesh_latent = mesh_latent + mesh_update
            direct_latent = direct_latent + direct_update
            inverse_latent = inverse_latent + inverse_update
        return self.decoder(cloth_latent)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def _tensor_bytes(value: torch.Tensor) -> tuple[tuple[int, ...], bytes]:
    array = value.detach().cpu().numpy().astype("<f4", copy=False)
    return tuple(array.shape), array.tobytes()


def _export_mlp(tensors: dict[str, tuple[tuple[int, ...], bytes]], prefix: str, mlp: TinyMlp) -> None:
    network = prefix if mlp.norm is None else f"{prefix}.0"
    for layer in (0, 2, 4):
        linear = mlp.layers[layer]
        assert isinstance(linear, nn.Linear)
        tensors[f"{network}.layers.{layer}.weight"] = _tensor_bytes(linear.weight)
        tensors[f"{network}.layers.{layer}.bias"] = _tensor_bytes(linear.bias)
    if mlp.norm is not None:
        tensors[f"{prefix}.1.weight"] = _tensor_bytes(mlp.norm.weight)
        tensors[f"{prefix}.1.bias"] = _tensor_bytes(mlp.norm.bias)


def export_tinyhood(
    model: TinyHood,
    teacher: Fine15Weights,
    output: Path | str,
    *,
    checkpoint_sha256: str,
) -> dict:
    tensors: dict[str, tuple[tuple[int, ...], bytes]] = {}
    _export_mlp(tensors, "model._learned_model.node_encoder", model.node_encoder)
    _export_mlp(tensors, "model._learned_model.edgeset_encoders.mesh", model.mesh_encoder)
    _export_mlp(tensors, "model._learned_model.edgeset_encoders.world", model.world_encoder)
    for index, block in enumerate(model.processor_steps):
        base = f"model._learned_model.processor_steps.{index}"
        _export_mlp(tensors, f"{base}.mesh_edge_processor", block.mesh_edge_processor)
        _export_mlp(tensors, f"{base}.world_edge_processor", block.world_edge_processor)
        _export_mlp(tensors, f"{base}.node_processor", block.node_processor)
    _export_mlp(tensors, "model._learned_model.decoder", model.decoder)
    tensors["model.nodetype_embedding.weight"] = _tensor_bytes(
        teacher.require("model.nodetype_embedding.weight", (9, 9))
    )
    for label in ("node", "mesh_edge", "world_edge", "output"):
        prefix = f"model._{label}_normalizer"
        for suffix in ("_acc_count", "_acc_sum", "_acc_sum_squared"):
            value = teacher.require(f"{prefix}.{suffix}")
            tensors[f"{prefix}.{suffix}"] = _tensor_bytes(value)
    return write_tensor_asset(output, tensors, checkpoint_sha256=checkpoint_sha256)


def _read(asset: TensorAsset, name: str, shape: tuple[int, ...]) -> torch.Tensor:
    view = asset.require(name, shape)
    return torch.frombuffer(bytearray(view.data), dtype=torch.float32).reshape(shape).clone()


def _load_mlp(asset: TensorAsset, prefix: str, mlp: TinyMlp) -> None:
    network = prefix if mlp.norm is None else f"{prefix}.0"
    with torch.no_grad():
        for layer in (0, 2, 4):
            linear = mlp.layers[layer]
            assert isinstance(linear, nn.Linear)
            linear.weight.copy_(_read(asset, f"{network}.layers.{layer}.weight", tuple(linear.weight.shape)))
            linear.bias.copy_(_read(asset, f"{network}.layers.{layer}.bias", tuple(linear.bias.shape)))
        if mlp.norm is not None:
            mlp.norm.weight.copy_(_read(asset, f"{prefix}.1.weight", tuple(mlp.norm.weight.shape)))
            mlp.norm.bias.copy_(_read(asset, f"{prefix}.1.bias", tuple(mlp.norm.bias.shape)))


def load_tinyhood(path: Path | str, device: torch.device | str = "cpu") -> TinyHood:
    asset = load_tensor_asset(path)
    model = TinyHood()
    _load_mlp(asset, "model._learned_model.node_encoder", model.node_encoder)
    _load_mlp(asset, "model._learned_model.edgeset_encoders.mesh", model.mesh_encoder)
    _load_mlp(asset, "model._learned_model.edgeset_encoders.world", model.world_encoder)
    for index, block in enumerate(model.processor_steps):
        base = f"model._learned_model.processor_steps.{index}"
        _load_mlp(asset, f"{base}.mesh_edge_processor", block.mesh_edge_processor)
        _load_mlp(asset, f"{base}.world_edge_processor", block.world_edge_processor)
        _load_mlp(asset, f"{base}.node_processor", block.node_processor)
    _load_mlp(asset, "model._learned_model.decoder", model.decoder)
    return model.to(device)
