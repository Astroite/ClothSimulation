"""Trainable HOOD-compatible student network with a configurable width and depth."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn

from .fine15 import Fine15Graph, Fine15Weights, aggregate_sum
from .formats import TensorAsset, load_tensor_asset, write_tensor_asset


# Defaults describe the first student that shipped (64 wide, 4 blocks). Both are constructor
# arguments now: GPU cost scales as blocks x latent^2, so depth is far cheaper than width and
# the useful architectures are narrow and deep rather than the other way round.
LATENT = 64
PROCESSOR_BLOCKS = 4


class TinyMlp(nn.Module):
    def __init__(self, input_dimension: int, output_dimension: int, *, latent: int = LATENT, layer_norm: bool = True):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dimension, latent),
            nn.ReLU(),
            nn.Linear(latent, latent),
            nn.ReLU(),
            nn.Linear(latent, output_dimension),
        )
        self.norm = nn.LayerNorm(output_dimension, eps=1.0e-5) if layer_norm else None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.layers(value)
        return self.norm(value) if self.norm is not None else value


class TinyProcessor(nn.Module):
    def __init__(self, latent: int = LATENT):
        super().__init__()
        self.mesh_edge_processor = TinyMlp(latent * 3, latent, latent=latent)
        self.world_edge_processor = TinyMlp(latent * 3, latent, latent=latent)
        self.node_processor = TinyMlp(latent * 3, latent, latent=latent)


class TinyHood(nn.Module):
    """HOOD input contract (20/12/9) with a configurable-width, configurable-depth processor."""

    def __init__(self, latent: int = LATENT, blocks: int = PROCESSOR_BLOCKS):
        super().__init__()
        # One lane owns one latent channel on the GPU, so the width must have a compiled
        # SPIR-V variant; see TINY_LATENT_VARIANTS in tools/compile_shaders.py. The depth is
        # bounded by the fixed Vulkan timestamp schedule.
        if latent not in (32, 64):
            raise ValueError(f"latent width {latent} has no compiled Vulkan shader variant")
        if not 1 <= blocks <= 15:
            raise ValueError(f"processor block count {blocks} is outside the Vulkan schedule")
        self.latent = latent
        self.node_encoder = TinyMlp(20, latent, latent=latent)
        self.mesh_encoder = TinyMlp(12, latent, latent=latent)
        self.world_encoder = TinyMlp(9, latent, latent=latent)
        self.processor_steps = nn.ModuleList(TinyProcessor(latent) for _ in range(blocks))
        self.decoder = TinyMlp(latent, 3, latent=latent, layer_norm=False)

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


def infer_architecture(asset: TensorAsset) -> tuple[int, int]:
    """Recover (latent, blocks) from a student checkpoint's tensor shapes.

    Mirrors inferTinyArchitecture in overlay/examples/gnncloth/fine15_gpu_layout.h so the
    Python reference and the Vulkan runtime agree on what a .vhood contains.
    """
    encoder = asset.tensors.get("model._learned_model.node_encoder.0.layers.0.weight")
    if encoder is None:
        raise ValueError("student checkpoint has no node encoder weight to infer the latent width from")
    shape = tuple(encoder.shape)
    if len(shape) != 2 or shape[1] != 20:
        raise ValueError(f"student node encoder does not take the expected 20 node features: {shape}")
    blocks = 0
    while f"model._learned_model.processor_steps.{blocks}.mesh_edge_processor.0.layers.0.weight" in asset.tensors:
        blocks += 1
    return shape[0], blocks


def load_tinyhood(path: Path | str, device: torch.device | str = "cpu") -> TinyHood:
    asset = load_tensor_asset(path)
    latent, blocks = infer_architecture(asset)
    model = TinyHood(latent=latent, blocks=blocks)
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
