#!/usr/bin/env python3
"""Export the official HOOD Fine15 checkpoint to strict VHOOD v1 FP32."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

POC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC_ROOT))

from real_scene.formats import load_tensor_asset, sha256_file, write_tensor_asset  # noqa: E402


UPSTREAM_COMMIT = "9bc1076195979ac6c027fdd729c6e960cad62f2a"
OFFICIAL_DATA_ID = "1RdA4L6Fy50VsKZ8k7ySp5ps5YtWoHSgs"
EXPECTED_CHECKPOINT_SHA256 = "bc92f1fb9a0ca1c9e476ad3981c3e4453bd66519ef16e6f2d6a52305c2aa13cb"


def load_state(path: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Fine15 checkpoint root must be a dictionary")
    state = checkpoint.get("training_module")
    if not isinstance(state, dict):
        raise ValueError("Fine15 checkpoint has no training_module state dictionary")
    tensors: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError(f"training_module contains a non-tensor entry: {name!r}")
        tensor = value.detach().cpu()
        if tensor.dtype not in (torch.float16, torch.float32, torch.float64, torch.int32, torch.int64):
            raise ValueError(f"unsupported checkpoint dtype for {name}: {tensor.dtype}")
        tensor = tensor.to(torch.float32).contiguous()
        if tensor.numel() == 0 or not torch.isfinite(tensor).all().item():
            raise ValueError(f"empty or non-finite checkpoint tensor: {name}")
        tensors[name] = tensor
    return checkpoint, tensors


def validate_fine15(tensors: dict[str, torch.Tensor]) -> dict:
    node_embedding = next((tensor for name, tensor in tensors.items() if name.endswith("nodetype_embedding.weight")), None)
    if node_embedding is None or tuple(node_embedding.shape) != (9, 9):
        raise ValueError("Fine15 NodeType embedding must be 9x9")
    processor_indices = set()
    for name in tensors:
        marker = "._learned_model.processor_steps."
        if marker in name:
            suffix = name.split(marker, 1)[1]
            processor_indices.add(int(suffix.split(".", 1)[0]))
    if processor_indices != set(range(15)):
        raise ValueError(f"Fine15 must contain processor steps 0..14, got {sorted(processor_indices)}")

    linear_shapes = [tuple(tensor.shape) for name, tensor in tensors.items() if name.endswith(".weight") and tensor.ndim == 2]
    if not any(shape == (128, 20) for shape in linear_shapes):
        raise ValueError("Fine15 node encoder 20->128 weight was not found")
    if not any(shape == (128, 12) for shape in linear_shapes):
        raise ValueError("Fine15 mesh encoder 12->128 weight was not found")
    if not any(shape == (128, 9) for shape in linear_shapes):
        raise ValueError("Fine15 world encoder 9->128 weight was not found")
    if not any(shape == (3, 128) for shape in linear_shapes):
        raise ValueError("Fine15 128->3 decoder weight was not found")

    normalizers = {}
    for label in ("output", "node", "mesh_edge", "world_edge"):
        prefix = f"model._{label}_normalizer."
        matches = {name.rsplit(".", 1)[-1]: tensor for name, tensor in tensors.items() if prefix in name}
        required = {"_acc_count", "_num_accumulations", "_acc_sum", "_acc_sum_squared"}
        if set(matches) != required:
            raise ValueError(f"Fine15 {label} normalizer fields differ: {sorted(matches)}")
        count = float(matches["_acc_count"].item())
        mean = matches["_acc_sum"] / max(count, 1.0)
        variance = torch.clamp(matches["_acc_sum_squared"] / max(count, 1.0) - mean.square(), min=0.0)
        std = torch.clamp(torch.sqrt(variance), min=1.0e-8)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError(f"Fine15 {label} normalizer is not finite")
        normalizers[label] = {"width": mean.numel(), "count": count}
    expected_widths = {"output": 3, "node": 17, "mesh_edge": 9, "world_edge": 9}
    if {name: value["width"] for name, value in normalizers.items()} != expected_widths:
        raise ValueError(f"Fine15 normalizer widths differ: {normalizers}")
    return {"processor_steps": 15, "normalizers": normalizers}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=POC_ROOT / ".work/hood_data/trained_models/fine15.pth")
    parser.add_argument("--output", type=Path, default=POC_ROOT / ".work/hood_data/fine15.vhood")
    parser.add_argument("--metadata", type=Path, default=POC_ROOT / ".work/hood_data/fine15.json")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    checkpoint_path = args.checkpoint.resolve()
    digest = sha256_file(checkpoint_path)
    if digest != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"unexpected official Fine15 SHA-256: {digest}")
    checkpoint, tensors = load_state(checkpoint_path)
    if args.list:
        for name in sorted(tensors):
            print(f"{name}: {tuple(tensors[name].shape)}")
    validation = validate_fine15(tensors)
    packed = {
        name: (tuple(tensor.shape), tensor.numpy().astype("<f4", copy=False).tobytes(order="C"))
        for name, tensor in tensors.items()
    }
    format_metadata = write_tensor_asset(args.output.resolve(), packed, checkpoint_sha256=digest)
    reloaded = load_tensor_asset(args.output.resolve())
    for name, tensor in tensors.items():
        view = reloaded.require(name, tensor.shape)
        if bytes(view.data) != packed[name][1]:
            raise ValueError(f"VHOOD reload differs for tensor {name}")
    metadata = {
        "schema_version": 1,
        "model": "HOOD Fine15 single-level CVPR baseline",
        "license": "MIT",
        "source_repository": "https://github.com/dolorousrtur/hood",
        "source_commit": UPSTREAM_COMMIT,
        "official_data_google_drive_id": OFFICIAL_DATA_ID,
        "checkpoint": {"path": str(checkpoint_path), "sha256": digest},
        "architecture": {
            "node_features": 20,
            "mesh_edge_features": 12,
            "world_edge_features": 9,
            "latent": 128,
            "output": 3,
            "mlp_hidden_layers": 2,
            "message_passing_steps": 15,
            "collision_radius_m": 0.03,
            "k_world_edges": 1,
            "use_current_obstacle_pos": True,
            "initial_timestep_s": 1.0 / 3.0,
            "regular_timestep_s": 1.0 / 30.0,
        },
        "default_material": {
            "density_kg_m2": 0.20022,
            "lame_mu": 23600.0,
            "lame_lambda": 44400.0,
            "bending_coefficient": 3.9625778333333325e-5,
        },
        "validation": validation,
        "checkpoint_root_keys": sorted(checkpoint),
        "file": format_metadata,
    }
    args.metadata.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.metadata.resolve().write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"tensors": len(tensors), "file_bytes": format_metadata["file_bytes"], "sha256": format_metadata["file_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
