#!/usr/bin/env python3
"""Export the official hierarchical HOOD PostCVPR checkpoint to VHOOD v1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

POC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC_ROOT))

from real_scene.formats import load_tensor_asset, sha256_file, write_tensor_asset  # noqa: E402
from real_scene.postcvpr import ARCHITECTURE  # noqa: E402
from tools.export_fine15 import load_state  # noqa: E402


UPSTREAM_COMMIT = "9bc1076195979ac6c027fdd729c6e960cad62f2a"
OFFICIAL_DATA_ID = "1RdA4L6Fy50VsKZ8k7ySp5ps5YtWoHSgs"
EXPECTED_CHECKPOINT_SHA256 = "155d2dd25e54756fc04b0d27996ebca3446b2a59d3a715bb1fb73407753ce5ea"
ARCHITECTURE_STRING = "f,c0|f,c0|f,c0|d:c0,c1|c0,c1|c0,c1|d:c1|c1|c1|u:c0,c1|c0,c1|c0,c1|u:f,c0|f,c0|f,c0"


def validate_postcvpr(tensors: dict[str, torch.Tensor]) -> dict:
    expected_embeddings = {
        "model.nodetype_embedding.weight": (9, 9),
        "model.vertexlevel_embedding.weight": (4, 4),
    }
    for name, shape in expected_embeddings.items():
        if name not in tensors or tuple(tensors[name].shape) != shape:
            raise ValueError(f"PostCVPR tensor {name} must be {shape}")
    required_encoders = {
        "model._learned_model.node_encoder.0.layers.0.weight": (128, 24),
        "model._learned_model.edgeset_encoders.mesh.0.layers.0.weight": (128, 12),
        "model._learned_model.edgeset_encoders.world.0.layers.0.weight": (128, 9),
        "model._learned_model.decoder.layers.4.weight": (3, 128),
    }
    for level in range(3):
        required_encoders[f"model._learned_model.edgeset_encoders.coarse{level}.0.layers.0.weight"] = (128, 12)
    for name, shape in required_encoders.items():
        if name not in tensors or tuple(tensors[name].shape) != shape:
            raise ValueError(f"PostCVPR tensor {name} must be {shape}")

    for block, (level, edge_names) in enumerate(ARCHITECTURE):
        step = block % 3
        prefix = f"model._learned_model.levels.{level}.{step}"
        for edge_name in (*edge_names, "world_edge"):
            name = f"{prefix}.edge_processor_dict.{edge_name}.0.layers.0.weight"
            if name not in tensors or tuple(tensors[name].shape) != (128, 384):
                raise ValueError(f"PostCVPR processor tensor is missing or malformed: {name}")
        node_name = f"{prefix}.node_processor_dict.node.0.layers.0.weight"
        expected_input = 128 * (2 + len(edge_names))
        if node_name not in tensors or tuple(tensors[node_name].shape) != (128, expected_input):
            raise ValueError(f"PostCVPR node processor tensor is missing or malformed: {node_name}")

    normalizers = {}
    expected_widths = {"output": 3, "node": 21, "mesh_edge": 9, "world_edge": 9}
    for label, width in expected_widths.items():
        prefix = f"model._{label}_normalizer."
        matches = {name.rsplit(".", 1)[-1]: tensor for name, tensor in tensors.items() if prefix in name}
        required = {"_acc_count", "_num_accumulations", "_acc_sum", "_acc_sum_squared"}
        if set(matches) != required:
            raise ValueError(f"PostCVPR {label} normalizer fields differ: {sorted(matches)}")
        if matches["_acc_sum"].numel() != width or matches["_acc_sum_squared"].numel() != width:
            raise ValueError(f"PostCVPR {label} normalizer width differs")
        count = float(matches["_acc_count"].item())
        if count < 1.0:
            raise ValueError(f"PostCVPR {label} normalizer count is invalid")
        mean = matches["_acc_sum"] / count
        variance = torch.clamp(matches["_acc_sum_squared"] / count - mean.square(), min=0.0)
        if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
            raise ValueError(f"PostCVPR {label} normalizer is not finite")
        normalizers[label] = {"width": width, "count": count}
    return {"message_passing_steps": 15, "hierarchy_levels": 3, "normalizers": normalizers}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=POC_ROOT / ".work/hood_data/trained_models/postcvpr.pth")
    parser.add_argument("--output", type=Path, default=POC_ROOT / ".work/hood_data/postcvpr.vhood")
    parser.add_argument("--metadata", type=Path, default=POC_ROOT / ".work/hood_data/postcvpr.json")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    digest = sha256_file(checkpoint)
    if digest != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"unexpected official PostCVPR SHA-256: {digest}")
    root, tensors = load_state(checkpoint)
    validation = validate_postcvpr(tensors)
    if args.list:
        for name in sorted(tensors):
            print(f"{name}: {tuple(tensors[name].shape)}")
    packed = {
        name: (tuple(tensor.shape), tensor.numpy().astype("<f4", copy=False).tobytes(order="C"))
        for name, tensor in tensors.items()
    }
    format_metadata = write_tensor_asset(args.output.resolve(), packed, checkpoint_sha256=digest)
    reloaded = load_tensor_asset(args.output.resolve())
    for name, tensor in tensors.items():
        if bytes(reloaded.require(name, tensor.shape).data) != packed[name][1]:
            raise ValueError(f"PostCVPR VHOOD reload differs for tensor {name}")
    metadata = {
        "schema_version": 1,
        "model": "HOOD hierarchical PostCVPR",
        "license": "MIT",
        "source_repository": "https://github.com/Dolorousrtur/HOOD",
        "source_commit": UPSTREAM_COMMIT,
        "official_data_google_drive_id": OFFICIAL_DATA_ID,
        "checkpoint": {"path": str(checkpoint), "sha256": digest},
        "architecture": {
            "node_features": 24,
            "mesh_edge_features": 12,
            "coarse_edge_features": 12,
            "world_edge_features": 9,
            "latent": 128,
            "output": 3,
            "hierarchy_levels": 3,
            "message_passing_steps": 15,
            "architecture_string": ARCHITECTURE_STRING,
            "collision_radius_m": 0.03,
            "k_world_edges": 1,
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
        "checkpoint_root_keys": sorted(root),
        "file": format_metadata,
    }
    args.metadata.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.metadata.resolve().write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"tensors": len(tensors), "floats": sum(t.numel() for t in tensors.values()), **format_metadata}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
