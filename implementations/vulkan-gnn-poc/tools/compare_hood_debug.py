#!/usr/bin/env python3
"""Compare --hood-verify GPU debug buffers with the pure-PyTorch first step."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

POC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC_ROOT))

from real_scene.fine15 import Fine15, Fine15Weights  # noqa: E402
from real_scene.runtime_scene import RuntimeScene  # noqa: E402


def error(gpu: np.ndarray, reference) -> dict[str, float]:
    difference = np.abs(gpu - reference.detach().cpu().numpy())
    return {"max_abs_error": float(difference.max(initial=0.0)), "mean_abs_error": float(difference.mean())}


def load(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    expected = int(np.prod(shape))
    if values.size != expected:
        raise ValueError(f"{path} contains {values.size} floats, expected {expected}")
    return values.reshape(shape)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--motion", default="ch10032_sprint")
    parser.add_argument("--model", type=Path, default=POC_ROOT / ".work/hood_data/fine15.vhood")
    parser.add_argument("--debug-root", type=Path, default=POC_ROOT / "results")
    args = parser.parse_args()

    scene = RuntimeScene.load(args.asset_root.resolve(), args.motion)
    weights = Fine15Weights.from_vhood(args.model.resolve())
    model = Fine15(weights)
    position = scene.cloth_target(0)
    proxy_position, proxy_normals = scene.proxy(0)
    proxy_target, _ = scene.proxy(1)
    trace = {}
    model.step(
        position=position,
        previous=position,
        rest_position=scene.cloth_rest,
        triangles=scene.cloth_triangles,
        mesh_senders=scene.cloth_senders,
        mesh_receivers=scene.cloth_receivers,
        mass=scene.cloth_mass,
        pin_mask=scene.cloth_pins,
        pin_target=scene.cloth_target(1),
        obstacle_position=proxy_position,
        obstacle_target=proxy_target,
        obstacle_normals=proxy_normals,
        timestep=1.0 / 3.0,
        trace=trace,
    )

    cloth_count = len(position)
    proxy_count = len(proxy_position)
    mesh_count = len(scene.cloth_senders)
    root = args.debug_root.resolve()
    node = load(root / "hood_debug_node_features.bin", (cloth_count + proxy_count, 20))
    mesh = load(root / "hood_debug_mesh_features.bin", (mesh_count, 12))
    direct = load(root / "hood_debug_world_direct_features.bin", (cloth_count, 9))
    inverse = load(root / "hood_debug_world_inverse_features.bin", (cloth_count, 9))
    latent = load(root / "hood_debug_node_latent.bin", (cloth_count + proxy_count, 128))
    world_cloth = trace["world_cloth"].cpu().numpy()
    active_obstacle = trace["active_obstacle"].cpu().numpy()
    cloth_reference = trace["cloth_node_features"].detach().cpu().numpy()
    node_mean, node_std = weights.normalizer("node")
    node_mean = node_mean.cpu().numpy().reshape(-1)
    node_std = node_std.cpu().numpy().reshape(-1)
    gpu_normals = node[:cloth_count, 12:15] * node_std[12:15] + node_mean[12:15]
    reference_normals = cloth_reference[:, 12:15] * node_std[12:15] + node_mean[12:15]
    normal_dots = (gpu_normals * reference_normals).sum(axis=1)
    report = {
        "cloth_node_features": error(node[:cloth_count], trace["cloth_node_features"]),
        "obstacle_node_features": error(node[cloth_count + active_obstacle], trace["obstacle_node_features"]),
        "mesh_features": error(mesh, trace["mesh_features"]),
        "world_direct_features": error(direct[world_cloth], trace["world_direct_features"]),
        "world_inverse_features": error(inverse[world_cloth], trace["world_inverse_features"]),
        "final_cloth_node_latent": error(latent[:cloth_count], trace["cloth_node_latent"]),
        "cloth_node_per_feature_max": np.abs(node[:cloth_count] - cloth_reference).max(axis=0).tolist(),
        "cloth_normal_dots": {
            "minimum": float(normal_dots.min()),
            "mean": float(normal_dots.mean()),
            "negative_count": int((normal_dots < 0.0).sum()),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
