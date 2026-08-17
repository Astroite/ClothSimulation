#!/usr/bin/env python3
"""Run deterministic Fine15 inference on baked CH10032 assets."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

POC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC_ROOT))

from real_scene.fine15 import Fine15, Fine15Weights  # noqa: E402
from real_scene.formats import Section, pack_f32, pack_u32, write_sectioned  # noqa: E402
from real_scene.runtime_scene import RuntimeScene  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--motion", default="ch10032_sprint")
    parser.add_argument("--model", type=Path, default=POC_ROOT / ".work/hood_data/fine15.vhood")
    parser.add_argument("--checkpoint", type=Path, default=POC_ROOT / ".work/hood_data/trained_models/fine15.pth")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--golden", type=Path)
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    scene = RuntimeScene.load(args.asset_root.resolve(), args.motion, args.device)
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    steps = min(args.steps, max(scene.frame_count - 1, 1))

    exported_weights = Fine15Weights.from_vhood(args.model.resolve(), args.device)
    direct_weights = Fine15Weights.from_checkpoint(args.checkpoint.resolve(), args.device)
    if set(exported_weights.tensors) != set(direct_weights.tensors):
        raise ValueError("checkpoint and VHOOD tensor names differ")
    maximum_weight_error = max(
        float((exported_weights.tensors[name] - direct_weights.tensors[name]).abs().max().item())
        for name in exported_weights.tensors
    )
    if maximum_weight_error > 0.0:
        raise ValueError(f"VHOOD differs from checkpoint by {maximum_weight_error}")

    model = Fine15(exported_weights)
    position = scene.cloth_target(0)
    previous = position.clone()
    initial_position = position.clone()
    rollout = []
    first_acceleration = None
    first_world_obstacle = None
    start = time.perf_counter()
    max_pin_error = 0.0
    for step in range(steps):
        target_frame = min(step + 1, scene.frame_count - 1)
        obstacle_frame = min(step, scene.frame_count - 1)
        obstacle_target_frame = target_frame
        obstacle_position, obstacle_normals = scene.proxy(obstacle_frame)
        obstacle_target, _ = scene.proxy(obstacle_target_frame)
        timestep = 1.0 / 3.0 if step == 0 else 1.0 / 30.0
        output = model.step(
            position=position,
            previous=previous,
            rest_position=scene.cloth_rest,
            triangles=scene.cloth_triangles,
            mesh_senders=scene.cloth_senders,
            mesh_receivers=scene.cloth_receivers,
            mass=scene.cloth_mass,
            pin_mask=scene.cloth_pins,
            pin_target=scene.cloth_target(target_frame),
            obstacle_position=obstacle_position,
            obstacle_target=obstacle_target,
            obstacle_normals=obstacle_normals,
            timestep=timestep,
        )
        if first_acceleration is None:
            first_acceleration = output.acceleration.detach().cpu()
            mapping = torch.full((len(position),), 0xFFFFFFFF, dtype=torch.int64)
            mapping[output.world_cloth.detach().cpu()] = output.world_obstacle.detach().cpu()
            first_world_obstacle = mapping
        target = scene.cloth_target(target_frame)
        max_pin_error = max(max_pin_error, float((output.position[scene.cloth_pins] - target[scene.cloth_pins]).abs().max().item()))
        if not torch.isfinite(output.position).all():
            raise ValueError(f"Fine15 produced NaN/Inf at step {step}")
        previous = output.effective_position.detach()
        position = output.position.detach()
        if step == 0 and steps > 1:
            previous = position.clone()  # Official baseline forces zero cloth velocity on rollout step 1.
        rollout.append(position.cpu())
    elapsed = time.perf_counter() - start
    if max_pin_error > 1.0e-6:
        raise ValueError(f"Fine15 pin drift exceeds tolerance: {max_pin_error}")

    if args.golden:
        assert first_acceleration is not None and first_world_obstacle is not None
        rollout_tensor = torch.stack(rollout)
        sections = [
            Section("info", 4, 4, pack_u32([steps, len(position), len(scene.cloth_senders), len(scene.proxy_positions)])),
            Section("initial_pos", len(position), 12, initial_position.cpu().numpy().astype("<f4").tobytes()),
            Section("first_accel", len(position), 12, first_acceleration.numpy().astype("<f4").tobytes()),
            Section("world_to", len(position), 4, first_world_obstacle.to(torch.uint32).numpy().astype("<u4").tobytes()),
            Section("rollout_pos", steps * len(position), 12, rollout_tensor.numpy().astype("<f4").tobytes()),
        ]
        write_sectioned(args.golden.resolve(), b"VHGOLD01", 1, sections)
    report = {
        "character": "CH10032",
        "motion": args.motion,
        "device": args.device,
        "steps": steps,
        "seconds": elapsed,
        "seconds_per_step": elapsed / steps,
        "cloth_vertices": len(position),
        "mesh_edges": len(scene.cloth_senders),
        "first_world_edges": int((first_world_obstacle != 0xFFFFFFFF).sum().item()) if first_world_obstacle is not None else 0,
        "max_pin_error": max_pin_error,
        "max_weight_error": maximum_weight_error,
        "position_aabb": [
            position.amin(dim=0).cpu().tolist(),
            position.amax(dim=0).cpu().tolist(),
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
