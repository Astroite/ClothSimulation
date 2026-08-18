#!/usr/bin/env python3
"""Generate a deterministic TinyHOOD rollout/golden file from baked runtime assets."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

POC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC_ROOT))

from real_scene.fine15 import Fine15, Fine15Weights  # noqa: E402
from real_scene.formats import Section, pack_u32, write_sectioned  # noqa: E402
from real_scene.runtime_scene import RuntimeScene  # noqa: E402
from real_scene.tinyhood import load_tinyhood  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--motion", default="ch10032_tpose")
    parser.add_argument("--asset-stem", default="ch10032")
    parser.add_argument("--model", type=Path, default=POC_ROOT / ".work/hood_data/tinyhood64x4.vhood")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--golden", required=True, type=Path)
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    torch.use_deterministic_algorithms(True)
    weights = Fine15Weights.from_vhood(args.model.resolve())
    builder = Fine15(weights)
    model = load_tinyhood(args.model.resolve())
    model.eval()
    scene = RuntimeScene.load(args.asset_root.resolve(), args.motion, asset_stem=args.asset_stem)
    output_mean, output_std = weights.normalizer("output")
    position = scene.cloth_target(0)
    previous = position.clone()
    initial = position.clone()
    rollout = []
    first_acceleration = None
    first_world = None
    started = time.perf_counter()
    with torch.no_grad():
        for step in range(args.steps):
            target_frame = min(step + 1, scene.frame_count - 1)
            obstacle_frame = min(step, scene.frame_count - 1)
            obstacle_position, obstacle_normals = scene.proxy(obstacle_frame)
            obstacle_target, _ = scene.proxy(target_frame)
            graph = builder.prepare_graph(
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
                timestep=1.0 / 3.0 if step == 0 else 1.0 / 30.0,
            )
            acceleration = model(graph) * output_std + output_mean
            predicted = graph.effective_position + (graph.effective_position - graph.effective_previous + acceleration)
            predicted[graph.pin_mask] = graph.pin_target[graph.pin_mask]
            if first_acceleration is None:
                first_acceleration = acceleration.clone()
                first_world = torch.full((len(position),), 0xFFFFFFFF, dtype=torch.int64)
                first_world[graph.world_cloth] = graph.active_obstacle[graph.world_obstacle]
            previous = graph.effective_position
            position = predicted
            if step == 0 and args.steps > 1:
                previous = position.clone()
            rollout.append(position.clone())
    assert first_acceleration is not None and first_world is not None
    rollout_tensor = torch.stack(rollout)
    sections = [
        Section("info", 4, 4, pack_u32([args.steps, len(position), len(scene.cloth_senders), len(scene.proxy_positions)])),
        Section("initial_pos", len(position), 12, initial.numpy().astype("<f4").tobytes()),
        Section("first_accel", len(position), 12, first_acceleration.numpy().astype("<f4").tobytes()),
        Section("world_to", len(position), 4, first_world.to(torch.uint32).numpy().astype("<u4").tobytes()),
        Section("rollout_pos", args.steps * len(position), 12, rollout_tensor.numpy().astype("<f4").tobytes()),
    ]
    write_sectioned(args.golden.resolve(), b"VHGOLD01", 1, sections)
    print(
        json.dumps(
            {
                "model": str(args.model.resolve()),
                "motion": args.motion,
                "steps": args.steps,
                "seconds": time.perf_counter() - started,
                "vertices": len(position),
                "mesh_edges": len(scene.cloth_senders),
                "first_world_edges": int((first_world != 0xFFFFFFFF).sum().item()),
                "golden": str(args.golden.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
