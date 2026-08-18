#!/usr/bin/env python3
"""Distill a HOOD-compatible 64-wide, four-block student from baked Fine15 rollout states."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

POC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC_ROOT))

from real_scene.fine15 import Fine15, Fine15Graph, Fine15Weights  # noqa: E402
from real_scene.formats import load_sectioned, sha256_file  # noqa: E402
from real_scene.runtime_scene import RuntimeScene  # noqa: E402
from real_scene.tinyhood import LATENT, PROCESSOR_BLOCKS, TinyHood, export_tinyhood, load_tinyhood  # noqa: E402


@dataclass
class DistillationSample:
    graph: Fine15Graph
    target_normalized_acceleration: torch.Tensor
    teacher_position: torch.Tensor
    step: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--threads", type=int, default=min(os.cpu_count() or 1, 16))
    parser.add_argument("--train-steps", type=int, default=48)
    parser.add_argument("--static-steps", type=int, default=120)
    parser.add_argument("--dagger-rounds", type=int, default=3)
    parser.add_argument("--dagger-steps", type=int, default=30)
    parser.add_argument("--dagger-epochs", type=int, default=3)
    parser.add_argument("--dagger-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fine15", type=Path, default=POC_ROOT / ".work/hood_data/fine15.vhood")
    parser.add_argument("--sprint-root", type=Path, default=POC_ROOT / ".work/real_scene/ch10032_sprint")
    parser.add_argument("--tpose-root", type=Path, default=POC_ROOT / ".work/real_scene/ch10032_tpose")
    parser.add_argument("--grid-root", type=Path, default=POC_ROOT / ".work/real_scene/hood_grid64")
    parser.add_argument("--checkpoint", type=Path, default=POC_ROOT / ".work/hood_data/tinyhood64x4.pt")
    parser.add_argument("--output", type=Path, default=POC_ROOT / ".work/hood_data/tinyhood64x4.vhood")
    parser.add_argument("--metadata", type=Path, default=POC_ROOT / ".work/hood_data/tinyhood64x4.json")
    parser.add_argument("--report", type=Path, default=POC_ROOT / "results/tinyhood64x4_python.json")
    return parser.parse_args()


def load_golden(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    asset = load_sectioned(path, expected_magic=b"VHGOLD01", expected_version=1)
    info = torch.frombuffer(bytearray(asset.require("info", count=4, stride=4).data), dtype=torch.uint32).to(torch.long)
    steps, vertices = int(info[0]), int(info[1])
    initial = torch.frombuffer(bytearray(asset.require("initial_pos", count=vertices, stride=12).data), dtype=torch.float32).reshape(vertices, 3)
    first_acceleration = torch.frombuffer(bytearray(asset.require("first_accel", count=vertices, stride=12).data), dtype=torch.float32).reshape(vertices, 3)
    rollout = torch.frombuffer(
        bytearray(asset.require("rollout_pos", count=steps * vertices, stride=12).data), dtype=torch.float32
    ).reshape(steps, vertices, 3)
    return initial, first_acceleration, rollout


def make_graph(
    builder: Fine15,
    scene: RuntimeScene,
    position: torch.Tensor,
    previous: torch.Tensor,
    step: int,
) -> Fine15Graph:
    target_frame = min(step + 1, scene.frame_count - 1)
    obstacle_frame = min(step, scene.frame_count - 1)
    obstacle_position, obstacle_normals = scene.proxy(obstacle_frame)
    obstacle_target, _ = scene.proxy(target_frame)
    return builder.prepare_graph(
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


def cache_sprint_samples(
    builder: Fine15,
    weights: Fine15Weights,
    scene: RuntimeScene,
    golden_path: Path,
) -> tuple[list[DistillationSample], float]:
    initial, first_acceleration, rollout = load_golden(golden_path)
    output_mean, output_std = weights.normalizer("output")
    position = initial.clone()
    previous = initial.clone()
    samples: list[DistillationSample] = []
    started = time.perf_counter()
    with torch.no_grad():
        for step, teacher_position in enumerate(rollout):
            graph = make_graph(builder, scene, position, previous, step)
            acceleration = teacher_position - 2.0 * graph.effective_position + graph.effective_previous
            if step == 0:
                error = (acceleration[~graph.pin_mask] - first_acceleration[~graph.pin_mask]).abs().max().item()
                if error > 1.0e-6:
                    raise ValueError(f"derived first acceleration differs from golden by {error}")
            target = (acceleration - output_mean) / output_std
            samples.append(DistillationSample(graph, target, teacher_position.clone(), step))
            previous = graph.effective_position.clone()
            position = teacher_position.clone()
            if step == 0 and len(rollout) > 1:
                previous = position.clone()
    return samples, time.perf_counter() - started


def sample_loss(model: TinyHood, sample: DistillationSample) -> torch.Tensor:
    prediction = model(sample.graph)
    mask = ~sample.graph.pin_mask
    return torch.mean((prediction[mask] - sample.target_normalized_acceleration[mask]).square())


def optimize_samples(
    model: TinyHood,
    samples: list[DistillationSample],
    *,
    epochs: int,
    learning_rate: float,
) -> list[float]:
    if epochs == 0:
        return []
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-6)
    order = list(range(len(samples)))
    losses = []
    for _ in range(epochs):
        random.shuffle(order)
        total = 0.0
        model.train()
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            loss = sample_loss(model, samples[index])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item())
        losses.append(total / len(samples))
    model.eval()
    return losses


def collect_on_policy_samples(
    model: TinyHood,
    teacher: Fine15,
    weights: Fine15Weights,
    scene: RuntimeScene,
    *,
    steps: int,
    student_blend: float,
) -> list[DistillationSample]:
    output_mean, output_std = weights.normalizer("output")
    position = scene.cloth_target(0)
    previous = position.clone()
    samples = []
    with torch.no_grad():
        for step in range(steps):
            graph = make_graph(teacher, scene, position, previous, step)
            teacher_normalized = teacher.predict_graph(graph)
            student_normalized = model(graph)
            blended = torch.lerp(teacher_normalized, student_normalized, student_blend)
            acceleration = blended * output_std + output_mean
            velocity = graph.effective_position - graph.effective_previous + acceleration
            predicted = graph.effective_position + velocity
            predicted[graph.pin_mask] = graph.pin_target[graph.pin_mask]
            samples.append(DistillationSample(graph, teacher_normalized, predicted.clone(), step))
            previous = graph.effective_position
            position = predicted
            if step == 0 and steps > 1:
                previous = position.clone()
            if not torch.isfinite(position).all():
                break
    return samples


def evaluate_teacher_forced(model: TinyHood, samples: list[DistillationSample], weights: Fine15Weights) -> dict:
    output_mean, output_std = weights.normalizer("output")
    sum_absolute = 0.0
    maximum_absolute = 0.0
    count = 0
    normalized_mse = 0.0
    with torch.no_grad():
        for sample in samples:
            prediction = model(sample.graph)
            mask = ~sample.graph.pin_mask
            difference_normalized = prediction[mask] - sample.target_normalized_acceleration[mask]
            normalized_mse += float(difference_normalized.square().mean().item())
            difference = difference_normalized * output_std
            sum_absolute += float(difference.abs().sum().item())
            maximum_absolute = max(maximum_absolute, float(difference.abs().max().item()))
            count += difference.numel()
    return {
        "samples": len(samples),
        "normalized_mse": normalized_mse / max(len(samples), 1),
        "acceleration_mean_abs": sum_absolute / max(count, 1),
        "acceleration_max_abs": maximum_absolute,
    }


def structure_metrics(scene: RuntimeScene, position: torch.Tensor, pin_target: torch.Tensor) -> dict:
    finite = torch.isfinite(position).all(dim=1)
    safe = torch.nan_to_num(position)
    rest_edges = scene.cloth_rest[scene.cloth_senders] - scene.cloth_rest[scene.cloth_receivers]
    current_edges = safe[scene.cloth_senders] - safe[scene.cloth_receivers]
    ratios = torch.linalg.vector_norm(current_edges, dim=-1) / torch.clamp(
        torch.linalg.vector_norm(rest_edges, dim=-1), min=1.0e-12
    )
    rest_triangles = scene.cloth_rest[scene.cloth_triangles]
    current_triangles = safe[scene.cloth_triangles]
    rest_normal = torch.linalg.cross(rest_triangles[:, 1] - rest_triangles[:, 0], rest_triangles[:, 2] - rest_triangles[:, 0])
    current_normal = torch.linalg.cross(current_triangles[:, 1] - current_triangles[:, 0], current_triangles[:, 2] - current_triangles[:, 0])
    area_ratios = torch.linalg.vector_norm(current_normal, dim=-1) / torch.clamp(
        torch.linalg.vector_norm(rest_normal, dim=-1), min=1.0e-12
    )
    pin_error = (safe[scene.cloth_pins] - pin_target[scene.cloth_pins]).abs()
    return {
        "invalid_vertices": int((~finite).sum().item()),
        "pin_max_abs": float(pin_error.max().item()) if pin_error.numel() else 0.0,
        "edge_ratio_mean": float(ratios.mean().item()),
        "edge_ratio_p95": float(torch.quantile(ratios, 0.95).item()),
        "edge_ratio_max": float(ratios.max().item()),
        "collapsed_fraction_lt_0_5": float((ratios < 0.5).float().mean().item()),
        "stretched_fraction_gt_1_5": float((ratios > 1.5).float().mean().item()),
        "area_ratio_mean": float(area_ratios.mean().item()),
        "degenerate_fraction_lt_0_1": float((area_ratios < 0.1).float().mean().item()),
        "flipped_fraction": float(((rest_normal * current_normal).sum(dim=-1) < 0.0).float().mean().item()),
        "bounds_min": safe.amin(dim=0).tolist(),
        "bounds_max": safe.amax(dim=0).tolist(),
    }


def rollout(
    model: TinyHood,
    builder: Fine15,
    weights: Fine15Weights,
    scene: RuntimeScene,
    steps: int,
    teacher_positions: torch.Tensor | None = None,
) -> dict:
    output_mean, output_std = weights.normalizer("output")
    position = scene.cloth_target(0)
    previous = position.clone()
    maximum_teacher_error = 0.0
    sum_teacher_error = 0.0
    teacher_values = 0
    active_world_edges = 0
    started = time.perf_counter()
    completed = 0
    with torch.no_grad():
        for step in range(steps):
            graph = make_graph(builder, scene, position, previous, step)
            acceleration = model(graph) * output_std + output_mean
            velocity = graph.effective_position - graph.effective_previous + acceleration
            predicted = graph.effective_position + velocity
            velocity[graph.pin_mask] = graph.pin_target[graph.pin_mask] - graph.effective_position[graph.pin_mask]
            predicted[graph.pin_mask] = graph.pin_target[graph.pin_mask]
            if teacher_positions is not None and step < len(teacher_positions):
                difference = (predicted - teacher_positions[step]).abs()
                maximum_teacher_error = max(maximum_teacher_error, float(difference.max().item()))
                sum_teacher_error += float(difference.sum().item())
                teacher_values += difference.numel()
            previous = graph.effective_position
            position = predicted
            if step == 0 and steps > 1:
                previous = position.clone()
            active_world_edges = len(graph.world_cloth)
            completed += 1
            if not torch.isfinite(position).all():
                break
    target_frame = min(completed, scene.frame_count - 1)
    result = {
        "requested_steps": steps,
        "completed_steps": completed,
        "seconds": time.perf_counter() - started,
        "active_world_edges": active_world_edges,
        "teacher_position_max_abs": maximum_teacher_error if teacher_positions is not None else None,
        "teacher_position_mean_abs": sum_teacher_error / max(teacher_values, 1) if teacher_positions is not None else None,
    }
    result.update(structure_metrics(scene, position, scene.cloth_target(target_frame)))
    return result


def main() -> int:
    args = parse_args()
    if args.epochs < 0 or args.train_steps <= 0 or args.static_steps <= 0:
        raise ValueError("epochs must be non-negative and step counts must be positive")
    if min(args.dagger_rounds, args.dagger_steps, args.dagger_epochs) < 0:
        raise ValueError("DAgger arguments must be non-negative")
    if args.resume and not args.checkpoint.is_file():
        raise ValueError(f"resume checkpoint does not exist: {args.checkpoint}")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(args.threads)

    weights = Fine15Weights.from_vhood(args.fine15.resolve())
    builder = Fine15(weights)
    sprint_scene = RuntimeScene.load(args.sprint_root.resolve(), "ch10032_sprint")
    samples, cache_seconds = cache_sprint_samples(
        builder, weights, sprint_scene, args.sprint_root.resolve() / "fine15_rollout.vhgold"
    )
    train_count = min(args.train_steps, len(samples) - 1)
    train_samples, validation_samples = samples[:train_count], samples[train_count:]
    model = TinyHood()
    if args.resume:
        saved = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    training_started = time.perf_counter()
    history = []
    order = list(range(len(train_samples)))
    for epoch in range(args.epochs):
        random.shuffle(order)
        total_loss = 0.0
        model.train()
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            loss = sample_loss(model, train_samples[index])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
        scheduler.step()
        model.eval()
        validation = evaluate_teacher_forced(model, validation_samples, weights)
        record = {
            "epoch": epoch + 1,
            "train_normalized_mse": total_loss / len(train_samples),
            "validation_normalized_mse": validation["normalized_mse"],
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    tpose_scene = RuntimeScene.load(args.tpose_root.resolve(), "ch10032_tpose")
    grid_scene = RuntimeScene.load(args.grid_root.resolve(), "hood_grid64", asset_stem="hood_grid64")
    dagger_history = []
    for dagger_round in range(args.dagger_rounds):
        blend = 0.25 + 0.5 * dagger_round / max(args.dagger_rounds - 1, 1)
        collection_started = time.perf_counter()
        on_policy = []
        on_policy.extend(
            collect_on_policy_samples(
                model,
                builder,
                weights,
                sprint_scene,
                steps=min(args.dagger_steps, sprint_scene.frame_count - 1),
                student_blend=blend,
            )
        )
        on_policy.extend(
            collect_on_policy_samples(
                model, builder, weights, tpose_scene, steps=args.dagger_steps, student_blend=blend
            )
        )
        on_policy.extend(
            collect_on_policy_samples(
                model, builder, weights, grid_scene, steps=args.dagger_steps, student_blend=blend
            )
        )
        collection_seconds = time.perf_counter() - collection_started
        losses = optimize_samples(
            model,
            train_samples + on_policy,
            epochs=args.dagger_epochs,
            learning_rate=args.dagger_learning_rate,
        )
        record = {
            "round": dagger_round + 1,
            "student_blend": blend,
            "on_policy_samples": len(on_policy),
            "collection_seconds": collection_seconds,
            "epoch_losses": losses,
            "validation": evaluate_teacher_forced(model, validation_samples, weights),
        }
        dagger_history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    training_seconds = time.perf_counter() - training_started
    model.eval()

    args.checkpoint.resolve().parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": {"node": 20, "mesh_edge": 12, "world_edge": 9, "latent": LATENT, "blocks": PROCESSOR_BLOCKS},
            "seed": args.seed,
            "model": model.state_dict(),
        },
        args.checkpoint.resolve(),
    )
    checkpoint_hash = sha256_file(args.checkpoint.resolve())
    export_info = export_tinyhood(model, weights, args.output.resolve(), checkpoint_sha256=checkpoint_hash)
    reloaded = load_tinyhood(args.output.resolve())
    reloaded.eval()
    with torch.no_grad():
        reload_max_abs = float((model(samples[0].graph) - reloaded(samples[0].graph)).abs().max().item())
    if reload_max_abs > 1.0e-6:
        raise ValueError(f"VHOOD reload mismatch: {reload_max_abs}")

    _, _, teacher_rollout = load_golden(args.sprint_root.resolve() / "fine15_rollout.vhgold")
    report = {
        "architecture": {"node": 20, "mesh_edge": 12, "world_edge": 9, "latent": LATENT, "blocks": PROCESSOR_BLOCKS},
        "parameter_count": model.parameter_count,
        "fine15_packed_float_count": 3_854_164,
        "parameter_ratio_vs_fine15": model.parameter_count / 3_854_164,
        "checkpoint_sha256": checkpoint_hash,
        "vhood": {key: value for key, value in export_info.items() if key != "tensors"},
        "vhood_reload_max_abs": reload_max_abs,
        "seed": args.seed,
        "threads": args.threads,
        "feature_cache_seconds": cache_seconds,
        "training_seconds": training_seconds,
        "epochs": args.epochs,
        "training_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "final_history": history[-1] if history else None,
        "dagger": dagger_history,
        "teacher_forced_train": evaluate_teacher_forced(model, train_samples, weights),
        "teacher_forced_validation": evaluate_teacher_forced(model, validation_samples, weights),
        "rollouts": {
            "ch10032_sprint": rollout(model, builder, weights, sprint_scene, len(teacher_rollout), teacher_rollout),
            "ch10032_tpose": rollout(model, builder, weights, tpose_scene, args.static_steps),
            "hood_grid64": rollout(model, builder, weights, grid_scene, args.static_steps),
        },
    }
    args.metadata.resolve().write_text(
        json.dumps(
            {
                "architecture": report["architecture"],
                "parameter_count": model.parameter_count,
                "training": {
                    "teacher": str(args.fine15.resolve()),
                    "teacher_sha256": sha256_file(args.fine15.resolve()),
                    "motion": "ch10032_sprint",
                    "samples": len(train_samples),
                    "epochs": args.epochs,
                    "seed": args.seed,
                },
                "file": export_info,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.resolve().write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
