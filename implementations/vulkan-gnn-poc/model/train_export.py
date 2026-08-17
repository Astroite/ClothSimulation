"""Train the fixed 10->16->3 graph network and export VGNN v1 assets."""

from __future__ import annotations

import argparse
import platform
from pathlib import Path

import numpy as np
import torch
from torch import nn

from vgnn import (
    HIDDEN_DIM,
    INPUT_DIM,
    INPUT_SCALE,
    OUTPUT_DIM,
    OUTPUT_SCALE,
    GoldenCase,
    Weights,
    infer_numpy,
    make_features,
    make_grid_graph,
    make_rest_positions,
    make_smooth_state,
    neighbor_mean_torch,
    oracle_acceleration,
    read_model,
    write_golden,
    write_manifest,
    write_model,
)


class ShaderFriendlyGNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self0 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.neighbor0 = nn.Linear(INPUT_DIM, HIDDEN_DIM, bias=False)
        self.self1 = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
        self.neighbor1 = nn.Linear(HIDDEN_DIM, OUTPUT_DIM, bias=False)

    def forward(self, x: torch.Tensor, graph) -> torch.Tensor:
        neighbor_x = neighbor_mean_torch(x, graph)
        hidden = torch.relu(self.self0(x) + self.neighbor0(neighbor_x))
        neighbor_hidden = neighbor_mean_torch(hidden, graph)
        return self.self1(hidden) + self.neighbor1(neighbor_hidden)


def make_training_set(graph, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for _ in range(count):
        displacement, velocity, external, pinned = make_smooth_state(graph, rng)
        inputs.append(make_features(displacement, velocity, external, pinned))
        targets.append(oracle_acceleration(displacement, velocity, external, pinned, graph))
    return np.stack(inputs), np.stack(targets)


def export_weights(model: ShaderFriendlyGNN) -> Weights:
    def array(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy().astype(np.float32, copy=True)

    return Weights(
        input_scale=INPUT_SCALE.copy(),
        output_scale=OUTPUT_SCALE.copy(),
        self0=array(model.self0.weight),
        neighbor0=array(model.neighbor0.weight),
        bias0=array(model.self0.bias),
        self1=array(model.self1.weight),
        neighbor1=array(model.neighbor1.weight),
        bias1=array(model.self1.bias),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "artifacts")
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--grid", type=int, default=16)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    np.random.seed(args.seed)

    graph = make_grid_graph(args.grid)
    inputs_np, targets_np = make_training_set(graph, args.samples, args.seed)
    x = torch.from_numpy(inputs_np * INPUT_SCALE)
    target = torch.from_numpy(targets_np / OUTPUT_SCALE)
    pin_mask = 1.0 - torch.from_numpy(inputs_np[..., 9:10])

    model = ShaderFriendlyGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=4.0e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x, graph) * pin_mask
        loss = torch.mean((prediction - target) ** 2)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 100 == 0:
            print(f"epoch={epoch + 1:04d} normalized_mse={loss.item():.8f}")

    weights = export_weights(model)
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "model.bin"
    model_info = write_model(model_path, weights)
    # Generate the golden output from the exact bytes consumed by the runtime.
    # The position round-trip is intentional: the shader reconstructs displacement
    # from two FP32 position buffers rather than receiving it directly.
    runtime_weights = read_model(model_path)

    golden_graph = make_grid_graph(32)
    rng = np.random.default_rng(args.seed + 1)
    displacement, velocity, external, pinned = make_smooth_state(golden_graph, rng)
    rest = make_rest_positions(golden_graph.grid_size)
    position = np.asarray(rest + displacement, dtype=np.float32)
    features = make_features(position - rest, velocity, external, pinned)
    expected = infer_numpy(features, golden_graph, runtime_weights)
    golden = GoldenCase(
        graph=golden_graph,
        rest_position=rest,
        position=position,
        velocity=velocity,
        pinned=pinned,
        external_acceleration=external[0],
        expected_acceleration=expected,
    )
    golden_info = write_golden(args.output / "golden.bin", golden)

    with torch.no_grad():
        training_prediction = model(x, graph) * pin_mask
        normalized_mse = float(torch.mean((training_prediction - target) ** 2))
        physical_mae = float(torch.mean(torch.abs((training_prediction - target) * torch.from_numpy(OUTPUT_SCALE))))

    manifest = {
        "format": "VGNN v1",
        "architecture": {
            "input": INPUT_DIM,
            "hidden": HIDDEN_DIM,
            "output": OUTPUT_DIM,
            "aggregation": "mean of 8-neighborhood without self loops",
            "layer": "W_self * h_i + W_neighbor * mean(h_j) + bias",
            "activation": "ReLU after layer 0",
        },
        "feature_order": [
            "displacement.x",
            "displacement.y",
            "displacement.z",
            "velocity.x",
            "velocity.y",
            "velocity.z",
            "external_acceleration.x",
            "external_acceleration.y",
            "external_acceleration.z",
            "pinned",
        ],
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "samples": args.samples,
            "grid_size": args.grid,
            "normalized_mse": normalized_mse,
            "physical_mae": physical_mae,
            "target": "18 * (mean(neighbor displacement) - displacement) - 0.9 * velocity + external acceleration",
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "license": "Generated by this PoC; repository license applies.",
        "model": model_info,
        "golden": golden_info,
    }
    write_manifest(args.output / "model.json", manifest)
    print(f"wrote {args.output / 'model.bin'}")
    print(f"wrote {args.output / 'golden.bin'}")
    print(f"physical_mae={physical_mae:.6f}")


if __name__ == "__main__":
    main()
