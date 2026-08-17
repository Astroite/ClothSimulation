"""Validate VGNN model/golden files and the NumPy reference path."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np

from vgnn import infer_numpy, make_features, read_golden, read_model


def expect_rejected(source: Path, mutation, message: str) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        target = Path(temp_dir) / source.name
        shutil.copyfile(source, target)
        data = bytearray(target.read_bytes())
        mutation(data)
        target.write_bytes(data)
        try:
            read_model(target)
        except ValueError:
            return
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path(__file__).parent / "artifacts")
    args = parser.parse_args()

    model_path = args.artifacts / "model.bin"
    weights = read_model(model_path)
    golden = read_golden(args.artifacts / "golden.bin")
    displacement = golden.position - golden.rest_position
    external = np.broadcast_to(golden.external_acceleration, (golden.graph.vertex_count, 3))
    features = make_features(displacement, golden.velocity, external, golden.pinned)
    actual = infer_numpy(features, golden.graph, weights)
    difference = np.abs(actual - golden.expected_acceleration)
    max_abs = float(difference.max())
    mean_abs = float(difference.mean())
    if max_abs > 1.0e-6 or mean_abs > 1.0e-7:
        raise AssertionError(f"reference mismatch: max={max_abs}, mean={mean_abs}")

    expect_rejected(model_path, lambda data: data.__setitem__(0, ord("X")), "bad magic accepted")
    expect_rejected(model_path, lambda data: data.__setitem__(4, 99), "bad version accepted")
    expect_rejected(model_path, lambda data: data.__setitem__(16, 11), "bad dimension accepted")
    expect_rejected(model_path, lambda data: data.__setitem__(-1, data[-1] ^ 0x01), "bad checksum accepted")
    with tempfile.TemporaryDirectory() as temp_dir:
        truncated = Path(temp_dir) / "model.bin"
        truncated.write_bytes(model_path.read_bytes()[:-4])
        try:
            read_model(truncated)
        except ValueError:
            pass
        else:
            raise AssertionError("truncated model accepted")

    print(f"reference_max_abs={max_abs:.9g}")
    print(f"reference_mean_abs={mean_abs:.9g}")
    print("negative_loader_tests=5/5")


if __name__ == "__main__":
    main()
