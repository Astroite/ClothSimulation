"""Compare ablation position dumps to quantify what the network contributes.

The VGNN weights are trained to imitate an analytic target (see the ``target``
field in ``model/artifacts/model.json``), and the XPBD stage afterwards projects
positions back onto rest lengths almost rigidly.  Either of those alone would be
enough to make the network's real influence hard to see, so this script measures
it directly: run the same deterministic scenario four times, changing only where
the acceleration comes from, and compare the resulting positions.

The four modes are chosen so the comparisons isolate different things:

``gnn``
    The network.
``analytic``
    The exact formula the network was trained to imitate.  ``gnn`` versus this
    is the network's own approximation error, accumulated over the rollout.
``gravity``
    The same formula with the neighbour coupling removed.  ``gnn`` versus this
    is what the graph structure contributes at all.
``zero``
    No acceleration whatsoever, which also removes gravity.  Useful only as an
    outer bound; do not use it as the denominator, because "gravity exists"
    would dominate the result.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"VABL"
VERSION = 1
HEADER = struct.Struct("<4s4I")
MODE_NAMES = {0: "gnn", 1: "analytic", 2: "zero", 3: "gravity"}


def read_dump(path: Path) -> tuple[int, np.ndarray]:
    data = path.read_bytes()
    if len(data) < HEADER.size:
        raise ValueError(f"{path} is shorter than its header")
    magic, version, vertex_count, mode, frames = HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ValueError(f"{path} has invalid magic {magic!r}")
    if version != VERSION:
        raise ValueError(f"{path} has unsupported version {version}")
    payload = data[HEADER.size :]
    expected = vertex_count * 3 * 4
    if len(payload) != expected:
        raise ValueError(f"{path} payload is {len(payload)} bytes, expected {expected}")
    positions = np.frombuffer(payload, dtype="<f4").reshape(vertex_count, 3)
    return mode, positions.astype(np.float64), frames


def distances(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    delta = np.linalg.norm(a - b, axis=-1)
    return {
        "mean_vertex_distance": float(delta.mean()),
        "max_vertex_distance": float(delta.max()),
        "l2_norm": float(np.linalg.norm(a - b)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnn", type=Path, required=True)
    parser.add_argument("--analytic", type=Path, required=True)
    parser.add_argument("--zero", type=Path, required=True)
    parser.add_argument("--gravity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dumps: dict[str, np.ndarray] = {}
    frame_counts: set[int] = set()
    sources = (
        ("gnn", args.gnn),
        ("analytic", args.analytic),
        ("zero", args.zero),
        ("gravity", args.gravity),
    )
    for name, path in sources:
        mode, positions, frames = read_dump(path)
        if MODE_NAMES.get(mode) != name:
            raise ValueError(f"{path} records mode {mode} ({MODE_NAMES.get(mode)}), expected {name}")
        dumps[name] = positions
        frame_counts.add(frames)

    shapes = {name: positions.shape for name, positions in dumps.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"dumps disagree on vertex count: {shapes}")
    if len(frame_counts) != 1:
        raise ValueError(f"dumps disagree on frame count: {frame_counts}")

    pairs = {
        "gnn_vs_analytic": ("gnn", "analytic"),
        "gnn_vs_gravity": ("gnn", "gravity"),
        "gnn_vs_zero": ("gnn", "zero"),
        "analytic_vs_gravity": ("analytic", "gravity"),
        "analytic_vs_zero": ("analytic", "zero"),
    }
    measured = {name: distances(dumps[a], dumps[b]) for name, (a, b) in pairs.items()}

    # The entire contribution of graph message passing is analytic minus gravity:
    # same formula, coupling term removed. The network's own error is gnn minus
    # analytic. Comparing the two answers "is the network's error larger than the
    # phenomenon it exists to model?" Do not use zero as a baseline here; that
    # would mostly measure "gravity exists".
    coupling_effect = measured["analytic_vs_gravity"]["l2_norm"]
    network_error = measured["gnn_vs_analytic"]["l2_norm"]
    error_over_coupling = (
        network_error / coupling_effect if coupling_effect > 0.0 else float("nan")
    )

    report = {
        "frames": frame_counts.pop(),
        "vertices": int(dumps["gnn"].shape[0]),
        **measured,
        "coupling_effect_l2": coupling_effect,
        "network_error_l2": network_error,
        "network_error_over_coupling_effect": error_over_coupling,
        "interpretation": (
            "coupling_effect_l2 is |analytic - gravity|: everything graph message passing "
            "contributes, since those two modes differ only by the neighbour term. "
            "network_error_l2 is |gnn - analytic|: how far the network drifts from the target it "
            "was trained to reproduce. A ratio above 1 means the network's approximation error "
            "is larger than the effect it exists to model, so the graph is not earning its cost "
            "in this scenario. That is expected here: XPBD's near-rigid distance constraints "
            "already enforce what the Laplacian term was approximating. It bounds what this PoC "
            "demonstrates to the deployment chain, not learned dynamics."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"vertices={report['vertices']} frames={report['frames']}")
    for key in pairs:
        entry = report[key]
        print(
            f"{key}: mean={entry['mean_vertex_distance']:.6g} "
            f"max={entry['max_vertex_distance']:.6g} l2={entry['l2_norm']:.6g}"
        )
    print(f"coupling_effect_l2={coupling_effect:.6g}  network_error_l2={network_error:.6g}")
    print(f"network_error_over_coupling_effect={error_over_coupling:.6g}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
