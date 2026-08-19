#!/usr/bin/env python3
"""Export a randomly initialised student checkpoint for GPU cost measurement.

Kernel timing does not depend on weight values, only on the shapes the shaders dispatch over.
So the cheapest way to choose a student architecture is to export random weights at each
candidate width/depth, benchmark them, and only then spend training time on the winner.

The output is a fully valid VHOOD -- it just predicts nonsense, so it is for
benchmark_hood_static.ps1 only, never for verify_hood.ps1 or a quality claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

POC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC_ROOT))

from real_scene.fine15 import Fine15Weights  # noqa: E402
from real_scene.formats import sha256_file  # noqa: E402
from real_scene.tinyhood import TinyHood, export_tinyhood  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent", type=int, required=True, choices=(32, 64))
    parser.add_argument("--blocks", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--fine15", type=Path, default=POC_ROOT / ".work/hood_data/fine15.vhood")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    output = args.output or POC_ROOT / f".work/hood_data/student{args.latent}x{args.blocks}_random.vhood"
    # The normalizers and the node-type embedding are copied from the teacher, exactly as a
    # trained student would, so the feature and encode passes see realistic magnitudes.
    teacher = Fine15Weights.from_vhood(args.fine15.resolve())
    model = TinyHood(latent=args.latent, blocks=args.blocks)
    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    info = export_tinyhood(model, teacher, output.resolve(), checkpoint_sha256="0" * 64)
    print(json.dumps({
        "latent": args.latent,
        "blocks": args.blocks,
        "parameter_count": model.parameter_count,
        "cost_units_blocks_times_latent_squared": args.blocks * args.latent * args.latent,
        "output": str(output),
        "file_bytes": info["file_bytes"],
        "sha256": sha256_file(output.resolve()),
        "random_weights": True,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
