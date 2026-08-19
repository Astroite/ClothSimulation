#!/usr/bin/env python3
"""Bake deterministic three-level HOOD PostCVPR coarse graphs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

POC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC_ROOT))

from real_scene.formats import load_sectioned  # noqa: E402
from real_scene.postcvpr_hierarchy import write_hierarchy  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--asset-stem", default="ch10032")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.asset_root.resolve()
    cloth_name = f"{args.asset_stem}.vcloth2" if args.asset_stem != "ch10032" else "ch10032_lower.vcloth2"
    cloth = load_sectioned(root / cloth_name, expected_magic=b"VCLTH002", expected_version=2)
    positions = cloth.require("positions", stride=12)
    triangle_view = cloth.require("triangles", stride=12)
    triangles = np.frombuffer(triangle_view.data, dtype="<u4").reshape(-1, 3).copy()
    output = args.output.resolve() if args.output else root / f"{args.asset_stem}.postcvpr.vhier"
    metadata = write_hierarchy(output, triangles, positions.count)
    print(json.dumps({"output": str(output), **metadata}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
