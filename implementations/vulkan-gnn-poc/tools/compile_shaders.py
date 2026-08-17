#!/usr/bin/env python3
"""Compile the PoC HLSL shaders to Vulkan SPIR-V with DXC."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


PROFILES = {
    ".comp": "cs_6_0",
    ".vert": "vs_6_0",
    ".frag": "ps_6_0",
}


def find_tool(name: str) -> str:
    sdk = os.environ.get("VULKAN_SDK")
    candidates = []
    if sdk:
        candidates.extend([Path(sdk) / "Bin" / f"{name}.exe", Path(sdk) / "Bin" / name])
    resolved = shutil.which(name)
    if resolved:
        candidates.append(Path(resolved))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(f"Could not find {name}; set VULKAN_SDK or add it to PATH")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).parents[1] / "overlay" / "shaders" / "hlsl" / "gnncloth")
    args = parser.parse_args()

    source = args.source.resolve()
    dxc = find_tool("dxc")
    validator = find_tool("spirv-val")
    shaders = sorted(path for path in source.iterdir() if path.suffix in PROFILES)
    if not shaders:
        raise RuntimeError(f"No HLSL stage files found in {source}")

    for shader in shaders:
        output = shader.with_suffix(shader.suffix + ".spv")
        command = [
            dxc,
            "-spirv",
            "-E", "main",
            "-T", PROFILES[shader.suffix],
            "-fspv-target-env=vulkan1.1",
            "-O3",
            "-Fo", str(output),
            str(shader),
        ]
        subprocess.run(command, check=True)
        subprocess.run([validator, "--target-env", "vulkan1.1", str(output)], check=True)
        print(f"compiled {shader.name} -> {output.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
