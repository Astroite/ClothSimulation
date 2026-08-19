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

# Extra latent widths to build the TinyHOOD student shaders at, beyond the 64 baked into
# tinyhood_mlp.hlsli. One lane owns one latent channel, so the width is also the workgroup
# size and has to be a compile-time constant -- hence a separate SPIR-V module per width
# rather than a runtime parameter. Each entry emits `<prefix><stem>.comp.spv`.
TINY_LATENT_VARIANTS = {
    32: "tiny32_",
}
TINY_VARIANT_SOURCES = (
    "tinyhood_encode.comp",
    "tinyhood_edge_update.comp",
    "tinyhood_node_update.comp",
    "tinyhood_integrate.comp",
)


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

    def build(shader: Path, output: Path, defines: tuple[str, ...] = ()) -> None:
        command = [
            dxc,
            "-spirv",
            "-E", "main",
            "-T", PROFILES[shader.suffix],
            "-fspv-target-env=vulkan1.1",
            "-O3",
        ]
        for define in defines:
            command += ["-D", define]
        command += ["-Fo", str(output), str(shader)]
        subprocess.run(command, check=True)
        subprocess.run([validator, "--target-env", "vulkan1.1", str(output)], check=True)
        print(f"compiled {shader.name} -> {output.name}")

    for shader in shaders:
        build(shader, shader.with_suffix(shader.suffix + ".spv"))

    for latent, prefix in sorted(TINY_LATENT_VARIANTS.items()):
        for stem in TINY_VARIANT_SOURCES:
            shader = source / stem
            if not shader.is_file():
                raise RuntimeError(f"TinyHOOD variant source is missing: {shader}")
            build(shader, source / f"{prefix}{stem}.spv", (f"HOOD_TINY_LATENT={latent}",))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
