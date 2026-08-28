#!/usr/bin/env python3
"""Compile the MLCloth HLSL stages to validated Vulkan 1.1 SPIR-V."""
from __future__ import annotations
import os
import shutil
import subprocess
from pathlib import Path

PROFILES = {".comp": "cs_6_0", ".vert": "vs_6_0", ".frag": "ps_6_0"}

# Named rather than counted. A bare count catches a missing stage but says nothing
# about *which* one, and it silently accepts a renamed file -- which then fails much
# later as a pipeline creation error inside the sample.
EXPECTED = {
    "point_transform.comp",  # Root_M-local cm -> world metres
    "cloth_normals.comp",    # area-weighted vertex normals over the baked triangle CSR
    "point.vert",
    "point.frag",
    "cloth.vert",
    "cloth.frag",
}

def find_tool(name: str) -> str:
    candidates: list[Path] = []
    if sdk := os.environ.get("VULKAN_SDK"):
        candidates.append(Path(sdk) / "Bin" / f"{name}.exe")
    if resolved := shutil.which(name):
        candidates.append(Path(resolved))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(f"Could not find {name}; set VULKAN_SDK or add it to PATH")

def main() -> int:
    source = Path(__file__).parents[1] / "overlay" / "shaders" / "hlsl" / "mlclothcpu"
    dxc = find_tool("dxc")
    validator = find_tool("spirv-val")
    shaders = sorted(path for path in source.iterdir() if path.suffix in PROFILES)
    found = {path.name for path in shaders}
    if found != EXPECTED:
        missing = sorted(EXPECTED - found)
        unexpected = sorted(found - EXPECTED)
        raise RuntimeError(
            f"HLSL stage set in {source} does not match the expected set; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for shader in shaders:
        output = shader.with_suffix(shader.suffix + ".spv")
        subprocess.run([dxc, "-spirv", "-E", "main", "-T", PROFILES[shader.suffix], "-fspv-target-env=vulkan1.1", "-O3", "-Fo", str(output), str(shader)], check=True)
        subprocess.run([validator, "--target-env", "vulkan1.1", str(output)], check=True)
        print(f"compiled and validated {shader.name} -> {output.name}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

