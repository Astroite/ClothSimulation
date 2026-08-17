"""Unreal Python fallback: export a native CH10032 AnimSequence to FBX.

Run through UnrealEditor-Cmd. Environment variables allow the wrapper script to
select another motion without editing this file:

  CH10032_ANIM_ASSET  Unreal object path
  CH10032_ANIM_OUTPUT absolute output FBX path
"""

from __future__ import annotations

import os
from pathlib import Path

import unreal


DEFAULT_ASSET = (
    "/Game/Developers/jinzhao/AICloth/CH_10032/Animation/04_Sprint/"
    "AS_C10032_ArmedSprint_Skirt.AS_C10032_ArmedSprint_Skirt"
)


def main() -> None:
    asset_path = os.environ.get("CH10032_ANIM_ASSET", DEFAULT_ASSET)
    output_text = os.environ.get("CH10032_ANIM_OUTPUT")
    if not output_text:
        raise RuntimeError("CH10032_ANIM_OUTPUT must be an absolute .fbx path")
    output = Path(output_text)
    if not output.is_absolute() or output.suffix.lower() != ".fbx":
        raise RuntimeError(f"invalid CH10032_ANIM_OUTPUT: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    sequence = unreal.load_asset(asset_path)
    if not sequence or not isinstance(sequence, unreal.AnimSequence):
        raise RuntimeError(f"could not load AnimSequence: {asset_path}")

    task = unreal.AssetExportTask()
    task.object = sequence
    task.filename = str(output)
    task.automated = True
    task.prompt = False
    task.replace_identical = True
    task.write_empty_files = False
    task.exporter = unreal.AnimSequenceExporterFBX()
    if not unreal.Exporter.run_asset_export_task(task):
        raise RuntimeError(f"AnimSequence FBX export failed: {asset_path}")
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Unreal reported success but did not create {output}")
    unreal.log(f"CH10032 animation exported: {asset_path} -> {output}")


main()
