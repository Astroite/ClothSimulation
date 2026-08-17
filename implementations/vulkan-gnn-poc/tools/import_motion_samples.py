#!/usr/bin/env python3
"""Copy the curated local motion subset into the Vulkan GNN PoC Assets tree.

The source dataset is intentionally not scanned wholesale. A fixed list of six
validated exports is copied with stable names, then a compact manifest with
relative paths and SHA-256 hashes is generated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


SAMPLES = (
    ("hml_001962", "baseline forward walk, turn, and stop"),
    ("hml_009402", "walk, right spin, and return to the start"),
    ("hml_000295", "squat and sustained lower-body bend"),
    ("hml_001977", "single left-leg kick"),
    ("hml_002926", "squat, vertical jump, and landing"),
    ("hml_011319", "full-body spin with arms extended"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_asset(source: Path, target: Path, overwrite: bool) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(f"Required source asset does not exist: {source}")

    source_hash = sha256(source)
    if target.exists():
        if target.is_file() and sha256(target) == source_hash:
            pass
        elif not overwrite:
            raise FileExistsError(
                f"Target exists with different contents: {target}; "
                "pass --overwrite to refresh it"
            )
        else:
            temporary = target.with_name(target.name + ".importing")
            shutil.copy2(source, temporary)
            temporary.replace(target)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".importing")
        shutil.copy2(source, temporary)
        temporary.replace(target)

    return {
        "path": target.as_posix(),
        "bytes": target.stat().st_size,
        "sha256": source_hash,
    }


def relative_source_path(value: str, source_root: Path) -> str:
    normalized = value.replace("\\", "/")
    prefix = source_root.as_posix().rstrip("/") + "/"
    if normalized.lower().startswith(prefix.lower()):
        return normalized[len(prefix) :]
    return normalized


def main() -> int:
    poc_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(r"F:\Projects\Anim"),
        help="Root of the existing motion export (default: F:\\Projects\\Anim)",
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=poc_root / "Assets",
        help="Destination Assets directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace destination files only when their contents differ",
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    assets_root = args.assets_root.resolve()
    samples_root = assets_root / "TrainingSamples"
    source_manifest_path = (
        source_root
        / "output"
        / "motion_export"
        / "manifests"
        / "exported_motion_manifest.json"
    )
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    entries = {entry["motion_id"]: entry for entry in source_manifest["entries"]}

    selected: list[dict[str, Any]] = []
    for motion_id, purpose in SAMPLES:
        if motion_id not in entries:
            raise KeyError(f"Motion {motion_id} is absent from {source_manifest_path}")
        entry = entries[motion_id]
        if entry.get("validation_status") != "passed":
            raise ValueError(f"Motion {motion_id} did not pass source FBX validation")

        sample_dir = samples_root / motion_id
        raw_source = (
            source_root
            / "output"
            / "motion_export"
            / "segments"
            / "raw"
            / f"{motion_id}.npz"
        )
        resampled_source = (
            source_root
            / "output"
            / "motion_export"
            / "segments"
            / "resampled_60fps"
            / f"{motion_id}.npz"
        )
        fbx_name = Path(entry["fbx_skeleton_path"]).name
        fbx_source = (
            source_root
            / "output"
            / "motion_export"
            / "export"
            / "smpl_body22"
            / "skeleton_only"
            / fbx_name
        )

        files = {
            "raw_npz": copy_asset(
                raw_source, sample_dir / f"{motion_id}_raw.npz", args.overwrite
            ),
            "resampled_60fps_npz": copy_asset(
                resampled_source,
                sample_dir / f"{motion_id}_60fps.npz",
                args.overwrite,
            ),
            "skeleton_fbx": copy_asset(
                fbx_source, sample_dir / fbx_name, args.overwrite
            ),
        }
        for file_entry in files.values():
            file_entry["path"] = Path(file_entry["path"]).relative_to(
                assets_root
            ).as_posix()

        selected.append(
            {
                "motion_id": motion_id,
                "purpose": purpose,
                "humanml3d_id": entry["humanml3d_id"],
                "amass_dataset": entry["amass_dataset"],
                "amass_source_path": relative_source_path(
                    entry["amass_source_path"], source_root
                ),
                "text_descriptions": entry["text_descriptions"],
                "action_tags": entry["action_tags"],
                "primary_tag": entry["primary_tag"],
                "gender": entry["gender"],
                "duration_s": entry["duration_s"],
                "frame_count": entry["frame_count"],
                "source_fps": entry["mocap_framerate"],
                "resampled_60fps_trusted": entry["resampled_60fps_trusted"],
                "motion_metrics": entry["motion_metrics"],
                "fbx_validation_status": entry["validation_status"],
                "fbx_validation_checks": entry["validation_checks"],
                "fbx_validation_details": entry["validation_details"],
                "files": files,
            }
        )

    character_root = assets_root / "Characters" / "CH10032"
    character_source = source_root / "Assets" / "CH_0032"
    character_files: dict[str, dict[str, Any]] = {}
    for source_name in ("SK_JZ_CH_10032_Body.FBX", "SK_JZ_CH_10032_Body.json"):
        file_entry = copy_asset(
            character_source / source_name,
            character_root / source_name,
            args.overwrite,
        )
        file_entry["path"] = Path(file_entry["path"]).relative_to(
            assets_root
        ).as_posix()
        character_files[source_name] = file_entry

    output_manifest = {
        "schema_version": 1,
        "source_manifest": {
            "path_within_source_root": relative_source_path(
                str(source_manifest_path), source_root
            ),
            "schema_version": source_manifest.get("schema_version"),
            "generated_at": source_manifest.get("generated_at"),
        },
        "selection_policy": (
            "Six validated, non-mirrored motions covering locomotion, turning, "
            "squatting, kicking, jumping/landing, and axial rotation."
        ),
        "data_terms_notice": (
            "These local data assets retain their original HumanML3D, AMASS, "
            "KIT, BMLmovi, SMPL, and character terms; the PoC MIT license does "
            "not grant redistribution rights."
        ),
        "motions": selected,
        "character_reference": {
            "name": "CH10032",
            "purpose": "body alignment, skinning reference, and collision proxy source",
            "files": character_files,
        },
    }

    manifest_path = samples_root / "manifest.json"
    manifest_text = json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n"
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != manifest_text:
        if not args.overwrite:
            raise FileExistsError(
                f"Manifest differs from generated contents: {manifest_path}; "
                "pass --overwrite to refresh it"
            )
    manifest_path.write_text(manifest_text, encoding="utf-8", newline="\n")
    print(f"Imported {len(selected)} motions into {samples_root}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
