"""Bulk-export the CH10032 animation and model set from the Z2Game project.

Runs inside Unreal Python. Reads tools/ch10032_export_manifest.json, exports
every selected asset, and writes an export report with sizes and SHA-256
hashes next to the outputs.

Reaching Python in this project takes a specific invocation. The project
patches FPythonScriptPlugin::StartupModule to return early when
``IsRunningCommandlet() || FApp::IsUnattended()``, so ``-run=pythonscript`` and
anything passing ``-unattended`` leave Python unavailable and no command-line
flag can re-enable it. Driving ``py <script>`` through ``-ExecCmds`` with
neither flag works, and that is what tools/export_ch10032_assets.ps1 does.

Configuration comes from the environment so the wrapper owns the CLI:

  CH10032_MANIFEST     path to the export manifest JSON
  CH10032_OUTPUT_ROOT  directory to write animations/, models/, report
  CH10032_TIER         'all' | 'skirt' | 'locomotion' (animations only)
  CH10032_ONLY         optional comma-separated asset ids
  CH10032_FORCE        '1' to re-export assets that already exist
"""

from __future__ import annotations

import hashlib
import json
import os

import unreal

MANIFEST_PATH = os.environ["CH10032_MANIFEST"]
OUTPUT_ROOT = os.environ["CH10032_OUTPUT_ROOT"]
TIER = os.environ.get("CH10032_TIER", "all")
ONLY = [i.strip() for i in os.environ.get("CH10032_ONLY", "").split(",") if i.strip()]
FORCE = os.environ.get("CH10032_FORCE") == "1"

ANIM_DIR = os.path.join(OUTPUT_ROOT, "animations")
MODEL_DIR = os.path.join(OUTPUT_ROOT, "models")
DATA_DIR = os.path.join(OUTPUT_ROOT, "data")
for directory in (OUTPUT_ROOT, ANIM_DIR, MODEL_DIR, DATA_DIR):
    os.makedirs(directory, exist_ok=True)

with open(MANIFEST_PATH, "r", encoding="utf-8") as handle:
    manifest = json.load(handle)
CONTENT_ROOT = manifest["content_root"]

results: list[dict] = []


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_path(entry: dict) -> str:
    """Resolve an entry to a full object path.

    Most entries are relative to the manifest's content_root, but an entry may
    carry an explicit object_path to reach an asset outside it -- the body
    mesh's real runtime physics asset lives under /Game/Prototype.
    """
    explicit = entry.get("object_path")
    if explicit:
        return f"{explicit}.{explicit.rsplit('/', 1)[-1]}"
    package = entry["package"]
    return f"{CONTENT_ROOT}/{package}.{package.rsplit('/', 1)[-1]}"


def record(entry: dict, status: str, output: str, **extra) -> None:
    row = {
        "id": entry["id"],
        "package": entry.get("package") or entry.get("object_path"),
        "purpose": entry.get("purpose"),
        "status": status,
        "output": output,
    }
    if status in ("exported", "skipped") and output and os.path.isfile(output):
        row["bytes"] = os.path.getsize(output)
        row["sha256"] = sha256_of(output)
    row.update(extra)
    results.append(row)
    unreal.log(f"[{status}] {entry['id']} -> {output}")


def already_done(path: str) -> bool:
    return (
        not FORCE
        and os.path.isfile(path)
        and os.path.getsize(path) > 0
    )


def export_to_fbx(entry: dict, exporter, expected_class, output: str) -> None:
    """Export one asset through an FBX exporter, verifying the file appeared."""
    if already_done(output):
        record(entry, "skipped", output)
        return

    asset = unreal.load_asset(object_path(entry))
    if asset is None:
        record(entry, "failed", output, error="load_failed")
        return
    if expected_class is not None and not isinstance(asset, expected_class):
        record(entry, "failed", output, error=f"unexpected_class:{type(asset).__name__}")
        return

    task = unreal.AssetExportTask()
    task.set_editor_property("object", asset)
    task.set_editor_property("filename", output)
    task.set_editor_property("automated", True)
    task.set_editor_property("prompt", False)
    task.set_editor_property("replace_identical", True)
    task.set_editor_property("write_empty_files", False)
    task.set_editor_property("exporter", exporter)

    ok = unreal.Exporter.run_asset_export_task(task)
    if not ok or not os.path.isfile(output) or os.path.getsize(output) == 0:
        record(entry, "failed", output, error="exporter_reported_failure" if not ok else "no_output")
        return
    record(entry, "exported", output)


def export_to_t3d(entry: dict, output: str) -> bool:
    """Dump an asset's full property tree as UE text.

    PhysicsAsset::SkeletalBodySetups is not a reflected editor property, so
    get_editor_property cannot reach the collision bodies. ObjectExporterT3D
    walks instanced subobjects and does emit them, which makes the .t3d the
    authoritative record here.
    """
    asset = unreal.load_asset(object_path(entry))
    if asset is None:
        record(entry, "failed", output, error="load_failed")
        return False
    task = unreal.AssetExportTask()
    task.set_editor_property("object", asset)
    task.set_editor_property("filename", output)
    task.set_editor_property("automated", True)
    task.set_editor_property("prompt", False)
    task.set_editor_property("replace_identical", True)
    task.set_editor_property("write_empty_files", False)
    task.set_editor_property("exporter", unreal.ObjectExporterT3D())
    ok = unreal.Exporter.run_asset_export_task(task)
    if not ok or not os.path.isfile(output) or os.path.getsize(output) == 0:
        record(entry, "failed", output, error="t3d_export_failed")
        return False
    return True


def summarise_physics_t3d(t3d_path: str) -> dict:
    """Light summary of the .t3d so the report is readable without parsing it.

    The .t3d stays the source of truth; this only pulls out the body count, the
    bones they hang off, and which primitive kinds appear.
    """
    with open(t3d_path, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()

    bodies: list[dict] = []
    current: dict | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("Begin Object Name=") and "SkeletalBodySetup" in line:
            current = {"setup": line.split('Name="', 1)[-1].split('"', 1)[0]}
            continue
        if current is None:
            continue
        if line.startswith("End Object"):
            bodies.append(current)
            current = None
        elif line.startswith("BoneName="):
            current["bone"] = line.split("=", 1)[1].strip('"')
        elif line.startswith("AggGeom="):
            current["geometry"] = [
                kind for kind in
                ("SphereElems", "BoxElems", "SphylElems", "ConvexElems",
                 "TaperedCapsuleElems", "LevelSetElems", "SkinnedLevelSetElems",
                 "SkinnedTriangleMeshElems")
                if kind in line
            ]
        elif line.startswith("CollisionTraceFlag="):
            current["collision_trace_flag"] = line.split("=", 1)[1]

    return {
        "body_count": len(bodies),
        "bones": sorted({b["bone"] for b in bodies if b.get("bone")}),
        "bodies": bodies,
    }


def dump_skeleton_bones(mesh) -> dict:
    """Bone hierarchy and bind transforms via SkeletonModifier.

    The reference skeleton is not a reflected property, but SkeletonModifier
    (UE 5.4+) exposes names, parents and transforms once bound to a skeletal
    mesh. Its signature is GetBoneTransform(BoneName, bGlobal = false), so both
    spaces are emitted under explicit names: parent_local_* matches the
    convention in the existing Assets/Characters/CH10032 sidecar (UE bones run
    down +X, so a spine bone reads as [bone_length, 0, 0]), while component_*
    is the same bone resolved against the component origin and is what bind
    alignment wants.
    """
    modifier = unreal.SkeletonModifier()
    modifier.set_skeletal_mesh(mesh)
    names = [str(n) for n in modifier.get_all_bone_names()]
    index_of = {name: i for i, name in enumerate(names)}

    bones = []
    for index, name in enumerate(names):
        parent = str(modifier.get_parent_name(name))
        if parent in ("None", ""):
            parent = None
        local = modifier.get_bone_transform(name, False)
        world = modifier.get_bone_transform(name, True)
        local_rotation = local.rotation.rotator()
        bones.append({
            "index": index,
            "name": name,
            "parent": parent,
            "parent_index": index_of.get(parent, -1) if parent else -1,
            "parent_local_translation": [
                local.translation.x, local.translation.y, local.translation.z
            ],
            "parent_local_rotation_rpy_deg": [
                local_rotation.roll, local_rotation.pitch, local_rotation.yaw
            ],
            "parent_local_scale": [
                local.scale3d.x, local.scale3d.y, local.scale3d.z
            ],
            "component_translation": [
                world.translation.x, world.translation.y, world.translation.z
            ],
        })
    return {
        "bone_count": len(bones),
        "transform_spaces": {
            "parent_local_*": "relative to the parent bone; comparable to the existing SK_JZ_CH_10032_Body.json sidecar",
            "component_*": "relative to the component origin",
        },
        "bones": bones,
    }


def selected(entries: list[dict], tier_filter: bool = False) -> list[dict]:
    chosen = entries
    if tier_filter and TIER != "all":
        chosen = [e for e in chosen if e.get("tier") == TIER]
    if ONLY:
        chosen = [e for e in chosen if e["id"] in ONLY]
    return chosen


# --- animations -------------------------------------------------------------
for entry in selected(manifest["animations"], tier_filter=True):
    export_to_fbx(
        entry,
        unreal.AnimSequenceExporterFBX(),
        unreal.AnimSequence,
        os.path.join(ANIM_DIR, f"{entry['id']}.fbx"),
    )

# --- skeletal meshes --------------------------------------------------------
exported_meshes: dict[str, object] = {}
for entry in selected(manifest["models"]):
    export_to_fbx(
        entry,
        unreal.SkeletalMeshExporterFBX(),
        unreal.SkeletalMesh,
        os.path.join(MODEL_DIR, f"{entry['id']}.fbx"),
    )
    mesh = unreal.load_asset(object_path(entry))
    if mesh is not None:
        exported_meshes[entry["id"]] = mesh

# --- physics assets (T3D text plus a parsed summary) ------------------------
for entry in selected(manifest["physics"]):
    t3d_output = os.path.join(DATA_DIR, f"{entry['id']}.t3d")
    json_output = os.path.join(DATA_DIR, f"{entry['id']}.json")
    if already_done(t3d_output) and already_done(json_output):
        record(entry, "skipped", t3d_output)
        continue
    if not export_to_t3d(entry, t3d_output):
        continue
    summary = summarise_physics_t3d(t3d_output)
    with open(json_output, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "asset_name": entry["id"],
                "asset_path": object_path(entry),
                "kind": entry["kind"],
                "purpose": entry.get("purpose"),
                "source_of_truth": os.path.basename(t3d_output),
                **summary,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    record(entry, "exported", t3d_output, body_count=summary["body_count"], summary=json_output)

# --- skeletons --------------------------------------------------------------
for entry in selected(manifest["skeletons"]):
    output = os.path.join(DATA_DIR, f"{entry['id']}.json")
    if already_done(output):
        record(entry, "skipped", output)
        continue

    payload = {
        "asset_name": entry["id"],
        "asset_path": object_path(entry),
        "kind": entry["kind"],
        "purpose": entry.get("purpose"),
    }
    source_model = entry.get("bone_source_model")
    mesh = exported_meshes.get(source_model) if source_model else None
    if mesh is not None:
        try:
            payload.update(dump_skeleton_bones(mesh))
            payload["bone_source_model"] = source_model
        except Exception as exc:  # noqa: BLE001 - fall back to the text dump
            payload["bone_dump_error"] = str(exc)
    elif source_model:
        payload["bone_dump_error"] = f"source model not exported: {source_model}"

    if "bones" not in payload:
        # No mesh binds to this skeleton, so text is all we can get.
        t3d_output = os.path.join(DATA_DIR, f"{entry['id']}.t3d")
        if export_to_t3d(entry, t3d_output):
            payload["source_of_truth"] = os.path.basename(t3d_output)
        else:
            continue

    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    record(entry, "exported", output, bone_count=payload.get("bone_count"))

# --- report -----------------------------------------------------------------
counts = {
    "requested": len(results),
    "exported": sum(1 for r in results if r["status"] == "exported"),
    "skipped": sum(1 for r in results if r["status"] == "skipped"),
    "failed": sum(1 for r in results if r["status"] == "failed"),
}
report_path = os.path.join(OUTPUT_ROOT, "export_report.json")
with open(report_path, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "schema_version": 1,
            "manifest": MANIFEST_PATH,
            "tier_filter": TIER,
            "only": ONLY,
            "counts": counts,
            "assets": results,
        },
        handle,
        indent=2,
        ensure_ascii=False,
    )

unreal.log(f"CH10032 export finished: {json.dumps(counts)} -> {report_path}")
for row in results:
    if row["status"] == "failed":
        unreal.log_warning(f"CH10032 export FAILED: {row['id']} {row.get('error')}")
