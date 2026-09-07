"""Read-only export of existing animation assets for the Blender demo bake.

Run via UnrealEditor-Cmd -ExecutePythonScript. No UE assets are saved or modified.
"""
import json
from pathlib import Path
import unreal

ROOT = Path(__file__).resolve().parents[1] / '.work/demo/source'
BASE = '/Game/Developers/jinzhao/AICloth/CH_10032/Animation/'
SOURCES = {
    'idle': '01_Idle_Standby/AS_C10032_UnarmedIdle_Standby',
    'walk': '02_Walk/AS_C10032_UnarmedWalk',
    'run': '03_Run/AS_C10032_UnarmedRun',
    'sprint': '04_Sprint/AS_C10032_UnarmedSprint',
    'jump_start': '06_Jump_Fall_Land/AS_C10032_UnarmedJump_IdleTo',
    'jump_land': '06_Jump_Fall_Land/AS_C10032_Armed02Jump_Land_Lit',
    'turn': '05_Strafe_Turn/AS_C10032_ArmedTurn_L180',
}

ROOT.mkdir(parents=True, exist_ok=True)
report = {}
for key, relative in SOURCES.items():
    asset_path = BASE + relative
    asset = unreal.load_asset(asset_path)
    if not isinstance(asset, unreal.AnimSequence):
        raise RuntimeError('Missing animation: ' + asset_path)
    task = unreal.AssetExportTask()
    task.object = asset
    task.filename = str(ROOT / (key + '.fbx'))
    task.automated = True
    task.prompt = False
    task.replace_identical = True
    task.exporter = unreal.AnimSequenceExporterFBX()
    task.options = unreal.FbxExportOption()
    if not unreal.Exporter.run_asset_export_task(task):
        raise RuntimeError('Export failed: ' + asset_path)
    report[key] = {'source': asset_path, 'duration': asset.get_play_length(), 'fbx': task.filename}
    unreal.log('[DemoExport] ' + key)
(ROOT / 'sources.json').write_text(json.dumps(report, indent=2), encoding='utf8')
unreal.log('[DemoExport] COMPLETE')
