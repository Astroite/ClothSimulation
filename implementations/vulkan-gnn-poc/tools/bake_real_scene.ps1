param(
    [string]$Motion = 'ch10032_sprint',
    [string]$AnimAsset = '/Game/Developers/jinzhao/AICloth/CH_10032/Animation/04_Sprint/AS_C10032_ArmedSprint_Skirt.AS_C10032_ArmedSprint_Skirt',
    # Point this at an FBX already on disk -- for example one of the 31 clips
    # tools/export_ch10032_assets.ps1 pulled into .work/ch10032_library/animations --
    # and pass -SkipUnrealExport to bake it without launching the editor again. The
    # default keeps the per-scene path the Unreal export writes to, so leaving it
    # unset reproduces the original single-clip behaviour exactly.
    [string]$AnimationFbx = '',
    [double]$Duration = 0.0,
    [switch]$SkipUnrealExport,
    [switch]$SkipFine15Golden,
    [switch]$StaticPose
)

$ErrorActionPreference = 'Stop'
$PocRoot = Split-Path -Parent $PSScriptRoot
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PocRoot)
$RuntimeRoot = Join-Path $PocRoot ".work/real_scene/$Motion"
if ([string]::IsNullOrWhiteSpace($AnimationFbx)) {
    $AnimationFbx = Join-Path $RuntimeRoot 'target_animation.fbx'
} else {
    $AnimationFbx = (Resolve-Path -LiteralPath $AnimationFbx).Path
    if (-not $SkipUnrealExport) {
        throw '-AnimationFbx names an existing FBX, so pass -SkipUnrealExport too; otherwise the Unreal export would overwrite it.'
    }
}
$UnrealEditor = 'E:\Main\Engine\Binaries\Win64\UnrealEditor-Cmd.exe'
$Project = 'E:\Main\Projects\Z2Game\Z2Game.uproject'
$Blender = 'C:\Program Files\Blender Foundation\Blender 4.5\blender.exe'

New-Item -ItemType Directory -Force -Path $RuntimeRoot | Out-Null

if (-not $SkipUnrealExport) {
    if (-not (Test-Path -LiteralPath $UnrealEditor -PathType Leaf)) { throw "UnrealEditor-Cmd not found: $UnrealEditor" }
    if (-not (Test-Path -LiteralPath $Project -PathType Leaf)) { throw "Unreal project not found: $Project" }
    # The project does not initialize Python in commandlet mode. Unreal's
    # built-in OBJ EXPORT command invokes the same AnimSequence FBX exporter.
    $Package = $AnimAsset.Substring(0, $AnimAsset.LastIndexOf('.'))
    $Exec = "OBJ EXPORT TYPE=/Script/Engine.AnimSequence NAME=$AnimAsset FILE=$AnimationFbx,QUIT_EDITOR"
    $LoadOverride = "-ini:Engine:[/Script/UnrealEd.UnrealEdEngine]:PackagesToBeFullyLoadedAtStartup=$Package"
    & $UnrealEditor $Project $LoadOverride "-ExecCmds=$Exec" -unattended -nop4 -nosplash -nullrhi -NoSound
    if ($LASTEXITCODE -ne 0) { throw "Unreal animation export failed with exit code $LASTEXITCODE" }
}

if (-not (Test-Path -LiteralPath $AnimationFbx -PathType Leaf)) { throw "Animation FBX is missing: $AnimationFbx" }
if (-not (Test-Path -LiteralPath $Blender -PathType Leaf)) { throw "Blender not found: $Blender" }

$BakeArguments = @(
    '--background', '--factory-startup', '--python', (Join-Path $PocRoot 'tools/bake_ch10032_scene.py'), '--',
    '--body', (Join-Path $PocRoot 'Assets/Characters/CH10032/SK_JZ_CH_10032_Body.FBX'),
    '--animation', $AnimationFbx,
    '--cloth', (Join-Path $PocRoot 'Assets/Meshes/CH10032_lower_sim.vcloth'),
    '--output-dir', $RuntimeRoot,
    '--motion', $Motion,
    '--duration', $Duration,
    '--fps', '30'
)
if ($StaticPose) { $BakeArguments += '--static-pose' }
& $Blender @BakeArguments
if ($LASTEXITCODE -ne 0) { throw "Blender scene bake failed with exit code $LASTEXITCODE" }

$Required = @('ch10032.vchar', "$Motion.vanim", 'ch10032_lower.vcloth2', 'scene.json')
foreach ($Name in $Required) {
    $Path = Join-Path $RuntimeRoot $Name
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf) -or (Get-Item -LiteralPath $Path).Length -eq 0) {
        throw "Blender did not generate a valid-looking asset: $Path"
    }
}

$Python = Join-Path $PocRoot '.venv/Scripts/python.exe'
& $Python (Join-Path $PocRoot 'tools/validate_real_assets.py') --asset-root $RuntimeRoot --motion $Motion
if ($LASTEXITCODE -ne 0) { throw 'Generated real-scene assets failed strict validation' }
if (-not $SkipFine15Golden) {
    $Model = Join-Path $PocRoot '.work/hood_data/fine15.vhood'
    $Checkpoint = Join-Path $PocRoot '.work/hood_data/trained_models/fine15.pth'
    if (-not (Test-Path -LiteralPath $Model -PathType Leaf) -or -not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
        throw 'Fine15 assets are missing. Run .\tools\fetch_hood_fine15.ps1 first, or use -SkipFine15Golden.'
    }
    & $Python (Join-Path $PocRoot 'tools/run_fine15_reference.py') --asset-root $RuntimeRoot --motion $Motion `
        --model $Model --checkpoint $Checkpoint --steps 100000 --golden (Join-Path $RuntimeRoot 'fine15_rollout.vhgold')
    if ($LASTEXITCODE -ne 0) { throw 'Fine15 Python golden rollout failed' }
}
Write-Host "CH10032 runtime assets ready: $RuntimeRoot"
