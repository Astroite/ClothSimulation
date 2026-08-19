[CmdletBinding()]
param(
    [string]$Project = 'E:\Main\Projects\Z2Game\Z2Game.uproject',
    [string]$MeshAsset = '/Game/Developers/jinzhao/AICloth/CH_10032/Model/SK_JZ_CH_10032_Body.SK_JZ_CH_10032_Body',
    [string]$AnimAsset = '/Game/Developers/jinzhao/AICloth/CH_10032/Animation/04_Sprint/AS_C10032_ArmedSprint_Skirt.AS_C10032_ArmedSprint_Skirt',
    [string]$Output = '',
    [string]$Model = '',
    [string]$UnrealEditor = 'E:\Main\Engine\Binaries\Win64\UnrealEditor-Cmd.exe'
)

$ErrorActionPreference = 'Stop'
$PocRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Output) { $Output = Join-Path $PocRoot '.work/clips/AS_C10032_ArmedSprint_Skirt.mldrv' }
if (-not $Model) { $Model = Join-Path $PocRoot '.work/runtime/model_NeuralRes4_NeuralRes4_final.enc' }
$Project = [System.IO.Path]::GetFullPath($Project)
$Output = [System.IO.Path]::GetFullPath($Output)
$Model = [System.IO.Path]::GetFullPath($Model)
$UnrealEditor = [System.IO.Path]::GetFullPath($UnrealEditor)
$Script = [System.IO.Path]::GetFullPath((Join-Path $PocRoot 'tools/bake_driver_clip_unreal.py'))
foreach ($Required in @($Project, $Model, $UnrealEditor, $Script)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) { throw "Required bake input is missing: $Required" }
}
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Output) | Out-Null

$Previous = @{}
$Variables = [ordered]@{
    MLCLOTH_BAKE_MODEL = $Model
    MLCLOTH_BAKE_MESH = $MeshAsset
    MLCLOTH_BAKE_ANIM = $AnimAsset
    MLCLOTH_BAKE_OUTPUT = $Output
    MLCLOTH_BAKE_FPS = '30'
}
try {
    foreach ($Pair in $Variables.GetEnumerator()) {
        $Previous[$Pair.Key] = [Environment]::GetEnvironmentVariable($Pair.Key, 'Process')
        [Environment]::SetEnvironmentVariable($Pair.Key, $Pair.Value, 'Process')
    }
    # Z2Game's WITH_PAPER_GAME engine branch deliberately skips Python startup for
    # commandlets and FApp::IsUnattended(). ExecutePythonScript still marks the
    # script itself unattended and exits on completion, so omit the global flag.
    & $UnrealEditor $Project "-ExecutePythonScript=$Script" -ScriptErrorsAreFatal -ForceEnablePython -nop4 -nosplash -nullrhi -NoSound -stdout -FullStdOutLogOutput
    if ($LASTEXITCODE -ne 0) { throw "Unreal driver bake failed with exit code $LASTEXITCODE" }
} finally {
    foreach ($Pair in $Variables.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($Pair.Key, $Previous[$Pair.Key], 'Process')
    }
}
if (-not (Test-Path -LiteralPath $Output -PathType Leaf) -or (Get-Item -LiteralPath $Output).Length -le 144) {
    throw "Unreal did not produce a valid-looking MLDRV001 clip: $Output"
}
Write-Host "Baked read-only 30 Hz driver clip: $Output"
