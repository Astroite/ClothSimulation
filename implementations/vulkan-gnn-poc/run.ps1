param(
    [ValidateSet('Grid', 'CH10032')]
    [string]$Scene = 'Grid',
    [ValidateSet(16, 32, 64)]
    [int]$Grid = 32,
    [string]$Motion = 'ch10032_sprint',
    [ValidateSet('Toy', 'Fine15', 'Toy2L')]
    [string]$Solver = 'Toy',
    [string]$AssetRoot = '',
    [string]$HoodModel = '',
    [switch]$CollisionProjection,
    [switch]$StaticPose
)

$ErrorActionPreference = 'Stop'
$PocRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Motion = $StaticPose ? 'ch10032_tpose' : $Motion
if ($StaticPose) { $Scene = 'CH10032'; if ($Solver -eq 'Toy') { $Solver = 'Fine15' } }
$UpstreamRoot = Join-Path $PocRoot '.work/Vulkan'
$Executable = Join-Path $UpstreamRoot 'build-gnn/bin/gnncloth.exe'
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw "Executable is missing. Run .\build.ps1 first: $Executable"
}

$Arguments = @('-s', 'hlsl')
function Quote-ProcessArgument([string]$Value) { return '"{0}"' -f $Value }
if ($Scene -eq 'CH10032' -or $Solver -in @('Fine15', 'Toy2L')) {
    if (-not $AssetRoot) { $AssetRoot = Join-Path $PocRoot ".work/real_scene/$Motion" }
    if (-not $HoodModel) { $HoodModel = Join-Path $PocRoot '.work/hood_data/fine15.vhood' }
    foreach ($Required in @(
        (Join-Path $AssetRoot 'ch10032.vchar'),
        (Join-Path $AssetRoot "$Motion.vanim"),
        (Join-Path $AssetRoot 'ch10032_lower.vcloth2'),
        $HoodModel
    )) {
        if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
            throw "CH10032 runtime asset is missing: $Required`nRun .\tools\fetch_hood_fine15.ps1 and .\tools\bake_real_scene.ps1 first."
        }
    }
    $RuntimeSolver = $Solver -eq 'Toy2L' ? 'toy2l' : 'fine15'
    $Arguments += @('--scene', 'ch10032', '--motion', $Motion, '--solver', $RuntimeSolver, '--asset-root', (Quote-ProcessArgument $AssetRoot), '--hood-model', (Quote-ProcessArgument $HoodModel))
    if ($CollisionProjection) { $Arguments += '--hood-collision-projection' }
} else {
    $Arguments += @('--gnn-grid', $Grid)
}

Push-Location $UpstreamRoot
try {
    Start-Process -FilePath $Executable -ArgumentList $Arguments -WorkingDirectory $UpstreamRoot -Wait
} finally {
    Pop-Location
}
