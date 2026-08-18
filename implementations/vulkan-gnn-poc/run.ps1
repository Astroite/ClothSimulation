param(
    [ValidateSet('Grid', 'CH10032', 'HoodGrid64')]
    [string]$Scene = 'Grid',
    [ValidateSet(16, 32, 64)]
    [int]$Grid = 32,
    [string]$Motion = 'ch10032_sprint',
    [ValidateSet('Toy', 'Fine15', 'TinyHood', 'Toy2L')]
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
if ($Scene -eq 'HoodGrid64') { $Motion = 'hood_grid64'; $Grid = 64; if ($Solver -eq 'Toy') { $Solver = 'Fine15' } }
$UpstreamRoot = Join-Path $PocRoot '.work/Vulkan'
$Executable = Join-Path $UpstreamRoot 'build-gnn/bin/gnncloth.exe'
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw "Executable is missing. Run .\build.ps1 first: $Executable"
}

$Arguments = @('-s', 'hlsl')
function Quote-ProcessArgument([string]$Value) { return '"{0}"' -f $Value }
if ($Scene -in @('CH10032', 'HoodGrid64') -or $Solver -in @('Fine15', 'TinyHood', 'Toy2L')) {
    $IsHoodGrid = $Scene -eq 'HoodGrid64'
    if (-not $AssetRoot) { $AssetRoot = Join-Path $PocRoot ($IsHoodGrid ? '.work/real_scene/hood_grid64' : ".work/real_scene/$Motion") }
    if (-not $HoodModel) { $HoodModel = Join-Path $PocRoot ($Solver -eq 'TinyHood' ? '.work/hood_data/tinyhood64x4.vhood' : '.work/hood_data/fine15.vhood') }
    $AssetStem = $IsHoodGrid ? 'hood_grid64' : 'ch10032'
    foreach ($Required in @(
        (Join-Path $AssetRoot "$AssetStem.vchar"),
        (Join-Path $AssetRoot "$Motion.vanim"),
        (Join-Path $AssetRoot ($IsHoodGrid ? 'hood_grid64.vcloth2' : 'ch10032_lower.vcloth2')),
        $HoodModel
    )) {
        if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
            $BakeHint = $IsHoodGrid ? '.\tools\bake_hood_grid_scene.ps1' : '.\tools\bake_real_scene.ps1'
            throw "HOOD runtime asset is missing: $Required`nRun .\tools\fetch_hood_fine15.ps1 and $BakeHint first."
        }
    }
    $RuntimeSolver = $Solver -eq 'Toy2L' ? 'toy2l' : ($Solver -eq 'TinyHood' ? 'tinyhood' : 'fine15')
    $Arguments += @('--scene', ($IsHoodGrid ? 'hoodgrid' : 'ch10032'), '--motion', $Motion, '--solver', $RuntimeSolver, '--asset-root', (Quote-ProcessArgument $AssetRoot), '--hood-model', (Quote-ProcessArgument $HoodModel))
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
