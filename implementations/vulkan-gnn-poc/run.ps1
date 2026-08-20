param(
    [ValidateSet('Grid', 'CH10032', 'HoodGrid64')]
    [string]$Scene = 'Grid',
    [ValidateSet(16, 32, 64)]
    [int]$Grid = 32,
    [string]$Motion = 'ch10032_sprint',
    [ValidateSet('Toy', 'Fine15', 'PostCvpr', 'TinyHood', 'Toy2L')]
    [string]$Solver = 'Toy',
    [string]$AssetRoot = '',
    [string]$HoodModel = '',
    [switch]$CollisionProjection,
    # Jacobi XPBD after the network. Needs a .vxpbd next to the scene assets, from
    # tools/bake_xpbd_constraints.py. Also toggleable live in the overlay.
    [switch]$Xpbd,
    [ValidateRange(0, 1024)]
    [int]$XpbdIterations = 128,
    [string]$XpbdAsset = '',
    # Side-by-side A/B/C comparison: one animation driving network-only, constraints-only and hybrid
    # at once, with the animation and playback speed switchable live. Implies -Xpbd and needs the
    # garment-level .vxpbd rather than the per-motion one.
    [switch]$Compare,
    # Equal GPU budget, not equal iterations: C is 0.924 ms of network plus 128 sweeps, which buys the
    # constraints-only branch about 228. See results/RECOVERY_SPEED_RESULTS.md section 0.
    [ValidateRange(0, 1024)]
    [int]$XpbdIterationsB = 228,
    [ValidateRange(1, 4)]
    [int]$FrameStep = 1,
    [ValidateRange(0.5, 3.0)]
    [double]$CompareSpacing = 1.2,
    # Stop on the final frame instead of looping, so the post-motion settle is observable -- that is
    # where C's overshoot resolves and A's does not. Also toggleable live in the overlay.
    [switch]$HoldLastFrame,
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
if ($Scene -in @('CH10032', 'HoodGrid64') -or $Solver -in @('Fine15', 'PostCvpr', 'TinyHood', 'Toy2L')) {
    $IsHoodGrid = $Scene -eq 'HoodGrid64'
    if (-not $AssetRoot) { $AssetRoot = Join-Path $PocRoot ($IsHoodGrid ? '.work/real_scene/hood_grid64' : ".work/real_scene/$Motion") }
    if (-not $HoodModel) { $HoodModel = Join-Path $PocRoot ($Solver -eq 'PostCvpr' ? '.work/hood_data/postcvpr.vhood' : ($Solver -eq 'TinyHood' ? '.work/hood_data/tinyhood64x4.vhood' : '.work/hood_data/fine15.vhood')) }
    # The executable's working directory is the upstream tree, not the caller's, so resolve any
    # caller-supplied relative path here. The defaults above are already absolute.
    $AssetRoot = [System.IO.Path]::GetFullPath($AssetRoot)
    $HoodModel = [System.IO.Path]::GetFullPath($HoodModel)
    $AssetStem = $IsHoodGrid ? 'hood_grid64' : 'ch10032'
    $RequiredAssets = @(
        (Join-Path $AssetRoot "$AssetStem.vchar"),
        (Join-Path $AssetRoot "$Motion.vanim"),
        (Join-Path $AssetRoot ($IsHoodGrid ? 'hood_grid64.vcloth2' : 'ch10032_lower.vcloth2')),
        $HoodModel
    )
    if ($Solver -eq 'PostCvpr') { $RequiredAssets += Join-Path $AssetRoot "$AssetStem.postcvpr.vhier" }
    foreach ($Required in $RequiredAssets) {
        if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
            $BakeHint = $IsHoodGrid ? '.\tools\bake_hood_grid_scene.ps1' : '.\tools\bake_real_scene.ps1'
            $FetchHint = $Solver -eq 'PostCvpr' ? '.\tools\fetch_hood_postcvpr.ps1' : '.\tools\fetch_hood_fine15.ps1'
            throw "HOOD runtime asset is missing: $Required`nRun $FetchHint and $BakeHint first."
        }
    }
    $RuntimeSolver = $Solver -eq 'Toy2L' ? 'toy2l' : ($Solver -eq 'PostCvpr' ? 'postcvpr' : ($Solver -eq 'TinyHood' ? 'tinyhood' : 'fine15'))
    $Arguments += @('--scene', ($IsHoodGrid ? 'hoodgrid' : 'ch10032'), '--motion', $Motion, '--solver', $RuntimeSolver, '--asset-root', (Quote-ProcessArgument $AssetRoot), '--hood-model', (Quote-ProcessArgument $HoodModel))
    if ($CollisionProjection) { $Arguments += '--hood-collision-projection' }
    if ($Compare) {
        # Comparison mode reads the garment-level constraint set, not the per-motion one, so one
        # calibration is shared by every clip the Animation dropdown offers. results/GATE_G0_RESULTS.md
        # section 11 measured that calibration as transferable across motions.
        if (-not $XpbdAsset) { $XpbdAsset = Join-Path (Split-Path -Parent $AssetRoot) ($IsHoodGrid ? 'hood_grid64.vxpbd' : 'ch10032_lower.vxpbd') }
        $XpbdAsset = [System.IO.Path]::GetFullPath($XpbdAsset)
        if (-not (Test-Path -LiteralPath $XpbdAsset -PathType Leaf)) {
            throw "Comparison mode needs a garment-level XPBD asset: $XpbdAsset`nBake it with .\.venv\Scripts\python.exe -B tools\bake_xpbd_constraints.py --scene sprint_start --output $XpbdAsset"
        }
        $Arguments += @('--hood-compare', '--hood-xpbd-iterations', $XpbdIterations,
            '--hood-xpbd-iterations-b', $XpbdIterationsB, '--hood-frame-step', $FrameStep,
            '--hood-compare-spacing', $CompareSpacing, '--hood-xpbd-asset', (Quote-ProcessArgument $XpbdAsset))
        if ($HoldLastFrame) { $Arguments += '--hood-hold-last-frame' }
    } elseif ($Xpbd) {
        if (-not $XpbdAsset) { $XpbdAsset = Join-Path $AssetRoot "$Motion.vxpbd" }
        $XpbdAsset = [System.IO.Path]::GetFullPath($XpbdAsset)
        if (-not (Test-Path -LiteralPath $XpbdAsset -PathType Leaf)) {
            throw "XPBD asset is missing: $XpbdAsset`nRun .\.venv\Scripts\python.exe -B tools\bake_xpbd_constraints.py --scene $Motion first."
        }
        $Arguments += @('--hood-xpbd', '--hood-xpbd-iterations', $XpbdIterations,
            '--hood-xpbd-asset', (Quote-ProcessArgument $XpbdAsset))
    }
} else {
    $Arguments += @('--gnn-grid', $Grid)
}

Push-Location $UpstreamRoot
try {
    Start-Process -FilePath $Executable -ArgumentList $Arguments -WorkingDirectory $UpstreamRoot -Wait
} finally {
    Pop-Location
}
