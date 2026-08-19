param(
    [string]$Motion = 'ch10032_sprint',
    [ValidateSet('Fine15', 'PostCvpr', 'TinyHood')]
    [string]$Solver = 'Fine15',
    [string]$AssetRoot = '',
    [string]$HoodModel = '',
    # A retrained student has its own golden rollout and its own result file. Override both so
    # verifying it never overwrites the numbers backing an existing result document.
    [string]$Golden = '',
    [string]$Output = '',
    # Verify the Jacobi XPBD path against a golden produced with
    # `run_tinyhood_reference.py --xpbd-asset`. The iteration count and the .vxpbd file have to be
    # the ones that golden was generated with, or the comparison is meaningless.
    [switch]$Xpbd,
    [int]$XpbdIterations = 128,
    [string]$XpbdAsset = ''
)

$ErrorActionPreference = 'Stop'
$PocRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$UpstreamRoot = Join-Path $PocRoot '.work/Vulkan'
$Executable = Join-Path $UpstreamRoot 'build-gnn/bin/gnncloth.exe'
if (-not $AssetRoot) { $AssetRoot = Join-Path $PocRoot ".work/real_scene/$Motion" }
if (-not $HoodModel) { $HoodModel = Join-Path $PocRoot ($Solver -eq 'PostCvpr' ? '.work/hood_data/postcvpr.vhood' : ($Solver -eq 'TinyHood' ? '.work/hood_data/tinyhood64x4.vhood' : '.work/hood_data/fine15.vhood')) }
# The executable's working directory is the upstream tree, not the caller's, so resolve any
# caller-supplied relative path here. The defaults above are already absolute.
$AssetRoot = [System.IO.Path]::GetFullPath($AssetRoot)
$HoodModel = [System.IO.Path]::GetFullPath($HoodModel)
$Golden = $Golden ? [System.IO.Path]::GetFullPath($Golden) : (Join-Path $AssetRoot ($Solver -eq 'PostCvpr' ? 'postcvpr_rollout.vhgold' : ($Solver -eq 'TinyHood' ? 'tinyhood64x4_rollout.vhgold' : 'fine15_rollout.vhgold')))
$StaticPose = $Motion -eq 'ch10032_tpose'
$OutputSupplied = [bool]$Output
$Output = $OutputSupplied ? [System.IO.Path]::GetFullPath($Output) : (Join-Path $PocRoot ($Solver -eq 'PostCvpr' ? 'results/postcvpr_verify.json' : ($Solver -eq 'TinyHood' ? 'results/tinyhood_verify.json' : ($StaticPose ? 'results/hood_static_verify.json' : 'results/hood_verify.json'))))
$ValidationSource = Join-Path $UpstreamRoot 'validation_output.txt'
# The four default pairs are pre-existing names and one of them (hood_verify.json ->
# hood_validation_output.txt) does not follow the stem, so keep the explicit table and only
# derive a name when the caller redirected the result file.
$ValidationOutput = $OutputSupplied ? (($Output -replace '\.json$', '') + '_validation_output.txt')
    : (Join-Path $PocRoot ($Solver -eq 'PostCvpr' ? 'results/postcvpr_verify_validation_output.txt' : ($Solver -eq 'TinyHood' ? 'results/tinyhood_verify_validation_output.txt' : ($StaticPose ? 'results/hood_static_verify_validation_output.txt' : 'results/hood_validation_output.txt'))))
$RequiredAssets = @($Executable, $HoodModel, $Golden)
if ($Solver -eq 'PostCvpr') { $RequiredAssets += Join-Path $AssetRoot 'ch10032.postcvpr.vhier' }
if ($Xpbd) {
    if (-not $XpbdAsset) { $XpbdAsset = Join-Path $AssetRoot "$Motion.vxpbd" }
    $XpbdAsset = [System.IO.Path]::GetFullPath($XpbdAsset)
    $RequiredAssets += $XpbdAsset
}
foreach ($Required in $RequiredAssets) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) { throw "$Solver verification asset is missing: $Required" }
}
if (Test-Path -LiteralPath $Output -PathType Leaf) { Remove-Item -LiteralPath $Output -Force }
if (Test-Path -LiteralPath $ValidationSource -PathType Leaf) { Remove-Item -LiteralPath $ValidationSource -Force }

$Arguments = @(
    '--scene', 'ch10032', '--motion', $Motion, '--solver', ($Solver -eq 'PostCvpr' ? 'postcvpr' : ($Solver -eq 'TinyHood' ? 'tinyhood' : 'fine15')),
    '--asset-root', ('"{0}"' -f $AssetRoot), '--hood-model', ('"{0}"' -f $HoodModel),
    '--hood-verify', '--hood-golden', ('"{0}"' -f $Golden), '--hood-verify-output', ('"{0}"' -f $Output),
    '-v', '-vl', '-s', 'hlsl'
)
if ($Xpbd) {
    $Arguments += @('--hood-xpbd', '--hood-xpbd-iterations', $XpbdIterations,
        '--hood-xpbd-asset', ('"{0}"' -f $XpbdAsset))
}
$PreviousLayerEnables = $env:VK_LAYER_ENABLES
try {
    $env:VK_LAYER_ENABLES = 'VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT'
    $Process = Start-Process -FilePath $Executable -ArgumentList $Arguments -WorkingDirectory $UpstreamRoot -WindowStyle Hidden -Wait -PassThru
} finally {
    $env:VK_LAYER_ENABLES = $PreviousLayerEnables
}
if ($Process.ExitCode -ne 0) { throw "$Solver Vulkan verification failed with exit code $($Process.ExitCode)" }
if (-not (Test-Path -LiteralPath $Output -PathType Leaf)) { throw "$Solver Vulkan verification did not produce a result file" }
$Result = Get-Content -LiteralPath $Output -Raw | ConvertFrom-Json
if (-not $Result.passed) { throw "$Solver Vulkan verification exceeded its error thresholds: $Output" }
if (-not (Test-Path -LiteralPath $ValidationSource -PathType Leaf)) { throw 'Khronos validation did not produce a log file' }
Copy-Item -LiteralPath $ValidationSource -Destination $ValidationOutput -Force
if (Select-String -LiteralPath $ValidationOutput -Pattern 'VUID-|Validation Error|SYNC-HAZARD' -Quiet) {
    throw "Khronos validation reported an error: $ValidationOutput"
}
Write-Host "$Solver Vulkan verification passed: $Output"
