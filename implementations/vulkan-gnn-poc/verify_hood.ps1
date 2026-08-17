param(
    [string]$Motion = 'ch10032_sprint',
    [string]$AssetRoot = '',
    [string]$HoodModel = ''
)

$ErrorActionPreference = 'Stop'
$PocRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$UpstreamRoot = Join-Path $PocRoot '.work/Vulkan'
$Executable = Join-Path $UpstreamRoot 'build-gnn/bin/gnncloth.exe'
if (-not $AssetRoot) { $AssetRoot = Join-Path $PocRoot ".work/real_scene/$Motion" }
if (-not $HoodModel) { $HoodModel = Join-Path $PocRoot '.work/hood_data/fine15.vhood' }
$Golden = Join-Path $AssetRoot 'fine15_rollout.vhgold'
$StaticPose = $Motion -eq 'ch10032_tpose'
$Output = Join-Path $PocRoot ($StaticPose ? 'results/hood_static_verify.json' : 'results/hood_verify.json')
$ValidationSource = Join-Path $UpstreamRoot 'validation_output.txt'
$ValidationOutput = Join-Path $PocRoot ($StaticPose ? 'results/hood_static_verify_validation_output.txt' : 'results/hood_validation_output.txt')
if (Test-Path -LiteralPath $Output -PathType Leaf) { Remove-Item -LiteralPath $Output -Force }
if (Test-Path -LiteralPath $ValidationSource -PathType Leaf) { Remove-Item -LiteralPath $ValidationSource -Force }

$Arguments = @(
    '--scene', 'ch10032', '--motion', $Motion, '--solver', 'fine15',
    '--asset-root', ('"{0}"' -f $AssetRoot), '--hood-model', ('"{0}"' -f $HoodModel),
    '--hood-verify', '--hood-golden', ('"{0}"' -f $Golden), '--hood-verify-output', ('"{0}"' -f $Output),
    '-v', '-vl', '-s', 'hlsl'
)
$PreviousLayerEnables = $env:VK_LAYER_ENABLES
try {
    $env:VK_LAYER_ENABLES = 'VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT'
    $Process = Start-Process -FilePath $Executable -ArgumentList $Arguments -WorkingDirectory $UpstreamRoot -WindowStyle Hidden -Wait -PassThru
} finally {
    $env:VK_LAYER_ENABLES = $PreviousLayerEnables
}
if ($Process.ExitCode -ne 0) { throw "Fine15 Vulkan verification failed with exit code $($Process.ExitCode)" }
if (-not (Test-Path -LiteralPath $Output -PathType Leaf)) { throw 'Fine15 Vulkan verification did not produce a result file' }
$Result = Get-Content -LiteralPath $Output -Raw | ConvertFrom-Json
if (-not $Result.passed) { throw "Fine15 Vulkan verification exceeded its error thresholds: $Output" }
if (-not (Test-Path -LiteralPath $ValidationSource -PathType Leaf)) { throw 'Khronos validation did not produce a log file' }
Copy-Item -LiteralPath $ValidationSource -Destination $ValidationOutput -Force
if (Select-String -LiteralPath $ValidationOutput -Pattern 'VUID-|Validation Error|SYNC-HAZARD' -Quiet) {
    throw "Khronos validation reported an error: $ValidationOutput"
}
Write-Host "Fine15 Vulkan verification passed: $Output"
