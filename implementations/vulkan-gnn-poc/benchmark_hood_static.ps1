param(
    [ValidateRange(0, 10000)]
    [int]$Warmup = 5,
    [ValidateRange(1, 10000)]
    [int]$Samples = 20,
    [ValidateSet('Fine15', 'Toy2L')]
    [string]$Solver = 'Fine15',
    [string]$Motion = 'ch10032_tpose',
    [string]$AssetRoot = '',
    [string]$HoodModel = '',
    [string]$Output = ''
)

$ErrorActionPreference = 'Stop'
$PocRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$UpstreamRoot = Join-Path $PocRoot '.work/Vulkan'
$Executable = Join-Path $UpstreamRoot 'build-gnn/bin/gnncloth.exe'
if (-not $AssetRoot) { $AssetRoot = Join-Path $PocRoot ".work/real_scene/$Motion" }
if (-not $HoodModel) { $HoodModel = Join-Path $PocRoot '.work/hood_data/fine15.vhood' }
if (-not $Output) { $Output = Join-Path $PocRoot ($Solver -eq 'Toy2L' ? 'results/hood_static_toy2l_timing.csv' : 'results/hood_static_timing.csv') }
$Output = [System.IO.Path]::GetFullPath($Output)
$ValidationSource = Join-Path $UpstreamRoot 'validation_output.txt'
$ValidationOutput = Join-Path $PocRoot ($Solver -eq 'Toy2L' ? 'results/hood_static_toy2l_validation_output.txt' : 'results/hood_static_validation_output.txt')

foreach ($Required in @(
    $Executable,
    (Join-Path $AssetRoot 'ch10032.vchar'),
    (Join-Path $AssetRoot "$Motion.vanim"),
    (Join-Path $AssetRoot 'ch10032_lower.vcloth2'),
    $HoodModel
)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) { throw "Static benchmark asset is missing: $Required" }
}
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Output) | Out-Null
if (Test-Path -LiteralPath $Output -PathType Leaf) { Remove-Item -LiteralPath $Output -Force }
if (Test-Path -LiteralPath $ValidationSource -PathType Leaf) { Remove-Item -LiteralPath $ValidationSource -Force }

function Quote-ProcessArgument([string]$Value) { return '"{0}"' -f $Value }
$Arguments = @(
    '--scene', 'ch10032', '--motion', $Motion, '--solver', ($Solver -eq 'Toy2L' ? 'toy2l' : 'fine15'),
    '--asset-root', (Quote-ProcessArgument $AssetRoot), '--hood-model', (Quote-ProcessArgument $HoodModel),
    '--hood-static-benchmark', '--hood-benchmark-warmup', $Warmup,
    '--hood-benchmark-samples', $Samples, '--hood-benchmark-output', (Quote-ProcessArgument $Output),
    '-v', '-vl', '-s', 'hlsl'
)
$PreviousLayerEnables = $env:VK_LAYER_ENABLES
try {
    $env:VK_LAYER_ENABLES = 'VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT'
    $Process = Start-Process -FilePath $Executable -ArgumentList $Arguments -WorkingDirectory $UpstreamRoot -WindowStyle Hidden -Wait -PassThru
} finally {
    $env:VK_LAYER_ENABLES = $PreviousLayerEnables
}
if ($Process.ExitCode -ne 0) { throw "Static Fine15 benchmark failed with exit code $($Process.ExitCode)" }
if (-not (Test-Path -LiteralPath $Output -PathType Leaf)) { throw "Static Fine15 benchmark did not produce $Output" }
if (-not (Test-Path -LiteralPath $ValidationSource -PathType Leaf)) { throw 'Khronos validation did not produce a log file' }
Copy-Item -LiteralPath $ValidationSource -Destination $ValidationOutput -Force
if (Select-String -LiteralPath $ValidationOutput -Pattern 'VUID-|Validation Error|SYNC-HAZARD' -Quiet) {
    throw "Khronos validation reported an error: $ValidationOutput"
}

Write-Host "Wrote $Output"
Import-Csv -LiteralPath $Output |
    Where-Object { $_.stage -in @('skin', 'toy_layer0', 'toy_layer1_integrate', 'features_world', 'encoder_total', 'processor_15_total', 'decoder_integrate', 'total') } |
    Format-Table stage, mean_ms, p95_ms, max_ms -AutoSize
