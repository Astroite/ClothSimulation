param(
    [ValidateSet('CH10032', 'HoodGrid64')]
    [string]$Scene = 'CH10032',
    [ValidateRange(0, 10000)]
    [int]$Warmup = 5,
    [ValidateRange(1, 10000)]
    [int]$Samples = 20,
    [ValidateSet('Fine15', 'TinyHood', 'Toy2L')]
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
$IsHoodGrid = $Scene -eq 'HoodGrid64'
if ($IsHoodGrid) {
    if ($Solver -eq 'Toy2L') { throw 'HoodGrid64 supports Fine15 or TinyHood' }
    $Motion = 'hood_grid64'
}
if (-not $AssetRoot) { $AssetRoot = Join-Path $PocRoot ($IsHoodGrid ? '.work/real_scene/hood_grid64' : ".work/real_scene/$Motion") }
$IsTiny = $Solver -eq 'TinyHood'
if (-not $HoodModel) { $HoodModel = Join-Path $PocRoot ($IsTiny ? '.work/hood_data/tinyhood64x4.vhood' : '.work/hood_data/fine15.vhood') }
$ResultStem = if ($IsHoodGrid) { $IsTiny ? 'tinyhood_grid64' : 'hood_grid64_fine15' } elseif ($IsTiny) { 'tinyhood_ch10032_tpose' } elseif ($Solver -eq 'Toy2L') { 'hood_static_toy2l' } else { 'hood_static' }
if (-not $Output) { $Output = Join-Path $PocRoot "results/${ResultStem}_timing.csv" }
$Output = [System.IO.Path]::GetFullPath($Output)
$StabilityOutput = [System.IO.Path]::GetFullPath((Join-Path $PocRoot "results/${ResultStem}_stability.json"))
$ValidationSource = Join-Path $UpstreamRoot 'validation_output.txt'
$ValidationOutput = Join-Path $PocRoot "results/${ResultStem}_validation_output.txt"
$AssetStem = $IsHoodGrid ? 'hood_grid64' : 'ch10032'

foreach ($Required in @(
    $Executable,
    (Join-Path $AssetRoot "$AssetStem.vchar"),
    (Join-Path $AssetRoot "$Motion.vanim"),
    (Join-Path $AssetRoot ($IsHoodGrid ? 'hood_grid64.vcloth2' : 'ch10032_lower.vcloth2')),
    $HoodModel
)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) { throw "Static benchmark asset is missing: $Required" }
}
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Output) | Out-Null
if (Test-Path -LiteralPath $Output -PathType Leaf) { Remove-Item -LiteralPath $Output -Force }
if (Test-Path -LiteralPath $StabilityOutput -PathType Leaf) { Remove-Item -LiteralPath $StabilityOutput -Force }
if (Test-Path -LiteralPath $ValidationSource -PathType Leaf) { Remove-Item -LiteralPath $ValidationSource -Force }

function Quote-ProcessArgument([string]$Value) { return '"{0}"' -f $Value }
$Arguments = @(
    '--scene', ($IsHoodGrid ? 'hoodgrid' : 'ch10032'), '--motion', $Motion, '--solver', ($Solver -eq 'Toy2L' ? 'toy2l' : ($IsTiny ? 'tinyhood' : 'fine15')),
    '--asset-root', (Quote-ProcessArgument $AssetRoot), '--hood-model', (Quote-ProcessArgument $HoodModel),
    '--hood-static-benchmark', '--hood-benchmark-warmup', $Warmup,
    '--hood-benchmark-samples', $Samples, '--hood-benchmark-output', (Quote-ProcessArgument $Output),
    '--hood-stability-output', (Quote-ProcessArgument $StabilityOutput),
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
if (-not (Test-Path -LiteralPath $StabilityOutput -PathType Leaf)) { throw "Static Fine15 benchmark did not produce $StabilityOutput" }
if (-not (Test-Path -LiteralPath $ValidationSource -PathType Leaf)) { throw 'Khronos validation did not produce a log file' }
Copy-Item -LiteralPath $ValidationSource -Destination $ValidationOutput -Force
if (Select-String -LiteralPath $ValidationOutput -Pattern 'VUID-|Validation Error|SYNC-HAZARD' -Quiet) {
    throw "Khronos validation reported an error: $ValidationOutput"
}

Write-Host "Wrote $Output"
Import-Csv -LiteralPath $Output |
    Where-Object { $_.stage -in @('skin', 'toy_layer0', 'toy_layer1_integrate', 'features_world', 'encoder_total', 'processor_4_total', 'processor_15_total', 'decoder_integrate', 'total') } |
    Format-Table stage, mean_ms, p95_ms, max_ms -AutoSize
Get-Content -LiteralPath $StabilityOutput
