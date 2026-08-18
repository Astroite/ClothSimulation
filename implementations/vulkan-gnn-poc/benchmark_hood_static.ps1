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
    [string]$Output = '',
    # Timing runs default to a clean device: no validation layers, no synchronization
    # validation. Both add per-dispatch CPU work that lets the GPU idle down between
    # the ~60 dispatches of a step, which inflates and destabilises every stage mean.
    # Correctness is verify_hood.ps1's job; pass -Validate only to time the validated path.
    [switch]$Validate,
    # Locked SM clock for reproducibility. The idle clock on this part is 735 MHz against
    # a 3105 MHz maximum, and a latency-bound step does not look like load to the clock
    # governor, so an unlocked device reports run-to-run means up to 2.2x apart on
    # identical work. 2700 MHz is the highest value this 4060 Ti holds exactly under this
    # workload (2900 falls back to 2745), and it matches what an unlocked run boosts to.
    # 0 disables locking (and restores nothing).
    [ValidateRange(0, 4000)]
    [int]$LockClockMHz = 2700
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
if ($Validate -and (Test-Path -LiteralPath $ValidationSource -PathType Leaf)) { Remove-Item -LiteralPath $ValidationSource -Force }

function Quote-ProcessArgument([string]$Value) { return '"{0}"' -f $Value }
$Arguments = @(
    '--scene', ($IsHoodGrid ? 'hoodgrid' : 'ch10032'), '--motion', $Motion, '--solver', ($Solver -eq 'Toy2L' ? 'toy2l' : ($IsTiny ? 'tinyhood' : 'fine15')),
    '--asset-root', (Quote-ProcessArgument $AssetRoot), '--hood-model', (Quote-ProcessArgument $HoodModel),
    '--hood-static-benchmark', '--hood-benchmark-warmup', $Warmup,
    '--hood-benchmark-samples', $Samples, '--hood-benchmark-output', (Quote-ProcessArgument $Output),
    '--hood-stability-output', (Quote-ProcessArgument $StabilityOutput),
    '-s', 'hlsl'
)
if ($Validate) { $Arguments += @('-v', '-vl') }

function Get-GpuState {
    $fields = 'clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,clocks_event_reasons.active'
    $row = (& nvidia-smi --query-gpu=$fields --format=csv,noheader 2>$null) | Select-Object -First 1
    if (-not $row) { return $null }
    $parts = $row -split ',\s*'
    return [ordered]@{ sm_clock = $parts[0]; memory_clock = $parts[1]; temperature_c = $parts[2]
        power_draw = $parts[3]; utilization = $parts[4]; throttle_reasons = $parts[5] }
}

$HasNvidiaSmi = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
$ClockLocked = $false
$PreviousLayerEnables = $env:VK_LAYER_ENABLES
try {
    if ($LockClockMHz -gt 0) {
        if (-not $HasNvidiaSmi) {
            Write-Warning 'nvidia-smi is unavailable; timing runs unlocked and stage means will be unstable'
        } else {
            & nvidia-smi -lgc "$LockClockMHz,$LockClockMHz" | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "Could not lock the SM clock to $LockClockMHz MHz" }
            $ClockLocked = $true
        }
    }
    $StateBefore = if ($HasNvidiaSmi) { Get-GpuState } else { $null }
    if ($Validate) { $env:VK_LAYER_ENABLES = 'VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT' }
    $Process = Start-Process -FilePath $Executable -ArgumentList $Arguments -WorkingDirectory $UpstreamRoot -WindowStyle Hidden -Wait -PassThru
    $StateAfter = if ($HasNvidiaSmi) { Get-GpuState } else { $null }
} finally {
    $env:VK_LAYER_ENABLES = $PreviousLayerEnables
    if ($ClockLocked) { & nvidia-smi -rgc | Out-Null }
}
if ($Process.ExitCode -ne 0) { throw "Static Fine15 benchmark failed with exit code $($Process.ExitCode)" }
if (-not (Test-Path -LiteralPath $Output -PathType Leaf)) { throw "Static Fine15 benchmark did not produce $Output" }
if (-not (Test-Path -LiteralPath $StabilityOutput -PathType Leaf)) { throw "Static Fine15 benchmark did not produce $StabilityOutput" }
if ($Validate) {
    if (-not (Test-Path -LiteralPath $ValidationSource -PathType Leaf)) { throw 'Khronos validation did not produce a log file' }
    Copy-Item -LiteralPath $ValidationSource -Destination $ValidationOutput -Force
    if (Select-String -LiteralPath $ValidationOutput -Pattern 'VUID-|Validation Error|SYNC-HAZARD' -Quiet) {
        throw "Khronos validation reported an error: $ValidationOutput"
    }
}

# A timing CSV is only comparable against another run measured the same way, so record
# how this one was measured next to it.
$EnvironmentOutput = if ($Output -match '_timing\.csv$') { $Output -replace '_timing\.csv$', '_environment.json' }
    else { [System.IO.Path]::ChangeExtension($Output, 'environment.json') }
[ordered]@{
    scene = $Scene; solver = $Solver; motion = $Motion; warmup = $Warmup; samples = $Samples
    validation_layers = [bool]$Validate
    synchronization_validation = [bool]$Validate
    locked_sm_clock_mhz = $ClockLocked ? $LockClockMHz : 0
    gpu_before = $StateBefore; gpu_after = $StateAfter
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $EnvironmentOutput -Encoding utf8

Write-Host "Wrote $Output"
if ($Validate) { Write-Host 'Measured WITH validation + synchronization validation: not a clean timing baseline' }
Write-Host ("SM clock: {0}" -f ($ClockLocked ? "locked $LockClockMHz MHz" : 'unlocked (stage means will be unstable)'))
# min_ms is the statistic that reproduces across runs; the mean absorbs clock excursions.
Import-Csv -LiteralPath $Output |
    Where-Object { $_.stage -in @('skin', 'toy_layer0', 'toy_layer1_integrate', 'features_world', 'encoder_total', 'processor_4_total', 'processor_15_total', 'decoder_integrate', 'total') } |
    Format-Table stage, min_ms, mean_ms, p95_ms, max_ms -AutoSize
Get-Content -LiteralPath $StabilityOutput
