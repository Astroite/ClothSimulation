[CmdletBinding()]
param(
    [string]$RuntimeDir = '', [string]$Model = '', [string]$Clip = '',
    [ValidateSet(1, 2, 4, 8)][int]$Threads = 1,
    [switch]$Verify, [switch]$Benchmark, [int]$Frames = 0, [switch]$Validation
)
$ErrorActionPreference = 'Stop'
$PocRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $RuntimeDir) { $RuntimeDir = Join-Path $PocRoot '.work/runtime' }
if (-not $Model) { $Model = Join-Path $RuntimeDir 'model_NeuralRes4_NeuralRes4_final.enc' }
if (-not $Clip) { $Clip = Join-Path $PocRoot '.work/clips/AS_C10032_ArmedSprint_Skirt.mldrv' }
$RuntimeDir = [System.IO.Path]::GetFullPath($RuntimeDir)
$Model = [System.IO.Path]::GetFullPath($Model)
$Clip = [System.IO.Path]::GetFullPath($Clip)
$Executable = Join-Path $PocRoot '.work/Vulkan/build-mlclothcpu/bin/mlclothcpu.exe'
$WorkingDirectory = Join-Path $PocRoot '.work/Vulkan'
$BenchmarkOutput = Join-Path $WorkingDirectory 'mlcloth_benchmark.csv'
$ValidationLog = Join-Path $WorkingDirectory 'validation_output.txt'
foreach ($Required in @($Executable, $Model, $Clip, (Join-Path $RuntimeDir 'AILab.dll'))) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) { throw "Required file is missing: $Required" }
}
$Arguments = @('-s', 'hlsl', '--runtime-dir', $RuntimeDir, '--model', $Model, '--clip', $Clip, '--threads', "$Threads")
if ($Verify) { $Arguments += '--verify' }
if ($Benchmark) { $Arguments += @('--benchmark', '--benchmark-output', $BenchmarkOutput) }
if ($Frames -gt 0) { $Arguments += @('--frames', "$Frames") }
if ($Validation) { $Arguments += @('-v', '-vl', '--sync-validation') }
if ($Validation -and (Test-Path -LiteralPath $ValidationLog -PathType Leaf)) {
    Remove-Item -LiteralPath $ValidationLog -Force
}
if ($Benchmark -and (Test-Path -LiteralPath $BenchmarkOutput -PathType Leaf)) {
    Remove-Item -LiteralPath $BenchmarkOutput -Force
}
Push-Location $WorkingDirectory
try {
    # A native GUI-subsystem executable is asynchronous under PowerShell unless
    # it participates in a pipeline. Out-Host preserves live output and waits.
    & $Executable @Arguments 2>&1 | Out-Host
    $ExitCode = $LASTEXITCODE
    if ($null -eq $ExitCode -or $ExitCode -ne 0) { throw "mlclothcpu exited with code $ExitCode" }
    if ($Validation -and (Test-Path -LiteralPath $ValidationLog -PathType Leaf)) {
        $ValidationFailures = Select-String -LiteralPath $ValidationLog -Pattern 'WARNING:|ERROR:'
        if ($ValidationFailures) {
            Get-Content -LiteralPath $ValidationLog | Out-Host
            throw "Vulkan validation emitted warning/error messages: $ValidationLog"
        }
    }
    if ($Benchmark) {
        if (-not (Test-Path -LiteralPath $BenchmarkOutput -PathType Leaf)) {
            throw "Benchmark did not produce a CSV (sample shortfall or write failure): $BenchmarkOutput"
        }
        $Rows = @(Import-Csv -LiteralPath $BenchmarkOutput)
        if ($Rows.Count -ne 1 -or [int]$Rows[0].samples -ne 1000) {
            throw "Benchmark CSV must contain exactly one 1,000-sample result: $BenchmarkOutput"
        }
    }
} finally { Pop-Location }
