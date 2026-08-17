$ErrorActionPreference = 'Stop'
$PocRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$UpstreamRoot = Join-Path $PocRoot '.work/Vulkan'
$Executable = Join-Path $UpstreamRoot 'build-gnn/bin/gnncloth.exe'
$Python = Join-Path $PocRoot '.venv/Scripts/python.exe'
$Results = Join-Path $PocRoot 'results'
New-Item -ItemType Directory -Force -Path $Results | Out-Null

& $Python (Join-Path $PocRoot 'model/verify_export.py')
if ($LASTEXITCODE -ne 0) { throw 'Python reference verification failed' }

& (Join-Path $PocRoot 'tests/build/vgnn_format_test.exe') (Join-Path $PocRoot 'model/artifacts/model.bin') (Join-Path $PocRoot 'model/artifacts/golden.bin')
if ($LASTEXITCODE -ne 0) { throw 'C++ binary-format verification failed' }

$ValidationLog = Join-Path $UpstreamRoot 'validation_output.txt'
if (Test-Path -LiteralPath $ValidationLog) { Clear-Content -LiteralPath $ValidationLog }
$PreviousLayerEnables = $env:VK_LAYER_ENABLES
try {
    $env:VK_LAYER_ENABLES = 'VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT'
    $Process = Start-Process -FilePath $Executable -ArgumentList @('--gnn-verify', '-s', 'hlsl', '-v', '-vl') -WorkingDirectory $UpstreamRoot -WindowStyle Hidden -Wait -PassThru
    if ($Process.ExitCode -ne 0) { throw "Vulkan verification exited with code $($Process.ExitCode)" }
    & (Join-Path $PocRoot 'smoke_modes.ps1')
} finally {
    $env:VK_LAYER_ENABLES = $PreviousLayerEnables
}

$Report = Get-Content -LiteralPath (Join-Path $UpstreamRoot 'gnn_verify.json') -Raw | ConvertFrom-Json
if (-not $Report.passed) { throw 'Vulkan numerical/stability verification failed' }
$ValidationText = Get-Content -LiteralPath $ValidationLog -Raw
if ($ValidationText -match '(?m): (ERROR|WARNING):') { throw 'Vulkan validation emitted an error or warning' }
Copy-Item -LiteralPath (Join-Path $UpstreamRoot 'gnn_verify.json') -Destination (Join-Path $Results 'gnn_verify.json') -Force
Copy-Item -LiteralPath $ValidationLog -Destination (Join-Path $Results 'validation_output.txt') -Force
Write-Host 'Python, format, Vulkan numerical, 1200-frame stability, reset replay, both solver modes, and synchronization validation passed.'
