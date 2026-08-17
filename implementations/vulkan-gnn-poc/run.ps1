param(
    [ValidateSet(16, 32, 64)]
    [int]$Grid = 32
)

$ErrorActionPreference = 'Stop'
$PocRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$UpstreamRoot = Join-Path $PocRoot '.work/Vulkan'
Push-Location $UpstreamRoot
try {
    Start-Process -FilePath (Join-Path $UpstreamRoot 'build-gnn/bin/gnncloth.exe') -ArgumentList @('--gnn-grid', $Grid, '-s', 'hlsl') -WorkingDirectory $UpstreamRoot -Wait
} finally {
    Pop-Location
}
