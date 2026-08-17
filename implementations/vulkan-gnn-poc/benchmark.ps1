$ErrorActionPreference = 'Stop'
$PocRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$UpstreamRoot = Join-Path $PocRoot '.work/Vulkan'
$Executable = Join-Path $UpstreamRoot 'build-gnn/bin/gnncloth.exe'
$Results = Join-Path $PocRoot 'results'
New-Item -ItemType Directory -Force -Path $Results | Out-Null

$Rows = @()
foreach ($Grid in @(16, 32, 64)) {
    $Output = Join-Path $Results "benchmark_$Grid.csv"
    $Process = Start-Process -FilePath $Executable -ArgumentList @('--gnn-benchmark', '--gnn-grid', $Grid, '--gnn-benchmark-output', $Output, '-s', 'hlsl') -WorkingDirectory $UpstreamRoot -WindowStyle Hidden -Wait -PassThru
    if ($Process.ExitCode -ne 0) { throw "Grid $Grid benchmark exited with code $($Process.ExitCode)" }
    $Rows += Import-Csv -LiteralPath $Output
}
$Rows | Export-Csv -LiteralPath (Join-Path $Results 'gnn_benchmark.csv') -NoTypeInformation
Write-Host "Wrote $Results\gnn_benchmark.csv"
