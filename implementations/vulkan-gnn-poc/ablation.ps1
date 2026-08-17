$ErrorActionPreference = 'Stop'
$PocRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$UpstreamRoot = Join-Path $PocRoot '.work/Vulkan'
$Executable = Join-Path $UpstreamRoot 'build-gnn/bin/gnncloth.exe'
$Python = Join-Path $PocRoot '.venv/Scripts/python.exe'
$Results = Join-Path $PocRoot 'results'
New-Item -ItemType Directory -Force -Path $Results | Out-Null

# Same deterministic scenario for each mode, changing only where the acceleration
# comes from. 'gravity' drops the neighbour coupling but keeps gravity, so it, not
# 'zero', is the baseline that isolates what the graph structure contributes.
$Frames = 600
$Dumps = @{}
foreach ($Mode in @('gnn', 'analytic', 'gravity', 'zero')) {
    $Dump = Join-Path $Results "ablation_$Mode.bin"
    $Dumps[$Mode] = $Dump
    $Process = Start-Process -FilePath $Executable -ArgumentList @(
        '--gnn-ablate', $Mode,
        '--gnn-ablate-dump', $Dump,
        '--gnn-ablate-frames', $Frames,
        '--gnn-grid', '32',
        '-s', 'hlsl'
    ) -WorkingDirectory $UpstreamRoot -WindowStyle Hidden -Wait -PassThru
    if ($Process.ExitCode -ne 0) { throw "Ablation run '$Mode' exited with code $($Process.ExitCode)" }
    if (-not (Test-Path -LiteralPath $Dump)) { throw "Ablation run '$Mode' produced no dump at $Dump" }
    Write-Host "ablation '$Mode' -> $Dump"
}

& $Python (Join-Path $PocRoot 'model/compare_ablation.py') `
    --gnn $Dumps['gnn'] `
    --analytic $Dumps['analytic'] `
    --gravity $Dumps['gravity'] `
    --zero $Dumps['zero'] `
    --output (Join-Path $Results 'gnn_ablation.json')
if ($LASTEXITCODE -ne 0) { throw 'Ablation comparison failed' }
