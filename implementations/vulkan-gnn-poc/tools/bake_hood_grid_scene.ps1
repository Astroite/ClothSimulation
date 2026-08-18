param(
    [ValidateSet(64)]
    [int]$Grid = 64,
    [string]$Output = ''
)

$ErrorActionPreference = 'Stop'
$PocRoot = Split-Path -Parent $PSScriptRoot
if (-not $Output) { $Output = Join-Path $PocRoot '.work/real_scene/hood_grid64' }
$Python = Join-Path $PocRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment is missing. Run .\bootstrap.ps1 first: $Python"
}

& $Python (Join-Path $PocRoot 'tools/bake_hood_grid_scene.py') --output-dir $Output --grid $Grid
if ($LASTEXITCODE -ne 0) { throw "HOOD grid scene bake failed with exit code $LASTEXITCODE" }

foreach ($Required in @('hood_grid64.vchar', 'hood_grid64.vanim', 'hood_grid64.vcloth2', 'scene.json')) {
    $Path = Join-Path $Output $Required
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf) -or (Get-Item -LiteralPath $Path).Length -eq 0) {
        throw "HOOD grid scene asset is missing: $Path"
    }
}
Write-Host "HOOD grid scene ready: $Output"
