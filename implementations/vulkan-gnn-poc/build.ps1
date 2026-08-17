$ErrorActionPreference = 'Stop'
$PocRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

$VenvPython = Join-Path $PocRoot '.venv/Scripts/python.exe'
if (Test-Path -LiteralPath $VenvPython) {
    & $VenvPython (Join-Path $PocRoot 'tools/compile_shaders.py')
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 (Join-Path $PocRoot 'tools/compile_shaders.py')
} else {
    & python (Join-Path $PocRoot 'tools/compile_shaders.py')
}
if ($LASTEXITCODE -ne 0) { throw 'Shader compilation failed' }

& (Join-Path $PocRoot 'bootstrap.ps1') -SkipFetch
if ($LASTEXITCODE -ne 0) { throw 'Bootstrap failed' }

& (Join-Path $PocRoot 'tools/build-with-vs.cmd')
if ($LASTEXITCODE -ne 0) { throw 'Native build failed' }

Write-Host "Built: $PocRoot\.work\Vulkan\build-gnn\bin\gnncloth.exe"
