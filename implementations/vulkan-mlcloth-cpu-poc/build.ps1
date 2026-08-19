$ErrorActionPreference = 'Stop'
$PocRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (Get-Command py -ErrorAction SilentlyContinue) { & py -3 (Join-Path $PocRoot 'tools/compile_shaders.py') } else { & python (Join-Path $PocRoot 'tools/compile_shaders.py') }
if ($LASTEXITCODE -ne 0) { throw 'Shader compilation or spirv-val failed' }
& (Join-Path $PocRoot 'bootstrap.ps1')
if ($LASTEXITCODE -ne 0) { throw 'Bootstrap failed' }
& (Join-Path $PocRoot 'tools/build-with-vs.cmd')
if ($LASTEXITCODE -ne 0) { throw 'Native build or unit tests failed' }
$Executable = Join-Path $PocRoot '.work/Vulkan/build-mlclothcpu/bin/mlclothcpu.exe'
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) { throw "Build produced no executable: $Executable" }
$BuiltAt = (Get-Item -LiteralPath $Executable).LastWriteTimeUtc
$NewerSources = foreach ($Root in @((Join-Path $PocRoot 'overlay/examples/mlclothcpu'), (Join-Path $PocRoot 'include'))) {
    Get-ChildItem -LiteralPath $Root -File | Where-Object { $_.LastWriteTimeUtc -gt $BuiltAt }
}
if ($NewerSources) { throw "mlclothcpu.exe is stale relative to: $($NewerSources.Name -join ', ')" }
Write-Host "Built and tested: $Executable"

