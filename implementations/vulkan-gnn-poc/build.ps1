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

# Ninja learns header dependencies by matching cl.exe's localised /showIncludes prefix. A
# build tree whose recorded msvc_deps_prefix does not byte-match what cl actually prints
# records no header dependencies at all, and then edits to fine15_gpu_layout.h,
# hood_runtime.inl or the .hlsli headers relink nothing while the build still reports
# success -- benchmarks silently measure the previous binary. Assert freshness directly
# rather than trusting the dependency scanner. Recover with:
#   Remove-Item .work/Vulkan/build-gnn/CMakeCache.txt
#   Remove-Item -Recurse .work/Vulkan/build-gnn/examples/CMakeFiles/gnncloth.dir
$Executable = Join-Path $PocRoot '.work/Vulkan/build-gnn/bin/gnncloth.exe'
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) { throw "Native build produced no executable: $Executable" }
$BuiltAt = (Get-Item -LiteralPath $Executable).LastWriteTimeUtc
$NewerSources = Get-ChildItem -LiteralPath (Join-Path $PocRoot '.work/Vulkan/examples/gnncloth') -File |
    Where-Object { $_.LastWriteTimeUtc -gt $BuiltAt }
if ($NewerSources) {
    $Listing = ($NewerSources | ForEach-Object { "  $($_.Name)  ($($_.LastWriteTime))" }) -join "`n"
    throw @"
gnncloth.exe ($($BuiltAt.ToLocalTime())) is older than its own sources, so the build did not
pick these up and any measurement against it is stale:
$Listing
Ninja is not tracking header dependencies in this build tree. Delete
.work/Vulkan/build-gnn/CMakeCache.txt and .work/Vulkan/build-gnn/examples/CMakeFiles/gnncloth.dir,
then rerun build.ps1.
"@
}

Write-Host "Built: $PocRoot\.work\Vulkan\build-gnn\bin\gnncloth.exe"
