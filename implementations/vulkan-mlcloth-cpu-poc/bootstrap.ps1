[CmdletBinding()]
param([switch]$SkipFetch, [switch]$OverwriteWorkEdits)

$ErrorActionPreference = 'Stop'
$PocRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Lock = Get-Content -LiteralPath (Join-Path $PocRoot 'upstream.lock.json') -Raw | ConvertFrom-Json
$UpstreamRoot = Join-Path $PocRoot '.work/Vulkan'
function Invoke-Git([string[]]$Arguments, [string]$WorkingDirectory) {
    & git -C $WorkingDirectory @Arguments
    if ($LASTEXITCODE -ne 0) { throw "git failed in $WorkingDirectory" }
}
if (-not (Test-Path -LiteralPath $UpstreamRoot)) {
    if ($SkipFetch) { throw "Upstream checkout is missing: $UpstreamRoot" }
    New-Item -ItemType Directory -Force -Path $UpstreamRoot | Out-Null
    Invoke-Git @('init') $UpstreamRoot
    Invoke-Git @('remote', 'add', 'origin', $Lock.repository) $UpstreamRoot
    Invoke-Git @('fetch', '--depth', '1', 'origin', $Lock.commit) $UpstreamRoot
    Invoke-Git @('checkout', '--detach', $Lock.commit) $UpstreamRoot
}
$ActualCommit = (& git -C $UpstreamRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $ActualCommit -ne $Lock.commit) { throw "Upstream commit mismatch: expected $($Lock.commit), got $ActualCommit" }
$GlmRoot = Join-Path $UpstreamRoot 'external/glm'
if (-not (Test-Path -LiteralPath (Join-Path $GlmRoot 'glm'))) {
    if ($SkipFetch) { throw "GLM checkout is missing: $GlmRoot" }
    New-Item -ItemType Directory -Force -Path $GlmRoot | Out-Null
    Invoke-Git @('init') $GlmRoot
    Invoke-Git @('remote', 'add', 'origin', $Lock.submodules.'external/glm'.repository) $GlmRoot
    Invoke-Git @('fetch', '--depth', '1', 'origin', $Lock.submodules.'external/glm'.commit) $GlmRoot
    Invoke-Git @('checkout', '--detach', $Lock.submodules.'external/glm'.commit) $GlmRoot
}
$ActualGlmCommit = (& git -C $GlmRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $ActualGlmCommit -ne $Lock.submodules.'external/glm'.commit) { throw "GLM commit mismatch: expected $($Lock.submodules.'external/glm'.commit), got $ActualGlmCommit" }

$Destinations = @(
    @{ Source = Join-Path $PocRoot 'overlay/examples/mlclothcpu'; Destination = Join-Path $UpstreamRoot 'examples/mlclothcpu' },
    @{ Source = Join-Path $PocRoot 'overlay/shaders/hlsl/mlclothcpu'; Destination = Join-Path $UpstreamRoot 'shaders/hlsl/mlclothcpu' },
    @{ Source = Join-Path $PocRoot 'overlay/shaders/hlsl/mlclothcpu'; Destination = Join-Path $UpstreamRoot 'shaders/glsl/mlclothcpu' }
)
$Copies = [System.Collections.Generic.List[object]]::new()
foreach ($Entry in $Destinations) {
    if (-not (Test-Path -LiteralPath $Entry.Source -PathType Container)) { throw "Overlay is missing: $($Entry.Source)" }
    New-Item -ItemType Directory -Force -Path $Entry.Destination | Out-Null
    foreach ($File in Get-ChildItem -LiteralPath $Entry.Source -File) {
        if ($Entry.Destination -like '*shaders/glsl*' -and $File.Extension -ne '.spv') { continue }
        $Copies.Add([pscustomobject]@{ Source=$File.FullName; Destination=(Join-Path $Entry.Destination $File.Name) })
    }
}
foreach ($File in Get-ChildItem -LiteralPath (Join-Path $PocRoot 'include') -File) {
    $Copies.Add([pscustomobject]@{ Source=$File.FullName; Destination=(Join-Path $UpstreamRoot "examples/mlclothcpu/$($File.Name)") })
}
if (-not $OverwriteWorkEdits) {
    foreach ($Copy in $Copies) {
        if ((Test-Path -LiteralPath $Copy.Destination -PathType Leaf) -and (Get-Item -LiteralPath $Copy.Destination).LastWriteTimeUtc -gt (Get-Item -LiteralPath $Copy.Source).LastWriteTimeUtc.AddSeconds(2)) {
            throw "Refusing to overwrite a newer .work file: $($Copy.Destination). Move it back to overlay/include or pass -OverwriteWorkEdits."
        }
    }
}
foreach ($Copy in $Copies) { Copy-Item -LiteralPath $Copy.Source -Destination $Copy.Destination -Force }
$ExamplesCmake = Join-Path $UpstreamRoot 'examples/CMakeLists.txt'
$CmakeText = Get-Content -LiteralPath $ExamplesCmake -Raw
if ($CmakeText -notmatch '(?m)^\s*mlclothcpu\s*$') {
    $CmakeText = $CmakeText -replace '(?m)^(\s*computecloth\s*)$', "`$1`r`n`tmlclothcpu"
    Set-Content -LiteralPath $ExamplesCmake -Value $CmakeText -NoNewline
}
Write-Host "Upstream ready: $UpstreamRoot @ $ActualCommit"
Write-Host 'Overlay installed: examples/mlclothcpu and shaders/*/mlclothcpu'

