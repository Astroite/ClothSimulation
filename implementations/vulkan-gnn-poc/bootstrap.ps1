param(
    [switch]$SkipFetch,
    # The overlay is the only copy of the sample and shaders; this script copies
    # it one way into .work. Pass this switch to discard newer edits made
    # directly inside .work instead of failing on them.
    [switch]$OverwriteWorkEdits
)

$ErrorActionPreference = 'Stop'
$PocRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Lock = Get-Content -LiteralPath (Join-Path $PocRoot 'upstream.lock.json') -Raw | ConvertFrom-Json
$WorkRoot = Join-Path $PocRoot '.work'
$UpstreamRoot = Join-Path $WorkRoot 'Vulkan'

function Invoke-Git {
    param([string[]]$Arguments, [string]$WorkingDirectory)
    & git -C $WorkingDirectory @Arguments
    if ($LASTEXITCODE -ne 0) { throw "git failed in $WorkingDirectory" }
}

if (-not (Test-Path -LiteralPath $UpstreamRoot)) {
    if ($SkipFetch) { throw "Upstream checkout is missing: $UpstreamRoot" }
    New-Item -ItemType Directory -Force -Path $UpstreamRoot | Out-Null
    Invoke-Git -Arguments @('init') -WorkingDirectory $UpstreamRoot
    Invoke-Git -Arguments @('remote', 'add', 'origin', $Lock.repository) -WorkingDirectory $UpstreamRoot
    Invoke-Git -Arguments @('fetch', '--depth', '1', 'origin', $Lock.commit) -WorkingDirectory $UpstreamRoot
    Invoke-Git -Arguments @('checkout', '--detach', $Lock.commit) -WorkingDirectory $UpstreamRoot
}

$ActualCommit = (& git -C $UpstreamRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $ActualCommit -ne $Lock.commit) {
    throw "Upstream commit mismatch: expected $($Lock.commit), got $ActualCommit"
}

$GlmRoot = Join-Path $UpstreamRoot 'external/glm'
if (-not (Test-Path -LiteralPath (Join-Path $GlmRoot 'glm'))) {
    if ($SkipFetch) { throw "GLM checkout is missing: $GlmRoot" }
    if (-not (Test-Path -LiteralPath $GlmRoot)) { New-Item -ItemType Directory -Force -Path $GlmRoot | Out-Null }
    Invoke-Git -Arguments @('init') -WorkingDirectory $GlmRoot
    Invoke-Git -Arguments @('remote', 'add', 'origin', $Lock.submodules.'external/glm'.repository) -WorkingDirectory $GlmRoot
    Invoke-Git -Arguments @('fetch', '--depth', '1', 'origin', $Lock.submodules.'external/glm'.commit) -WorkingDirectory $GlmRoot
    Invoke-Git -Arguments @('checkout', '--detach', $Lock.submodules.'external/glm'.commit) -WorkingDirectory $GlmRoot
}

$ActualGlmCommit = (& git -C $GlmRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $ActualGlmCommit -ne $Lock.submodules.'external/glm'.commit) {
    throw "GLM commit mismatch: expected $($Lock.submodules.'external/glm'.commit), got $ActualGlmCommit"
}

$ExampleDestination = Join-Path $UpstreamRoot 'examples/gnncloth'
$HlslDestination = Join-Path $UpstreamRoot 'shaders/hlsl/gnncloth'
$GlslDestination = Join-Path $UpstreamRoot 'shaders/glsl/gnncloth'
New-Item -ItemType Directory -Force -Path $ExampleDestination, $HlslDestination, $GlslDestination | Out-Null

# Enumerate every authoritative source and where it lands, then verify none of
# the destinations carries newer edits before copying anything. Copy-Item
# preserves LastWriteTime, so a freshly bootstrapped tree compares equal; only a
# real edit inside .work makes a destination newer. The tolerance absorbs
# timestamp granularity differences between volumes.
$OverlayShaderRoot = Join-Path $PocRoot 'overlay/shaders/hlsl/gnncloth'
$Copies = [System.Collections.Generic.List[object]]::new()
function Add-Copy {
    param([string]$Source, [string]$DestinationDirectory)
    $Copies.Add([pscustomobject]@{
        Source      = $Source
        Destination = Join-Path $DestinationDirectory (Split-Path -Leaf $Source)
    })
}

Add-Copy (Join-Path $PocRoot 'overlay/examples/gnncloth/gnncloth.cpp') $ExampleDestination
Add-Copy (Join-Path $PocRoot 'overlay/examples/gnncloth/vgnn_format.h') $ExampleDestination
foreach ($File in Get-ChildItem -LiteralPath $OverlayShaderRoot -File) {
    Add-Copy $File.FullName $HlslDestination
    if ($File.Extension -eq '.spv') { Add-Copy $File.FullName $GlslDestination }
}
foreach ($Artifact in @('model.bin', 'golden.bin')) {
    Add-Copy (Join-Path $PocRoot "model/artifacts/$Artifact") $HlslDestination
    Add-Copy (Join-Path $PocRoot "model/artifacts/$Artifact") $GlslDestination
}

if (-not $OverwriteWorkEdits) {
    $Tolerance = [TimeSpan]::FromSeconds(2)
    $Stale = foreach ($Copy in $Copies) {
        if (-not (Test-Path -LiteralPath $Copy.Destination -PathType Leaf)) { continue }
        $SourceTime = (Get-Item -LiteralPath $Copy.Source).LastWriteTimeUtc
        $DestinationTime = (Get-Item -LiteralPath $Copy.Destination).LastWriteTimeUtc
        if (($DestinationTime - $SourceTime) -gt $Tolerance) {
            [pscustomobject]@{ Path = $Copy.Destination; Newer = $DestinationTime; Overlay = $SourceTime }
        }
    }
    if ($Stale) {
        $Detail = ($Stale | ForEach-Object {
            "  $($_.Path)`n    .work: $($_.Newer.ToLocalTime())  overlay: $($_.Overlay.ToLocalTime())"
        }) -join "`n"
        throw @"
Refusing to overwrite newer edits inside .work. These destinations are newer
than their overlay sources, so bootstrap would silently discard them:
$Detail

Copy the changes back into overlay/ (the tracked, authoritative copy), or rerun
with -OverwriteWorkEdits to discard them.
"@
    }
}

foreach ($Copy in $Copies) {
    Copy-Item -LiteralPath $Copy.Source -Destination $Copy.Destination -Force
}

$ExamplesCmake = Join-Path $UpstreamRoot 'examples/CMakeLists.txt'
$CmakeText = Get-Content -LiteralPath $ExamplesCmake -Raw
if ($CmakeText -notmatch '(?m)^\s*gnncloth\s*$') {
    $CmakeText = $CmakeText -replace "(?m)^(\s*computecloth\s*)$", "`$1`r`n`tgnncloth"
    Set-Content -LiteralPath $ExamplesCmake -Value $CmakeText -NoNewline
}

Write-Host "Upstream ready: $UpstreamRoot @ $ActualCommit"
Write-Host "Overlay installed: examples/gnncloth and shaders/*/gnncloth"
