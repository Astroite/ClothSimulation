param(
    [switch]$SkipFetch
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
Copy-Item -LiteralPath (Join-Path $PocRoot 'overlay/examples/gnncloth/gnncloth.cpp') -Destination $ExampleDestination -Force
Copy-Item -LiteralPath (Join-Path $PocRoot 'overlay/examples/gnncloth/vgnn_format.h') -Destination $ExampleDestination -Force
Copy-Item -Path (Join-Path $PocRoot 'overlay/shaders/hlsl/gnncloth/*') -Destination $HlslDestination -Force

Get-ChildItem -LiteralPath $HlslDestination -Filter '*.spv' | Copy-Item -Destination $GlslDestination -Force
Copy-Item -LiteralPath (Join-Path $PocRoot 'model/artifacts/model.bin') -Destination $HlslDestination -Force
Copy-Item -LiteralPath (Join-Path $PocRoot 'model/artifacts/golden.bin') -Destination $HlslDestination -Force
Copy-Item -LiteralPath (Join-Path $PocRoot 'model/artifacts/model.bin') -Destination $GlslDestination -Force
Copy-Item -LiteralPath (Join-Path $PocRoot 'model/artifacts/golden.bin') -Destination $GlslDestination -Force

$ExamplesCmake = Join-Path $UpstreamRoot 'examples/CMakeLists.txt'
$CmakeText = Get-Content -LiteralPath $ExamplesCmake -Raw
if ($CmakeText -notmatch '(?m)^\s*gnncloth\s*$') {
    $CmakeText = $CmakeText -replace "(?m)^(\s*computecloth\s*)$", "`$1`r`n`tgnncloth"
    Set-Content -LiteralPath $ExamplesCmake -Value $CmakeText -NoNewline
}

Write-Host "Upstream ready: $UpstreamRoot @ $ActualCommit"
Write-Host "Overlay installed: examples/gnncloth and shaders/*/gnncloth"
