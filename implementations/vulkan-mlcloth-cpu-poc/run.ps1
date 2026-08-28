[CmdletBinding()]
param(
    [string]$RuntimeDir = '', [string]$Model = '', [string]$Clip = '', [string]$Mesh = '',
    # Directory scanned for selectable animations. Defaults to the chosen clip's own directory,
    # so a bake directory becomes a selectable set with no extra flag.
    [string]$ClipDir = '',
    [ValidateSet(1, 2, 4, 8)][int]$Threads = 1,
    [switch]$Verify, [switch]$Benchmark, [int]$Frames = 0, [switch]$Validation, [switch]$Points,
    # CPU XPBD post-constraint. -XpbdIterations 0 is the no-regression arm: it must reproduce
    # the pure-network solved hash byte for byte, which mlcloth_verify.json reports.
    [switch]$Xpbd,
    [int]$XpbdIterations = 8,
    [switch]$XpbdTwoSided,
    [double]$XpbdStretchCompliance = 0.0,
    [double]$XpbdBendCompliance = 0.05,
    [double]$XpbdAreaFloor = 0.0,
    [double]$XpbdAreaCompliance = 0.0,
    # Negative leaves the guide off, so the prediction is only the initial state. The useful
    # band is narrow: alpha-tilde = compliance / dt^2, so at 30 Hz anything under about 0.5
    # is a hard guide in disguise and anything over a few is inert.
    [double]$XpbdGuideCompliance = -1.0,
    [double]$XpbdGuideTrust = 0.0,
    # Side by side: left is the network alone, right is the network plus constraints. One
    # inference feeds both, so it costs one extra solve and nothing else.
    [switch]$Compare,
    [double]$CompareSpacingCm = 90,
    [string]$Capsules = '',
    [switch]$NoBody,
    [switch]$NoCollision,
    # Which garment pieces take contacts, largest-first, comma separated. Empty means all,
    # which measurement shows lifts the fitted pieces off the body; '2' is the skirt.
    [string]$CollisionPieces = '2'
)
$ErrorActionPreference = 'Stop'
$PocRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $RuntimeDir) { $RuntimeDir = Join-Path $PocRoot '.work/runtime' }
if (-not $Model) { $Model = Join-Path $RuntimeDir 'model_NeuralRes4_NeuralRes4_final.enc' }
if (-not $Clip) {
    # Prefer the offline bake directory when it exists: it holds one clip per exported
    # animation, so the overlay's animation selector has a set to choose from. The single
    # Unreal-baked clip stays the fallback so a checkout without an export still runs.
    $OfflineDir = Join-Path $PocRoot '.work/clips/offline'
    $Offline = if (Test-Path -LiteralPath $OfflineDir -PathType Container) {
        @(Get-ChildItem -LiteralPath $OfflineDir -Filter '*.mldrv' -File | Sort-Object Name | Select-Object -First 1)
    } else { @() }
    $Clip = if ($Offline.Count -eq 1) { $Offline[0].FullName }
            else { Join-Path $PocRoot '.work/clips/AS_C10032_ArmedSprint_Skirt.mldrv' }
}
# The mesh is optional. Defaulting to the baked path when it exists means a fresh
# checkout still runs as the original point-cloud sample rather than failing on a
# missing asset, and -Points forces that mode back on demand.
if (-not $Mesh) {
    $DefaultMesh = Join-Path $PocRoot '.work/mesh/ch10032_cloth2607.mlmesh'
    if (Test-Path -LiteralPath $DefaultMesh -PathType Leaf) { $Mesh = $DefaultMesh }
}
$RuntimeDir = [System.IO.Path]::GetFullPath($RuntimeDir)
$Model = [System.IO.Path]::GetFullPath($Model)
$Clip = [System.IO.Path]::GetFullPath($Clip)
if ($Mesh) { $Mesh = [System.IO.Path]::GetFullPath($Mesh) }
if ($ClipDir) { $ClipDir = [System.IO.Path]::GetFullPath($ClipDir) }
$Executable = Join-Path $PocRoot '.work/Vulkan/build-mlclothcpu/bin/mlclothcpu.exe'
$WorkingDirectory = Join-Path $PocRoot '.work/Vulkan'
$BenchmarkOutput = Join-Path $WorkingDirectory 'mlcloth_benchmark.csv'
$ValidationLog = Join-Path $WorkingDirectory 'validation_output.txt'
foreach ($Required in @($Executable, $Model, $Clip, (Join-Path $RuntimeDir 'AILab.dll'))) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) { throw "Required file is missing: $Required" }
}
if ($Mesh -and -not (Test-Path -LiteralPath $Mesh -PathType Leaf)) { throw "Mesh asset is missing: $Mesh" }
$Arguments = @('-s', 'hlsl', '--runtime-dir', $RuntimeDir, '--model', $Model, '--clip', $Clip, '--threads', "$Threads")
if ($Mesh) { $Arguments += @('--mesh', $Mesh) }
if ($ClipDir) { $Arguments += @('--clip-dir', $ClipDir) }
if (-not $Capsules) {
    $DefaultCapsules = Join-Path $PocRoot '.work/capsules/ch10032.mlcap'
    if (Test-Path -LiteralPath $DefaultCapsules -PathType Leaf) { $Capsules = $DefaultCapsules }
}
if ($Capsules) { $Arguments += @('--capsules', [System.IO.Path]::GetFullPath($Capsules)) }
if ($NoBody) { $Arguments += '--no-body' }
if ($NoCollision) { $Arguments += '--no-collision' }
if ($CollisionPieces) { $Arguments += @('--collision-pieces', $CollisionPieces) }
if ($Compare) { $Arguments += @('--compare', '--compare-spacing-cm', "$CompareSpacingCm") }
if ($Points) { $Arguments += '--points' }
if ($Xpbd) {
    if (-not $Mesh) { throw 'XPBD needs the topology: bake it with bake_cloth_topology.ps1, or drop -Xpbd.' }
    $Arguments += @(
        '--xpbd',
        '--xpbd-iterations', "$XpbdIterations",
        '--xpbd-stretch-compliance', "$XpbdStretchCompliance",
        '--xpbd-bend-compliance', "$XpbdBendCompliance",
        '--xpbd-area-floor', "$XpbdAreaFloor",
        '--xpbd-area-compliance', "$XpbdAreaCompliance",
        '--xpbd-guide-compliance', "$XpbdGuideCompliance",
        '--xpbd-guide-trust', "$XpbdGuideTrust"
    )
    if ($XpbdTwoSided) { $Arguments += '--xpbd-two-sided' }
}
if ($Verify) { $Arguments += '--verify' }
if ($Benchmark) { $Arguments += @('--benchmark', '--benchmark-output', $BenchmarkOutput) }
if ($Frames -gt 0) { $Arguments += @('--frames', "$Frames") }
if ($Validation) { $Arguments += @('-v', '-vl', '--sync-validation') }
if ($Validation -and (Test-Path -LiteralPath $ValidationLog -PathType Leaf)) {
    Remove-Item -LiteralPath $ValidationLog -Force
}
if ($Benchmark -and (Test-Path -LiteralPath $BenchmarkOutput -PathType Leaf)) {
    Remove-Item -LiteralPath $BenchmarkOutput -Force
}
Push-Location $WorkingDirectory
try {
    # A native GUI-subsystem executable is asynchronous under PowerShell unless
    # it participates in a pipeline. Out-Host preserves live output and waits.
    & $Executable @Arguments 2>&1 | Out-Host
    $ExitCode = $LASTEXITCODE
    if ($null -eq $ExitCode -or $ExitCode -ne 0) { throw "mlclothcpu exited with code $ExitCode" }
    if ($Validation -and (Test-Path -LiteralPath $ValidationLog -PathType Leaf)) {
        $ValidationFailures = Select-String -LiteralPath $ValidationLog -Pattern 'WARNING:|ERROR:'
        if ($ValidationFailures) {
            Get-Content -LiteralPath $ValidationLog | Out-Host
            throw "Vulkan validation emitted warning/error messages: $ValidationLog"
        }
    }
    if ($Benchmark) {
        if (-not (Test-Path -LiteralPath $BenchmarkOutput -PathType Leaf)) {
            throw "Benchmark did not produce a CSV (sample shortfall or write failure): $BenchmarkOutput"
        }
        $Rows = @(Import-Csv -LiteralPath $BenchmarkOutput)
        if ($Rows.Count -ne 1 -or [int]$Rows[0].samples -ne 1000) {
            throw "Benchmark CSV must contain exactly one 1,000-sample result: $BenchmarkOutput"
        }
    }
} finally { Pop-Location }
