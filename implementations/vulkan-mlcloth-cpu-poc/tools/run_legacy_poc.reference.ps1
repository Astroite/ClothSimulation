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
    # Side by side: A the network alone, B the constraints alone with no network at all, C both.
    # One inference feeds A and C, so the mode costs one extra solve for C and one for B. Needs a
    # mesh baked with reference clips, because B takes its pin target from the measured bind.
    [switch]$Compare,
    [double]$CompareSpacingCm = 90,
    # Equal CPU budget, not equal iterations: C is 2.558 ms of inference plus 2.57 ms of solve,
    # which at 0.341 ms per iteration buys B about 15. See mlclothcpu.cpp.
    [ValidateRange(0, 128)]
    [int]$XpbdIterationsB = 15,
    # Stop on the final frame instead of looping, so the post-motion settle is observable.
    [switch]$HoldLastFrame,
    # Frame decimation, which is the speed axis. The timestep is deliberately not scaled.
    [ValidateRange(1, 4)]
    [int]$FrameStep = 1,
    [string]$Capsules = '',
    # What stands in for the character: the skinned render mesh, the collision capsules, or
    # nothing. The capsules are still what contacts and the reported penetration use; they are
    # a ragdoll envelope several centimetres wider than the skin, which is why the skin is what
    # gets drawn. Needs .work/body/*.mlbody -- bake it with tools/bake_mlcloth_body.py.
    [ValidateSet('mesh', 'capsules', 'none')]
    [string]$Body = 'mesh',
    [string]$BodyMesh = '',
    [switch]$NoBody,
    # Drop the gradient sky and the gridded floor. They are what gives a hem a height and the
    # side-by-side branches a shared baseline, so this is for screenshots and for checking
    # whether a dark patch is shading or geometry.
    [switch]$NoSky,
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
if ($NoBody) { $Arguments += '--no-body' } else { $Arguments += @('--body', $Body) }
if (-not $BodyMesh) {
    $DefaultBody = Join-Path $PocRoot '.work/body/ch10032_body.mlbody'
    if (Test-Path -LiteralPath $DefaultBody -PathType Leaf) { $BodyMesh = $DefaultBody }
}
if ($BodyMesh) { $Arguments += @('--body-mesh', [System.IO.Path]::GetFullPath($BodyMesh)) }
if ($NoCollision) { $Arguments += '--no-collision' }
if ($CollisionPieces) { $Arguments += @('--collision-pieces', $CollisionPieces) }
if ($Compare) {
    $Arguments += @('--compare', '--compare-spacing-cm', "$CompareSpacingCm",
        '--xpbd-iterations-b', "$XpbdIterationsB")
}
if ($HoldLastFrame) { $Arguments += '--hold-last-frame' }
if ($FrameStep -gt 1) { $Arguments += @('--frame-step', "$FrameStep") }
if ($Points) { $Arguments += '--points' }
if ($NoSky) { $Arguments += '--no-sky' }
if ($Xpbd -or $Compare) {
    # `-Compare` turns the solver on inside the exe, so its knobs have to be forwarded here too.
    # They were not, and nothing said so: `-Compare -XpbdAreaFloor 1.0` silently ran with the
    # exe's own default of 0. The verify report now carries the values the solver was configured
    # with, so a dropped flag shows up as a number rather than as a puzzling result.
    if (-not $Mesh) {
        $Which = if ($Xpbd) { '-Xpbd' } else { '-Compare' }
        throw "XPBD needs the topology: bake it with bake_cloth_topology.ps1, or drop $Which."
    }
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
