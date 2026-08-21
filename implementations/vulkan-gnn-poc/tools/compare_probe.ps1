# Runs the A/B/C comparison for a fixed number of simulation steps and returns the structure JSON.
#
# This exists because the renderer writes results/<name>.json from its destructor, so a bounded run
# needs the window opened, stepped, then closed -- the same mechanism capture_screenshot.ps1 uses.
# Its purpose is the cross-language check: `edge_length_ratio.p95` here is measured against the
# authored rest mesh with the same directed edge list tools/train_student.py::edge_ratios uses, so it
# is directly comparable to the Python probe's `edge_p95`, and agreement between the two is what
# makes the interactive comparison trustworthy.
#
# Needs an interactive desktop (a visible window), like capture_screenshot.ps1 and smoke_modes.ps1.
param(
    [string]$Motion = 'ch10032_sprint',
    [ValidateSet('Fine15', 'TinyHood')]
    [string]$Solver = 'TinyHood',
    [string]$HoodModel = '',
    [ValidateRange(1, 10000)]
    [int]$Steps = 120,
    [string]$Branches = 'ABC',
    [ValidateRange(0, 1024)]
    [int]$XpbdIterations = 128,
    [ValidateRange(0, 1024)]
    [int]$XpbdIterationsB = 228,
    [ValidateRange(1, 4)]
    [int]$FrameStep = 1,
    # Drops --hood-compare so the run goes down the original single-branch path. With -Branches C the
    # two should agree: that is the regression guard for the branch loop.
    [switch]$Single,
    # Defaults to the garment-level constraint set. Point it at a per-motion .vxpbd to measure how
    # much the calibration clip alone moves the result.
    [string]$XpbdAsset = '',
    # Loop the clip instead of stopping on its last frame. Off by default because the Python probe
    # clamps, so holding is what makes the two comparable past clip_exhausted_at.
    [switch]$Loop,
    # Soft-guided multirate options, matching tools/recovery_probe.py's. -XpbdIterations is PER
    # SUBSTEP, so -Substeps 4 -XpbdIterations 32 is the equal-budget partner of the default 1 x 128.
    # All default to the historical behaviour.
    [ValidateRange(1, 8)]
    [int]$Substeps = 1,
    [switch]$Guide,
    [double]$GuideCompliance = 10.0,
    [double]$GuideTrustRatio = 0.0,
    [double]$AreaFloor = 0.0,
    [double]$AreaCompliance = 0.0,
    [string]$Output = ''
)

$ErrorActionPreference = 'Stop'
$PocRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$UpstreamRoot = Join-Path $PocRoot '.work/Vulkan'
$Executable = Join-Path $UpstreamRoot 'build-gnn/bin/gnncloth.exe'
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) { throw "Executable is missing. Run .\build.ps1 first: $Executable" }
if (-not $HoodModel) { $HoodModel = Join-Path $PocRoot '.work/hood_data/student32x12_r1.vhood' }
if (-not $Output) { $Output = Join-Path $PocRoot ("results/compare_{0}_{1}x_{2}.json" -f $Motion, $FrameStep, $Branches) }
$AssetRoot = Join-Path $PocRoot ".work/real_scene/$Motion"
$XpbdAsset = $XpbdAsset ? $XpbdAsset : (Join-Path $PocRoot '.work/real_scene/ch10032_lower.vxpbd')
foreach ($Required in @($AssetRoot, $HoodModel, $XpbdAsset)) {
    if (-not (Test-Path -LiteralPath $Required)) { throw "Comparison input is missing: $Required" }
}

$RuntimeSolver = $Solver -eq 'TinyHood' ? 'tinyhood' : 'fine15'
$Arguments = @(
    '-s', 'hlsl', '--scene', 'ch10032', '--motion', $Motion, '--solver', $RuntimeSolver,
    '--asset-root', ('"{0}"' -f [System.IO.Path]::GetFullPath($AssetRoot)),
    '--hood-model', ('"{0}"' -f [System.IO.Path]::GetFullPath($HoodModel)),
    '--hood-xpbd-asset', ('"{0}"' -f [System.IO.Path]::GetFullPath($XpbdAsset)),
    '--hood-xpbd-iterations', $XpbdIterations,
    '--hood-frame-step', $FrameStep, '--hood-pause-after', $Steps,
    '--hood-stability-output', ('"{0}"' -f [System.IO.Path]::GetFullPath($Output))
)
if (-not $Loop) { $Arguments += '--hood-hold-last-frame' }
$Arguments += @('--hood-xpbd-substeps', $Substeps)
if ($Guide) { $Arguments += @('--hood-xpbd-guide', '--hood-xpbd-guide-compliance', $GuideCompliance) }
if ($GuideTrustRatio -gt 0) { $Arguments += @('--hood-xpbd-guide-trust-ratio', $GuideTrustRatio) }
if ($AreaFloor -gt 0) { $Arguments += @('--hood-xpbd-area-floor', $AreaFloor,
    '--hood-xpbd-area-compliance', $AreaCompliance) }
$Arguments += $Single ? @('--hood-xpbd') : @('--hood-compare', '--hood-compare-branches', $Branches,
    '--hood-xpbd-iterations-b', $XpbdIterationsB)

if (Test-Path -LiteralPath $Output) { Remove-Item -LiteralPath $Output }
Add-Type -AssemblyName System.Windows.Forms | Out-Null
Add-Type -Namespace GnnCompare -Name Win -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("user32.dll")]
public static extern bool ShowWindow(System.IntPtr handle, int command);
'@
$Process = Start-Process -FilePath $Executable -ArgumentList $Arguments -WorkingDirectory $UpstreamRoot -WindowStyle Normal -PassThru
try {
    for ($Attempt = 0; $Attempt -lt 200 -and $Process.MainWindowHandle -eq 0; ++$Attempt) {
        Start-Sleep -Milliseconds 25
        $Process.Refresh()
    }
    if ($Process.MainWindowHandle -eq 0) { throw 'The Vulkan window did not appear' }
    # Must stay visible: the upstream sample base stops rendering while minimised, and the simulation
    # is driven from the render loop, so a minimised run advances zero steps and writes a JSON full of
    # zeroed positions. Steps advance at hoodFps (30), so allow that wall time plus slack for pipeline
    # creation, asset upload and the settle step.
    [void][GnnCompare.Win]::ShowWindow($Process.MainWindowHandle, 9)
    $Budget = [int]($Steps / 30.0 * 1000) + 25000
    Write-Host ("Running {0} for {1} steps at {2}x ({3} s budget)..." -f $Motion, $Steps, $FrameStep, [int]($Budget / 1000))
    Start-Sleep -Milliseconds $Budget
} finally {
    if (-not $Process.HasExited) {
        [void]$Process.CloseMainWindow()
        if (-not $Process.WaitForExit(15000)) { Stop-Process -Id $Process.Id -Force }
    }
}
if (-not (Test-Path -LiteralPath $Output)) { throw "The run produced no structure JSON: $Output" }
Write-Host "Wrote $Output"
Get-Content -LiteralPath $Output -Raw | Write-Host
