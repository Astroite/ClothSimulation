[CmdletBinding()]
param(
    # Root of an MLCloth training export: the dataset directory that holds
    # cloth_topology.bin and one subdirectory per clip. A single clip directory works
    # too, because the compact export writes cloth_topology.bin one level up from the
    # clip (it is shared by every clip of a garment), so both layouts are searched.
    [Parameter(Mandatory = $true)][string]$ExportRoot,
    [string]$Model = '',
    [string]$Output = '',
    [string]$Report = '',
    [int]$AssetIndex = 0,
    [int]$Lod = 0,
    # Which clip supplies the rest configuration. A T-pose reference clip is the
    # defensible choice and is picked automatically when the export has one; a
    # mid-motion clip is not a rest pose, so anything else has to be named explicitly.
    [string]$RestClip = '',
    # -1 means the clip's last frame. The T-pose reference is a settle under gravity, so
    # its final frame is the rested garment; frame 0 is the instant after a hard reset,
    # which is a pose nobody chose.
    [int]$RestFrame = -1,
    # Clips used to measure which boundary loops ride the body. Several dissimilar
    # motions, because the classification has to hold across all of them. Left empty, a
    # deterministic spread is picked and printed.
    [string[]]$AttachmentClip = @(),
    [int]$AttachmentClipCount = 4,
    [int]$AttachmentStep = 4,
    [double]$Density = 0,
    # Optional assertion, not a setting: the up axis is derived from the reference
    # transform inside cloth_sim.bin. Supply this only to make the bake fail if the
    # derived axis is not what you expected.
    [ValidateSet('x', 'y', 'z')][string]$UpAxis = ''
)

$ErrorActionPreference = 'Stop'
$PocRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Model) { $Model = Join-Path $PocRoot '.work/runtime/model_NeuralRes4_NeuralRes4_final.enc' }
if (-not $Output) { $Output = Join-Path $PocRoot '.work/mesh/ch10032_cloth2607.mlmesh' }
if (-not $Report) { $Report = Join-Path $PocRoot '.work/mesh/ch10032_cloth2607.report.json' }
$ExportRoot = [System.IO.Path]::GetFullPath($ExportRoot)
$Model = [System.IO.Path]::GetFullPath($Model)
$Output = [System.IO.Path]::GetFullPath($Output)
$Report = [System.IO.Path]::GetFullPath($Report)

if (-not (Test-Path -LiteralPath $ExportRoot -PathType Container)) { throw "Export directory does not exist: $ExportRoot" }
if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) { throw "Encoded model is missing: $Model. Run prepare_runtime.ps1 first." }

# Several clips share one garment, so several cloth_sim.bin is normal; several
# cloth_topology.bin is not, and picking one arbitrarily could pair a mesh with the
# wrong garment. Identical copies are fine -- the compact export writes the same bytes
# beside the reference clip -- so they are collapsed by hash rather than rejected.
$TopologyFiles = @(Get-ChildItem -LiteralPath $ExportRoot -Filter 'cloth_topology.bin' -File -Recurse -ErrorAction SilentlyContinue)
if ($TopologyFiles.Count -eq 0) {
    throw "Could not find cloth_topology.bin under $ExportRoot. Was the training export run with cloth simulation data enabled?"
}
$DistinctTopologies = @($TopologyFiles | Group-Object { (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash })
if ($DistinctTopologies.Count -gt 1) {
    $Paths = ($TopologyFiles | ForEach-Object { $_.FullName }) -join "`n  "
    throw "Found $($DistinctTopologies.Count) different cloth_topology.bin under ${ExportRoot}; pass a narrower -ExportRoot.`n  $Paths"
}
$Topology = $TopologyFiles[0].FullName

# A clip directory is one that carries both files the measurement needs.
$ClipDirectories = @(
    Get-ChildItem -LiteralPath $ExportRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object {
            (Test-Path -LiteralPath (Join-Path $_.FullName 'cloth_sim.bin') -PathType Leaf) -and
            (Test-Path -LiteralPath (Join-Path $_.FullName 'manifest.json') -PathType Leaf)
        } | Sort-Object Name
)
if ($ClipDirectories.Count -eq 0) {
    # A single clip directory was passed directly rather than a dataset root.
    if (Test-Path -LiteralPath (Join-Path $ExportRoot 'cloth_sim.bin') -PathType Leaf) {
        $ClipDirectories = @(Get-Item -LiteralPath $ExportRoot)
    } else {
        throw "No clip directory with cloth_sim.bin and manifest.json found under $ExportRoot"
    }
}

if ($RestClip) {
    $RestDirectory = if (Test-Path -LiteralPath $RestClip -PathType Container) { (Get-Item -LiteralPath $RestClip) }
                     else { $ClipDirectories | Where-Object { $_.Name -eq $RestClip } | Select-Object -First 1 }
    if (-not $RestDirectory) { throw "-RestClip '$RestClip' is not a clip directory under $ExportRoot" }
} else {
    $RestDirectory = $ClipDirectories | Where-Object { $_.Name -like '*TPose*' } | Select-Object -First 1
    if (-not $RestDirectory) {
        $Names = ($ClipDirectories | ForEach-Object { $_.Name }) -join ', '
        throw "No T-pose reference clip found under $ExportRoot, so the rest configuration is ambiguous. Pass -RestClip explicitly. Available: $Names"
    }
}
$Simulation = Join-Path $RestDirectory.FullName 'cloth_sim.bin'
$Manifest = Join-Path $RestDirectory.FullName 'manifest.json'

if ($AttachmentClip.Count -eq 0) {
    # A deterministic spread across the dataset: motion clips only (the rest clip holds
    # a single pose and the _x0.5 variants are the same motion retimed, so neither adds
    # an independent test of what is attached), sampled evenly and then printed. Printing
    # matters more than the choice -- a silently sampled subset reads as full coverage.
    $Candidates = @($ClipDirectories | Where-Object { $_.FullName -ne $RestDirectory.FullName -and $_.Name -notlike '*_x0.5' })
    if ($Candidates.Count -eq 0) {
        Write-Warning "No motion clips available, so pins will fall back to the rest-pose height rule. See pin_rule_warning in the report."
    } else {
        $Take = [Math]::Min($AttachmentClipCount, $Candidates.Count)
        $AttachmentClip = @(0..($Take - 1) | ForEach-Object {
            $Candidates[[int][Math]::Floor($_ * $Candidates.Count / $Take)].FullName
        })
    }
}
if ($AttachmentClip.Count -gt 0) {
    Write-Host "Measuring loop attachment on $($AttachmentClip.Count) clip(s):"
    $AttachmentClip | ForEach-Object { Write-Host "  $(Split-Path -Leaf $_)" }
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Output) | Out-Null

$Converter = Join-Path $PocRoot 'tools/bake_cloth_topology.py'
$Arguments = @(
    $Converter,
    '--topology', $Topology,
    '--sim', $Simulation,
    '--model', $Model,
    '--manifest', $Manifest,
    '--asset-index', "$AssetIndex",
    '--lod', "$Lod",
    '--rest-frame', "$RestFrame",
    '--attachment-step', "$AttachmentStep",
    '--output', $Output,
    '--report', $Report
)
foreach ($Clip in $AttachmentClip) { $Arguments += @('--attachment-clip', $Clip) }
if ($Density -gt 0) { $Arguments += @('--density', "$Density") }
if ($UpAxis) { $Arguments += @('--up-axis', $UpAxis) }

$Python = if (Get-Command py -ErrorAction SilentlyContinue) { 'py' } else { 'python' }
$PythonArguments = if ($Python -eq 'py') { @('-3') + $Arguments } else { $Arguments }
& $Python @PythonArguments
if ($LASTEXITCODE -ne 0) { throw "Topology conversion failed with exit code $LASTEXITCODE" }
if (-not (Test-Path -LiteralPath $Output -PathType Leaf)) { throw "Conversion produced no mesh: $Output" }

# Re-read the result with the strict C++ parser, which re-derives the edge set and
# boundary structure independently. Passing this is the difference between "the
# writer wrote something" and "the runtime will load it".
$Validator = Join-Path $PocRoot 'tests/build/mesh_validate.exe'
if (Test-Path -LiteralPath $Validator -PathType Leaf) {
    & $Validator $Model $Output
    if ($LASTEXITCODE -ne 0) { throw "The baked mesh was rejected by mesh_validate: $Output" }
} else {
    Write-Warning "mesh_validate.exe not built, so the C++ cross-check was skipped. Run build.ps1 to enable it."
}
Write-Host "Baked cloth topology: $Output"
