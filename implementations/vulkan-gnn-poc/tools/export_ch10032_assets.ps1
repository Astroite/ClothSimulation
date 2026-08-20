<#
.SYNOPSIS
    Bulk-export the CH10032 animation and model set from the Z2Game project.

.DESCRIPTION
    Thin wrapper: launches the editor once and hands the work to
    tools/export_ch10032_assets.py, which does the exporting and writes
    <OutputRoot>/export_report.json.

    The invocation is load-bearing. The project patches
    FPythonScriptPlugin::StartupModule to return early when
    IsRunningCommandlet() || FApp::IsUnattended(), so Python never initializes
    under -run=pythonscript or under -unattended, and no command-line flag can
    re-enable it. Driving `py <script>` through -ExecCmds with neither flag
    does work. Two further constraints, both measured on this engine build:

      * -nullrhi must NOT be passed. SkeletalMesh FBX export asserts without a
        render resource (USkinnedMeshComponent::GetCPUSkinnedVertices ->
        check(MeshObject)).
      * The console-command route needs no PackagesToBeFullyLoadedAtStartup ini
        override, unlike `OBJ EXPORT`, which only ever resolves the single
        preloaded package and would need one editor launch per asset.

    Exports are resumable: an asset whose output already exists and is
    non-empty is skipped unless -Force is passed.

.EXAMPLE
    .\tools\export_ch10032_assets.ps1
    .\tools\export_ch10032_assets.ps1 -Tier skirt
    .\tools\export_ch10032_assets.ps1 -Only sprint_skirt,body -Force
#>
param(
    [string]$UnrealEditor = 'E:\Main\Engine\Binaries\Win64\UnrealEditor-Cmd.exe',
    [string]$Project = 'E:\Main\Projects\Z2Game\Z2Game.uproject',
    [string]$OutputRoot,
    [ValidateSet('all', 'skirt', 'locomotion')]
    [string]$Tier = 'all',
    [string[]]$Only = @(),
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$PocRoot = Split-Path -Parent $PSScriptRoot
$ManifestPath = Join-Path $PSScriptRoot 'ch10032_export_manifest.json'
$ExportScript = Join-Path $PSScriptRoot 'export_ch10032_assets.py'
if (-not $OutputRoot) { $OutputRoot = Join-Path $PocRoot '.work/ch10032_library' }

foreach ($required in @($UnrealEditor, $Project, $ManifestPath, $ExportScript)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Missing required path: $required" }
}
# -ExecCmds splits on commas, so a comma in either path would truncate the command.
foreach ($path in @($ExportScript, $OutputRoot)) {
    if ($path -like '*,*') { throw "Path must not contain a comma: $path" }
}
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

# `pwsh -File script.ps1 -Only a,b` binds one literal string, not an array.
$onlyIds = $Only | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } |
    Where-Object { $_ }

$env:CH10032_MANIFEST = $ManifestPath
$env:CH10032_OUTPUT_ROOT = $OutputRoot
$env:CH10032_TIER = $Tier
$env:CH10032_ONLY = ($onlyIds -join ',')
$env:CH10032_FORCE = if ($Force) { '1' } else { '0' }

$log = Join-Path $OutputRoot 'unreal_export.log'
Write-Host "Exporting CH10032 assets (tier=$Tier) to $OutputRoot"
Write-Host "Editor log: $log"

# No -unattended and no -run=: both would disable Python. No -nullrhi: it
# breaks SkeletalMesh export.
& $UnrealEditor $Project "-ExecCmds=py $ExportScript,QUIT_EDITOR" `
    -nosplash -NoSound -nop4 -abslog="$log" | Out-Null

$reportPath = Join-Path $OutputRoot 'export_report.json'
if (-not (Test-Path -LiteralPath $reportPath -PathType Leaf)) {
    throw "Export produced no report. Python probably never ran; check $log for 'LogPython'."
}

$report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
Write-Host ''
Write-Host "requested=$($report.counts.requested) exported=$($report.counts.exported) skipped=$($report.counts.skipped) failed=$($report.counts.failed)"
$report.assets | Where-Object { $_.status -eq 'exported' } |
    ForEach-Object { Write-Host ("  {0,-24} {1,8:N2} MB" -f $_.id, ($_.bytes / 1MB)) }

$failed = @($report.assets | Where-Object { $_.status -eq 'failed' })
if ($failed.Count -gt 0) {
    Write-Host ''
    Write-Warning "$($failed.Count) asset(s) failed:"
    $failed | ForEach-Object { Write-Warning "  $($_.id): $($_.error)" }
    exit 1
}
Write-Host "report: $reportPath"
