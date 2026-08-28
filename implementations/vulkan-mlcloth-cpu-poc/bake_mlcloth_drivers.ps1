[CmdletBinding()]
param(
    # Dataset root of an MLCloth training export. Every clip directory under it is baked;
    # the T-pose reference is skipped because it holds a single pose and would be refused
    # by the no-motion check.
    [Parameter(Mandatory = $true)][string]$ExportRoot,
    [string]$Model = '',
    [string]$OutputDir = '',
    [string]$Report = '',
    # An Unreal-produced .mldrv to compare the matching bake against. Defaults to the
    # existing sprint clip when both it and its export counterpart are present, because
    # an offline bake that has never been checked against the editor's own AnimPose
    # evaluation is an assumption rather than a result.
    [string]$CrossCheck = '',
    [string]$CrossCheckClip = 'AS_C10032_ArmedSprint_Skirt'
)

$ErrorActionPreference = 'Stop'
$PocRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Model) { $Model = Join-Path $PocRoot '.work/runtime/model_NeuralRes4_NeuralRes4_final.enc' }
if (-not $OutputDir) { $OutputDir = Join-Path $PocRoot '.work/clips/offline' }
if (-not $Report) { $Report = Join-Path $OutputDir 'batch.json' }
$ExportRoot = [System.IO.Path]::GetFullPath($ExportRoot)
$Model = [System.IO.Path]::GetFullPath($Model)
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
$Report = [System.IO.Path]::GetFullPath($Report)

if (-not (Test-Path -LiteralPath $ExportRoot -PathType Container)) { throw "Export directory does not exist: $ExportRoot" }
if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) { throw "Encoded model is missing: $Model. Run prepare_runtime.ps1 first." }

if (-not $CrossCheck) {
    $Candidate = Join-Path $PocRoot ".work/clips/$CrossCheckClip.mldrv"
    $Counterpart = @(Get-ChildItem -LiteralPath $ExportRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "*$CrossCheckClip" })
    if ((Test-Path -LiteralPath $Candidate -PathType Leaf) -and $Counterpart.Count -eq 1) {
        $CrossCheck = $Candidate
        $CrossCheckClip = $Counterpart[0].Name
    } else {
        Write-Warning "No Unreal-baked counterpart found, so this run is not cross-checked against the editor. Pass -CrossCheck to enable it."
    }
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$Arguments = @(
    (Join-Path $PocRoot 'tools/bake_mlcloth_drivers.py'),
    '--model', $Model,
    '--export-root', $ExportRoot,
    '--output-dir', $OutputDir,
    '--report', $Report
)
if ($CrossCheck) { $Arguments += @('--cross-check', $CrossCheck, '--cross-check-clip', $CrossCheckClip) }

$Python = if (Get-Command py -ErrorAction SilentlyContinue) { 'py' } else { 'python' }
$PythonArguments = if ($Python -eq 'py') { @('-3') + $Arguments } else { $Arguments }
& $Python @PythonArguments
if ($LASTEXITCODE -ne 0) { throw "Driver clip bake failed with exit code $LASTEXITCODE" }

# Hand one clip to the real AILab runtime. The Python side can produce a well-formed
# container that the C++ parse_clip still refuses -- the model hash lock and the float
# count fields are checked there, not here -- so "the baker wrote files" is not the same
# claim as "the runtime will infer from them".
$Integration = Join-Path $PocRoot 'tests/build/runtime_integration.exe'
$RuntimeDir = Join-Path $PocRoot '.work/runtime'
$Sample = @(Get-ChildItem -LiteralPath $OutputDir -Filter '*.mldrv' -File | Sort-Object Name | Select-Object -First 1)
if ((Test-Path -LiteralPath $Integration -PathType Leaf) -and (Test-Path -LiteralPath (Join-Path $RuntimeDir 'AILab.dll') -PathType Leaf) -and $Sample.Count -eq 1) {
    & $Integration $RuntimeDir $Model $Sample[0].FullName
    if ($LASTEXITCODE -ne 0) { throw "The AILab runtime rejected a baked clip: $($Sample[0].FullName)" }
} else {
    Write-Warning "runtime_integration.exe or the AILab runtime is absent, so no clip was fed to the real inference runtime."
}
Write-Host "Baked driver clips: $OutputDir"
