[CmdletBinding()]
param(
    # Automation inputs only. Presentation settings are changed in the Demo UI.
    [string]$Manifest = '',
    [ValidateRange(0, 2147483647)][int]$Frames = 0,
    [switch]$ValidateGpu,
    [switch]$ValidateAsset,
    [switch]$ValidatePresentation,
    [switch]$ValidateGnn,
    [switch]$ValidateTemporal,
    [string]$Screenshot = ''
)
$ErrorActionPreference = 'Stop'
$DemoRoot = $PSScriptRoot
$Executable = Join-Path $DemoRoot '.work/Vulkan/build-mlclothcpu/bin/mlclothcpu.exe'
if (-not (Test-Path -LiteralPath $Executable)) {
    throw 'Demo executable missing. Run bootstrap.ps1, then tools/build-with-vs.cmd.'
}
if (-not $Manifest) { $Manifest = Join-Path $DemoRoot '.work/demo/demo.json' }
$Manifest = (Resolve-Path -LiteralPath $Manifest).Path
$DemoArguments = @('--demo-manifest', $Manifest)
if ($ValidatePresentation) {
    if ($Frames -eq 0) { $Frames = 180 }
    $DemoArguments += @('--validate-presentation','1','--fixed-frame-dt','0.00694444444444444')
}
if ($ValidateGnn -or $ValidateTemporal) {
    if ($Frames -eq 0) { $Frames = 24 }
    $DemoArguments += @('--hybrid-algorithm','gnn','--validate-gnn','1','--dump-gnn','1','--fixed-frame-dt','0.0166666666666667')
    if ($ValidateTemporal) { $DemoArguments[$DemoArguments.IndexOf('gnn')] = 'temporal' }
}
if (($ValidateGpu -or $ValidateAsset) -and $Frames -eq 0) { $Frames = 1 }
if ($Frames -gt 0) { $DemoArguments += @('--frames', "$Frames") }
if ($ValidateGpu) { $DemoArguments += @('--validate-gpu', '1') }
if ($ValidateAsset) { $DemoArguments += @('--validate-asset', '1') }
if ($Screenshot) {
    $DemoArguments += @('--screenshot', [IO.Path]::GetFullPath($Screenshot))
}
Push-Location (Join-Path $DemoRoot '.work/Vulkan')
try {
    # PowerShell does not consistently wait for a GUI-subsystem executable invoked
    # with &: wait explicitly and preserve quoting for manifests containing spaces.
    $QuotedArguments = ($DemoArguments | ForEach-Object { '"' + $_.Replace('"', '\"') + '"' }) -join ' '
    $Launch = @{FilePath=$Executable; ArgumentList=$QuotedArguments; Wait=$true; PassThru=$true}
    if ($Frames -gt 0) { $Launch.WindowStyle = 'Hidden' }
    $DemoProcess = Start-Process @Launch
    if ($DemoProcess.ExitCode -ne 0) { throw "Demo exited with code $($DemoProcess.ExitCode)" }
    $Checks = @()
    if ($ValidateGpu) { $Checks += 'gpu-validation.json' }
    if ($ValidateAsset) { $Checks += 'gpu-asset-validation.json' }
    if ($ValidatePresentation) { $Checks += 'presentation-validation.json' }
    if ($ValidateGnn -or $ValidateTemporal) { $Checks += 'gnn-integration-validation.json' }
    foreach ($Check in $Checks) {
        $Report = Get-Content -LiteralPath (Join-Path (Split-Path $Manifest -Parent) $Check) -Raw | ConvertFrom-Json
        if (-not $Report.passed) { throw "Validation failed: $Check" }
    }
    if ($ValidateGnn -or $ValidateTemporal) {
        $ReferenceArguments = @('--directory', (Join-Path (Split-Path $Manifest -Parent) 'gnn-validation'))
        if ($ValidateTemporal) {
            $TemporalManifest = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
            $TemporalPath = $TemporalManifest.temporal_gnn_model
            if (-not $TemporalPath) { $TemporalPath = '../temporal_v4/temporal-init.vthood' }
            if (-not [IO.Path]::IsPathRooted($TemporalPath)) { $TemporalPath = Join-Path (Split-Path $Manifest -Parent) $TemporalPath }
            $ReferenceArguments += @('--temporal', $TemporalPath)
        }
        & (Join-Path $DemoRoot '../vulkan-gnn-poc/.venv/Scripts/python.exe') (Join-Path $DemoRoot 'tools/validate_demo_gnn.py') @ReferenceArguments
        if ($LASTEXITCODE -ne 0) { throw 'GNN independent reference validation failed' }
    }
} finally { Pop-Location }
