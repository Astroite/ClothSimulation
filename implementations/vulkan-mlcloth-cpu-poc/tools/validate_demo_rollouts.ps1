[CmdletBinding()]
param([string]$Manifest = '', [int]$ShortLoops = 20, [int]$LongLoops = 3,
      [float]$BendCompliance = 10000, [ValidateSet(0,1)][int]$ContactGuide = 1, [string[]]$Clips=@(), [string]$Blender = 'blender')
$ErrorActionPreference = 'Stop'
$TaskRoot = Split-Path $PSScriptRoot -Parent
if (-not $Manifest) { $Manifest = Join-Path $TaskRoot '.work/demo/demo-hybrid-collision.json' }
$Manifest = (Resolve-Path -LiteralPath $Manifest).Path
$TaskAssets = Split-Path $Manifest -Parent
$TaskExe = Join-Path $TaskRoot '.work/Vulkan/build-mlclothcpu/bin/mlclothcpu.exe'
$TaskOutput = Join-Path $TaskAssets ('rollouts-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $TaskOutput | Out-Null
$TaskManifestData = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
foreach($Id in $Clips){if($Id -notin @($TaskManifestData.clips.id)){throw "Unknown animation: $Id"}}
$TaskSummary = @()
foreach ($Clip in $TaskManifestData.clips) {
    if($Clips.Count -gt 0 -and $Clip.id -notin $Clips){continue}
    $Loops = if ($Clip.id -eq 'complex') { $LongLoops } else { $ShortLoops }
    if ($Loops -lt 1) { throw 'Loop counts must be positive' }
    $FrameCount = [int][Math]::Ceiling($Clip.duration * 60 * $Loops) + 2
    $ClipOutput = Join-Path $TaskOutput $Clip.id
    New-Item -ItemType Directory -Path $ClipOutput | Out-Null
    $Screenshot = Join-Path $ClipOutput 'final.ppm'
    $TaskArguments = @('--demo-manifest',$Manifest,'--animation',$Clip.id,'--frames',"$FrameCount",
        '--fixed-frame-dt','0.0166666666666667','--capture-metrics','1','--metric-warmup','15',
        '--bend-compliance',"$BendCompliance",'--contact-guide',"$ContactGuide",'--tethers','0','--dump-physics','1','--screenshot',$Screenshot)
    $Quoted = ($TaskArguments | ForEach-Object { '"' + $_.Replace('"','\"') + '"' }) -join ' '
    $Process = Start-Process -FilePath $TaskExe -WorkingDirectory (Join-Path $TaskRoot '.work/Vulkan') -ArgumentList $Quoted -WindowStyle Hidden -PassThru
    $Process.Id | Set-Content -LiteralPath (Join-Path $TaskOutput 'owned-process.pid')
    $Process.WaitForExit()
    if ($Process.ExitCode -ne 0) { throw "$($Clip.id) exited with $($Process.ExitCode)" }
    & $Blender --factory-startup -b --python-exit-code 1 -P (Join-Path $PSScriptRoot 'measure_demo_snapshot.py') -- --snapshot (Join-Path $TaskAssets 'physics-snapshot.json') --output (Join-Path $TaskAssets 'snapshot-quality.json') *> (Join-Path $ClipOutput 'quality.log')
    if ($LASTEXITCODE -ne 0) { throw "Geometry diagnostic failed for $($Clip.id)" }
    foreach ($Name in @('physics-snapshot.json','snapshot-quality.json','performance.json')) {
        Copy-Item -LiteralPath (Join-Path $TaskAssets $Name) -Destination (Join-Path $ClipOutput $Name)
    }
    $Metrics = Get-Content -LiteralPath (Join-Path $ClipOutput 'performance.json') -Raw | ConvertFrom-Json
    $Quality = Get-Content -LiteralPath (Join-Path $ClipOutput 'snapshot-quality.json') -Raw | ConvertFrom-Json
    $Reached = $Metrics.simulation_time -ge ($Clip.duration * $Loops - 0.001) -and -not $Metrics.paused
    $Finite = @($Quality.actors | Where-Object { -not $_.finite }).Count -eq 0
    $TaskSummary += @{clip=$Clip.id; loops=$Loops; reached_end=$Reached; final_finite=$Finite; simulation_time=$Metrics.simulation_time;
        animation_sha256=(Get-FileHash -LiteralPath (Join-Path $TaskAssets $Clip.animation) -Algorithm SHA256).Hash;
        driver_sha256=(Get-FileHash -LiteralPath (Join-Path $TaskAssets $Clip.drivers) -Algorithm SHA256).Hash;
        frame_p95_ms=$Metrics.frame_p95_ms; quality=$Quality.actors; qualification=$false}
    @{manifest=$Manifest; bend_compliance=$BendCompliance; contact_aware_guide=($ContactGuide -ne 0); executable_sha256=(Get-FileHash -LiteralPath $TaskExe -Algorithm SHA256).Hash; qualification=$false;
      limitation='Fixed time-step rollout and final snapshot diagnostics; not continuous quality sampling or realtime performance qualification.';
      clips=$TaskSummary} | ConvertTo-Json -Depth 12 | Set-Content -Encoding utf8 -LiteralPath (Join-Path $TaskOutput 'summary.json')
    Write-Output "$($Clip.id): loops=$Loops reached=$Reached finite=$Finite frame_p95=$($Metrics.frame_p95_ms) ms"
    if (-not $Reached -or -not $Finite) { throw "Rollout failed for $($Clip.id); see $ClipOutput" }
}
Write-Output "Rollout diagnostics: $TaskOutput"
