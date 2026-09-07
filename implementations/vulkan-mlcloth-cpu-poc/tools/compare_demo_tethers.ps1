[CmdletBinding()]
param([string]$Manifest='',[string]$Blender='blender')
$ErrorActionPreference='Stop'
$TaskRoot=Split-Path $PSScriptRoot -Parent
if(-not $Manifest){$Manifest=Join-Path $TaskRoot '.work/demo/demo.json'}
$TaskManifest=(Resolve-Path -LiteralPath $Manifest).Path
$TaskAssets=Split-Path $TaskManifest -Parent
$TaskOutput=Join-Path $TaskAssets ('tether-comparison-'+(Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $TaskOutput | Out-Null
$TaskExe=Join-Path $TaskRoot '.work/Vulkan/build-mlclothcpu/bin/mlclothcpu.exe'
$Summary=@()
foreach ($Clip in @(@{id='walk';frames=212},@{id='jump';frames=175},@{id='turn';frames=206})) {
    foreach ($Enabled in @(0,1)) {
        $Output=Join-Path $TaskOutput ($Clip.id+'-'+$Enabled)
        New-Item -ItemType Directory -Path $Output | Out-Null
        $Arguments=@('--demo-manifest',$TaskManifest,'--animation',$Clip.id,'--frames',"$($Clip.frames)",
            '--fixed-frame-dt','0.0166666666666667','--tethers',"$Enabled",'--dump-physics','1','--screenshot',(Join-Path $Output 'final.ppm'))
        $Quoted=($Arguments | ForEach-Object { '"'+$_.Replace('"','\"')+'"' }) -join ' '
        $Process=Start-Process -FilePath $TaskExe -WorkingDirectory (Join-Path $TaskRoot '.work/Vulkan') -ArgumentList $Quoted -WindowStyle Hidden -PassThru
        $Process.Id | Set-Content -LiteralPath (Join-Path $TaskOutput 'owned-process.pid')
        $Process.WaitForExit()
        if ($Process.ExitCode -ne 0) { throw "Demo failed: $($Clip.id) tethers=$Enabled" }
        Copy-Item -LiteralPath (Join-Path $TaskAssets 'physics-snapshot.json') -Destination $Output
        & $Blender --factory-startup -b --python-exit-code 1 -P (Join-Path $PSScriptRoot 'measure_demo_snapshot.py') -- --snapshot (Join-Path $Output 'physics-snapshot.json') --output (Join-Path $Output 'quality.json') *> (Join-Path $Output 'quality.log')
        if ($LASTEXITCODE -ne 0) { throw 'Geometry measurement failed' }
        $Quality=Get-Content -LiteralPath (Join-Path $Output 'quality.json') -Raw | ConvertFrom-Json
        $Snapshot=Get-Content -LiteralPath (Join-Path $Output 'physics-snapshot.json') -Raw | ConvertFrom-Json
        if ($Snapshot.physics.tethers -ne [bool]$Enabled) { throw 'Requested tether state was not applied' }
        $Summary+=@{clip=$Clip.id;tethers=[bool]$Enabled;physics=$Snapshot.physics;actors=$Quality.actors}
        @{qualification=$false;purpose='Algorithm A/B snapshots; performance qualification deferred by user';
          executable_sha256=(Get-FileHash -LiteralPath $TaskExe -Algorithm SHA256).Hash;cases=$Summary} |
            ConvertTo-Json -Depth 16 | Set-Content -LiteralPath (Join-Path $TaskOutput 'summary.json') -Encoding utf8
        Write-Output "$($Clip.id) tethers=$Enabled complete"
    }
}
Write-Output "Comparison: $TaskOutput"
