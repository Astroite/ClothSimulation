[CmdletBinding()]
param([Parameter(Mandatory)][string]$Manifest,[string[]]$Clips=@('jump','turn'),[ValidateSet(0,1)][int]$ContactGuide=1,[string]$Blender='blender')
$ErrorActionPreference='Stop'
$TaskRoot=Split-Path $PSScriptRoot -Parent
$TaskManifest=(Resolve-Path -LiteralPath $Manifest).Path
$TaskAssets=Split-Path $TaskManifest -Parent
$TaskOutput=Join-Path $TaskAssets ('collision-candidate-'+(Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $TaskOutput | Out-Null
$TaskExe=Join-Path $TaskRoot '.work/Vulkan/build-mlclothcpu/bin/mlclothcpu.exe'
$Data=Get-Content -LiteralPath $TaskManifest -Raw | ConvertFrom-Json
$Summary=@()
foreach ($Id in $Clips) {
    $Clip=@($Data.clips|Where-Object id -eq $Id)
    if ($Clip.Count -ne 1) { throw "Unknown animation: $Id" }
    $Frames=[int][Math]::Ceiling($Clip[0].duration*60)+2
    $Output=Join-Path $TaskOutput $Id
    New-Item -ItemType Directory -Path $Output | Out-Null
    $Arguments=@('--demo-manifest',$TaskManifest,'--animation',$Id,'--frames',"$Frames",'--fixed-frame-dt','0.0166666666666667',
        '--tethers','0','--contact-guide',"$ContactGuide",'--dump-physics','1','--screenshot',(Join-Path $Output 'final.ppm'))
    $Quoted=($Arguments|ForEach-Object{'"'+$_.Replace('"','\"')+'"'}) -join ' '
    $Process=Start-Process -FilePath $TaskExe -WorkingDirectory (Join-Path $TaskRoot '.work/Vulkan') -ArgumentList $Quoted -WindowStyle Hidden -PassThru
    $Process.Id | Set-Content -LiteralPath (Join-Path $TaskOutput 'owned-process.pid')
    $Process.WaitForExit()
    if ($Process.ExitCode -ne 0) { throw "Demo failed for $Id" }
    Copy-Item -LiteralPath (Join-Path $TaskAssets 'physics-snapshot.json') -Destination $Output
    & $Blender --factory-startup -b --python-exit-code 1 -P (Join-Path $PSScriptRoot 'measure_demo_snapshot.py') -- --snapshot (Join-Path $Output 'physics-snapshot.json') --output (Join-Path $Output 'quality.json') *> (Join-Path $Output 'quality.log')
    if ($LASTEXITCODE -ne 0) { throw 'Geometry measurement failed' }
    $Quality=Get-Content -LiteralPath (Join-Path $Output 'quality.json') -Raw | ConvertFrom-Json
    if (@($Quality.actors|Where-Object{-not $_.finite -or $_.time -lt $Clip[0].duration}).Count) { throw "Incomplete or non-finite rollout: $Id" }
    $Summary+=@{clip=$Id;actors=$Quality.actors}
    @{qualification=$false;manifest=$TaskManifest;collision=$Data.collision;contact_aware_guide=($ContactGuide -ne 0);executable_sha256=(Get-FileHash -LiteralPath $TaskExe -Algorithm SHA256).Hash;
      purpose='Collision quality snapshots; performance qualification deferred by user';cases=$Summary} |
        ConvertTo-Json -Depth 16 | Set-Content -LiteralPath (Join-Path $TaskOutput 'summary.json') -Encoding utf8
    Write-Output "$Id complete"
}
Write-Output "Collision candidate: $TaskOutput"
