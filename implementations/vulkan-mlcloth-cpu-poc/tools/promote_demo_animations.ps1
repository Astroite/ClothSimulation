[CmdletBinding()]
param([Parameter(Mandatory)][string]$Candidate)
$ErrorActionPreference='Stop'
$TaskRoot=Split-Path $PSScriptRoot -Parent
$TaskCurrent=(Resolve-Path -LiteralPath (Join-Path $TaskRoot '.work/demo')).Path
$TaskCandidate=(Resolve-Path -LiteralPath $Candidate).Path
if($TaskCurrent -eq $TaskCandidate){throw 'Candidate must be separate from the active assets'}
if(Get-Process mlclothcpu -ErrorAction SilentlyContinue){throw 'Finish the running Demo before publishing animation resources'}
foreach($Report in @('asset-validation.json','hybrid-blender-validation.json')){
    $Data=Get-Content -LiteralPath (Join-Path $TaskCandidate $Report) -Raw | ConvertFrom-Json
    if(-not $Data.passed){throw "Candidate validation failed: $Report"}
}
foreach($Name in @('character.dmp','cloth-attached.dmp','collision-hybrid.dmp','leg-capsules.dmp')){
    if((Get-FileHash -LiteralPath (Join-Path $TaskCurrent $Name)).Hash -ne (Get-FileHash -LiteralPath (Join-Path $TaskCandidate $Name)).Hash){throw "Animation-only promotion would change static asset: $Name"}
}
$Manifest=Get-Content -LiteralPath (Join-Path $TaskCurrent 'demo.json') -Raw | ConvertFrom-Json
$Replacement=Get-Content -LiteralPath (Join-Path $TaskCandidate 'demo-hybrid.json') -Raw | ConvertFrom-Json
if($Manifest.model_sha256 -ne $Replacement.model_sha256){throw 'Model identity changed'}
$Names=@('C10032_Demo.blend','blender-animation-validation.json','asset-validation.json','hybrid-blender-validation.json')
foreach($Clip in @('idle','walk','run','sprint','jump','turn','complex')){
    foreach($Suffix in @('.dma','.mldrv','.reference.dmp')){$Names+=$Clip+$Suffix}
}
foreach($Name in $Names){if(-not (Test-Path -LiteralPath (Join-Path $TaskCandidate $Name))){throw "Missing candidate file: $Name"}}
$Backup=Join-Path $TaskCurrent ('animation-history/'+(Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $Backup | Out-Null
Copy-Item -LiteralPath (Join-Path $TaskCurrent 'demo.json') -Destination (Join-Path $Backup 'demo.json')
$Changed=@()
try{
    foreach($Name in $Names){
        $Target=Join-Path $TaskCurrent $Name
        if(Test-Path -LiteralPath $Target){Copy-Item -LiteralPath $Target -Destination (Join-Path $Backup $Name)}
        Copy-Item -LiteralPath (Join-Path $TaskCandidate $Name) -Destination ($Target+'.tmp')
        $Changed+=$Name
        Move-Item -LiteralPath ($Target+'.tmp') -Destination $Target -Force
    }
    $Manifest.clips=$Replacement.clips
    $Manifest | Add-Member -NotePropertyName animation_pipeline -NotePropertyValue $Replacement.animation_pipeline -Force
    $Manifest | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath (Join-Path $TaskCurrent 'demo.json.tmp') -Encoding utf8
    Move-Item -LiteralPath (Join-Path $TaskCurrent 'demo.json.tmp') -Destination (Join-Path $TaskCurrent 'demo.json') -Force
}catch{
    foreach($Name in $Changed){$Saved=Join-Path $Backup $Name;if(Test-Path -LiteralPath $Saved){Copy-Item -LiteralPath $Saved -Destination (Join-Path $TaskCurrent $Name) -Force}}
    Copy-Item -LiteralPath (Join-Path $Backup 'demo.json') -Destination (Join-Path $TaskCurrent 'demo.json') -Force
    throw
}
@{candidate=$TaskCandidate;backup=$Backup;files=$Names;qualification=$false;note='Independent pose/skin validation passed. Cloth rollout qualification remains separate.'} |
    ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $Backup 'promotion.json') -Encoding utf8
Write-Output "Animation resources published; previous version retained at $Backup"
