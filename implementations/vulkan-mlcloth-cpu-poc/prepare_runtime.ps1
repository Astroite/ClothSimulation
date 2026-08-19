[CmdletBinding()]
param([string]$MLClothRoot = 'E:\PRJ\Projects\PaperGame\Plugins\PaperGame\MLCloth')

$ErrorActionPreference = 'Stop'
$PocRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$MLClothRoot = [System.IO.Path]::GetFullPath($MLClothRoot)
$ManifestPath = Join-Path $MLClothRoot 'Models/manifest.json'
$RuntimeDir = Join-Path $PocRoot '.work/runtime'
if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) { throw "MLCloth model manifest is missing: $ManifestPath" }
$Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
$Matches = @($Manifest.models | Where-Object { $_.id -eq 'ch10032-cloth2607-neuralres4' })
if ($Matches.Count -ne 1) { throw 'Expected exactly one ch10032-cloth2607-neuralres4 model entry' }
$Model = $Matches[0]
if ($Model.usage -ne 'MLClothVertex' -or $Model.status -ne 'active' -or
    $Model.modelType -ne 2 -or $Model.driverCount -ne 45 -or
    $Model.vertexCount -ne 5294 -or $Model.driverFeatureLen -ne 1969 -or
    $Model.drivenFeatureLen -ne 16394 -or $Model.pcaDim -ne 512) {
    throw 'Vertex model manifest dimensions/type/status do not match the PoC contract'
}
$ModelSource = Join-Path $MLClothRoot $Model.path
if (-not (Test-Path -LiteralPath $ModelSource -PathType Leaf)) { throw "Model is missing: $ModelSource" }
$ModelFile = Get-Item -LiteralPath $ModelSource
$ModelHash = (Get-FileHash -LiteralPath $ModelSource -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ModelFile.Length -ne [int64]$Model.bytes -or $ModelHash -ne $Model.sha256.ToLowerInvariant()) {
    throw "Model manifest verification failed: bytes=$($ModelFile.Length), sha256=$ModelHash"
}

$DllNames = @('opencv_world440.dll', 'samplerate.dll', 'sent2pron.dll', 'MNN.dll', 'AILab.dll')
$Sources = @($ModelSource)
foreach ($Name in $DllNames) {
    $Path = Join-Path $MLClothRoot "ThirdParty/Win64/$Name"
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Runtime DLL is missing: $Path" }
    $Sources += $Path
}
New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
$Files = foreach ($Source in $Sources) {
    $Destination = Join-Path $RuntimeDir (Split-Path -Leaf $Source)
    Copy-Item -LiteralPath $Source -Destination $Destination -Force
    $Item = Get-Item -LiteralPath $Destination
    [pscustomobject]@{
        name = $Item.Name
        source = [System.IO.Path]::GetFullPath($Source)
        bytes = $Item.Length
        sha256 = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToLowerInvariant()
        version = $Item.VersionInfo.FileVersion
    }
}
$RuntimeManifest = [ordered]@{
    schemaVersion = 1
    preparedUtc = [DateTime]::UtcNow.ToString('o')
    sourceRoot = $MLClothRoot
    sourceManifest = [System.IO.Path]::GetFullPath($ManifestPath)
    modelId = $Model.id
    files = @($Files)
}
$RuntimeManifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $RuntimeDir 'runtime-manifest.json') -Encoding utf8NoBOM
Write-Host "Verified and prepared MLCloth runtime: $RuntimeDir"

