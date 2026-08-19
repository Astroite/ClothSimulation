param(
    [string]$Archive = '',
    [switch]$RemoveArchive
)

$ErrorActionPreference = 'Stop'
$PocRoot = Split-Path -Parent $PSScriptRoot
$WorkRoot = Join-Path $PocRoot '.work'
if (-not $Archive) { $Archive = Join-Path $WorkRoot 'hood_auxiliary.zip' }
$Checkpoint = Join-Path $WorkRoot 'hood_data/trained_models/postcvpr.pth'
$ExpectedArchiveSha256 = '3b68239bea3f298f9456680e34cf0204c90512ba1e43233febb375a90038a2a4'
$ExpectedCheckpointSha256 = '155d2dd25e54756fc04b0d27996ebca3446b2a59d3a715bb1fb73407753ce5ea'
$GoogleDriveId = '1RdA4L6Fy50VsKZ8k7ySp5ps5YtWoHSgs'

New-Item -ItemType Directory -Force -Path $WorkRoot | Out-Null
if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf) -or (Get-FileHash -LiteralPath $Checkpoint -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ExpectedCheckpointSha256) {
    if (-not (Test-Path -LiteralPath $Archive -PathType Leaf) -or (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ExpectedArchiveSha256) {
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'uv is required to run the isolated gdown downloader' }
        & uvx --from gdown gdown --id $GoogleDriveId --output $Archive
        if ($LASTEXITCODE -ne 0) { throw 'Official HOOD auxiliary archive download failed' }
    }
    $ArchiveDigest = (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ArchiveDigest -ne $ExpectedArchiveSha256) { throw "Unexpected HOOD archive SHA-256: $ArchiveDigest" }
    & tar -xf $Archive -C $WorkRoot hood_data/trained_models/postcvpr.pth
    if ($LASTEXITCODE -ne 0) { throw 'Could not extract postcvpr.pth from the official HOOD archive' }
}

$CheckpointDigest = (Get-FileHash -LiteralPath $Checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
if ($CheckpointDigest -ne $ExpectedCheckpointSha256) { throw "Unexpected PostCVPR SHA-256: $CheckpointDigest" }
$Python = Join-Path $PocRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Create the reference environment first with: uv venv --python 3.11 .venv' }
& $Python (Join-Path $PocRoot 'tools/export_postcvpr.py') --checkpoint $Checkpoint
if ($LASTEXITCODE -ne 0) { throw 'PostCVPR VHOOD export failed' }
if ($RemoveArchive -and (Test-Path -LiteralPath $Archive -PathType Leaf)) {
    Remove-Item -LiteralPath $Archive -Force
}
Write-Host "PostCVPR VHOOD ready: $(Join-Path $WorkRoot 'hood_data/postcvpr.vhood')"
