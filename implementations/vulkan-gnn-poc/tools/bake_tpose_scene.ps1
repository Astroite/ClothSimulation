param(
    [switch]$SkipUnrealExport,
    [switch]$SkipFine15Golden
)

$ErrorActionPreference = 'Stop'
$PocRoot = Split-Path -Parent $PSScriptRoot
& (Join-Path $PocRoot 'tools/bake_real_scene.ps1') `
    -Motion 'ch10032_tpose' `
    -AnimAsset '/Game/Developers/jinzhao/AICloth/CH_10032/Animation/00_Pose/AS_C10032_Tpose.AS_C10032_Tpose' `
    -StaticPose `
    -SkipUnrealExport:$SkipUnrealExport `
    -SkipFine15Golden:$SkipFine15Golden
