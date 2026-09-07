[CmdletBinding()]
param([switch]$ExportSources, [switch]$CollisionOnly, [string]$Blender = 'blender',
      [string]$SourceBlend = 'F:\ArtWorks\Cloth\CH\AS_C10032_Body_JSH_V09_Chongdingxiang_tiaowu_Repaired_TposeWarmup_60fps_Preview.blend')
$ErrorActionPreference = 'Stop'
$PocRoot = $PSScriptRoot
if ($ExportSources) {
    $Exporter = Join-Path $PocRoot 'tools/export_demo_sources_unreal.py'
    & 'E:\Main\Engine\Binaries\Win64\UnrealEditor-Cmd.exe' 'E:\Main\Projects\Z2Game\Z2Game.uproject' "-ExecutePythonScript=$Exporter" -ScriptErrorsAreFatal -ForceEnablePython -nop4 -nosplash -nullrhi -NoSound -stdout -FullStdOutLogOutput
    if ($LASTEXITCODE -ne 0) { throw 'Source animation export failed' }
}
if (-not $CollisionOnly) {
& $Blender --factory-startup -b --python-exit-code 1 -P (Join-Path $PocRoot 'tools/test_demo_pose_blend.py')
if ($LASTEXITCODE -ne 0) { throw 'Animation transition regression failed' }
foreach ($Clip in @('idle','walk','run','sprint','jump_start','jump_land','turn')) {
    if (-not (Test-Path -LiteralPath (Join-Path $PocRoot ".work/demo/source/$Clip.fbx"))) { throw "Missing $Clip.fbx; run prepare_demo.ps1 -ExportSources" }
}
& $Blender --factory-startup -b $SourceBlend --python-exit-code 1 -P (Join-Path $PocRoot 'tools/bake_demo_blender.py') -- --clips all
if ($LASTEXITCODE -ne 0) { throw 'Blender demo bake failed' }
}
$Manifest = Join-Path $PocRoot '.work/demo/demo.json'
if (-not (Test-Path -LiteralPath $Manifest)) { throw 'Bake did not produce demo.json; inspect Blender log' }
$LegacyManifest = Join-Path $PocRoot '.work/demo/demo-original-stm.json'
if (-not (Test-Path -LiteralPath $LegacyManifest)) { Copy-Item -LiteralPath $Manifest -Destination $LegacyManifest }
foreach ($Stage in @('build_demo_collision','bake_demo_bindings','bake_demo_hybrid_collision','validate_demo_collision','validate_demo_hybrid_blender')) {
    Write-Host "Preparing $Stage"
    $Log = Join-Path $PocRoot ".work/$Stage.log"
    & $Blender --factory-startup -b --python-exit-code 1 -P (Join-Path $PocRoot "tools/$Stage.py") *> $Log
    if ($LASTEXITCODE -ne 0) { throw "$Stage failed; see $Log" }
}
$Candidate = Join-Path $PocRoot '.work/demo/demo-hybrid-collision.json'
$AssetValidator = Join-Path $PocRoot 'tests/build/demo_asset_tests.exe'
if (Test-Path -LiteralPath $AssetValidator) {
    & $AssetValidator $Candidate *> (Join-Path $PocRoot '.work/hybrid-asset-validation.log')
    if ($LASTEXITCODE -ne 0) { throw 'Hybrid assets failed independent Blender-reference validation' }
}
$TemporaryManifest = $Manifest + '.tmp'
Copy-Item -LiteralPath $Candidate -Destination $TemporaryManifest -Force
Move-Item -LiteralPath $TemporaryManifest -Destination $Manifest -Force
Write-Host "Demo assets: $Manifest"
