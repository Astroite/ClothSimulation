param(
    [ValidateRange(256, 2500)]
    [int]$TargetTriangles = 1280,
    [string]$Input = '',
    [string]$Output = '',
    [string]$Blender = 'C:\Program Files\Blender Foundation\Blender 4.5\blender.exe'
)

$ErrorActionPreference = 'Stop'
$PocRoot = Split-Path -Parent $PSScriptRoot
if (-not $Input) { $Input = Join-Path $PocRoot '.work/real_scene/ch10032_tpose/ch10032_lower.vcloth2' }
if (-not $Output) { $Output = Join-Path $PocRoot 'Assets/Meshes/CH10032_lower_sim_lod1.vcloth2' }
if ([System.IO.Path]::GetFullPath($Input) -eq [System.IO.Path]::GetFullPath($Output)) {
    throw 'LOD output must not overwrite the source cloth used by distillation'
}
foreach ($Required in @($Blender, $Input, (Join-Path $PSScriptRoot 'build_cloth_lod.py'))) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) { throw "Required LOD input is missing: $Required" }
}

& $Blender --background --factory-startup --python (Join-Path $PSScriptRoot 'build_cloth_lod.py') -- `
    --input $Input --output $Output --target-triangles $TargetTriangles
if ($LASTEXITCODE -ne 0) { throw "Blender cloth LOD bake failed with exit code $LASTEXITCODE" }

Write-Host "Standalone cloth LOD ready: $Output"
