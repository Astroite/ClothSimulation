# GPU-free regression checks. Everything here runs without a Vulkan device, so it
# is suitable for a CI runner; verify.ps1 and benchmark.ps1 need real hardware.
#
# To wire this to a provider once the repository has a remote, call this script
# from the job after installing the Vulkan SDK (needed for dxc and spirv-val).
$ErrorActionPreference = 'Stop'
$PocRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

$Python = Join-Path $PocRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    if (Get-Command py -ErrorAction SilentlyContinue) { $Python = 'py' } else { $Python = 'python' }
}

Write-Host '== Python reference and negative loader tests'
& $Python (Join-Path $PocRoot 'model/verify_export.py')
if ($LASTEXITCODE -ne 0) { throw 'Python reference verification failed' }

Write-Host '== C++ binary-format loader tests'
$FormatTest = Join-Path $PocRoot 'tests/build/vgnn_format_test.exe'
if (Test-Path -LiteralPath $FormatTest) {
    & $FormatTest (Join-Path $PocRoot 'model/artifacts/model.bin') (Join-Path $PocRoot 'model/artifacts/golden.bin')
    if ($LASTEXITCODE -ne 0) { throw 'C++ binary-format verification failed' }
} else {
    Write-Host "  skipped: $FormatTest not built"
}

Write-Host '== Generated shader constants match vgnn.py'
& $Python (Join-Path $PocRoot 'model/write_shader_constants.py') --check
if ($LASTEXITCODE -ne 0) { throw 'Generated shader constants are stale' }

# The committed SPIR-V is what makes the Vulkan SDK optional at run time, so it
# has to stay in step with the HLSL sources. DXC output is byte-reproducible for
# these shaders, which makes an exact comparison the right check.
Write-Host '== Committed SPIR-V matches a fresh compile'
$ShaderDir = Join-Path $PocRoot 'overlay/shaders/hlsl/gnncloth'
$Backup = Join-Path ([System.IO.Path]::GetTempPath()) ("gnncloth_spv_" + [System.Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $Backup | Out-Null
try {
    Copy-Item -Path (Join-Path $ShaderDir '*.spv') -Destination $Backup -Force
    & $Python (Join-Path $PocRoot 'tools/compile_shaders.py')
    if ($LASTEXITCODE -ne 0) { throw 'Shader compilation failed' }

    $Drifted = @()
    foreach ($Original in Get-ChildItem -LiteralPath $Backup -Filter '*.spv') {
        $Rebuilt = Join-Path $ShaderDir $Original.Name
        if (-not (Test-Path -LiteralPath $Rebuilt)) { $Drifted += "$($Original.Name) (missing after rebuild)"; continue }
        $a = [System.IO.File]::ReadAllBytes($Original.FullName)
        $b = [System.IO.File]::ReadAllBytes($Rebuilt)
        $same = $a.Length -eq $b.Length
        if ($same) {
            for ($i = 0; $i -lt $a.Length; ++$i) { if ($a[$i] -ne $b[$i]) { $same = $false; break } }
        }
        if (-not $same) { $Drifted += $Original.Name }
    }
    if ($Drifted.Count -gt 0) {
        throw ("Committed SPIR-V is stale for: " + ($Drifted -join ', ') + ". Run build.ps1 and commit the regenerated .spv.")
    }
    Write-Host "  $((Get-ChildItem -LiteralPath $Backup -Filter '*.spv').Count) shaders match"
} finally {
    Remove-Item -LiteralPath $Backup -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host 'GPU-free checks passed: Python reference, binary format, SPIR-V freshness.'
