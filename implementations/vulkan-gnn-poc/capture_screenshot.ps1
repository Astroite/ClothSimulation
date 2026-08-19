param(
    [ValidateSet('Grid', 'CH10032', 'HoodGrid64')]
    [string]$Scene = 'Grid',
    [ValidateSet(16, 32, 64)]
    [int]$Grid = 32,
    [string]$Motion = 'ch10032_sprint',
    [ValidateSet('Fine15', 'PostCvpr', 'TinyHood', 'Toy2L')]
    [string]$Solver = 'Fine15',
    [ValidateRange(0, 10000)]
    [int]$SimulationSteps = 0,
    [ValidateRange(0, 30000)]
    [int]$WarmupMilliseconds = 2000,
    [string]$AssetRoot = ''
)

$ErrorActionPreference = 'Stop'
$PocRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$UpstreamRoot = Join-Path $PocRoot '.work/Vulkan'
$Executable = Join-Path $UpstreamRoot 'build-gnn/bin/gnncloth.exe'
$OutputName = if ($Solver -eq 'PostCvpr' -and $Scene -eq 'HoodGrid64') { 'results/postcvpr_grid64.png' } elseif ($Solver -eq 'PostCvpr') { 'results/postcvpr_ch10032_tpose.png' } elseif ($Solver -eq 'TinyHood' -and $Scene -eq 'HoodGrid64') { 'results/tinyhood_grid64.png' } elseif ($Solver -eq 'TinyHood' -and $Motion -eq 'ch10032_tpose') { 'results/tinyhood_ch10032_tpose.png' } elseif ($Solver -eq 'TinyHood') { 'results/tinyhood_ch10032.png' } elseif ($Scene -eq 'HoodGrid64') { 'results/hood_grid64_fine15.png' } elseif ($Scene -eq 'CH10032' -and $Motion -eq 'ch10032_tpose' -and $Solver -eq 'Toy2L') { 'results/hood_ch10032_tpose_toy2l.png' } elseif ($Scene -eq 'CH10032' -and $Motion -eq 'ch10032_tpose') { 'results/hood_ch10032_tpose.png' } elseif ($Scene -eq 'CH10032') { 'results/hood_ch10032.png' } else { 'results/gnn_cloth.png' }
$Output = Join-Path $PocRoot $OutputName
$Arguments = @('-s', 'hlsl', '-vs', '-w', '1280', '-h', '720')
if ($Scene -in @('CH10032', 'HoodGrid64')) {
    $IsHoodGrid = $Scene -eq 'HoodGrid64'
    if ($IsHoodGrid) { $Motion = 'hood_grid64' }
    if (-not $AssetRoot) { $AssetRoot = Join-Path $PocRoot ($IsHoodGrid ? '.work/real_scene/hood_grid64' : ".work/real_scene/$Motion") }
    # The executable's working directory is the upstream tree, not the caller's, so resolve any
    # caller-supplied relative path here. The default above is already absolute.
    $AssetRoot = [System.IO.Path]::GetFullPath($AssetRoot)
    $Arguments += @(
        '--scene', ($IsHoodGrid ? 'hoodgrid' : 'ch10032'), '--motion', $Motion, '--solver', ($Solver -eq 'Toy2L' ? 'toy2l' : ($Solver -eq 'PostCvpr' ? 'postcvpr' : ($Solver -eq 'TinyHood' ? 'tinyhood' : 'fine15'))),
        '--asset-root', $AssetRoot,
        '--hood-model', (Join-Path $PocRoot ($Solver -eq 'PostCvpr' ? '.work/hood_data/postcvpr.vhood' : ($Solver -eq 'TinyHood' ? '.work/hood_data/tinyhood64x4.vhood' : '.work/hood_data/fine15.vhood')))
    )
    if ($SimulationSteps -gt 0) { $Arguments += @('--hood-pause-after', $SimulationSteps) }
} else {
    $Arguments += @('--gnn-grid', $Grid)
}

Add-Type -AssemblyName System.Drawing
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class GnnWindowCapture {
    [StructLayout(LayoutKind.Sequential)]
    public struct Rect { public int Left, Top, Right, Bottom; }
    [DllImport("user32.dll")]
    public static extern bool GetWindowRect(IntPtr handle, out Rect rect);
    [DllImport("user32.dll")]
    public static extern bool SetForegroundWindow(IntPtr handle);
    [DllImport("user32.dll")]
    public static extern bool ShowWindow(IntPtr handle, int command);
    [DllImport("user32.dll")]
    public static extern bool SetWindowPos(IntPtr handle, IntPtr insertAfter, int x, int y, int width, int height, uint flags);
    [DllImport("user32.dll")]
    public static extern bool PostMessage(IntPtr handle, uint message, IntPtr wParam, IntPtr lParam);
    [DllImport("user32.dll")]
    public static extern bool PrintWindow(IntPtr handle, IntPtr deviceContext, uint flags);
}
'@

$Process = Start-Process -FilePath $Executable -ArgumentList $Arguments -WorkingDirectory $UpstreamRoot -WindowStyle Normal -PassThru
try {
    for ($Attempt = 0; $Attempt -lt 100 -and $Process.MainWindowHandle -eq 0; ++$Attempt) {
        Start-Sleep -Milliseconds 20
        $Process.Refresh()
    }
    if ($Process.MainWindowHandle -eq 0) { throw 'The Vulkan window did not appear' }
    Start-Sleep -Milliseconds $WarmupMilliseconds
    if ($SimulationSteps -eq 0) {
        [void][GnnWindowCapture]::PostMessage($Process.MainWindowHandle, 0x0100, [IntPtr]0x50, [IntPtr]0)
        [void][GnnWindowCapture]::PostMessage($Process.MainWindowHandle, 0x0101, [IntPtr]0x50, [IntPtr]0)
    }
    [void][GnnWindowCapture]::ShowWindow($Process.MainWindowHandle, 9)
    [void][GnnWindowCapture]::SetForegroundWindow($Process.MainWindowHandle)
    [void][GnnWindowCapture]::SetWindowPos($Process.MainWindowHandle, [IntPtr](-1), 40, 40, 1280, 720, 0x0040)
    Start-Sleep -Milliseconds 300
    $Rect = New-Object GnnWindowCapture+Rect
    if (-not [GnnWindowCapture]::GetWindowRect($Process.MainWindowHandle, [ref]$Rect)) { throw 'GetWindowRect failed' }
    $Width = $Rect.Right - $Rect.Left
    $Height = $Rect.Bottom - $Rect.Top
    $Bitmap = New-Object System.Drawing.Bitmap($Width, $Height)
    $Graphics = [System.Drawing.Graphics]::FromImage($Bitmap)
    try {
        try {
            $Graphics.CopyFromScreen($Rect.Left, $Rect.Top, 0, 0, $Bitmap.Size)
        } catch {
            $DeviceContext = $Graphics.GetHdc()
            try {
                if (-not [GnnWindowCapture]::PrintWindow($Process.MainWindowHandle, $DeviceContext, 2)) {
                    throw 'CopyFromScreen and PrintWindow both failed'
                }
            } finally {
                $Graphics.ReleaseHdc($DeviceContext)
            }
        }
        $Bitmap.Save($Output, [System.Drawing.Imaging.ImageFormat]::Png)
    } finally {
        $Graphics.Dispose()
        $Bitmap.Dispose()
    }
    Write-Host "Wrote $Output"
} finally {
    if (-not $Process.HasExited) {
        [void]$Process.CloseMainWindow()
        if (-not $Process.WaitForExit(2000)) { Stop-Process -Id $Process.Id -Force }
    }
}
