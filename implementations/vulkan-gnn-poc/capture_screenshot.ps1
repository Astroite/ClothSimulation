param(
    [ValidateSet(16, 32, 64)]
    [int]$Grid = 32,
    [ValidateRange(0, 30000)]
    [int]$WarmupMilliseconds = 2000
)

$ErrorActionPreference = 'Stop'
$PocRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$UpstreamRoot = Join-Path $PocRoot '.work/Vulkan'
$Executable = Join-Path $UpstreamRoot 'build-gnn/bin/gnncloth.exe'
$Output = Join-Path $PocRoot 'results/gnn_cloth.png'

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
}
'@

$Process = Start-Process -FilePath $Executable -ArgumentList @('--gnn-grid', $Grid, '-s', 'hlsl', '-vs', '-w', '1280', '-h', '720') -WorkingDirectory $UpstreamRoot -WindowStyle Normal -PassThru
try {
    for ($Attempt = 0; $Attempt -lt 100 -and $Process.MainWindowHandle -eq 0; ++$Attempt) {
        Start-Sleep -Milliseconds 20
        $Process.Refresh()
    }
    if ($Process.MainWindowHandle -eq 0) { throw 'The Vulkan window did not appear' }
    Start-Sleep -Milliseconds $WarmupMilliseconds
    [void][GnnWindowCapture]::PostMessage($Process.MainWindowHandle, 0x0100, [IntPtr]0x50, [IntPtr]0)
    [void][GnnWindowCapture]::PostMessage($Process.MainWindowHandle, 0x0101, [IntPtr]0x50, [IntPtr]0)
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
        $Graphics.CopyFromScreen($Rect.Left, $Rect.Top, 0, 0, $Bitmap.Size)
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
