$ErrorActionPreference = 'Stop'
$PocRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$UpstreamRoot = Join-Path $PocRoot '.work/Vulkan'
$Executable = Join-Path $UpstreamRoot 'build-gnn/bin/gnncloth.exe'

if (-not ('GnnSmokeWindow' -as [type])) {
    Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class GnnSmokeWindow {
    [DllImport("user32.dll")]
    public static extern bool PostMessage(IntPtr handle, uint message, IntPtr wParam, IntPtr lParam);
    [DllImport("user32.dll")]
    public static extern bool ShowWindow(IntPtr handle, int command);
}
'@
}

$Process = Start-Process -FilePath $Executable -ArgumentList @('--gnn-grid', '32', '-s', 'hlsl', '-v', '-vl') -WorkingDirectory $UpstreamRoot -WindowStyle Normal -PassThru
try {
    for ($Attempt = 0; $Attempt -lt 100 -and $Process.MainWindowHandle -eq 0; ++$Attempt) {
        Start-Sleep -Milliseconds 20
        $Process.Refresh()
    }
    if ($Process.MainWindowHandle -eq 0) { throw 'The Vulkan smoke-test window did not appear' }
    [void][GnnSmokeWindow]::ShowWindow($Process.MainWindowHandle, 0)
    # Exercise the default moving kinematic sphere on the GNN/XPBD path before
    # switching solvers, so synchronization validation observes moving contact.
    Start-Sleep -Milliseconds 2000
    foreach ($Key in @(0x47, 0x52, 0x47, 0x52)) {
        [void][GnnSmokeWindow]::PostMessage($Process.MainWindowHandle, 0x0100, [IntPtr]$Key, [IntPtr]0)
        [void][GnnSmokeWindow]::PostMessage($Process.MainWindowHandle, 0x0101, [IntPtr]$Key, [IntPtr]0)
        Start-Sleep -Milliseconds 250
    }
} finally {
    if (-not $Process.HasExited) {
        [void][GnnSmokeWindow]::PostMessage($Process.MainWindowHandle, 0x0010, [IntPtr]0, [IntPtr]0)
        if (-not $Process.WaitForExit(2000)) { Stop-Process -Id $Process.Id -Force }
    }
}
Write-Host 'Moving-sphere contact plus mass-spring/GNN switch/reset smoke test completed.'
