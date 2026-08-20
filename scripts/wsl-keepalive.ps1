<#
.SYNOPSIS
  Keep the WSL distro alive so Docker containers keep running.

.DESCRIPTION
  WSL2 terminates a distro once no `wsl.exe` client process is attached to it.
  That takes down systemd, dockerd and every running container -- which presents
  as containers exiting on their own with status 0 or 255, minutes after they
  started, with nothing wrong in their logs.

  `vmIdleTimeout` in .wslconfig does NOT fix this on Windows 10: that setting is
  Windows 11 only and is silently ignored here.

  The reliable fix is to keep one long-lived WSL process running. This script
  launches a hidden `wsl.exe ... sleep infinity` and is idempotent -- running it
  twice will not start a second one.

  NOTE: the keepalive dies when you log off or reboot. Re-run it after logging
  back in, or register it as a logon scheduled task:

    $a = New-ScheduledTaskAction -Execute 'powershell.exe' `
         -Argument '-WindowStyle Hidden -ExecutionPolicy Bypass -File "<full path>\wsl-keepalive.ps1"'
    $t = New-ScheduledTaskTrigger -AtLogOn
    Register-ScheduledTask -TaskName 'WSL keepalive' -Action $a -Trigger $t

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\scripts\wsl-keepalive.ps1

.EXAMPLE
  # Stop it (containers will then die on the next idle sweep):
  .\scripts\wsl-keepalive.ps1 -Stop
#>
param(
    [string] $Distro = "Ubuntu",
    [switch] $Stop
)

$marker = "sprite-stack-keepalive"

function Get-Keepalive {
    Get-CimInstance Win32_Process -Filter "Name = 'wsl.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine -match $marker }
}

if ($Stop) {
    $procs = Get-Keepalive
    if (-not $procs) { Write-Output "No keepalive running."; return }
    $procs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
    Write-Output "Keepalive stopped. WSL will shut the distro down when idle."
    return
}

if (Get-Keepalive) {
    Write-Output "Keepalive already running for '$Distro'."
    return
}

# The marker string is only there so this script can find its own process later;
# "sleep infinity" is what actually holds the distro open.
Start-Process -FilePath "wsl.exe" `
    -ArgumentList @('-d', $Distro, '--', 'sh', '-c', "# $marker`nexec sleep infinity") `
    -WindowStyle Hidden

Start-Sleep -Seconds 2
if (Get-Keepalive) {
    Write-Output "Keepalive started -- '$Distro' will stay up until you log off or run -Stop."
} else {
    Write-Error "Keepalive failed to start."
}
