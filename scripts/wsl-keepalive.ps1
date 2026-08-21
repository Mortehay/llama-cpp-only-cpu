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
  back in, or register it as a logon task with -Install (needs Administrator --
  the keepalive itself does not, only registering the task does).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\scripts\wsl-keepalive.ps1

.EXAMPLE
  # Stop it (containers will then die on the next idle sweep):
  .\scripts\wsl-keepalive.ps1 -Stop
#>
param(
    [string] $Distro = "Ubuntu",
    [switch] $Stop,
    [switch] $Install,
    [switch] $Uninstall
)

$marker = "sprite-stack-keepalive"
$taskName = "WSL keepalive"

function Test-Elevated {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    (New-Object Security.Principal.WindowsPrincipal($id)).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

if ($Install -or $Uninstall) {
    # Register-ScheduledTask REQUIRES elevation on this machine.
    #
    # The comment-based help above used to say otherwise, copied from the
    # README. Measured 2026-08-21: without elevation it fails with
    #   Register-ScheduledTask : Access is denied.
    #   HRESULT 0x80070005
    # The keepalive itself still needs no elevation - only registering it as a
    # logon task does.
    if (-not (Test-Elevated)) {
        Write-Error @"
-Install/-Uninstall require Administrator. Register-ScheduledTask fails with
"Access is denied" (HRESULT 0x80070005) otherwise.

Open Win+X -> "Windows PowerShell (Admin)", then:
  powershell -ExecutionPolicy Bypass -File "$PSCommandPath" -Install
"@
        return
    }

    if ($Uninstall) {
        if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
            Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
            Write-Output "Removed the '$taskName' logon task."
        } else {
            Write-Output "No '$taskName' task registered."
        }
        return
    }

    $action = New-ScheduledTaskAction -Execute 'powershell.exe' `
        -Argument ('-WindowStyle Hidden -ExecutionPolicy Bypass -File "' +
                   $PSCommandPath + '" -Distro ' + $Distro)
    $trigger = New-ScheduledTaskTrigger -AtLogOn
    # ExecutionTimeLimit 0 = never kill it. The default is 3 days, after which
    # the task engine would stop the keepalive and every container would die
    # exactly as if it had never been started.
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero)
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
        -Settings $settings -Force `
        -Description "Holds the WSL2 distro open so Docker containers survive. See scripts/wsl-keepalive.ps1" | Out-Null
    Write-Output "Registered '$taskName' to run at logon."
    Write-Output "It starts on your NEXT logon; this session still needs the script run once."
    return
}

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
