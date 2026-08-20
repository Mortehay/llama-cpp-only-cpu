<#
.SYNOPSIS
  Publish WSL2-hosted service ports to the home LAN. Run as Administrator.

.DESCRIPTION
  WSL2 sits behind NAT: a port bound inside WSL is reachable from Windows via
  localhost, but NOT from other machines on the LAN. WSL's networkingMode=mirrored
  would remove the need for this, but it requires Windows 11 22H2+ and this host
  is Windows 10 -- so we forward explicitly.

  This script is idempotent: it clears and re-adds each proxy entry.

  RE-RUN THIS AFTER EVERY WSL RESTART. The WSL IP is assigned per boot and the
  portproxy entries silently point at a stale address once it changes.

.EXAMPLE
  # From an elevated PowerShell:
  .\scripts\lan-expose.ps1

.EXAMPLE
  # Undo everything this script created:
  .\scripts\lan-expose.ps1 -Remove
#>
param(
    [int[]] $Ports = @(8001, 7860, 8002, 3000, 3001),
    [string] $Distro = "Ubuntu",
    [switch] $Remove
)

$ErrorActionPreference = "Stop"
$FirewallRule = "WSL2 sprite services (LAN)"

if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()
        ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "Must run as Administrator (netsh portproxy and firewall rules require elevation)."
}

if ($Remove) {
    foreach ($p in $Ports) {
        netsh interface portproxy delete v4tov4 listenport=$p listenaddress=0.0.0.0 2>$null | Out-Null
        Write-Output "removed proxy for port $p"
    }
    Remove-NetFirewallRule -DisplayName $FirewallRule -ErrorAction SilentlyContinue
    Write-Output "removed firewall rule '$FirewallRule'"
    return
}

# Ask WSL for its own address. `hostname -I` returns the eth0 address first.
$wslIp = (wsl -d $Distro -- bash -lc "hostname -I | awk '{print `$1}'").Trim()
if (-not $wslIp) { Write-Error "Could not determine the WSL IP for distro '$Distro'. Is it running?" }
Write-Output "WSL ($Distro) address: $wslIp"

foreach ($p in $Ports) {
    netsh interface portproxy delete v4tov4 listenport=$p listenaddress=0.0.0.0 2>$null | Out-Null
    netsh interface portproxy add v4tov4 listenport=$p listenaddress=0.0.0.0 `
        connectport=$p connectaddress=$wslIp | Out-Null
    Write-Output "0.0.0.0:$p  ->  ${wslIp}:$p"
}

Remove-NetFirewallRule -DisplayName $FirewallRule -ErrorAction SilentlyContinue
New-NetFirewallRule -DisplayName $FirewallRule -Direction Inbound -Action Allow `
    -Protocol TCP -LocalPort $Ports -Profile Private | Out-Null
Write-Output "firewall rule '$FirewallRule' allows TCP $($Ports -join ',') on Private networks only"

Write-Output ""
Write-Output "--- active proxies ---"
netsh interface portproxy show v4tov4

$lanIp = (Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "172.*" -and $_.PrefixOrigin -ne "WellKnown" } |
    Select-Object -First 1).IPAddress
Write-Output ""
Write-Output "Other machines on the LAN should reach the sprite API at: http://${lanIp}:8001"
