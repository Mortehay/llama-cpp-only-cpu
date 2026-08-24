<#
.SYNOPSIS
  Create and attach a dedicated ext4 VHD on D: to hold model weights.

.DESCRIPTION
  MODELS_DIR has lived inside the distro's own ext4.vhdx, which sits on C:.
  That file grows and never shrinks, so every model download permanently eats
  C: until someone remembers to run compact-wsl-disk.ps1.

  The obvious alternative -- point MODELS_DIR at /mnt/d -- is wrong, and this
  project already measured why: DrvFs/9p ran at 44 MB/s against 3.9 GB/s on
  ext4, and a pipeline load took 62s from /mnt/d against 10s from ext4. See
  scripts/archive-models.sh, which is why D: is cold storage only.

  This script gets both properties at once: a real ext4 filesystem (no 9p
  penalty) whose bytes live on the D: spindle (C: stops growing). The weights
  are on D: exactly as asked, but through a filesystem Linux owns end to end.

  NOT PERSISTENT ACROSS REBOOTS. `wsl --mount` attaches for the life of the
  WSL VM. Re-run this script with -AttachOnly after every reboot, in the same
  slot as scripts/lan-expose.ps1. That is the cost of this approach and it is
  the reason the reboot runbook exists.

  REQUIREMENTS
    - Administrator. Neither diskpart nor `wsl --mount` works without it.
    - The distro is shut down during creation; every container stops. Bring the
      stack back up afterwards with `make up`.

.EXAMPLE
  # First time, from an ELEVATED PowerShell:
  powershell -ExecutionPolicy Bypass -File .\scripts\setup-models-vhd.ps1

.EXAMPLE
  # After a reboot -- attach the existing VHD, do not recreate it:
  powershell -ExecutionPolicy Bypass -File .\scripts\setup-models-vhd.ps1 -AttachOnly
#>
param(
    [string] $VhdPath    = "D:\wsl-models.vhdx",
    [int]    $SizeGB     = 25,
    [string] $Distro     = "Ubuntu",
    [string] $MountName  = "models",
    [switch] $AttachOnly
)

$ErrorActionPreference = "Stop"

# An elevated window's output is invisible to whatever launched it. This bit
# this project once for 30 minutes: diskpart's error went only to the UAC
# console and the caller saw nothing at all. Tee everything.
$logPath = Join-Path $env:TEMP "setup-models-vhd.log"
try { Start-Transcript -LiteralPath $logPath -Force | Out-Null } catch { }

function Write-Step { param([string] $m) Write-Host "==> $m" }

function Assert-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $pr = New-Object Security.Principal.WindowsPrincipal($id)
    if (-not $pr.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "Administrator required. diskpart cannot create a vdisk and 'wsl --mount' cannot attach one otherwise."
    }
}

function Get-WslDisks {
    # NAME,SIZE for every whole disk the distro can see. Compared before and
    # after the attach to identify the new device, rather than guessing /dev/sdX
    # by position -- the ordering is not stable and formatting the wrong disk
    # would destroy the Postgres data directory.
    $out = & wsl.exe -d $Distro -e bash -lc "lsblk -dn -o NAME,SIZE 2>/dev/null" 2>$null
    if (-not $out) { return @() }
    ($out -replace "`0", '') -split "`r?`n" |
        Where-Object { $_.Trim() } |
        ForEach-Object { ($_ -split '\s+')[0] }
}

Assert-Admin

$driveLetter = (Split-Path -Qualifier $VhdPath)
Write-Step "Target VHD : $VhdPath ($SizeGB GB, expandable)"
Write-Step "Log        : $logPath"

$before = @()

if (-not $AttachOnly) {
    if (Test-Path -LiteralPath $VhdPath) {
        throw "$VhdPath already exists. Re-run with -AttachOnly to attach it, or delete it first if you really mean to start over."
    }

    $free = (Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='$driveLetter'").FreeSpace
    if ($free -lt ($SizeGB * 1GB * 0.10)) {
        throw ("Not enough free space on {0} for even a sparse {1} GB VHD: {2:N1} GB free." -f $driveLetter, $SizeGB, ($free / 1GB))
    }
    Write-Step ("Free on {0}: {1:N1} GB" -f $driveLetter, ($free / 1GB))

    # diskpart reads its script file as ANSI/ASCII. $env:TEMP on this machine is
    # under a Cyrillic profile name, so use -LiteralPath for every file
    # operation here: PowerShell 5.1 otherwise reads the '~' in the 8.3 short
    # name as a home-directory reference and throws a PSArgumentException that
    # -ErrorAction cannot suppress. The VHD path itself is plain ASCII, so the
    # 8.3 dance compact-wsl-disk.ps1 needs does not apply to it.
    $script = Join-Path $env:TEMP "create-models-vhd.txt"
    $lines = @(
        "create vdisk file=`"$VhdPath`" maximum=$($SizeGB * 1024) type=expandable",
        "exit"
    )
    Set-Content -LiteralPath $script -Value $lines -Encoding Ascii

    Write-Step "Creating the VHD with diskpart"
    $out = & diskpart.exe /s $script 2>&1 | Out-String
    Write-Host $out
    Remove-Item -LiteralPath $script -Force -ErrorAction SilentlyContinue

    if (-not (Test-Path -LiteralPath $VhdPath)) {
        throw "diskpart did not create $VhdPath. Full output is above and in $logPath."
    }
}

if (-not (Test-Path -LiteralPath $VhdPath)) {
    throw "$VhdPath does not exist. Run without -AttachOnly to create it."
}

# The distro must be running for lsblk to answer, and must see a consistent
# device list before the attach.
Write-Step "Enumerating disks before attach"
& wsl.exe -d $Distro -e true | Out-Null
$before = Get-WslDisks
Write-Host ("    before: " + ($before -join ", "))

if ($AttachOnly) {
    Write-Step "Attaching $VhdPath as '$MountName'"
    & wsl.exe --mount --vhd $VhdPath --name $MountName
    if ($LASTEXITCODE -ne 0) { throw "wsl --mount failed with exit code $LASTEXITCODE." }
} else {
    Write-Step "Attaching bare so the disk can be formatted"
    & wsl.exe --mount --vhd $VhdPath --bare
    if ($LASTEXITCODE -ne 0) { throw "wsl --mount --bare failed with exit code $LASTEXITCODE." }

    $after = Get-WslDisks
    Write-Host ("    after:  " + ($after -join ", "))
    $new = @($after | Where-Object { $before -notcontains $_ })

    if ($new.Count -ne 1) {
        & wsl.exe --unmount $VhdPath 2>&1 | Out-Null
        throw ("Expected exactly one new disk, found {0} ({1}). Refusing to format: picking the wrong device would destroy the Postgres data directory." -f $new.Count, ($new -join ", "))
    }

    $dev = "/dev/$($new[0])"
    Write-Step "Formatting $dev as ext4 (label: $MountName)"
    & wsl.exe -d $Distro -u root -e bash -lc "mkfs.ext4 -F -L $MountName $dev"
    if ($LASTEXITCODE -ne 0) { throw "mkfs.ext4 failed on $dev." }

    Write-Step "Re-attaching as a mounted filesystem"
    & wsl.exe --unmount $VhdPath | Out-Null
    & wsl.exe --mount --vhd $VhdPath --name $MountName
    if ($LASTEXITCODE -ne 0) { throw "wsl --mount failed with exit code $LASTEXITCODE." }
}

$mountPoint = "/mnt/wsl/$MountName"
Write-Step "Verifying"
& wsl.exe -d $Distro -e bash -lc "df -h $mountPoint | tail -1"

# Docker runs as root inside the distro and the sprite containers write the HF
# cache; without this the first download fails on a read-only-to-user mount.
& wsl.exe -d $Distro -u root -e bash -lc "chown -R 1000:1000 $mountPoint && chmod 755 $mountPoint"

Write-Host ""
Write-Host "DONE. Model storage is live at $mountPoint (ext4, on $driveLetter)."
Write-Host ""
Write-Host "Next:"
Write-Host "  1. Set MODELS_DIR=$mountPoint in compose/develop/.env"
Write-Host "  2. Restore what you still need:"
Write-Host "       ./scripts/archive-models.sh restore <name>"
Write-Host "  3. make up"
Write-Host ""
Write-Host "AFTER EVERY REBOOT, re-attach it -- the mount does not survive:"
Write-Host "  powershell -ExecutionPolicy Bypass -File .\scripts\setup-models-vhd.ps1 -AttachOnly"

try { Stop-Transcript | Out-Null } catch { }
