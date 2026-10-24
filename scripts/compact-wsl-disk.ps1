<#
.SYNOPSIS
  Return space freed inside WSL back to the Windows drive.

.DESCRIPTION
  The WSL distro lives in a dynamically expanding ext4.vhdx. That file GROWS as
  the distro writes, and it NEVER shrinks on its own. Deleting files inside WSL
  -- models, docker build cache, anything -- frees space that `df` inside the
  distro reports as available, while Windows still sees the VHDX at its high
  water mark.

  Measured on this project 2026-08-21: `df` inside WSL reported 70 GB used while
  ext4.vhdx was 119 GB on disk. 49 GB had been freed inside and none of it had
  come back to C:, which was down to 20.5 GB free.

  This script compacts the VHDX with diskpart, which is the safe route.

  NOT USED HERE: `wsl --manage <distro> --set-sparse true`. It would make the
  disk stay compact automatically, but WSL refuses it outright:

    Sparse VHD support is currently disabled due to potential data corruption.
    To force ... --set-sparse true --allow-unsafe

  Do not pass --allow-unsafe on a distro holding a Postgres data directory.
  Running this script occasionally is the cheaper trade.

  REQUIREMENTS
    - Administrator. diskpart cannot attach a vdisk without it.
    - The distro is shut down first; this script does that for you. Every
      container stops as a result, so bring the stack back up afterwards.

.EXAMPLE
  # From an ELEVATED PowerShell (Win+X -> "Windows PowerShell (Admin)"):
  powershell -ExecutionPolicy Bypass -File .\scripts\compact-wsl-disk.ps1

.EXAMPLE
  # Report the numbers without changing anything:
  powershell -ExecutionPolicy Bypass -File .\scripts\compact-wsl-disk.ps1 -WhatIfOnly

.EXAMPLE
  # Compact AND bring the stack back without a second manual pass:
  powershell -ExecutionPolicy Bypass -File .\scripts\compact-wsl-disk.ps1 -Restore

  # From inside WSL, which is what `make compact-wsl` does for you:
  #   make compact-wsl        (elevated, -Restore, detached)
  #   make compact-wsl-dry    (numbers only, no UAC, changes nothing)
#>
param(
    [string] $Distro = "Ubuntu",
    [switch] $WhatIfOnly,
    [switch] $Restore
)

$ErrorActionPreference = "Stop"

function Get-DistroVhdx {
    param([string] $Name)
    $lxss = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss"
    if (-not (Test-Path $lxss)) { return $null }
    foreach ($k in Get-ChildItem $lxss) {
        $p = Get-ItemProperty $k.PSPath
        if ($p.DistributionName -eq $Name) {
            # BasePath can carry a \\?\ prefix; diskpart will not take that.
            $base = $p.BasePath -replace '^\\\\\?\\', ''
            $vhdx = Join-Path $base "ext4.vhdx"
            if (Test-Path $vhdx) { return $vhdx }
        }
    }
    return $null
}

function Get-AsciiSafePath {
    param([string] $Path)
    # diskpart reads its script file as ANSI/ASCII. This machine's user profile
    # is "C:\Users\<cyrillic>", and writing that path with -Encoding Ascii turns
    # the name into "??", after which diskpart fails with:
    #   The filename, directory name, or volume label syntax is incorrect.
    #   There is no virtual disk selected.
    # Quoting does not help -- the characters are already destroyed by the time
    # diskpart reads the file. The 8.3 short name is pure ASCII and names the
    # same file, so use it whenever the long path is not ASCII.
    if ($Path -match '^[\x20-\x7E]+$') { return $Path }
    try {
        $fso = New-Object -ComObject Scripting.FileSystemObject
        $short = $fso.GetFile($Path).ShortPath
        if ($short -and $short -match '^[\x20-\x7E]+$') { return $short }
    } catch {
        # fall through
    }
    return $null
}

function Get-FreeGb {
    param([string] $Drive)
    $d = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='$Drive'"
    if ($null -eq $d) { return $null }
    return [math]::Round($d.FreeSpace / 1GB, 1)
}

$vhdx = Get-DistroVhdx -Name $Distro
if (-not $vhdx) {
    Write-Error "Could not find ext4.vhdx for distro '$Distro'. Check `wsl -l -v`."
    return
}

$drive = (Split-Path -Qualifier $vhdx)
$sizeBefore = [math]::Round((Get-Item $vhdx).Length / 1GB, 1)
$freeBefore = Get-FreeGb -Drive $drive

Write-Output "Distro     : $Distro"
Write-Output "VHDX       : $vhdx"
Write-Output "Allocated  : $sizeBefore GB"
Write-Output "$drive free    : $freeBefore GB"

if ($WhatIfOnly) {
    Write-Output ""
    Write-Output "Run without -WhatIfOnly to compact. Compare 'Allocated' against"
    Write-Output "the 'used' column of 'df -h /' inside WSL -- the gap is what"
    Write-Output "compaction can return."
    return
}

$id = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($id)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error @"
Must run as Administrator -- diskpart cannot attach a vdisk otherwise.

Open Win+X -> "Windows PowerShell (Admin)", then:
  powershell -ExecutionPolicy Bypass -File "$PSCommandPath"
"@
    return
}

$keepalive = Join-Path $PSScriptRoot "wsl-keepalive.ps1"

Write-Output ""
Write-Output "Stopping the keepalive (it would restart the distro immediately)..."
# Measured 2026-10-23: `wsl --shutdown` on its own is NOT enough. The keepalive
# holds a hidden `wsl.exe ... sleep infinity`, which reattaches within seconds;
# the distro came back Running 3s after a shutdown and every container with it.
# diskpart would then be attaching a live disk. Stop it through its own
# interface, then prove the file is actually free before touching it.
if (Test-Path -LiteralPath $keepalive) {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $keepalive -Stop 2>&1 |
        ForEach-Object { Write-Output "  $_" }
}

Write-Output "Shutting down WSL (this stops every container)..."
wsl.exe --shutdown | Out-Null
Start-Sleep -Seconds 5

# An exclusive handle is the only proof the VM let go. Without this check a
# compact can run against a disk that is back in use and report success having
# reclaimed nothing -- which looks identical to "there was nothing to reclaim".
$released = $false
foreach ($attempt in 1..20) {
    try {
        $fs = [System.IO.File]::Open($vhdx, 'Open', 'ReadWrite', 'None')
        $fs.Close()
        $released = $true
        break
    } catch {
        Start-Sleep -Seconds 3
    }
}
if (-not $released) {
    Write-Error @"
The VHDX is still locked after 60s, so NOTHING was compacted.
Something keeps reattaching the distro. Find it with:
  Get-Process wsl
Close those clients -- a WSL terminal, an IDE attached to the distro, another
session running commands -- and run this again.
"@
    # The keepalive was stopped above. Put it back before giving up, or the
    # distro tears down minutes later and every container exits 255 with clean
    # logs -- a failure that reads like an OOM and is not one.
    if (Test-Path -LiteralPath $keepalive) {
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $keepalive 2>&1 |
            ForEach-Object { Write-Output "  $_" }
    }
    return
}
Write-Output "VHDX released."

# diskpart reads a script file; it will not take these on stdin reliably.
$dpPath = Get-AsciiSafePath -Path $vhdx
if (-not $dpPath) {
    Write-Error @"
The VHDX path is not ASCII and no 8.3 short name is available for it:
  $vhdx

diskpart cannot address this file. 8.3 name generation is probably disabled on
this volume; check with:
  fsutil 8dot3name query C:

Enabling it only affects files created afterwards, so the existing directories
would still have no short name. In that case compact the disk from a tool that
takes a Unicode path, or move the distro to an ASCII path with:
  wsl --manage $Distro --move D:\wsl\$Distro
"@
    return
}
if ($dpPath -ne $vhdx) {
    Write-Output "Using 8.3 short path for diskpart (long path is not ASCII):"
    Write-Output "  $dpPath"
}

$script = @"
select vdisk file="$dpPath"
attach vdisk readonly
compact vdisk
detach vdisk
exit
"@
$tmp = Join-Path $env:TEMP "compact-wsl-$([guid]::NewGuid().ToString('N')).txt"
# ASCII: diskpart does not read UTF-8 with a BOM correctly.
#
# -LiteralPath throughout, not -Path. On this machine $env:TEMP resolves to the
# 8.3 short form "C:\Users\74EA~1\AppData\Local\Temp", and PowerShell 5.1 treats
# a "~" in a path as a home-directory reference. -Path then fails with
#   An object at the specified path C:\Users\74EA~1 does not exist.
# which is a PSArgumentException from parameter binding, so -ErrorAction
# SilentlyContinue does NOT suppress it. -LiteralPath skips that expansion.
Set-Content -LiteralPath $tmp -Value $script -Encoding Ascii

Write-Output "Compacting. On a 119 GB disk this took roughly 10-15 minutes; do not interrupt it."

# Log to a file as well as the console. This script runs in an elevated window
# that nothing else can read, so a silent diskpart failure is otherwise
# invisible -- which is exactly what happened on the first attempt: the window
# stayed open, the VHDX never shrank, and the reason was only on screen.
$log = Join-Path $env:TEMP "compact-wsl-disk.log"
"=== $(Get-Date -Format s) compacting $vhdx ===" | Set-Content -LiteralPath $log -Encoding Ascii
"diskpart script:" | Add-Content -LiteralPath $log -Encoding Ascii
$script | Add-Content -LiteralPath $log -Encoding Ascii

try {
    $out = & diskpart.exe /s $tmp 2>&1 | Out-String
    Write-Output $out.Trim()
    $out | Add-Content -LiteralPath $log -Encoding Ascii
    "exit code: $LASTEXITCODE" | Add-Content -LiteralPath $log -Encoding Ascii
} catch {
    "EXCEPTION: $_" | Add-Content -LiteralPath $log -Encoding Ascii
    Write-Output "diskpart threw: $_"
} finally {
    Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
}
Write-Output "Log written to: $log"

$sizeAfter = [math]::Round((Get-Item $vhdx).Length / 1GB, 1)
$freeAfter = Get-FreeGb -Drive $drive

Write-Output ""
Write-Output "Allocated  : $sizeBefore GB -> $sizeAfter GB"
Write-Output "$drive free    : $freeBefore GB -> $freeAfter GB"
Write-Output ("Reclaimed  : {0} GB" -f [math]::Round($sizeBefore - $sizeAfter, 1))
if (-not $Restore) {
    Write-Output ""
    Write-Output "WSL is down. Bring the stack back up:"
    Write-Output "  1. powershell -ExecutionPolicy Bypass -File .\scripts\wsl-keepalive.ps1"
    Write-Output "  2. wsl:  make up"
    Write-Output "  3. wsl:  make lan-check   (only run lan-expose if that FAILS)"
    return
}

# -Restore puts the stack back in the same elevated window, so the operator is
# not left with a dead box. Docker's systemd unit is enabled, but restart
# policies are NOT reliable here (measured 2026-10-23: after a sweep, only an
# unrelated stack came back), so `make up` is what actually settles it, and it
# runs the migrations.
#
# DO NOT put `2>&1` on these native calls, and do not restore "Stop" above this
# line. Measured 2026-10-24: `$ErrorActionPreference = "Stop"` (line ~54) plus
# `2>&1` on a NATIVE command makes PowerShell 5.1 wrap every stderr line in a
# NativeCommandError and TERMINATE on the first one. docker compose writes all
# of its progress to stderr, so the restore died on compose's first line
# ("Container sprite_worker Creating"), left a half-created container whose
# name stayed reserved until dockerd restarted, and never reached lan-check.
# The fix is both halves: merge stderr INSIDE bash so PowerShell only ever sees
# stdout, and drop to "Continue" for the rest of the run.
$ErrorActionPreference = "Continue"

Write-Output ""
Write-Output "Restarting the keepalive..."
if (Test-Path -LiteralPath $keepalive) {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $keepalive |
        ForEach-Object { Write-Output "  $_" }
}

$repoWin = Split-Path $PSScriptRoot -Parent
$repoWsl = (& wsl.exe -d $Distro -e wslpath -a "$repoWin") -replace "`r|`n", ""

Write-Output ""
Write-Output "Bringing containers up ($repoWsl)..."
Write-Output "  (a cold build after a cache prune takes 15-25 min; not a hang)"
& wsl.exe -d $Distro -e bash -lc "cd '$repoWsl' && make up 2>&1" |
    ForEach-Object { Write-Output "  $_" }

# The forwards target the distro IP, which usually survives a restart -- checked
# 2026-10-23, unchanged and all five still answering. So CHECK (no UAC) rather
# than re-running lan-expose blind.
Write-Output ""
Write-Output "Checking LAN forwards..."
& wsl.exe -d $Distro -e bash -lc "cd '$repoWsl' && make lan-check 2>&1" |
    ForEach-Object { Write-Output "  $_" }
Write-Output ""
Write-Output "If lan-check printed STALE above, run from WSL:  make lan-expose"
