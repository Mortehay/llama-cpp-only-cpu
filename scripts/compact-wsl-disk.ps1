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
#>
param(
    [string] $Distro = "Ubuntu",
    [switch] $WhatIfOnly
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

Write-Output ""
Write-Output "Shutting down WSL (this stops every container)..."
wsl.exe --shutdown | Out-Null
Start-Sleep -Seconds 10

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
Write-Output ""
Write-Output "WSL is down. Bring the stack back up:"
Write-Output "  1. powershell -ExecutionPolicy Bypass -File .\scripts\wsl-keepalive.ps1"
Write-Output "  2. wsl:  sudo service docker start  &&  make up"
Write-Output "  3. ELEVATED powershell: .\scripts\lan-expose.ps1   (WSL IP changed)"
