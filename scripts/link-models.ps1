<#
.SYNOPSIS
  Surface the downloaded model cache as `models\` inside the project folder.

.DESCRIPTION
  Model weights live on WSL's ext4 filesystem, not in the repo, and that is
  deliberate: the Windows-side path is reached over 9p/DrvFs, which measured
  44 MB/s against 3.9 GB/s on ext4 for the same read. A ~7GB checkpoint is the
  difference between a two second load and a two and a half minute one - and
  something2 caps a provider call at five minutes and cannot poll, so moving
  the weights into the repo would not just be slow, it would break the
  integration.

  So this does NOT move anything. It creates a directory SYMLINK at the repo
  root pointing at the real cache, so the models are browsable from Explorer,
  VS Code and the project tree while every read still goes through ext4.
  Docker keeps bind-mounting MODELS_DIR directly and never sees this link.

  The target is read from MODELS_DIR in compose/develop/.env, so this keeps
  working if that path changes.

  REQUIRES an elevated PowerShell, or Developer Mode enabled. Windows will not
  let an unprivileged process create a symlink, and a junction (mklink /J,
  which needs no privilege) cannot point at a UNC path - which is what a WSL
  path is from the Windows side. There is no unprivileged route.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\scripts\link-models.ps1

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\scripts\link-models.ps1 -Remove
#>
param(
    [string] $Distro = "Ubuntu",
    [string] $EnvFile = "compose/develop/.env",
    [switch] $Remove
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$link = Join-Path $repo "models"

if ($Remove) {
    if (Test-Path $link) {
        $item = Get-Item $link -Force
        if ($item.LinkType) {
            # Remove-Item on a directory symlink deletes the LINK, not the
            # target - but only when the item is fetched with -Force and
            # deleted without -Recurse. Recursing here would walk into the
            # target and start deleting 30GB of model weights.
            $item.Delete()
            Write-Output "removed symlink $link (weights untouched)"
        } else {
            Write-Error "$link is a real directory, not a symlink. Refusing to delete it."
        }
    } else {
        Write-Output "nothing to remove at $link"
    }
    return
}

$envPath = Join-Path $repo $EnvFile
if (-not (Test-Path $envPath)) { Write-Error "No env file at $envPath (run 'make env')." }

$modelsDir = (Select-String -Path $envPath -Pattern '^\s*MODELS_DIR\s*=\s*(.+?)\s*$' |
              Select-Object -First 1).Matches.Groups[1].Value
if (-not $modelsDir) { Write-Error "MODELS_DIR is not set in $envPath." }

if (-not $modelsDir.StartsWith("/")) {
    Write-Output "MODELS_DIR is '$modelsDir' - a repo-relative path, so the weights are"
    Write-Output "already inside the project. Nothing to link."
    return
}

$target = "\\wsl.localhost\$Distro" + ($modelsDir -replace '/', '\')
Write-Output "MODELS_DIR : $modelsDir"
Write-Output "target     : $target"

if (-not (Test-Path $target)) {
    Write-Error "Cannot reach $target. Is the '$Distro' distro running? (wsl -l -v)"
}

if (Test-Path $link) {
    $item = Get-Item $link -Force
    if (-not $item.LinkType) {
        Write-Error "$link already exists as a real directory. Move or delete it first."
    }
    if ($item.Target -eq $target) {
        Write-Output "already linked, nothing to do."
        return
    }
    Write-Output "re-pointing existing link (was: $($item.Target))"
    $item.Delete()
}

try {
    New-Item -ItemType SymbolicLink -Path $link -Target $target -ErrorAction Stop | Out-Null
} catch {
    Write-Output ""
    Write-Output "Could not create the symlink: $($_.Exception.Message)"
    Write-Output ""
    Write-Output "Windows requires elevation (or Developer Mode) for symlinks. Re-run from"
    Write-Output "an Administrator PowerShell:"
    Write-Output "  powershell -ExecutionPolicy Bypass -File .\scripts\link-models.ps1"
    exit 1
}

Write-Output ""
Write-Output "Linked. models\ now shows the cache; the bytes stay on ext4."
Get-ChildItem $link | Select-Object -First 10 Name, Length | Format-Table -AutoSize
