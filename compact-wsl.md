# Compacting the WSL disk (and bringing the stack back)

Runbook. The *why* and the measurements live in
[`.ai/project-context.md`](.ai/project-context.md) under "Disk space: `df`
inside WSL is not free disk space"; this file is the ordered checklist so a
session does not have to rediscover the steps. Last run: **2026-09-12** —
C: reported ~17 GB free before, **50.8 GB free after** (2026-08-21 run:
ext4.vhdx 119.0 GB -> 71.7 GB, C: 25.9 GB -> 73.2 GB).

## Why this is needed at all

The Ubuntu distro lives in a dynamically expanding `ext4.vhdx` on C:. It grows
whenever the distro writes and **never shrinks on its own**. Deleting inside
WSL (models, Docker build cache) frees space that `df` reports as available
while Windows still sees the file at its high-water mark. `df` inside WSL is
therefore meaningless for "can I download this"; only the Windows side counts.

`wsl --manage <distro> --set-sparse true` would make it shrink automatically,
but WSL refuses it without `--allow-unsafe` and the distro holds the Postgres
data directory. Occasional manual compaction is the cheaper trade.

## When to run it

1. Check the Windows side, not `df`:
   ```powershell
   Get-PSDrive C, D | Select-Object Name, @{n='FreeGB';e={[math]::Round($_.Free/1GB,1)}}
   ```
2. Free space *inside* the distro first — compaction only returns space that
   is already free in ext4. The usual biggest consumer is not the models:
   ```bash
   docker system df            # 2026-08-23: 40.9 GB build cache, 31.8 GB reclaimable
   docker builder prune        # cost: next no-cache rebuild re-pulls the torch wheel (15-25 min)
   ```
3. Then compact.

## Procedure

All of this from **one elevated PowerShell** (Win+X -> "Windows PowerShell
(Admin)"). `diskpart` cannot attach a vdisk unelevated, and an elevated
window's output is invisible to anything else — that is why the script tees to
a log.

```powershell
cd "C:\Users\Нє\Projects\Нова папка\llama-cpp-only-cpu"
powershell -ExecutionPolicy Bypass -File .\scripts\compact-wsl-disk.ps1
```

What it does, in order: `wsl --shutdown` (**every container stops**, the
keepalive dies), resolves the VHDX to its 8.3 short name (the Cyrillic profile
path breaks diskpart otherwise), runs `diskpart compact vdisk`, detaches.
About **7 minutes**. Output is teed to `%TEMP%\compact-wsl-disk.log`; if the
window shows nothing useful, read that file. Exit code 0 and
`DiskPart successfully compacted the virtual disk file` is success.

Dry run that only reports the numbers: add `-WhatIfOnly`.

## Bringing the stack back

`wsl --shutdown` undoes three things that do not come back on their own. Same
elevated window, in this order:

```powershell
# 1. Models VHD — ONLY if D:\wsl-models.vhdx exists (see "Deferred" below).
#    As of 2026-09-12 it does NOT exist and this line throws
#    "D:\wsl-models.vhdx does not exist" — skip it until the move is done.
powershell -ExecutionPolicy Bypass -File .\scripts\setup-models-vhd.ps1 -AttachOnly

# 2. LAN port-proxies. The WSL IP changes per boot; stale entries 404 silently.
powershell -ExecutionPolicy Bypass -File .\scripts\lan-expose.ps1

# 3. Keepalive. Without it the distro tears down minutes later and containers
#    exit 0 with clean logs.
powershell -ExecutionPolicy Bypass -File .\scripts\wsl-keepalive.ps1
```

Then from **inside WSL** (Docker Engine runs there, not Docker Desktop):

```bash
cd "/mnt/c/Users/Нє/Projects/Нова папка/llama-cpp-only-cpu"
make up
```

`make up` runs the model downloader before `compose up`, so it is also the
step that would silently re-download weights if `MODELS_DIR` pointed at an
empty directory. Docker itself auto-starts (systemd unit is enabled); no
`service docker start` needed.

Verify before sending traffic:

```bash
docker ps                   # api, worker, db, redis all Up
make gpu-health             # the worker's CUDA context, not just nvidia-smi
```

## Deferred: moving `MODELS_DIR` onto a D: VHD

**Current reality (2026-09-12), which contradicts project-context.md:**
`MODELS_DIR=/home/markunn/sprite-data/models` (`compose/develop/.env`) — inside
the C: VHDX, 28 GB. **No models VHD exists**: `D:\wsl-models.vhdx` is absent,
`lsblk` shows no attached disk, `/mnt/wsl/models` does not exist. D: holds only
the cold archive (`D:\wsl-model-archive`, DrvFs — never run a model from it).

**Decision: not now.** Reasons:

- Pressure is gone after compaction (50.8 GB free; the audio spike needs
  ~10 GB of weights plus build layers).
- Moving weights does not end compaction — the build cache stays on C: and
  was the larger consumer.
- It adds a real failure mode the current setup does not have: if the VHD is
  not re-attached before `make up`, Docker bind-mounts an **empty**
  `/mnt/wsl/models`, the downloader refills it on the C: VHDX, and nothing
  warns. Needs a guard in `make up` first.
- Both C: and D: are SATA SSDs, so there is no speed gain either way; the only
  win is which drive the bytes occupy.

**When it is worth doing:** C: is tight again *because of weights* (not build
cache), or the roster grows past what compaction keeps recovering. D: has
69.5 GB free.

**How, when the time comes** (elevated PowerShell unless noted):

1. Create — the script default is `-SizeGB 25`, which cannot hold the
   current 28 GB, let alone +10 GB. Size it for the roster:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\scripts\setup-models-vhd.ps1 -SizeGB 50
   ```
   Shuts the distro down, creates + formats ext4, mounts at `/mnt/wsl/models`.
2. Copy, inside WSL, with the stack down:
   `rsync -a /home/markunn/sprite-data/models/ /mnt/wsl/models/`
   Compare `du -sh` on both sides before deleting the source.
3. Repoint `MODELS_DIR=/mnt/wsl/models` in `compose/develop/.env`.
4. Add a guard so `make up` refuses when `/mnt/wsl/models` is not a mount
   point (`mountpoint -q /mnt/wsl/models`), otherwise step 1 of "Bringing the
   stack back" becomes a silent re-download.
5. `make up`, then `make gpu-health` and one generation.
6. Delete the old directory inside WSL, then run this runbook once more —
   the freed 28 GB only reaches C: after compaction.
7. Docs: correct project-context.md "D: now holds the models" (it will be
   true only then), and un-comment step 1 above.

## Traps already measured (do not re-learn)

- `-Encoding Ascii` turns the Cyrillic profile name into `??` — the script
  uses the 8.3 short path for diskpart; keep that when editing it.
- `~` in that short name is read by PowerShell 5.1 as home-dir; use
  `-LiteralPath` for anything under `$env:TEMP`.
- `scripts/*.ps1` must stay ASCII-only (5.1 reads ANSI without a BOM).
- `wsl-keepalive.ps1 -Install` (logon task) needs Administrator; running the
  keepalive itself does not.
