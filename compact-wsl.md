# Compacting the WSL disk (and bringing the stack back)

Runbook. The *why* and the measurements live in
[`.ai/project-context.md`](.ai/project-context.md) under "Disk space: `df`
inside WSL is not free disk space"; this file is the ordered checklist so a
session does not have to rediscover the steps.

**Last run: 2026-10-24** — ext4.vhdx 342.8 GB -> **221.5 GB**, C: 16.3 GB free
-> **137.6 GB free**. **Reclaimed 121.3 GB**, by far the best result of any run
— because for once the deletion pass came first and gave compaction 137 GB of
slack to work with instead of 18 GB.

> **Quote the settled figure, not the peak: C: ended at 111.1 GB free, a net
> +94.8 GB.** Compaction left 137.6 GB, then `make up` rebuilt the sprite
> images against an empty build cache and took ~26 GB straight back (images
> 22.28 GB -> 40.36 GB, VHDX 221.5 -> 248.0 GB) before a single sprite was
> generated. **That rebuild cost is part of the price of `builder prune -a`
> and should be budgeted with it.** It is also the clearest argument for the
> cache cap in [0018](.ai/decisions/0018-disk-retention-is-a-policy.md): the
> cache is now 0 B and capped at 20 GB, so the next round cannot climb back to
> 81 GB.

Earlier runs: 2026-10-23 ext4.vhdx
288.0 GB -> 252.6 GB, C: 73.2 -> 108.7 GB free (35.4 GB in 46 minutes);
2026-09-12 C: ~17 GB -> 50.8 GB free; 2026-08-21 ext4.vhdx 119.0 GB -> 71.7 GB,
C: 25.9 GB -> 73.2 GB.

**What the 2026-10-24 pass deleted**, in the order that matters (`/home` and
every model left untouched): the `mesh-worker` + `rig-worker` images and their
containers (47.5 GB, lane closed by 0016), `nvidia/cuda:12.8.1-devel-ubuntu24.04`
(14.6 GB orphan), the `aod-gpubox` stack with its 22.65 GB `aod_ollama_models`
volume (`compose down` **without** `-v`, keeping `aod_pg_data`), then
`docker builder prune -a -f`. ext4 used went **325 GB -> 206 GB**.

> **The ordering rule paid out, measured for a third time.** Before the images
> were deleted the build cache reported 81.22 GB total with only **13.66 GB
> reclaimable**. After deleting them -- without touching the cache -- the same
> cache reported **49.77 GB** reclaimable, and the prune then released the
> **full 81.22 GB**. Pruning first would have left roughly 67 GB on the disk.

> **The same day that run finished, the disk refilled: 252.6 GB -> 342.8 GB,
> C: 108.7 GB free -> 16.3 GB free. +90 GB in hours, from ONE experiment.**
> Measured 2026-10-23 evening. `mesh-worker` (22.3 GB) and `rig-worker`
> (25.2 GB) were built for the [0016](.ai/decisions/0016-rig-gate-retest-and-off-card-3d.md)
> rig-gate retest, `sprite-worker`/`sprite-generator` were rebuilt, and the
> build cache regrew from ~20 GB post-prune to **81.22 GB**. `/home` did not
> grow at all.
>
> Two lessons, both of which contradict what this file said before:
>
> - **Compaction was not available as the fix.** ext4 reported **325 GB used
>   inside a 342.8 GB file** — ~18 GB of slack. Every earlier run had tens of
>   GB of gap to harvest; this one had none. Compaction returns only what is
>   already free in ext4, so when the distro is genuinely full it yields
>   nothing. Delete inside first, *then* compact. Check `df` against the VHDX
>   size before budgeting an hour for `compact vdisk`.
> - **This is the real mechanism behind "WSL ate the disk", and it is not
>   about weights.** The VHDX never shrinks, so WSL converts every transient
>   build *peak* into permanent allocation. A 48 GB throwaway experiment is a
>   48 GB permanent cost until someone runs a ~1 hour compaction. The fix is
>   retention policy, not reclamation —
>   [0018](.ai/decisions/0018-disk-retention-is-a-policy.md).

## The short version

```bash
make compact-wsl-dry    # numbers only. No UAC, no shutdown, nothing changes.
make compact-wsl        # the real thing. UAC prompt, WSL goes down, comes back.
```

`make compact-wsl` is **not** interactive from this shell and does not block:
the make process running it lives inside the distro the script shuts down, so
waiting here would mean waiting to be killed. The elevated window is a Windows
process, outlives the shutdown, and restores the stack itself. **Watch that
window, not the shell you typed into.**

## Why this is needed at all

The Ubuntu distro lives in a dynamically expanding `ext4.vhdx` on C:. It grows
whenever the distro writes and **never shrinks on its own**. Deleting inside
WSL (models, Docker build cache) frees space that `df` reports as available
while Windows still sees the file at its high-water mark. `df` inside WSL is
therefore meaningless for "can I download this"; only the Windows side counts.

`wsl --manage <distro> --set-sparse true` would make it shrink automatically,
but WSL refuses it outright:

```
Sparse VHD support is currently disabled due to potential data corruption.
To force ... --set-sparse true --allow-unsafe
```

Do not pass `--allow-unsafe` on a distro holding a Postgres data directory.
Occasional manual compaction is the cheaper trade.

## Free space INSIDE the distro first

Compaction only returns space that is already free in ext4, so this step is
what actually determines the result. **`fstrim` is not part of it** — `/` is
mounted with `discard` (`findmnt -no OPTIONS /`) and `fstrim --dry-run /`
reports `0 B`, so the host already learns about freed blocks as they are freed.
Nothing to prepare.

Check the Windows side, never `df`:

```powershell
Get-CimInstance Win32_LogicalDisk | Select-Object DeviceID,
    @{n='FreeGB';e={[math]::Round($_.FreeSpace/1GB,1)}}
```

Then find the space. **The biggest consumer is usually not the models** —
measured 2026-10-23 evening, Docker held **173 GB** (`/var/lib/containerd`
151 GB + `/var/lib/docker` 22 GB) against **149 GB** for `/home`. Measure all
three, because images and build cache live under `containerd` on this box:

```bash
# Use `wsl -d Ubuntu -u root`: sudo in this distro PROMPTS for a password,
# and a sudo command with no tty hangs forever instead of failing.
du -sxh /var/lib/containerd /var/lib/docker /home
docker system df
```

### Delete dead images BEFORE pruning the cache

**The order changes the yield, and getting it backwards is the most common way
this step under-delivers.** `docker builder prune` reclaims only cache layers
that are *private* to the cache; anything still shared with a live image is
netted off. So an image you are about to delete anyway is, until you delete it,
actively protecting tens of GB of cache from the prune.

Measured both ways on 2026-10-23. In the morning, prune-first: cache total
76.49 GB, 33.12 GB shared with live images, **56 GB freed**. That evening,
*before* the dead images were removed, the cache read 81.22 GB total with only
**13.66 GB reclaimable** — because 47.5 GB of mesh/rig images were holding its
layers.

```bash
docker image rm llama-cpp-only-cpu-mesh-worker:latest \
                llama-cpp-only-cpu-rig-worker:latest   # then prune
docker builder prune -a -f
```

Other candidates, in descending order of how easy they are to get wrong:

| What | Size (2026-10-23) | Notes |
|---|---|---|
| `mesh-worker` + `rig-worker` images | **47.5 GB** | Lane closed by 0016. Delete these FIRST. |
| `docker builder prune -a` | 13.7 GB alone, far more after the above | Yield depends entirely on ordering. |
| `nvidia/cuda:12.8.1-devel-ubuntu24.04` | **14.6 GB** | True orphan — see below. |
| `aod_ollama_models` volume | 22.7 GB | Separate `aod-gpubox` project — see below. |
| unused *tagged* images | 15 GB | **Do NOT use `docker image prune -a`** — see below. Name images explicitly. |
| `make prune-job-cache` | 4.5 GB | **Loses redraw.** See below. |
| `models/Qwen2.5-3B-Instruct-Q4_K_M.gguf` | 1.8 GB | Dead since 0013; `legacy-llm` only. |

Traps in that table:

- **`docker image prune -a` is unsafe on this box, and most unsafe exactly when
  you reach for it.** It removes every image no *container* references — and
  this stack routinely sits with containers stopped or missing entirely. Checked
  2026-10-23 evening: `llama-cpp-only-cpu-sprite-worker:latest` (9.42 GB) had
  **no container at all**, and `ghcr.io/ggml-org/llama.cpp:full-cuda` never has
  one, so `prune -a` would have deleted the live digest-pinned base and a
  current sprite image. **Always `docker image rm <name>` by name.**
- **`nvidia/cuda:12.8.1-devel-ubuntu24.04` is safe to delete; the two that
  look similar are not.** No Dockerfile references 12.8.1 — `mesh_worker` and
  `rig_worker` both build `FROM nvidia/cuda:13.0.3-devel-ubuntu22.04`, and the
  sprite images are `python:3.11-slim`-based, so it shares no layers with
  anything live. It is 19 months old. Note that **13.0.3 is not on the box as
  a tag**, so re-running 0016 re-pulls it.
- **Do NOT delete `ghcr.io/ggml-org/llama.cpp:full-cuda` (10.3 GB).** Its image
  ID is `sha256:85e30d01b550…`, which is exactly the digest pinned at
  `compose/develop/sprite_generator/Dockerfile.cuda:15`. Deleting the tag forces
  a 10.3 GB re-pull on the next sprite build. The link has measured 1–9 MiB/s,
  so that is hours, not minutes.
- **`aod_ollama_models` belongs to another project of yours**, `aod-gpubox` at
  `C:\Users\linuser\Projects\auto-orbit-data-external-ai`. If you tear that
  stack down, `compose down` **without** `-v`: images rebuild and ollama
  re-pulls its weights, but `aod_pg_data` (172 MB) is the one irreplaceable
  piece. Dropping the stack plus the models volume and keeping the database is
  ~32 GB.
- **`make prune-job-cache` deletes sheet evidence.** `d3083b62` and `2ff8af36`
  are the Q3_K_S-versus-Q2_K comparison jobs cited in CLAUDE.md and
  `.ai/specs/sheet-quality/`. Pruning them is fine only if you no longer need
  to re-measure that result. Use `older=` / `only=` to scope it.
- **Do not delete `models/blobs/`.** It looks like an orphan at the models root
  but carries `.huggingface-shared-blobs` — it is the HF shared blob store, and
  it is why `models--Qwen--Qwen-Image-2512` shows as 44K. Deleting it breaks
  every repo whose blobs live there.

## What `make compact-wsl` does, in order

`scripts/compact-wsl-disk.ps1`, elevated, with `-Restore`:

1. Resolves the VHDX from the registry (`HKCU:\...\Lxss`, matched on
   `DistributionName`) — no hardcoded GUID.
2. **Stops the keepalive** via `wsl-keepalive.ps1 -Stop`.
3. `wsl --shutdown`. **Every container stops.**
4. **Waits for an exclusive handle on the VHDX**, up to 60s, and aborts if it
   never arrives.
5. `diskpart`: `attach vdisk readonly`, `compact vdisk`, `detach vdisk`.
   Output is teed to `%TEMP%\compact-wsl-disk.log`.
6. Restarts the keepalive, runs `make up`, then `make lan-check`.

Steps 2 and 4 were added 2026-10-23 and are not optional — see the traps below.

### How long, and how much

`compact vdisk` gives no progress output; a silent window is not a hung one.
The 2026-10-23 run took **46 minutes on a 288 GB file** (that run used
`Optimize-VHD`; `diskpart compact vdisk` measured 7-15 minutes on the smaller
119 GB disk in 2026-08). Budget an hour and do not interrupt it.

**The `df` gap is an upper bound, not a forecast.** On 2026-10-23 the distro
reported 225 GB used inside a 288 GB file — a 63 GB gap — and compaction
returned **35.4 GB**, leaving 27.6 GB still allocated. Compaction reclaims only
blocks it can prove are unused; ext4 free space being fragmented is normal.
Quote the gap as a ceiling, never as the expected result.

## Bringing the stack back

With `make compact-wsl` this is automatic (step 6). If you ran the script by
hand without `-Restore`, or it aborted:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\wsl-keepalive.ps1
```

```bash
make up          # also runs the migrations
make lan-check   # expect five "ok" lines
```

Three corrections to the pre-2026-10-23 version of this section:

- **`setup-models-vhd.ps1 -AttachOnly` is gone from the sequence.** There is no
  `D:\wsl-models.vhdx`; `MODELS_DIR` is on the distro root. Running that script
  *without* `-AttachOnly` would create an empty disk.
- **Run `make lan-check`, not `lan-expose`.** The forwards target the distro IP,
  which usually survives a restart — checked 2026-10-23, unchanged and all five
  still answering. Only reach for `lan-expose` (UAC) when `lan-check` fails.
- **`sudo service docker start` is not needed**, but **do not count on the
  sprite stack coming back on its own.** The systemd unit is enabled and
  dockerd does start. Restart policies are the unreliable part: on 2026-10-23
  *morning* all ten containers were up before `make up` ran, yet after the
  distro was swept that *evening* only the `aod-gpubox` stack came back —
  `sprite_generator`, `redis_broker` and `stats_db` were `exited`,
  `model_downloader` and `model_orchestrator` sat in `created` having never
  started, and **there was no `sprite_worker` container on the box at all**
  (its image was present and unreferenced). So the earlier "all ten were up"
  is one observation, not a property. **Always run `make up` and check
  `docker ps` rather than assuming** — and note that a missing container, not
  merely a stopped one, is a state this stack reaches on its own.

Verify before sending traffic:

```bash
docker ps          # 10 containers Up
make gpu-health    # the worker's CUDA context, not just nvidia-smi
```

## Traps already measured (do not re-learn)

- **`wsl --shutdown` alone does not release the disk.** The keepalive holds a
  hidden `wsl.exe ... sleep infinity` and reattaches within seconds — measured
  2026-10-23, the distro was `Running` again 3s after a shutdown. diskpart would
  then be compacting a live disk. Stop the keepalive first and **prove** the file
  is free by taking an exclusive handle; the script does both. `Get-Process wsl`
  returning nothing is the tell that it is really down.
- **`PS C:\Windows\system32>` is not proof of elevation.** `system32` is just a
  default working directory. A run that is not elevated fails at the admin check
  having changed nothing. The script now self-elevates, so the UAC prompt is the
  only thing to watch for.
- **Running a `.ps1` by path hits the execution policy**, not permissions:
  `cannot be loaded because running scripts is disabled`. Invoke it as
  `powershell -ExecutionPolicy Bypass -File <path>`, which is per-process and
  changes no system setting.
- **`Tee-Object` writes UTF-16 in PowerShell 5.1**, so a log read as bytes comes
  back as `R E S U L T = F A I L`. Use `Add-Content -Encoding utf8`.
- **`2>&1` on a NATIVE command, under `$ErrorActionPreference = "Stop"`, kills
  the script on the first line of ordinary progress output.** PowerShell 5.1
  wraps each stderr line from a native exe in a `NativeCommandError`, and
  `Stop` makes the first one terminating. **docker compose writes all of its
  progress to stderr**, so this is not an edge case — it fires immediately.
  Measured 2026-10-24: the `-Restore` step died on compose's first line,

  ```
  wsl.exe :  Container sprite_worker Creating
  At ...\compact-wsl-disk.ps1:283 char:1
  + & wsl.exe -d $Distro -e bash -lc "cd '$repoWsl' && make up" 2>&1 |
  + CategoryInfo : NotSpecified: ( Container sprite_worker Creating :String)
  ```

  leaving a **half-created container whose name stayed reserved even though it
  appeared in neither `docker ps` nor `docker ps -a`** — `make up` then failed
  with *"The container name /sprite_worker is already in use"* and
  `docker rm -f` by name *and* by id both silently did nothing. **Only a
  `systemctl restart docker` cleared it.** `lan-check` never ran at all.
  Fixed both ways: merge stderr **inside** bash (`bash -lc "... make up 2>&1"`)
  so PowerShell sees only stdout, and drop to `"Continue"` for the restore
  block. The restore step had never completed whenever `make up` had real work
  to do; it only looked fine when compose printed nothing.
- `-Encoding Ascii` destroys a non-ASCII profile path — the script falls back to
  the 8.3 short name for diskpart. Keep that when editing it. (The current
  profile is `C:\Users\linuser`, which is ASCII, so the fallback is dormant
  rather than dead.)
- `~` in an 8.3 name is read by PowerShell 5.1 as a home-dir reference; use
  `-LiteralPath` for anything under `$env:TEMP`.
- `scripts/*.ps1` must stay **ASCII-only** (5.1 reads ANSI without a BOM).
- `wsl-keepalive.ps1 -Install` (logon task) needs Administrator; running the
  keepalive itself does not.

## Deferred: moving `MODELS_DIR` onto another drive

**Current reality (2026-10-23):** `MODELS_DIR=/home/markunn/sprite-data/models`
(`compose/develop/.env`), inside the C: VHDX, **126 GB** and growing — it was
28 GB when this section was first written. No models VHD exists. The drive
letters have also shifted since: **C: 476.9 GB (108.7 free that morning,
16.3 free by the evening), D: 222.9 GB (188.7 free), E: 111.8 GB (69.4
free)**. The old note saying "D: has 69.5 GB
free" is describing what is now E:, and `D:\wsl-model-archive` is **absent** —
the 28 GB cold archive parked in 2026-08 is not retrievable, so anything deleted
must come from the Hub again.

**Decision: still not now.**

> **CORRECTION 2026-10-23 (evening). The paragraph that stood here was wrong,
> and acting on it would have cost a multi-hour migration that fixed nothing.**
> It read: *"the balance has shifted — weights are now the dominant consumer
> (126 GB against 22 GB of Docker), which is the condition this section named
> as the trigger."*
>
> **Docker is the dominant consumer, not weights.** Measured the same evening:
>
> | | |
> |---|---|
> | `/var/lib/containerd` | **151 GB** |
> | `/var/lib/docker` | 22 GB |
> | **Docker total** | **173 GB** |
> | `/home` (models ~126 + job cache 4.3 + repo) | **149 GB** |
>
> **`22 GB` is `/var/lib/docker` alone, and on this box that number is
> meaningless on its own.** Docker here uses the **containerd image store**, so
> images and build cache live in `/var/lib/containerd` while
> `/var/lib/docker` holds little but volumes. "Docker usage" must always mean
> **both** directories summed. The error looks like a conflation of two
> coincidentally equal figures: containerd measured 126 GB that morning (it is
> quoted correctly in "Free space INSIDE the distro first" above) and the
> models tree is also ~126 GB.
>
> So the trigger condition below was **never met**, and moving `MODELS_DIR` to
> D: would have left the actual 173 GB problem sitting on C:.

Reasons to hold anyway:

- Weights are not what fills this disk. Two reclaim passes over Docker
  artifacts (dead images, then build cache) free more than the entire models
  tree, and cost only re-pulls.
- It adds a failure mode the current setup does not have: if the VHD is not
  re-attached before `make up`, Docker bind-mounts an **empty** `/models`. Since
  0014 that is less silent than it was — `make up` runs `models-warn`, which
  compares `models.toml` against `MODELS_DIR` and prints the missing required
  entries — but a `mountpoint -q` guard is still the honest fix.
- D: is now the obvious target with 188.7 GB free, not the 69.5 GB this section
  was written against.

**When it is worth doing:** C: is tight again *because of weights* while the
build cache is already pruned **and the dead-image sweep in "Free space INSIDE
the distro first" has been run**.
State the evidence as `du -sx /var/lib/containerd /var/lib/docker /home`, all
three, before quoting this condition — the 2026-10-23 version of this section
was wrong precisely because it skipped `containerd`.

**How, when the time comes** (elevated PowerShell unless noted):

1. Create, sized for the roster — the script default `-SizeGB 25` cannot hold
   the current 126 GB:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\scripts\setup-models-vhd.ps1 -SizeGB 200
   ```
   Shuts the distro down, creates + formats ext4, mounts at `/mnt/wsl/models`.
   **Check the script's target drive letter first** — it predates the reshuffle.
2. Copy, inside WSL, with the stack down:
   `rsync -a /home/markunn/sprite-data/models/ /mnt/wsl/models/`
   Compare `du -sh` on both sides before deleting the source. Note `models/blobs`
   is the shared blob store and must come across intact.
3. Repoint `MODELS_DIR=/mnt/wsl/models` in `compose/develop/.env`.
4. Add a guard so `make up` refuses when `/mnt/wsl/models` is not a mount point
   (`mountpoint -q /mnt/wsl/models`). `models-warn` warns; it does not refuse.
5. `make up`, then `make gpu-health` and one generation.
6. Delete the old directory inside WSL, then run this runbook once more — the
   freed space only reaches C: after compaction.
7. Docs: correct `.ai/project-context.md` and un-comment the attach step in
   "Bringing the stack back".
