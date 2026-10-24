# 0018 - Disk space on this box is a retention policy, not a reclamation chore

Date: 2026-10-23 (reclaim executed and rules 1+3 applied 2026-10-24)
Status: **Accepted, PARTIALLY IMPLEMENTED.** Rule 1 (cache cap) is live in
`/etc/docker/daemon.json`; rule 3 (ordering) was exercised and measured a third
time during the 2026-10-24 reclaim. **Rules 2 and 4 are still unimplemented.**
Supersedes the premise of the "Deferred: moving `MODELS_DIR` onto another
drive" section in [`compact-wsl.md`](../../compact-wsl.md), corrected the same
day.

**Result of the 2026-10-24 reclaim: `ext4.vhdx` 342.8 GB -> 221.5 GB, C: free
16.3 GB -> 137.6 GB. 121.3 GB returned, no weights touched.**

> **Read this before running the compaction runbook a fourth time.** The short
> version: compaction has been run three times and the last one lasted hours.
> It is not a fix, because it does not address what fills the disk. The disk is
> filled by Docker build artifacts for experiments, and WSL's
> never-shrinking VHDX turns each transient peak into permanent cost.

## Why this exists

Four fill-ups: 2026-08-21, 2026-09-12, 2026-10-23 morning, 2026-10-23 evening.
The last gap is the one that matters. A 46-minute compaction that morning took
`ext4.vhdx` from 288.0 GB to 252.6 GB and left C: with 108.7 GB free. **By that
evening the file was 342.8 GB and C: had 16.3 GB free — +90 GB in hours.**

Measured that evening, inside the distro:

| | |
|---|---|
| `/var/lib/containerd` | **151 GB** |
| `/var/lib/docker` | 22 GB |
| `/home` (models ~126, job cache 4.3, repo) | **149 GB** |
| ext4 used / VHDX size | **325 GB / 342.8 GB** |

What had been created since the compaction, from `docker images`:

| Image | Size | Built |
|---|---|---|
| `llama-cpp-only-cpu-rig-worker:latest` | 25.2 GB | 4 hours earlier |
| `llama-cpp-only-cpu-mesh-worker:latest` | 22.3 GB | 5 hours earlier |
| `llama-cpp-only-cpu-sprite-worker:latest` | 9.42 GB | 2 hours earlier |
| `llama-cpp-only-cpu-sprite-generator:latest` | 9.42 GB | 2 hours earlier |

plus the build cache regrowing from ~20 GB post-prune to **81.22 GB**. `/home`
did not grow at all. The two largest images on the box existed to run the
[0016](0016-rig-gate-retest-and-off-card-3d.md) rig-gate retest — whose decision,
written the same day, is *"Do not revive the mesh -> rig conveyor."*

So 47.5 GB was resident for a lane that had just been formally killed, and
nothing in the process said to remove it.

## Decision

**Treat disk as a retention problem with four standing rules.**

### 1. Cap the build cache so it cannot reach 81 GB again

**Applied 2026-10-24.** `/etc/docker/daemon.json` now reads:

```json
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    },
    "builder": {
        "gc": {
            "enabled": true,
            "defaultKeepStorage": "20GB"
        }
    }
}
```

The cache went ~20 GB -> 81.22 GB in a single day of builds. A cap makes that
structurally impossible and costs only cache misses on builds that were going
to re-pull anyway. Requires a dockerd restart, which stops every container.

> **`daemon.json` already carried the `nvidia` runtime. MERGE, never
> overwrite.** Writing a bare `builder` block over it would remove
> `nvidia-container-runtime` and take the GPU away from the worker — which
> would present as generation failures, not as a disk change, and would send
> the next session to `make gpu-health` chasing a fault that is not there.
> Backup kept at `/etc/docker/daemon.json.bak-20261024`; validate with
> `python3 -c 'import json; json.load(open("/etc/docker/daemon.json"))'`
> before restarting dockerd.

### 2. An ADR that closes a lane takes its images off the box in the same session

0016 closed the mesh/rig lane and left 47.5 GB behind. The teardown belongs in
the session that writes the decision, while it is still obvious which images
were experimental. Concretely, for 0016: `docker image rm
llama-cpp-only-cpu-mesh-worker:latest llama-cpp-only-cpu-rig-worker:latest`.

### 3. Delete dead images BEFORE pruning the build cache

`docker builder prune` reclaims only cache layers *private* to the cache;
anything still shared with a live image is netted off. An image you are about
to delete is, until deleted, protecting its cache layers from the prune.
Measured both ways on 2026-10-23: prune-first that morning freed **56 GB** of a
76.49 GB cache (33.12 GB shared); that evening, with the dead mesh/rig images
still present, an 81.22 GB cache reported only **13.66 GB reclaimable**.

### 4. A pre-build disk guard that reads the Windows side

`df` inside WSL reports the VHDX's nominal ~1007 GB and is actively misleading.
A `make` target that reads `Get-CimInstance Win32_LogicalDisk` and warns (or
refuses) below a threshold would have made the evening's second 25 GB build an
informed choice. This is the weakest of the four — it addresses visibility, not
retention — but it is also the cheapest.

## What this costs

- **Cache misses.** A 20 GB cap means some rebuilds re-pull layers they would
  otherwise have had. The torch wheel is the expensive one (15-25 min). This is
  a deliberate trade: the cap's whole purpose is to stop paying permanent disk
  for cached bytes.
- **Re-pulls when an experiment is revived.** Removing `mesh-worker` and
  `rig-worker` means reviving 0016 re-pulls
  `nvidia/cuda:13.0.3-devel-ubuntu22.04` and rebuilds ~48 GB. 0016 says do not
  revive it, so this is the correct side of the trade — but it is a real cost
  if that decision is ever reversed.
- **A dockerd restart** to apply rule 1, which stops every container including
  the `aod-gpubox` stack.

## Alternatives rejected

- **Keep compacting.** This is the status quo and it has now failed visibly:
  three runs, the last lasting hours, and on the fourth fill-up compaction was
  not even *available* — 325 GB used inside a 342.8 GB file is ~18 GB of slack,
  and compaction returns only what is already free in ext4. It also costs ~1
  hour and takes the whole stack down.
- **Move `MODELS_DIR` to a D: ext4 VHD as the fix.** This was the queued plan
  and it rests on a measurement error: `compact-wsl.md` claimed weights
  dominated at "126 GB against 22 GB of Docker", where 22 GB is
  `/var/lib/docker` **alone** and omits `/var/lib/containerd`'s 151 GB. Docker
  is 173 GB against `/home`'s 149 GB. Moving weights would have left the actual
  problem on C: after a multi-hour migration that adds a failure mode (an
  unmounted VHD makes `/models` silently empty). **Still worth doing eventually**
  for the permanent ~126 GB of headroom and to decouple weights from Docker
  churn — but as a follow-up, not as this fix. D: has 188.7 GB free.
- **Move the whole VHDX to D: (`wsl --manage <distro> --move`).**
  `.ai/project-context.md` recommends this over a models-only VHD, to keep
  everything on ext4. **It does not fit.** The file is 342.8 GB against D:'s
  188.7 GB free, and would not fit after a compaction back to ~250 GB either.
  A models-only ext4 VHD is the only viable variant on current drive sizes.
- **`wsl --manage <distro> --set-sparse true`.** Already rejected in
  `.ai/project-context.md`: WSL refuses it without `--allow-unsafe`, citing
  potential data corruption, and this distro holds the Postgres data directory.
- **Deleting weights or the sheet job cache.** Considered and rejected for this
  fill-up: `/home` did not grow, so it is not the cause. Deleting GGUFs means
  Hub-only recovery (the D: cold archive is gone) and `make prune-job-cache`
  would forfeit the `d3083b62` / `2ff8af36` quant comparison evidence cited in
  CLAUDE.md and `.ai/specs/sheet-quality/`.

## Implementation notes

Reclaim order for the current fill-up, highest confidence first:

1. `docker image rm` the `mesh-worker` and `rig-worker` images — 47.5 GB.
2. `docker image rm nvidia/cuda:12.8.1-devel-ubuntu24.04` — 14.6 GB. A true
   orphan: nothing builds from 12.8.1, and it shares no layers with the
   `python:3.11-slim`-based sprite images.
3. `docker builder prune -a -f` — yield is now much larger than the 13.66 GB
   reported while the images were resident.
4. Optionally the `aod-gpubox` stack, ~32 GB. **`compose down` without `-v`** —
   images rebuild and ollama re-pulls, but `aod_pg_data` (172 MB) is the one
   irreplaceable piece. This is another project's running stack; confirm at the
   keyboard first.
5. Only then `make compact-wsl`. Budget ~1 hour for a 342.8 GB file.

**Do not** delete `ghcr.io/ggml-org/llama.cpp:full-cuda` (10.3 GB). Its image ID
`sha256:85e30d01b550…` is the exact digest pinned at
`compose/develop/sprite_generator/Dockerfile.cuda:15`; removing the tag forces a
10.3 GB re-pull at the 1-9 MiB/s this link has measured.

Use `wsl -d Ubuntu -u root` throughout. `sudo` in this distro prompts for a
password and **hangs indefinitely** with no tty rather than failing.

## What the restore exposed (2026-10-24)

Reclaiming the space was the easy half. Putting the stack back found two
separate defects, both now recorded as traps in `compact-wsl.md`:

- **`scripts/compact-wsl-disk.ps1 -Restore` had never completed.** Line ~54
  sets `$ErrorActionPreference = "Stop"`, and the restore block piped native
  commands through `2>&1`. PowerShell 5.1 wraps native stderr in a
  `NativeCommandError`, `Stop` makes the first one terminating, and **docker
  compose writes all of its progress to stderr** — so the script died on
  compose's first line and never ran `lan-check`. It only ever looked healthy
  when `make up` had nothing to print. Fixed by merging stderr inside bash and
  dropping to `"Continue"` for the restore block.
- **That abort left container state Docker could not clean up by itself.** A
  half-created `sprite_worker` held its name while appearing in neither
  `docker ps` nor `docker ps -a`; `docker rm -f` by name and by id both did
  nothing. Later a `sprite_generator` went `dead` (failed filesystem unmount)
  and blocked `docker compose up`, which sat at 0% CPU indefinitely rather
  than erroring. **`systemctl restart docker` was the only thing that cleared
  either.** A compose that is pinned at 0% CPU is stuck, not working.

## Not verified

- **Rules 2 and 4 are not implemented.** No teardown step exists in any
  process document beyond this ADR, and the pre-build disk guard has no
  implementation, not even a target name.
- **`defaultKeepStorage: "20GB"` is an untested threshold.** It is chosen to sit
  just above the ~20 GB the cache held after the morning prune, on the
  assumption that a working set that fit then will fit again. Not measured
  against a cold `--no-cache` rebuild of `sprite-worker`, and **no build has
  yet run to completion under the cap**, so its practical effect on build
  times is unknown.
- **Whether the sprite images' 9.42 GB each was net-new** is unknown; they were
  rebuilds and may have replaced similar-sized predecessors, so the 90 GB
  attribution is sound in total but not exact per image.
- **The repaired `-Restore` path has not been re-run.** The fix is reasoned
  from the documented PowerShell 5.1 behaviour and the exact error, and the
  file parses and is ASCII-clean, but it has not been exercised end to end;
  that needs the next compaction.
- **`make gpu-health` has not been run since the `daemon.json` edit.** The
  nvidia runtime was preserved and validated as JSON, but GPU access has not
  been confirmed by actually generating anything.
- `%TEMP%\compact-wsl-disk.log` was absent while `scripts/compact-wsl-disk.ps1`
  is modified-uncommitted, so the `Add-Content -Encoding utf8` logging that
  `compact-wsl.md` describes may never have run. Verify on the next compaction.
