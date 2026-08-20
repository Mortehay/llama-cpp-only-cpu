# Sprite Generator

A local, GPU-accelerated pixel-art **sprite sheet generator**, exposed as a REST
API for the home network and as a browser tool for iterating on prompts.

> **Repo name is historical.** This began as a CPU-only `llama.cpp` cluster.
> The live component is now `src/sprite_generator/`; the `llm-server`,
> `collector`, and `orchestrator` services are legacy from that original
> purpose. The previous README is kept at `README.md.bak`.

## Requirements

- **NVIDIA GPU** with a reasonably current driver (developed against a
  RTX 3060 12 GB, driver 610.88 / CUDA 13.3).
- **WSL2 with Docker Engine installed inside the distro.** Docker Desktop is
  *not* used — see `.ai/decisions/0001-gpu-pivot-and-engine-choice.md`.
- **NVIDIA Container Toolkit** installed in the same distro.

> **Run every `make` target from inside WSL**, not from Windows PowerShell or
> Git Bash. The Docker daemon lives in the distro.

A CPU-only fallback still works: without `nvidia-smi` on PATH the Makefile skips
the CUDA overlay. Expect it to be slow to the point of impracticality.


---

## First run

```bash
# 1. Create compose/develop/.env (or copy it yourself from .env.example)
make env

# 2. Fill in HF_TOKEN — required for gated repos (FLUX, Llama, Gemma).
#    Also set the storage paths — see "Storage" below. Model weights and the
#    Postgres data dir MUST be on WSL ext4; generated images stay in the repo.
$EDITOR compose/develop/.env

# 3. Confirm the GPU is visible *from inside a container*. Host nvidia-smi
#    passing proves nothing about container access.
make gpu-check

# 4. Build the GPU image on its own first. This pulls the torch wheel
#    (~2.4 GB) and takes 15-25 minutes on a cold cache — do it before wiring
#    in the rest so a slow download is not tangled up with model fetching.
docker compose -f compose/develop/docker-compose.yml \
               -f compose/develop/docker-compose.cuda.yml \
               --env-file compose/develop/.env build sprite-worker

# 5. Download models and start everything.
make build
```

Then open **http://localhost:8001**.

| Service | URL | Purpose |
|---|---|---|
| `sprite_generator` | http://localhost:8001 | Sprite API + browser tool |
| `sprite_worker` | *(internal)* | Celery worker — **owns the GPU** |
| `stats_db` | `localhost:5432` | Postgres: sprite history, settings |
| `redis_broker` | `localhost:6379` | Celery broker |
| `model_orchestrator` | http://localhost:7860 | Download/delete GGUF models |
| `stats_collector` | http://localhost:8002 | Legacy LLM stats proxy |
| `open_webui` | http://localhost:3000 | Legacy chat UI |
| `grafana` | http://localhost:3001 | Legacy metrics |


---

## GPU configuration

The Makefile detects `nvidia-smi` and layers `compose/develop/docker-compose.cuda.yml`
over the base compose file automatically. That overlay:

- builds from `Dockerfile.cuda`, which installs torch from the
  PyPI default (current CUDA), matching driver 610.88;
- reserves the GPU for **`sprite-worker` only**;
- sets `COMPUTE_DEVICE=cuda` on the worker and `COMPUTE_DEVICE=cpu` on the API
  process, which imports torch but never runs inference and must not hold VRAM.

`src/sprite_generator/tasks.py` resolves `DEVICE`/`DTYPE` once at import.
`COMPUTE_DEVICE=auto` (the default when unset) autodetects. If you set `cuda`
and CUDA is unavailable, it logs a loud warning and falls back to CPU rather
than failing silently.

### VRAM notes

- The Windows desktop consumes ~0.5 GB. Budget against **~11.7 GB** free.
- FLUX uses `enable_model_cpu_offload()` rather than `.to("cuda")` — a full FLUX
  pipeline at fp16 is ~24 GB. Offload streams weights through **system RAM**, so
  the WSL memory cap is the real ceiling. `~/.wslconfig` sets `memory=11GB`.
- `sprite-worker` runs `--concurrency=1` deliberately. A second worker would
  double-load pipelines into VRAM.


---

## Storage

| What | Path | In container | Set by |
|---|---|---|---|
| Model weights | `/home/<you>/sprite-data/models` — **WSL ext4** | `/models` | `MODELS_DIR` |
| Generated sprites | `./images` — **in the repo** | `/app/images` | `IMAGES_DIR` |
| Scratch frames | tmpfs — **RAM**, 2 GB cap | `/app/images/temp` | compose `tmpfs:` |
| Postgres data | `/home/<you>/sprite-data/db` — **WSL ext4** | `/var/lib/postgresql/data` | `DB_DATA_DIR` |

The split is deliberate and measured on this host:

| | `/mnt/c` (9p) | WSL ext4 |
|---|---|---|
| Write 512 MB | 48 MB/s | 54 MB/s |
| Read 512 MB | 44 MB/s | 3.9 GB/s (cached) |
| Create 2000 small files | 17,556 ms | 494 ms |

- **Generated sprites live in the repo** so they are visible in Explorer and the
  IDE. A 1 MB PNG costs ~20 ms to write either way — the 9p penalty is
  irrelevant for small, write-once files.
- **Model weights must not.** An HF snapshot is hundreds of files totalling GBs;
  at 44 MB/s with a 35x small-file penalty a cold load takes minutes instead of
  seconds, and 9p reads do not benefit from the page cache, so every worker
  restart or model switch pays it again.
- **Postgres must not**, and not only for speed: `initdb` needs POSIX ownership
  that DrvFs/9p cannot provide (`could not change permissions of directory`),
  and 9p fsync semantics are not safe for a database.

To browse the ext4 side from Windows, open this in Explorer (pin it for
convenience) or add it as a folder in your IDE:

```
\\wsl.localhost\Ubuntu\home\<you>\sprite-data
```

Models are also manageable from the orchestrator UI at http://localhost:7860
without touching the filesystem.

---

## Home-network exposure

**WSL2 is NAT'd.** Ports bound inside WSL are reachable from Windows via
`localhost` but *not* from other machines on the LAN. `networkingMode=mirrored`
would fix this but needs Windows 11 22H2+; on Windows 10, forward explicitly:

```powershell
# From an ELEVATED PowerShell, on the Windows side:
.\scripts\lan-expose.ps1

# To undo:
.\scripts\lan-expose.ps1 -Remove
```

**Re-run it after every WSL restart** — the WSL IP is assigned per boot, and
stale portproxy entries fail silently.


---

## Keeping the stack running (WSL2 gotcha)

**WSL2 terminates the distro once no `wsl.exe` client process is attached**,
taking systemd, dockerd and every running container with it. Containers appear
to exit on their own with status 0 or 255, minutes after starting, with nothing
wrong in their own logs. The giveaway is in `journalctl`:

```
systemd[1]: Startup finished in 20.877s     <- a FULL systemd boot, again
```

Start a keepalive before running the stack, and leave it running:

```powershell
# Windows PowerShell (elevation NOT required):
powershell -ExecutionPolicy Bypass -File .\scripts\wsl-keepalive.ps1
```

Notes:

- `vmIdleTimeout` in `.wslconfig` does **not** fix this on Windows 10 — that
  setting is Windows 11 only and is silently ignored.
- `uptime` inside WSL is misleading here: it reports the VM, which stays up
  while the distro cycles. Check `journalctl` for repeated "Startup finished"
  lines instead.
- The keepalive dies at logoff/reboot. Re-run it, or register it as a logon
  scheduled task — the command is in the script's header comment.
- Keeping any interactive WSL terminal open has the same effect.

---

## Make commands

| Command | Description |
|---|---|
| `make env` | Create `compose/develop/.env` from the example if missing |
| `make gpu-check` | Verify the GPU is reachable from inside a container |
| `make up` | Start the stack (downloads missing models first) |
| `make build` | Build images, download models, start |
| `make recreate` | Force-recreate the sprite services (fixes stale container DNS) |
| `make rebuild-clean` | Full no-cache rebuild |
| `make down` / `make stop` | Stop the stack |
| `make logs` | Tail a service (`SERVICE_NAME=sprite-generator make logs`) |
| `make download repo=<hf-repo> file=<f>` | Fetch a model into the running stack |
| `make warm` | Preload models into VRAM out-of-band (avoids request timeouts) |
| `make smoke` | Verify GPU, model discovery and a txt2img round-trip |
| `make test-flow` | Verify the full core -> spritesheet workflow |
| `make sync-models` | Download anything in `models.txt` that is missing |

### Changing the CUDA version

`torch` is installed in `Dockerfile.cuda` from PyPI with **no index override**,
in its own layer, and is deliberately **not** listed in `requirements.cuda.txt`.
PyPI's default torch is a current-CUDA build, which driver 610.88 (CUDA 13.3)
runs natively. The build asserts `torch.version.cuda` is non-null so a CPU-only
wheel fails the build instead of shipping an image that silently runs on CPU.

**If you ever run this against an older driver**, torch must be pinned to a
matching CUDA build, and neither pip index flag works alone:

- **`--extra-index-url` without a version pin** merely *adds* an index. pip
  resolves across it and PyPI and installs the highest version — which may be a
  newer-CUDA build than the driver supports. It imports without error and
  reports `torch.cuda.is_available() == False`, indistinguishable from a missing
  container runtime.
- **Exclusive `--index-url`** forces the right CUDA build but also forces torch's
  *dependencies* onto the PyTorch index, where `typing_extensions` trips pip's
  name-consistency check and the sdist fallback needs `flit_core`, which isn't
  published there. The install fails outright.

The fix is both together: `--extra-index-url <torch-index>` plus
`torch==<ver>+cuXXX`. That local version tag exists only on the PyTorch index,
so it forces the exact wheel while dependencies still come from PyPI.


---


### Step 2 (spritesheet) does not produce usable output

Step 1 (core sprite) works well. Step 2 does not, after four rewrites, and the
reason is not a bug that one more fix will clear.

What was tried, and what each attempt taught:

| Attempt | Approach | Result |
|---|---|---|
| 1 | Original FLUX path | Could not load at all — wrong weights wired up |
| 2 | img2img on a 512x128 strip | Core squashed 4:1; model denoised from a smear |
| 3 | Per-frame img2img at 128x128 | RGB static — SD1.5 cannot generate below ~512 |
| 4 | Per-frame at 512, downscaled | Real sprites, but a different character per frame (strength 0.95 discarded the reference) and each frame held its own mini-row |
| 5 | One row per action, alpha-sliced, strength 0.65 | Model rendered vertical stacks, not a row; slicer assumes a row |

The core obstacle: **the model's output layout is not stable.** Given similar
prompts it returns a horizontal row, a vertical stack, or a loose grid, so no
fixed slicing rule holds. Solving it properly needs explicit spatial control
(ControlNet pose per frame), identity conditioning (IP-Adapter), or a LoRA
trained on the exact sheet layout wanted — all standard ComfyUI workflows, and
all a poor fit for hand-rolling against diffusers.

What is worth keeping regardless of engine: background removal
(`remove_background`), alpha-based row slicing (`split_row_by_alpha`,
`fit_into_frame`), sampling reconciliation (`resolve_sampling_params`), the
sprite DB and history, the A1111 facade, and the whole test harness. Those are
domain logic, not engine logic.

**Recommendation:** ship step 1, which produces good single sprites in ~4s, and
move step 2 onto ComfyUI (phase 3) rather than continuing to hand-tune it.
## Known issues

- **FLUX step 2 cannot load on a clean machine.** `tasks.py` expects
  `/models/flux1-schnell.safetensors` plus a hardcoded HF snapshot hash, while
  `models.txt` downloads `flux-2-klein-4b-Q8_0.gguf`. Neither is wired to the
  other. Deliberately unfixed — the planned ComfyUI migration removes this code
  path entirely. Until then, step 2 has no working pipeline.

Fixed in the GPU pivot, listed here because they shaped the current code:

- The per-action `try/except` in `generate_spritesheet_task` was commented out,
  so one failed action aborted the sheet with no error recorded. It now records
  the failure and **continues with the remaining actions** — losing every strip
  already rendered is the expensive outcome on a multi-minute GPU job. An
  all-actions-failed run returns an explicit error instead of constructing a
  zero-height image.
- `POST /api/settings/{key}` was dead code that always returned an error.
  Removed.
- The Settings tab's CPU/CUDA radio wrote to `app_settings` that nothing read,
  and the device is resolved from `COMPUTE_DEVICE` at worker import, so it could
  not have worked. Replaced with a read-only readout backed by
  `GET /api/compute-info`, which round-trips through Celery to the worker.
- The diagnostics panel reported the **browser's** WebGL renderer as "Hardware",
  which describes the machine viewing the page, not where inference runs.

See `.ai/project-context.md` for measured hardware constraints and the
something2 integration contract.
