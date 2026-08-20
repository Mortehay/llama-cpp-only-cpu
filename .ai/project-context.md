# Project Context

## What this is

Despite the repo name, `llama-cpp-only-cpu` is no longer a CPU LLM cluster. The
live component is `src/sprite_generator/` — a FastAPI + Celery service that
generates pixel-art sprite sheets, with a Jinja/vanilla-JS browser UI and
Postgres history. The llama.cpp/collector/orchestrator services around it are
legacy from the original purpose. **The root `README.md` still describes the old
project and is stale.**

## Where it is going

Three deliverables, in priority order:

1. **A REST image-generation service for the home LAN** — text2img, img2img, and
   image2text, consumed by other machines.
2. **A local browser testing tool** — prompt/model/seed comparison for iterating
   on complicated pixel-art sprites.
3. **LoRA training** on curated outputs, so sprite consistency stops depending
   on prompt engineering alone.

The first external consumer is the admin panel of
[Mortehay/something2](https://github.com/Mortehay/something2).

## Measured hardware (2026-08-19)

These are measured, not assumed, and several are binding constraints:

| | |
|---|---|
| CPU | Intel i3-8100 — **4 cores / 4 threads, no HT** |
| Host RAM | 15.9 GB |
| GPU | RTX 3060 **12 GB**, driver 610.88 → **CUDA 13.3** (was 551.86 / CUDA 12.4) |
| Host disk | C: 157 GB free |
| WSL | Ubuntu 26.04 LTS ("resolute"), ext4 VHD 954 GB free |
| OS | Windows 10 Pro 19045 |

Consequences worth remembering:

- **CPU inference is not viable here.** 4 threads, and FLUX at bfloat16 needs
  ~11.5 GB. The GPU pivot is a necessity, not an optimization.
- **~0.5 GB of VRAM is consumed** by the Windows desktop after the driver
  update (~3.8 GB before it). Budget against ~11.7 GB free.
- WSL got only ~7 GB by default. `~/.wslconfig` now sets `memory=11GB`,
  `swap=16GB`, `processors=4`. This budget covers the *whole* WSL2 VM.
- Host RAM is the real ceiling for FLUX, not VRAM — `enable_model_cpu_offload`
  streams weights through system RAM.

## Environment topology

- **Docker Engine runs inside WSL2 Ubuntu. Docker Desktop is not used** — see
  [decisions/0001](decisions/0001-gpu-pivot-and-engine-choice.md). Run all
  `make` targets from inside WSL, not from Windows.
- Docker Engine 29.7.2, Compose v5.5.0, NVIDIA Container Toolkit 1.20.0.
- Verify GPU reachability with `make gpu-check` — it tests from *inside a
  container*, which is the only check that matters. Host `nvidia-smi` passing
  proves nothing about container access.
- **All persistent data lives on WSL ext4**, not `/mnt/c`, via
  `MODELS_DIR` / `IMAGES_DIR` / `DB_DATA_DIR` in `compose/develop/.env`. Postgres
  especially must not sit on `/mnt/c`: `initdb` needs POSIX ownership that
  DrvFs/9p cannot provide, and 9p fsync is unsafe for a database. The repo stays
  on the Windows side; only the multi-GB weights need ext4 I/O.
- The repo path contains Cyrillic characters and a space
  (`.../Нова папка/...`). Quote paths in any script that touches it.

### Home-network exposure (unsolved)

**WSL2 is NAT'd, so a port bound inside WSL is not reachable from other LAN
machines.** WSL's `networkingMode=mirrored` would fix this but requires
Windows 11 22H2+; this host is Windows 10. The remaining option is
`netsh interface portproxy` on the Windows side plus a firewall rule, which
needs admin and must be redone when the WSL IP changes. This directly gates
deliverable #1 and is not yet done.

### WSL2 distro lifetime (bites constantly)

**WSL2 terminates the distro when no `wsl.exe` client is attached**, killing
systemd, dockerd and all containers. Symptom: containers exit with status 0 or
255 shortly after starting, with clean logs and no OOM. Confirm by looking for
repeated `systemd[1]: Startup finished in Ns` in `journalctl` — that is a full
boot, not a service restart.

- Fix: run `scripts/wsl-keepalive.ps1` (a hidden `wsl.exe ... sleep infinity`).
- `vmIdleTimeout` in `.wslconfig` does **not** help: Windows 11 only, silently
  ignored on Windows 10.
- `uptime` inside WSL reports the **VM**, which survives while the distro
  cycles. Do not use it to rule out a reboot.

### PowerShell script encoding

Windows PowerShell 5.1 reads `.ps1` files as **ANSI unless they have a BOM**.
A UTF-8 em-dash or curly quote written by a Unix tool becomes mojibake mid-string
and produces `The string is missing the terminator`. Keep `scripts/*.ps1`
ASCII-only.


### Code reloading

`src/sprite_generator/` is bind-mounted, so Python changes need no rebuild — but
the two services behave differently:

- **`sprite-generator` (API)** runs `uvicorn --reload` and picks up changes
  automatically.
- **`sprite-worker` (Celery) does NOT hot-reload.** After editing `tasks.py` you
  must restart it, or the worker keeps running the old module and rejects new
  tasks with `Received unregistered task of type '...'` — which looks like a
  routing bug rather than a stale process:

  ```bash
  docker compose ... restart sprite-worker
  ```

Only dependency changes (`requirements*.txt`, Dockerfile) need a rebuild.
### Celery + CUDA

The worker runs `--pool=solo` (set in `docker-compose.cuda.yml`). Celery's
default prefork pool forks after importing `tasks.py`; if anything initialized a
CUDA context in the parent, every child dies with *"Cannot re-initialize CUDA in
forked subprocess"*. `tasks.py` therefore must not touch CUDA at import —
`torch.cuda.is_available()` is fork-safe, `get_device_properties()` is not.
## The something2 provider contract

something2 already has a **generic remote AI provider system** (`docs/ai-providers.md`).
Do not design a new protocol — impersonate one it already speaks.

- Admin registers: base URL, optional auth header, a JSON request template with
  `{{prompt}} {{model}} {{seed}} {{width}} {{height}} {{frames}}`, a models
  discovery path + pointer, and an image pointer.
- Worked examples: **Automatic1111** (`POST /sdapi/v1/txt2img`, models
  `GET /sdapi/v1/sd-models`, models pointer `$[*].model_name`, image pointer
  `images[0]`) and OpenAI-images (`data[0].b64_json`).
- **Synchronous only.** ComfyUI-style submit/poll is explicitly unsupported
  (their SOMET-334). `AI_PROVIDER_GENERATE_TIMEOUT_MS` defaults to 5 minutes.
  Our Celery/poll design must hide behind a blocking façade.
- something2 slices sheets itself (columns/rows/directions). It wants **one grid
  PNG** that divides evenly into the declared grid; ≤32 MB.

Caveat: this was read from their published `docs/ai-providers.md` on `main`.
Their actual calling code has not been read.

## Known-broken areas

- **Step 2 (spritesheet) does not produce usable output** after four rewrites —
  the model's row/stack/grid layout is not stable, so no fixed slicing rule
  holds. See README "Known issues". Step 1 works well.
- (historical) The FLUX path could not load at all: `tasks.py` expected
  `/models/flux1-schnell.safetensors` plus a hardcoded HF snapshot hash, while
  `models.txt` downloads `flux-2-klein-4b-Q8_0.gguf`. Neither the safetensors
  nor the snapshot is ever fetched, and the GGUF is never loaded. Deliberately
  **not fixed in diffusers** — the ComfyUI migration deletes that path.
- The per-action try/except now records failures and continues (was: commented out).
- `POST /api/settings/{key}` removed (was dead code that always errored).
- The Settings UI compute toggle is now a read-only readout via
  `GET /api/compute-info` (was a dead toggle plus stale hardware copy).
