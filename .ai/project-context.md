# Project Context

Companion docs: [domain.md](domain.md) for vocabulary that is ambiguous in this
codebase — "core" alone means three different things — and
[decisions/](decisions/) for choices that are hard to reverse.

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

> **Revised by [decisions/0005](decisions/0005-back-to-2d-modern-editors.md)
> (2026-08-23).** The 3D pivot is reversed; the target is still whole-sheet
> consistency (~150 cells per character, all matching) but it is now reached
> with a 2026 instruction-editing model rather than a mesh.
>
> - **Deliverable 1 needs reshaping, not dropping.** The something2 contract is
>   a *synchronous* A1111 facade with a 240s budget. A 150-cell sheet cannot be
>   generated inside it on this hardware under any plan, so the facade becomes a
>   cache reader in front of a background job.
> - **Deliverable 2 stands**, and matters more than before: the conveyor now has
>   more stages worth comparing side by side, not fewer.
> - **Deliverable 3 (LoRA training) is deferred, not moot.** 0004 argued the 3D
>   pivot removed the consistency problem LoRA training solves. With 2D back,
>   the problem returns - but `Qwen-Image-Edit-2511` plus a locked palette is
>   the cheaper answer, and training is only worth revisiting if that is
>   measured to be insufficient.

## Measured hardware (2026-08-19)

These are measured, not assumed, and several are binding constraints:

| | |
|---|---|
| CPU | Intel i3-8100 — **4 cores / 4 threads, no HT** |
| Host RAM | 15.9 GB |
| GPU | RTX 3060 **12 GB**, driver 610.88 → **CUDA 13.3** (was 551.86 / CUDA 12.4) |
| Host disk | C: 222.9 GB total, **47.7 GB free on 2026-08-22** (20.5 GB on 2026-08-21; 157 GB on 2026-08-19 — the ext4 VHDX grew to 119 GB). D: 111.8 GB total, **72.6 GB free — usable for model storage when C: is tight** |
| WSL | Ubuntu 26.04 LTS ("resolute"), WSL 2.7.12. ext4 VHD reports ~1007 GB nominal — **this number is meaningless, see "Disk space" below** |
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

- **`sprite-generator` (API)** runs `uvicorn --reload`, but **the reload does not
  actually fire for Python edits made from Windows.** Measured 2026-08-20:
  editing `a1111.py` produced no `WatchFiles detected changes` line and the old
  module kept serving; a `docker restart sprite_generator` was required. The
  repo lives on `/mnt/c`, and inotify events do not propagate across the 9p
  bind mount, so the watcher never sees the write. Assume the API needs a
  restart for `.py` changes, exactly like the worker.
- **Jinja templates are the exception** and do reload: they are read from disk
  per request, so `templates/*.html` edits take effect on the next page load
  with no restart. This split is confusing in practice — an HTML change appears
  instantly while the Python change beside it silently does not.
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

- (historical) **Step 2 produced no usable output** after four rewrites, because
  the model's row/stack/grid layout was not stable and no fixed slicing rule
  held. Fixed by no longer asking for a layout at all: each frame is generated
  separately from the core and the strip is composed in PIL, where layout is
  arithmetic rather than a sample from a distribution.
- **Directional actions: cardinals work, diagonals do not.** `move left`/`right`
  turn via derived view cores, and `move up` now renders a genuine BACK view
  (no face) since step 2 moved to `Onodofthenorth/SD_PixelArt_SpriteSheet_Generator`,
  which ships trained front/back/left/right view triggers — see
  `tasks.VIEW_TRIGGERS`. **Diagonals still do not exist and still silently
  collapse onto cardinals**: matching is contiguous-substring, and
  `"move up right"` contains `"move up"`. Full diagnosis in
  [decisions/0003](decisions/0003-directional-sprites-and-view-cores.md).
- `attack_melee` uses the profile `ATTACK` cycle, so it silently receives a
  derived side core. Shipped, never verified.
- `burning` still speckles at strength 0.75 on some frames.
- (historical) The FLUX path could not load at all: `tasks.py` expected
  `/models/flux1-schnell.safetensors` plus a hardcoded HF snapshot hash, while
  `models.txt` downloads `flux-2-klein-4b-Q8_0.gguf`. Neither the safetensors
  nor the snapshot is ever fetched, and the GGUF is never loaded. Deliberately
  **not fixed in diffusers** — the ComfyUI migration deletes that path.
- The per-action try/except now records failures and continues (was: commented out).
- `POST /api/settings/{key}` removed (was dead code that always errored).
- The Settings UI compute toggle is now a read-only readout via
  `GET /api/compute-info` (was a dead toggle plus stale hardware copy).

## Image model selection constraints

Full reasoning in [decisions/0002](decisions/0002-image-model-selection.md). The
parts worth having in context before touching model choice:

- **Step 1 and step 2 are independent choices.** Step 2 re-renders the core
  through img2img, so the core model does not need to be a pixel-art model — it
  needs one clean, centred, single character. Step 2 supplies the style.
- **Step 2 is locked to SD1.5.** `poses.py` authors COCO-18 skeletons against
  `control_v11p_sd15_openpose`, and `get_sd_pipeline` refuses ControlNet on
  SDXL. Pose conditioning works; moving step 2 off SD1.5 discards it.
- **Prefer non-distilled checkpoints.** Anything matching `DISTILLED_MARKERS`
  (`turbo`, `schnell`, `lightning`, `lcm`) runs at guidance 0, which disables
  classifier-free guidance and makes every negative prompt a silent no-op. That
  is the root cause behind `_isolate_largest_sprite`.
- **Latency is not the constraint people assume.** The A1111 façade calls
  `generate_raw_task` — a *single* txt2img, 240s budget.
  `generate_spritesheet_task` is not exposed through the façade at all and runs
  async through the browser UI with no cap.
- **`is_sdxl` is a name heuristic, not a config read.** `get_sd_pipeline` picks
  the pipeline class from whether the repo name contains `sdxl` or `turbo`. An
  SDXL repo named otherwise (`…-diffusion-xl`,
  `stabilityai/stable-diffusion-xl-base-1.0`) gets an SD1.5 pipeline class and
  fails to load.
- **No model-discovery endpoint exists.** A new checkpoint must be added to three
  hardcoded lists: both dropdowns in `templates/index.html` and `KNOWN_MODELS`
  in `a1111.py`.
- **Trigger words are attached to the wrong models.** `generate_core_task`
  injects `PixelartFSS` — Onodofthenorth's SD1.5 trigger — into prompts going to
  SDXL-Turbo, where it is inert; step 2 strips it and runs Onodofthenorth
  *without* its trigger. Comparing checkpoints without moving each model's own
  trigger with it is not a fair test.

### Disk space: `df` inside WSL is not free disk space

The distro lives in a **dynamically expanding `ext4.vhdx`** that grows and
**never shrinks on its own**. Deleting files inside WSL frees space that `df`
reports as available while Windows still sees the VHDX at its high-water mark.

Measured 2026-08-21: `df -h /` inside WSL reported **70 GB used of a nominal
1007 GB**, while `ext4.vhdx` was **119 GB on disk** and **C: had 20.5 GB free of
222.9 GB**. Roughly 49 GB had been freed inside the distro and none of it had
come back.

- Always check the Windows side too:
  `Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3"`.
  The VHDX path comes from `HKCU:\Software\...\Lxss` (`BasePath`), not a fixed
  location.
- `df` inside the container is actively misleading here: it reports the ext4
  VHDX's **nominal** ~1007 GB, so a model download can look free when C: is
  nearly full. Check `Get-PSDrive C` on the Windows side before pulling weights.

**D: now holds the models, and the way it holds them matters.** Updated
2026-08-23. Two distinct uses of D:, which are easy to confuse:

1. **Cold archive** — `/mnt/d/wsl-model-archive`, plain DrvFs, managed by
   `scripts/archive-models.sh` and `scripts/archive-all-models.sh`. All 28 GB of
   weights were parked here on 2026-08-23. **Never run a model from here** — 9p
   measured at 44 MB/s against 3.9 GB/s on ext4, 62s to load a pipeline against
   10s.
2. **Live storage** — a dedicated **ext4 VHD** on D:, created by
   `scripts/setup-models-vhd.ps1` and attached with `wsl --mount --vhd`. This is
   what `MODELS_DIR` points at. Native ext4 (no 9p penalty), bytes on the D:
   spindle (C: stops growing).

The mount **does not survive a reboot**. Re-attach it in the same slot as
`lan-expose.ps1`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup-models-vhd.ps1 -AttachOnly
```

That is the price of this arrangement, and forgetting it looks like every model
having vanished.
- Reclaim with `scripts/compact-wsl-disk.ps1` (**Administrator** — `diskpart`
  cannot attach a vdisk otherwise).
- **Do not use `wsl --manage <distro> --set-sparse true`.** WSL refuses it with
  *"Sparse VHD support is currently disabled due to potential data corruption"*
  and only proceeds with `--allow-unsafe`. This distro holds the Postgres data
  directory; occasional manual compaction is the cheaper trade.

**The biggest consumer is usually not the models.** Same date, `docker system df`
showed a **40.93 GB build cache with 31.79 GB reclaimable** — larger than every
model deletion combined. `docker builder prune` is the first thing to try; the
cost is that the next no-cache rebuild re-pulls the torch wheel (15-25 min).

**Moving weights to `/mnt/d` is not a fix.** That is DrvFs/9p, and the storage
benchmarks in the README apply in full: 35x small-file penalty, no page cache.
An HF snapshot is hundreds of files. If a different physical drive is genuinely
needed, move the **VHDX** (`wsl --manage <distro> --move <path>`) or attach a
second ext4 VHD, so the filesystem stays ext4 — never bind-mount weights from a
Windows drive.

#### Result, and two path traps the Cyrillic profile causes

Ran 2026-08-21: **ext4.vhdx 119.0 GB -> 71.7 GB, C: free 25.9 GB -> 73.2 GB.**
diskpart took about 7 minutes and exited 0. No data was lost - all model
directories and the Postgres dir were intact afterwards.

Both bugs hit on the way there came from `C:\Users\<cyrillic>`, and neither is
fixed by quoting:

1. **`-Encoding Ascii` destroys the path.** The diskpart script file must not
   carry a UTF-8 BOM, but writing it as ASCII turns the Cyrillic profile name
   into `??`, and diskpart then reports *"The filename, directory name, or
   volume label syntax is incorrect"* / *"There is no virtual disk selected"*
   (exit `-2147024809`). Fix: resolve the VHDX to its **8.3 short name** first
   (`Scripting.FileSystemObject.GetFile($p).ShortPath` ->
   `C:\Users\74EA~1\...\EXT4~1.VHD`), which is pure ASCII and names the same
   file.
2. **`~` in that short name breaks `-Path`.** PowerShell 5.1 treats `~` as a
   home-directory reference, so `Remove-Item -Path "$env:TEMP\..."` fails with
   *"An object at the specified path C:\Users\74EA~1 does not exist"*. That is a
   `PSArgumentException` from parameter binding, so **`-ErrorAction
   SilentlyContinue` does not suppress it**. Fix: use `-LiteralPath` for every
   file operation under `$env:TEMP`.

Generalise: on this machine, prefer `-LiteralPath` in PowerShell, and expect any
tool that reads an ANSI/ASCII script file to need the 8.3 short path.

**An elevated window's output is invisible to anything else.** The first attempt
failed silently for 30 minutes because diskpart's error went only to the UAC
console. `scripts/compact-wsl-disk.ps1` now tees everything to
`%TEMP%\compact-wsl-disk.log`; keep that when editing it.

#### Docker does auto-start after all

The README's boot runbook says `sudo service docker start` is required because
"Docker does not auto-start in WSL2". Measured 2026-08-21 after a full
`wsl --shutdown`: the daemon was reachable with no manual start, and
`systemctl is-enabled docker` returns **enabled** - this distro runs systemd, so
the unit starts on boot. The manual step is harmless but no longer necessary.

## Model roster (2026-08-21)

Full reasoning and measurements in
[decisions/0002](decisions/0002-image-model-selection.md). The short version:

| role | model | size |
|---|---|---|
| text | `Qwen3-8B-Q8_0.gguf` via llama.cpp | 8.2 GB |
| image gen | `stable-diffusion-xl-base-1.0` **+** one of three pixel-art LoRAs | 6.7 GB + 0.08-0.32 GB |
| pose / step 2 | `All-In-One-Pixel-Model` + `control_v11p_sd15_openpose` | 5.4 GB |

Rules that are easy to get wrong:

- **`"<base>+<lora>"` is a valid model string.** `get_sd_pipeline` splits on `+`,
  loads the right-hand side with `load_lora_weights` and calls `fuse_lora`. A big
  base plus a small style LoRA is the ONLY configuration measured to produce
  structurally real pixel art — 86.6% of pixels from 32 colours, 74.7%
  blockiness. Do not judge a LoRA by file size; it is a delta, not a model.
- **A LoRA only fuses onto the base it was trained against.** Check `base_model`
  in the repo's card data first. `Limbicnation/pixel-art-lora` is FLUX, not SDXL.
- **Step 2 must stay SD1.5.** ControlNet works on SDXL now, but
  `thibaud/controlnet-openpose-sdxl-1.0` is **3.6x weaker** than
  `control_v11p_sd15_openpose` — measured motion 9.57 vs 34.02 at the same
  strength. Step 1 gets SDXL for the art, step 2 gets SD1.5 for the pose.
- **`POSED_STRENGTH` below ~0.75 silently disables pose conditioning.** At 0.60
  the img2img init latent wins and the skeletons do nothing, while every log line
  still says "pose-conditioned".
- **Each checkpoint needs its own trigger word** — see `CORE_TRIGGERS`. A trigger
  from a different checkpoint is inert at best; `PixelartFSS` actively requests a
  four-character sheet.
- **Dedicated image-editing models do not fit.** Their transformers quantise to
  ~9 GB but their text encoders (15.45 GB for Qwen-Image-Edit, T5 for FLUX
  Kontext) have no GGUF and exceed both the 12 GB card and the 11 GB WSL cap.
  SDXL img2img plus ControlNet already covers image-to-image at zero extra cost.
- **Sizing intuition from chat LLMs does not transfer.** GGUF quant tiers give
  chat models a smooth 4/8/16 GB ladder; image models jump 4 -> 6.5 -> 10 -> 20
  -> 24 GB, and quality tracked style-fit rather than size in every test here.

### Keepalive: register it, and know it needs elevation

`scripts/wsl-keepalive.ps1 -Install` registers the logon task;
`-Uninstall` removes it. **Requires Administrator** — `Register-ScheduledTask`
fails with `Access is denied` / `HRESULT 0x80070005` otherwise. The README and
the script's own help both claimed no elevation was needed; both were wrong and
are corrected. Running the keepalive itself stays unprivileged — only
registering the task does not.

The task sets `ExecutionTimeLimit` to zero on purpose: the default is three
days, after which the task engine stops the keepalive and every container dies
looking exactly like the ordinary distro-teardown bug.

Measured 2026-08-21: the keepalive died three separate times in one session,
each time taking the whole stack with it mid-work — including a
multi-GB download and a running generation. `wsl --shutdown` (e.g. from
`scripts/compact-wsl-disk.ps1`) also kills it, so re-run the script after any
deliberate shutdown; the logon task only fires at logon.
