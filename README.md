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

## Starting the stack (after every reboot)

Nothing here survives a reboot on its own. Run these in order — steps 1 and 4
are the two that get forgotten, and both fail in ways that do not look like
their cause.

```powershell
# --- Windows PowerShell -------------------------------------------------

# 1. Keep the distro alive. Do this FIRST. Without it WSL2 tears the distro
#    down as soon as no wsl.exe client is attached, and every container dies
#    with a clean exit code and nothing wrong in its logs.
powershell -ExecutionPolicy Bypass -File .\scripts\wsl-keepalive.ps1
```

```bash
# --- inside WSL (Ubuntu) ------------------------------------------------

# 2. Docker does not auto-start in WSL2 - there is no boot sequence to hook.
sudo service docker start

# 3. Bring the stack up. `make up` waits for db, redis and the sprite
#    services to report healthy, then applies migrations.
cd "$(find /mnt/c/Users -maxdepth 5 -type d -name llama-cpp-only-cpu -print -quit 2>/dev/null)"
make up

# 5. Load a checkpoint into VRAM before anything calls the API. A cold model
#    takes minutes to fetch and load, which blows past every client timeout -
#    including something2's 5 minute cap, which cannot poll.
make warm
```

```powershell
# --- back in an ELEVATED Windows PowerShell -----------------------------

# 4. Re-publish the ports. REQUIRED after every reboot: WSL2 gets a new IP
#    each boot, the portproxy entries persist pointing at the OLD one, and
#    they fail silently - LAN clients just hang. See "Home-network exposure".
powershell -ExecutionPolicy Bypass -File .\scripts\lan-expose.ps1
```

Then check it actually works, rather than assuming:

```bash
make smoke                                    # GPU in use, models listed, txt2img round-trip
./scripts/verify-something2-contract.sh <lan-ip>   # what something2 will see
```

### If WSL will not start at all

```
WSL2 is unable to start since virtualization is not enabled on this machine.
Error code: Wsl/Service/CreateInstance/CreateVm/HCS/HCS_E_HYPERV_NOT_INSTALLED
```

Virtualization is off at the firmware level. Confirm from Windows:

```powershell
(Get-CimInstance Win32_ComputerSystem).HypervisorPresent          # expect True
(Get-CimInstance Win32_Processor).VirtualizationFirmwareEnabled   # expect True
```

Enable **Intel VT-x** (or AMD **SVM**) in BIOS/UEFI, then **fully power off and
on** — a warm restart does not always re-read the setting. If it is already on
in firmware, the Windows components are missing; from an elevated prompt:

```powershell
wsl.exe --install --no-distribution     # re-enables Virtual Machine Platform + WSL
```

Reboot again afterwards.


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

### Seeing the models in the project tree

Weights used to sit in `<repo>/models`; the compose mount still defaults to it
(`${MODELS_DIR:-../../models}`) and only `.env` redirects it to ext4. Moving
them was a speed decision, and it cost the visibility that having them in the
repo gave. You can have both — link the cache back into the project instead of
moving it:

```powershell
# ELEVATED PowerShell. Windows requires elevation for symlinks, and a junction
# (mklink /J, no privilege needed) cannot target a UNC path, which is what a
# WSL path is from the Windows side. There is no unprivileged route.
powershell -ExecutionPolicy Bypass -File .\scripts\link-models.ps1

# Undo (removes the link only — the weights are never touched):
powershell -ExecutionPolicy Bypass -File .\scripts\link-models.ps1 -Remove
```

`models\` then appears at the repo root and browses normally in Explorer, VS
Code and the project tree, while every read still goes through ext4. Docker is
unaffected: it bind-mounts `MODELS_DIR` directly and never sees the link. The
script reads the target from `MODELS_DIR`, so it keeps working if that changes,
and it says so and exits if `MODELS_DIR` is already repo-relative.

`models` is gitignored as a single entry rather than `models/*` — otherwise git
tracks a symlink pointing at a path that exists on exactly one machine.

**If you would rather have the real files in the repo**, set
`MODELS_DIR=../../models` in `compose/develop/.env` and move the cache across.
It works. Expect a cold SDXL-Turbo load to go from ~2 s to ~160 s, the HF cache
to lose its blob symlinks, and something2's 300 s no-polling cap to become
reachable on a cold model.

To browse the whole ext4 side from Windows, open this in Explorer (pin it for
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

This runs in **Windows PowerShell, as Administrator**. Not in WSL — `netsh
portproxy` and the Windows firewall are Windows-side, and forwarding *into*
WSL's NAT is the entire point.

Two separate things block it on a default Windows 10 install, and they need
different fixes:

| Symptom | Cause | Fix |
|---|---|---|
| `cannot be loaded because running scripts is disabled` | Execution policy is `Restricted` | Launch `powershell.exe` with `-ExecutionPolicy Bypass`. The flag cannot be applied to a session that is already running, so `.\scripts\lan-expose.ps1` on its own will always fail. |
| `Must run as Administrator` | `netsh portproxy` and `New-NetFirewallRule` require elevation | Open the window with Win+X -> "Windows PowerShell (Admin)" |

Open an elevated window, then:

```powershell
cd (Get-ChildItem $HOME -Recurse -Depth 4 -Directory -Filter llama-cpp-only-cpu -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
powershell -ExecutionPolicy Bypass -File .\scripts\lan-expose.ps1

# To undo everything it created:
powershell -ExecutionPolicy Bypass -File .\scripts\lan-expose.ps1 -Remove
```

Or, from a normal (non-elevated) PowerShell already sitting in the repo — this
spawns the elevated window for you, via a UAC prompt:

```powershell
Start-Process powershell -Verb RunAs -ArgumentList `
  "-NoExit -ExecutionPolicy Bypass -File `"$PWD\scripts\lan-expose.ps1`""
```

To stop passing the flag every time, `Set-ExecutionPolicy -Scope CurrentUser
RemoteSigned` makes local scripts runnable permanently. The Administrator
window is still required — that part is `netsh`, not policy.

The script defaults to distro `Ubuntu`; pass `-Distro <name>` if `wsl -l -v`
shows something else.

**Re-run it after every WSL restart** — the WSL IP is assigned per boot, and
stale portproxy entries fail silently.

Then confirm the service is actually reachable from the network, not just from
this machine:

```bash
./scripts/verify-something2-contract.sh <lan-ip>
```

`localhost` passing proves nothing about LAN reachability; this is the half
that matters for something2.


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
- The keepalive dies at logoff/reboot. Register it as a logon task so you stop
  having to remember (normal PowerShell, **no** elevation needed):

  ```powershell
  $a = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument `
       ('-WindowStyle Hidden -ExecutionPolicy Bypass -File "' +
        (Join-Path $PWD 'scripts\wsl-keepalive.ps1') + '"')
  $t = New-ScheduledTaskTrigger -AtLogOn
  Register-ScheduledTask -TaskName 'WSL keepalive' -Action $a -Trigger $t
  ```

  This does **not** cover `lan-expose.ps1`, which needs elevation and so cannot
  be triggered the same way without a task configured to run with highest
  privileges.
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


### Step 2 (spritesheet): how it works, and what it still cannot do

Step 2 produces a usable sheet now — a transparent 512x256 sheet, one
consistent character across all frames, in about 10 seconds per action. It took
six attempts, and the last one only worked after a step-1 defect was found.

| Attempt | Approach | Result |
|---|---|---|
| 1 | Original FLUX path | Could not load at all — wrong weights wired up |
| 2 | img2img on a 512x128 strip | Core squashed 4:1; model denoised from a smear |
| 3 | Per-frame img2img at 128x128 | RGB static — SD1.5 cannot generate below ~512 |
| 4 | Per-frame at 512, downscaled | Real sprites, but a different character per frame (strength 0.95 discarded the reference) and each frame held its own mini-row |
| 5 | One row per action, alpha-sliced, strength 0.65 | Model rendered vertical stacks, not a row; slicer assumes a row |
| 6 | Per-frame at 512, **shared seed**, strength 0.55, composed in PIL | Works |

Three things had to be true at once, and every earlier attempt broke at least
one of them.

**Stop asking the model for a layout.** Attempts 2, 4 and 5 all requested "a row
of N frames" and hoped. The same prompt returns a row, a vertical stack or a
loose grid on consecutive runs; layout is not something classifier-free guidance
steers well. Step 1 already renders a single centred character reliably, so
step 2 now renders one frame at a time and composes the strip in PIL, where
layout is arithmetic instead of a sample from a distribution.

**The seed is what holds identity, not the strength.** Attempt 4 was the right
approach at the wrong settings. Measured on a 4-frame walk (mean absolute
inter-frame difference, 0-255):

| strength | motion | drift from frame 0 | |
|---|---|---|---|
| 0.35 | 1.7 | 2.1 | barely moves |
| 0.45 | 3.0 | 3.7 | used for static actions |
| 0.55 | 4.7 | 5.8 | used for dynamic actions |
| 0.65 | 8.0 | 10.1 | design starts changing (a hat appeared mid-strip) |
| 0.55, seed per frame | **23.2** | 24.2 | four different characters |

Same strength, only the seed policy changed: 4.7 versus 23.2. Every frame now
uses `parent_seed` — not `parent_seed + i` — so the noise and the denoising
trajectory are identical and only the prompt differs.

**The core was carrying stowaways.** The real reason step 2 output looked wrong
for so long: SDXL-Turbo runs at guidance 0, which switches classifier-free
guidance off entirely, which makes the negative prompt — including "multiple
characters, group, crowd" — a no-op. Cores were coming out as one large
character ringed by five small copies of itself, and step 2 faithfully carried
all six into every frame. No prompt can fix that on a distilled checkpoint, so
`_isolate_largest_sprite` keeps the largest connected opaque region and deletes
the rest. Geometry, not prompting.

A related bug hid behind it: the core was handed to img2img with a transparent
background, `.convert("RGB")` turned that black, and `remove_background` refused
to key a dark corner — so a finished sheet came back 100% opaque on a black
backdrop while every colour-based check passed. The core is now composited onto
white first, corner sampling takes a majority vote instead of a brightness
threshold, and `make test-flow` fails a sheet under 5% transparent.

**What it still cannot do.** Motion is a shift in posture and limb position, not
a real walk cycle, and asking for a side profile does not turn the character
around. That is inherent to img2img: it preserves composition, and pose *is*
composition — the same knob controls both, so identity and pose cannot be
separated. Frames also often keep the model's ground-shadow ellipse, which is
connected to the feet and so survives both the background key and the
largest-blob filter; negative prompting reduces it but does not clear it.

Separating pose from identity needs a second conditioning channel, which means
ControlNet (OpenPose). That is available in diffusers directly
(`StableDiffusionControlNetImg2ImgPipeline` plus a ~1.4GB ControlNet
checkpoint) — a much smaller step than the full ComfyUI migration this section
used to recommend, and the natural next one.
## Known issues

- **The FLUX branch of step 2 cannot load on a clean machine.** `tasks.py`
  expects `/models/flux1-schnell.safetensors` plus a hardcoded HF snapshot hash,
  while `models.txt` downloads `flux-2-klein-4b-Q8_0.gguf`. Neither is wired to
  the other. Unfixed and low priority: step 2 runs on
  `Onodofthenorth/SD_PixelArt_SpriteSheet_Generator` (SD1.5), which works, and
  FLUX is a worse fit for this job anyway — schnell is distilled, so guidance is
  off and negative prompts do nothing, which is the exact trap that hid the
  duplicate-character bug in step 1.

- **Sprites keep the model's ground shadow.** A green ellipse under the feet,
  connected to the sprite, so neither the background key nor the largest-blob
  filter removes it. Cosmetic; see the step 2 section above.

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
