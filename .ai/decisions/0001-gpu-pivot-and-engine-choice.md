# 0001 — GPU pivot, container runtime, and inference engine

Date: 2026-08-19
Status: Accepted (phases 3–5 not yet implemented)

## Context

The project generated sprites on CPU via diffusers, on a host with an i3-8100
(4 threads) and a WSL memory cap of ~7 GB. An RTX 3060 12 GB is available. The
project simultaneously needs to become a LAN REST service consumed by the
something2 admin panel.

## Decisions

### 1. Pivot inference to CUDA; make device a runtime setting

`tasks.py` hardcoded CPU in six places, including `torch.set_num_threads(12)` on
a 4-thread host. Device and dtype now resolve once at import from
`COMPUTE_DEVICE`, which compose sets per service.

fp16 on CUDA (the 3060 has weak bf16 throughput and fp16 halves VRAM); the CPU
path keeps float32, and bfloat16 for FLUX. The worker gets the GPU; the API
process is pinned to CPU so it cannot hold VRAM. FLUX on CUDA uses
`enable_model_cpu_offload()` rather than `.to("cuda")` — a full FLUX pipeline at
fp16 is ~24 GB, double the card.

torch is installed in its own `Dockerfile.cuda` layer, taking **PyPI's default
build with no index override**, and is deliberately absent from
`requirements.cuda.txt`.

> **Superseded 2026-08-19, same day.** The NVIDIA driver was updated from
> **551.86 (CUDA 12.4)** to **610.88 (CUDA 13.3)** partway through this work.
> The pinning described below existed only to match the *old* driver; PyPI's
> default torch now runs natively and the workaround was removed. Free VRAM also
> rose from ~8.4 GB to ~11.7 GB, which relaxes pipeline budgeting.
>
> The history is kept because it is the exact trap anyone will re-enter if the
> driver is ever downgraded or the image is built for an older machine.

This took two failed attempts and both are worth recording, because **neither
index flag works on its own**.

*Attempt 1 — `--extra-index-url` inside the requirements file.* That does not
pin anything; it merely *adds* an index, after which pip resolves across it and
PyPI together and installs the highest version found. PyPI's default `torch`
build won, and the image shipped `torch 2.13.0+cu130` against a CUDA 12.4
driver. It imported fine and reported `torch.cuda.is_available() == False` —
indistinguishable from a missing container runtime:

```
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old
(found version 12040)
```

*Attempt 2 — exclusive `--index-url`.* This forces the right CUDA build but also
forces torch's **dependencies** to resolve from the PyTorch index. That index
serves `typing_extensions` under a name pip rejects as inconsistent
(`expected 'typing-extensions', but metadata has 'typing_extensions'`), so pip
falls back to the sdist, which needs `flit_core` as a build dependency — not
published on that index at all. The install fails outright:

```
ERROR: Could not find a version that satisfies the requirement flit_core<4,>=3.11
       (from versions: none)
```

*Resolution, if you ever need it again.* Pin `torch==<ver>+cuXXX` **and** use
`--extra-index-url`. The `+cuXXX` local version tag exists only on the PyTorch
index, so that exact wheel must be chosen, while ordinary dependencies still
resolve from PyPI. (For CUDA 12.4 on cp310 the newest was `2.6.0+cu124`.)

*Resolution actually shipped.* The driver was updated instead, so
`Dockerfile.cuda` now just installs `torch` from PyPI. Updating the driver is
the better fix whenever it is available: it removes a version-matching
constraint permanently rather than encoding it in the build.

`Dockerfile.cuda` asserts `torch.version.cuda` is non-null at build time, so a
CPU-only wheel fails the build instead of silently shipping an image that runs
on CPU. The assertion no longer checks a *specific* CUDA version — with a
current driver, any CUDA build is acceptable.

**Operational note:** the first failure was masked by running the build as
`docker compose build ... | tail -30`, which discards the pipeline's exit status
and reported success for a failed build. Use `set -o pipefail` when piping build
output.

### 2. Docker Engine inside WSL2 Ubuntu — not Docker Desktop

**This decision was reversed twice; the reasoning matters.**

Initially: Docker Engine in WSL, for leaner RAM on a 16 GB box and to avoid 9p
bind-mount latency.

Then reversed to Docker Desktop, because this is **Windows 10**, where WSL's
`networkingMode=mirrored` is unavailable (Windows 11 22H2+ only). Docker
Desktop forwards published ports to the Windows host automatically, which would
have made LAN exposure free.

Then reversed back, on evidence. The local Docker Desktop install is broken:

```
[main.wsl] failed to read component versions:
    open /opt/docker-desktop/componentsVersion.json: no such file or directory
[main.wslexec][E] wsl.exe --unregister docker-desktop: WSL_E_DISTRO_NOT_FOUND
[enginedependencies] still waiting for init control API after 33.4s
→ backend process exited
```

The `docker-desktop` distro is registered but unprovisioned, and the
`com.docker.service` privileged helper is absent. Repair would mean
unregistering that distro — which on the merged single-distro layout destroys
all local images and volumes — plus an admin reinstall.

**Chosen:** Docker Engine + NVIDIA Container Toolkit installed directly in
Ubuntu. Non-destructive, scriptable, no Desktop RAM overhead. The cost is that
LAN exposure now needs `netsh interface portproxy` — one admin command, judged
cheaper than the repair.

Consequence: **all `make` targets must be run from inside WSL.**

### 3. ComfyUI as the inference engine; this repo as the sprite brain

Options weighed: (A) keep diffusers and add CUDA, (B) ComfyUI as engine with
this repo orchestrating, (C) build a backend-registry gateway first, (E) do LoRA
training first.

**Chosen: B**, with A's fixes retained as prerequisites (they were required
regardless), C's seam emerging in phase 3, and E as phase 5.

Reasons:

- A synchronous façade over an asynchronous engine must be built anyway to meet
  something2's 5-minute sync constraint. B's main cost is therefore already
  sunk.
- The maintenance currently broken in this repo — FLUX key hot-patching,
  hardcoded snapshot hashes, a downloaded GGUF nothing loads — is exactly the
  category ComfyUI absorbs.
- ComfyUI's VRAM/offload management is materially better than the `pipes` dict
  with mid-flight `pipes.clear()`, which matters on a card with ~8.5 GB free.

Rejected A because every new model family becomes hand-written pipeline code.
Rejected C-first as speculative: a gateway fanning out to backends this host
cannot run concurrently is architecture for hardware we do not have.

### 4. Impersonate Automatic1111 rather than define a protocol

something2's provider system already speaks A1111 and OpenAI-images. Exposing
`POST /sdapi/v1/txt2img` and `GET /sdapi/v1/sd-models` means **zero code changes
in something2** — the admin only registers a provider.

### 5. Do not repair the FLUX diffusers path

`tasks.py` and `models.txt` disagree about which FLUX weights exist. Fixing it
in diffusers is work that decision 3 deletes. Interim: point step 2 at SDXL
img2img so the two-step workflow stays testable.

## Consequences

- `docker-compose.cuda.yml` now exists; the Makefile had referenced it for a
  long time, so every make target previously failed on a GPU host.
- `.env.example` is tracked, and `make env` bootstraps `.env`.
- Model weights live on WSL ext4 via `MODELS_DIR`; the repo stays on `/mnt/c`.
- LAN exposure remains unsolved and is the main risk to deliverable #1.
