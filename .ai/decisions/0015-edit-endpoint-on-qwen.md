# 0015 - /api/edit runs Qwen-Image-Edit, not FLUX Kontext

Date: 2026-10-21
Status: accepted
Supersedes: the editor choice in [0002](0002-image-model-selection.md) (that
document's SDXL/LoRA findings stand; only `EDIT_BASE` changes)
Related: [0005](0005-back-to-2d-modern-editors.md) (the Qwen GGUF lane),
[0012](0012-qwen-image-default-lane.md) (subprocess-per-stage),
[0014](0014-model-provisioning-manifest.md) (`models.toml`)

## What changed

`POST /api/edit` now runs **Qwen-Image-Edit-2511** (Q2_K GGUF transformer +
the 4-step Lightning LoRA) through `qwen_edit.py` in a child process, instead
of **FLUX Kontext NF4** in the worker process.

`tasks.EDIT_ENGINE` selects it; it defaults to `qwen-gguf`. `flux-nf4` still
reaches the old `get_edit_pipeline` path.

## Why

Not because Kontext was worse. **Because it could not run at all here, and the
reason was a provisioning gap rather than a hardware one.**

`Meatfucker/Flux.1-Kontext-dev-bnb-nf4` is declared under `[[dead]]` in
`models.toml`, so `make provision` never fetched it, and the worker runs
`HF_HUB_OFFLINE=1`. Every call died inside `from_pretrained`:

    OSError: Cannot load model Meatfucker/Flux.1-Kontext-dev-bnb-nf4:
    the model is not fully cached locally ... outgoing traffic has been disabled

So the endpoint was reachable and could only fail. That is the shape of bug
0014 exists to catch, arriving by the one route the manifest test cannot see:
the model was *declared dead on purpose*, and the feature that needed it was
switched off, so nothing was inconsistent until the feature was switched on.

The alternative was to declare it and download ~12.5 GB. Qwen-Image-Edit-2511
needs **no download**: all five repos the lane uses were already cached and
verified present 2026-10-21 —

    ovedrive/Qwen-Image-Edit-2511-4bit                  (text encoder, NF4)
    Qwen/Qwen-Image-Edit-2511                           (config)
    unsloth/Qwen-Image-Edit-2511-GGUF                   (qwen-image-edit-2511-Q2_K.gguf)
    fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA       (not used here; see below)
    lightx2v/Qwen-Image-Edit-2511-Lightning             (4-step, used)

— because `/api/jobs` already builds every sprite sheet with them.

### The paradox this resolves

`tasks.py` says, above `FLUX_EDIT_BASE`, that Qwen-Image-Edit "does not fit
this card": its transformer is 10.71 GB and bitsandbytes 4-bit layers cannot
be split or CPU-offloaded. **That is true, and it is about the NF4 build.** It
is not true of the Q2_K GGUF at 7.1 GB, which is the entire reason
`qwen_edit.py` exists — see its docstring, which opens by quoting and
overturning that same line. Both statements are correct about different
builds. Anyone re-reading the NF4 numbers should read `qwen_edit.py` first.

## How

One subprocess, two memory-disjoint passes — the same shape as
`turnaround_split`, and a child process for the reason 0012 gives: process
exit is the only guaranteed-complete VRAM release, so the worker's own CUDA
context stays clean for the next SDXL job. The worker is `--pool=solo`, so a
poisoned context there would take plain sprite generation down with it.

    qwen_edit.py edit-one   encode_only (text encoder alone, unloaded)
                            denoise_only (GGUF transformer, then VAE)

`tasks._run_qwen_edit` reads the child's `STAGE` lines for the progress bar,
kills it at `QWEN_EDIT_TIMEOUT_S` (900 s - a hang detector, not a budget), and
trips the GPU breaker on a CUDA-fault signature in its output, because a fault
in a child never reaches the parent's breaker on its own.

### Three details that are decisions, not defaults

- **`with_angles=False`.** The camera LoRA pins *pose* as well as framing
  (measured, see `denoise_only`). For a turnaround that is what you want; for
  an instruction edit it is the opposition — asked to remove a figure it would
  hold the composition it was trained to hold. `EDIT_LORAS` therefore stays
  empty on this path.
- **`cfg_scale` is not passed through.** `denoise_only` pins
  `true_cfg_scale=1.0` because Lightning is step-distilled: above 1.0 it does
  not error, it produces over-guided washed-out output that reads like a
  prompt problem — the same trap 0002 records for SDXL-Turbo.
  `/api/edit-capabilities` reports `guidance_range: [1.0, 1.0]` to say so
  rather than inviting a caller to tune a nailed-down dial.
- **The `Form` defaults changed**, from `steps=20, cfg_scale=4.0` to
  `steps=0, cfg_scale=1.0`. The old pair was right for Kontext and is the
  documented way to wash out a distilled model. `steps=0` means "the engine's
  own default" (4 here) rather than pinning one engine's quirk into the API.

### Gateway

`model_gateway.GATED["tasks.edit_image_task"]` now resolves to `QWEN_EDIT`,
the **same label as `build_sheet_job`**. That is deliberate: it is the same
transformer on the card, so an edit queued behind a sheet build is not
deferred, while one queued behind an SDXL job is. The lambda reads
`tasks.EDIT_ENGINE` rather than hardcoding, so `flux-nf4` does not leave the
gateway gating against a model the worker is not loading.

## What stayed

- **`EDIT_ENABLED` is still required**, and still defaults to off. Its
  original justification was NF4-specific (12.52 GB resident against an 11 GB
  WSL cap, kernel OOM-killing the worker mid-load at component 4 of 7 with no
  traceback). That cap is now 42 GB, ~38 GB available, and the Qwen route runs
  in a subprocess, so a hard kill costs one edit rather than the worker. It
  stays opt-in because an edit holds the whole card for its duration.
  - **Spelling tripwire.** `EDIT_ENABLE` (no `D`) is read by nothing. Set that
    and `env` inside the container shows it while `/api/edit` keeps answering
    503 `enable_with: EDIT_ENABLED=1`. It must be set on **both**
    `sprite-generator` and `sprite-worker` — the API answers the 503 from the
    same constant the worker reads — and the containers must be **recreated**,
    not restarted, for compose to re-read the environment.
- **`get_edit_pipeline` and the FLUX constants**, behind `EDIT_ENGINE`. Not
  dead weight: their comments are the measurement of why NF4 editing does not
  fit a 12 GB card, and the function now refuses with a message naming the
  *choice* rather than failing minutes later with a message about a cache.
- **`Meatfucker/...` under `[[dead]]`**, with the `why` rewritten. The string
  still appears in `tasks.py`, so `make test-models-manifest` still needs the
  declaration.

## Known limits

- **Output is square**, at `EDIT_MAX_SIDE` (512). `denoise_only` renders
  `size x size`, so a wide source is squashed rather than letterboxed. Every
  concept core is already square; anything else is worth resizing first. The
  FLUX path used `thumbnail` and preserved aspect — that difference is real.
- **Encode happens at 384 px** (`QWEN_ENCODE_SIZE`) while the render is 512.
  Deliberate and pre-existing: the text encoder is a VL model, so image
  resolution drives sequence length and VRAM. Vision tokens only *condition*
  the embedding; the edit is still applied at render resolution.
- **A 10.5 GiB free-VRAM preflight** (`QWEN_GPU_BUDGET`) refuses rather than
  loading onto a short card. The same constant `load_pipeline` uses, so the
  two paths cannot drift. On this host a warm brain or a resident pipeline is
  enough to trip it; that is a refusal, not a fault.
