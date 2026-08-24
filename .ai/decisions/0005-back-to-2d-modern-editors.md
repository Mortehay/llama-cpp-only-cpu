# 0005 - Abandon the 3D conveyor; rebuild step 2 on a 2026 instruction editor

Date: 2026-08-23
Status: Proposed. Supersedes the *direction* of
[0004](0004-pivot-to-3d-conveyor.md) and reopens most of
[0003](0003-directional-sprites-and-view-cores.md).

## Why this exists

The conveyor has been rebuilt four times and still does not produce the
reference sheets. This ADR records the diagnosis and the plan, and is written
after looking at the *output* rather than at the code.

## Diagnosis: three separate failures, fought as one

`images/sheet_3025b822691a.png` (newest sheet, 2026-08-22 20:41, 1024x512,
cells 256px) shows all three at once.

### 1. There is no pixelation stage at all

The cells are 256px anti-aliased, gradient-shaded renders. The supplied
reference sheets are 32-48px cells, hard 1px edges, roughly 16-32 colours.
`images/core_d92c08a1c54a.png` makes it unmistakable: a smooth red gradient
with anti-aliased edges at 512px. That is not pixel art; it is a painting of
pixel art.

Nothing in `tasks.py` quantises a palette, snaps to a pixel grid, or hard
thresholds alpha. `fit_into_frame` scales with a resampling filter that
*creates* intermediate colours. The style has been chased entirely through
prompt text and checkpoint choice, which is the one place it cannot be fixed.

**Cheapest failure to fix, and worth fixing first**: deterministic post
processing, no model, no download, no VRAM, and it changes what every existing
output looks like.

### 2. The model generation is three years out of date

The stack is SD1.5 + SDXL + `control_v11p_sd15_openpose` + IP-Adapter, with
hand-authored COCO-18 skeletons in `poses.py` (527 lines) and per-action
`strength` tuning in `action_prompts.json`. Every one of those pieces exists to
work around the same limitation: **SD1.5/SDXL cannot follow an instruction about
an image.** They can only be nudged with an init latent, a control map and an
adapter. The ADR 0002/0003 measurements are a careful record of that nudging
failing.

The 2026 answer is an instruction-following edit model trained for subject
consistency. `Qwen-Image-Edit-2511` (Dec 2025) lists "mitigate image drift" and
"improved character consistency" as headline changes.

### 3. The 8-direction problem was solved with 3D, and did not need to be

ADR 0004 concluded that ~150 consistent cells is structurally impossible with 2D
sampling and pivoted to `TripoSR -> UniRig -> Blender`. The mesh stage was built
and works. The rig stage was built and **fails on 4 of 5 generated meshes**, with
no measured input property predicting which succeed.

That leg is unnecessary. `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA`
(Apache 2.0) re-renders an input image from 96 camera poses -
**8 azimuths x 4 elevations x 3 distances** - trained on Gaussian-Splatting
renders. The prompt form is literally
`<sks> right side view eye-level shot medium shot`. It does in one 2D edit what
`derive_view_core`, `VIEW_TRIGGERS`, the 3/4-view work and the entire mesh+rig
conveyor were built to approximate.

ADR 0004's reasoning was not wrong given what was on the table. What it missed
is that the tool arrived.

### Why the Qwen LoRAs were ruled out before, and why that call was wrong

`tasks.EDIT_LORAS` is empty, with this comment:

> the two that were wanted - fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA
> (turn the subject) and prithivMLmods/...-Object-Remover - are trained against
> Qwen-Image-Edit, whose transformer does not fit this card.

True **only of the bitsandbytes NF4 path** the code takes. Not true of GGUF.
`unsloth/Qwen-Image-Edit-2511-GGUF` publishes Q2_K at 7.47 GB, Q3_K_M at
9.92 GB, Q4_0 at 11.9 GB, and `diffusers` loads GGUF transformers through
`from_single_file` + `GGUFQuantizationConfig` - which `tasks.py` **already
imports at line 22** and never uses for this.

"Does not fit this card" was concluded without testing the quantisation format
that exists specifically to make it fit.

## Decision

### The new conveyor

    concept        key          angles           poses            pixelate        pack
    ------------   ----------   --------------   --------------   -------------   ------------
    Z-Image-Turbo  chroma-key   Qwen-Edit-2511   Qwen-Edit-2511   palette-lock    grid PNG
    6B, 8 steps    + RMBG-2.0   +Angles LoRA     +AnyPose LoRA    + NN downscale  + atlas JSON
    (or SDXL)      fallback     8 azimuths       N frames/action  + alpha thresh  + RGBA

Every stage after `concept` is an *edit of one image*, never a fresh sample.
That is the property ADR 0004 went to 3D to obtain.

### Palette lock is the consistency trick that costs nothing

Extract one 16-32 colour palette from the concept image, store it with the
character, quantise **every** cell against it. At 48px a hue shift is what reads
as "a different character" - far more than silhouette detail does. Locking the
palette removes most visible drift for free, in numpy, with no model involved.
Do this before spending any more VRAM on identity preservation.

### Model roster

| Role | Model | Size | Licence | Why |
|---|---|---|---|---|
| Concept | `Tongyi-MAI/Z-Image-Turbo` | 6B, GGUF ~6 GB | Apache 2.0 | 8 NFEs, far better prompt adherence than SDXL, fits with room to spare |
| Editor | `Qwen/Qwen-Image-Edit-2511` GGUF Q3_K_M | 9.92 GB | Apache 2.0 | Only model with a purpose-built 8-azimuth LoRA and a pose-transfer LoRA |
| Angles | `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA` | small | Apache 2.0 | 8 directions, directly |
| Pose | `lilylilith/AnyPose` (Qwen Edit 2511) | small | see card | Pose from a reference image, no ControlNet |
| Speed | `lightx2v/Qwen-Image-Edit-2511-Lightning` 4-step | small | see card | ~10x fewer steps; documented best at LoRA strength 0.7 |
| Matte | `briaai/RMBG-2.0` or BiRefNet | ~1 GB | see card | Fallback when chroma-key fails |

**Fast fallback if the 20B editor is too slow in practice:**
`black-forest-labs/FLUX.2-klein-4B` (Apache 2.0, 4 steps, multi-reference
editing, FP8/NVFP4 checkpoints published by BFL). No angles LoRA, so directions
must come from instruction text - worth a bake-off, not an assumption.

### What gets retired

- `poses.py` (527 lines of hand-authored COCO-18 skeletons) - AnyPose takes a
  reference image instead.
- `derive_view_core`, `VIEW_TRIGGERS`, the IP-Adapter wiring, and the per-action
  `strength` / `ip_adapter_scale` tuning in `action_prompts.json`.
- `src/mesh_worker/`, `src/rig_worker/`, `compose/develop/mesh_worker/`,
  `compose/develop/rig_worker/` - the 3D leg. Their two Docker images alone are
  **21.2 GB and 24.6 GB**, the largest single disk consumers on this machine.

Keep the ADRs. They are the record of *why* this cost was paid and are the most
valuable thing in the repo. Retire the code to a branch or `legacy/` rather than
deleting it.

## Hardware plan, and the blocker that has to clear first

### C: was down to 12.9 GB free (measured 2026-08-22)

Down from 47.7 GB earlier the same day. Nothing in the roster above can be
downloaded until this clears. Three levers, largest first:

1. `docker image rm` on `mesh-worker` (21.2 GB) and `rig-worker` (24.6 GB) once
   the 3D leg is formally retired.
2. `docker builder prune` - 47.58 GB of build cache, 17.23 GB reclaimable.
   Cost: the next no-cache rebuild re-pulls the torch wheel (15-25 min).
3. `scripts/archive-all-models.sh` - parks all 28 GB of live weights on D:.
   Reversible with `archive-models.sh restore <name>`.

All three free space *inside* the ext4 VHDX. C: itself recovers only after
`scripts/compact-wsl-disk.ps1` (Administrator).

### Put the models on D:, but as ext4 - not as a bind mount

D: has ~60 GB free. `scripts/archive-models.sh` already treats D: as **cold
storage only**, with measurements behind it: 44 MB/s over 9p against 3.9 GB/s on
ext4, and 62s to load a pipeline from `/mnt/d` against 10s from ext4.

To move *live* weights onto the D: spindle without paying that, attach a
**second ext4 VHD**:

    # Windows, elevated - one time
    wsl --mount --vhd D:\wsl-models.vhdx --name models

Format ext4 once, mount it, point `MODELS_DIR` at it. The filesystem stays ext4
so the 9p penalty never applies; the bytes live on the D: physical disk so C:
stops growing. `project-context.md` already names this escape hatch; it is now
the critical path, not an option.

### VRAM / RAM budget on a 3060 12 GB + 16 GB host

Free VRAM ~11.7 GB; WSL capped at 11 GB RAM by `~/.wslconfig`. The 20B editor
does not fit naively. It fits under a **three-pass batch structure**, which is
the part of this plan that makes 150 cells feasible at all:

1. **Encode pass.** Load the Qwen2.5-VL text encoder (4-bit, ~5 GB), encode
   *every* prompt for the whole sheet, cache embeddings to disk, unload.
2. **Denoise pass.** Load the GGUF transformer once and keep it resident for all
   ~150 cells. No per-cell reload - that is the difference between a 40-minute
   job and an overnight one.
3. **Decode pass.** VAE is small; decode and pixelate on the way out.

Raise `~/.wslconfig` to `memory=12GB`, `swap=24GB` (swap needs the disk cleared
first, or place it on the new D: VHD).

**Honest throughput estimate, to be measured not assumed:** 4-step Lightning at
512px on a 3060 is roughly 8-20 s/cell once resident. A 150-cell character is
~30-50 minutes. This is a batch job, not an interactive one.

## Consequences for the something2 contract

`a1111.py` is a **synchronous** txt2img facade with a 240 s budget, because
something2 refuses submit/poll (their SOMET-334). A 150-cell sheet cannot be
produced inside 240 s under any plan on this hardware.

The facade must stop being the generator and become a **cache reader**: sheets
are produced by a background job, stored, and the facade serves the finished
PNG. A request for a sheet that does not exist yet enqueues it and returns an
error something2 can retry, rather than blocking. Needs confirming against
something2's actual calling code, which per
`.ai/specs/something2-provider/contract.md` has still never been read.

## Order of work

Ordered so each step produces evidence before the next spends money on
downloads.

| # | Step | Cost | Proves |
|---|---|---|---|
| 1 | Pixelation + palette-lock + alpha module, applied to existing outputs | none | whether "mess" is partly post-processing. Immediate, visible |
| 2 | Clear C:, attach the D: ext4 VHD, repoint `MODELS_DIR` | none | downloads become possible at all |
| 3 | Tiles / items / props path: Z-Image-Turbo -> key -> pixelate | 6 GB | end-to-end transparent asset, no animation, no directions. Ships first |
| 4 | Qwen-Edit GGUF loads at all on this card, one cell | 10 GB | the claim this ADR overturns |
| 5 | Angles LoRA: one concept -> 8 directions, compared | small | the 3D leg was unnecessary |
| 6 | AnyPose: one direction -> 6-8 action frames | small | animation without `poses.py` |
| 7 | Three-pass batch runner, one full character sheet | none | the throughput estimate above |
| 8 | Rework `a1111.py` into a cache reader + job queue | none | something2 integration survives |

Steps 3 and 4-7 are independent. Step 3 delivers usable assets for tiles, items
and static entities while the character conveyor is still being proven.

## Findings from the first build (2026-08-23)

Three things that cost time and are not obvious from any model card.

### `snapshot_download` treats a mistyped filter as success

The repo is `unsloth/Qwen-Image-Edit-2511-GGUF`; its files are
`qwen-image-edit-2511-*.gguf` — **lower case**. An `allow_patterns` entry with
the repo's capitalisation matches nothing, and huggingface_hub does not raise.
It prints `Fetching 0 files`, returns a valid path to an empty snapshot, and
exits 0. 10 GB silently did not arrive and the fetch script reported success.

`scripts/fetch-qwen-edit.py` now treats an empty snapshot as a failure. Any
future `allow_patterns` needs the same guard — this is a property of the
library, not of this repo.

### Filters on these repos are mandatory, not tidiness

`lightx2v/Qwen-Image-Edit-2511-Lightning` is **107.7 GB**: 4- and 8-step LoRAs in
bf16 *and* fp32, a 20.5 GB fp8 full checkpoint, and 60 split block files. The
849 MB file we want is under 1% of it. An unfiltered `snapshot_download` fills
the disk long before it finishes.

### The worker image is older than its own requirements

`requirements.cuda.txt` lists `bitsandbytes>=0.43.0`, but the running
`sprite-worker` image was built 2026-08-20 and does not contain it —
`scripts/check-diffusers-api.py` catches this. It matters because the NF4 text
encoder cannot load without it, so the 2D route does not run on the current
image at all.

Everything else needed is already present and current:

| | |
|---|---|
| diffusers | 0.39.0 — has `QwenImageEditPlusPipeline`, `QwenImageTransformer2DModel`, `GGUFQuantizationConfig` |
| transformers | 5.15.1 — has `Qwen2_5_VLForConditionalGeneration`, `BitsAndBytesConfig` |
| gguf | 0.19.0 |
| torch | 2.13.0+cu130 |

The fix is a `sprite-worker` rebuild. Note the cost: `Dockerfile.cuda` installs
torch behind a **pip cache mount**, and `docker builder prune` (run the same day
to reclaim 38.4 GB) discards cache mounts along with layer cache — so the next
rebuild re-downloads the ~2.4 GB torch wheel. Budget 15-25 minutes.

**Run `scripts/check-diffusers-api.py` before trusting an image.** It is offline
and instant, and it answers "can this container run the plan at all" without
waiting for a 10 GB download to finish first.

### Do not run a model download and an image build at the same time

Measured directly. With the 9.92 GB GGUF fetch and the `sprite-worker` rebuild
running together on this connection, pip crawled at **24-40 kB/s** and the GGUF
stopped advancing altogether for four minutes. Stopping the fetch took pip to
**5-7 MB/s** immediately - two orders of magnitude, from one change.

This is worse than ordinary bandwidth sharing, and it matters because both jobs
LOOK healthy while it happens: the build log keeps printing and the `.incomplete`
blob still exists, so the natural reading is "slow connection" rather than "these
two are fighting". Sequence them.

Interrupting a download is cheap, but **not because it resumes** - it does not.
A killed fetch leaves its `.incomplete` blob behind, and the next run starts a
new one under a different temp suffix rather than continuing it. Measured here:
a 1.0 GB partial was abandoned, the file re-downloaded from zero, and a stale
3.68 GB orphan was left in `blobs/` next to the finished 9.92 GB file. Sweep them:

```bash
find "$MODELS_DIR" -name '*.incomplete' -delete   # as root; containers write these
```

The cost of interrupting is therefore the whole partial file, not the current
chunk. Still worth it here - re-downloading 1 GB at 5 MB/s beat sharing the pipe
at 30 kB/s - but decide with that number, not the wrong one.

### The API confirms the batch split is possible, with one correction

`scripts/inspect-qwen-api.py` against diffusers 0.39.0:

    encode_prompt(prompt, image, device, num_images_per_prompt,
                  prompt_embeds, prompt_embeds_mask, max_sequence_length)
    __call__(..., prompt_embeds, prompt_embeds_mask, ...)
    model_cpu_offload_seq = text_encoder->transformer->vae

`__call__` accepts pre-computed embeddings, so the encode-then-denoise split is
supported. **But `encode_prompt` takes an image**: the text encoder is a
vision-language model, so an embedding is a function of the (prompt, image)
pair. The 8 directions of one turnaround share a concept image and can be
encoded together; a stage whose input is the previous stage's OUTPUT cannot be.

So the three-pass structure above is **per-stage, not global**. The claim that
one encode pass covers "EVERY prompt for the sheet" is wrong and is corrected
here.

### Host RAM is tighter than the config claims

`~/.wslconfig` asks for `memory=11GB`; the distro reports **10 GiB total, ~7 GiB
available**. Against ~15 GiB of component residency under
`enable_model_cpu_offload`, this swaps. It runs, slowly. `qwen_edit.py` warns
rather than refusing, because the usual failure is not an exception - it is the
kernel OOM-killing the worker with no traceback, which reads like the container
crashing for no reason.

## RESULT (2026-08-23): the route works. 8 directions, one character, 2D.

`images/_turnaround8.png` (raw, 4096x512) and `images/_sheet8.png` (finished,
384x64). One concept image in, eight directions out, **including genuine back
views with no face** - the exact thing ADR 0003 measured as unreachable and
ADR 0004 built the TripoSR -> UniRig conveyor to obtain. It cost a 295 MB LoRA.

Acceptance check on the finished sheet:

| | |
|---|---|
| size | 384x64, 8 cells of 48x64 |
| transparent | 69.3% |
| partial alpha | **0 px** |
| colours | **24** |
| baseline | row 62 on all 8 cells |
| verdict | PASS |

### The working configuration - do not change one part without re-measuring

| | |
|---|---|
| transformer | `qwen-image-edit-2511-Q2_K.gguf` (7.47 GB) |
| placement | **resident on the GPU**, `pipe.to("cuda")` - NOT offloaded |
| text encoder | NF4 from `ovedrive/...-4bit`, loaded and freed in its own pass |
| LoRAs | angles @ 1.0 + Lightning 4-step @ 0.7 |
| steps | 4, `true_cfg_scale=1.0` |
| size | 512, passed **explicitly** as height/width |
| decode | after the transformer is released |

Measured: **7.9 s/step, ~33 s/cell**, VRAM 2.7/12.0 GiB free while resident.
A 150-cell character is therefore **~80-90 minutes**, not the 30-50 estimated
earlier in this document. That estimate was optimistic and is corrected here.

### The counter-intuitive finding: do not offload on this machine

Every guide says to use `enable_model_cpu_offload` on a 12 GB card. Here it is
**wrong**, and it is wrong for a reason specific to this box: free VRAM
(11.7 GiB) exceeds available host RAM (~9 GiB). Offload parks idle components
in the scarcer resource, and the bill arrives at the END of the forward pass
when accelerate moves the 7.1 GB transformer back to RAM. Measured: denoising
completed in 64 s and the container then died with `unexpected EOF` - the OOM
killer - before one line of the decode ran.

Keeping the model on the card instead: same work, 31 s, no host RAM touched.

**The general rule for this hardware: compare free VRAM against available host
RAM before reaching for offload.** On most machines RAM is plentiful and the
advice holds. On this one it is inverted.

### Four failures on the way, and what each actually was

| Symptom | Real cause |
|---|---|
| Exit 1, no traceback, whole WSL VM down | Host RAM. Text encoder + transformer in one pipeline needs ~14 GiB. Fixed by the encode/denoise split |
| `AttributeError: 'NoneType' has no attribute 'detach'` | `encode_prompt` returns `prompt_embeds_mask=None` when the mask is all-ones. It is the MASK, not the embeddings - both `.detach()` calls sit on one line |
| `CUDA driver error: device not ready` | bitsandbytes out-of-VRAM in disguise. Its fast 4-bit kernel needs a 64-aligned inner dimension; unaligned, it DEQUANTISES each weight to bf16 and discards the whole NF4 saving. Fixed by resizing the image before encoding |
| `RuntimeError: shape '[1,12,12,16,2,2]' is invalid for input of size 262144` | The pipeline silently ignored `size` and rendered at its own ~1-megapixel default. Fixed by passing height/width explicitly |

Two things that did NOT help and were reverted:

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` broke a stage that had been
  passing ("memory mapping failed with OOM" under WSL).
- Raising `~/.wslconfig` `memory` to 12GB left Windows under 4 GB and got the VM
  killed repeatedly. Reverted to 11GB. The split made the extra GB unnecessary -
  **fixing the algorithm beat buying a gigabyte.**

And one hard incompatibility: **`enable_sequential_cpu_offload` cannot be used
with a GGUF transformer.** accelerate moves parameters to the "meta" device and
a `GGUFParameter` loses its `quant_type`, giving a bare `KeyError: None` that
names neither offload nor GGUF. `qwen_edit.denoise_only` now refuses it with an
explanation instead.

### Why not a bigger quant, and why more RAM/swap does not buy one

Asked and measured 2026-08-23. Q3_K_M (9.92 GB) was retried with the working
GPU-resident configuration. It **never finished loading**: the log froze at
"Stage 2/2: loading transformer" for 41 minutes and had to be killed.

The host RAM ceiling binds at model LOAD, not at inference:

| quant | host RAM to load | result |
|---|---|---|
| Q2_K (7.47 GB) | fits in ~9 GiB available | loads in ~90 s, then 33 s/cell |
| Q3_K_M (9.92 GB) | needs ~10 GiB | thrashes indefinitely, never loads |

So "we are no longer RAM-bound" - stated earlier in this document - is true of
the working configuration and **false of any larger quant**. Corrected here.

Three levers were considered and all are refused:

- **Give WSL more RAM (leave Windows ~1.5 GB).** Windows hosts the hypervisor,
  `dxgkrnl`, and the WDDM driver that manages the 12 GB of VRAM. Starving it does
  not enlarge the guest's budget, it removes the supervisor. Measured: 12GB
  (~3.9 GB left for Windows) already got the VM killed repeatedly.
- **Grow swap to 20-25 GB on D:.** `swapFile=` can point at a Windows path with
  no 9p penalty, so the config is legal. But swap converts an OOM kill into an
  indefinite stall, which is exactly what the 41-minute freeze above IS. Paging
  a 10 GB model per step is not a throughput fix.
- **Accept Q3_K_M with offload.** Dies on host RAM at the end of the forward
  pass instead. Already measured.

And the gain would not be visible anyway: cells are 48x64 against a 24-colour
locked palette, so the pixelation stage discards far more detail than the quant
difference introduces. Same argument ADR 0004 made correctly about TripoSR's
textures.

**Spend effort on the ground patches and the azimuth mapping instead** - those
are defects visible in the output; the quant is not.

### Known-imperfect, and honestly so

- **Ground shadow patches: 6 of 8 directions fixed, front and back not.**
  See "Ground patches" below.
- **Keying tolerance is sharp-edged.** The model renders on black and the
  character wears dark denim, so at tolerance 40 the flood fill leaked through
  the clothing and hollowed the sprite out. 10 works. Generating on a chroma-key
  background would remove the whole class of problem.
- **Cell 1 is framed larger** than the other seven; the shared-scale logic in
  `sheet.py` handles this when composing a real sheet, but the underlying
  generation is inconsistent.
### Azimuth mapping: checked, broken, fixed (2026-08-23)

Verified with `scripts/label-turnaround.py`, which reflows the 4096px strip into
a labelled 4x2 grid - at strip scale the cells are thumbnails and the question
"which way is this character facing" cannot be answered at all.

The first pass was **wrong in exactly two of eight cells**, and they were the two
that matter most for a 2D game: `right side` and `left side` both returned a
near-frontal figure with the face fully visible, where a profile was needed. The
front/back axis was already perfect.

Three hypotheses were tested and rejected before the cause was found:

| Hypothesis | Test | Result |
|---|---|---|
| Lightning LoRA diluting the angles LoRA | `--no-lightning --steps 20` | still frontal |
| Elevated camera flattening the rotation | `--elevation eye` | still frontal |
| Q2_K too coarse to steer the LoRA | implied by the fix below | **disproved** |

**The cause was the prompt, not the model.** The LoRA card says to use its
trigger tokens alone; that is true for front/back and false at 90 degrees.
Appending plain language - "strict profile view, the character faces exactly
sideways, only one eye visible, nose in silhouette" - produces clean, correct,
opposite-facing profiles at Q2_K.

This also settles the hardware question above for good: **the model can render
profiles on this card at the smallest quant.** No amount of extra RAM, swap or
quantisation headroom would have fixed it, because nothing was short.

Implemented as `AZIMUTH_HINTS` in `qwen_edit.py`, keyed per azimuth. **Do not
apply it globally.** Telling the FRONT view that the character faces sideways is
a contradiction, and the model resolves contradictions by picking one at random
per seed - which would trade two broken cells for eight unstable ones.

All eight directions now verify correct, and the finished sheet passes:
70.6% transparent, 0 partial-alpha px, 24 colours, baseline row 62 on every cell.

## Ground patches (2026-08-23): 6 of 8 fixed, and why the other two are not

The concept image has **no** ground patch - the model invents one. That is not
a defect in the concept: `fal/...-Multiple-Angles-LoRA` was trained on
Gaussian-Splatting renders of real captured scenes, which always have a floor,
and it carries that prior into every direction.

### Suppressing it by prompt was tried and abandoned

It works, but it costs more than it is worth. Extra prompt tokens lengthen the
sequence, and stage 1's bitsandbytes dequant fallback allocates against that
length - the run OOM'd on a 1.02 GiB buffer. Two real fixes came out of chasing
it, and both are keepers:

- **`QWEN_ENCODE_SIZE` (default 384), decoupled from render size.** The text
  encoder is a VL model, so vision tokens dominate the sequence. Encoding
  smaller shortens it; the image is still fed to stage 2 at full render size,
  and that is what the edit is applied to. Dropped the fallback allocation from
  1.02 GiB to 120 MiB.
- **`torch.no_grad()` around `encode_prompt`.** This one is a genuine bug and it
  does not look like one. `pipe.__call__` carries `@torch.no_grad()`;
  `encode_prompt` called directly does not, so every dequantised bf16 weight the
  fallback materialises is retained for a backward pass that never comes. A 5 GB
  NF4 encoder sat at **8.77 GiB**. Anyone calling a pipeline's sub-methods
  directly needs this.

### The geometric fix, and its limit

`strip_ground_patch` moved from `tasks.py` to `pixelate.py` (pure PIL/numpy, so
every stage can use it without importing celery/torch) and is exposed as
`--strip-ground` / `--ground-ratio`.

It detects ground by WIDTH, not colour - feet are narrow, the thing they stand
on spreads out. **That works for 6 of 8 directions and cannot work for the other
two**, and the reason is structural rather than a tuning miss:

| ratio | turned views (se, e, ne, nw, w, sw) | front + back |
|---|---|---|
| 1.8 (default) | clean | patch remains |
| 1.4 | clean | **legs amputated** |

Front and back present the body at its widest across the shoulders, which
inflates the reference width and pushes the patch ratio under the cut. Lowering
the threshold to catch it then lets the hysteresis walk run up through the
boots - exactly the failure the original docstring predicted, now confirmed on
these two views.

**Default stays 1.8.** A remaining patch is cosmetic; amputated legs are not.

Closing the last two properly needs a detector that is not width-based - the
honest options are a per-direction rule, or keying against a chroma background
generated for the purpose rather than the black the model chooses. Not attempted
here.

## Actions (2026-08-23): plain instructions work; AnyPose not needed

`images/_walk_px.png` - a 4-frame walk cycle, 192x64, PASS: 23 colours, 0
partial-alpha px, baseline row 62 on all four, cell-height spread **1px**.

### AnyPose was researched and NOT used

`lilylilith/AnyPose` was the plan. Reading its card first was worth it:

- it needs **two** input images (character + a pose reference we do not have)
- it wants **both** its 295 MB adapters at 0.7, stacked on the two already loaded
- its trigger is a long sentence, and prompt length is VRAM-bound in stage 1 here
- its own card warns it "struggles with 2D art styles" - i.e. exactly our case

Qwen-Image-Edit-2511 is already an instruction-following editor, so the free
test came first: just ask for the pose. It works. No download, no extra
adapters, no pose references.

### The real finding: camera and pose compete, and the LoRA weight is the dial

Three runs, one variable each:

| camera prompt | angles LoRA | pose | framing |
|---|---|---|---|
| omitted | off | **strong, correct** | destroyed - bigger, eye-level, scale differs per frame |
| carried through | 1.0 | **flattened** - 4 near-identical standing poses | perfect |
| carried through | **0.4** | **correct walk cycle** | **preserved** |

Two things fall out of this, and neither was obvious:

1. **The angle prompt is not just for turnarounds - it is what pins framing.**
   An action prompt REPLACES it unless carried through explicitly, and the model
   then reverts to its own default framing and camera height.
2. **The angles LoRA pins POSE as well as framing.** At 1.0 that is exactly
   right for a turnaround, where the body must not move between directions, and
   exactly wrong for an action. `angles_scale` is the dial; 0.4 is the measured
   balance. Default in `action_frames` is 0.5.

### Write frame prompts as POSES, not as verbs

"walking" is a process, and the model has to guess which instant to draw - which
is how four frames of a walk come back as four frames of standing. Naming limb
positions ("left leg forward, right arm forward, mid stride") makes each frame a
different, checkable target. See `ACTIONS` in `scripts/action-test.py`.

### Enclosed background: the second keying pass

A wide stance traps a wedge of background BETWEEN THE LEGS. It touches no
border, so the corner flood fill structurally cannot reach it, and it survives
as a solid black slab through the middle of the sprite - visible on walk frame 3
before the fix.

Clearing every near-background pixel instead is simpler and wrong: this
character's outline and darkest shading sit close to black, and a global
threshold punches holes through them. `key_background` now removes whole
CONNECTED REGIONS above `min_enclosed` (default 200 px) - a trapped wedge is
large and contiguous, shading is small and scattered.

### Still open

- **Ground patches on action frames.** `strip_ground_patch` is deliberately NOT
  applied here: a wide stance genuinely is wider at the bottom, and clipping it
  is worse than the patch.
- Only `walk` is verified. `attack` and `idle` are written in
  `scripts/action-test.py` and untested.
- Frames are not yet composed into a full sheet via `sheet.py` - that wiring
  exists and is tested (`make smoke-sheet`) but has not been run on real action
  output.

## Full sheet (2026-08-23): end to end, and the two weak directions

`images/_char_walk.png` - 192x256, 4 directions x walk x 4 frames. PASS:
24 colours, 0 partial-alpha px, baseline row 62 on all 16 cells, atlas declares
`columns 4 / rows 4` for something2.

**Side rows are production quality.** Front and back are not: they barely
animate, and they carry the ground patches. That is the SAME two directions
that defeat `strip_ground_patch`, and the cause is likely shared - front and
back present the least foreshortening, so a stride changes the silhouette least
and the model has the least to work with. Worth knowing before generating 150
cells: **6 of 8 directions are good, 2 need work.**

### Two operational requirements, both learned the hard way

**1. Restart WSL before a generation run.** Not superstition - measured. The
NF4 text encoder's dequant fallback needs a **1.02 GiB contiguous** allocation,
and WSL's GPU memory manager could not provide it despite 5.8 GiB free:

    memory allocation failed with OOM on device 0 while trying to allocate
    1090519040 bytes (free: 6223298560, total: 12884246528)

Four things were tried and did NOT fix it: shortening the prompt (the buffer is
sized from the WEIGHT shape, not the sequence - the reported
`inner dimension (3420)` is identical either way), `max_split_size_mb:256`,
`expandable_segments:True` (made it worse), and explicitly freeing latents and
pipelines between passes. `wsl --shutdown` fixed it on the first try.

**2. Stop `llm_engine` and friends first.** llama.cpp holds the same 12 GB card,
and `stats_collector` polls it, which can trigger a model load mid-run.
`make up` starts them; a generation run should not have them running.

### View-aware poses: helps the back row, barely helps the front

`src/sprite_generator/actions.py` now keys frame poses to VIEW FAMILY, because
the first sheet used one prompt set for all directions and "left leg forward,
right arm forward, mid stride" is a **sagittal** description - it reads in
profile and is foreshortened to nothing head-on. A walk seen from the front is
not a stride, it is a knee lift, which is also what a hand-drawn RPG-Maker
front-walk row shows.

Result, measured: the BACK row gained visible arm and silhouette variation. The
FRONT row improved only slightly. Profile rows remain clearly the best.

So this was the right diagnosis but only a partial fix, and the honest summary
stands: **6 of 8 directions are good, front and back are weaker.** Remaining
ideas, none tried: lower `angles_scale` further for front/back only; generate
front/back at more steps; or accept it, since a side-facing sprite is what most
2D games use for diagonal movement anyway (see `action_prompts.json`).

### Why build-sheet.py is FIVE processes, not one function

Loading and releasing a large quantised model fragments the CUDA allocator under
WSL past the point of reuse. Splitting turnaround from actions was **not
enough**: encode and denoise each load their own model, so a second load inside
one stage still failed - on a **30 MiB** allocation with **2.23 GiB free**.

Every stage that loads a model now gets its own process, with embeddings
persisted between them as `.pt` files:

    build-sheet.py turnaround-encode <concept.png> --directions s,e,n,w
    build-sheet.py turnaround-denoise
    build-sheet.py actions-encode --actions walk --frames 4
    build-sheet.py actions-denoise
    build-sheet.py compose <out.png> --cell 48x64

`scripts/build-sheet.sh` drives all five and stops the GPU-contending services
first. Explicit `del` of latents, VAE and pipeline improved the failure
(72 MiB -> 2 MiB) but never removed it - the arena itself is what fragments. A
fresh process gets a fresh allocator, and the model was loading twice either
way, so the only cost is ~10 s of torch import per stage.

The split also makes the run resumable and each stage inspectable on disk, which
matters when a full character is an hour of GPU time and a crash in `compose`
should not discard the generation.

Measured: 4 directions + 16 action cells = **~20 minutes** including both model
loads. Extrapolating to 8 directions x 4 actions x 6 frames (192 cells) gives
roughly **2 hours** - more than the 80-90 minutes estimated earlier, because
that estimate ignored the turnaround pass and the per-stage model loads.

## The job API (2026-08-24): something2 polls after all

`src/sprite_generator/jobs.py`, table in `migrations/012_jobs.sql`, verified by
`scripts/verify-jobs-api.py --submit` (12 checks).

**The synchronous-only constraint recorded throughout this ADR was wrong.** It
came from something2's published `docs/ai-providers.md` and their SOMET-334,
read with an explicit caveat that their calling code had never been reviewed -
and the caveat was the accurate part. The project owner states something2
queues the task, does not wait, and polls later, either per-task or for every
task it has sent. All it needs back immediately is a task number.

So the facade is no longer the integration point for sheets. `a1111.py` stays
for single-image txt2img, which fits in one request; sheets go through
`/api/jobs`. See the rewritten `.ai/specs/something2-provider/contract.md`.

Three design points worth keeping:

- **A `jobs` table, not Celery's result backend.** Celery results expire on a
  TTL and do not survive a broker flush. "Hand me an id now, ask about it
  whenever" needs the id to still resolve tomorrow.
- **`sheet_url` appears only when done**, so a client can branch on its presence
  instead of string-matching status.
- **Asking for the sheet early returns 409, not 404.** A client that conflates
  those abandons live jobs.

The Celery task **shells out to each build stage as a subprocess** rather than
importing them. That is the allocator-fragmentation fix again: a worker that
loaded models in-process would reintroduce exactly the failure the five-stage
split exists to avoid. The task holds no CUDA context of its own.

## First full character (2026-08-24): 96 cells, one hour, unattended

`sheet_daad88a1-*.png` - 192x1536, 8 directions x (walk, attack, idle) x 4
frames, submitted through the job API and left alone. PASS: 24 colours, 0
partial-alpha px, **baseline row 62 on all 96 cells**.

Measured cost: **~59 minutes for 96 cells + 8 turnaround**, matching the
33 s/cell estimate almost exactly. The five-stage split held up over a full run
with no OOM.

### Per-action results, honestly

| action | result |
|---|---|
| **walk** | Good in **all 8 directions.** The view-aware knee-lift phrasing works - the front, back and quarter rows animate visibly, which the earlier 4-direction test had not shown |
| **idle** | Correct everywhere. Subtle shoulder variation, character stable |
| **attack** | **Excellent in the two profiles, broken in the other six** - see below |

So the "6 of 8 directions" caveat recorded earlier is now **wrong for walk**:
with per-family poses, all eight are usable. It was a prompt problem, not a
model limitation, and it took a full 8-direction run to see that - the
4-direction test was too small a sample to distinguish "front views are weak"
from "my front-view prompt was weak".

### The T-pose: the same trap, twice

`attack` frame 3 asked for "both fists thrust toward the viewer". A forward
punch head-on is pure foreshortening - there is nothing for the camera to see -
so the model drew the nearest thing it CAN draw: **arms straight out sideways.
A literal T-pose, in all six non-profile directions.**

This is the identical mistake as writing a walk as a stride, made again in a
different action, and it generalises into a rule:

> **A frame pose must change the SILHOUETTE as seen from that camera.** Motion
> along the view axis does not exist in the output. Profile cameras see
> sagittal motion (strides, thrusts); front and back cameras see vertical and
> lateral motion (knee lifts, overhead swings, arms going wide).

`attack` front/back is now an **overhead swing**, which moves the arms
vertically and therefore reads from any angle. Re-run and verified
(`sheet_499c921c-*.png`, 192x512, PASS): the T-pose is gone, all six
non-profile rows show arms-up -> coming-down -> hunched follow-through ->
return, and the two profile rows are unchanged.

**All three actions are now usable in all eight directions.**

## Two self-inflicted regressions from this session

Recorded because both were introduced by a change that looked purely
housekeeping.

**`HF_HUB_OFFLINE=1` broke generation.** Every weight was cached, so switching
the worker offline looked free. It was not: `from_single_file` loads the
transformer's architecture config through its own path and **does not forward
`cache_dir`**. Online that is invisible - the config lands in the container's
default `~/.cache/huggingface` and dies with the container. Offline it fails
with `Qwen/Qwen-Image-Edit-2511 does not appear to have a file named
config.json` while that file is demonstrably in `/models`.

Two changes were needed, not one:
- fetch `Qwen/Qwen-Image-Edit-2511` with `allow_patterns=["transformer/config.json",
  "model_index.json"]` - a few KB, no weights;
- set **`HF_HUB_CACHE=/models`**, so the whole hub stack uses the mounted cache
  rather than only the calls that happen to pass `cache_dir=`.

**The fetch script re-downloaded a quant that had just been deleted.** Its
default was `Q3_K_M` while `qwen_edit.GGUF_FILE` was `Q2_K`. A routine re-run
silently began pulling back the 9.9 GB quant that had been removed for not
fitting the card. **These two defaults must agree**; they are now both Q2_K.

### A warning that is not a bug

The denoise stage warns `prompt_embeds_mask is not provided`. Checked against
the installed source rather than "fixed":

```python
if prompt_embeds_mask.all():
    prompt_embeds_mask = None
```

diffusers nulls an all-ones mask itself, then warns it is missing. Prompts are
encoded one at a time, so there is never padding, so the mask is always
all-ones. Nothing to do.

## Sources

- <https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA>
- <https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF>
- <https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning>
- <https://huggingface.co/lilylilith/AnyPose>
- <https://huggingface.co/Tongyi-MAI/Z-Image-Turbo>
- <https://huggingface.co/black-forest-labs/FLUX.2-klein-4B>
- <https://huggingface.co/briaai/RMBG-2.0>
- <https://huggingface.co/docs/diffusers/main/en/quantization/gguf>
