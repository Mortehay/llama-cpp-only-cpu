# 0002 - Image model selection: which checkpoints, and what constrains the choice

Date: 2026-08-20
Status: Accepted

## Context

SDXL-Turbo (step 1) and `Onodofthenorth/SD_PixelArt_SpriteSheet_Generator`
(step 2) were judged to produce weak sprites. The question raised was whether to
replace them, and with what.

Inspecting the actual output first, three separate causes were visible and only
one of them is the model:

| Cause | Evidence | Fixed by a new checkpoint? |
|---|---|---|
| No pixel grid or palette enforcement anywhere in the pipeline | `images/core_891f3ae55287.png` is a continuous-tone illustration *of* pixel art: anti-aliased outlines, unbounded palette, no consistent grid | No |
| Pipeline defects | `images/sheet_e2516f7c1ad5.png` - two of four strips keep a grey background rectangle (`remove_background` failed); sprite fills ~25% of the canvas in `core_e6857fcaeffb.png`, so effective sprite resolution is ~60px before downscale | No |
| Model ceiling | SD1.5 finetune from 2022; SDXL-Turbo clamped to 4 steps / guidance 0 by `resolve_sampling_params` | Yes |

Recorded because a future "the sprites look bad, swap the model" impulse should
start from this table rather than repeat the diagnosis.

## Decisions

### 1. Step 1 and step 2 are independent model choices

Step 2 hands the core to img2img and re-renders it, so the **core does not need
to be a pixel-art model** - it needs to produce one clean, well-composed,
centred character. Step 2 supplies the pixel style.

### 2. Step 2 is locked to the SD1.5 family

`poses.py` authors COCO-18 skeletons against `lllyasviel/control_v11p_sd15_openpose`,
and `get_sd_pipeline` explicitly refuses ControlNet on SDXL (the openpose
checkpoint is trained against SD1.5's UNet). Pose conditioning has been run and
works. Moving step 2 to SDXL or FLUX discards that work and requires re-tuning
`poses.py`'s `Y` proportions against a different ControlNet.

Step 1 is unconstrained.

### 3. Prefer non-distilled checkpoints for step 1

SDXL-Turbo runs at guidance 0, which switches classifier-free guidance off,
which makes every negative prompt a no-op. That is the documented root cause of
the duplicate-character bug that `_isolate_largest_sprite` exists to work around
(see README, step 2 section). **Any non-distilled checkpoint restores negative
prompts**, and that is the single largest model-side win available - larger than
the difference between any two pixel-art finetunes.

The latency objection is weaker than it looks: the A1111 facade calls
`generate_raw_task`, a *single* txt2img, with a 240s budget. The multi-action
`generate_spritesheet_task` is not exposed through the facade at all and runs
async through the browser UI with no time cap. 30 steps of SDXL fits 240s
comfortably.

### 4. Rejected: FLUX.1-schnell

Proposed and rejected. schnell is distilled - guidance 0, negative prompts dead -
so it repeats the exact trap in decision 3 at 6x the VRAM cost. It is 12B, so
`enable_model_cpu_offload` streams through system RAM against an 11GB WSL cap.
And it is not SD1.5, so it cannot drive step 2's ControlNet. `FLUX.1-dev` is the
coherent FLUX choice (non-distilled, real CFG) but is gated, slower, and
non-commercially licensed.

### 5. Rejected: LLM-emitted pixel grids

`huggingface.co/blog/AINovice2005/pixel-art-bench` was raised as a model source.
It benchmarks small *text* LLMs (Phi-3.5-mini, Qwen3-1.7B, Llama, SmolLM)
emitting a palette plus a 24x24 index grid as JSON - no diffusion involved. The
idea is appealing because a palette-indexed grid is exact pixel art by
construction, which is the thing diffusion cannot do. The benchmark's own
numbers rule it out for now: best model ~0.42 pixel-art quality, ~0.74 render
success, at 24x24.

## What actually drops in without pipeline changes

Verified against the HuggingFace API (file listings, not model cards):

| Repo | Family | Verdict |
|---|---|---|
| `John6666/super-pixelart-xl-m-v1-v10-sdxl` | SDXL | Loads. Full diffusers layout, `text_encoder_2/`, safetensors |
| `PublicPrompts/All-In-One-Pixel-Model` | SD1.5 | Loads. DreamBooth sprite model, trigger `pixelsprite`. ControlNet-compatible |
| `kohbanye/pixel-art-style` | SD1.5 | Loads. Trigger `pixelartstyle`. ControlNet-compatible |
| `stablediffusionapi/pixel-art-diffusion-xl` | SDXL | **Will not load** - name trap, see below |
| `pixelparty/pixel-party-xl` | SDXL UNet | **Will not load** - bare UNet, no `model_index.json`; must be grafted onto an SDXL base pipeline |
| `nerijs/pixel-art-xl` | SDXL LoRA | **Will not load** - `get_sd_pipeline` never calls `load_lora_weights` |

Neither working candidate publishes an fp16 variant. `get_sd_pipeline` already
falls back to default weights, at the cost of a slower first load and higher
peak host RAM - which matters against the 11GB WSL cap.

## Consequences and traps

### The `is_sdxl` name heuristic gates what is usable

```python
is_sdxl = "sdxl" in llm_name.lower() or "turbo" in llm_name.lower()
```

The pipeline class is chosen from the **repo name string**, not from
`model_index.json`. Any SDXL repo whose name lacks `sdxl`/`turbo` is handed an
SD1.5 pipeline class and fails. This catches `pixel-art-diffusion-xl` (has `xl`,
not `sdxl`) and would catch `stabilityai/stable-diffusion-xl-base-1.0`.

Only `sdxl`- or `turbo`-named repos are drop-in. Reading `_class_name` from
`model_index.json` instead would remove the constraint.

### Adding a model touches three hardcoded lists

There is no model-discovery endpoint. A new checkpoint must be added to:

- `templates/index.html` - the step 1 `#core-llm` dropdown
- `templates/index.html` - the step 2 `#sheet-llm` dropdown
- `a1111.py` `KNOWN_MODELS` - what something2 sees at `/sdapi/v1/sd-models`

Data-only edits, but all three or the model is unreachable from that entry point.

### Trigger words are currently attached to the wrong models

`generate_core_task` hardcodes `PixelartFSS` into the step 1 prompt. That is
*Onodofthenorth's SD1.5 trigger*, and step 1 runs SDXL-Turbo, where it is a
meaningless token. Step 2 does the reverse: it **strips** `PixelartFSS` and runs
Onodofthenorth without its own trigger, so that finetune is only weakly
activated.

Each candidate has its own trigger (`pixelsprite`, `pixelartstyle`). Benchmarking
checkpoints without moving the trigger with the model tests them with their
finetune barely engaged, and the comparison is not meaningful. **Unfixed** - the
model swap was requested without code changes.

## Measured results (2026-08-21)

Four models, identical subject and seed (112233), each given **its own trigger
word**, through `POST /sdapi/v1/txt2img`. That route matters: `generate_raw_task`
does no prompt rewriting, so each checkpoint could be tested with its correct
trigger without editing `tasks.py`.

| Model | Prompt prefix | Result | Gen |
|---|---|---|---|
| `stabilityai/sdxl-turbo` | - | One character, but continuous-tone: anti-aliased outlines, soft shading, no pixel grid. Ground shadow present. | 26.6s |
| `Onodofthenorth/SD_PixelArt_SpriteSheet_Generator` | `PixelartFSS` | **Four characters in a row** on grey | 3.9s |
| `John6666/super-pixelart-xl-m-v1-v10-sdxl` | `pixel art sprite` | **Pure RGB noise** | 19.8s |
| `PublicPrompts/All-In-One-Pixel-Model` | `pixelsprite` | **Real pixel art**: hard grid, bounded palette, one character | **3.0s** |

Images: `images/raw_86069c1d91fc.png`, `raw_196cfdd00816.png`,
`raw_0dd2e0ac2cf9.png`, `raw_1ddcbc59ddf7.png` respectively.

### Decision: `PublicPrompts/All-In-One-Pixel-Model` for both steps

The only candidate that produced genuine pixel art, and it is SD1.5, so it is
also ControlNet-compatible and satisfies the step 2 constraint in decision 2.
One model can serve both steps with pose conditioning intact. Trigger word is
`pixelsprite` (sprites) / `16bitscene` (scenes).

### `super-pixelart-xl` is not usable and is no longer served

It loads without error and returns undenoised latents. This is **not** the fp16
SDXL VAE failure the README documents - that produces solid black, not noise.
Most likely a scheduler or prediction-type mismatch against this diffusers
version. Removed from `KNOWN_MODELS` rather than left to fail silently with a
200 to something2. Recommending it on HuggingFace file listings alone was wrong;
only the bench caught it.

### Correction: guidance 0 is not the whole duplicate-character story

Decision 0001 and the README attribute duplicated characters in step 1 to
SDXL-Turbo running at guidance 0, which makes negative prompts inert, and
`_isolate_largest_sprite` exists as a geometry workaround for it.

Measured here: Onodofthenorth at **cfg 7.5** - classifier-free guidance fully on
- with `multiple characters, group, crowd, twins, clones` in the negative prompt
still returned four characters in a row.

`PixelartFSS` expands to **"Front Sprite Sheet"**. The trigger *is* a request for
a sheet, and no guidance setting overrides what the finetune was trained to do.
So `generate_core_task` prepending `PixelartFSS` - in a step whose entire job is
to produce exactly one character - is not merely inert on SDXL as previously
recorded. On any SD1.5 pixel checkpoint it actively asks for the defect that
`_isolate_largest_sprite` then removes.

Unfixed; the model swap was requested without code changes. It is a one-string
change in `generate_core_task` and it is the highest-value one left.

### Latency is a non-issue

3.0s and 3.9s for the SD1.5 models at 20 steps, 19.8s for SDXL at 1024px/25
steps, against a 240s façade budget. Non-distilled sampling costs nothing that
matters here. The 26.6s for SDXL-Turbo was first-load warmup, not sampling.

## Superseded 2026-08-21: step 2 is no longer locked to SD1.5

Decision 2 above said step 2 could not leave the SD1.5 family, because
`poses.py` targets `control_v11p_sd15_openpose` and `get_sd_pipeline` refused
ControlNet on SDXL. That was a property of the code, not a law, and it has been
changed.

### What changed

**Family detection reads the config, not the repo name.** `_is_sdxl_checkpoint`
calls `DiffusionPipeline.load_config` and tests `_class_name` for "XL". The old
rule - `"sdxl" in name or "turbo" in name` - was a guess about naming that
excluded `stabilityai/stable-diffusion-xl-base-1.0` outright, which is the very
model this decision now wants. The name heuristic survives only as a fallback
for an unreadable config (offline, cold cache).

**ControlNet is now family-matched rather than refused.** A ControlNet's blocks
carry the dimensions of the UNet it was trained against, so the SD1.5 openpose
checkpoint genuinely cannot condition SDXL. Instead of declining, callers still
pass `OPENPOSE_CONTROLNET` and `get_sd_pipeline` substitutes
`thibaud/controlnet-openpose-sdxl-1.0` when the target is SDXL. No call site has
to know which family it is driving, so `_patch.py` is untouched.

`StableDiffusionXLControlNetImg2ImgPipeline` handles the SDXL + ControlNet +
img2img combination.

**poses.py needs no change.** Both openpose checkpoints consume the same COCO-18
skeleton format, which is what `render_skeleton` emits. Its `Y` proportions may
still want retuning if SDXL draws a different body plan than SD1.5 did - that is
tuning, not a structural blocker.

### The new constraint is VRAM, not architecture

SDXL UNet (~5GB fp16) + SDXL ControlNet (~2.5GB) + VAE and two text encoders is
roughly **9.5GB against ~11.7GB free**. That is the tightest combination this
service loads; the SD1.5 + ControlNet path measured 3.43GB reserved.

`pipe.to(DEVICE)` is therefore wrapped: on `torch.cuda.OutOfMemoryError` it
retries with `enable_model_cpu_offload()`, which streams weights through system
RAM. That works but is much slower, and the WSL memory cap (11GB) becomes the
real ceiling for that path rather than VRAM - the same trap decision 0001
recorded for FLUX.

### Status

Code loads cleanly - `sprite_worker` reports `celery@... ready` with no import
error, and the API advertises `stabilityai/stable-diffusion-xl-base-1.0`.
**Not yet exercised**: no image has been generated through the SDXL ControlNet
path, and the ~9.5GB VRAM estimate is arithmetic, not a measurement. The OOM
fallback has never fired.

## 2026-08-21: SDXL ControlNet works, and POSED_STRENGTH was disabling it

### The SDXL path is verified end to end

First pose-conditioned sheet on `stabilityai/stable-diffusion-xl-base-1.0`
succeeded (`images/sheet_7cff0e8420cf.png`). Every piece of the new code fired:

```
Pose cycles found; loading ControlNet for per-frame pose.
'stabilityai/stable-diffusion-xl-base-1.0' is SDXL; substituting
    'thibaud/controlnet-openpose-sdxl-1.0' for the SD1.5 openpose checkpoint.
'move right': pose-conditioned, strength 0.6
```

`_patch.py` passes the SD1.5 constant and never learns it was swapped, which is
the point of doing the substitution inside `get_sd_pipeline`.

**VRAM: 11006 MiB of 12288 (89.6%)** with SDXL + ControlNet resident. The earlier
~9.5GB estimate in this document was **too low** - real headroom is about 1.2GB.
The `enable_model_cpu_offload` fallback did not fire, so the arrangement fits,
but nothing else can share the card while it is loaded. The ControlNet
checkpoint is 4.7GB on disk.

### The pose looked frozen, and it was not SDXL's fault

The first SDXL sheet held identity perfectly and barely moved - the exact
identity-preserved/pose-frozen trade ControlNet was introduced to break. Running
the identical sheet on `PublicPrompts/All-In-One-Pixel-Model` (SD1.5 +
`control_v11p_sd15_openpose`) reproduced it exactly, which clears
`thibaud/controlnet-openpose-sdxl-1.0` of being the weak link.

Ruled out with evidence, in order:

| Hypothesis | Result |
|---|---|
| SDXL ControlNet is weak | No - SD1.5 fails identically |
| Skeletons not generated | No - 4 frames, 1.69% coverage, max 255 |
| Skeleton frames duplicated | No - pose 0 ankles (0.64, 0.38), pose 2 (0.38, 0.64), a proper mirror |
| Skeleton malformed | No - correct COCO-18 limb colours, clear side-view walk |
| Conditioning scale too low | No - already `1.0` |
| Kwargs not reaching the pipeline | No - built and splatted correctly |

### Cause: POSED_STRENGTH = 0.60

Strength sweep, same core and action, SD1.5, mean absolute inter-frame
difference and drift from frame 0:

| strength | motion | drift | transparent |
|---|---|---|---|
| **0.60** | **9.60** | 7.07 | 55.0% |
| 0.75 | 34.02 | 30.93 | 57.0% |
| 0.85 | 22.08 | 16.93 | 66.9% |
| 0.95 | 24.04 | 18.33 | 70.6% |

At 0.60 the img2img init latent still encodes the front-facing core and
ControlNet cannot redirect it. `_patch.py` originally shipped `0.75`; lowering
it to `0.60` turned the skeletons, the COCO-18 authoring and the whole SDXL
ControlNet path into no-ops while every log line still said
"pose-conditioned".

**Treat 0.60 as wrong rather than 0.75 as proven.** The 0.75/0.85/0.95 ordering
is non-monotonic, which single-sample measurements should not be - one action,
one core, one seed carries real noise. Only the gap between 0.60 and the rest is
unambiguous.

Note also that the README's existing strength table calls motion 23.2 "four
different characters". That table was measured **without** ControlNet, so it does
not transfer directly - separating pose from identity is the entire claim being
tested - but it is a reason to check identity visually and not only by the metric.

### Confirmed: POSED_STRENGTH = 0.75

The single-sample sweep above was re-run across **three different cores** (three
seeds, three characters) and **two actions** with different pose cycles, at both
strengths. Mean absolute inter-frame difference, SD1.5 +
`control_v11p_sd15_openpose`:

| core | action | motion @0.60 | motion @0.75 | change |
|---|---|---|---|---|
| 22 | move right | 9.60 | 34.02 | 3.5x |
| 18 | move right | 8.82 | 24.28 | 2.8x |
| 22 | attack | 5.29 | 19.87 | 3.8x |
| 20 | move right | 40.72 | 85.54 | 2.1x (see below) |

**Four pairs out of four move the same direction**, including `attack`, which
uses the ATTACK cycle rather than WALK_SIDE - so this is not specific to the walk
skeletons. Excluding the broken core: mean motion **7.90 -> 26.06**.

The non-monotonic 0.75/0.85/0.95 ordering in the first sweep was single-sample
noise, as suspected there. `POSED_STRENGTH` is now **0.75** in `tasks.py`.

### Separate defect: core 20 produces unusable sheets at any strength

`sheet_9154e7fcd74a.png` (core 20, 0.60) carries tall green block "totem"
columns flanking the character in every frame, plus a grey background band that
`remove_background` did not key. Its drift of **89.03** is an artifact of that
junk changing between frames, not of pose motion, which is why it is excluded
from the mean above.

The subject is also rendered as a photographic-looking human rather than a pixel
sprite, so the defect probably starts in the core, not in step 2. Same family as
the grey rectangles in `sheet_e2516f7c1ad5.png`. **Not investigated.**

## 2026-08-21: per-checkpoint trigger words for step 1

`CORE_TRIGGERS` in `tasks.py` maps a repo id to the token that activates its
finetune. `generate_core_task` prepends it and logs which one it used.

Only two entries, and the omission is deliberate:

| checkpoint | trigger |
|---|---|
| `PublicPrompts/All-In-One-Pixel-Model` | `pixelsprite` |
| `kohbanye/pixel-art-style` | `pixelartstyle` |
| `Onodofthenorth/SD_PixelArt_SpriteSheet_Generator` | **none usable** |

Onodofthenorth ships only sheet triggers (FSS/RSS/LSS/BSS - "Front/Right/Left/
Back Sprite Sheet"). `PixelartFSS` measurably returns **four characters in a
row**, and it does so at guidance 7.5 with "multiple characters, group, crowd,
twins, clones" in the negative prompt. Training beats guidance; there is no
prompt-side fix. Step 1 wants exactly one character, so that checkpoint gets no
trigger.

An earlier revision of `generate_core_task` prepended `PixelartFSS`
unconditionally, to every model. That was removed before this change.

### Effect, measured on the 512px core

| | opaque % | bbox-fill % | blobs | unique colours |
|---|---|---|---|---|
| sdxl-turbo (old default) | 32.9 | 52.8 | **16** | 56821 |
| Onodofthenorth (old default) | **6.4** | 71.8 | 1 | 14496 |
| All-In-One + `pixelsprite` | 30.6 | 70.8 | 1 | 37716 |
| All-In-One + `pixelsprite` | 22.8 | 78.9 | 1 | 23979 |

The sprite now fills **3.5-4.8x more of the frame** than the old Onodofthenorth
core, which matters because step 2 downscales to a 128px cell - 6.4% of a 512px
canvas is roughly a 130px figure, so almost nothing survived. One connected
blob, versus 16 for sdxl-turbo.

### Still unsolved: nothing here produces a bounded palette

**24k-38k unique colours.** These outputs have pixel-art *structure* - hard
blocky shapes - with anti-aliased colour inside the blocks. Real pixel art is
16-32 colours on an exact grid. No checkpoint tested produces that, including
the ones that look most convincing, and no trigger or model swap changes it.

That needs a deterministic post-pass: nearest-neighbour downscale to a true
grid, then median-cut or fixed-palette quantisation. Recorded here because
"the sprites still do not look like pixel art" will otherwise keep being
misdiagnosed as a model problem. n=2; one of the two cores also had a
background-removal artifact.

## CORRECTION 2026-08-21: the SDXL openpose ControlNet IS weak

Earlier in this document, under "The pose looked frozen, and it was not SDXL's
fault", the table records `thibaud/controlnet-openpose-sdxl-1.0` as cleared
because SD1.5 reproduced the same frozen pose. **That test was run at
POSED_STRENGTH 0.60, which suppressed pose conditioning on BOTH families**, so it
could not distinguish them. It was a valid observation and an invalid inference.

Re-measured at 0.75, where strength is no longer the limiter:

| config | motion |
|---|---|
| SD1.5 + `control_v11p_sd15_openpose` @0.75 | **34.02** |
| SDXL + pixel-art-xl LoRA + `controlnet-openpose-sdxl-1.0` @0.75 | **9.57** |
| SD1.5 + `control_v11p_sd15_openpose` @0.60 | 9.60 |

The SDXL ControlNet at 0.75 delivers the motion of SD1.5 with conditioning
effectively off. It is **3.6x weaker** than the SD1.5 checkpoint.

### Consequence for the model roster

**The SD1.5 half of the stack cannot be deleted.** Pose conditioning is the whole
reason step 2 exists, and only `control_v11p_sd15_openpose` drives it, which
requires an SD1.5 base. The tempting single-family SDXL-only stack does not work.

Current best split:

| step | model | why |
|---|---|---|
| 1 (core) | `stable-diffusion-xl-base-1.0` **+** `nerijs/pixel-art-xl` | the only config producing structurally real pixel art |
| 2 (frames) | `PublicPrompts/All-In-One-Pixel-Model` + `control_v11p_sd15_openpose` @0.75 | the only config producing real pose change |

The cost of the split is that step 2 re-renders the core in All-In-One's weaker
style, so the SDXL+LoRA quality does not survive into the sheet. Unresolved.

Untested: whether the SDXL ControlNet engages at all at 0.85/0.95, and whether a
higher `controlnet_conditioning_scale` compensates. `CONTROLNET_SCALE` has been
1.0 for every measurement in this document and was never swept.

## 2026-08-21: LoRA support, and the first real pixel art

### The earlier "no model can produce a bounded palette" conclusion was wrong

This document previously concluded that no checkpoint produces genuinely
quantised pixel art and that only a deterministic post-pass could. That was true
of every checkpoint tested **at the time**, and false as a general claim. The
missing option was not a bigger model - it was a **style LoRA on a big base**.

A LoRA is a low-rank delta over the base UNet's attention weights, tens to
hundreds of MB. It is not a model and cannot be judged by file size; it is the
mechanism by which a general 6.7GB base is taught one specific style.

### `<base>+<lora>` in the model name

`get_sd_pipeline` now splits `llm_name` on `+`. The right-hand side is loaded
with `load_lora_weights` and then **`fuse_lora`** - folding the delta into the
base weights rather than keeping a second tensor set consulted every forward
pass. Fusing also survives later pipeline surgery (ControlNet swap, VAE config
edits) that can silently drop an unfused adapter.

Encoding it in the name means nothing else changed: UI dropdowns, `KNOWN_MODELS`,
the A1111 facade's `sd_model_checkpoint` and the `pipes` cache key all treat
`base+lora` as one more model string. A failed LoRA load is caught and degrades
to the bare base **with a warning**, because an SDXL LoRA on an SD1.5 base (rank
mismatch) is the common mistake and should not take down a generation.

### Measured

`stabilityai/stable-diffusion-xl-base-1.0` + `nerijs/pixel-art-xl`, prompt
containing the LoRA's trigger "pixel art", 25 steps, cfg 7, 1024px, 21s:

| config | colours | top-32 share | blockiness |
|---|---|---|---|
| sdxl-turbo (old default) | 60252 | 36.9% | 25.2% |
| All-In-One + `pixelsprite` | 37444 | 64.0% | 50.2% |
| **SDXL + pixel-art-xl LoRA** | 81788 | **86.6%** | **74.7%** |

**86.6% of pixels come from 32 colours** and 74.7% of horizontally-adjacent pixel
pairs are identical - a real pixel grid. The raw colour count is highest only
because it renders at 1024 and carries a thin anti-aliased tail over the
remaining 13%. This is the first output in the project that is structurally
pixel art rather than an illustration imitating one.

`top-32 share` and `blockiness` are the metrics worth tracking; raw unique-colour
count is misleading and should not be used alone.

### Roster

Three image-generation configs share ONE 6.7GB base, at 0.4GB total extra disk:

| model string | status |
|---|---|
| `...xl-base-1.0+nerijs/pixel-art-xl` | measured, best |
| `...xl-base-1.0+Muapi/soft-pixel-art-xl` | untested |
| `...xl-base-1.0+ntc-ai/SDXL-LoRA-slider.pixel-art` | untested |

`Limbicnation/pixel-art-lora` is excluded: its `base_model` is
`black-forest-labs/FLUX.2-klein-4B`, and LoRA ranks are tied to the UNet they
were trained on.

### Deleted

`stabilityai/sdxl-turbo` (6619MB), `kohbanye/pixel-art-style` (4070MB),
`Onodofthenorth/SD_PixelArt_SpriteSheet_Generator` (5230MB) - 15.9GB. All three
were superseded by measurement, not preference: turbo is distilled so negative
prompts are inert, Onodofthenorth only ships sheet triggers, kohbanye was never
used for a single generation.

## 2026-08-21: final roster after the cleanup

Model store went **41GB -> 25GB**. Deleted: `sdxl-turbo` (6619MB),
`kohbanye/pixel-art-style` (4070MB), `Onodofthenorth/...` (5230MB),
`Qwen2.5-14B-Instruct-Q4_K_M.gguf` (8572MB) = **24.5GB**. Added 8.4GB.

| role | model | size | VRAM | status |
|---|---|---|---|---|
| text | `Qwen3-8B-Q8_0.gguf` | 8.2GB | 10833MiB | verified, 1.8s warm |
| image gen | `stable-diffusion-xl-base-1.0` + `nerijs/pixel-art-xl` | 6.7 + 0.16GB | ~7GB | measured best |
| image gen | same base + `Muapi/soft-pixel-art-xl` | +0.32GB | ~7GB | untested |
| image gen | same base + `ntc-ai/SDXL-LoRA-slider.pixel-art` | +0.08GB | ~7GB | untested |
| pose (step 2) | `All-In-One-Pixel-Model` + `control_v11p_sd15_openpose` | 4.0 + 1.4GB | ~3.4GB | measured, only working pose |
| pose (SDXL) | `thibaud/controlnet-openpose-sdxl-1.0` | 4.7GB | 11GB | 3.6x weaker, **deletion candidate** |

**Qwen2.5-14B -> Qwen3-8B was an upgrade, not a downgrade.** Q8_0 on 8B is
near-lossless where Q4_K_M on 14B is not, both land at ~8.3GB, and Qwen3 is a
2025 architecture against 2024.

### Dedicated image-EDITING models do not fit this hardware

Requested: two. Not viable, and the blocker is not the transformer:

| Qwen-Image-Edit-2511 component | size |
|---|---|
| transformer (GGUF Q3_K_M replaces the 38GB original) | 9.24GB |
| **text_encoder - NOT published in any GGUF repo** | **15.45GB** |
| vae | 0.24GB |
| **per model** | **~24.9GB** |

The transformer quantises to a usable 9.24GB. The text encoder does not: no GGUF
exists, `GGUFQuantizationConfig` applies to the transformer rather than text
encoders, and `QwenImageEditPipeline` wants a real transformers model there.
At fp16 it is **15.45GB against a 12GB card and an 11GB WSL RAM cap**, so
`enable_model_cpu_offload` would swap rather than stream. Two such models would
be ~50GB of disk on top of everything else.

`FluxKontextPipeline` is present in diffusers 0.39.0 and has the same shape of
problem: quantisable transformer, unquantised T5 encoder.

**What already covers "changing images" at zero extra cost:** SDXL img2img
(`StableDiffusionXLImg2ImgPipeline`, what step 2 already runs) and ControlNet
conditioning. Both are loaded and working. A dedicated 20B editing model buys
instruction-following edits ("remove the object", "rotate the character"), which
is a genuine capability gap - but it is a hardware gap, not a model-choice one.

### Verified this session

diffusers 0.39.0, torch 2.13.0+cu130. `QwenImageEditPipeline`,
`QwenImageEditPlusPipeline`, `FluxKontextPipeline`, `GGUFQuantizationConfig` all
present; the `gguf` package imports and exposes `GGUFReader`. So the GGUF path is
available whenever a model appears whose *encoder* also fits.

## CORRECTION 2026-08-21: image editing IS affordable, via pre-quantised NF4

The section above concluded that dedicated image-editing models do not fit this
hardware, on the grounds that the Qwen-Image-Edit text encoder is 15.45GB with no
GGUF available. **That reasoning only considered GGUF.** It missed
bitsandbytes NF4, and repos that ship *already quantised*:

| repo | total | transformer | text encoder | `model_index.json` |
|---|---|---|---|---|
| `seochan99/Qwen-Image-Edit-2511-bnb-nf4` | **16.79GB** | 10.71GB | **5.80GB** | yes |
| `Meatfucker/Flux.1-Kontext-dev-bnb-nf4` | 12.52GB | 6.24GB | 5.89GB | yes |

The encoder is **5.80GB rather than 15.45GB**, and ships that way - there is no
fp16 download to quantise at load. Both are complete diffusers pipelines
loadable with `from_pretrained`.

Two distinctions that were conflated the first time:

- `GGUFQuantizationConfig` quantises the **transformer/UNet**. It does not apply
  to text encoders, which is why searching for a GGUF encoder found nothing and
  looked like a dead end.
- **bitsandbytes NF4 does apply to text encoders**, and pre-quantised repos ship
  both halves. `BitsAndBytesConfig` is importable from diffusers *and*
  transformers without the backend installed, so its presence proves nothing -
  `bitsandbytes` itself must be present or `from_pretrained` raises on NF4
  weights. It was missing; `bitsandbytes>=0.43.0` is now in both requirements
  files, and 0.50.1 is installed in the running worker.

### Chosen: Qwen-Image-Edit-2511

Not the smaller FLUX Kontext, because **both requested editing capabilities are
LoRAs against this exact base**:

| LoRA | size | capability |
|---|---|---|
| `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA` | 0.62GB | re-render a subject from another angle |
| `prithivMLmods/Qwen-Image-Edit-2511-Object-Remover` | 0.44GB | remove an object |

That is the same shape as the pixel-art LoRAs: **one base, several capabilities,
~1GB each** rather than a separate multi-GB model per task. "Multiple angles" is
also directly relevant to sprite work - turning a character to face another
direction is the thing img2img provably cannot do (README, step 2).

VRAM: NF4 weights stay quantised in VRAM, so the transformer is ~10.7GB against
11.7GB free. The encoder must be offloaded - encode the prompt, free it, then
run the transformer - which `enable_model_cpu_offload` does. Untested.

## CORRECTION 2026-08-21 (again): NF4 helps, but bnb 4-bit cannot be offloaded

The correction above claimed pre-quantised NF4 makes image editing affordable.
Downloading `seochan99/Qwen-Image-Edit-2511-bnb-nf4` (16.79GB) and loading it
proved that **half right**. Measured placement with
`device_map="balanced", max_memory={0:"6GiB","cpu":"16GiB"}`:

```
transformer  -> cuda:0   (all 10.71GB, one indivisible block)
text_encoder -> cpu      (offloaded correctly)
vae          -> cuda:0
torch.cuda.memory_allocated: 10.95 GB
```

**`max_memory` was ignored for the transformer.** bitsandbytes 4-bit layers
cannot be split across devices or offloaded to CPU - they must be fully
GPU-resident. The text encoder offloads because it ends up on the meta/CPU path,
but the 10.71GB transformer does not, leaving ~0.75GB of an 11.7GB card for the
activations of a 20B model.

Symptoms, in the order they appear, none of which say "out of memory":

1. `RuntimeError: CUDA driver error: device not ready` - during load, or during
   inference if the weights just fit.
2. `!handles_.at(i) INTERNAL ASSERT FAILED ... CUDACachingAllocator.cpp:467` on
   the **next** request. The first failure leaves the CUDA context unusable and
   only a worker restart clears it.

Lowering `EDIT_GPU_BUDGET` does not help: the constraint is one indivisible
block, not a budget. Downscaling the source to 512 does not help either - it
reduces activations, not the 10.71GB floor.

### What this actually rules in and out

The deciding number is the **transformer** size, because that is the part that
must be wholly resident:

| pipeline | NF4 transformer | text encoder (offloadable) | fits 11.7GB? |
|---|---|---|---|
| `Qwen-Image-Edit-2511-bnb-nf4` | **10.71GB** | 5.80GB | no - 0.75GB left for activations |
| `Flux.1-Kontext-dev-bnb-nf4` | **6.24GB** | 5.89GB | plausible - ~5GB for activations |

So the roster rule is: judge an editing pipeline by its transformer, not its
total. Total size is misleading precisely because the encoder half offloads and
the transformer half cannot.

## FINAL 2026-08-21: image editing does not fit this machine. RAM, not VRAM.

FLUX Kontext was chosen over Qwen-Image-Edit on the transformer rule above
(6.24GB vs 10.71GB) and downloaded (12.52GB, verified complete). It does not run
either, and the reason is different from Qwen's.

Placement with `device_map="balanced"` is correct - measured:

```
transformer     -> cuda:0        (6.24GB)
text_encoder    -> cuda:0        (CLIP, 0.23GB)
text_encoder_2  -> cpu           (T5, 5.89GB)
vae             -> cuda:0
torch.cuda.memory_allocated: 6.62 GB
```

But a static device map has **no runtime hooks**, so calling the CPU-resident T5
raises `Cannot copy out of meta tensor; no data!`.

`enable_model_cpu_offload` installs those hooks - and it works by keeping every
weight in **system RAM** and streaming each component to the GPU as it is
called. That needs all 12.52GB resident in RAM. The WSL VM is capped at 11GB.
The worker was killed mid-load:

```
18:54:10  Task received
18:55:19  Loading pipeline components...  4/7    <- then silence
18:55:21  Connected to redis / mingle: all alone <- fresh Celery start
```

Silent death at component 4 of 7 - the T5 - with no traceback, followed by an
immediate respawn. That is a **kernel OOM kill**, and it is invisible in
`docker inspect` (`OOMKilled=false`) because the container has no memory limit
of its own; the ceiling is the VM's.

### Why raising the WSL cap does not rescue it

Host RAM is 15.9GB. The weights need 12.52GB plus allocator overhead. Setting
`memory=13GB` in `.wslconfig` would leave Windows ~2.9GB, which thrashes; and
12GB is still short of 12.52GB. There is no setting that satisfies both sides.

### The rule, stated properly

An editing pipeline must satisfy BOTH, and they pull in opposite directions:

- **transformer <= free VRAM** (~11.7GB), because bitsandbytes 4-bit layers
  cannot be split or offloaded, and
- **total weights <= WSL RAM** (11GB), because `enable_model_cpu_offload` -
  the only thing that makes the offloaded half callable - keeps everything in
  system RAM.

| pipeline | transformer | total | verdict |
|---|---|---|---|
| Qwen-Image-Edit-2511 NF4 | 10.71GB | 16.79GB | fails both |
| FLUX.1-Kontext-dev NF4 | 6.24GB | 12.52GB | passes VRAM, fails RAM by 1.5GB |

**Not a model-selection problem.** It needs more system RAM (24GB+ would settle
it comfortably) or a smaller editing model than currently exists. The earlier
"NF4 makes editing affordable" correction was too optimistic; the original
"does not fit" was right, for a reason I had not yet found.

### What image-to-image capability actually exists here

SDXL img2img and ControlNet conditioning, both loaded and working, at no extra
cost. What is missing is *instruction* editing - "remove the shield", "show this
character from the side" - which is exactly what the two requested LoRAs were
for. That capability is unavailable on this hardware, not merely unconfigured.
