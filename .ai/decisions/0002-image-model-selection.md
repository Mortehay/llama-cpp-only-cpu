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
