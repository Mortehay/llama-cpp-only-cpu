# 0012 - Qwen-Image-2512 (GGUF) as the UI's default step-1 model

Date: 2026-09-28
Status: **Accepted. Step-1 integration built 2026-09-28** (`qwen_t2i.py`,
`tasks._generate_core_gguf`, roster `gguf:` prefix + `ui_default`, facade
busy gate, timeout 240 -> 285s). End to end through the real worker: 257.6s
cold (transformer load 59s on a cold file cache vs ~20s warm), result a clean
single-subject core. Takes direction E
of [0011](0011-brain-tiers-and-colibri.md) off the shelf. Measurements below
are from a smoke test and one 12-image bench; the step-2 quant question is
still open.

## Why this exists

The owner downloaded a batch of large GGUF files into `D:\downloads`, called
them "LLM images", and asked for the ones that fit to become selectable for
image and sprite generation. **None of them is an LLM.** Reading the GGUF
headers (`general.architecture` plus the tensor-name prefixes, since several
files carry a wrong architecture label) showed diffusion transformers, one
video model, and one text encoder that has no diffusion model to go with it.
In this repo's terms they are *image models* (`llm_name`) - see `domain.md`.

## What was in the batch (headers read 2026-09-27/28)

| file | size | actually is | diffusers 0.40 loads it? | verdict |
|---|---|---|---|---|
| `Qwen-Image-2512-Q3_K_M.gguf` | 9.69 GB | Qwen-Image 20B, 60 blocks, Q3_K/Q4_K mix (not unsloth's 9.93 GB build) | yes - `QwenImageTransformer2DModel.from_single_file` | **taken** |
| `qwen-image-lite-iq4_xs.gguf` | 10.17 GB | same 60-block layout, labelled `pig`, IQ4_XS + F16 | probably (IQ4_XS is supported); not run | parked - what "lite" means is unknown, and it is larger |
| `Magic-Wan-Image-V3-Q5_K_M.gguf` | 10.06 GB | Wan 14B (40 blocks) fine-tuned for stills | yes (`WanTransformer3DModel`), but needs a UMT5-XXL encoder not on disk | parked |
| `stable-diffusion-v3-5-large-pure-Q4_0.gguf` | 7.75 GB | SD3.5 Large packed with its text encoders and VAE, stable-diffusion.cpp layout (`model.` / `cond_stage_model.` / `first_stage_model.`) | no - that packaging needs a second runtime | parked |
| `qwen-image-edit-2511-Q3_K_L.gguf` | 9.85 GiB | Qwen-Image-**Edit**-2511 (has the `__index_timestep_zero__` marker) | yes | **does not fit** - 0.0 GiB free after placement (Open #2) |
| `Qwen-Image-Edit-2511_clear_Q3_K_L.gguf` | 9.65 GiB | Qwen-Image-Edit, 60 blocks, no 2511 marker tensor | yes | not run - larger than the Q3_K_M that already missed |
| `qwen-image-edit-iq4_xs.gguf` | 10.2 GB | Qwen-Image-Edit, version not stated | probably | not pursued |
| `krea2_turbo_edit-Q6_K.gguf` | 9.86 GB | labelled `qwen_image`, but a custom 28-block design (`txtfusion`, `egg_w=6144`) | no | **rejected** - no loader, and "turbo" means guidance 0 |
| `SenseNova-U1.5-8B-MoT-Q3_K_M.gguf` | 10.17 GB | combined understand+generate model (MoT), mislabelled `wan` | no | **rejected** |
| `joyai_image_edit_plus-Q4_0.gguf` | 9.57 GB | custom `double_blocks` design, mislabelled `wan` | no | **rejected** |
| `LLaDA-Image-Base-text_encoder-Q4_K_M.gguf` | 8.55 GB | only the LLaDA2-MoE *text encoder* of LLaDA-Image-Turbo | useless alone | **rejected** |
| `cosmos-predict2-14b-v2w-480p...q5_1.gguf` | 10.09 GB | 14B video-to-world model | - | **rejected** |

**A header's `general.architecture` is not evidence.** Four of these files
carry a label (`wan`, `qwen_image`, `pig`) that the tensor names contradict.
Read the tensor prefixes and block count.

## Why it loads here with almost no new weights

Qwen-Image-2512 shares everything but the transformer with the Qwen-Image-Edit
work already on disk (`qwen_edit.py`, ADR 0005). The HF LFS hashes are
identical for the text encoder across `Qwen/Qwen-Image`, `Qwen-Image-2512` and
`Qwen-Image-Edit-2511`, and identical for the VAE between 2512 and Edit-2511.
So the pipeline is:

- transformer: the GGUF above, at `/models/image-gguf/transformers/` (host
  `/home/markunn/sprite-data/models/image-gguf/transformers/`). Moved from
  `/models/gguf/` the same day - see "Where image GGUFs live" below
- NF4 text encoder, VAE, tokenizer: `ovedrive/Qwen-Image-Edit-2511-4bit`
  (already cached)
- transformer + scheduler **config** and `model_index.json`: `Qwen/Qwen-Image-2512`
  (a few KB, fetched once - the worker runs `HF_HUB_OFFLINE`).

### Where image GGUFs live - and why not `/models/gguf/`

They were first copied to `/models/gguf/`. That put them in front of
`llm_engine`: llama.cpp runs as a router with `--models-dir /models` and offers
as a **chat model** every `.gguf` directly in `/models` plus every folder that
directly contains one. `/v1/models` returned `["Qwen2.5-3B-Instruct-Q4_K_M",
"gguf"]`, and `worlds._llm_model()` takes entry `[0]` - an ordering change away
from asking the router to load a 9.7 GB image transformer as a text model.
Found by the audio session the same day, which hardened its own picker.

Fixed at the root: moved to `/models/image-gguf/transformers/`, two levels
deep, which the router does not scan (the same reason the HF-cache GGUF under
`models--unsloth--...` was never listed). The router reads its model list **at
startup only**, so `llm_engine` needed a restart; afterwards `/v1/models`
returned only the Qwen2.5-3B. `core_models.GGUF_DIR` carries the warning.

Do **not** point `config=` at the Edit repo: its transformer config carries
`zero_cond_t: true`, which 2512's does not. The pipeline class is plain
`QwenImagePipeline`, not `QwenImageEditPlusPipeline`.

## Measured

Worker container, RTX 3060 12 GB, diffusers 0.40, bf16, transformer resident
(no CPU offload), text encoder in a **separate process** first (the rule from
`qwen_edit.encode_only`: never both halves in one process).

| stage | figure |
|---|---|
| text encoder load + 2 prompts | 33.5 s, 4.8 GiB peak |
| transformer placed on card | 19.6-23 s, 9.25 GiB allocated |
| 512 px, 20 steps, true CFG 4 | **7.33 s/step**, ~144 s/image, 9.55 GiB peak, **1.3 GiB free** |
| after exit | 788 MiB used - nothing leaked |

**Bench**, 4 subjects x 3 seeds (101/202/303), something2-shaped prompts passed
through `split_negations`, both models judged by the production cutout
helpers (`remove_background` -> `_isolate_largest_sprite` -> kept %):

| | Qwen-Image-2512 (512, 20 st, cfg 4) | SDXL + nerijs (1024, 25 st, cfg 7) |
|---|---|---|
| contact sheets, by eye | **0/12** | **2/12** (boots 202, potion 202) |
| mean cutout kept | 98.2% | 77.4% |
| wrong framing | 2/12 (knight 202 = helmet only; knight 303 cropped at waist) | 0/12 |
| time/image, warm | ~144 s | ~20-26 s |
| owner's judgement | "much better" | |

Three measurement notes, each of which would mislead a re-run:

- **Production would have accepted an SDXL contact sheet.** Boots 202 kept
  23.9%, just above the 20% `cutout_failed` floor. This confirms the CLAUDE.md
  warning that an accepted cutout can be the largest item *of* a sheet.
- **The bench's automatic "multi" flag over-counts SDXL.** Its drop shadows and
  grey gradient backgrounds leave extra blobs after the cutout (potion 101,
  boots 101 are single objects). The by-eye count is the one quoted above.
- **The 0002 style figures could not be reproduced.** No script for "86.6% of
  pixels from 32 colours / 74.7% blockiness" was committed. The bench's
  top-32-colour coverage and `measure.pixel_scale` did not discriminate (every
  image: scale 1, confidence 1.0). Pixel style is judged by eye for now.

The bench is `scripts/bench-qwen-image.py`. Each image is written to the
`generations` ledger, so a run is visible on the **Activity** tab.

## Decisions

### D1. Selectable everywhere, default only in the UI

Qwen-Image-2512 becomes a roster entry, and the **UI preselects it** for step 1
(React and legacy). `core_models.default_model()` stays SDXL + nerijs.

Why the split: `default_model()` is also the `Form` default in `main.py`, so
it decides the model for every script and caller that omits `llm_name` - the
exact leak 0011/ticket 12 closed for SDXL-Turbo. Making the slow model the
silent default there would slow batch callers 6x without anyone having chosen
it.

### D2. Not offered to something2

A cold Qwen run is ~144 s denoise plus ~55 s encode and load, against
something2's 240 s ceiling. The entity-cutout retry loop (3 attempts,
`ENTITY_CUTOUT_BUDGET_S` 180 s) could not complete one retry. Their side is
synchronous-only, so an async 202-and-poll path is not an option. `a1111.KNOWN_MODELS`
is unchanged.

**Reversed 2026-09-28, same day, at the owner's request** ("longer, but the
quality is much better"). The facade budget is 285 s now, not 240. Qwen runs
through `generate_raw_task` via `tasks._raw_image_gguf`: square only (the
facade refuses other sizes with 400), rendered at 512 and scaled NEAREST to
the requested size, followed by the same cutout checks as SDXL. Measured:
Q3_K_M at 1024x1024 with cutout took **253.5 s** end to end, one clean
subject, 100% kept, with no time left for a cutout retry. The fast Q2_K +
Lightning variant (2b) is `KNOWN_MODELS[0]`, measured at 106.8-112.7 s cold.
Model switching between these and SDXL now goes through the model gateway
(`.ai/specs/model-gateway/`).

### D3. Step 2 is already the Qwen-Edit conveyor - nothing to switch

Corrected the same day, after reading the code rather than the docs. Both UIs
already send step 2 to the Qwen-Image-Edit job conveyor: React
`SheetGenerator.tsx` and legacy `generator.js` both `POST /api/jobs` ->
`build_sheet_job` -> `scripts/build-sheet.py` (ADR 0005). The SD1.5 +
`control_v11p_sd15_openpose` path (`POST /api/generate_sheet` ->
`generate_spritesheet_task`) has **no UI caller**; only the Retry endpoint and
`scripts/two-step-test.sh` reach it, and `scripts/check-ui.py` guards against a
UI calling it. There is no selector between the two.

So "Qwen by default at step 2" is already true. What remains is the quant
(Open #2). Re-exposing SD1.5 as a selectable sheet path would be new work, not a
default flip - and `generate_sheet`'s own `Form` default is still
`stabilityai/sdxl-turbo`, which 0011 D1 set out to remove.

### D4. Keep the 9.69 GB quant

Published unsloth quants for 2512: Q2_K 7.33 GB, Q3_K_S 9.22 GB, Q3_K_M
9.93 GB, Q4_0 11.85 GB and up. Nothing between Q2_K and the file in hand is
meaningfully smaller, and Q4 does not fit. A smaller quant only buys 1024 px
output or coexistence with another resident model; neither is needed for
sprites that are pixelated to <=128 px. Q2_K is the fallback if 1024 matters.

## Open

1. ~~**Per-image decode.**~~ **Answered 2026-09-28: it fits.** The bench
   batched decodes (copied from `qwen_edit.py`, whose measured failure was CPU
   offload, not the VAE). The integrated path decodes inside the same pipeline
   call with the transformer resident and VAE tiling on: 1.27 GiB free after
   decode.
2. ~~**Step-2 quant.**~~ **Answered 2026-09-28: stay on Q2_K.**
   `qwen-image-edit-2511-Q3_K_L` is 10.58 GB on disk (9.85 GiB - the table
   above mixed units). Placed resident it left **0.0 GiB free** and died at the
   first denoise with `CUDA driver error: device not ready` - the out-of-VRAM
   form qwen_edit.py documents, not a context fault: `make gpu-health` right
   after said OK, 11.2 GB free. Same core, same 4 directions on Q2_K: 8.4
   s/step, 33 s/cell, fine. The `_clear` Q3_K_L (10.36 GB) is also above the
   Q3_K_M that already missed by 24 MiB, so it was not run. A bigger Edit quant
   needs CPU offload, which 0005 measured as the worse trade on this box.
   `QWEN_EDIT_GGUF_FILE` now accepts an absolute local path, so a future
   quant can be tried without a Hub round trip.

   **Q3_K_S tried the same day - fits, not adopted.** unsloth
   `qwen-image-edit-2511-Q3_K_S.gguf`, 9.22 GB, sha256 `d613d933...` verified
   against the Hub (an FDM file mid-download has the FINAL size, preallocated,
   with holes - size is not evidence of completion, the hash is). Same core,
   seed 0, s/e/n/w at 512px:

   | | Q2_K (production) | Q3_K_S |
   |---|---|---|
   | VRAM free after placement | 2.7 GiB | **0.7 GiB** |
   | s/step | 8.4 | 8.8 |
   | back view | correct, no face | **wrong - drew the face** |
   | edges | clean | more white fringe |

   One seed, so the back-view miss may be chance - but nothing here buys back
   two-thirds of the headroom on a card with this fault history. The file is
   kept at `/models/image-gguf/transformers/` for a multi-seed comparison if one is ever wanted;
   switching is `QWEN_EDIT_GGUF_FILE` in compose plus a worker restart.
2b. **Lightning for step 1 (2026-09-28): quality holds, VRAM does not - on
   this quant.** `lightx2v/Qwen-Image-2512-Lightning` 8-step bf16 LoRA
   (0.85 GB) on the 9.69 GB Q3_K_M, 8 steps, true CFG 1, the distillation's
   own scheduler (shift 3, from ModelTC's generate_with_diffusers.py). Same
   bench, same 12 cases:

   | | baseline (20 st, cfg 4) | Lightning 8 st, cfg 1 |
   |---|---|---|
   | contact sheets | 0/12 | **0/12** - holds with the negative prompt inert |
   | cutout kept | 98.2% | 100% |
   | steady s/image | ~144 | **~30** |
   | VRAM free after placement | 1.3 GiB | **0.00 GiB** |
   | timings | stable | 332, 308, then ~30, then 99-103 s; placement 244 s |

   By eye: equal or crisper pixel art; knight 303 now full-body (was cropped);
   knight 202 still helmet-only; **less variety** - the three slime seeds are
   near-identical, the usual cost of step distillation. The erratic timings
   with a full card look like WDDM spilling VRAM to system RAM - inferred from
   the pattern, not measured. Not shippable at 0 GiB on a card with this fault
   history. Next: the 7.33 GB Q2_K transformer, which should leave ~2 GiB with
   the LoRA - if Q2_K does not cost the quality that Lightning kept.

   **Q2_K + Lightning 8 (same day): shippable, slightly rougher.** unsloth
   `qwen-image-2512-Q2_K.gguf`, 7.33 GB, sha256 `176678f0...` verified, at
   `/models/image-gguf/transformers/`. Same bench:

   | | Q3_K_M 20 st cfg 4 | Q3_K_M + L8 | **Q2_K + L8** |
   |---|---|---|---|
   | contact sheets | 0/12 | 0/12 | **0/12** |
   | s/image | ~144 | ~30, spikes to 332 | **24, flat** (placement 51 s) |
   | VRAM free after placement | 1.3 GiB | 0.00 | **3.12 GiB** |
   | knights full-body | 1/3 | 2/3 | **3/3**, stockier |
   | texture / edges | cleanest | crisp | **noisier** slime edges, less potion detail |
   | background | white | white | faint grey (cutout still kept 98-100%) |

   Prompt adherence equal or better ("obsidian" boots come out black);
   fidelity visibly lower than the 20-step baseline. A speed/quality trade,
   not a free win - so it is the owner's call which the UI preselects.

2c. **A persistent Qwen process (spike, 2026-09-28): works, costs RAM.** Each
   request today starts two fresh subprocesses, so ~75 of a cold ~100-110 s
   is re-loading. Spike: one process loads the NF4 encoder and the Q2_K +
   Lightning transformer once, parks both in host RAM, and per request moves
   encoder -> card -> encode -> host, then transformer -> card -> denoise.
   Three requests, all correct images:

   | | R1 | R2 | R3 |
   |---|---|---|---|
   | total | 43.9 s | 45.9 s | 35.9 s |
   | of which CPU<->GPU swaps | ~9.4 s | ~18.5 s | ~9.8 s |
   | denoise 8 + decode | 28.3 s | 25.8 s | 25.6 s |

   One-time load ~85 s. Encoder `.to("cpu")` is the slowest move (~4 s, NF4).
   **Resident host RAM ~17 GiB** (RSS 13.3 GiB after load, 16.9 GiB after the
   first round trip) against a WSL VM of 23 GiB total - the reason it is not
   built: the worker would sit one audio or SDXL job away from the OOM killer.
   Precondition if wanted: raise `.wslconfig` `memory=` (still 24 GB from the
   i3 era; the host now has 47.8 GB) and make the gateway release it on a
   model switch. Spike VRAM was measured with the gateway's SD1.5 pipeline
   (2,054 MiB) still resident in the worker, which a real run evicts first.

   **Built the same day** (`qwen_server.py`; `tasks._qwen_server_*`;
   `.wslconfig` memory 24 -> 36 GB, `free` now 35 GiB). Through the real queue:

   | | end to end | server timings |
   |---|---|---|
   | request 1 (starts the child, READY 23.4 s) | 57.7 s | 34.1 s |
   | request 2 | **38.9 s** | encode 6.2 + place 2.5 + denoise 23.6 |
   | request 3 | **34.0 s** | 32.0 s |
   | per-request subprocesses, warm page cache | 57-61 s | - |
   | per-request subprocesses, cold | 100-130 s | - |

   The honest comparison is ~60 s -> ~35-40 s warm (~40% faster), not the
   ~100 s cold figure. VRAM parks at ~0.9 GB between requests; a gateway
   switch to SDXL stopped the child and host RAM used fell 19.5 -> 3.8 GB.
   Any child failure falls back to the per-request path - proven when a first
   build died at birth (its orphan check treated the Celery worker, PID 1 in
   the container, as init) and all three requests still succeeded, slower.
   The worker now passes its pid in `QWEN_SERVER_PARENT`.

3. **A pixel-style metric that works.** Measure after the production pixelate
   step, or drop the pretence and keep the by-eye grid as the gate.
4. **Framing misses** (helmet-only, cropped). A prompt/cutout problem to watch,
   not yet a pattern at n=12.
