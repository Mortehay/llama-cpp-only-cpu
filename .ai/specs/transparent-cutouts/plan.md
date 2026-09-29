# Transparent cutouts - plan

Status: **approved 2026-09-29; the design CHANGED after measurement - see
"Outcome" at the end.** The prompt rewrite and the stricter refusal shipped;
the BiRefNet slices below are deferred, not needed for the reported bug.

## Problem (measured 2026-09-29)

The fast Qwen model (`gguf:qwen-image-2512-Q2_K+lightning8`) returns cutouts
whose background is still there. The last three something2 entity cutouts
shipped at **3.4%, 6.3% and 6.6% transparent**. The two before them were
40-52%.

Cause: the prompt asks for a "transparent background" (something2 sends
`"... single object, with transparent background, only slime"`), so the model
**paints the grey/white transparency checkerboard into the pixels**.
`remove_background` flood-fills from one corner colour with tolerance 22, so it
cannot clear a two-tone pattern. The refusal floor only rejects results under
2% transparent, so these passed as successes. It is not Qwen-specific: any
model given that phrase can draw the checker.

## Decisions (owner, 2026-09-29)

| # | Question | Answer |
|---|---|---|
| 1 | What must be transparent | Object cutouts (something2 cutout/entity), UI character cores, **and any request whose prompt asks for a transparent background**. Tiles, terrain and maps stay opaque. |
| 2 | Method | AI background removal: **BiRefNet** (MIT). Not RMBG-1.4/2.0, which are non-commercial and the owner may sell the game. |
| 3 | Reach | **All models** (SDXL, SD1.5, Qwen); it replaces the flood fill on the in-scope paths. |
| 4 | Edges | **Hard outline** (each pixel fully opaque or fully transparent). Inner translucency (slime, glass) is kept only if the test set shows it looks right; otherwise the inside is opaque. Nice-to-have, not a must. |
| 5 | Prompt | **Rewrite** transparent-background phrases to "isolated on a plain solid white background" and add "checkerboard, transparency grid" to the negative. The request stays transparent-required. |
| 6 | Failure | **Retry on a new seed, then refuse.** Uses the existing budget (`ENTITY_CUTOUT_ATTEMPTS` 3, `ENTITY_CUTOUT_BUDGET_S` 180 s; a fast-Qwen attempt is ~35-40 s). something2 gets 422; a UI core fails with Retry. An opaque square is never a success. |

## Terms

- **Transparent-required**: the request has the cutout flag, is a UI core, or
  its prompt asks for a transparent background.
- **Painted checkerboard**: the transparency pattern drawn *into the pixels*,
  as opposed to real alpha.

## Facts the design rests on

- Worker has `torch`, `torchvision`, `transformers`; no matting model cached.
- `HF_HUB_OFFLINE=1` in the worker, so the weights need a one-off
  `docker compose run -e HF_HUB_OFFLINE=0 ...` fetch (existing pattern, see
  compose comments at the fetch-qwen-edit example). C: has 195 GB free.
- Worker torch has **2 CPU threads**, so CPU inference is likely too slow.
  Expect the GPU.
- VRAM: the slow Q3_K_M refuses to start below `QWEN_MIN_FREE_MB` (10 GB
  free), so BiRefNet must NOT stay resident on the card. Plan: keep it in host
  RAM and move it to the GPU only for the ~1 s it runs, the same park/place
  trick the persistent Qwen process uses.
- Callers of `remove_background`:

  | Caller | Makes | In scope |
  |---|---|---|
  | `tasks._finish_raw` | something2 entity cutouts | yes |
  | `tasks._finish_core` | UI cores | yes |
  | `tasks` side-view cores (~1881) | derived core views | yes |
  | `map_tasks` props (~793) | map objects | yes |
  | `tasks` sheet frames (~2889) | SD1.5 sheet path, no UI caller | yes (free once shared) |
  | `main.py` crop (~607) | manual crop, **API process, no GPU** | **no** - stays flood fill |

## Slices

### Slice 0 - measure before building (no production change)

1. Fetch `ZhengPeng7/BiRefNet_lite` and `ZhengPeng7/BiRefNet` (both MIT;
   confirm the licence in each repo card before use).
2. `scripts/bench-cutout.py`: a fixed test set of **8 subjects x 3 seeds x 2
   models** (fast Qwen, SDXL + nerijs). Subjects: translucent slime, glass
   potion, ghost, white-armour knight, obsidian boots, pine tree, plus the two
   latest real something2 entity prompts from the ledger. Each subject runs
   twice: the prompt as sent, and the rewritten prompt. The raw pre-cutout
   image is saved.
3. Run each image through flood fill, BiRefNet_lite and BiRefNet (GPU, and CPU
   once for timing). Output: one contact sheet on a magenta backdrop and a
   `--json` sibling with transparent %, isolate-kept %, time and VRAM peak.
4. From that, decide: lite vs full; GPU vs CPU; the hard-alpha threshold; the
   inner-translucency verdict; and the new refusal thresholds (see risks).
   **Owner reviews the contact sheet** before slice 1.

Cost: ~25-30 min of GPU. Hold it through the gateway (a pinned model) so
something2 gets 503s rather than contention.

### Slice 1 - something2 path (the reported bug)

- `cutout.py` (worker only): lazy-loaded BiRefNet, parked in host RAM and
  placed on the GPU per call; `cut_out(img) -> (RGBA, stats)` with a hard
  outline, and inner alpha only if slice 0 approved it.
- `rewrite_transparency(prompt, negative) -> (prompt, negative, required)` next
  to `split_negations`, and a unit test.
- `generate_raw_task`: transparent-required = the cutout flag OR a detected
  phrase. The rewrite is applied, then `cut_out` replaces flood fill +
  `_isolate_largest_sprite` stage 1. The largest-blob isolation and the
  kept-fraction guard stay, since BiRefNet can return a contact sheet as
  several blobs.
- New refusal: background residue above the slice-0 threshold ->
  `cutout_failed` -> the existing seed retry.
- **Accept:** replaying the 3 failing ledger requests (slime and the two
  before it) returns images with no grey/white squares on a magenta preview,
  or a 422 - never an opaque square. Time per attempt rises by at most ~2 s
  (slice-0 figure).

### Slice 2 - the other in-scope paths

- `_finish_core`, side-view cores, map props and sheet frames call `cut_out`
  in place of `remove_background`. The UI core path gains the same
  retry-on-new-seed loop.
- `CUTOUT_ENGINE=birefnet|floodfill` env (default birefnet) as the rollback
  switch. The API process always uses flood fill (the crop route).
- **Accept:** a UI core on fast Qwen and on SDXL comes out transparent on the
  magenta preview; a map prop likewise; tiles unchanged (opaque).

### Slice 3 - docs

The doc updates below, applied once approved.

## Out of scope

- Real alpha matting (ViTMatte etc.) for see-through subjects.
- Tiles, terrain, maps, and the manual crop route.
- Re-cutting images already on disk (possible later with the same function).

## Risks and open questions

- **The thresholds were tuned on flood fill** (20% kept floor, 99.5% clear
  ceiling, measured on 297 cutouts). They must be re-derived on BiRefNet masks
  in slice 0, not carried over.
- **"Background residue" needs a definition that works when the subject
  touches the frame edge.** A border-ring rule was tried before and was wrong
  for exactly that case (a tree at 40% border-clear). Candidate: the fraction
  of pixels BiRefNet calls background that are still opaque after thresholding,
  which should be ~0 by construction; plus a painted-checker detector (two
  alternating tones on a grid) on what remains.
- BiRefNet on pixel art is **unmeasured**. It is trained mostly on photos, so
  thin 1-px outlines may be eroded. Slice 0 exists to find out.
- The pinned-hold during slice 0 blocks something2 for its duration.

## Proposed doc updates (apply in slice 3, after approval)

1. **`.ai/decisions/0013-ai-background-removal.md`** (new ADR):
   - Context: the painted checkerboard and the measured 3-7% results.
   - Decision: BiRefNet for all object paths.
   - Rejected: RMBG (licence); repair-only flood fill (can't separate a
     two-tone backdrop by colour, as `key-checkerboard.py` already learned);
     ViTMatte (scope).
   - Consequences: thresholds re-derived; flood fill retained only for the API
     crop route and as the rollback engine.
2. **`CLAUDE.md`** tripwire under "Models": *"'transparent background' in a
   POSITIVE prompt makes models PAINT a checkerboard. It is rewritten on
   transparent-required paths (`rewrite_transparency`); background removal is
   BiRefNet (0013). Do not reach for colour-matching the checker: white
   subjects are the same colour as the light square."*
3. **`.ai/domain.md`**: the two terms above ("transparent-required", "painted
   checkerboard").

## Outcome (2026-09-29) - measured, and the plan above was changed by it

**Bench** (images and ledger rows in Activity under "bench: qwen-image-2.1
(claude)"; per-image numbers in `/app/images/bench_q21_compare.json`):

| Set | Model | Transparent share | Picture |
|---|---|---|---|
| A: 4 subjects x 3 seeds, "plain white background" | fast Qwen (prod cutout) | 12/12 clean, 41-64% | best detail |
| A | SDXL + nerijs (prod cutout) | 11/12, one grey leftover (18.5%) | good |
| A | Qwen-Image-2.1 Q8 (sd.cpp, native RGBA) | 12/12 clean, 64-79% | crude: "obsidian" -> black silhouettes, flat blobs |
| B: something2's "with transparent background" x 3 subjects x 3 seeds | fast Qwen (prod cutout) | **0/9: painted checker, 3-18%**, all reported success | detailed |
| B | SDXL + nerijs | 6/9 clean; 10.4%, 26.8%, 33% leftovers | good |
| B | Qwen-Image-2.1 | 9/9 clean | crude |

**Conclusion.** The bug is the phrase, not the model and not the remover:
the same fast Qwen, through the same flood fill, is clean when the prompt says
"plain white background". Qwen-Image-2.1 has perfect alpha but is not a
quality replacement (research licence; weights kept in
`/models/image-gguf/qwen-image-2.1/`, sd.cpp CUDA build in `~/sdcpp`).

**Shipped** (instead of slices 1-2):
- `tasks.rewrite_transparency()`: transparent-background phrases become
  "isolated on a plain white background" (+ checkerboard terms in the
  negative). Runs before `split_negations` in `generate_raw_task`, and in
  `_core_prompt` for UI cores. Asking for transparency makes the request a
  cutout (decision 1). Test: `make test-transparency-rewrite`.
- `CUTOUT_MIN_CLEAR` = 25% in `_finish_raw`, in the measured gap (broken <=
  18.5%, clean >= 41.4%). A refusal is `cutout_failed`, so the existing seed
  retry runs first, then 422 (decision 6). Known miss: 26.8% SDXL leftover.
- The interim `KNOWN_MODELS[0]` = SDXL was reverted: fast Qwen is the default
  again.

**Deferred:** BiRefNet/ToonOut (slices 0-2) - still the answer for a white
subject on a white backdrop, which the flood fill can eat; not needed for the
reported bug. Map props already prompt "plain flat white background".

## Segmenter (2026-09-29, same day) - the deferred slice, after all

The prompt fix left real gaps, found by auditing EVERY path that feeds the
web RPG game (the owner's requirement: anything used in the game, directly or
via sprite-sheet generation, needs a real transparent background):

| Path | Before today | Now |
|---|---|---|
| something2 entities | flood fill | rewrite + BiRefNet + 25% floor + seed retry |
| UI cores (source of every sheet) | flood fill, **no check**: 38/200 under 25%, all broken | rewrite + BiRefNet + floor + retry (new loop) |
| side-view cores | flood fill, no check | BiRefNet (via `remove_background`) |
| map props | flood fill + own refusal | BiRefNet (via `remove_background`) |
| sprite sheets | `pixelate.key_background` per cell | unchanged; 8 recent sheets all >= 31.5% transparent |
| tiles / terrain | none, on purpose | unchanged (must stay opaque) |
| manual crop (API, no GPU) | flood fill | unchanged (fallback) |

**Spike** (`BiRefNet` vs `ToonOut` vs shipped flood fill, 19 images: today's
failures plus 5 known-good):
- Both segmenters removed every painted checker, the white floor, the dark
  backdrop and the goblin's enclosed bow pockets.
- On a brick wall, a mountain scene and a room interior, BiRefNet cut the
  subject out; **ToonOut kept the scenery**. The research tip that ToonOut
  suits illustrations did not hold here, so plain BiRefNet ships.
- Neither damaged the known-good cutouts. Both failed one garden-grid scene;
  on item sheets both keep the largest item (as `_isolate_largest_sprite`).
- ~0.3 s per image, 2 GB peak VRAM during the call.
- ToonOut's `.pth` needs `squeeze_N.` -> `squeeze_module.N.` and prefix
  stripping to load strictly into the HF BiRefNet; a first non-strict load
  silently mixed weights. Weights kept in `/models` in case of a re-test.

**Shipped:** `cutout.py` - BiRefNet (MIT, `ZhengPeng7/BiRefNet`, 444 MB in
`/models`), fp16, parked in host RAM and placed on the GPU per call so it never
holds VRAM against the slow Qwen core's 10 GB free check. Mask snapped to
0/255 at 128 - hard pixel-art edges. `remove_background` uses it on the CUDA
worker and falls back to the flood fill on the API process, on
`CUTOUT_ENGINE=floodfill`, or if it fails to load. Deps `einops kornia timm`
added to requirements.

**Live check** through the real job path: goblin-archer core 59.9%
transparent with the bow pockets gone; something2's slime prompt 58.9%.
