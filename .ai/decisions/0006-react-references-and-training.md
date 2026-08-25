# 0006 - React front end, reference-driven style, and training on one 3060

Date: 2026-08-25
Status: Proposed. Builds on [0005](0005-back-to-2d-modern-editors.md), which
retired the 3D leg and established the 2D conveyor that now works.

## Why this exists

0005 fixed *how* sprites are made. This ADR is about *whose style they are*.
The conveyor currently produces competent generic pixel art; the goal is art
that drops into something2 without looking foreign. That needs reference
examples, and reference examples need somewhere to live, something to measure
them, and - for style specifically - training.

Six requirements, in the user's numbering:

1. Move the front end to React.
2. Add `reference-core`, `reference-sprite`, `reference-tile` tabs for examples.
3. Offer models that can actually be trained on this hardware.
4. Fix the codebase where it blocks the above.
5. Authenticated queue API for image/sprite/tile generation.
6. A gallery that shows everything generated.

## What was measured first, 2026-08-25

Facts, not impressions. Each one changes the plan.

| Finding | Consequence |
|---|---|
| `#tab-settings` carried an inline `style="display:none;"` while `switchTab` only toggles a class. Inline style beats a class selector, so Settings could never open. | Fixed in this session. It is also the argument for item 1: this bug is unrepresentable in React. |
| `sprite_images` holds **2** undeleted rows. `jobs` holds **20** rows, **13** with a finished `sheet_path`. The gallery reads only `sprite_images`. | 13 finished sheets have never been visible in the UI. Item 6 is a missing join, not a styling task. |
| `tasks.py` is 142 KB in one module. | Any feature touching generation edits a file too large to reason about. Split before adding tile/training paths. |
| `mesh_worker` and `rig_worker` are still built and mounted by compose, though 0005 retired the 3D leg. | Dead build weight and dead images. Remove. |
| `/models` cache is 24 GB and already holds `stabilityai/stable-diffusion-xl-base-1.0` plus four pixel-art LoRAs (`nerijs/pixel-art-xl`, `PublicPrompts/All-In-One-Pixel-Model`, `Muapi/soft-pixel-art-xl`, `ntc-ai/SDXL-LoRA-slider.pixel-art`). | The training base is already on disk. No 7 GB download to start. |
| WSL root has 879 GB free; C: has 49 GB. GPU is a 3060, 12 GB, ~2.4 GB resident at idle. | Training has disk room. VRAM is the binding constraint, not storage. |
| `jobs.py` already implements queue semantics: `POST /api/jobs` -> 202 + UUID, poll `GET /api/jobs/{id}`, batch `GET /api/jobs?ids=&since=`. Auth is one shared bearer token, and a **no-op when `API_TOKEN` is unset**. | Item 5 is mostly done. What is missing is real key management and non-sheet job kinds. |

## Decisions

### D1. Backend contract before React, despite item 1 being listed first

React built against today's API would be rewritten within days: the asset model
is about to change (item 6 needs one asset list, not two tables), and job kinds
must generalise beyond sheets (item 5 needs tiles and images). Porting the UI
twice is the expensive order.

So: one migration and the API contract land first (about a day), then React is
built once against the final shape. Item 1 still ships early - it is second, not
sixth.

### D2. SDXL LoRA at 768 px is the training target. Not Flux, not Qwen.

| Candidate | Verdict |
|---|---|
| **SDXL LoRA, 768 px** | **Chosen.** Base already cached. 12 GB is the documented working minimum at 768 with bf16 + gradient checkpointing + Adafactor or AdamW8bit, rank 32-64, 10-30 images. Mature tooling, and four pixel-art SDXL LoRAs are already local as baselines to beat. |
| Flux.1-dev LoRA | Deferred, not rejected. Trainable in 12 GB via FluxGym/ai-toolkit, but slow enough that iteration suffers. Revisit once SDXL proves the loop and the dataset is known-good. |
| SD 1.5 LoRA | Fallback. Trains comfortably, but quality is a step down and SDXL already fits. |
| Qwen-Image-Edit-2511 (~20 B) | Not trainable here at any setting. **Keeps its job unchanged** as the untrained pose editor - it is good at it, and the turnaround/action stages depend on it. |

Training targets the **concept** stage only. The pose, pixelation, palette and
sheet stages stay deterministic. This is deliberate: it keeps the trained
component small, and keeps every stage that already works out of the blast
radius.

### D3. Measurement first, training second - they solve different problems

Stated plainly because it sets expectations:

- **Palette, cell grid, colour count, outline width and camera elevation are
  measured** from references and applied as hard constraints. Three examples are
  enough. This lands in phase 3 and needs no GPU.
- **Style** - line quality, shading idiom, how a face reads at 48 px - is what
  training buys, and it needs 20+ consistent examples.

`QWEN_ISO_ELEVATION` currently defaults to `eye`, chosen in 0005 because
`elevated` cropped legs. That fixed a symptom without reference to something2's
actual projection. A ground tile's width:height ratio *is* the projection angle,
so `reference-tile` resolves this by measurement. This is likely the single
largest correctness win in the plan, and it costs no training.

### D4. One GPU lease

Training and generation cannot share 12 GB. The queue must serialise them
rather than discovering the conflict as a CUDA OOM inside an hour-long run. A
training run is a job kind holding an exclusive GPU lease.

### D5. Auth: hashed per-client keys, single-token compatibility retained

`api_keys(id, name, key_hash, scopes, created_at, last_used_at, revoked_at)`.
Bearer token, hashed at rest, scoped (`generate`, `read`, `admin`). The existing
`API_TOKEN` continues to work so something2 does not break mid-migration, but
**the no-op-when-unset default is removed** once keys exist: an unset token
silently disabling auth is a trap.

## Plan

Phases are ordered by dependency. Each ends in something demonstrable.

### Phase 0 - unblock (half a day)
- [x] Settings tab renders.
- [ ] Drop `mesh_worker` / `rig_worker` from compose; delete their images.
- [ ] Split `tasks.py`: extract the sheet-build orchestration and the legacy
      A1111 path into modules. Mechanical, test-covered, no behaviour change.

### Phase 1 - data model and API (1 day)
- [ ] Migration 013: `api_keys`, `reference_assets`, `style_profiles`,
      `training_runs`; add `kind` to `jobs`.
- [ ] `GET /api/assets` - one paginated, filterable list over generated images
      *and* job sheets. This is what makes item 6 possible.
- [ ] Key management endpoints + hashed verification.
- [ ] Generalise `JobSpec` to `kind: sheet|core|tile|train`.

### Phase 2 - React (2 days)
- [ ] `frontend/` - Vite + React + TypeScript. Dev server proxies `/api` to
      8001; production build emitted to `static/dist` and served by FastAPI, so
      deployment stays one container.
- [ ] Port Core Generator, Spritesheet Generator, Settings, Gallery.
- [ ] Typed API client generated from the OpenAPI schema FastAPI already emits -
      so a backend change breaks the build, not the page.

### Phase 3 - reference tabs and measurement (1.5 days)
- [ ] Three tabs, one upload component, `kind` differing.
- [ ] On upload, measure and display: tiles -> projection ratio and implied
      elevation; sprites -> cell grid, palette, colour count, outline width;
      cores -> isolation verdict via the existing `concept.judge`.
- [ ] "Derive style profile" writes palette + cell + elevation, and the
      generator consumes it.
- [ ] **Checkpoint: regenerate one character against a derived profile and
      compare to the same character without it.** If measurement alone closes
      most of the gap, the training phase gets cheaper.

### Phase 4 - training (2-3 days, mostly unattended)
- [ ] `train_lora.py`: SDXL LoRA, 768 px, bf16, gradient checkpointing,
      Adafactor, rank 32, trigger token, templated captions.
- [ ] Training as a queued job kind under the GPU lease, with progress and loss
      streamed to the same job UI as generation.
- [ ] Fixed evaluation prompt set; before/after contact sheet against the four
      existing pixel-art LoRAs as baselines.
- [ ] Trained LoRA becomes a selectable option in core generation.

### Phase 5 - tiles and gallery (1 day)
- [ ] Tile generation path, honouring the measured projection.
- [ ] Gallery over `/api/assets`: filter by kind, source job, date; delete.

## Risks

- **12 GB is the ceiling, not the comfort zone.** If 768 px SDXL training OOMs,
  the ladder down is rank 32 -> 16, then 768 -> 640, then SD 1.5. Recorded now
  so the fallback is not invented mid-failure.
- **Dataset quality dominates.** 20 inconsistent examples produce a worse LoRA
  than none. The reference tabs measure consistency and will say so before a
  run is queued.
- **`tasks.py` refactor.** Incremental, one extraction per commit.
- **Training does not fix geometry.** If the camera angle is wrong, a style
  LoRA will faithfully reproduce the wrong angle. Phase 3 before phase 4 is not
  negotiable.

## What this does not do

No multi-user accounts (keys are per-client, not per-person). No cloud training
fallback. No change to the Qwen pose/turnaround stages, which work.
