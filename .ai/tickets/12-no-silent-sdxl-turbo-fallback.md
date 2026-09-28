# No caller silently gets SDXL-Turbo when it omits the model

## What To Build

Every place that falls back to `stabilityai/sdxl-turbo` when `llm_name` is
missing uses the roster instead. Turbo was dropped from `core_models.CORE_MODELS`
(distilled, guidance 0, lost the 2026-08-21 bench) but kept being served by
defaults. See `.ai/decisions/0011` "The finding that reshaped the ask".

## Blocked By

None.

## Scope

- [x] `main.py` `/api/warm` and `/api/generate_core` form defaults ->
  `core_models.default_model()` (done 2026-09-27).
- [x] `static/js/generator.js` no longer hardcodes turbo when the model select
  is absent; it omits the field so the server default applies (done 2026-09-27).
- [ ] `main.py` `/api/generate_sheet` default and the retry fallback
  (`llm_actual = ... else "stabilityai/sdxl-turbo"`). **Not `default_model()`**:
  the sheet task is step 2, which is locked to SD1.5 (ControlNet openpose), and
  the roster default is SDXL. Decide the step-2 default first - the code says
  `Onodofthenorth/SD_PixelArt_SpriteSheet_Generator` was measured, the roster
  has `PublicPrompts/All-In-One-Pixel-Model` - then split the retry fallback by
  `image_type` (core -> `default_model()`, sheet -> the step-2 default).
- [ ] Count how much traffic actually hit turbo:
  `SELECT llm_name, count(*) FROM sprite_images GROUP BY 1 ORDER BY 2 DESC`,
  and `generations.model` likewise. Record the figure here.

## Out Of Scope

Removing turbo support from `get_sd_pipeline` (an explicit request may still
name it); the distilled-guidance clamp in `resolve_sampling_params`.

## Acceptance Criteria

- [ ] `grep -rn "sdxl-turbo" src/sprite_generator/{main.py,static,templates}`
  returns only comments.
- [ ] `POST /api/generate_core` with no `llm_name` queues the SDXL + pixel-art
  LoRA default, visible in the worker log line naming the model.
- [ ] A retry of a legacy row with `llm_name='Unknown'` does not load turbo.

## Test Seam

The endpoint's recorded `llm_name` in `sprite_images` for a request that omits it.

## Verification

    docker compose ... restart sprite-generator   # .py does not hot-reload
    curl -H "Authorization: Bearer $KEY" -F prompt="pixel art knight" http://localhost:8001/api/generate_core
    # then check the row's llm_name / worker log

## Suggested Route

`/implement` for the remaining two boxes, then `/review-code`.
