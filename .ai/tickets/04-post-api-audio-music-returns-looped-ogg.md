# `POST /api/audio {"kind":"music"}` returns a looped OGG and the request is a ledger row

## What To Build

The first end-to-end path: a bearer-authenticated `POST /api/audio` for
`kind=music` either cache-reads a finished track by name or builds one in the
solo worker (evicting SDXL, running ACE-Step, mastering, writing WAV + OGG),
records the request in `generations`, and returns `audio[0]` + `info` per
`.ai/specs/audio/contract.md`. Plus the read routes and a verifier script.

## Blocked By

01, 03.

## Scope

- `generations.KINDS += ("music", "ambience")`.
- `tasks.py: generate_audio_task(kind, name, style, prompt, negative, lyrics,
  bpm, time_signature, seed, duration_s, gen_id)`:
  - refuse via `gpu_fault_block_reason()` like the other GPU tasks;
  - evict everything in `pipes`, `gc.collect()`, `torch.cuda.empty_cache()`;
    preflight `torch.cuda.mem_get_info()` against `AUDIO_GPU_BUDGET` (env,
    default from ticket 01's number) and refuse with a message if short;
  - load ACE-Step turbo in the shape 01 decided (in-process cached in
    `pipes` under an `audio:` key, or venv subprocess);
  - generate with `[Instrumental]`, then `audio_master.cut_to_bars` +
    `crossfade_loop`, write `audio_music_<name>_<uuid>.wav` and `.ogg` flat in
    `IMAGES_DIR`;
  - drop the model from `pipes` (so the next SDXL load has the card),
    `release_vram_cache("audio")`;
  - `except` path uses `is_cuda_fault` / `trip_gpu_breaker` exactly as
    `generate_raw_task` does, returning `{"error", "gpu_faulted": True,
    "retry_after_s"}`;
  - on success returns `{file_path, master_path, seed, duration_s,
    sample_rate, loop_start, loop_end, seam_rms_jump, bpm, time_signature}`;
  - closes its own ledger row via `gen_id` (`generations.finish` /
    `generations.fail`) - ticket 05 relies on this.
- `src/sprite_generator/audio.py` router, included from `main.py`:
  - `POST /api/audio` (scope `generate`): validate `kind` in
    (`music`, `ambience`) - ambience returns `501` until ticket 09; resolve
    `style` (default roster entry) and `prompt` override; **cache first**:
    newest `done` row for (`kind`, `lower(name)`) whose `file_path` exists ->
    `200`, `served_from: cache`, via `generations.record`; otherwise
    `generations.begin(kind, name, route="/api/audio", prompt, model,
    seed, params, caller)`, `_long_job_ahead()`-style refusal if a non-audio
    job holds the worker (recorded as failed activity, `503`), queue the
    task, `attach_task`, block with `AsyncResult.get(timeout=
    AUDIO_GENERATE_TIMEOUT_S)`; on success re-read the row and return the
    payload. The overshoot branch is ticket 05 - here it may temporarily
    behave like the tile facade, but the row-closing contract above is built
    now.
  - `GET /api/audio?kind=&name=&limit=` (scope `read`): ledger rows for
    audio kinds with `url`, `master_url`, `info`.
  - `GET /api/audio/{kind}/{name}` (`?master=1`) -> `FileResponse`, `404` +
    JSON `{"kind","name","detail"}` when no done row.
  - `GET /api/audio/{kind}/{name}/info`, `GET /api/audio/styles`.
  - Response `info` fields exactly as the contract lists, with `seed` the one
    actually used and `sample_rate` whatever the model emitted.
- `scripts/verify-audio-api.py --submit` (bearer from `SPRITE_API_KEY`):
  miss -> `200`, `audio[0]` decodes to an OGG whose `read_loop_tags` equals
  `info.loop_start/loop_end`, duration within 2 s of requested bars; second
  call -> `cached: true` under 500 ms; `GET` unknown name -> `404` with the
  JSON body; `GET /api/audio/styles` lists the default.
- Env: `AUDIO_GENERATE_TIMEOUT_S` (default from 01's warm number + load, just
  under what something2 will be told), `AUDIO_GPU_BUDGET`.

## Out Of Scope

Overshoot semantics (05), UI (06, 07), LLM (08), ambience generation (09).

## Acceptance Criteria

- [ ] `verify-audio-api.py --submit` passes end to end against the running stack.
- [ ] The Activity tab lists the request with `principal_name` while it runs and `done` after (playback is ticket 06).
- [ ] `make gpu-health` after the build reports OK, and an SDXL txt2img (the 2026-09-12 check: base + `nerijs/pixel-art-xl`, 512, 20 steps) still succeeds afterwards - eviction works in both directions.
- [ ] A second `POST` for the same name never touches the GPU (worker log shows no load).
- [ ] A `gpu_faulted` task result becomes `503` + `Retry-After` at the API and is never re-queued.

## Test Seam

`POST /api/audio` and the `GET` routes, via `scripts/verify-audio-api.py`;
`generate_audio_task` is exercised only through them.

## Verification

    make gpu-health
    SPRITE_API_KEY=... python scripts/verify-audio-api.py --submit
    make gpu-health
    # then one SDXL txt2img through /sdapi/v1/txt2img

## Implementation Notes

- Copy `_serve_tile` for the request flow and `generate_raw_task` for the
  task's breaker wrapping. Read both fully first.
- The API process is `COMPUTE_DEVICE=cpu`; nothing in `audio.py` may import
  torch.
- `tasks.py` must not touch CUDA at import - the ACE-Step import goes inside
  the task function.
- Ambience `501` is deliberate: the route exists so the kind validation and
  the contract are one place, and 09 fills the branch.

## Review Focus

Eviction in both directions; that the ledger row is closed by the task, not
only by the request; that `seed` in `info` is the seed used; that nothing is
retried on `gpu_faulted`.

## Suggested Route

`/implement`, then `/review-code`.

## Status 2026-09-28 (read from code; no box ticked without the named check)

Router, task and engine exist, and the ambience half is proven in product (0010 "In product"). **The music half cannot run**: `audio_engine._generate_music` needs `/opt/acestep/.venv`, which no product image has - ticket 15. No acceptance box can be ticked for `kind=music` until then.
