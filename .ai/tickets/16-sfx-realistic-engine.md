# `POST /api/audio {"kind":"sfx"}` returns a one-shot cue from the realistic engine

## What To Build

The third kind (0010 D7/D8, contract "`sfx`"): a cue roster, the engine
precedence, and the `realistic` engine on Stable Audio Open 1.0 with one-shot
mastering. No loop.

## Blocked By

None for the engine itself (SAO 1.0 is on disk and proven). Planning must
first settle the three contract "Undecided" items: name shape, variants per
cue, one cue vs a pack per request.

## Scope

- `audio_styles`: `kind="sfx"` entries - start with `slash`, `hit`,
  `pickup`, `spell`, `footstep`, `ui_click` - each with a `realistic` recipe
  (prompt, negative), an optional `retro` recipe (ticket 17), and a default
  engine.
- Engine resolution in one function: request `engine` > world `sfx_engine`
  from `<WORLDS_DIR>/<world>.gen.json` > cue default > `realistic`; returns
  `(engine, engine_from)`; refuses (422) an engine the cue has no recipe for.
- `audio_master`: one-shot mastering - trim leading silence to a fixed
  threshold, short fade-out, peak/loudness normalise. No loop tags.
- `audio_engine`: `sfx` branch reusing the ambience pipeline load;
  `DURATION_BOUNDS["sfx"] = (0.1, 3.0)`; output under `audio/sfx/`.
- `generations.KINDS += ("sfx",)`; cache key includes `engine`.
- `smoke-audio.py`: precedence table (every level, and the refusal), onset
  trim on a synthetic clip with leading silence.

## Acceptance Criteria

- [ ] `verify-audio-api.py --submit --kind sfx` passes for three cues.
- [ ] Onset measured: first sample above threshold within 10 ms of file start, on every cue.
- [ ] Listening pass on sub-second cues recorded; if SAO 1.0 is poor there, that is the trigger to evaluate SAO Small (0010 D7), not a reason to tune blindly.
- [ ] `info.engine_from` correct for each precedence level.

## Suggested Route

`/plan-feature` for the three undecided items, then `/implement`.

## Status 2026-09-28

Implemented in the audio worktree (`feat/audio-generation`): cue roster and
`resolve_engine` precedence (`audio_styles`), one-shot mastering and
`sfx_paths` (`audio_master`), `generate_sfx` one-load batches
(`audio_engine`), `tasks.generate_sfx_task` (row per cue, breaker-wrapped),
`POST /api/audio/sfx` + `/api/audio/sfx-pack`, `GET /api/audio/styles?kind=sfx`,
`sfx_engine` on `WorldSpec`/`WorldEdit`, and an sfx panel on the Audio tab
(single cue, pack builder, per-variant players). Owner decisions: name
`<cue>/<entity>`, caller chooses 1-5 variants, pack endpoint too.

- [x] Onset: 0-5 ms on every variant measured (bar <= 10 ms).
- [x] `info.engine_from` correct per level: `cue` and `world` exercised
      through the facade; `request` and the refusals in smoke + facade.
- [x] Facade end to end through Celery (temporary worktree worker): build,
      cache hit, fewer-variants cache hit, world precedence, 422/404
      refusals, pack with a mix of cached and new cues.
- [ ] `verify-audio-api.py --submit --kind sfx` - the verifier has no sfx
      mode yet, and there is no bearer in this session.
- [ ] **Listening pass on sub-second cues** - the owner's call; files are in
      `audio/sfx/realistic/` and `audio/sfx/_experiment/`.
- [ ] Not deployed: the running stack mounts the shared checkout, which does
      not have this code until the branch is merged there.

Findings: the pipeline's full 47.6 s denoise window and its batched decode
(GPU fault) - 0010 "sfx, measured". smoke-audio 18/18.
