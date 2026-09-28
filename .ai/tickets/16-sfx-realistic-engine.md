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
