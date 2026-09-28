# The `retro` sfx engine renders cues with sfxr-style synthesis, no GPU

## What To Build

The second sfx engine (0010 D7): procedural 8-bit synthesis. Each cue's
`retro` recipe is a preset plus parameter ranges; the rules (and later the
brain) pick values inside them. Runs in the API or worker process in
milliseconds, deterministic from the seed.

## Blocked By

16 (roster shape, precedence, one-shot mastering, `sfx` kind).

## Scope

- Decide first: vendor an existing Python sfxr port (license must be
  permissive - check before copying a line) or write a small one (square /
  saw / sine / noise, envelope, frequency slide, vibrato, low/high-pass).
- `retro` recipes for the cues where 8-bit makes sense (`hit`, `pickup`,
  `spell`, `ui_click`, `slash`); `footstep` may have none - then the
  precedence refusal applies.
- Same seed + same recipe -> byte-identical WAV (asserted in the smoke).
- No GPU, no eviction: must not go through the solo worker's model path.

## Acceptance Criteria

- [ ] `smoke-audio.py` covers determinism and every retro recipe.
- [ ] `POST /api/audio {"kind":"sfx","engine":"retro","name":"pickup"}` returns in under 1 s with the worker busy on an image job.
- [ ] A world with `sfx_engine: retro` in its `.gen.json` yields retro cues with `engine_from: world`.

## Suggested Route

`/implement`, then `/review-code`.

## Status 2026-09-28

Implemented: `audio_retro.py` - written here, not vendored (no licence
question): per-cue PRESETS of parameter ranges, a seeded draw, and an
sfxr-style synth (square/saw/sine/held-noise, attack/sustain/decay with
punch, exponential slide, arpeggio, vibrato, one-pole low/high-pass,
bit-crush). Recipes: slash, hit, pickup, spell, ui_click. **footstep has
none, on purpose** - refused with the level named. Retro renders INLINE in
the API process, before the worker/busy check; a mixed pack sends only its
realistic cues to the worker. No seed -> seed derived from `<cue>/<entity>`,
so a name is always the same sound. `audio_engine.sfx_ledger_params` is the
one ledger shape for both engines.

- [x] smoke-audio 19/19 in the API image: determinism (byte-identical),
      name-seeding, every recipe audible with onset <= 10 ms, 25 ms/variant
      worst, roster and presets agree.
- [x] `POST /api/audio/sfx {"cue":"pickup","engine":"retro"}` in 218 ms
      end to end (ledger included); cache 143 ms.
- [x] World with `sfx_engine: retro` -> `engine_from: world`.
- [x] Mixed pack (1 cached realistic + 3 retro) in 566 ms, no GPU.
- [ ] "with the worker busy on an image job" - not staged; the retro path
      never reads the worker or the queue, so it cannot wait on it.
- [ ] Listening pass - files in `audio/sfx/retro/`.
