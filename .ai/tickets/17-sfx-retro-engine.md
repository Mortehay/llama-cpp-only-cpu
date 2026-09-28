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
