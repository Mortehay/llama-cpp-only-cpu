# `context` picks a style through the LLM, with a rules fallback and a recorded `author`

## What To Build

`POST /api/audio` accepts `context` (a map/level description); the LLM on
`llm-server` selects a roster entry and fills its slots; the result is
reproducible and the response says what happened. A non-empty `prompt`
bypasses it. The Audio tab gets a "Propose from context" button.

## Blocked By

04, 07.

## Scope

- `audio_styles._llm_style_plan(context, kind) -> (style_id, slots, author)`
  on the `worlds._llm_biome_plan` pattern: `LLM_URL`, `_llm_model()` reuse
  (or a shared helper if lifting it is small), `POST /v1/chat/completions`
  with a system prompt that lists the roster ids and slot ranges and demands
  JSON; validate against the roster; an invented style is **dropped and
  named** in `author` ("dropped invented style(s): sea_shanty"); out-of-range
  slots clamped and named; cold-start (`--sleep-idle-seconds 120`) retried
  once like worlds does.
- `audio_styles._rules_style_plan(context, kind)`: keyword table
  (`tavern|inn|ale` -> tavern; `mine|cave|crypt|dungeon` -> dungeon;
  `battle|siege|war` -> battle; `village|farm|market` -> village; else
  default). Used when the LLM is unreachable, times out, or returns nothing
  usable; `author` says so.
- `POST /api/audio`: `context` field; precedence `prompt` > (`style` given
  explicitly) > `context` via LLM > default. `info.author` and the ledger
  `params.author` record the outcome; `info.prompt` is the rendered prompt
  actually sent.
- Tab: "Propose from context" textarea + button -> `POST
  /api/audio/propose` (scope `read`; no GPU) returning `{style, slots,
  author, prompt}` and filling the form; generation is still the user's
  click.
- `smoke-audio.py`: 6 fixed contexts -> expected rules-fallback styles; the
  LLM path is covered by `verify-audio-api.py --llm` (non-deterministic;
  asserts only that `author` names the model and the style is in the roster).

## Out Of Scope

The LLM writing prompts free-form; per-map memory of past choices; any
change to `worlds.py` beyond a shared helper if extracted.

## Acceptance Criteria

- [ ] `POST /api/audio/propose {"context":"abandoned dwarven mine, danger"}` returns a roster style (LLM's choice, named in `author`) or the rules choice `dungeon` with `author` saying the LLM was unavailable.
- [ ] The same `context` with the LLM bypassed (`prompt` set) renders the same prompt twice.
- [ ] `smoke-audio.py` rules cases pass; `verify-audio-api.py --llm` passes with `llm_engine` running.
- [ ] With `llm_engine` stopped, generation still works and `author` records the fallback.

## Test Seam

`audio_styles._rules_style_plan` (smoke) and `POST /api/audio/propose`
(verifier).

## Verification

    docker exec sprite_generator python /app/scripts/smoke-audio.py
    SPRITE_API_KEY=... python scripts/verify-audio-api.py --llm
    docker compose ... stop llm-server && SPRITE_API_KEY=... python scripts/verify-audio-api.py --llm --expect-fallback

## Implementation Notes

- `llm-server` shares the card; a music request right after a propose pays
  the LLM's sleep and the audio model's load. Expected; documented in 0010.
- Never let the LLM emit the model prompt directly - the roster template
  renders it.

## Review Focus

Validation of the LLM's JSON (no crash on garbage), the drop-and-name path,
and that `author` is honest when the fallback ran.

## Suggested Route

`/implement`, then `/review-code`.
