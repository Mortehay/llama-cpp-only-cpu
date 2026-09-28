# The fixed 10-track loop corpus exists and 0010 is marked accepted with numbers

## What To Build

The acceptance artefact the contract's bar refers to: ten music tracks (5
styles x 2 seeds) and five ambience clips generated through the product
path, kept with their seam metric and a per-track listening verdict, so any
later change to mastering, roster or model is a before/after against the
same set. Then 0010 moves from *proposed* to *accepted*.

## Blocked By

07, 09.

## Scope

- `scripts/build-audio-corpus.py`: generates the set through `POST
  /api/audio` with fixed names (`corpus-<style>-<seed>`), copies the OGG +
  WAV to `IMAGES_DIR/audio-corpus/`, writes `corpus.json` rows: `kind`,
  `style`, `seed`, `bpm`, `time_signature`, `duration_s`, `loop_start`,
  `loop_end`, `seam_rms_jump`, `seconds_generated`, `served_from`,
  `verdict` (filled by hand: `inaudible` | `audible` | `click`), `vocals`
  (bool), `note`.
- Listening pass in the Audio tab with the Seam button, verdicts written
  into `corpus.json` by hand.
- `.ai/specs/audio/corpus.md`: the Markdown summary (counts per verdict,
  vocal leaks, mean/max seam metric, generation times) - per-row data stays
  in the JSON, the `reference-audit.md` pattern.
- 0010: status -> accepted; the acceptance section records the pass/fail
  against the bar (>= 8/10 inaudible, 0 vocals on the default style, all
  music >= 60 s); if the bar is missed, 0010 says so and names the next
  dial (crossfade length, sft + CFG, or a different style template) as an
  open decision rather than quietly lowering the bar.
- `contract.md` placeholders replaced by measured defaults/max.

## Out Of Scope

Tuning to pass the bar (a separate, measured change if needed); a listening
UI beyond what 07 has.

## Acceptance Criteria

- [ ] `corpus.json` has 15 rows with verdicts, and the files exist.
- [ ] `corpus.md` summarises them; numbers match the JSON.
- [ ] 0010 is `accepted` or explicitly `accepted with a named gap`, with the contract's placeholders gone.
- [ ] `CLAUDE.md` gets one tripwire line if the corpus revealed one (e.g. "a rhythmic style loops audibly - do not judge the mastering on ambient pieces").

## Test Seam

Not applicable - the corpus is the fixture other work tests against.

## Verification

    SPRITE_API_KEY=... python scripts/build-audio-corpus.py
    # listen, fill verdicts, then:
    python scripts/build-audio-corpus.py --summarise > .ai/specs/audio/corpus.md

## Implementation Notes

- Generate the corpus once the roster is frozen for v1; changing a template
  afterwards invalidates the row for that style, and the JSON should say
  which roster version produced it (a hash of `STYLES` is enough).
- Fixed seeds are the point (the entity-cutout work used 12 fixed seeds); a
  re-roll to make the numbers look better is the thing this ticket exists
  to prevent.

## Review Focus

That verdicts are recorded per row, not summarised into a feeling, and that
a missed bar is written as missed.

## Suggested Route

`/implement`, then `/review-code` on 0010 and `corpus.md`.

## Status 2026-09-28

`scripts/build-audio-corpus.py` written: 10 music (5 styles x seeds 11, 42)
+ 5 ambience through the real authenticated `POST /api/audio`, fixed names
(`corpus-<style>-<seed>`), newest file per name copied to `audio/corpus/`,
`corpus.json` with EMPTY verdict/vocals/note fields - preserved across
re-runs, never invented. Waits on 503 building/busy per Retry-After.

- [ ] Run it (needs the same key as ticket 10; ~15-20 min of GPU).
- [ ] Owner's listening pass fills the verdicts; then `corpus.md` and 0010
      -> accepted.
