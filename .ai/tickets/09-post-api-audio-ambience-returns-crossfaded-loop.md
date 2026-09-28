# `POST /api/audio {"kind":"ambience"}` returns a 20-45 s crossfaded loop

## What To Build

The second artefact kind: a stationary texture from Stable Audio Open 1.0,
through the same task, facade, ledger, Activity row and tab as music -
crossfade-only mastering (no bpm, no bar cut), with ambience entries in the
roster. Replaces ticket 04's `501` branch.

## Blocked By

02, 04.

## Scope

- `audio_styles.STYLES` ambience entries (`kind="ambience"`): `forest`,
  `cave`, `village_day`, `night`, `rain`; template + `negative` list
  (`vocals, music, melody, singing`), duration bounds 20-45, default 30.
  `GET /api/audio/styles?kind=ambience` filters.
- `generate_audio_task` ambience branch: evict `pipes`, preflight VRAM, load
  `StableAudioPipeline` (fp16) under an `audio:ambience` key, generate with
  `negative_prompt`, `audio_end_in_s=duration_s`; `audio_master.crossfade_loop`
  only; `loop_start=0`, `loop_end=len`; write WAV + OGG with tags; drop from
  `pipes`, `release_vram_cache`. Same breaker wrapping as music.
- `POST /api/audio` removes the `501`; `duration_s` validated per kind.
- Tab: enable `ambience` in the kind selector; the form hides bpm-related
  slots for it.
- `verify-audio-api.py --submit --kind ambience`: `200`, OGG 20-45 s,
  `read_loop_tags` == `info`, second call cached; both kinds for one name
  coexist (`GET /api/audio/music/x` and `/ambience/x` both `200`).
- `smoke-audio.py`: ambience roster entries render with the negative list.

## Out Of Scope

Layering music + ambience here (something2 does it), per-area ambience,
any change to the music path.

## Acceptance Criteria

- [ ] `verify-audio-api.py --submit --kind ambience` passes.
- [ ] Loop audition in the tab: no audible click at the seam on the three spike prompts.
- [ ] `make gpu-health` OK afterwards and a music generation still works after an ambience one (eviction across the two audio models).
- [ ] 0010 measured row for ambience has the in-product numbers if they differ from the spike.

## Test Seam

`POST /api/audio` with `kind=ambience`, via `verify-audio-api.py`.

## Verification

    SPRITE_API_KEY=... python scripts/verify-audio-api.py --submit --kind ambience
    make gpu-health

## Implementation Notes

- The model is gated: ticket 02's one-time fetch must have happened on this
  host; `HF_HUB_OFFLINE=1` stays set.
- Two audio models plus SDXL never coexist: the `pipes` dict holds one.

## Review Focus

That the ambience branch shares the music branch's breaker/eviction code
rather than copying it, and that `duration_s` bounds are per kind.

## Suggested Route

`/implement`, then `/review-code`.

## Status 2026-09-28 (read from code; no box ticked without the named check)

The ambience path runs end to end (0010 "In product": 37-41 s cold, 0.07 s cached, seams 0.77-2.38 dB, gpu-health OK). Unchecked: the verifier run named in the criteria, the tab audition (07 not built), and music-after-ambience eviction (music cannot run - ticket 15).
