# Style roster and loop mastering pass a no-GPU smoke

## What To Build

The two pure-Python modules everything else calls, plus the smoke that proves
them without a GPU: `audio_styles.py` (the roster, contract "Style is a
roster, not a string") and `audio_master.py` (bar-aligned cut, crossfade,
seam metric, WAV/OGG writing with loop tags).

## Blocked By

01 (settles whether the model hands back a WAV path or a tensor, and the
sample rate; the OGG/loop-tag tooling choice follows from it).

## Scope

- `src/sprite_generator/audio_styles.py`
  - `STYLES`: dict keyed by roster id. Music entries: `medieval_fantasy`
    (default), `tavern`, `dungeon`, `battle`, `village`. Each: `kind`
    (`music`), `template` with slots `{tempo_bpm}`, `{mood}`, `{featured}`,
    per-slot allowed values/ranges, a fixed `time_signature`, `negative`
    list, `lyrics="[Instrumental]"`.
  - `DEFAULT = "medieval_fantasy"`, `render(style_id, **slots) -> dict` with
    `prompt`, `negative`, `lyrics`, `bpm`, `time_signature`; unknown style ->
    `KeyError`, out-of-range slot -> clamped and reported.
  - `roster() -> list` in the shape `GET /api/audio/styles` will return
    (`id`, `label`, `kind`, `default`, slot metadata). Mirrors how
    `core_models.CORE_MODELS` is consumed by the UI.
- `src/sprite_generator/audio_master.py` (numpy only; no torch import - the
  API process imports it):
  - `bar_seconds(bpm, time_signature)`.
  - `cut_to_bars(samples, sr, bpm, time_signature, min_s) -> (samples, bars)`:
    largest whole-bar count `>= min_s`, else raise with a message that says
    what length was available.
  - `crossfade_loop(samples, sr, ms) -> samples`: equal-power blend of the
    final `ms` into the head, so a `loop_end -> loop_start` jump is
    continuous.
  - `seam_rms_jump(samples, sr, window_ms) -> float`: RMS of the last window
    vs the first window, as the metric the corpus records.
  - `write_wav(path, samples, sr)`, `write_ogg(path, samples, sr, loop_start,
    loop_end)`: OGG Vorbis with `LOOPSTART` / `LOOPLENGTH` vorbis comments
    (sample offsets). Tooling: `soundfile` if the image's libsndfile writes
    Vorbis, else `ffmpeg` (apt) - decide here, add to
    `requirements.cuda.txt` / `Dockerfile.cuda`, and note the rebuild.
  - `read_loop_tags(path) -> (loop_start, loop_end)` for the smoke and the
    verifier.
- `scripts/smoke-audio.py`, runnable as
  `docker exec sprite_generator python /app/scripts/smoke-audio.py`, in the
  `smoke-world-gen.py` style (numbered cases, `ok`/`FAIL`, exit code):
  - every roster entry renders and contains `[Instrumental]` and a
    `no vocals` negative;
  - `bar_seconds` for (120, "4"), (90, "3"), (140, "6") matches hand values;
  - `cut_to_bars` on 125 s at 120 bpm 4/4 returns 62 bars = 124.0 s;
  - a synthetic 2-tone signal: `seam_rms_jump` above threshold before
    `crossfade_loop`, below after;
  - `write_ogg` + `read_loop_tags` round-trips the offsets exactly;
  - the WAV and OGG decode to the same length within one frame.

## Out Of Scope

Any model call, the facade, the LLM, ambience entries (ticket 09 adds them
to the same roster).

## Acceptance Criteria

- [ ] `smoke-audio.py` passes inside the running API container with no GPU involved.
- [ ] `render("medieval_fantasy")` with default slots produces a prompt a human would recognise as the default template from the contract.
- [ ] `write_ogg` output plays in a browser `<audio>` element (manual, once).
- [ ] If new apt/pip deps were added, the worker image was rebuilt and `make gpu-health` still OK.

## Test Seam

`audio_styles.render`, `audio_styles.roster`, and the `audio_master`
functions, via `scripts/smoke-audio.py`.

## Verification

    docker exec sprite_generator python /app/scripts/smoke-audio.py

## Implementation Notes

- Keep `audio_master` free of torch so the API can compute `seam_rms_jump`
  for the verifier without a CUDA context.
- Files land flat in `IMAGES_DIR` as `audio_<kind>_<name>_<uuid>.{wav,ogg}`
  because `generations._url` takes a basename.

## Review Focus

Whether the crossfade actually blends tail into head (a common bug fades the
tail to silence instead), and whether loop tags are sample offsets, not
seconds.

## Suggested Route

`/implement`, then `/review-code`.

## Status 2026-09-28 (read from code; no box ticked without the named check)

Code exists: `audio_styles.py` (5 music + 5 ambience entries), `audio_master.py`, `scripts/smoke-audio.py`. libsndfile/soundfile/mutagen are in both images (0010). Not re-run this session. `audio_paths` now writes under `<AUDIO_DIR>/<kind>/`.
