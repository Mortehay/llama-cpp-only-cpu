# A track can be generated, auditioned at the seam, and downloaded from the Audio tab

## What To Build

The local testing surface: a new `Audio` tab (`#audio`) that drives
`POST /api/audio` by hand, lists finished audio, plays it, auditions the loop
seam with Web Audio, and downloads OGG or WAV. A `503` during a long build
reads as "building", not as an error.

## Blocked By

05, 06.

## Scope

- `frontend/src/tabs/Audio.tsx`, registered in `App.tsx` `TABS` after
  `worlds` as `{ id: 'audio', label: 'Audio' }`; `Worlds.tsx` is the sibling
  to copy for layout and state handling.
- Form: `kind` (music | ambience - ambience disabled with a note until
  ticket 09 lands), `name` (map name; free text), `style` from
  `GET /api/audio/styles` (roster; default preselected), slot controls the
  roster exposes (tempo band, mood, featured), `seed` (-1 = random),
  `duration_s` within the roster/contract bounds, and a collapsed "prompt
  override" textarea that, when non-empty, bypasses the style.
- Submit -> `api.audio.generate(body)`. On `200` show the result; on `503`
  with `reason: building` show a "building - N s" state and poll
  `GET /api/audio?kind=&name=` every 5 s until a `done` row appears; on
  `reason: gpu_faulted` / `busy` show the message and `Retry-After`.
- List: `GET /api/audio?limit=50` newest first, each with `<audio controls
  preload="none">`, `style`, `duration_s`, `bpm`, `seed`, `seam_rms_jump`,
  `served_from`, `principal_name`.
- **Loop audition**: a "Loop" button per row that decodes the OGG with
  `AudioContext.decodeAudioData` (fetched with the bearer via
  `fetchObjectUrl` or a `fetch` -> `arrayBuffer`), plays an
  `AudioBufferSourceNode` with `loop = true`, `loopStart` / `loopEnd` from
  `info` converted to seconds; and a "Seam" button that starts playback 3 s
  before `loopEnd` so the join is heard immediately. One source at a time;
  stop on tab change.
- Download OGG / WAV: `fetchObjectUrl('/api/audio/<kind>/<name>?master=1')`
  -> anchor with `download` attribute.
- `api.ts`: `api.audio.styles()`, `.generate()`, `.list()`, `.info()`;
  types for the response.
- Rebuild with `scripts/build-frontend.sh`.

## Out Of Scope

LLM "propose" (08), ambience generation (09), editing or deleting tracks,
waveform display.

## Acceptance Criteria

- [ ] From `#audio`, generating `music` for a new name produces a row that plays.
- [ ] "Seam" audibly plays the join; "Loop" plays continuously across it without the page reloading the file.
- [ ] Both downloads deliver a file of the right type (OGG plays, WAV is larger and the master).
- [ ] With `AUDIO_GENERATE_TIMEOUT_S=1` on the API, the tab shows "building" and then the finished row without a manual refresh.
- [ ] A `401` (no token in Settings) shows the same guidance other tabs show.

## Test Seam

Not applicable - UI over the routes 04/05 verify. Manual browser check.

## Verification

    bash scripts/build-frontend.sh
    # open http://<host>:8001/#audio with a token set in Settings

## Implementation Notes

- The `<img>` 401 lesson: `<audio src="/images/...">` is fine (open mount);
  anything under `/api/audio/` must be fetched with the bearer.
- `loopStart`/`loopEnd` are sample offsets in `info`; divide by
  `info.sample_rate` for Web Audio.
- Keep the `AudioContext` singleton at module scope; browsers block
  autoplay, so create it on first click.

## Review Focus

Polling backoff (no tight loop on `503`), a single active audio source, and
that the bearer never leaks into an `<audio src>` query string.

## Suggested Route

`/implement`, then `/review-code`. Optionally `human-ui-designer` for the
form layout if it grows past the Worlds pattern.

## Status 2026-09-28

Implemented: `frontend/src/tabs/Audio.tsx`, registered as `#audio` after
Worlds; `audioApi` appended to `api.ts` (kept out of the `api` object, which
another session was editing). Form: kind, style (roster), every slot the
roster exposes, seed, seconds, collapsed prompt override. **The name is
pre-filled fresh** (`test-<kind>-<stamp>`) because the endpoint is cache-first
by name - a repeated name returns the old take, which read as "the button does
nothing". 503 `building` is shown as "still building" and the list polls every
5 s while any row is running; `busy`/`gpu_faulted` are shown with Retry-After
and never auto-retried. Each take: `<audio>`, Loop (Web Audio, loop points
converted from samples to seconds), Seam (starts 3 s before loop end), OGG and
WAV downloads from the open `/audio/` mount (no bearer needed). Music shows a
note that it cannot run until ticket 15.

Verified: `tsc -b` (strict) + vite build pass. One ambience take generated
through the worker with the new house-style roster (`smooth-forest-check`,
72.5 s cold, seam 0.29 dB); its OGG and WAV serve 200 from `/audio/`;
`gpu-health` OK afterwards. **Not verified: clicking through the tab in a
browser** (Generate, Loop, Seam, downloads) - no box ticked until that.
