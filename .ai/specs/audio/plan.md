# Audio: implementation plan

Companion to [contract.md](contract.md) (the surface) and
[decisions/0010](../../decisions/0010-audio-generation.md) (why). This is the
order of work, the seams it hooks into, and what proves each step. Written
2026-09-12 against the code as it is; every `file:symbol` below was read, not
assumed.

Status: **plan, not started.** Slice 0 is a spike whose outcome can change
slices 1-5; do not start slice 1 before slice 0 has filled 0010's measured row.

## Goal, in one sentence

something2 asks `POST /api/audio` for `music:<map>` or `ambience:<map>`, gets
an OGG loop with loop points (or a `503` that tells it when to come back), and
every request, its author, and its result are listenable on the Activity tab
and reproducible from the Audio tab.

## What exists that this reuses (read before writing anything)

| Seam | Where | What it gives audio |
|---|---|---|
| Tile facade | `a1111.py:_serve_tile`, `_long_job_ahead`, `_tile_payload` | cache-first, build-within-budget, refusal-as-activity. **Differs on overshoot**: tile REVOKES the task and returns `504`; audio must NOT revoke - it returns `503` + `Retry-After` and lets the build finish (contract D5). Copy the shape, not the timeout branch. |
| Breaker | `tasks.py:gpu_fault_block_reason`, `trip_gpu_breaker`, `_CUDA_FAULT_MARKERS`; `a1111.py:850-899` for the `503` + `Retry-After` translation | audio tasks wrap generation the same way; a `gpu_faulted` result is never retried |
| Eviction | `tasks.py:get_sd_pipeline` ~1098 (`pipes.clear(); gc.collect(); torch.cuda.empty_cache()`), the qwen-edit preflight at ~3475 (`mem_get_info` before load, refuse if short) | the audio model is loaded through the same `pipes` dict so SDXL is evicted first, and dropped afterwards so SDXL can come back; preflight refuses rather than OOMs |
| VRAM release | `tasks.py:release_vram_cache` | called after each audio generation, as after each image |
| Ledger | `generations.begin / attach_task / finish / fail / record`, `KINDS` at `generations.py:59`; `_url()` maps `file_path` to `/images/<basename>` | `music` and `ambience` are two more `KINDS`; `file_path` set, `job_id` NULL. `assets_v` and `activity_v` do not filter by kind, so rows appear in Activity and Gallery with **no migration** |
| Static media | `main.py` `/audio` mount (was `/images` until 2026-09-28 - see contract "Files on disk"), `NoStoreStaticFiles`, **unauthenticated** | Activity's `Row` renders `item.url` in an `<img>`; audio gets an `<audio controls>` on the same URL. The contract's "fetch with bearer" applies to `/api/audio/...` routes, not to the static mount |
| LLM | `worlds.py:_llm_model`, `_llm_biome_plan` (`POST {LLM_URL}/v1/chat/completions`, `LLM_TIMEOUT`, cold-start retry, invented-and-dropped reporting) | `audio_styles._llm_style_plan` is the same function with a different vocabulary |
| Roster pattern | `core_models.CORE_MODELS`, `trigger_for()`; `GET /api/core-models` | `audio_styles.STYLES`, `GET /api/audio/styles` |
| Tabs | `frontend/src/App.tsx:17-33` (`TABS`, hash routing), `api.ts:fetchObjectUrl`, `Worlds.tsx` as the closest sibling (form + list + preview + download) | the Audio tab |
| Frontend build | `scripts/build-frontend.sh` (node in a container; no node on the host or in WSL) | required after any `.tsx` change |
| No-GPU smoke pattern | `scripts/smoke-world-gen.py` (34 cases, `docker exec sprite_generator python /app/scripts/...`) | `scripts/smoke-audio.py` for the mastering math and the roster |
| Contract verifier | `scripts/verify-jobs-api.py` | `scripts/verify-audio-api.py` |

## Slices

### Slice 0 - the spike (blocking; fills 0010's measured row)

Nothing else is planned on assumed numbers. Runs in a **throwaway image
derived from the worker image**, never in the shared one, because ACE-Step
may pin `transformers`/`torch` and pip will happily downgrade the versions
diffusers and the Qwen paths depend on (worker is transformers 5.17.0, torch
2.14.0+cu130, diffusers 0.40.0 as of 2026-09-12).

1. `compose/develop/sprite_generator/Dockerfile.audio-spike`: `FROM
   llama-cpp-only-cpu-sprite-worker:latest`, `pip install` ACE-Step 1.5 from
   its repo, then `pip check` and a `pip freeze` diff against the worker.
   **Record the diff.** Three outcomes:
   - clean or additive -> slice 1 installs it in the worker image
     (`requirements.cuda.txt`);
   - it downgrades transformers/torch -> ACE-Step gets its **own venv inside
     the worker image** (`/opt/acestep`), invoked as a **subprocess** from the
     Celery task after eviction. Sequential, same worker, VRAM fully freed on
     exit - this is not the "second GPU process" D6 forbids, because nothing
     else can hold the card while the solo worker is inside the call;
   - it does not install at all -> stop and reassess (DiffRhythm next).
2. Download the `acestep-v15-turbo` DiT to `MODELS_DIR` (MIT, ungated).
   Check `Get-PSDrive` on Windows first.
3. Generate, via the library or CLI, **the fixed 10-prompt set** (5 music
   styles x 2 seeds, `[Instrumental]` as the only lyric, `bpm` and
   `time_signature` set, `audio_duration=120`, wav out). Measure per track:
   VRAM peak (`torch.cuda.max_memory_allocated` in-process; host-side
   `Get-Counter` on Windows), seconds cold (first) and warm, and whether any
   vocal leaked (listen; note it).
4. Same for **Stable Audio Open 1.0** via diffusers `StableAudioPipeline`
   (gated: one-time HF token accept; `HF_HUB_OFFLINE=1` means the download
   happens through the downloader, not at first request). 3 ambience prompts,
   `negative_prompt="vocals, music, melody"`, 30 s.
5. Confirm ACE-Step's `audio_format` accepts `wav` (or that the library
   returns a tensor the task can write itself). The WAV master is
   non-negotiable.
6. Write the measured row in 0010 and pick: in-image vs venv-subprocess, turbo
   vs sft, and the real `duration_s` max from the seconds-per-track number.

**Exit criteria:** 0010's measured row has numbers; a decision on the two
forks is written there; the spike image is deleted (`docker rmi`).

### Slice 1 - `music:<name>` end to end, no LLM, listenable in Activity

The first user-visible path: a request produces a track that plays in the
Activity tab.

- `src/sprite_generator/audio_styles.py`: `STYLES` roster (start with
  `medieval_fantasy`, `tavern`, `dungeon`, `battle`, `village`), each a
  template with slots (`tempo_bpm`, `mood`, `featured`), a fixed
  `time_signature`, a negative list, and the `[Instrumental]` lyric.
  `DEFAULT = "medieval_fantasy"`. `render(style, **slots) -> (prompt, params)`.
  Pure Python, no torch.
- `src/sprite_generator/audio_master.py`: pure numpy. `bar_seconds(bpm,
  sig)`, `cut_to_bars(samples, sr, bpm, sig, min_s)`, `crossfade_loop(samples,
  sr, ms)`, `seam_rms_jump(samples, sr, window_ms)`, `write_wav`,
  `write_ogg_with_loop_tags` (ffmpeg in the image + `mutagen` for
  `LOOPSTART`/`LOOPLENGTH`; decide in slice 0 whether `soundfile` suffices).
  No torch import - the API process may call the analysis half.
- `tasks.py: generate_audio_task(kind, name, style, prompt, params, seed,
  duration_s)`: evict via `pipes`, preflight `mem_get_info` against
  `AUDIO_GPU_BUDGET`, load the model (or spawn the venv subprocess), generate,
  master, write `audio_<kind>_<name>_<uuid>.wav` + `.ogg` **flat in
  `IMAGES_DIR`** (because `generations._url` takes a basename), drop the model
  from `pipes`, `release_vram_cache()`. Wrapped in the breaker exactly like
  `generate_raw_task`. Returns `{file_path, master_path, seed, duration_s,
  sample_rate, loop_start, loop_end, seam_rms_jump}` or `{error, gpu_faulted,
  retry_after_s}`.
- `src/sprite_generator/audio.py` (router): `POST /api/audio` (scope
  `generate`) implementing contract D5 - `generations.begin` at request time
  with `kind`, `name`, `principal`; cache read via the newest done row for
  (`kind`, lower(`name`)) - the index from migration 017 already covers it;
  build with `AsyncResult.get(timeout=AUDIO_GENERATE_TIMEOUT_S)`; on timeout
  **do not revoke**, leave the row `running` with its `celery_task_id`, return
  `503` + `Retry-After: <estimate>`; the task's own completion calls
  `generations.finish` so the row closes without the request. `GET
  /api/audio`, `GET /api/audio/{kind}/{name}` (`?master=1`), `GET
  /api/audio/{kind}/{name}/info`, `GET /api/audio/styles` (scope `read`).
  `404` + JSON body when nothing is done for that name.
- `generations.KINDS += ("music", "ambience")`.
- `Activity.tsx:Row`: if `item.url` ends in `.ogg`/`.wav`, render
  `<audio controls preload="none" src={item.url}>` instead of the `<img>`.
  `Gallery.tsx`: same switch, or filter audio kinds out of the image grid -
  whichever is smaller; audio in the gallery is not a requirement.
- `scripts/smoke-audio.py`: no GPU. Roster renders every style; bar
  arithmetic on 3 bpm/sig pairs; a synthetic sine loop's seam RMS jump is
  below threshold after crossfade and above it before; OGG tags read back
  equal to the WAV offsets.
- `scripts/verify-audio-api.py --submit`: miss -> build -> `200` with
  `audio[0]` decoding to a valid OGG whose tags match `info`; second call ->
  `cached: true` in milliseconds; unknown name on GET -> `404`.

**Acceptance:** from a bare `curl` with a bearer, `POST /api/audio
{"kind":"music","name":"spike-1"}` returns a 120 s OGG; it appears in
Activity within the poll interval with `principal_name` and plays inline;
`make gpu-health` afterwards reports the normal warm state (~4.4 GB headroom)
and the next SDXL txt2img still works. That last check is the one that proves
eviction in both directions.

### Slice 2 - the Audio tab

- `frontend/src/tabs/Audio.tsx`, registered in `App.tsx` `TABS` as
  `{ id: 'audio', label: 'Audio' }` (hash `#audio`). Form: kind, name, style
  (from `/api/audio/styles`), seed, duration, prompt override (collapsed by
  default - the roster is the default path). Submit -> shows the pending row
  from `/api/audio?name=` and polls it (the request itself may `503`; the tab
  treats that as "still building", which is what it means).
- List of finished audio rows with `<audio controls>`, a **loop audition**
  button (Web Audio: `AudioBufferSourceNode.loop = true`, `loopStart`/`loopEnd`
  from `info`, plus a "play the seam" mode that starts 3 s before `loop_end`),
  download OGG / WAV (`fetchObjectUrl` with the bearer for the `/api/audio/..`
  routes).
- `api.ts`: `api.audio.styles()`, `.generate()`, `.list()`, `.info()`.
- Build with `scripts/build-frontend.sh`; verify in the browser at `#audio`.

**Acceptance:** generate a track from the tab, hear it, audition the seam,
download both files; a `503` during a long build shows as "building" not as
an error dialog.

### Slice 3 - LLM style selection

- `audio_styles._llm_style_plan(context, kind) -> (style, slots, author)`:
  the `worlds._llm_biome_plan` shape - `LLM_URL`, `_llm_model()`, cold-start
  retry, JSON answer validated against the roster, invented styles dropped and
  named in `author`, rules fallback (`_rules_style_plan`, keyword table) when
  the LLM is down or wrong. Never writes the model prompt directly.
- `POST /api/audio` gains `context`; `prompt` non-empty bypasses the LLM.
  `info.author` records what happened.
- Tab: "Propose from context" fills the style and slots and shows `author`.
- `smoke-audio.py`: the rules fallback maps 6 fixed contexts to the expected
  styles; the LLM path is exercised by `verify-audio-api.py --llm` only (it
  needs `llm-server` and is not deterministic).

**Acceptance:** `context: "abandoned dwarven mine, danger"` yields `dungeon`
(or the LLM's choice, named), and the same context twice yields the same
prompt when the LLM is bypassed.

### Slice 4 - `ambience:<name>`

- Second model path in `generate_audio_task`: Stable Audio Open 1.0 through
  diffusers, `negative_prompt` from the style's negative list, 20-45 s.
  Mastering: crossfade only, no bar cut (no bpm). Weights via the downloader
  (gated; token handled once, out of band).
- Ambience entries in the roster (`forest`, `cave`, `village_day`, `night`,
  `rain`), separate from music entries by a `kind` field.
- Tab and Activity need nothing new beyond the kind selector.

**Acceptance:** `POST /api/audio {"kind":"ambience","name":"spike-1"}` returns
a 30 s OGG that loops without an audible click; both kinds for one map name
coexist and are fetched separately.

### Slice 5 - something2 connector and the acceptance corpus

- `contract.md` gains the "values to enter in their admin" table for the audio
  provider kind: base URL, `audio[0]` pointer, request template with their
  `{{prompt}}`-style placeholders, the timeout to set from slice 0's number.
- `scripts/verify-audio-api.py --lan <ip>` run from another LAN machine
  (portproxy is what it actually tests).
- The **fixed 10-track loop corpus**: generated once, kept under
  `IMAGES_DIR/audio-corpus/` with a `corpus.json` (style, seed, bpm, seam RMS
  jump, listening verdict). This is the before/after for any later change to
  mastering, and the artefact 0010's acceptance bar refers to.
- 0010 status moves from *proposed* to *accepted* with the numbers.

## Included scope

Slices 0-5 above: one music model, one ambience model, roster + LLM
selection, tile-shaped facade with the D5 overshoot semantics, ledger kinds,
Activity playback, Audio tab, verifier script, no-GPU smoke, connector values
for something2, loop corpus.

## Excluded scope (v1 non-goals, from the contract)

Vocals/lyrics; adaptive or per-area music within a map; MIDI; iOS; stitching
past native length; a second GPU process; anything on something2's side
beyond the values table (their audio provider kind is their work); Gallery
audio browsing beyond not-breaking; per-user quotas.

## Assumptions (each is checked by slice 0 or marked)

- ACE-Step 1.5 turbo runs on this card in WSL inside the worker's torch
  2.14/cu130 stack, or in an isolated venv beside it. *(slice 0)*
- Its output can be obtained as WAV/tensor, not only MP3. *(slice 0)*
- `[Instrumental]` as the only lyric keeps vocals out often enough that the
  10-track set passes; if not, sft + `guidance_scale` is the escalation, at
  8-16 GB. *(slice 0)*
- A 120 s track on a 3060 lands well under something2's 5-minute default,
  including one SDXL eviction. *(slice 0; the 3090 figure suggests so)*
- Stable Audio Open 1.0's gated download can be done once, out of band.
  *(marked; nobody has done it on this host)*
- The unauthenticated `/images` mount is acceptable for audio as it already is
  for every generated image. *(pre-existing property, not a new decision - but
  worth a line in the auth lockdown plan)*

## Verification strategy

- Per slice, above. The pattern is the repo's: a no-GPU smoke for the pure
  parts (`smoke-audio.py`), a contract verifier for the surface
  (`verify-audio-api.py`), `make gpu-health` before and after any GPU slice,
  and one SDXL txt2img after the first audio generation to prove eviction
  works in both directions.
- Nothing here is verified by "it looked right" except the listening verdicts
  in the corpus, which are recorded per track so they can be disagreed with.

## Risks and unresolved questions

- **Dependency conflict** (slice 0). The most likely way this plan changes
  shape. The venv-subprocess fallback is the mitigation, at the cost of a
  model load per call (turbo 2B from ext4: seconds, not minutes).
- **Blocking request threads.** `POST /api/audio` blocks a uvicorn worker
  thread for up to `AUDIO_GENERATE_TIMEOUT_S`, as the tile facade already
  does. With the solo worker, concurrent audio requests queue behind each
  other and each holds a thread; `_long_job_ahead`-style refusal keeps the
  count small, but a burst from something2 (no retries, but many maps) can
  still pin several. Measure with `verify-audio-api.py --burst 5` in slice 5.
- **Overshoot bookkeeping.** With no revoke, a `running` row outlives its
  request. The task must close the row itself (`finish`/`fail` by
  `celery_task_id`), or Activity shows a permanent "on the worker". Slice 1
  tests the timeout path deliberately with `AUDIO_GENERATE_TIMEOUT_S=1`.
- **Three model swaps per cold track** (LLM wake, audio load, SDXL reload).
  Real, and why the facade is cache-first; the tab is where maps get
  pre-warmed. If slice 0 shows the audio load alone is >60 s, consider
  keeping it resident *instead of* SDXL between audio requests (the `pipes`
  dict already allows exactly one) rather than a bigger card.
- **something2 has no audio provider kind today.** Slice 5's table is for a
  connector that does not exist yet; until it does, milestone 1 is the tab.

## Next route

This is more than one implementation pass. Run `to-tickets` on this plan:
slice 0 is a single ticket that blocks everything; slices 1-5 split roughly
per bullet with the acceptance line as each ticket's exit criterion.
