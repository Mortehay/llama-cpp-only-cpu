# Audio: music and ambience for something2 maps

Design reasoning is in
[decisions/0010](../../decisions/0010-audio-generation.md). This is the
surface, written BEFORE the code and before the model is chosen. Every number
marked *placeholder* is a placeholder until the research step measures it on
this card; do not quote one as a fact.

Status: **draft, pre-implementation** (2026-09-12). Confirmed in a grill
session; nothing here has been run.

## Two kinds, named by map

| kind | what | length | model class |
|---|---|---|---|
| `music` | a musical bed for a map, looped endlessly by the player | 60-240 s, default 120 *(placeholder)* | long-form text-to-music |
| `ambience` | a stationary texture - wind, birds, dungeon drips - layered under the music | 20-45 s, default 30 *(placeholder)* | short-clip text-to-audio |

They are **separate artefacts** and something2 plays them as two Web Audio
sources at once. Rejected: one mixed file per map. Ambience is stationary, so
it loops cleanly at almost any cut with a crossfade, while the music model gets
seams wrong more often; splitting them puts the hard property (a seamless loop
over minutes) on the artefact that needs it and lets their side swap ambience
per area without touching the music.

The **name is the map name**, the same string `map:<name>` and the worlds
surface use. Alternates are `?variant=` on the same name, not a second
namespace. Like tiles and entities, names are not unique in the ledger: a
re-roll writes a new row and the newest finished one wins.

## Format

- **Delivered: OGG Vorbis**, with `LOOPSTART` and `LOOPLENGTH` tags (sample
  offsets) so an engine that reads them loops without ever seeing the tail.
- **Kept on disk: the WAV master** beside the OGG. The master is the source of
  truth for loop points; the OGG is an encoding of it.
- **Not MP3.** Encoder delay and padding put silence at the seam, so a looped
  MP3 gaps. This is the whole reason the format question had an answer.
- **Not MIDI.** Direction A (neural audio) produces samples, not notes. MIDI
  would only exist under the symbolic direction, which was set aside - see 0010.
- Target platform is **browser, desktop-first**. iOS Safari is a non-goal;
  if it ever matters the answer is an AAC sibling, not a change here.

Loop playback on their side: `AudioBufferSourceNode.loop = true` with
`loopStart`/`loopEnd` taken from `info` - sample-accurate. `<audio loop>` is the
fallback and may gap on some browsers.

## The facade: tile-shaped

something2 is the **requester**, through a new audio provider kind on their
side (their image provider cannot carry this: it decodes `images[0]` as a PNG
and slices it). The behaviour copies the `tile:` facade, not the `map:` one:

    POST /api/audio            (scope: generate)

```json
{
  "kind": "music",
  "name": "emerald-reach",
  "style": "medieval_fantasy",
  "prompt": "",
  "context": "a green valley giving way to cold highlands",
  "seed": -1,
  "duration_s": 120
}
```

1. **Cache first.** Newest finished row with this `kind` and `name` -> `200`
   in milliseconds, `cached: true`, `served_from: cache`.
2. **Otherwise build, blocking up to `AUDIO_GENERATE_TIMEOUT_S`.** Done in
   time -> `200`, `cached: false`.
3. **Over budget -> `503` + `Retry-After`, and the build KEEPS RUNNING** in the
   worker. Their next request for the same name is a cache hit. This is what
   makes a sync-shaped facade survivable for a build that takes minutes.

A `503` carries a `reason`, and the three are not interchangeable:

| `reason` | means | `Retry-After` |
|---|---|---|
| `building` | over budget, still running, not cancelled | 60 |
| `busy` | a non-audio job holds the solo worker; nothing was queued | 120 |
| `gpu_faulted` | the breaker is open; **never retried** by this service | the cooldown |

A second request for a name already building **joins** it rather than
queueing a second build - verified 2026-09-13 with
`AUDIO_GENERATE_TIMEOUT_S=1`: one running row for the name, the abandoned
build closed its own row with its file and loop metadata, and the next
request was a 0.08 s cache read. The header is sent lowercase (`retry-after`),
as HTTP allows; a client that looks for `Retry-After` in a case-sensitive map
will not find it.

**Never a placeholder.** The `map:` facade returns a magenta cross with
`cached: false` when a map is provisional; the audio equivalent would be
silence, and a caller that stores it as final keeps a silent map forever.
Not-ready is always non-2xx here.

A `gpu_faulted` result is **not** retried and not re-queued - that is the storm
the breaker exists to stop. The breaker's `503` and the over-budget `503` are
distinguishable by body, and both carry `Retry-After`.

### Response

```json
{
  "audio": ["<base64 ogg>"],
  "info": {
    "kind": "music",
    "name": "emerald-reach",
    "style": "medieval_fantasy",
    "prompt": "<the prompt actually sent to the model>",
    "author": "style chosen by Qwen2.5-3B-Instruct-Q4_K_M; ...",
    "seed": 1234,
    "duration_s": 118.4,
    "sample_rate": 44100,
    "loop_start": 0,
    "loop_end": 5223168,
    "cached": false,
    "served_from": "generated",
    "generation_id": "<uuid>"
  }
}
```

- The response pointer for their admin is `audio[0]`, by analogy with
  `images[0]`. ~3 MB of base64 for two minutes, under their 32 MB body cap.
- `seed` is the seed **actually used**. As with entity cutouts, a caller-pinned
  seed may be overridden if the loop-mastering step rejects the take; the
  returned value is the truth.
- `loop_start` / `loop_end` are **sample offsets**, and they are the same
  numbers as the OGG tags.
- `author` says what the LLM did, including when it fell back and why -
  the `worlds.py` convention.

Read-only siblings, for their player and for the tab:

    GET  /api/audio                         (scope: read)  the ledger, filtered ?kind=&name=
    GET  /api/audio/{kind}/{name}           (scope: read)  the OGG, 404 + JSON body if none
    GET  /api/audio/{kind}/{name}?master=1  (scope: read)  the WAV
    GET  /api/audio/{kind}/{name}/info      (scope: read)  the info block above

Not-ready on GET is **`404`**, not `409`: for the jobs API `409` meant "exists,
unfinished", but a map with no track is the normal state and their player
should treat missing and unfinished identically - play nothing.

## Style is a roster, not a string

Styles live in a Python module (the `core_models.CORE_MODELS` pattern - one
list the tab dropdown, the facade and the LLM all read). Each entry is a prompt
template with variable slots (tempo band, mood, featured instrument) and a
negative list. `medieval_fantasy` is the default.

The LLM (`llm-server`, the same one `worlds.py` uses) **selects** from the
roster given `context`, fills the slots, and never free-forms into the model.
The response records what it chose and anything it invented and dropped; when
it is asleep or wrong the rules fallback picks by keyword. `prompt` non-empty
is an explicit override and bypasses the LLM entirely.

Why: a free-form LLM prompt straight into a music model is unreproducible - the
same map yields a different prompt each run, so a re-roll cannot be compared to
the last one, and "medieval fantasy RPG by default" has nowhere to live except
a system prompt. With a roster, something2 can name a style without depending
on the LLM being awake.

**House style: smooth and medieval, for every entry** (owner, 2026-09-28,
after listening to the spike tracks). Period acoustic instruments played
legato - battle is a noble march on frame drums, not war drums and brass;
dungeon is a drone with viola da gamba, not piano; tavern is lilting, not
raucous. Tempo bands and mood values were narrowed so no slot leads back out.
Because ACE-Step turbo has **no CFG** and never receives the negative, the
style lives in the POSITIVE template (`audio_styles._SMOOTH`). Positive
templates contain no negations; the ambience ones used to end in "no music",
which asks for music.

**Every template carries `no vocals` in its negative.** Both candidate
long-form models are song-trained; vocals leaking into an instrumental bed is
the expected failure. A model with no negative-prompt path scores against
itself in research.

## Where it runs

**The existing solo Celery worker, with pipeline eviction.** Not a second GPU
process. The card is 12 GB; SDXL holds ~6.8 GB warm and `llm-server` holds
8-9 GB until it sleeps. A second process contending for the card reproduces
the `dxgkio_make_resident: -12` fault in `project-context.md`. So one track
costs up to three model swaps (LLM wake, audio model load, SDXL re-warm) - which
is why the facade is cache-first and why the tab exists to pre-warm.

## The ledger and the UI

Every request writes a `generations` row at request time - `kind` gains
`music` and `ambience`, `file_path` set and `job_id` NULL, the row owns the
file, `assets_v` gets an `audio` arm. Activity shows the request the moment it
arrives, with `principal_name` first (the API key is the only identity that
separates callers on this topology).

Listening in the browser: **fetch with a bearer into a blob URL**, never a
bare `<audio src="/api/audio/...">` - auth is enforced and a navigation-style
media load sends no `Authorization` header. This is the `<img src>` 401 bug
again, and it applies before the first track exists.

A new **Audio tab** drives the same `POST /api/audio` by hand: pick kind,
name, style (or let the LLM propose), seed, duration; generate; list; listen
with a **loop audition** control that plays the last few seconds into the
first few, because the seam is the property under test; download OGG or WAV.

## Acceptance (what "works" means, before anyone tunes)

- **Loop.** On a fixed set of ~10 tracks across the roster, kept so a later
  change is a before/after: the crossfade is bar-aligned from beat tracking,
  there is no RMS discontinuity above threshold at the seam, and a listening
  pass finds the seam inaudible on at least 8, "audible but not a click"
  tolerated on the rest.
- **Style.** The default template yields instrumental, no vocals, on the same
  set.
- **Length.** No stitching in v1. A duration the model cannot reach natively
  is refused with a message; stitched continuations drift and are a separate
  decision.
- **Facade.** Seconds per 120 s track on this card, **cold and warm**,
  including the SDXL eviction. That number goes to something2 as their
  `AI_PROVIDER_GENERATE_TIMEOUT_MS`, and `AUDIO_GENERATE_TIMEOUT_S` sits just
  under it so a slow build surfaces as our `503` with a message rather than
  their opaque abort.

## Files on disk (2026-09-28)

Audio has its own tree, not IMAGES_DIR: `AUDIO_DIR` (host `../../audio`, the
repo root beside `images/`; container `/app/audio`), laid out as
`<kind>/<name>_<uid>.{ogg,wav}`, plus `_takes/` (raw model output, deleted
after mastering) and `spike/`. Served open at `/audio/<kind>/...` for the same
reason `/images` is open - an `<audio>` tag sends no bearer and the names are
unguessable. The ledger's `url` keeps the subfolder. Rows written before this
were moved by `scripts/move-audio-out-of-images.py`.

## `sfx`: one-shot cues (draft, 2026-09-28 - see 0010 D7/D8)

A third kind. A **cue** is a roster entry for one game event (`slash`, `hit`,
`pickup`, `spell`, `footstep`, `ui_click`, ...), 0.1-3 s, played once, never
looped - so no loop tags; the property under test is a hard onset, a clean
tail and consistent loudness, not a seam.

Two **engines**, `realistic` (Stable Audio Open 1.0) and `retro`
(sfxr-style synthesis, no GPU). Resolved by precedence, most specific first:
request `engine` > world `sfx_engine` in `<world>.gen.json` > the cue's
default > `realistic`. `info.engine_from` says which level decided. A level
that names an engine the cue has no recipe for is a `422` with a message, not
a silent fall-through. The cache key is `(sfx, engine, cue, name)`, so a
world's engine change takes effect on the next request.

Decided by the owner, 2026-09-28, and implemented (ticket 16):

- **Name `<cue>/<entity>`**, entity optional (`hit/slime`, `ui_click`). The
  ledger name is `<engine>:<cue>/<entity>`, lower-cased - that is the cache
  key, so a realistic and a retro `hit/slime` never share a slot.
- **Variants: the caller chooses 1-5, default 3.** A cached cue with at
  least as many variants as asked is served from cache (first N).
- **Single cues and packs.**

      POST /api/audio/sfx       {"cue","entity?","engine?","world?","variants":3,"seed?"}
          -> {"audio": ["<b64 ogg>", ...one per variant],
              "info": {"kind":"sfx","cue","entity","engine","engine_from",
                       "name","prompt","seed","sample_rate",
                       "variants":[{"url","duration_s","onset_ms"}],
                       "cached","served_from","generation_id"}}
      POST /api/audio/sfx-pack  {"items":[{"cue","entity?","engine?"}],
                                 "world?","variants":3,"seed?"}      (max 40)
          -> {"items":[<the info block above, plus "audio">, or
                       {"cue","name","error"} for a cue that failed],
              "count","failed"}
      GET  /api/audio/styles?kind=sfx   the cue roster, with each cue's engines

  A pack is ONE worker task and ONE model load; each cue owns its own ledger
  row. The same D5 rules apply: 503 `building` (not cancelled), `busy`,
  `gpu_faulted`. Refusals: unknown cue or variants outside 1-5 -> 422; an
  engine without a recipe -> 422 naming the level it came from; `world`
  that does not exist -> 404. `sfx_engine` is set on a world with
  `POST/PATCH /api/worlds` and lives only in its `.gen.json`.
- Files: `audio/sfx/<engine>/<cue>/<entity>_<uid>_v<n>.{ogg,wav}`, served at
  `/audio/sfx/...`. **No loop tags** - a cue that carried LOOPSTART would
  loop forever in an engine that honours it.

Measured 2026-09-28 (realistic, Stable Audio Open, 100 steps): ~8 s per
variant after a ~3-4 s load; a 3-cue / 8-variant pack in 79 s; onset 0-5 ms
on every variant; 2.85 GB peak. Two findings made that possible - see 0010
"sfx".

**Retro (ticket 17).** Procedural 8-bit, rendered in the API process - never
queued on the worker, so it answers while a GPU job runs. Cues: slash, hit,
pickup, spell, ui_click; **footstep has no retro recipe** and is refused
(422, "it offers realistic"). Without a `seed` the seed is derived from
`<cue>/<entity>`, so a name always yields the same sound; variants use
`seed..seed+n-1`, each recorded in `info.variants`. Measured: 218 ms for a
3-variant cue end to end, 25 ms of synthesis per variant; a pack mixing a
cached realistic cue with three retro cues in 566 ms with no GPU.

## Non-goals for v1

Vocals or lyrics; adaptive or per-area music within a map; MIDI export; iOS;
stitching past the model's native length; any second GPU process; sync
generation that assumes the 5-minute cap is enough.

## Open before planning

- **Model choice is sourced, not measured** (2026-09-12, table in 0010).
  Candidates after the read: **ACE-Step 1.5** for `music` (MIT, 48 kHz stereo,
  10-600 s native, takes `bpm` / `time_signature` / `key_scale` so the loop cut
  is bar arithmetic; instrumental via `[Instrumental]` as the only lyric; the
  turbo variant fits the card but has no CFG) and **Stable Audio Open 1.0** for
  `ambience` (47 s, real `negative_prompt`, Community License, gated). MusicGen
  is out on CC-BY-NC. Still unmeasured on this card: VRAM alongside SDXL
  eviction in WSL, instrumental quality on the medieval template, seconds per
  120 s track cold and warm. Expect `sample_rate: 48000` from ACE-Step, not the
  44100 in the example above.
- ~~ACE-Step wants Python 3.11-3.12; the worker image is `python:3.10-slim`.~~
  **Done 2026-09-12**: both sprite Dockerfiles are on `python:3.11-slim`,
  rebuilt and verified (details in 0010). The in-container shape (D6) stands.
- **Does something2 play audio at all today?** If not, milestone 1 is
  "download from the tab, drop into their assets" and the connector is
  milestone 2.
- **Worker image rebuild.** Audio deps (torchaudio, model libraries, a vorbis
  encoder, beat tracking) mean a rebuild, 15-25 min for the torch wheel, and
  the "no CUDA at import" rule applies to whatever gets imported.
