# something2 audio integration - handoff spec for the something2 repo

For: an agent (or person) working in **`Mortehay/something2`**. Written from
the audio generator's side (`llama-cpp-only-cpu`, "the generator"), after
reading something2 at HEAD `2ddd1bf` (2026-09-27). Every `path:line` below is
into THAT repo and was read, not assumed - re-check them if HEAD has moved.

Owner decisions (2026-09-28), not open for re-litigation here:

- **A** - audio travels over something2's existing **AI connector**, extended
  with a media kind. Not a second connector, not an offline import.
- **2** - each world (map) gets **context sets** of music (several tracks per
  context) plus **ambience layers** that play together.
- **SFX mirror VFX** - sound bindings copy the existing VFX binding model
  (library table + moment->names jsonb on entity/item types).
- **Batch** - audio is generated in bulk through the existing **art console**
  queue, like images.

## What something2 has today (read at `2ddd1bf`)

- **No audio anywhere** - no player, assets, columns, settings or tickets.
  The biome spec lists music as a non-goal
  (`docs/superpowers/specs/2026-07-28-biome-data-model-design.md:347`).
- **The AI connector is image-only by construction.**
  - No kind/media column; the design comment explains why none was needed for
    images (`backend/migrations/1714440360000_ai_providers.js:53-54`).
  - **One active provider, enforced by a unique index**
    (`...360000_ai_providers.js:80-83`), and unpinned generation goes to it
    (`backend/src/services/generationTarget.js:44-46`).
  - The result is stored as `static.png` with `image/png`
    (`backend/src/services/remoteImageProvider.js:511`, key built at `:309-314`).
    Decoding does no magic-byte check (`:234-265`), so base64 OGG would be
    stored **mislabelled** rather than rejected.
  - `GET /api/assets/*` sets a content type only for `.png`/`.json`
    (`backend/src/index.js:3197-3198`).
  - Template variables: only `{{prompt}} {{model}} {{seed}} {{width}}
    {{height}} {{frames}}` (`remoteImageProvider.js:348-355`).
- **Worlds** are the maps: `worlds` rows linked by `map_links`, containing
  `villages`, `waypoints`, `world_chests`. A new world field needs a migration
  AND the player whitelist `WORLD_CONTENT_FIELDS` (`index.js:4003-4007`) AND
  the map-spec validator `WORLD_KEYS` (`backend/seeds/mapSpec.js:125-133`) AND
  the admin PUT field list (`index.js:4158`). World change reaches the client
  as the `transition` frame (`Game.js:769`).
- **VFX is the template for SFX.** Moments `attack|impact|miss|trail`
  (`backend/src/authority/vfx.js:10`), defaults per kind (`:29-34`), bindings
  `{"<moment>": "<vfx_effects.name>"}` in `entity_types.vfx` / `item_types.vfx`
  jsonb with no FK (`1714440034000_vfx_effects.js:48-52`,
  `1714440170000_vfx_entity_bindings.js:15-47`), resolved server-side and sent
  as names on the `state` frame's `attacks`/`impacts` (`server.js:3108-3128`,
  client `Game.js:1302-1317`). Other one-shots: `picked` (`Game.js:641`),
  `chestOpened` (`:697`), `{type:'vfx'}` (`:703-711`). **No death event** on
  the wire.
- **Art console** batch queue: `art_jobs` (`subject_kind` is free text, "a new
  subject kind is a registry entry", `1714440530000_art_jobs.js:28-31`),
  kinds registry `catalogSubjects.js`, dispatcher `artDispatcher.js` - one
  remote call per job, concurrency `ART_DISPATCH_CONCURRENCY` default 1
  (`:42`), and it **requires `image_key`** (`:324-327`).

## What the generator offers (live now)

Base URL `http://<windows-lan-ip>:8001`. Auth header `Authorization`, value
**`Bearer <key>`** (their `authHeaders` sends it verbatim). Use a key minted for
something2 only (Settings tab -> Create key, name `something2-audio`, scopes
`read,generate`) - one key per consumer is how the generator's Activity tab
tells callers apart.

| Kind | Call | Response |
|---|---|---|
| music | `POST /api/audio {"kind":"music","name":N,"style"?:S,"context"?:TEXT,"seed"?}` | `audio[0]` base64 OGG, loop tags inside; `info.loop_start/loop_end` (samples), `sample_rate` 48000 |
| ambience | same with `"kind":"ambience"` | `audio[0]`, 44100 Hz, loop tags |
| sfx | `POST /api/audio/sfx {"cue","entity"?,"engine"?,"world"?,"variants":1-5}` | `audio[*]` one base64 OGG per variant, **no** loop tags; `info.variants[].url` |
| sfx pack | `POST /api/audio/sfx-pack {"items":[{cue,entity?,engine?}],"engine"?,"variants"}` | `items[*]` as above; a top-level `engine` is the default for items without one |
| discovery | `GET /api/audio/styles?kind=music|ambience|sfx` | pointer `$[*].value` |

- **Synchronous, cache-first by name.** A repeated name is a ~0.1 s cache hit;
  a new one builds (music ~27 s warm, ~60 s cold; realistic sfx ~8 s/variant;
  retro sfx < 1 s). Over 240 s the answer is **`503` + `Retry-After`, reason
  `building`, and the build continues** - retry later and it is a cache hit.
  `503 busy` = a long GPU job holds the worker; `503 gpu_faulted` = do not
  retry in a loop. Their 300 s `AI_PROVIDER_GENERATE_TIMEOUT_MS` is correct.
- Styles (music): `medieval_fantasy` (default), `tavern`, `dungeon`, `battle`,
  `village`. Ambience: `forest` (default), `cave`, `village_day`, `night`,
  `rain`. All in a smooth medieval house style.
- Cues (sfx): `slash`, `hit`, `pickup`, `spell`, `ui_click`, `miss`,
  `chest_open`, `death`, `waypoint` (realistic and retro), `footstep`
  (realistic only).
- Batches: the generator keeps Stable Audio resident between consecutive
  ambience/sfx jobs (measured 2026-09-28: a warm cue 8.5 s vs 41 s cold), so
  the art console's one-request-per-item queue does not pay a load per item. Engine precedence: request `engine` >
  the generator-side world's `sfx_engine` > cue default > `realistic`.
- Full surface: `.ai/specs/audio/contract.md` in the generator repo.

## Vocabulary: their moments -> the generator's cues

| something2 moment / event | cue | note |
|---|---|---|
| `attack` (melee) | `slash` | entity = the weapon or creature |
| `attack` (caster/skill) | `spell` | entity = the spell ("an ice spell") |
| `impact` | `hit` | entity = the thing hit |
| `miss` | `miss` | a light whoosh; added 2026-09-28 |
| `picked` | `pickup` | entity = the item |
| `chestOpened` | `chest_open` | added 2026-09-28 |
| death (needs a new event on their wire) | `death` | added 2026-09-28; entity = the creature |
| `waypointActivated` | `waypoint` | added 2026-09-28 |
| UI clicks (client-only) | `ui_click` | |
| footsteps (client-only, movement) | `footstep` | realistic only |

Contexts -> music style (a starting default; the admin may override per set):
`explore` -> `medieval_fantasy`, `combat` -> `battle`, `village` -> `village`,
`tavern`/`inn` -> `tavern`, `dungeon`/deep biomes -> `dungeon`.
Name each track by world, context and index, e.g. `emerald-reach:combat:2` -
the name is the cache key, so a re-roll is a new index, not a new name scheme.

## Slices (build in order; each is shippable)

### S1 - The connector learns a media kind

- Migration: `ai_providers.media text NOT NULL DEFAULT 'image' CHECK (media
  IN ('image','audio'))`. Replace the single-active index with **one active per
  media**: `UNIQUE ON ai_providers (media) WHERE is_active`.
- `generationTarget.resolveGenerationTarget` takes the media and only ever
  picks a provider of that media - an audio provider must never receive an
  image job (today rule 3 would hand it every unpinned image generation).
- An audio result path beside the image one: store `<bucket>/audio/<kind>/<name>/<job>/<n>.ogg`
  with `audio/ogg`; check the `OggS` magic before storing; read `audio[0]`
  (music/ambience) or `audio[*]` (sfx variants) via a per-provider
  `response_audio_pointer`.
- `GET /api/assets/*`: serve `.ogg` as `audio/ogg`, `.wav` as `audio/wav`.
- Template variables for audio: `{{name}} {{kind}} {{style}} {{context}}
  {{cue}} {{entity}} {{variants}} {{seed}}`.

Acceptance: an audio provider row pointing at the generator can be active at
the same time as the image provider; an image job still goes to the image
provider; an audio job stores a playable `.ogg` served with `audio/ogg`; a
non-OGG body is rejected with a message.

### S2 - Audio libraries (the `vfx_effects` pattern)

- `audio_tracks(name PK, kind music|ambience, key, loop_start, loop_end,
  sample_rate, style, seed, source)` and `sound_effects(name PK, cue,
  entity, engine, variant_keys jsonb, source)`.
- Admin CRUD screens like `VfxEffectsAdmin.jsx`; refuse deleting a name still
  bound anywhere (copy `vfxReferences`/`orphanConflict`, `index.js:1813-1829`).

### S3 - World audio: context sets + ambience layers

- `worlds.audio jsonb`, shape:
  `{"music": {"explore": ["t1","t2"], "combat": ["t3"], "village": [...]},
    "ambience": ["forest-day", "stream"], "crossfade_ms": 2000, "shuffle": true}`
  Names reference `audio_tracks`; dropdowns, never free text (the VFX lesson).
- Optional `biomes.audio` with the same shape as the default when a world has
  none.
- Add `audio` to `WORLD_CONTENT_FIELDS`, `WORLD_KEYS`, and the admin PUT list,
  or players never receive it / specs reject it.
- Context rules (client): `combat` while any creature is in `chase` toward the
  player; `village` inside a village radius; else `explore`. Switch with a
  crossfade; never restart a context's track on re-entry within N seconds.

### S4 - SFX bindings (mirror of VFX)

- `entity_types.sfx` / `item_types.sfx` jsonb: `{"<moment>": ["name", ...]}` -
  a LIST, so variants are bindings, not a special case. Moments: the VFX four
  plus `picked`, `chest`, `death`, `ui`.
- Server: resolve alongside VFX and send names on the same frames (no new
  frame for attack/impact). Add a death event if `death` is to be used.

### S5 - Client audio

- Music player: context sets, shuffle within a set, crossfade on context or
  world change (`transition` frame), loops honour `loop_start/loop_end`
  (Web Audio `AudioBufferSourceNode.loop` with `loopStart/loopEnd`).
- Ambience: every layer in `worlds.audio.ambience` plays at once, looped.
- **SFX mixer** - several sounds at once, without mush:
  random variant per play (never the same twice in a row), a polyphony cap
  (e.g. 12 voices, drop the quietest/oldest), a per-name cooldown (~60 ms), a
  small pitch jitter (+-3%), distance attenuation from the event's x/y.
- Volumes (master/music/ambience/sfx) in the client settings snapshot
  (`Game.js:1002-1010`). Browsers block audio until a user gesture - start
  the context on the first click/keypress.

### S6 - Batch through the art console

- Subject kinds: `world_music` (per world x context x index), `world_ambience`,
  `entity_sfx` (per entity x moment, with variants).
- `artDispatcher` must accept an `audio_key` result (today it requires
  `image_key`, `:324-327`) and write into the S2 libraries instead of
  `catalog_art`.
- Keep `ART_DISPATCH_CONCURRENCY=1` for the generator: its GPU runs one job at
  a time. The generator keeps its audio model warm between consecutive audio
  jobs, so a batch of sfx does not pay a model load per item.

## Not in scope / do not

- Do not register the generator as an **image** provider for audio, and do not
  activate an audio provider before S1's per-media index exists - it would
  take over all image generation.
- Do not store audio under `static.png` or serve it as `image/png`.
- Do not free-text track or sound names in bindings.
- No adaptive stems, no per-area music inside a world, no iOS - generator-side
  v1 non-goals.

## Verification from something2's side

    curl -s -H 'Authorization: Bearer <key>' http://<ip>:8001/api/audio/styles?kind=music
    curl -s -X POST -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' \
      -d '{"kind":"ambience","name":"s2-check-forest"}' http://<ip>:8001/api/audio \
      | python3 -c "import sys,json,base64; d=json.load(sys.stdin); open('t.ogg','wb').write(base64.b64decode(d['audio'][0])); print(d['info'])"

A second call with the same name must return in well under a second with
`"cached": true`.
