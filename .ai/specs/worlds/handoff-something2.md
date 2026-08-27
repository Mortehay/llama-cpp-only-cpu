# Handoff: build the something2 side of world/map delivery

**For the Claude session in the `Mortehay/something2` checkout.** Written
2026-08-27 from the sprite-generator side. Nothing in this document has been
run against something2 - it is a request, not a report.

## Goal

something2 should be able to browse, preview, download and re-generate worlds
and maps produced by the sprite-generator service, without leaving something2.

Four things to build, in the order they pay off:

1. **A list** of what the service holds - worlds and maps - with enough per-row
   detail to choose one.
2. **Preview in the browser** before downloading anything.
3. **Download and seed** a chosen world spec.
4. **Create / edit / regenerate** from something2, rather than only collecting
   what was made elsewhere.

## The provider

| | |
|---|---|
| Base URL | `http://192.168.0.217:8001` |
| Verified | 2026-08-27, `GET /api/auth/mode` from the LAN returned `200` |
| Auth | **Bearer, enforced.** `Authorization: Bearer <key>` |
| Getting a key | Minted on the sprite-generator box - ask the user; do not expect one in this repo |
| Scopes | `read` for list/preview/download, `generate` for create/edit/delete |

`/api/auth/mode` is deliberately open, so it is a safe reachability probe.

**Gotcha that will waste an afternoon:** that host is WSL2 behind a
`netsh portproxy`, and the proxy target goes stale on every WSL restart with no
error - LAN clients just hang. If everything times out, it is almost certainly
that, not your code. The user re-runs `scripts/lan-expose.ps1` elevated to fix
it.

## What the service exposes

### Worlds - region specs, the `*.map.json` your seeder reads

```
GET    /api/worlds                      -> every region, with a verdict each
GET    /api/worlds/{name}               -> the spec  (?download=true for a file)
GET    /api/worlds/{name}/report        -> is it empty? without fetching the spec
GET    /api/worlds/{name}/preview.png   -> one screenful per world
POST   /api/worlds                      -> generate a new region
PATCH  /api/worlds/{name}               -> change one thing and rebuild
DELETE /api/worlds/{name}
```

**These are SYNCHRONOUS.** A region spec is arithmetic over a biome table - no
model, no GPU, milliseconds - so unlike sprites and tiles there is no 202, no
polling, and no queue. This is the one surface here that fits your existing
blocking-call model directly.

`POST` body: `{name, worlds, target_per_screen, size, theme, author, overwrite}`.
`PATCH` body: any subset of `{worlds, target_per_screen, size, chunk_size,
biome_cell, theme, reauthor}` - fields you do not name are carried over,
including the biome plan, so raising the creature target does not redraw the
region's character.

`GET /api/worlds` rows carry `editable` (false for regions generated before edit
support - they can be replaced but not patched) and `params` (what to prefill an
edit form with).

### Maps - painted tilemaps

```
POST   /api/maps            -> 202 + job id     (NOT synchronous)
GET    /api/jobs/{id}       -> poll it
GET    /api/maps/{id}       -> the tilemap JSON (409 until ready)
GET    /api/jobs/{id}/sheet -> the map picture PNG
```

Maps are minutes to hours of GPU, so this half **cannot** be driven
synchronously. Treat maps as "collect what exists", not "generate on demand",
until your SOMET-334 lands.

## The constraint that shapes all of this

Your AI-provider system carries **images** - `images[0]`, an image pointer, one
blocking POST (`docs/ai-providers.md`: "Sync services only"). That means:

- **Preview PNGs can go through the AI connector as-is.** No new transport.
- **Spec JSON cannot.** A `*.map.json` is not an image and there is no pointer
  that makes it one.

So the recommended split, and the thing to push back on if it looks wrong:

| Need | Route |
|---|---|
| Show a preview | AI connector, or a plain authenticated `GET` of `/preview.png` |
| List / download / create / edit | **A small HTTP client of your own.** Four to six `fetch` calls with a bearer - not a provider registration |

Do not try to force the spec through the provider system. It is the wrong shape
and the failure will be confusing.

## Where this lands in your codebase

From reading your repo on 2026-08-27 - verify, do not trust:

- `backend/seeds/maps/*.map.json` - where a downloaded spec belongs.
- `make seed-map SPEC=<name>` / `make reseed-map SPEC=<name>` - how it is
  applied. Your README warns **only one spec per database**: "two specs seeded
  together leave the second one's worlds unreachable."
- `backend/seeds/mapSpec.js` - your validator. **Run a generated spec through it
  early**; the generator was written against `vale-region.map.json`'s shape, not
  against this validator, and that gap is the likeliest source of a surprise.
- `ai_providers` table + `services/remoteImageProvider.js` +
  `services/generationTarget.js` - the existing provider path, if you route
  previews through it.
- `backend/scripts/seed-map.js` - what consumes the spec.

## What a generated spec contains

`worlds[]` and `links[]`, matching vale-region: `key`, `name`, `grid`, `seed`,
`width`, `height`, `chunk_size`, `biomes`, `biome_cell`,
`allowed_creature_types`, `density`, `level_band`, `is_entry`,
`allows_fast_travel`; `entry_spawn` + `village` on the entry world; `roads`,
`safe_road_radius` and one `waypoints` entry per world.

Two things worth knowing before you read one:

- **`allowed_creature_types` is DERIVED from the world's biomes** - the union
  of their `biomes.creature_types`.

  > **This was wrong in the first version of this document and of the
  > generator, and something2's own validator caught it.** It said
  > `allowed_creature_types` used the P4 bestiary, on the evidence that your
  > `pens` entries use P4 names. They do - but pens are authored placement, not
  > biome spawning. Spawning is gated by `biomes.creature_types` (SOMET-315),
  > and a spec this service reported as `3376 creatures, 0 empty` validated as
  > seeding **zero** in all six worlds. Existing in `entity_types` is not the
  > same as being spawnable. Fixed structurally: derived, so the intersection
  > cannot be empty.
- **Coordinates mix units**: layout in TILES, positions in PIXELS at 100 px per
  tile. `roads` are tiles; `waypoints`/`spawn_x` are pixels.

`pens` and `chest` are deliberately **not** generated - those are authored
content decisions.

## Decisions already made here, so you do not re-litigate them

- Density is **solved, not named**. The caller asks for creatures-per-screen and
  the generator picks each world's tier after the biome multiplier. Your
  `normal` tier is ~4/screen and Meadow halves it to ~2, which is where "too
  much empty space" came from.
- The LLM picks **biomes only**, never counts.
- The spec is **repaired** to meet flora and creature-variety floors before it
  is written.
- Leaders (`Champion`/`Apex`) are capped at one per world, because your
  `computeAuras` is O(leaders x population) every tick.

Full reasoning: `.ai/decisions/0008-world-specs-for-something2.md` and
`.ai/specs/worlds/contract.md` in the sprite-generator repo.

## Acceptance criteria

1. A list in something2 showing every region the service holds, with worlds,
   creature count, mean per-screen, and whether any world is flagged empty.
2. A preview image for a chosen region, rendered in the browser, **before**
   anything is downloaded.
3. A chosen region downloaded to `backend/seeds/maps/` and seeded successfully
   with `make seed-map`.
4. A region created from something2 and appearing in the list without a manual
   step on the other machine.
5. An existing region edited - raise its creature target, confirm the preview
   and report change, confirm the biomes do **not**.
6. Auth failures and an unreachable provider both surface as a message, not an
   empty list. (This exact bug shipped on the generator side and looked
   identical to "nothing generated yet".)

## Open questions for you

1. ~~**Does a generated spec pass `seeds/mapSpec.js`?**~~ **ANSWERED
   2026-08-27, and it did not** - 14 road errors (points on the wall ring;
   valid range is 1..size-2) and 6 spawnability errors (see the correction
   above). Both fixed; `emerald-reach` regenerated and awaiting a re-run.
   Everything structural passed: grid, links, villages, entry_spawn,
   waypoints, safe_road_radius, level_band, density and the unknown-key check.

   **Still open, and now blocking variety:** the five biomes this generator can
   select (Meadow, Mire, Deep Forest, Arid Dunes, Frozen Waste) admit only
   **four creatures between them** - Slime, Wolf, Bat, Skeleton. The other 27
   biomes in your table are P4. To use them this side needs `color`,
   `creature_density` and `flora_types` per biome. Defaulting those would
   repeat the mistake above: computing confidently from a catalog not actually
   in hand.
2. **Where should this UI live** - the admin panel beside the AI providers, or
   its own screen?
3. **Should something2 cache downloaded specs**, or re-fetch each time?
4. **Is `level_band` interpreted the way the bestiary's own level ranges
   suggest?** Assumed here, never verified against your placement code.

## Verify on this side

    docker exec sprite_generator python /app/scripts/smoke-world-api.py    # 9 cases
    docker exec sprite_generator python /app/scripts/smoke-world-gen.py    # 22 cases

Both run in under a second, with no model and no GPU.
