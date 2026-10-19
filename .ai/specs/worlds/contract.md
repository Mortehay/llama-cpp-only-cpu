# World specs: the API something2 consumes

Design and findings are in
[decisions/0008](../../decisions/0008-world-specs-for-something2.md). This is
the surface.

## Why this one is synchronous

Every other generation surface here is `202` + poll, and something2 cannot poll
("Sync services only", their `docs/ai-providers.md`). A world spec is arithmetic
over a biome table - no model, no GPU, milliseconds - so it is served inline.
**This is the only endpoint here something2 can call directly today without any
facade.**

It does still need a small client on their side: their provider system carries
IMAGES (`images[0]`), and a map spec is JSON. Four plain HTTP calls with a
bearer, not a provider registration.

## Endpoints

    POST   /api/worlds                      -> 201, the spec + its report
    PATCH  /api/worlds/{name}               -> change one thing and rebuild
    GET    /api/worlds                      -> every generated region
    GET    /api/worlds/{name}               -> the map spec  (?download=true for a file)
                                               + ETag: "<sha256 of the bytes>"
    PUT    /api/worlds/{name}/spec          -> store a HAND-EDITED spec as-is
    GET    /api/worlds/{name}/report        -> is it empty?  (without fetching the spec)
    GET    /api/worlds/{name}/preview.png   -> one screenful per world
    DELETE /api/worlds/{name}

Scopes: `generate` for POST/PATCH/PUT/DELETE, `read` for the rest.

A region is three files: `<name>.map.json` (their format, untouched),
`<name>.preview.png`, and `<name>.gen.json` - the generation parameters, kept
BESIDE the spec rather than inside it so their seeder never sees a field it did
not ask for. The sidecar is what makes `PATCH` possible: without it an edit
would have to reverse engineer the request from its own output. A region
generated before it existed lists as `editable: false` and 409s on PATCH.
Since 2026-10-19 there can be a FOURTH file, `<name>.edit.json` - the hand-edit
marker; see **Hand edits** below for why it is not a key in the sidecar.

Generated files live in `WORLDS_DIR` (default `/app/images/worlds`) as
`<name>.map.json` and `<name>.preview.png`. The **spec file is the artefact** -
it is what `make seed-map SPEC=<name>` reads - so the listing is a directory
rather than a table, and there is no migration.

## Hand edits: the other direction (2026-10-19)

Admins also author maps in the game's **World Map Editor**, and those edits get
written back into the same `.map.json`. Until now the box could not receive
them: `PATCH` takes GENERATION PARAMETERS and rebuilds, which is exactly what
must not happen to a map somebody drew. `PUT /api/worlds/{name}/spec` takes the
full spec, in the shape `GET` returns, and **does not regenerate**.

It **stores unknown keys unchanged**. Three optional fields their format
carries and this generator does not emit have to survive a round trip: a
world's `graph: [x, y]` (a dragged graph position, which wins over
`grid x 220` on their side), a link's `kind: "edge"` (a compass doorway with an
explicit `edge` and no grid-delta rule), and a link's `to_name` (a world NAME
in ANOTHER region, used instead of a spec-local `to`). That list will grow on
their side, not here, so the body is taken as a dict rather than through a
model that would silently drop whatever it has not been taught.

| code | when |
|---|---|
| 200 | `{name, sha256, hand_edited, edited_at, problems, spec_url, preview_url}`, plus the new `ETag` |
| 404 | unknown region - checked BEFORE `If-Match`, so a caller is not sent off to fix a precondition on a region that does not exist |
| 409 | `{detail: "spec changed on the box", sha256: <current>}` - the `If-Match` is stale |
| 422 | `{detail: [...]}` - structurally unusable, see below |
| 428 | no `If-Match` at all |

**`If-Match` is required, not optional.** Two admins editing one region is the
ordinary case this exists for, and a blind write loses one of them with no
trace; a 409 costs a re-read. The ETag is the sha256 of the stored file's exact
bytes, quoted, and it is accepted back quoted, bare, or `W/`-prefixed.
`If-Match: *` is honoured as RFC 9110 means it - "whatever is there now" - and
is a deliberate force-write, not the header a sync should send.

A spec that goes out and comes back unchanged is **byte-identical**: the PUT
re-serialises through the same `to_json` generation uses, so the ETag does not
move. That is what makes "has this region changed?" answerable by comparing one
string.

**Validation is structural only.** Their editor is the authority on what a
hand-authored region should contain, so a spec that is merely thin, crowded or
oddly linked is STORED and reported - the `problems` list comes back with the
200. A 422 refuses only a body this service could not serve afterwards: not an
object, no worlds, a world with no `key`, duplicate keys, a link whose `from`
or `to` names no world here and carries no `to_name`, `kind: "edge"` without a
valid `edge`, or anything `world_gen.report` itself cannot read - that last one
would leave the region's own `/report` a 500 for ever. The body's `name` is
compared to the path **as a slug**: a region created as "Emerald Reach" is
stored at `emerald-reach.map.json` with `"name": "Emerald Reach"` inside it,
and a literal comparison would refuse that region its own spec back.

### A hand-edited region refuses to be regenerated by accident

The flag lives in a fourth file, `<name>.edit.json`, rather than as a key in
the `.gen.json` sidecar. The sidecar means exactly one thing - "how this was
generated" - and both the listing and `check-artifacts.py` read its PRESENCE as
"this region can be regenerated", so a sidecar holding a flag but no parameters
would list as editable and then fail on the first PATCH. A region from before
sidecars existed has no `.gen.json` at all, and can still be hand-edited.

- `PATCH /api/worlds/{name}` -> 409 `{detail: "region has hand edits; pass
  discard_hand_edits=true to regenerate"}`. With `"discard_hand_edits": true`
  it regenerates as before and the flag clears.
- `POST /api/worlds` with `overwrite: true` follows the same rule: an overwrite
  IS a regeneration, and a guard on one door is not a guard.
- `GET /api/worlds` items and `GET /api/worlds/{name}/report` both carry
  `hand_edited` and `edited_at` (UTC, ISO 8601, second precision).
- `editable` still tracks the `.gen.json` sidecar, which a hand edit leaves
  alone. A hand-edited region is both `editable: true` and `hand_edited: true`:
  it CAN be regenerated, and that has to be asked for by name.
- `make check-artifacts` prints a hand-edited region as `?  ... not compared`
  rather than STALE. It is not what this code produces, by design - and
  reporting every region the owner has edited as stale is the fastest way to
  teach everyone to ignore that check.

Two things outside the router had to learn the same lesson, and both were a
`KeyError` on `l["to"]` waiting for the first cross-region link:
`world_gen.report`, which would have refused the very specs this endpoint
exists to accept, and `world_preview.render`, which cost the region its preview
silently while the write succeeded. The preview also falls back to
`graph / 220` for a world that carries no `grid`.

## Submit body

```json
{
  "name": "emerald-reach",
  "worlds": 6,
  "target_per_screen": 6.0,
  "size": 128,
  "theme": "a green valley giving way to cold highlands",
  "author": "rules",
  "overwrite": false
}
```

`target_per_screen` is the whole interface. **Do not think in tier names** - the
generator picks those per world so the target is actually hit after the biome
multiplier. For calibration, their tiers land at:

| tier | per 1000 tiles | per screen |
|---|---|---|
| sparse | 9 | 2 |
| normal | 18 | 4 |
| dense | 36 | 8 |
| horde | 62 | 14 |
| swarm | 89 | 20 |

Under **3 per screen a world reads as empty space**; over ~22 the socket cost
their own measurement records starts to bite.

`author` is `rules` or `llm`. The LLM picks **biomes only** - never counts - and
the response's `author` field says what actually happened, including when it
fell back and why. It also names anything it invented and dropped:

    "biomes authored by Qwen2.5-3B-Instruct-Q4_K_M (5/6 worlds);
     dropped invented biome(s): Arctic Tundra"

Model: **superseded 2026-10-01 by the gated brain (0013).** `worlds.py` calls
`text.run_text`, so the author is whichever brain `brain_engine.BRAINS` has
active — by default Qwen3.6-35B-A3B, not a llama.cpp router model. The example
string above still names `Qwen2.5-3B-Instruct-Q4_K_M` because `author` reports
whatever actually ran; expect a brain id there now.

The old route was `Qwen2.5-3B-Instruct-Q4_K_M` via llm-server, declared in
`downloader/models.txt`, at ~0.7 s warm and ~13 s on the first call after two
idle minutes. **That path is retired** along with `llm_engine` (compose profile
`legacy-llm`), and its 1.8 GB GGUF is dead weight on disk. Latency is now the
brain's; cold loads are bounded by `brain_engine`'s own ceiling, not by
`--sleep-idle-seconds`. **`WORLD_LLM_MODEL` does not exist** — verified
2026-10-13, the name appears in this document and nowhere in the code or
compose. Setting it does nothing; the brain is chosen by `brain_engine`.
A refusal (`TextRefused`, another model holds the card) is a rule fallback,
not an error.

Whatever the author, the spec is repaired to meet the flora and variety floors
before it is written - see [0008](../../decisions/0008-world-specs-for-something2.md)
D7. Both authors currently produce zero problems on a 6-world region.

## What a generated world carries

`key`, `name`, `grid`, `seed`, `width`, `height`, `chunk_size`, `biomes`,
`biome_cell`, `allowed_creature_types`, `density`, `level_band`, `is_entry`,
`allows_fast_travel` - plus, on the entry world, `entry_spawn` and `village`.

**Roads and waypoints** are emitted for every world:

- `roads` - `[[row, col], [row, col]]` polylines in TILE coordinates, one from
  the middle of the world to each edge it exits by. Structure is the other half
  of "not empty": scattered creatures on undifferentiated ground still read as
  nothing in particular, while a road tells a player where the exits are and,
  through `safe_road_radius: 2`, carves a corridor they can move along at the
  densities these regions now ask for.
- `waypoints` - exactly one, where the roads meet. Their
  `one_waypoint_per_world` migration makes that a constraint, not a preference.

The **entry village contains its own spawn point** and its `gate_edge` opens
onto an edge the world actually exits by. Both are true of every village in
vale-region, and neither is automatic: an offset village drops a new player
beside the walls rather than inside them, and a gate onto an unlinked edge opens
onto the map boundary.

## Tiles and pixels, and why it is checked rather than documented

Layout is in TILES; positions are in PIXELS, at **100 px per tile**
(`n * 100 + 50`) - matching vale-region, where a village at `min_row 44` has
`spawn_y 4550`.

The trap is that **no range check can catch a confusion between them**. Tile 64
of a 128-wide world is also a perfectly legal pixel coordinate, since 64 is
inside that world's 12800 px extent. The mistake survives every bounds test and
surfaces only as an entity huddled in the top-left corner of a seeded map.

What separates them exactly is the offset: every position emitted here is a tile
CENTRE, so a real pixel value is always `n * 100 + 50`. A tile index almost
never is. `world_gen.check_units` runs over every world on every generation -
via `report()`, so it cannot be skipped - and flags:

- a pixel field holding something that is not a tile centre, or is outside the
  world's pixel extent;
- a road point outside the tile grid, which is the same mistake in reverse,
  since roads are tiles.

`pens` and `chest` are **not** emitted. Those are authored content decisions -
their own 34-world region uses one chest - and inventing them would be filling
a world with things nobody chose.

## The report

The reason to look before downloading:

```json
{
  "ok": false,
  "totals": { "worlds": 6, "creatures": 3081, "min_per_screen": 4.3,
              "max_per_screen": 7.0, "creature_types": 29,
              "leaders": 0, "empty_worlds": 0 },
  "worlds": [ { "key": "...", "density": "horde", "per_screen": 7.0,
                "biome_multiplier": 0.5, "variety": 8, "leaders": 0,
                "flora": ["bush", "Tree", "Stone"], "verdict": "ok" } ],
  "problems": ["..."]
}
```

- `per_screen` is the only number a human can judge. `density` alone is
  misleading, because `biome_multiplier` scales it.
- `verdict` is `ok` / `EMPTY` / `CROWDED`.
- `variety` is distinct creature types. Under 5 a world reads repetitive
  whatever its density.
- **Creature counts are scatter only.** Their tiers also carry packs, whose
  shape was not fully visible in the source read, so every figure is a floor -
  never an overstatement. See `PACKS_NOT_MODELLED` in `world_gen.py`.

## Consuming it from something2

```
GET  /api/worlds                      # what exists
GET  /api/worlds/<name>/report        # decide
GET  /api/worlds/<name>/preview.png   # or look
GET  /api/worlds/<name>?download=true # take it
                                      # -> backend/seeds/maps/<name>.map.json
                                      # -> make seed-map SPEC=<name>
```

Only one spec should be seeded per database: their README warns that "two specs
seeded together leave the second one's worlds unreachable". Use
`make reseed-map SPEC=<name>` to replace.

## Looking at one in a browser

`http://<host>:8001/#worlds` opens the tab directly - tabs live in the URL hash,
so a tab is linkable, a reload keeps its place, and the back button steps
between tabs instead of leaving the app. `#maps` and `#ref-map` likewise.

## Verify

    docker exec sprite_generator python /app/scripts/smoke-world-gen.py
    docker exec sprite_generator python /app/scripts/smoke-world-api.py

34 cases in the first, no model, no GPU, under a second. They assert their tier table is
reproduced exactly, that the Meadow trap really does land at 2/screen, that
solving beats naming, that links are reciprocal, that leaders stay rationed, and
that the emitted shape matches `vale-region.map.json`.

`smoke-world-api.py` covers the ROUTER - 18 cases over files, sidecars and
lifecycle, including the hand-edit ones, which are all about losing work: a
stale `If-Match`, a refused write that wrote anyway, a regeneration over an
admin's map. None of those raises anything on its own. Both suites run inside
`make test-maps`.
