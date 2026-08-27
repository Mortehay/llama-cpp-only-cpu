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
    GET    /api/worlds/{name}/report        -> is it empty?  (without fetching the spec)
    GET    /api/worlds/{name}/preview.png   -> one screenful per world
    DELETE /api/worlds/{name}

Scopes: `generate` for POST/PATCH/DELETE, `read` for the rest.

A region is three files: `<name>.map.json` (their format, untouched),
`<name>.preview.png`, and `<name>.gen.json` - the generation parameters, kept
BESIDE the spec rather than inside it so their seeder never sees a field it did
not ask for. The sidecar is what makes `PATCH` possible: without it an edit
would have to reverse engineer the request from its own output. A region
generated before it existed lists as `editable: false` and 409s on PATCH.

Generated files live in `WORLDS_DIR` (default `/app/images/worlds`) as
`<name>.map.json` and `<name>.preview.png`. The **spec file is the artefact** -
it is what `make seed-map SPEC=<name>` reads - so the listing is a directory
rather than a table, and there is no migration.

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

Model: `Qwen2.5-3B-Instruct-Q4_K_M`, in `downloader/models.txt`. ~0.7 s warm,
~13 s on the first call after two idle minutes (llama.cpp's router loads on
demand and `--sleep-idle-seconds 120` unloads); that cold start is retried
internally, not surfaced as a failure. Override with `WORLD_LLM_MODEL`.

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

27 cases, no model, no GPU, under a second. They assert their tier table is
reproduced exactly, that the Meadow trap really does land at 2/screen, that
solving beats naming, that links are reciprocal, that leaders stay rationed, and
that the emitted shape matches `vale-region.map.json`.
