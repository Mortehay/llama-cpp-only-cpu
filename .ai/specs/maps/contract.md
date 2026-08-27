# Maps: job kind, wire format, and the something2 facade

> **DRAFT. Nothing consumes this yet, and that is deliberate** - see
> [decisions/0007](../../decisions/0007-maps-paint-then-quantize.md) D5. The
> schema is free to change for exactly as long as something2 reads only the
> picture.
>
> **All six open questions are now closed in [plan.md](plan.md) section 0.** The
> one that mattered - how a painted biome colour binds to a terrain tile - is
> answered "the declared terrain set forces the palette, so quantization IS the
> binding". Read that before this.

Vocabulary is load-bearing here. `biome painting`, `tilemap`, `region graph`,
`map picture` are four different objects - see [domain.md](../../domain.md),
"Map is overloaded".

## The pipeline

```
region graph      biome painting     tilemap          placements       picture
---------------   ----------------   --------------   --------------   ----------
CPU llama.cpp     map LoRA, low-res  quantize the     library lookup   composite
~2 KB JSON        palette-locked     painting to      + rule scatter   tiles +
regions, roads,   one colour per     terrain ids      + queued jobs    entities
landmarks         terrain            via measure.py   for the gaps
                                     palette match
```

The **biome painting is data, not art**. It is the layout in visual form, and it
is what makes the picture and the ground incapable of disagreeing. Nothing shows
it to a user.

## Job kind

A map is a row in `jobs` with `kind='map'`, polled through the same
`GET /api/jobs/{id}` as sheets and tiles. No migration: `jobs.kind` is TEXT with
a `'sheet'` default and no CHECK constraint.

This follows `tiles.py` exactly, for the reason stated in its docstring - one
polling contract, several job shapes - and the map router should be
`tiles.py`-shaped for the same reason: a map spec has nothing in common with a
sheet spec.

```
POST   /api/maps           -> 202 {job_id, size, terrains, poll}
GET    /api/jobs/{id}      -> the job, as for any kind
GET    /api/maps/{id}      -> the tilemap + placements  (409 until terrain done)
GET    /api/jobs/{id}/sheet -> the map picture PNG      (409 until done)
```

`/api/maps/{id}` returns **409 until the terrain is final**, not until the map is
complete. A provisional map is a served state, not an unfinished one.

## Layers

Four authored layers, and the split between them is load-bearing:

| # | layer | shape | drawn |
|---|---|---|---|
| 1 | `terrain` | grid, one id per tile | Pass A, flat |
| 2 | `roads` | grid, one id per tile | Pass A, over terrain |
| 3 | `creatures` | object list | **Pass B, shared depth sort** |
| 4 | `props` | object list | **Pass B, same sort** |

**Layers 3 and 4 are separate data and ONE draw pass.** In an isometric view a
prop one tile in front of a creature must cover it, and the same creature one
tile further forward must cover the prop. Two flat passes cannot express that -
whichever is drawn second always wins. They stay separate in the FILE so they
can be authored, filtered or regenerated independently, and merge only at
compositing time. something2's renderer reaches the same conclusion for its own
Pass B: *"ground items must join the same sort rather than being drawn in a
later pass, or they would render on top of entities they are actually behind"*.

Within one depth, props settle behind creatures, so a character standing in a
doorway reads as being in front of it.

**Roads are ground, not objects.** They are painted in Pass A over the terrain
of the same cell, which is what stops a road from ever drawing over a creature
standing on it. `road_layer` also refuses to lay road across unwalkable terrain
- a road across a lake is not a road.

**Not layers here:** players and projectiles. Both are runtime - their renderer
draws blasts and VFX *after* the depth-sorted pass, and a player is never in a
map file. A `-1` under-ground layer is unused and deliberately not reserved.

Object sprites are anchored by their **bottom centre** to the middle of their
tile, because a sprite stands on the ground rather than filling the diamond.
Out-of-bounds placements raise rather than being clamped to the edge, where they
would look like art bugs instead of generation bugs.

## Populating layers 3 and 4

A `scatter` rule on the map spec says what goes where:

```json
{ "layer": "props", "asset": "core_oak_a1b2.png", "want": "oak tree",
  "terrain": ["grass"], "density": 0.06, "spacing": 3 }
```

- **`terrain` is named, not indexed.** Ids are positional; a caller should not
  have to count the terrain list to say "on grass". An unknown name is a 400
  listing the valid ones, not a rule that silently matches nothing.
- **`asset` costs no GPU; `want` is a gap.** Library first, per
  [0007](../../decisions/0007-maps-paint-then-quantize.md) D6. A rule with only
  a `want` is still placed - with placeholder art and `status: "pending"` - so
  the map is usable now and improves later.
- **`density`** is the fraction of matching tiles to occupy;
  **`spacing`** is the minimum tiles between two of the same rule. Spacing is
  not decoration: uniform sampling clumps, and three trees on adjacent tiles
  beside a bare stretch reads as a mistake rather than as nature.
- **Scatter avoids roads.** A tree in the road is not a charming detail.
- Placement is **deterministic from the map's seed**, so re-rendering a map
  does not move its contents.

The placeholder is deliberately ugly - a magenta box and cross. One that reads
as finished is worse than a missing one, because it gets cached downstream as
the real thing.

## Wire format

```json
{
  "id": "...",
  "complete": false,
  "pending": ["windmill", "shrine"],
  "size": { "w": 128, "h": 128 },
  "projection_ratio": 2.0,
  "terrains": [
    { "id": 0, "name": "grass", "tile": "/images/tile_grass_a1b2.png",
      "color": "#4a7c3f", "walkable": true }
  ],
  "layers": {
    "terrain": [[0, 0, 1, 1]],
    "roads":   [[0, 1, 1, 0]]
  },
  "entities": [
    { "asset": "oak_tree_a1b2", "x": 12, "y": 7,
      "layer": "props", "status": "placed" },
    { "asset": "wolf_c3d4", "x": 12, "y": 8,
      "layer": "creatures", "status": "placed" },
    { "asset": null, "want": "windmill", "x": 20, "y": 20,
      "layer": "props", "status": "pending", "job": "..." }
  ],
  "region_graph": { }
}
```

Three things a consumer must get right:

- **`complete: false` is not an error.** The terrain is final and walkable; some
  entity placements are still standing in with placeholder art and will resolve
  without the map being re-requested. A caller that treats it as failure abandons
  a usable map; a caller that caches it as final keeps a placeholder forever.
- **`terrains[].color` is the binding back to the biome painting**, not a display
  hint. It is the palette entry that produced every tile carrying that id.
- **`projection_ratio` must match the style profile the tiles were cut at.** Tiles
  measured at different ratios do not tessellate, and the seam is invisible on
  one tile and obvious across a field.

`layers.roads` is present and empty rather than absent when unused, so a
consumer never has to branch on whether the key exists. `entities` carries
creatures and props in one list with a `layer` tag - see Layers above.

`region_graph` is carried through for provenance and debugging. Nothing renders
it.

## The something2 facade

something2 is sync-only: one POST, one response, no polling
([something2-provider/contract.md](../something2-provider/contract.md)). A map
build is minutes to hours, so **maps are authored here and collected there**.

The facade is a cache reader. It returns an already-built map immediately and
never triggers a build inside the request.

| | |
|---|---|
| Crosses today | The **map picture**, through the existing AI connector (`images[0]`). Zero laptop-side code |
| Served, unread | The **tilemap + placements** at `GET /api/maps/{id}`, until something2 gains a client |
| Never | Generation on demand. That needs their SOMET-334 |

**The facade is keyed by map name** ([plan.md](plan.md) Q6). Maps are authored
here and named; prompt-keying is fragile and id-keying means copying UUIDs into
the something2 admin by hand.

## Reference kind

`map` becomes a fourth reference kind alongside `core`, `sprite`, `tile`. No
migration - `reference_assets.kind` is TEXT.

Touch list: `references.py:46`, `training.py:44`, `measure.py` MEASURERS,
`api.ts:154`, `App.tsx` TABS, `ReferenceTab.tsx` COPY/HEADLINE.

Training uses the existing path with `kinds: ["map"]`. **Train maps as their own
adapter**, for the reason `training.py` already records for tiles: one adapter
over several kinds teaches a single trigger to mean two things at once.
`MIN_IMAGES = 8` applies - below it the adapter memorises, and the symptom
appears hours later as output ignoring the prompt.

**`measure_map()` measures the palette** - count, entries most-common-first, a
palette-locked verdict, size and aspect. A tile upload answers "what camera does
this world use?"; a map upload answers "what colours does this world's terrain
come in?", and those become the **default terrain colours** a map's forced
palette starts from. That is its real job, not the verdict copy.

## Settled

Every question this document opened is answered in [plan.md](plan.md) section 0.
The load-bearing one:

**The declared terrain set forces the palette.** Terrains are authored up front
with a name, a prompt and a colour; the biome painting is snapped to exactly
those colours via `pixelate(..., palette=)`, which already accepts an explicit
palette and matches in Lab. So a tile id is a **lookup, not a match** - and
"picture and ground agree" becomes structural rather than a property to test.

Defaults for those colours come from `measure_map()` above, so the declared
palette starts as the one the reference art already uses.

## Prerequisites

- **Placeholder art must exist and must look obviously provisional** before D7
  ships. A placeholder that reads as finished is worse than a missing map.
- **Map jobs need the reaping that `aece983` gave training runs.** A failed
  entity job otherwise strands its map at `complete: false` permanently.
- **The Maps tab must fetch images with the bearer into a blob URL**
  (`useAuthedObjectUrl`), as `Tiles.tsx` does. A plain `<img src>` against an
  authed route 401s the moment a key is minted.
