# Maps: implementation plan

Plans [decisions/0007](../../decisions/0007-maps-paint-then-quantize.md) against
[contract.md](contract.md). The seven decisions there are settled and not
re-litigated here. This document closes the six open questions and cuts the work
into slices that each produce something you can look at.

## 0. The blocking question, closed

**Q1: how does a painted biome colour bind to a terrain tile?**

**Decided: the declared terrain set forces the palette. Quantization IS the
binding.** You declare terrains up front - each with a name, a generation prompt
and a colour - and the biome painting is snapped to exactly those colours. The
tile id is then a direct lookup, not a match.

The plumbing already exists and needs no new algorithm:

| Piece | Where | Does |
|---|---|---|
| `pixelate(img, ..., palette=)` | `pixelate.py:323` | Already accepts an **explicit** palette rather than extracting one |
| `snap_to_palette(rgb, palette)` | `pixelate.py:114` | Nearest entry, matched in **Lab** - perceptual, not RGB distance |
| `palette_of(arr)` | `measure.py:110` | Exact distinct colours, most common first - the source of default terrain colours |

Why this over auto-extract-then-label, which was the obvious alternative:

- **Auto-extraction gives N anonymous colours that still have to be named.** That
  is a labelling step per map, forever. Forcing the palette moves the naming to
  the terrain set, where it is authored once and reused.
- **It is deterministic.** Every pixel lands on a declared terrain or the run is
  wrong in a way that is visible immediately, rather than a nearest-match that is
  quietly 8% wrong.
- **"They must agree" becomes structural**, not a property to be tested.

**The risk this creates, and its mitigation.** Forcing a palette the LoRA was not
trained toward gives muddy assignments - a forest painted in a green that is
nearer your declared *swamp* green than your *forest* green. So **default terrain
colours are derived from the map references' own measured palette**:
`palette_of()` returns most-common-first, and those are by construction the
colours the reference art actually uses. Declared colours start as the ones the
model is already inclined to paint.

## The other five, closed

| | Question | Decision |
|---|---|---|
| Q2 | What `measure_map()` measures | Palette (count + entries, most-common-first), palette-locked verdict, size and aspect. `usable` = bounded palette. It feeds Q1's defaults, which is its real job |
| Q3 | Map size | Default **64x64 tiles**; bounds 32..256. Biome painting is **1 px per tile**, so the painting is literally 64x64 px. Terrain ids fit a byte with room to spare |
| Q4 | Recomposite or composite-on-read | **Recomposite on change, write-through.** The picture is served as a static PNG that something2 caches; composing per request re-pays the cost forever. Bump a version on change |
| Q5 | Which GGUF | A **3B instruct at Q4_K_M** (~2 GB), CPU. Use llama.cpp **GBNF grammar** to constrain output to the region-graph schema - this is the reliability mechanism, not prompt wording. *Needs verifying against `/models`; see Assumptions* |
| Q6 | Facade key | **By name.** Maps are authored here and named. Prompt-keying is fragile; id-keying means copying UUIDs by hand into the something2 admin |

## Scope

**Included:** `map` reference kind and tab; `measure_map()`; map LoRA training
through the existing path; terrain sets; biome painting; quantization to a
tilemap; picture compositing; entity placements with library-first + rule
scatter; provisional state; CPU region graph; sync cache-reader facade keyed by
name.

**Excluded:**

- **something2 client code.** Per 0007 D5, the picture crosses over the existing
  connector and the tilemap is served-but-unread. No laptop-repo work in this
  plan.
- **SOMET-334 (submit/poll).** Stays open, stays the right eventual fix.
- **The full `tasks.py` split.** See the precondition below - this plan avoids
  making it worse rather than fixing it.
- **A map editor UI.** Terrain sets are authored; entity positions are not
  hand-placed.
- **Multi-layer terrain** (cliffs, bridges, overpasses). One terrain layer.

## Precondition, and a deliberate compromise

`project-context.md` says split `tasks.py` (150 KB) before adding paths. A full
split is its own project and would block maps for days.

**Compromise: no map code goes into `tasks.py`, at all, from the first commit.**
Map tasks live in their own module, following the `tile_geometry.py` precedent -
that module exists precisely because the worker must import geometry without
dragging in a FastAPI router. The full split stays a separate ticket and this
plan does not enlarge the problem.

## Slices

Each slice is end-to-end and ends in something visible.

### Slice 1 - Reference - Map tab (no GPU)

Add `map` to `KINDS` (`references.py:46`) and `REFERENCE_KINDS`
(`training.py:44`); write `measure_map()` into `measure.py` MEASURERS; add the
kind to `api.ts:154`, `App.tsx` TABS, and `ReferenceTab.tsx` COPY/HEADLINE.

*Verifiable:* drop ~10 map images, see palette, colour count and a usable/why
verdict per card. No migration, no GPU, no worker restart beyond the API.

### Slice 2 - Train the map adapter

Run the existing training path with `kinds: ["map"]`. **A separate adapter from
characters and tiles**, for the reason `training.py` already documents: one
trigger cannot mean two things.

*Verifiable:* a run reaches `done` and an adapter appears in `/models/loras`.
Blocked until >= 8 usable references exist (`MIN_IMAGES`).

### Slice 3 - First map you can look at (no entities, no LLM) - MOSTLY DONE

Shipped 2026-08-26: `map_geometry.py`, `map_tasks.py` (registered through the
Celery app's `include`, so `tasks.py` gained one name rather than a stage),
`maps.py` (`POST /api/maps`, `GET /api/maps/{id}`, `GET /api/maps`), the Maps
tab, and `scripts/smoke-map-build.py`. Verified end to end on a real job row:
32x32 grid -> 2048x1024 picture, ids round-tripped exactly, zero interior holes.

Two departures from the plan as written, both deliberate:

- **A terrain may name an existing tile instead of generating one.** Library
  first, exactly as D6 has entity placements work. A map whose terrains all
  reuse tiles costs no GPU at all, which is what made the path testable before
  the adapter exists.
- **`painting_from` accepts a map reference as the layout.** Same reason: it
  decouples this slice from Slice 2 entirely. The generate-from-prompt path is
  written but unproven, because without the adapter its output is not worth
  looking at.

Storage reuses `sheet_path` (picture) and `atlas_path` (tilemap JSON), so
`/api/jobs/{id}/sheet` serves the picture unchanged and **no migration was
needed**.

**Still open in this slice:** the generated-painting path has never produced a
map worth keeping, and cannot until Slice 2 trains the adapter.

Terrain set in the Maps tab; `POST /api/maps`; `kind='map'` job; biome painting
at 1 px per tile with the forced palette; quantize to `layers.terrain`;
generate one tile per terrain through the existing tile path; composite the
picture. `GET /api/maps/{id}`.

*Verifiable:* a 64x64 map renders with no seams, and `GET /api/maps/{id}` returns
a grid whose every id resolves to a terrain. **This is the slice that proves or
kills the whole design.**

**Finding, 2026-08-26: `diamond_mask` could not tessellate, and never could.**
The deterministic half of this slice is built (`map_geometry.py`,
`scripts/smoke-map-build.py`, 11/11) and it failed on first run with 450
transparent pixels inside a composited field. The cause was not the new code:
`tile_geometry.diamond_mask` drew its polygon with vertices at `(width-1,
height/2)` and `(width/2, height-1)`, which makes the rhombus asymmetric -
measured row widths `1,5,9,13,16,20,24,28,32,28,23,19,14,10,5,1`, summing to
**248 opaque pixels of the required 256**.

Tiles sit on a `(width/2, height/2)` lattice, so tessellation requires

    w(y) + w(y + height/2) == width      for every y

and the old mask gave `1 + 32 = 33`. The docstring asserted the width-1 form was
what made tiles interlock; it was what stopped them. Rebuilt row by row, each
row measured from its own midline, and the invariant is now asserted directly on
the mask at four tile sizes so it cannot regress.

This is a **pre-existing defect in the shipped tile stage**, not a map problem -
`cut_tile` now returns exactly `width*height/2` opaque pixels with no partial
alpha, where it previously returned 3% fewer. Ground tiles generated before this
carry the old shape and will show pinholes when tiled; regenerate them if that
matters.

### Slice 4 - Entity placements and provisional - DONE

Shipped 2026-08-27: `map_geometry.scatter`, `map_tasks._populate`, the
`scatter` rules on `MapSpec`, and the provisional state. Verified end to end on
a real job: 32 entities over two layers, 24 placed from the library and 8
pending, `complete: false` with `pending: ["wolf"]`, none on water and none on
the road.

Four decisions worth keeping:

- **Terrain is named in a rule, not indexed.** Ids are positional and a caller
  should not have to count the terrain list to say "on grass". A typo is a 400
  naming the valid options, rather than a rule that silently matches nothing.
- **Spacing, not pure random.** Uniform sampling clumps - three trees on
  adjacent tiles beside a bare stretch reads as a mistake rather than as
  nature. Rejecting candidates within `spacing` costs nothing.
- **Scatter avoids roads.** A tree in the middle of the road is not a charming
  detail, and `safe_road_radius` exists to keep that corridor clear.
- **The placeholder is deliberately ugly** - magenta box and cross. One that
  reads as finished is worse than a missing one, because it gets cached
  downstream as the real thing.

**The resolver landed 2026-08-27**, closing the gap this section used to
record. `resolve_map_props` generates each missing `want`, files it in the prop
library, and composites the map again; `scripts/smoke-map-resolve.py` covers it
in 23 cases. Verified end to end with no GPU: a 24x24 map with 36 placements
went from 36 placeholders to 26 placed, stayed `complete: false`, and named the
one want that could not be generated.

Five decisions worth keeping:

- **The resolver is its own `jobs` row** (`kind='map_props'`), not a
  continuation of the map job. That was the whole trick for the reaping this
  slice owed: `fail_stranded_jobs` already sweeps `jobs` without caring about
  `kind`, so a resolver whose Celery message was lost is reaped for free. No
  new sweep, no new table.
- **`props_status` on the served map is what makes the reaping useful.** The
  reaper turns a lost job into a `failed` row, but a row nobody reads changes
  nothing - `complete: false` still means both "working" and "dead". The route
  now reports which, with `final` saying whether waiting will help.
- **A finished resolver on an incomplete map reads `partial`, never `done`.**
  Found by running it: the live check reported `state: "done"` on a map still
  missing a wolf, which sends a caller back to wait for art that is not coming.
  Same infinite wait, arriving by a different route.
- **One resolver row per map, not one per prop.** N rows would each have to
  re-composite and rewrite the same picture, racing on it, while the solo pool
  serialises them anyway.
- **The resolver re-draws, it does not re-scatter.** It reloads the placements
  and the saved terrain tiles from disk, so it costs no GPU and nothing can
  move. A map whose trees shuffled when a windmill resolved would be a baffling
  bug to be handed.

Two things fell out of it. Terrain tiles are now **saved per map and named in
the wire format**, which the contract had always documented and the build had
never emitted - so a consumer can finally draw the grid rather than only look
at the picture. And a resolved prop is **registered in the gallery**, so the
library grows and the second map wanting a windmill pays nothing.

**Still open:** maps built before this exist without saved terrain tiles, so
they can never be re-composited. Nothing tries - they have no `props_job` - and
they stay provisional until rebuilt.

### Slice 4 - original scope

Library-first lookup, rule scatter by biome density, queued jobs for gaps,
placeholder art, `complete` / `pending`. Reaping for stranded entity jobs,
modelled on `aece983`.

*Verifiable:* a map with scattered trees; a map naming a prop you do not have
serves `complete: false` with a visible placeholder and resolves itself once the
entity job lands.

### Slice 5 - Region graph - DONE

Shipped 2026-08-27 as `regions.py`, 25 smoke cases in `scripts/smoke-regions.py`.

**Verified on a real build with no GPU:** `port Port Haven` at (12,11) sat on
actual shoreline with nothing relaxed, a road of 14 tiles reached it, all six
places were on land, and no scatter prop landed on a landmark. Re-rolling the
seed moved every place and left the terrain byte-identical.

Five decisions worth keeping:

- **The graph is read OFF the finished terrain, not authored before it.** This
  reverses the pipeline in [contract.md](contract.md), and it is the decision
  the whole slice rests on. A landmark has to sit on real ground, and which
  ground is real is not known until the painting has been quantised. Authoring
  first leaves only two moves afterwards - reject a good graph over one town in
  a lake, or move it silently, at which point the graph and the terrain
  disagree and the graph was never authoritative. Derived from the terrain, it
  *cannot* disagree.
- **The JSON Schema is the reliability mechanism, and it is built from the
  map.** llama.cpp compiles `response_format: json_schema` to GBNF itself, so
  no hand-written grammar was needed - Q5's assumption held, better than
  expected. But the schema has to be per-map: told in prose that "ports need
  shoreline", the 3B model named three ports on a landlocked map. Removing
  `port` from the enum makes it unsayable. **The prompt is a hint; the schema
  is the rule.**
- **Constraints relax one at a time, and say which.** A port on a map with no
  sea is still placed, and carries `relaxed: ["shoreline"]`. A graph that
  silently ignores its own hints is worse than one that admits it could not
  honour them.
- **An unroutable pair is dropped, not forced.** Roads are axis-aligned L and Z
  routes, never a pathfinder: a shortest path round a coastline is a staircase
  that reads as a goat track. Two places with no straight dry route between
  them are simply not connected by road, and the graph says so.
- **A placed region becomes an ordinary pending prop**, keyed on its KIND
  rather than its name - a library keyed on "Saltmere" is a library of one.
  Slice 4's resolver and reaper cover them with nothing new built.

**The Z route search covers the whole grid, not the span between the two
places.** Found by a failing test: two towns on the same side of a bay have no
dry column *between* them - the way out is to go the other way first.

*Original criterion, met:* a town sits at a river mouth and a road reaches it.
Re-rolling the graph changes the semantics while terrain stays put.

### Slice 6 - The something2 facade - DONE

Shipped 2026-08-27: `maps.resolve_name`, `GET /api/maps/by-name/{name}`, the
`map:` branch in the A1111 facade, `scripts/smoke-map-facade.py` (12 cases) and
`scripts/verify-map-facade.py` for the cross-machine check.

Four decisions worth keeping:

- **The name travels in the PROMPT, as `map:<name>`.** The contract promises
  zero laptop-side code, and something2's connector substitutes into a fixed
  body shape - the prompt is the only field guaranteed to carry an arbitrary
  string. This is NOT the prompt-keying Q6 rejects: that was matching whatever
  text a caller sent against whatever maps happened to exist. An explicit
  prefix is unambiguous, cannot be hit by accident, and fails loudly.
  `override_settings.map` is accepted too - the same two-channel shape `cutout`
  and `lora_scale` already use, and for the same reason.
- **It never generates.** A cache miss is a 404 in milliseconds. A facade that
  falls through to the model looks identical on a hit and destroys the caller
  on a miss: a map build is minutes to hours inside a request with a 240s
  budget. The smoke case asserts *nothing was queued*, not merely that the
  status was 404 - those are different claims and only one of them is the one
  that matters.
- **Only `done` rows whose files are still on disk resolve.** Right name, wrong
  state is the dangerous near-miss - it would have something2 seed a world from
  a file that is about to be overwritten.
- **`info` carries `complete` and `pending`.** A consumer caching `images[0]`
  needs to know whether the picture still has magenta placeholders on it. It is
  the one thing this response can say that a generated image never has to.

**Not verified from another machine yet.** `verify-map-facade.py` checks 401
without a bearer over real HTTP and passes; the authed half needs a scoped key,
which is the user's to issue rather than this session's to mint.

Sync cache-reader keyed by map name. Picture over the existing AI connector.

*Verifiable:* a script in the shape of `verify-something2-contract.sh` fetches a
named map's picture from another machine with a scoped bearer, and gets 401
without one.

## Verification strategy

Mirror what this repo already does rather than inventing a second style.

- **`scripts/check-map.py`**, the `check-sprite.py` analogue and equally
  non-optional. Asserts: every tile id resolves to a terrain; the painting is
  palette-locked to exactly the declared colours; no transparent hairlines
  between tiles; the picture's dimensions match the grid and projection.
- **A no-model smoke test**, per `smoke-sheet.py`: quantization and compositing
  under test with a hand-made painting and solid-colour tiles. Runs on CPU, in
  CI-time, with no GPU.
- **`scripts/validate-terrains.py`**, per `validate-actions.py`: assert the
  terrain set's colours are mutually distinguishable in Lab. Two terrains closer
  than a threshold cannot be quantized apart, and that is a config error worth
  catching before an hour of GPU rather than after.

## Acceptance criteria

1. Ten map images uploaded to Reference - Map, each measured, each with a
   verdict.
2. A map adapter trained from them and selectable.
3. A named 64x64 map generated end to end, whose picture has no seams and whose
   tilemap resolves every id.
4. The picture and the ground agree - spot-check three coordinates: what the
   picture shows is what `layers.terrain` says.
5. A map with a missing prop serves `complete: false` with a visible placeholder,
   and reaches `complete: true` without being re-requested.
6. A road in the region graph connects the two landmarks it names.
7. something2 fetches that map's picture by name, from another machine, with a
   scoped bearer, in one request.

## Assumptions

- **A suitable 3B GGUF is cached in `/models`, or one 2 GB download is
  acceptable.** Not verified - `/models` lives on WSL ext4 and was not inspected
  from Windows. Check with `scripts/list-models.sh` before Slice 5.
- >= 8 usable map references will exist before Slice 2. Below `MIN_IMAGES` the
  adapter memorises and the symptom appears hours later.
- One style profile per map. Tiles measured at different projection ratios do not
  tessellate.
- Terrain count stays under ~16. Beyond that, forced-palette separation in Lab
  gets unreliable - which `validate-terrains.py` is there to catch.

## Risks

Carried from 0007, plus two this plan introduces:

- **The composited picture is large.** Isometric width is `(w + h) * tile_w / 2`;
  a 64x64 map at 64 px tiles is a **4096 px** picture. That crosses the connector
  fine but is not free to composite or store, and it scales with the *sum* of the
  dimensions. Cap it, or downscale the served picture.
- **Forced palette can paint mud.** Mitigated by deriving defaults from the
  references' measured palette, but if the LoRA is weak the painting may not
  separate cleanly into the declared terrains. **Slice 3 is where this shows up**,
  which is why Slice 3 comes before entities and the LLM.
- `tasks.py` growth - addressed by the precondition above, not solved.
- Placeholder art does not exist and is load-bearing for Slice 4.
- A failed entity job stranding a map, per `aece983`.
- Two integration paths coexisting while D5 holds; the unread one rots unless
  `check-map.py` exercises it.

## Two hazards found by building it, 2026-08-27

Both are about the palette, both are silent, and both were measured on real map
references rather than reasoned about.

**A declared colour that does not match the art costs you the terrain.** A
reference whose sea is a muted blue, quantised against a navy `#2850c8`,
produced **2.4% water**. Against the sea's actual colour, 49%. `validate_terrains`
cannot catch this - it checks that the declared colours are far enough APART,
which says nothing about whether any of them is in the painting.

**A near-neutral terrain is a sink.** Grey sits close to the middle of Lab
space, so it is the nearest match for anything desaturated. Dropping a grey
`stone` from that same reference took water from **2.4% to 58% without touching
the water colour**. A mid-grey terrain quietly eats every washed-out region of
the painting.

Neither is knowable before the painting exists, so the build now reports
coverage warnings at the one moment it can: a terrain under 0.5% or over 85% is
named in the tilemap's `warnings` and in the worker log. This is the same
failure `validate_terrains` was written for - "a map silently missing a terrain
gives no hint which one it lost" - closed at the other end.

**Fixed 2026-08-27.** `GET /api/maps/palette/{reference_id}` reads the
colours off a reference and reports the share each would capture, and the Maps
tab has a button that fills them in. This is what `measure_map()` was built for
and what contract.md always said its job was.

It reports rather than decides: near-neutral suggestions are flagged as sinks
with their chroma, and a reference that cannot supply as many terrains as were
asked for says so - `extract_palette` caps at the distinct colours in the art,
so eight asked of a three-colour painting quietly returned three. That last one
was found by a test whose own premise was wrong, which expected the surplus to
come back badly separated rather than not at all.

The coverage numbers are what the REFERENCE quantises to, not a preview of a
generated map. A map painted from a prompt may land elsewhere, and the build's
own warnings remain the last word.

## The card filled up, and the first diagnosis was wrong, 2026-08-27

**The symptom.** After a map build called llama.cpp and then queued prop
generation, one prop failed with `CUDA driver error: device not ready` and a
retry failed with a `CUDACachingAllocator` internal assert. `nvidia-smi` showed
**11957 / 12288 MiB used with an empty job queue**. Restarting the worker
freed it to 213 MiB.

**The wrong diagnosis, recorded because it was convincing.** Plan Q5 assumed
*CPU* llama.cpp; the deployed preset is `--n-gpu-layers 99`, so the region
graph had just put a 3B model on the same 12 GB card as SDXL. That fit the
timeline exactly - the failures began the first time a map build called the
LLM - and the recommendation was to move the text model to CPU.

**The measurement that killed it.** `tasks.describe_device` on the worker, with
the queue empty and llama.cpp asleep:

```
allocated   6.63 GB     what the pipeline is really using
reserved   11.32 GB     what PyTorch is sitting on
total      12.00 GB
```

Then the decisive one: **waking llama.cpp from sleep moved VRAM by exactly
0 MiB.** Contention was never it. `nvidia-smi` reports the RESERVED figure, and
the worker simply never handed its caching allocator's spare blocks back - so
the card read as full with nothing running, and the next large allocation hit
a pool with no contiguous room. That is the fragmented-allocator failure
ADR 0005 already describes.

**The fix**, in `map_tasks._release_vram`: one `empty_cache()` when a whole
resolve finishes, not between props. Measured live through a real build:
`released 4.67 GB of reserved VRAM (11.32 -> 6.65)`. Reserved now tracks
allocated to within 0.02 GB. The pipeline stays cached deliberately.

**Do not move the text model to CPU on the strength of the paragraph above** -
that was the wrong call, and the version of this document committed in
`518a848` recommended it.

Two things worth keeping from the mistake. The hypothesis was never tested
until it was refuted; waking the model was a thirty-second check available from
the first minute. And `nvidia-smi` answers a different question from the one
being asked - it reports what a process has reserved, not what it needs.

**How much headroom this leaves is not established.** The card now shows
~5.35 GB free with the pipeline resident, and it is tempting to call that
enough for a training run. The only peak figure anyone has for the trainer
(`scripts/train-lora.py`, ADR 0006) predates the current trainer and was never
re-measured against it. Treat the headroom as unknown until a run is executed
and its peak recorded.

`POST /api/maps/{id}/resolve` was written during this and is still worth
having: it retries only the missing art, at a different seed each attempt,
without repainting terrain that was never wrong.
