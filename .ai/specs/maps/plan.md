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

### Slice 3 - First map you can look at (no entities, no LLM)

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

### Slice 4 - Entity placements and provisional

Library-first lookup, rule scatter by biome density, queued jobs for gaps,
placeholder art, `complete` / `pending`. Reaping for stranded entity jobs,
modelled on `aece983`.

*Verifiable:* a map with scattered trees; a map naming a prop you do not have
serves `complete: false` with a visible placeholder and resolves itself once the
entity job lands.

### Slice 5 - Region graph

CPU llama.cpp behind a GBNF grammar; validate against terrain (nothing in water,
nothing out of bounds); rules scatter the rest.

*Verifiable:* a town sits at a river mouth and a road reaches it. Re-rolling the
graph changes the semantics while terrain stays put.

### Slice 6 - The something2 facade

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
