# 0007 - Maps: paint then quantize, and why the LLM does not draw the grid

Date: 2026-08-26
Status: Proposed. Extends [0006](0006-react-references-and-training.md) with a
fourth reference kind and a fourth job kind. Does not revise 0005 or 0006.

## Why this exists

The ask was "a maps tab and a maps reference tab, to train an LLM to generate
beautiful maps for a pixel-art RPG, served to something2". Taken literally that
is not buildable, for two reasons worth recording because both are easy to
re-propose:

1. **You cannot train a text LLM on map images.** Making an LLM emit layouts
   needs (image -> tile grid) supervised pairs. That means hand-labelling every
   collected example into a grid first, which is the whole project. What *can*
   be trained on collected map images is a **diffusion LoRA**, which is already
   built - `training.py` is one reference kind away from it.
2. **A 12 GB card cannot hold a useful GGUF and SDXL at once.** The README says
   so outright, and `stats_collector` polling `llm_engine` mid-run has already
   caused a model load during a sheet build.

So the LLM stays, but with a much smaller job than the one it was given.

## The requirement that decided everything

The map must be **walkable ground and a picture that agree** - the menu map has
to show the town where the player actually finds it. Once that is required, a
whole family of designs dies: any design that generates the picture and the grid
separately can produce two things that disagree, and no amount of prompt care
fixes it.

## Decisions

### D1. The biome painting IS the layout. Paint, then quantize.

One low-res, palette-locked denoise produces a biome painting - one colour per
terrain. The tilemap is derived from it by matching each pixel to a terrain and
compositing the corresponding tile through the existing rhombus mask. The map
picture is composited from those same tiles.

Picture and ground cannot disagree, because they are the same artefact read
twice.

Rejected alternatives:

| Candidate | Verdict |
|---|---|
| **Paint then quantize** | **Chosen.** Reuses `pixelate.py` palette-lock and `measure.py` palette machinery - the two strongest existing pieces - and satisfies "they must agree" by construction |
| Diffuse a full-resolution map | Beautiful and unusable. No tessellation guarantee, no grid, not walkable |
| Noise / WFC layout, no diffusion | Playable and characterless. Nothing the collected references teach ever reaches the output, which defeats the reference-and-training half of the ask |
| LLM emits the tile grid | Rejected, see D2 |

### D2. The LLM writes a region graph. It never writes coordinates.

Split by cardinality, which is also a split by what each method is good at:

- **LLM, tens of items, semantic.** Town at the river mouth; road from town to
  bridge; boss lair far from spawn. This is the one thing noise cannot do, and
  it is why an LLM is in the design at all.
- **Rules, hundreds of items, statistical.** Trees where biome is forest at
  density 0.3. Blue-noise scatter under a terrain mask.

Asked for ~400 placements an LLM reliably produces trees inside lakes, two towns
on one tile, out-of-bounds coordinates, and output truncated at the token limit.
The repair layer for that becomes most of the work, and the result is still not
reproducible run to run. A ~2 KB region graph is small enough to validate
completely and cheap enough to re-roll.

### D3. CPU llama.cpp, small model. It must not touch the card.

The region graph is ~2 KB of JSON. A 3B Q4 on the i3-8100 takes tens of seconds,
which is nothing against a map build, and it can run **while SDXL paints**
rather than fighting it for VRAM. The base compose already uses the CPU `full`
tag.

The GPU alternative is worse than it looks: inference saves ~2 s and costs two
extra ~90 s model loads, plus a new way for a map job to OOM a sheet job.

### D4. A sync cache-reader facade here, not submit/poll in something2

something2 is sync-only - one POST, one response
([contract.md](../specs/something2-provider/contract.md), their
`remoteImageProvider.js:11`). This already makes the tile path "not connectable
today" per [api-auth-lockdown/plan.md](../specs/api-auth-lockdown/plan.md).
Maps inherit that wall exactly, and harder: a map build is minutes to hours.

Maps are therefore **authored here** and **collected there**. The facade returns
an already-built map instantly; something2 never waits on generation.

This is chosen over teaching something2 to poll (their SOMET-334) despite
SOMET-334 being the real root cause and fixing tiles, maps and sheets at once.
The reason is ordering, not merit: the map schema is going to change several
times, and client code written against a moving schema is rewritten. SOMET-334
stays the right eventual fix.

### D5. The picture crosses now; the data waits

something2's connector carries images only (`images[0]`). Walkable ground is
data. Both cannot cross today.

So the **map picture** goes over the existing connector immediately, with zero
laptop-side code, and the tilemap plus entity placements are served at a stable
URL, documented, and deliberately unused until something2 is ready. The schema
stays free to move for exactly as long as nothing consumes it.

Encoding the tilemap into an index PNG to sneak it through the image connector
was considered and rejected: something2 still needs decode code, so it does not
actually reach zero, and it buys that non-zero cost with a 255-terrain cap and
an undebuggable wire format.

### D6. Entity placements reference the library first, and generate the gaps

A map places **entity assets that already exist**; props that do not exist yet
get a queued generation job and a placeholder in the meantime. Generating every
prop per map would put a ten-prop map at roughly five hours on this card, and
re-rolling the map would re-pay all of it.

### D7. Provisional is a served state, not an error

A map with unsatisfied placements is served with `complete: false`, a `pending`
list, and placeholder art. Withholding it means one failed entity job blocks a
whole map indefinitely - which is the shape of the stranded-training-run bug
fixed in `aece983`, and the reason that failure mode is called out in Risks
below rather than discovered later.

## What this costs in this repo

Smaller than it looks. `jobs.kind` is TEXT with no CHECK constraint and
`reference_assets.kind` likewise, so **neither new kind needs a migration**:

| Change | Where |
|---|---|
| `map` reference kind | `references.py:46`, `training.py:44`, `measure.py` MEASURERS, `api.ts:154`, `App.tsx` TABS, `ReferenceTab.tsx` COPY/HEADLINE |
| `map` job kind | a `tiles.py`-shaped router, no schema change |
| `measure_map()` | New. See Open questions - this is not yet specified |
| Maps tab | Must use `useAuthedObjectUrl` from the first commit, per the `Tiles.tsx` precedent |

## Risks

- **`tasks.py` is 150 KB**, and `project-context.md` already says to split it
  before adding tile and training paths. The map path is larger than either.
  Adding it unsplit is the single most likely source of regret here.
- **Placeholder art is load-bearing and does not exist.** If it is not obviously
  provisional, a map that looks finished and is not gets cached downstream as
  final.
- **A failed entity job strands a map at `complete: false` forever**, exactly as
  stranded runs blocked training before `aece983`. Whatever reaping that commit
  added should be the model for this, not a second invention.
- **The map LoRA needs >= 8 usable references** (`MIN_IMAGES`). Below that it
  memorises, and the failure shows up hours later as output ignoring the prompt.
- **Two integration paths coexist** for as long as D5 holds: the picture over
  the connector, and the data at a URL nothing reads. The second one rots
  silently unless something exercises it.

## Open questions - closed 2026-08-26 in [specs/maps/plan.md](../specs/maps/plan.md)

All six are answered in that document's section 0. The one that blocked
everything downstream of D1 was **how a painted biome colour binds to a terrain
tile**, and the answer changes D1's shape enough to record here:

**The declared terrain set forces the palette; quantization IS the binding.**
Terrains are authored up front with a name, a prompt and a colour, and the biome
painting is snapped to exactly those colours - `pixelate()` already takes an
explicit `palette=` and matches in Lab. A tile id becomes a **lookup rather than
a match**, which is what makes "the picture and the ground agree" structural
instead of a property to be tested.

The alternative - extract N colours from the painting and label them afterwards -
was rejected because it makes naming a per-map chore forever and leaves every
assignment a nearest-match that can be quietly wrong.

One consequence worth carrying: **defaults for the declared colours come from the
map references' own measured palette**, so the forced palette starts as the one
the LoRA is already inclined to paint. Without that, forcing a palette the model
was not trained toward produces mud, and that failure surfaces in Slice 3.

The remaining five (what `measure_map()` measures, map size bounds, recomposite
vs composite-on-read, which GGUF, the facade key) are settled in the same
section and are not repeated here.

## What this does not do

- It does not teach something2 to poll. SOMET-334 remains open and remains the
  correct eventual fix for tiles, maps and sheets together.
- It does not make maps generatable on demand from something2. They are authored
  here.
- It does not train anything on the region graph. The LLM is prompted, not
  fine-tuned, and nothing in this ADR changes that.
