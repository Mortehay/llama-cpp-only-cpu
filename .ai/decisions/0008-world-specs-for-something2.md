# 0008 - Generate something2 world specs, and solve density rather than name it

Date: 2026-08-27
Status: Proposed. Independent of [0007](0007-maps-paint-then-quantize.md) - that
ADR is about map ART, this one is about world DATA. They share no code.

## Why this exists

The ask was a tab that generates worlds for something2 with better tile and
entity distribution - "not so many empty space, creatures in normal amount, and
other entities as trees and stones in normal amount".

Before designing anything, their repo was read. Three findings changed the plan,
and all three are the reason this is a small module rather than a large one.

## What was measured in something2, 2026-08-27

| Finding | Consequence |
|---|---|
| **They already have world generation.** `make seed-map SPEC=vale-region` reads `backend/seeds/maps/*.map.json`; catalogs come from `make seed-catalogs`. | Nothing here invents a format. This produces THEIR file. |
| **`DENSITY_TIERS` is per 1000 tiles**, and one screen is ~225 tiles (fixed 1280x720 canvas, 128x64 iso diamond). sparse 9, normal 18, dense 36, horde 62, swarm 89. | Creatures per screen is `perThousand * 0.225`: 2, 4, 8, 14, 20. |
| **`biomes.creature_density` multiplies it, and Meadow is 0.5.** | The starter world is thin twice: default tier, halved by its biome. Their own migration says 'normal' on 64x64 is "~12 scattered creatures". |
| **Their comment: every checked-in spec uses only sparse/normal/dense** - the top two tiers are "theoretical". | The emptiness is not a missing feature. It is authored-in. |
| **`bestiaryP4.js` holds 288 creatures** - 32 families x 9 roles, levels 1-50 - and `spawn_tiles` is EMPTY on every one, so biome `creature_types` is the only gate. | The 5 starter biomes reference four legacy names (Slime, Wolf, Bat, Skeleton). 284 creatures are seeded and unreachable. |
| **`entity_types.chance` is catalog-global** (`services/decorationDefs.js`); there is no per-world decoration density. | Trees and stones cannot be tuned per world. Biome choice is the only lever, so flora is REPORTED, not tuned. |
| **`computeAuras` is O(leaders x population) every tick**, unscoped. Their measurement: 4500 creatures at 50 leaders is 6.8 ms/tick; at 200 leaders, 25.8 ms - past the whole 16 ms frame. Champion is the only aura-carrying role. | Leaders must be rationed. More creatures is safe; more LEADERS is not. |

## Decisions

### D1. Solve for creatures-per-screen. Never name a tier.

The generator takes a target and picks the tier per world that lands closest
**after** the biome multiplier. A Meadow world asking for 6/screen gets `horde`;
a Mire world asking for the same gets `sparse`. Naming a tier and hoping is
precisely what produces the empty starting area, because the thin biome is
invisible at authoring time.

Rejected: exposing the tier names directly. That is the interface that already
failed - the number a human can judge is creatures per screen, and it is the one
their own tier table was tuned against.

### D2. Derive creatures from the biomes. NOT from the bestiary.

> **CORRECTED 2026-08-27, and the original decision was wrong in a way that
> made the whole report lie.** The text below the correction is kept because
> the mistake is instructive and easy to make again.
>
> something2's validator, run by their session with catalogs loaded, reported
> that all six worlds of a region this service called
> `creatures: 3376, empty_worlds: 0, ok: true` would seed **ZERO creatures**.
>
> **Existing is not the same as being spawnable.** `entity_types` holds 293
> creature rows including the whole P4 bestiary, which is why their
> "unknown creature type" check never fired on the names below. But spawning is
> gated by a SECOND catalog, `biomes.creature_types`, and a world seeds nothing
> unless its `allowed_creature_types` intersects the union of its biomes' lists
> (their SOMET-315). Every P4 name this generator produced was real, and none
> of them were spawnable.
>
> **What made the mistake convincing:** their own `pens` entries use P4 names,
> so a P4 name in a spec looked verified against real data. Pens are authored
> placement, not biome spawning — the one place those names ARE correct.
>
> Three facts from their live table, none of which are visible in
> `seeds/data/biomes.js`:
>
> - Only **Swarm / Skirmisher / Line** roles appear in any biome list. The
>   other six P4 roles exist in `entity_types` and can never spawn from a biome
>   anywhere in the catalog.
> - The families this generator leaned on hardest — Beast, Woodland, Tundra,
>   Swamp, Desert — appear in **no** biome list at all.
> - The five biomes it can select admit **four creatures between them**:
>   Slime, Wolf, Bat, Skeleton. The other 27 biomes are P4.
>
> **The fix is structural, not a filter.** `allowed_creature_types` is now
> DERIVED from the chosen biomes, so the intersection is non-empty by
> construction and this class cannot regress.
>
> **The counter was fixed in the same pass, and that matters more than the
> selection.** The tier arithmetic says what a world is BUDGETED for; it said
> nothing about whether anything could spawn. A green verdict over a region
> that seeds nothing is worse than no verdict, because it is trusted. The
> report now takes the intersection first and the arithmetic second.
>
> **Resolved same day.** something2 sent the full 32-row table — colour,
> `creature_density`, `path_tile`, `flora_types` — and the spec re-validated
> clean (0 errors, both passes). Selectable biomes went 5 → 32 and variety 4 →
> 37. Three things came with it that changed the design, in D8 below.

### D8. A region DESCENDS. Surface at the entry, deep further out.

The four-creature ceiling and something2's coherence warning turned out to be
the same problem with one answer.

The eight surface biomes admit only Slime, Wolf, Bat and Skeleton between them;
the 24 deep ones carry the P4 families. So variety is unreachable without going
underground — but scattering deep biomes among surface ones validates and reads
as nonsense (their words: a "green river valley" spanning Catacombs and
Emberdepths). Descending into them is the ordinary shape of a region, and it
fixes both at once. Measured: a surface-only region reaches 13 creature kinds, a
descending one reaches 37.

The depth split is the ONE editorial judgement this module makes about
something2's content rather than reading a number from their database.
Highlands, Storm Coast and Verdant Jungle carry P4 creatures but read as
outdoors, so they are classed surface and act as the bridge. Flagged to them for
correction.

**The LLM has to be told about the split.** Given only biome names it picked
eight surface biomes for a surface-sounding theme, never went underground, and
capped itself at 13 kinds — valid, coherent, and quietly half the region it
could have been. A model steered only by a theme string will follow the theme's
surface reading.

**Three corrections that arrived with their table**, each worth keeping:

- **Frozen Waste's flora was wrong here, and it was nobody's fault but this
  session's.** The first draft of this line blamed drift between
  `seeds/data/biomes.js` and something2's live table. something2 then diffed all
  32 biomes across four columns and found **zero drift** — their seed file and
  database agree exactly. Worse for me: the extraction script in this session
  had also read it correctly (`IceRock, pine_tree`). The wrong value was
  introduced by **hand-transcribing** the table into `world_gen.py`, where
  Arid Dunes' flora got copied onto Frozen Waste's row. Correct data, correct
  tooling, and a copy-paste error between them. Transcribe tables with a
  script, or check them against the extraction afterwards.
- **Seven biomes carry no flora by design.** A uniform "ground reads bare" floor
  would have fired on all seven forever while the repair chased something
  unrepairable — the floor overruling the author. They are exempt by name.
- **The bandwidth warning was calibrated wrong.** Set at 250 KiB/s it fired on a
  6.8/screen world, the configuration something2 had just called fine. That is
  the crying-wolf failure that gets a warning ignored. Now 400 KiB/s — above
  dense (~300), below horde (~515) and swarm (~736) — so it flags what they
  cautioned about and nothing else. The projection is an approximate upper
  bound: their measured row comes out lower than the per-creature figure alone
  predicts, so their measurements beat this arithmetic where they disagree.

### D2 (original, superseded). Use the P4 bestiary, banded by level.

`allowed_creature_types` is filled from the 288-creature catalog, matched to the
biome's families and filtered to roles whose level range OVERLAPS the world's
band. This is what turns "the same four things forever" into 29 distinct kinds
across six worlds.

The level band is **five levels wide, not three**. Roles are banded tightly
(Beast Swarm is 1-2, Skirmisher 2-3), so a 3-wide band reaches only two roles
per family and the entry world reads repetitive however many creatures are in
it. Measured: widening the band took the entry world from 4 kinds to 8, and the
region from 17 to 29.

### D3. Ration leaders, for their runtime and not for game design.

At most one leader role per world, because their own measurement shows leader
count - not headcount - is what bends the tick cost. This is the one place the
generator refuses something the catalog would allow.

### D4. Synchronous. No job, no queue, no facade.

A spec is arithmetic over a biome table: no diffusion, no GPU. So this is the
ONE surface here that fits inside something2's single blocking POST
("Sync services only", their `docs/ai-providers.md`). Tiles and sheets cannot
be served that way and needed the cache-reader of 0007 D4; this needs nothing.

### D5. The LLM picks biomes. It never picks numbers.

Tens of decisions with meaning - which biome each world is, in journey order -
authored by a model; every count solved deterministically afterwards. Same split
as 0007 D2, for the same reason: a model has no feedback loop on a
creatures-per-screen target and does not need one.

**Live since 2026-08-27** on `Qwen2.5-3B-Instruct-Q4_K_M` (1.80 GB), added to
`downloader/models.txt`. Warm calls take ~0.7 s. Three things had to be right,
and each failed first:

| | |
|---|---|
| llama.cpp runs in **router mode** here (`--models-dir`), so a request without a `model` field is a **400**, not a default. | The first version reported "LLM unavailable" against a perfectly healthy server. |
| The router loads models **on demand**; the first call after `--sleep-idle-seconds 120` spends ~13 s loading and answers with a body that is not completion JSON. | One retry turns a cold start into a 0.7 s success. A cold start is not a failure. |
| A 3B asked for `[["Meadow"]]` answers `[[Meadow]]`. | The parser reads almost-JSON by scanning bracket groups for names that exist. Validation is unchanged, so nothing unsafe passes. |

Model size is the decision, not the model. `--models-max 1` means whichever
model is resident is the only one, so a 9 GB chat model would have to be evicted
and reloaded around every generation. At 1.80 GB this coexists with a diffusion
pipeline; the 8-9 GB entries in `models.txt` explicitly cannot.

**Correction to an earlier draft of this ADR**, which said the LLM must stay on
the CPU and never touch the card. That is not how this stack is configured:
`docker-compose.cuda.yml` puts `llm-server` on the GPU deliberately and makes
sharing work with `--sleep-idle-seconds`, which releases the card after two idle
minutes. The relevant control is the model's SIZE, not its device.

### D7. The model authors; the code guarantees.

Validation alone was not enough. Told to build "a harsh journey", the model
pairs Arid Dunes with Frozen Waste - both carry two flora types - and the world
reads bare between its creatures. Measured: 4 problems from an LLM plan against
0 from the deterministic one.

Telling it which biomes are flora-poor took that from 4 to 1. Getting to 0 took
a **repair pass**: any world still under the flora or variety floor gains one
biome, chosen for what it actually adds. Both repairs apply to every author,
because the deterministic pairing can land two thin biomes together too.

One addition at most. A third biome band starts to muddy what a world IS, and a
world needing two additions was a bad pairing that the report should surface
rather than something to paper over.

Prompting is how a model is steered; it is not how a guarantee is made.

### D6. Preview one screenful per world, not a shrunken world

A whole 128x128 world scaled to a thumbnail turns both 40 creatures and 4000
into grey mush. A screen is the unit a player experiences and the unit their
table was tuned against, so each cell in the preview IS one screen and the dots
in it ARE the creatures-per-screen figure printed beside it. Emptiness becomes
visible rather than described, before `make seed-map` is ever run.

## What this cannot fix

- **Decoration density.** No per-world lever exists in something2. A world whose
  biomes carry two flora types will read bare between its creatures, and the
  report says so rather than pretending otherwise. Fixing it properly means a
  change on their side - a per-world decoration multiplier, or richer
  `flora_types` on the thin biomes.
- **The starter biomes themselves.** `Meadow` at `creature_density: 0.5` and
  four legacy `creature_types` is authored in their seeds. This works around it
  per world; it does not repair the catalog.

## Risks

- **Their schema is read, not agreed.** Field names come from
  `vale-region.map.json` and `seeds/mapSpec.js` as of 2026-08-27. A rename on
  their side breaks generated specs silently, since nothing here validates
  against their validator.
- **`level_band` semantics are assumed** to gate creature levels the way the
  bestiary's own ranges suggest. Not verified against their placement code.
- **Bandwidth.** Their measurement puts swarm on 224x224 at ~940 KiB/s down one
  socket. The generator will happily produce that if asked; the report flags it
  as CROWDED, but nothing refuses it.

## D9 - The four-creature ceiling was fixed at the source, 2026-08-27

Every statement above about "four creatures between them" is now **historical**.
something2 widened the five starter biomes' `creature_types` and applied it to
its database; this generator's table mirrors it.

```
Meadow        Slime, Wolf,     + Beast Swarm / Skirmisher / Line
Deep Forest   Wolf, Bat, Skel. + Woodland Swarm / Skirmisher / Line
Arid Dunes    Skeleton, Bat,   + Desert Swarm / Skirmisher / Line
Frozen Waste  Bat, Skeleton,   + Tundra Swarm / Skirmisher / Line
Mire          Slime, Bat,      + Swamp Swarm / Skirmisher / Line
```

Measured here after mirroring: the five together go from **4 kinds to 19**, the
entry world from 2 to 5, and a 9-world region from 37 kinds to **49**. A
3-world region now reports 22 kinds, 0 empty, 0 problems.

**It did not invent the pairings, and neither did this side.**
`scripts/bestiary/template.js` has a LINES table in which every P4 line already
declares its home biome, and these five biomes are exactly the five it names.
This repo had independently written the same relationship down in
`BIOME_FAMILIES` - Beast for Meadow, Woodland for Deep Forest, Desert for Arid
Dunes, Tundra for Frozen Waste, Swamp for Mire. Two sides reading a
relationship the data already held.

**Why it was invisible for two slices.** The P3 biomes shipped EMPTY and were
caught precisely because they were empty. These five shipped populated - with
legacy names - so nothing flagged them. An incomplete list is much harder to
see than a missing one.

**The safety argument, checked before it was applied:**
`creatureTileCandidates` in `services/mapService.js` intersects a world's
`allowed_creature_types` with its biome's list, so the world allowlist stays
authoritative and a biome can only ever REMOVE candidates, never add. No
already-seeded world changes behaviour.

**Transcribing it is the risky part, so the check is arithmetic.** The Frozen
Waste row in D2 above was wrong once from hand-copying exactly this table. The
five lists must union to 19 - a single mistyped or duplicated name would not -
and `smoke-world-gen` asserts that number rather than the shape.

### The density retraction, recorded

something2 also withdrew its reading that entry `dense` at 5.7/screen and final
`sparse` at 4.2 was a bug. The tier NAME is not felt density once the biome
multiplier applies: `sparse` in Infernal Gate (x2.5) outnumbers `dense` in
Meadow (x0.5). A constant `target_per_screen` producing flat felt density is
this generator working as specified. Its level-band half stands and is verified
monotonic.

Whether difficulty should RAMP with depth is a design question for the user,
not for either generator, and is deliberately not implemented on a peer's
say-so.
