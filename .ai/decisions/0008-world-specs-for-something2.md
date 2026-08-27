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

## D10 - The biome multiplier does not scale anything, 2026-08-27

**This supersedes the founding premise of `world_gen.py`, D1's motivation, and
every creature-per-screen figure in this document above.**

The module was built on the claim that a starter world is thin twice over: the
default tier, times a Meadow `creature_density` of 0.5. Solving for a target
"after the biome multiplier" was its headline feature.

**No such halving happens.** something2 settled it structurally:

```
resolveDensity(tier, width, height)      <- arity 3, no biome parameter
target = round(perThousand * area / 1000)
```

A world's total cannot depend on the multiplier, because the multiplier is not
in scope where the total is computed. And `creatureDensityField.js` says so in
its own header: *"PURELY REDISTRIBUTIVE ... A world's total comes from its
density tier and MAX_WORLD_CREATURES, never from here."* The field normalises
to mean 1.0 over interior tiles, so in a **single-biome** world a uniform
weight cancels completely - Meadow at 0.5 alone behaves exactly like a biome at
2.5 alone. It is a relative weight BETWEEN biomes inside one world, clamped to
[0.15, 1.5].

### What it cost

Compensating for a multiplier that does not apply **over-populated the entry
world of every region**. emerald-reach's hub came out `horde` - 1016 creatures
on 128x128, 14 per screen against a requested 6 - while its siblings sat at
295. That is the 3.4x outlier something2 flagged, and it is real.

The contradiction was visible in this project's own report for two days:
`creatures` (from the `resolveDensity` mirror) and `per_screen` (with the
multiplier) described different worlds, and every world was off by exactly its
own multiplier. Nobody noticed while both sides quoted the numbers at each
other.

### The fix

- `per_screen` is derived from the RESOLVED count - `scatter / area * 225` -
  so it cannot drift from `creatures` again, and a world clamped at
  `MAX_WORLD_CREATURES` reports the density it will actually have.
- `choose_density` ignores biomes. It is much less clever and much more
  correct; the tiers are coarse, so nearest-tier-to-target still earns its
  place.
- `biome_multiplier` is still reported, documented as describing distribution
  WITHIN a world and nothing else.
- **New:** a caveat when the achieved mean misses the requested target by more
  than 15%. The tiers are 2.0 / 4.1 / 8.1 / 14.0 / 20.0 per screen, so a target
  of 6.0 yields 4.1 - a 32% shortfall that was previously masked, because the
  bogus compensation happened to push thin biomes up a tier.

### The retraction of a retraction

something2 had observed that density runs backwards - entry dense, depths
sparse - and **withdrew it** after this project argued the tier name is not
felt density once the multiplier applies. That compensation does not exist, so
the original observation was correct and was withdrawn on a false premise. A
player descending emerald-reach now walks from 1016 creatures to 147, a 7x
fall, while level bands ramp 3-5 up to 11-15.

Whether difficulty should RAMP with depth remains the user's design question,
not this generator's, and nothing is implemented on either side's say-so. But
it is now a question about real numbers.

### The pattern, named

Four bugs this week share one shape: **internally consistent, arithmetically
checkable, describing a mechanism nobody had read.** The creature counter, the
`depth` field, the spiral-vs-index bands, and now the multiplier itself.

The fifth version is worse and worth naming separately: a *correct measurement
withdrawn* because a plausible story was offered for why it did not matter. A
retraction deserves the same standard of proof as the claim it withdraws.

`level_band` is still assumed on both sides and unread. It is the obvious
candidate to be next.

## D11 - Every world declares a level band, 2026-08-27

**This reverses D8's "surface worlds carry no band".**

`scripts/seed-map.js` writes a world's level range as:

```js
w.level_band ? w.level_band[0] : 1,
w.level_band ? w.level_band[1] : 1,
```

**An omitted band is not "unspecified". It is level 1-1.** A valid-looking
range that pins the world to level-1 creatures and is indistinguishable
afterwards from a world that asked for exactly that. The same trap as
`Number('') === 0`: a default that is itself a legal value, so nothing
downstream can tell "unset" from "the lowest setting".

D8 omitted the band on surface worlds, copying vale-region's convention. That
convention is correct **for vale-region**, whose surface ring IS the level-1
starting area. It does not transfer to a region that DESCENDS. emerald-reach
seeded as:

```
hub, 1, 2   (omitted) -> 1-1
3           [3,5]
```

Three worlds at level 1, then a jump to 3-5.

`SURFACE_BAND = [1, 2]`, and every world now carries a band. Never `null` -
an absent key and a null are the same to that ternary.

**Why [1,2] and not the deep formula.** The deep ramp's lowest rung is `3 +
0*2 = 3`, so reusing it would make a starting meadow a level 3-5 world. That
is a design change wearing a bug fix's clothes. [1,2] keeps the entry a
level-1 area - which is what omitting the band was always meant to convey -
and leads into the deep ramp's [3,5] with neither gap nor overlap.

**The one behavioural change:** surface worlds seed as 1-2 where they
previously seeded as 1-1. Deliberate, and the smallest change that removes the
reliance on a default.

**This cannot affect shipped content.** vale-region and p5-descent are not
regenerated by this code and the seeder is unchanged; the only thing removed is
this generator's dependence on the default. Whether the seeder should
distinguish absent from 1-1 at all is something2's question, and it declined to
change it to satisfy a third spec - correctly, since two checked-in specs rely
on the current behaviour.

### The tally, by origin

Five bugs of one shape this week - internally consistent, arithmetically
checkable, describing a mechanism nobody had read:

| | |
|---|---|
| the creature counter | this side |
| the `depth` field | this side |
| the biome multiplier (D10) | this side |
| spiral-vs-index bands | something2 |
| `level_band` omitted | something2's advice, applied here |

The last one is the instructive one: a convention was read off a spec file and
passed on without the seeder that gives it meaning. A pattern without its
meaning is not a convention, it is a coincidence.

## D12 - A green suite says nothing about what is being served, 2026-08-27

The sixth of the week's class, and the widest.

D11's fix was committed, its 33 smoke cases passed, and the region something2
downloads **still had three level-1 worlds**. The code was right, the tests were
right, the artifact was wrong.

**Nothing in the suite could have caught it.** Every smoke script runs as
`docker exec sprite_generator python /app/scripts/...`, which starts a fresh
interpreter and imports from disk. It therefore tests the code AS WRITTEN and
never the code AS SERVED. The uvicorn process had not been restarted since the
commit, so it kept serving the pre-fix module while every file on disk looked
correct - the `uvicorn --reload` trap this project already had a memory note
about.

something2's diagnosis was sharper than "it did not deploy": the density TIERS
had changed in the same artifact, so D10's fix was live and D11's was not. One
process had been bounced between the two commits. That asymmetry is what proved
it was a deploy boundary rather than a bad commit.

**Why it matters past the deploy.** something2's admin downloads and seeds THE
ARTIFACT, not this repository. A person reading "fixed in 110479f" and clicking
Download and Seed would get three level-1 worlds, and it would validate 0/0 on
their side forever.

### `scripts/check-artifacts.py`

Regenerates every stored region from its own stored parameters and compares the
result to the artifact on disk. A mismatch means the artifact predates the
running code. Not a smoke test - it has no fixtures and asserts nothing about
correctness, only that the thing being served and the thing in the code are the
same thing.

**Run it after deploying, and before telling anyone a fix has landed.**

Mutation-checked rather than trusted, because a guard that cannot go red is
exactly the failure being guarded against: removing one `level_band` from the
artifact produced `emerald_reach_hub.level_band: '(absent)' -> [1, 2]`, one
difference, exit code 1.

### The pattern this closes

Twice now the artifact and the claim about it were checked separately, and only
the claim was checked:

- a test asserted surface worlds carry NO band, passed every run, and encoded
  the defect as a requirement (D11);
- then the fix for that defect did not reach the service, while the commit
  message defended it (here).

Both are the same error at different distances: **verifying the description
instead of the thing.**

### The instrument did it too, 2026-08-27

`check-artifacts.py` has two halves. The region half regenerates and diffs. The
map half, added later, checks that finished maps are internally whole - and it
ended every run with:

```
  --    no finished maps to check
maps checked, all whole.
```

There are no map rows in this database. Every map built here today was built by
calling the functions directly, through smoke scripts and one-off `docker exec`
runs, so no `kind='map'` job row was ever written. The map half has therefore
checked **nothing, ever**, and reported a pass for it every time.

The region half never had this bug - it counts what it checked and says
`0 region(s)` when it finds none. Only the map half collapsed "no faults" into
"all whole".

So the file written to stop this class of error contained an instance of it,
one paragraph below a docstring saying that calling this 'verified' would be
"the overclaim this whole file exists to stop". Writing the warning is not the
same as obeying it - which is the same error again, at the shortest distance
yet: **the description and the thing were the same file.**

Fixed: `check_maps` returns `(broken, checked)`, and an empty set now prints
`NO maps were checked. This says nothing about maps.` A zero denominator is
never a pass.

**The general rule, worth applying to every guard here:** a check that reports
success must report its denominator. "Nothing was wrong" and "nothing was
looked at" produce identical output otherwise, and only one of them is good
news.

### Two more, in one afternoon, 2026-08-27

Both mine, both found within an hour of writing the section above, which is
the useful part: knowing the pattern did not stop me producing it twice.

**A switch that was on and did nothing.** The map painting stopped fitting on
the card, so `_map_pipe` enabled VAE tiling. `pipe.enable_vae_tiling()` does
not exist in diffusers 0.40, so the call moved to `pipe.vae.enable_tiling()` -
and I added a check that `vae.use_tiling` was True afterwards, on the reasoning
that a switch which silently failed to flip is worth less than no switch. The
check passed. The decode still ran whole and still OOMed.

`_decode` gates on `z.shape[-1] > vae.tile_latent_min_size`, and that is
`sample_size / 8` = 128 for the SDXL VAE, while a 1024x1024 image has a 128x128
latent. `128 > 128` is false. The flag was set, the branch was unreachable, and
my verification confirmed the flag. **I checked the description of the
behaviour rather than the behaviour**, in a check written specifically to avoid
that. The fix lowers the thresholds and wraps `tiled_decode` to count calls, so
the run now reports `decode: tiled (1 call(s))` - the path, not the flag.

**A metric that scored two coastlines at zero.** `map-discriminator.py` asks
whether a map adapter paints a coast, screening on the share of the image that
reads as water. It reported **0.0% for both adapters** and printed a confident
verdict: "No coast either way."

Both images are roughly a fifth open water, with cliffs, bays and islands. I
found out by opening them.

The screen measured the four-colour palette fit rather than the pixels. Four
median-cut colours over a map that varied never allocated an entry to the
water at all - it averaged into the greens - and the entry that did emerge,
`#488269`, is a teal sitting just under the 0.47 blue cutoff. Two independent
reasons, both invisible from the number.

The metric now counts pixels, and reports **20.0%** for the image it scored at
zero.

### What these two add to D12

D12 said: verify the artifact, not the claim about it. These sharpen it, because
in both cases a verification step existed and ran:

- the tiling check tested a **flag that the behaviour did not follow**;
- the coast screen tested a **summary the artifact did not survive**.

**A derived value is a description too.** A flag, a palette fit, a percentage,
a passing assertion - each is a claim about the artifact, and checking one is
not checking the artifact. The only things that closed either of these were
running the actual code path and looking at the actual picture.

Which is also why `map-discriminator.py` writes its images to disk and says so
in its output. The number is a screen; the file is the evidence.

### And the investigation those two were serving was already dead

The map adapter was retrained specifically to test whether a hash-polluted
caption set was what made the first adapter paint a monochromatic green island.
33 minutes of GPU, and a peer session's experiment design that was better than
my hypothesis.

Three arms at the same prompt and seed - first adapter, retrained adapter, and
the first adapter with its trigger withheld:

| arm | water | flatness |
|---|---|---|
| `mapstyle` | 20.0% | 13.1 |
| `mapstyle2` | 15.6% | 15.1 |
| `mapstyle`, no trigger | 20.6% | 14.3 |

**Every arm paints a coast.** Bays, cliffs, islands, forest. The green island
does not reproduce, so neither candidate explanation can be tested - and the
recorded first-adapter measurement does not reproduce either: 14.6 flatness with
an all-green palette came back as 13.1 with a teal in it.

That last part is the finding. **The original run was not recorded, so only a
description of it survives**, and the description is now the only evidence that
the defect ever existed. The green island was most likely a property of that
one run's unwritten conditions rather than of the adapter.

The caption bug was real and worth fixing on its own terms - 1,863 tile captions
reduced to the single word "of" - and the fix is verified against real rows. But
it was never shown to fix anything, and saying it was would be the overclaim
this whole document is about. The investigation is closed as **cause unknown**,
not as solved.

**Cost of not recording a run: 33 minutes of training, four failed generations,
and an experiment that could not have answered its question.** Cheaper next
time: write the prompt, seed, model string and output path next to any result
worth citing later.

Two footnotes, both corrections to the above:

- The training peak WAS recorded. The trainer logs `Peak VRAM` per run, and
  `mapstyle` and `mapstyle2` both read 5.60/12.00 GiB. I reported it lost
  because the ad-hoc sampler I had set up did not survive; I had not checked
  whether the thing already recorded it. **Look for the existing record before
  announcing that a measurement is gone.**
- The 5.60 GiB allocator figure and the 6741 MiB device figure are the same
  run, not two numbers in tension: the ~1.1 GiB gap is CUDA context, cuBLAS
  and cuDNN workspaces.

### D14 - The OOM was not what I said it was, 2026-08-27

Recorded because the correction is more instructive than the fix.

The map painting stopped fitting on the card. I fixed it by tiling the VAE
decode, and the fix is right - three runs failed without it, three succeeded
with it, and the production map build now completes. **The explanation attached
to it was wrong.**

I wrote that llama.cpp holds 3.6 GB permanently, that it does not release on
sleep, and that a 12 GB card therefore has ~8.4 GB for diffusion. The error
message I quoted in the same paragraph says otherwise:

```
total 12.00 GiB, 3.18 GiB free, 7.71 GiB allocated by PyTorch
```

`12.00 - 3.18 - 7.71` leaves **1.11 GB** for everything else. And `7.71 + 3.6`
would have left 0.69 GB free, not the 3.18 the error reported. The arithmetic
contradicted the evidence sitting next to it and I did not check it.

The 3.6 GB was a real reading, taken at a different moment with llama.cpp
awake, carried into an explanation of a failure it was not present for.
llama.cpp does release on sleep - it entered the sleeping state two minutes
after this build's region-naming call and idles at ~390 MB.

**The cause is OPEN, and my replacement for it was no better founded.** I
swapped the llama.cpp story for a float32-VAE-transient story, which fits
`expandable_segments:True` changing nothing and the 55 MB
reserved-but-unallocated - and does not fit a 512 MB request failing against
3.18 GB free. If 3.18 GB were genuinely free, that allocation succeeds. One of
those three numbers is not what it appears to be. A peer caught it, having made
the mirror-image mistake: they corrected my sleep claim and then reused the
residency as the cause, quoting the same error and not doing the subtraction
either - fixing the smaller half of a wrong claim while carrying the larger
half forward, and feeling audited.

What is established, and is enough to justify the fix: three decodes failed
untiled, three succeeded tiled, and the production map build now completes.

The raw tracebacks are gone. Those runs called worker functions directly, so
nothing persisted to `docker logs`; only our paraphrase survives - the same
position the green island left us in. It is reproducible now if closing it is
ever worth the GPU: `queue_map` makes the failing path reachable through the
queue, where logs persist.

**The practical note that survives is a peer's, and is better than mine:** the
LLM and the diffusion model do contend, but on a TIMER. Sleep fires about two
minutes after the last request, and a map build names its regions immediately
before painting - inside that window every time. A standing "8.4 GB available"
would have looked like a constant to plan around. It is a collision to avoid,
and tiling avoids it.

**What this adds to D12.** The other entries are about verifying a description
instead of the thing. This one is narrower and nastier: *the evidence was
already in front of me, in the same paragraph, and the claim did not match it.*
Not a missing measurement - an unread one. Arithmetic in a commit message is a
claim like any other, and nobody checks it because it looks like a citation.

### The mechanism behind most of these: routing around the real path

A peer's generalisation, and it explains a whole class at once. Each of these
lost its evidence the same way:

- the map rows - built by calling worker functions directly, so
  `check-artifacts.py` had no denominator;
- the OOM tracebacks - same ad-hoc path, so nothing reached `docker logs` and
  only a paraphrase survives;
- the training VRAM sampler - an ad-hoc `/tmp/vram.log` that died, while the
  trainer's own `Peak VRAM` line had it all along;
- the green island - generated outside any recorded run, so the conditions
  cannot be recovered.

**Every one is someone routing around the real path and losing the recording
that comes with it.** The real path was unreachable for a reason that felt
trivial at the time - a scope check on an endpoint nobody wanted to mint a key
for - and the shortcut silently traded away the audit trail.

`maps.queue_map` is the fix for the specific case: the endpoint now adds
exactly the scope check, and the queueing path is callable by operational
scripts, so there is no longer a reason to reach past it. The general rule:
**when you find yourself calling an internal function because the supported
path needs a credential, the recording is what you are giving up.** Make the
supported path reachable instead.

### D15 - A declared terrain colour must exist in the art, 2026-08-27

The first two maps built through the queue both warned:

```
terrain 'water' (#2850c8) covers 0.0% of this map
```

Both were painted from a prompt asking for a coast, and one used the trained
map adapter, whose painting is measurably 20% open water. The water was in the
art. It did not survive quantisation.

Measured, on the adapter's own painting:

| declared terrain | Lab distance from the painted water | share captured |
|---|---|---|
| water `#2850c8` | 77.7 | 0.0% |
| stone `#6e6e73` | **26.0** | **97.5%** |

The painted water is `#39888f`, a desaturated teal. The declared water is a
saturated blue three times further away in Lab than a mid-grey. So a *grey*
is the nearest declared colour to every water pixel, and quantisation - which
IS the tile-id binding - assigns the whole sea to stone.

Correcting that one colour to `#39888f`, with nothing else changed, moves water
from 0.0% to 20.0% and stone from 43.7% to 24.5%. That matches the 20% the
pixel metric reads directly off the image. Verified on CPU by re-quantising the
existing painting; no GPU needed to check this.

**This is sharper than the near-neutral-sink hazard already recorded in the
plan.** That one says neutral colours attract. This says something stronger:
a declared colour that does not appear in the art is not merely under-used -
it can be *entirely* displaced by a colour that has no business representing
it, and the loss is silent apart from the coverage warning.

The demo terrain set in the smoke tests is the trap: `#2850c8`, `#4a7c3f`,
`#c8be78`, `#6e6e73` are legible primaries chosen to be far apart from each
other, which is the wrong criterion. They need to be close to the ART.
`GET /api/maps/palette/{reference_id}` exists to derive them from a reference
and is the right way to pick them.

**And the guard did its job.** `_coverage_warnings` reported 0.0% on both
builds. Nothing was silently wrong - it was loudly wrong for two builds while
I read the warning as noise and went looking for a model problem. Three
explanations of mine were wrong before I measured: the default model, then the
adapter, then the prompt. The measurement took two minutes and no GPU.

## D13 - Density is uniform across a region, and that is a decision to make

D10 removed the biome multiplier because it described nothing real. It was also
**the only thing making two worlds in one region differ in density**, so its
removal took the variation with it. Every world in a region now gets the same
tier, because one target maps to one tier.

```
before D10   horde, dense, normal, normal, normal, normal, normal, sparse
after D10    normal x8      - 295 creatures, 4.1/screen, every world
```

This is not a regression: the variation came from a mechanism that turned out
to be fiction, so it was never describing anything. But "every world identical"
is unlikely to be what anyone wants, and something2's original *density runs
backwards* finding is now retired rather than fixed.

**If per-world density is wanted it has to come from somewhere real** - a
per-depth target, or an explicit density in the plan. That is a feature
decision and belongs to the user, not to either generator. Nothing is
implemented on a peer's observation.

Two consequences worth recording either way:

- **The tier snapping is now visible.** A target of 6.0 gives 4.1 on all eight
  worlds - the 32% shortfall of D10, applied uniformly instead of being masked
  per-world. The `caveats` entry says so.
- **The socket guard is all-or-nothing per region.** It can no longer say
  "this world is the problem", only "this region is".

### The guard-reachability check

something2 raised that the 400 KiB/s guard might have become **unreachable** -
worse than wrong, because an unreachable guard reads as "checked and fine".

**Measured, and it is reachable.** The claim that `normal` is the only tier a
target can reach is not so; every tier is reachable and the target picks which:

```
sparse  2.0/screen    75 KiB/s          targets  0.1-3.0
normal  4.0          149                         3.1-6.0
dense   8.1          298                         6.1-11.0
horde  13.9          513  fires                 11.1-16.9
swarm  20.0          737  fires                 17.0-30.0
```

A real region at target 14.0 emits four socket warnings.

The suggestion was worth taking anyway, as an instrument rather than a fix:
`smoke-world-gen` now asserts the threshold sits INSIDE the reachable set -
some tier under it, some tier over. A threshold above every reachable tier
cannot fire; one below all of them is noise. Both are invisible otherwise,
because **a guard that never fires and a guard with nothing to catch look
identical from the outside.**

That is the same instrument as `check-artifacts.py` in D12: compare the thing
against what it claims to cover, rather than against itself.

### A track record is evidence about a person, not about a claim

The guard-reachability claim in D13 was wrong, and it arrived with a strong
recommendation to act on it immediately - singled out from a message where
everything else was explicitly deferred to a human, on the grounds that this
one needed no design decision. The tier table that refutes it was one function
call away and nobody made it, on either side.

It was worth checking **because** the source had been right five times that
day, not despite it. A track record feels like evidence about the claim while
only ever being evidence about the claimant, which makes it the one input that
grows more persuasive exactly as it grows more dangerous.

This is the same object as the other six, one level up. Each of those was a
description trusted in place of the thing it described: a comment, a
convention, a test, a commit message, a count. A reputation is another such
description, and the cheapest one to mistake for the thing.

It also names the failure that produced D10 from the other direction: a
plausible mechanism was offered for why something2's density measurement did
not matter, and it withdrew a correct finding on the strength of it. Same
error, opposite roles.

**Checking cost one function call in both directions and was not paid in
either.**
