# 0009 - The character training dataset, and why both adapters failed

Date: 2026-08-27
Status: Proposed. Extends [0006](0006-react-references-and-training.md), whose
Phase 4 shipped a trainer that runs, and whose risk register named "dataset
quality dominates" without ever measuring it.

## Why this exists

Two adapters have been trained and both produced a lattice of framed cells
instead of a subject. `curate-training-set.py` diagnosed that for `tile` and
fixed it there. Nobody ran the same pass over `sprite` and `core` - the two
kinds `POST /api/training` trains on **by default**.

This ADR is that pass. 483 references measured, every one looked at, and the
trainer read line by line alongside them.

The short version: **the dataset cannot teach what it is being asked to teach,
and four separate things in the trainer would damage it even if it could.**

## What was measured, 2026-08-27

`scripts/audit-character-refs.py`, over `/app/images`:

| | sprite | core |
|---|---|---|
| references | 149 | 334 |
| **usable as-is** | **2** | **137** |
| needs a human look | 1 | 69 |
| cannot teach | 146 | 128 |
| two or more subjects in one file | 126 (116 have ≥5) | 120 |
| transparency checkerboard baked into the pixels | 12 | 7 (see below) |
| no pixel grid, >1k colours | 128 | 325 |
| short side under 160px | 8 | 0 |
| near-duplicate of another reference | 24 by hash | 2 |

The core column is the surprise, and it took two corrections to get right - both
recorded under D1 below, because both were this audit repeating a mistake the
project had already made once.

**Those figures are measured over the FILESYSTEM. Over the database they are
worse.** The audit globs `images/`, and 43 of the 149 `ref_sprite_*.png` files
there belong to references the user has already soft-deleted. Restricted to
LIVE references:

| | sprite | core |
|---|---|---|
| live references (`deleted = false`) | 106 | 334 |
| **usable as-is** | **0** | **137** |
| needs a human look | 0 | 69 |
| cannot teach | 106 | 128 |

Every one of the three sprite references that survived the audit - the tree on
a plinth, the hollow fragment, the pine tree - is already deleted. Over the
live set there is not even a prop left: **106 live sprite references, none of
them usable, and none of them a character.**

And what training would consume TODAY: 102 sprite rows and 333 core rows are
live and `trainable = true`, so the default `kinds=["sprite","core"]` run reads
435 images, of which this audit rejects 230 - including all 12 checkerboarded
sheets, every one of which is still `trainable = true`.

**The core checkerboard count is 7 here and 8 in truth.** The eighth is
`ref_core_a9a6b26fb952`: its left and right borders two-tone perfectly at period
15, but the creature runs to the top and bottom edges, and `baked_checkerboard`
requires BOTH axes. Verified by eye and by measurement. Documented in the
function rather than patched - the both-axes rule is what stops every striped
and gradient backdrop being called a checker, and the only real fix (segment the
subject first) is circular, because the checker is what breaks segmentation.
`scripts/audit-entity-refs.py` catches it by taking the union with a second,
independently written detector.

That cross-check is worth more than the one number it corrected. The two
detectors share no code, and they agree exactly on 12 sprite checkerboards and
on 2 core references usable as-is. Agreement between independent
implementations is the strongest evidence these figures are real; the single
disagreement is the one above, and it resolved against this script.

Three findings matter more than the totals.

### 0. `core` is in better shape than `sprite`, and that is the plan

137 of 334 core references are single-subject creature illustrations on a flat
backdrop, at sane sizes - a real dataset. What they are NOT is one style: they
span painterly, cel-shaded, vector and pixel art, so a single trigger over all
137 learns an average of four idioms. That is a curation problem with a known
answer (pick one artist's subset), not a data-acquisition problem.

The `sprite` side has no such fallback. See below.

### 1. The `sprite` tab holds almost no sprites

141 of 149 have no detectable pixel grid and a median of 62,000 distinct
colours. Looking at them explains it: RPG Maker walk-cycle sheets, weapon-icon
boards, explosion strips, 3D-rendered knights, painted concept art, character
line-art, one Shutterstock-watermarked fire loop, and one screenshot of a
tileset store page with browser chrome still in it.

`measure_sprite` reads palette, cell grid and outline width off these. That
measurement is meaningless on a JPEG collage, which migration 014 already
worked out - and then routed around by making `trainable` permissive, so
training kept consuming them.

**And the two survivors are not characters.** The audit's `keep` verdict means
"can teach", not "is what we want", so it is worth saying what the two are.
`ref_sprite_bd7e43ed3cff` is a pixel-art tree standing on a grey stone plinth -
a base, not part of the tree. `ref_sprite_6f8abf25aa2b` is a hollow pale-blue
architectural fragment, a piece of some larger tileset, not a whole object at
all. Both were looked at, not inferred.

So the honest headline for this tab is not "2 of 149 usable" but **0 of 149
usable character references**. The two that pass every mechanical test are
props, and one of them is a fragment.

Neither this script nor `audit-entity-refs.py` can detect that, and neither
should try: "is this a piece of a thing?" has no non-circular test - a hollow
centre is correct for an archway or a barrel hoop - and "is this a character?"
is a semantic question, not a measurement. It is recorded as a known blind spot
in both tools rather than papered over with a rule that would misfire.

### 2. Nothing in the set is isometric

The game is isometric. Of 483 references, the isometric ones are roughly six
terrain images. Every character reference is four-direction top-down (RPG
Maker), straight side-on, or a front-facing concept illustration.

A style LoRA does not fix geometry - 0006 says so in its own risk register. So
the current dataset could train perfectly and still produce characters in the
wrong projection.

### 3. The one good subset is the one the pipeline throws away

Twelve `sprite` references are 8-directional character turnarounds by one
artist, one palette, hard edges, a real pixel grid, at a high three-quarter
angle - which is what an isometric game actually needs. They are the only
stylistically consistent character art in the set.

Every one of them is disqualified by a transparency checkerboard painted into
the pixels. That disqualifies them twice: trained on directly the adapter
learns the checker, and `split-sheets.py` refuses them because the checker
reads as foreground (~85%, outside its `FG_MIN..FG_MAX` band).

`scripts/key-checkerboard.py` keys the checker back to real alpha, and those
12 files become **103 single-character cells at a median 194px short side** -
above `measure.MIN_TRAIN_SIDE`, and the only training-grade character material
here. They are written to `images/recovered/` and are ready to train:

    make key-checkerboard      # 12 files -> images/recovered/keyed
    make recover-cells         # -> images/recovered/cells, 103 cells

The keying is a flood fill from the border, not a colour match. A colour match
removes the checker and also punches holes through white armour and pale skin,
because those genuinely ARE the light square's colour - you cannot separate
them by colour, only by connectivity.

Three filters earn their place, and the thresholds are taken from the data
rather than chosen. `--min-side 160` drops 63 cells that would be upscaled into
mush. `--max-aspect 2.0` drops the 4 TITLE BANNERS, which are perfectly good
connected components that read as subjects - they sit at 2.83-2.97 and the most
elongated real character at 1.43, so the gap is wide.

`--drop-edge-slivers` fixes a defect found by the entity audit reading this
output: 9 cells carried a fragment of the NEIGHBOURING character. `find_cells`
returns a bounding BOX, and subjects on a sheet are not box-shaped - a wizard's
staff leans up and left, overhanging the box of the character beside him, so
the crop takes it. The cell is single-subject by component analysis and still
contains a sliver of somebody else. 26 fragments erased across 103 cells.

Both of its conditions are load-bearing, and the second is what makes it safe:
a fragment must be small RELATIVE to the main subject *and* touch the crop
edge. Several cells are a dwarf WITH a wolf beside him - two large components,
both wanted - and a subject's own detached parts (a spark, a dropped item) sit
inside the cell rather than clipped by its boundary. Verified by eye across all
103 after the change: every companion survived.

### What splitting the OTHER sheets would give, and why it is not the answer

Segmenting all 245 contact sheets yields 4,889 sprite cells and 2,247 core
cells. The sprite cells have a **median short side of 61px**, and only 5.5%
clear 160px.

That is the tile-side failure exactly - `curate-training-set.py` recorded a
63px median there and concluded "no amount of splitting creates detail that was
never captured". The same sentence applies here. Splitting is worth doing for
the material that survives it; it is not a way to turn 149 asset packs into a
dataset.

## Four defects in the trainer, all invisible in the loss curve

Found by reading `scripts/train-lora.py` against the data it was being fed.
None of these would fail a run, print a warning, or bend the loss curve - which
is why a run that "worked end to end, loss falls, adapter saves" contained all
four.

**Captions were the filename hash.** `caption_for` appended the file stem, and
every reference is stored as `ref_sprite_<12 hex>.png`, so every caption read:

    <something2-style> pixel art sprite, 004228080a22

Not a weak caption - a harmful one. A unique random token per image gives the
text encoder a per-image handle, which is the memorisation a style LoRA exists
to avoid, and it dilutes the trigger meant to carry the style. `ref_map_` was
not even in the prefix-strip list, so those read `ref map 112233445566`. The
`label` column has existed since migration 013 and was never passed through.

**Every image was centre-cropped to a square.** Only past 3:1 was anything
refused, so an ordinary 512x1024 full-body character reference trained as a
torso, head and feet discarded. Median aspect across the set is 1.45 for
sprites - this was not an edge case, it was most of them.

**Pixel art was resampled with LANCZOS.** A 64px sprite enlarged to 1024 with a
smooth filter is a blur, and the blur is what gets learned. Same mechanism as
the tile side's "mush", in the one place it is most avoidable.

**SDXL's size conditioning was a constant.** Every sample was tagged
`(1024, 1024, 0, 0, 1024, 1024)` - "native resolution, uncropped" - whatever it
actually was, and whatever crop had just been applied to it.

### Two more, found by running the fixed trainer's CPU half over the recovered set

Both fixes were wrong on first attempt, and only running them against real
files showed it. Worth stating because the lesson is the same one this ADR is
about: a change that looks right in a diff and produces no error is not
evidence of anything.

**The caption fix leaked on the new filenames.** It stripped a fixed list of
prefixes - `ref_sprite_`, `ref_core_` and so on. `split-sheets.py` names its
output `cell_sprite_sprite_<hash>_000`, which that list cannot strip, so all
103 recovered cells got a caption ending in a hash: the original bug, on the
files it matters most for. Filtering token by token rather than by prefix fixes
it for any naming scheme.

**The NEAREST rule routed all 103 cells to LANCZOS.** The test for "is this
pixel art" was a small palette or a clean pixel grid, and the recovered cells
have neither: 4k-27k distinct colours, drawn at 1:1. By palette size they look
like renders. They are not - the style simply shades heavily, and its edges
STEP. Measuring edge softness separates them cleanly (hard-edged cells p90 0.45,
painted art p10 0.52), and with that third signal 95 of 103 route to NEAREST.
Without it, the single most important fix in this ADR would have done nothing
on the single most important dataset in it.

## Decisions

### D1. `trainable` gets a real gate for characters, and it explains itself

`judge_trainable` rejects three things: empty, under 160px, over 3:1. That was
a deliberate over-correction after `usable` rejected 100 of 106 sprites for
having too many colours, and it went too far the other way.

`scripts/audit-character-refs.py` adds the gate the character kinds never had -
contact sheets, baked checkerboards, duplicates, and non-pixel-art in the
`sprite` tab - and writes `trainable_why` per row so the UI can say why an
image is absent instead of silently dropping it.

Three verdicts, not two. `reject` is what provably cannot teach; `review` is
what a human must look at. Collapsing them is how the tile side rejected 100
real sprites.

**Two corrections this audit needed before its own numbers were trustworthy.**
Both are the same failure the project has hit twice already - a rule imported
from a question it was not written to answer - so they are recorded rather than
quietly fixed:

* **It counted subjects with `split-sheets.find_cells`.** That function answers
  "what can I safely CROP OUT of this?", so it discards any piece over 55% of
  the frame. Asked "how many subjects are here?", it returns nothing for a
  creature centred on white and cropped tightly. 169 core references came back
  unsegmentable; measuring them showed **141 were exactly one correctly-framed
  subject**. That is migration 014's finding - "cropped tightly, which is what
  good reference art looks like" - reappearing under a different rule. Subject
  counting is now its own function with a 0.95 cap, and it still consults
  `find_cells` afterwards to catch packed atlases whose items fuse.
* **It counted an artist's signature as a second subject.** "ID EV GAHIN" and
  "KING 2023.5" are perfectly good connected components. The fix is a relative
  size test, and the threshold was measured rather than chosen: across the 55
  references that segment into 2-4 components, the ratio of second-largest to
  largest is sharply bimodal - 36 at or below 0.078, 19 at or above 0.151, and
  nothing in between. 0.12 sits in the empty gap. Small components are now
  reported as "detached marks beside the subject" - a `review` note, never a
  rejection, because they are at least four different things and only a person
  can tell which: an artist's signature, a title bar, a drop shadow sitting
  clear of the feet, and loose specks of the subject's own art. The first
  wording of that message asserted "usually a signature", which was wrong for
  `ref_sprite_790879270b49` - a detached shadow plus stray leaves.

The result of both: `core` went from 2 keep / 205 review to 137 keep / 69
review, and the review pile became a pile worth working through - dark-backdrop
art with no separable foreground, parchment character sheets, and icon boards -
rather than most of the dataset.

**It does not judge the camera, and says so.** Whether a character is drawn in
the game's projection is not readable from one character image - there is no
ground plane in it - which is exactly why `measure_tile` reads projection off a
TILE. That stays a human call.

### D2. `sprite` and `core` are separate adapters

`kinds=["sprite", "core"]` is the current default and it is the tile-mixing
mistake wearing a disguise. `core` is painted concept art at high resolution;
`sprite` is meant to be finished pixel art. One trigger over both learns an
average of two art forms. The mixed-kinds warning now covers this case; the
default is left alone rather than changed underneath a queued run.

### D3. Fix the trainer before running it again

All four defects above are fixed, plus the standard things that were absent:
noise offset (SDXL cannot otherwise produce the flat dark and light fields
pixel art is made of), min-SNR-gamma loss weighting, a warmup-then-cosine LR
schedule where there was no schedule, and shuffled sampling without replacement
in place of `random.randrange` per step.

`scripts/test-train-prep.py` covers the GPU-free half. That is the first test
this script has had, and it exists because the caption bug is precisely the
kind that survives "the run finished".

### D4. Measure the tile projection before training anything

0006 D3 called this "likely the single largest correctness win in the plan, and
it costs no training", and Phase 3's checkpoint - regenerate one character
against a derived profile - is still unticked. Given finding 2, it is now also
a precondition: there is no point training a character adapter until the target
projection is a measured number rather than a guess.

## What to acquire, and why each rule is there

Every rule below is one of the failure modes above, stated as a requirement.

1. **One subject per file.** 245 of 483 references are sheets, and sheets are
   what both adapters learned to draw.
2. **Real alpha.** No checkerboard, no white box, no baked drop shadow. The
   background trains too.
3. **Short side 256-512px at an integer pixel scale** - art drawn at 48-64px
   and exported ×4 or ×8 with nearest-neighbour. Below 160px the trainer is
   inventing detail; smooth-scaled exports arrive as blur.
4. **Palette-locked, under ~64 colours, hard edges, no anti-aliased
   silhouette.** This is what makes it pixel art rather than a render.
5. **The game's projection, 4 or 8 directions per character**, at the elevation
   the tile set measures. An idle frame per direction is enough.
6. **One artist across the whole set.** 20-40 coherent examples beat 400 mixed
   ones; the current 483 span painterly, cel-shaded, vector and pixel, which is
   why no consistent style can be extracted from them.
7. **No text, banners, watermarks, UI chrome or parchment frames.** The set
   contains D&D character sheets, store screenshots and a Shutterstock
   watermark, all of which are learnable.
8. **No duplicates.** A repeat is extra weight on one example, not extra data.

The 150 keyed Mesgard cells satisfy 1-4 and 6-8, and are the closest thing here
to 5. They are the reference for what to look for, and the first place to look
is more work by the same artist.

One thing to keep out: **do not train on this pipeline's own output.** Migration
013 records the reason - it is how a style collapses.

## What has NOT been done

- **The verdicts are not in the database, but not for the reason first given.**
  This originally read "Docker was not running, so `--apply` has not been
  executed" - which turned an untested write path into an environmental excuse.
  The database was reachable the whole time: Docker's CLI pipe was broken while
  the container ran on, and `llm_monitoring` holds 2,555 references, 573 of
  them trainable. Nobody had looked.

  `--apply` is now TESTED - `scripts/test-apply-verdicts.py`, 9 checks,
  `make test-apply-verdicts`. It copies production's `reference_assets` schema
  out read-only, rebuilds it in a throwaway database, and exercises the write
  there: a blocking row flips and carries its reason, a clean row does not
  flip, a soft-deleted row is untouched, pre-existing `metrics` survive, the
  suffix match writes exactly one row rather than splashing across the table,
  and the run reports what did NOT land.

  Two defects the first version of that test could not have caught, both found
  by a sibling session hitting them for real:

  * **It seeded the paths its own invocation produced.** Seed and query agreed,
    every assertion passed, and a path-matching bug would have been invisible -
    a test cannot falsify an assumption it shares. It now seeds
    `/app/images/<name>` (production's format) while `--data` points somewhere
    else, so the two must genuinely disagree for the match to be exercised.
  * **`applied: 0` and "nothing needed changing" were the same sentence.** They
    are opposite outcomes; one is a clean no-op and the other is every finding
    failing to match. `--apply` now also reports the count that matched no row.

  The matching itself changed too: `LIKE '%<name>'` treats the UNDERSCORES in
  every one of these filenames as single-character wildcards. It is now an
  exact `right(file_path, n)` suffix comparison with no pattern semantics.

  Checked two ways, because they answer different questions. **Mutation:**
  reverting to a whole-path match reddens three checks and prints `applied: 0`
  - it proves the suite CAN fail. **A decoy row:** the fixture now seeds
  `refXspriteY<hex>.png`, reachable only by a wildcard, and restoring the LIKE
  predicate makes the run report `applied: 2`, exit 0, and leave that row
  holding the reject reason belonging to a different file. That one is the
  stronger check. A wrong-row write is invisible in the rowcount, and mutation
  says nothing about whether the fixture was written from the same belief that
  produced the bug - which is precisely how both this session and a sibling
  shipped a "fixed" path match with every assertion green.

  Running it against the REAL database is still not done, and deliberately:
  that write changes what the UI shows and what the next training run consumes,
  across 573 trainable rows. It is a decision, not a verification step.
- **No training run has been executed against the fixed trainer.** The
  GPU-free half is tested; the loop changes - noise offset, min-SNR, the LR
  schedule, per-sample conditioning - are reviewed, not run.
- **No evaluation harness.** 0006 Phase 4 still lists "fixed evaluation prompt
  set; before/after contact sheet against the four existing pixel-art LoRAs"
  as unticked, and nothing here changes that. Loss will keep failing to tell
  anyone whether an adapter got better.
- **`MIN_IMAGES` is still 8**, against 0006 D3's own "20+ consistent examples".
  Left alone deliberately: raising a gate blocks runs, and that is the user's
  call rather than this pass's.
