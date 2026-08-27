# Entity cutouts: why the pipeline cannot produce them, and what the training set is teaching instead

Date: 2026-08-27. Status: investigation, no code behaviour changed.

The goal this answers: **generate one entity, centred, on a transparent
background, with no pedestal, no ground patch, no backdrop and nothing else in
frame** — because the output is composited into a pixel-art RPG, so every pixel
that is not the entity is waste that has to be removed by hand or shows up in
the game.

Companion artefacts, all reproducible:

| What | Where |
|---|---|
| The auditor | [`scripts/audit-entity-refs.py`](../../../scripts/audit-entity-refs.py) |
| Per-file marks | `images/_audit/core_marks.md`, `images/_audit/sprite_marks.md` |
| Full measurements | `images/audit_core.json`, `images/audit_sprite.json` |
| Contact sheets per defect | `images/_audit/core_*.png`, `images/_audit/sprite_*.png` |

---

## The one-paragraph answer

Nothing in this system can emit transparency from a model. SDXL has no alpha
channel, so **every transparent pixel this pipeline has ever produced was made
by `remove_background`**, a corner flood fill that deletes background-coloured
regions connected to the border. That single fact decides everything else: the
model's only job is to produce a subject on a **flat, keyable, uniform**
backdrop with nothing fused to it, and the cutout stage does the rest. Measured
against that job, the reference set teaches almost exactly the opposite, the
trainer discards the little transparency the set does have, and the entity
generation path skips the cleanup every other path in the codebase applies.

---

## 1. Where transparency is actually decided

Three stages, and only the last one can create an alpha channel.

1. **Training** — `scripts/train-lora.py`. Composites every reference onto
   opaque grey `#808080` before the VAE sees it
   (the `alpha_composite` in [`train-lora.py`](../../../scripts/train-lora.py)'s image-prep helper). Alpha cannot
   survive here and is not meant to; SDXL's latent space has no channel for it.
2. **Generation** — `generate_raw_task` / `generate_core_task` in `tasks.py`.
   Returns opaque RGB. Prompt and negative prompt are the only levers, and on
   distilled checkpoints (`sdxl-turbo`, guidance 0) the negative prompt does
   nothing at all — `resolve_sampling_params` strips it and says so.
3. **Cutout** — `remove_background` ([`tasks.py:1402`](../../../src/sprite_generator/tasks.py)),
   plus `_isolate_largest_sprite` and `strip_ground_patch`. This is the only
   stage that produces alpha.

The consequence is worth stating plainly, because it inverts the intuitive
plan: **you cannot train a model to output transparent backgrounds.** What you
can train is a model that reliably produces a *flat, unique, border-connected*
backdrop and never fuses anything to the subject — because that is the input
`remove_background` can actually cut.

---

## 2. What the reference set teaches (334 `core` images)

`ref_core_*` is the entity reference set. Audited with
`scripts/audit-entity-refs.py`, every finding checked against contact sheets by
eye:

```
kind=core  images: 334
  fit for entity training:      1
  BLOCKING (certain):         115
  REVIEW  (measured, judge):  218

    323  background is flat, not transparent
     98  multi-subject
     50  stray marks beside the subject
     37  pedestal or ground patch
      8  checkerboard baked in
      7  opaque background
      6  off centre
      4  full bleed
      2  dense atlas
      1  clean
```

**Of 334 entity references, 4 have any real transparency, and 1 comes through
every rule clean** — `ref_core_08e39eb3c931`. Call it **2 by eye**:
`ref_core_ca0070408096` is flagged only by the stray-marks rule, and its
"strays" are the detached drips of a dripping beast, which are the art (see §7).
The rest are concept-art boards, asset atlases and marketplace pages: figures on
white, black, grey, parchment and gradient backdrops.

### The defects, in order of how much they cost

**Opaque backdrops (323).** The defining problem. These are not cutouts and
were never cut out. Most are *keyable* — a flat white or grey that the flood
fill would remove — so the images are recoverable, but as stored every one of
them teaches "an entity comes with a backdrop". Seven are worse: ornate
full-page character sheets on parchment with gilt borders and body text, where
there is no flat region to key at all.

**Multi-subject sheets (98).** Icon grids, weapon racks, tree collections,
character line-ups. This is the failure mode this repo has already been burned
by twice: `CORE_TRIGGERS` in `tasks.py` records that a sheet-trained trigger
returned four characters in a row *at guidance 7.5 with duplicate-suppression
negatives*, and `8e738ef` records a terrain adapter that learned grids and kept
drawing them when asked for a single tile — *after* the sheets had been split
into single-subject cells (`43af8c1`), which by itself did not fix it.
**Training beats guidance.** A quarter of the entity set is teaching layout.

**Baked-in checkerboards (8).** The nastiest, because it survives review: the
transparency checkerboard is *painted into the pixels*. The image looks like a
clean cutout in any viewer that draws a checkerboard behind alpha, and it is a
fully opaque square of grey squares. `remove_background` cannot touch it — a
flood fill keys one colour and this is two — so an adapter trained on these
learns to draw the grey checker pattern *as part of the subject*.
Files: `039fd1bc0d72`, `189b515f46c9`, `3089d99f1c8a`, `9e6ee38ae4c6`,
`a9a6b26fb952`, `bf85cd390051`, `cff50aa39462`, `fe898e015a5d`. Two of them
check in pink and navy rather than grey, which is how the grey assumption in
the first version of the detector got caught.

**Pedestals and ground patches (37, needs eyes).** Bases, plinths, dirt discs,
water pools and contact shadows fused to the feet. `NEGATIVE_SINGLE` exists
precisely because of this — the comment at
[`tasks.py:376-390`](../../../src/sprite_generator/tasks.py) explains that a
fused ground patch "is neither background-coloured nor a separate blob, so
neither `remove_background` nor `_isolate_largest_sprite` can touch it". This
count is the honest one: the detector is `strip_ground_patch`'s own width rule
and inherits its documented gap, so it is reported with its number for a human
to judge rather than acted on blind. Confirmed by eye in the contact sheet:
`3214bbee74f0` (tree on a green ground disc), `f197486c7e9a` (creature standing
in water), `2ad3524afa1b`, `5121288de0f5`, `02fd319439e9`.

**Stray marks beside the subject (50, needs eyes).** A detached blob clear of
the entity — most often a **drop shadow floating free of the feet**, sometimes a
loose speck of foliage or a signature. Added late, after
`scripts/audit-character-refs.py` flagged one of the four images this note had
called clean.

It is worth spelling out why it needed its own rule, because it fell exactly
between the other two: `pedestal_ratio` measures *width at the base*, so it only
ever sees a shadow **fused** to the feet, and `subject_count` ignores blobs under
1% of the frame, which is the right floor for "is this a multi-subject sheet"
and far too coarse here — the shadow under `ref_sprite_790879270b49` is 0.22% of
the frame. A detached shadow was invisible to both.

This matters for **training specifically**. `_isolate_largest_sprite` deletes
these at generation time, so they never reach a finished sprite; but nothing
deletes them from the training set, and an adapter simply learns to draw the
litter. Reported as review, not blocking, because a detached blob is sometimes
real art — floating leaves, sparks off a torch, a gleam beside a blade.

**Watermarks and licensed studio art.** Not automatically detectable and worth
naming anyway: `ref_core_039fd1bc0d72.png` carries "THE UNLIVING © ROCKETBRUSH
STUDIO LTD." burned into the bottom-left corner. An adapter trained on it can
reproduce the lettering. There is also a licensing question here that is not a
technical one, and it is not mine to answer.

### The `sprite` set is not a second chance (149 images)

`ref_sprite_*` was audited the same way, since it is the other set an entity
adapter might draw on:

```
kind=sprite  images: 149
  fit for entity training:      1
  BLOCKING (certain):         129
  REVIEW  (measured, judge):   19

    132  background is flat, not transparent
     98  multi-subject
     23  dense atlas
     14  stray marks beside the subject
     12  checkerboard baked in
      8  too small
      4  pedestal or ground patch
      1  clean
```

**This is not a defect of the sprite tab.** A sprite reference is *supposed* to
be a sheet — that is what the kind means, and 121 of 149 being sheets or atlases
is the set working as intended. It is simply the wrong shape for entity
training, and it is recorded here so that "just train on the sprites instead"
is answered with a number rather than tried.

Four looked like the target shape — single pixel-art plants, centred, on real
transparency — and **three of them have a detached drop shadow**:
`25371f4ebb61` and `790879270b49` carry a grey ellipse sitting clear of the
trunk, `5b28b6c2f199` a loose speck at the base. All three confirmed by eye.
The one that survives is `06074972c0ad`.

`audit-character-refs.py` also finds `06074972c0ad` notable, and **rejects it**
— 24,268 colours and no pixel grid, so it is not the finished pixel art the
`sprite` tab means. The agreement is narrower than a matching count would
suggest and is stated carefully in §7: the two audits agree this image is
cleanly *cut out*, and disagree about whether it is *usable*, because they ask
different questions.

Across both sets: **483 references, 2 fit to teach an entity cutout** —
`ref_core_08e39eb3c931` and `ref_sprite_06074972c0ad`.

### The dataset that DOES work: `images/recovered/cells` (103 images)

Added after the fact, and it is the most useful number in this note.

Another session keyed the 12 checkerboard-backed RPG-Maker sheets and split them
into single-character cells (`images/recovered/`). Run this audit's rules over
them:

```
kind=core  images: 103
  fit for entity training:    103
  BLOCKING (certain):           0
  REVIEW  (measured, judge):    0

    103  clean
```

**Every one of them, clean.** Checked by eye across the contact sheets: single
whole characters, centred, on real transparency, no pedestals, no ground
shadows, no backdrop. This is the shape the entity conveyor needs, and it did
not exist in any `ref_*` set.

So the conclusion in §4 changes in an important way. It is not "there is no
usable entity data" — it is **"the usable entity data is the recovered cells,
and the `ref_core` set is not it"**. 103 clean single-subject cutouts is a
plausible floor for a first entity adapter; 1 was not.

### The sliver bug, and why it was not a margin setting

The first pass over these cells flagged 9 with a detached sliver beside the
character — on `cell_sprite_sprite_291203136719_00*` plainly a staff belonging
to the NEIGHBOURING character in the source sheet.

The obvious reading, and the one this note first wrote down, was "the cell
margin is too generous". That is **wrong**, and the correct diagnosis is worth
recording because it generalises to every grid split this repo will ever do:
`find_cells` returns a bounding BOX, and characters are not box-shaped. A
wizard's staff leans up and to the left, so it overhangs the corner of the box
belonging to the character beside him. The cell is genuinely single-subject by
component analysis *and* contains a piece of someone else. No amount of
shrinking the pad fixes that.

Fixed at the split by the session that owns it, as an opt-in
`--drop-edge-slivers` in `scripts/split-sheets.py`, on two conditions that must
BOTH hold — a blob is dropped only if it is small **relative to the main
subject** and **touches the crop edge**. Each alone is unsafe: the relative test
by itself destroys the dozen dwarf-and-wolf cells where a wolf companion stands
beside the ranger and both are wanted; the edge test by itself clips a subject's
own detached parts. Together they are precise, because a piece of the neighbour
is necessarily clipped by the boundary it came across, while a spark or a
dropped item sits inside the cell.

26 fragments erased across the 103 cells. Verified here independently
afterwards: all 9 previously-flagged cells re-read clean with the characters
fully intact — staff and its lit tip, belt dagger, both swords, cloaks — and the
audit now returns 103 clean, 0 review.

### `ref_tile` and `ref_map`: measured, and confirmed out of scope

These two sets hold the other 2,115 references. This note originally asserted
they were a different job and skipped them. Asserting is not measuring, so they
were run too — and the assertion holds, in the strongest available form:

**0 of 2,115 tile or map references are entity-shaped.** The test was
deliberately generous — real alpha in use, exactly one subject, under 60% frame
coverage, touching at most one edge — and nothing matched. There is no misfiled
entity hiding in the tile tab, and no additional training data to recover there.

**The rest of the tile numbers are artefacts and must not be quoted.** Applying
an entity rule to a ground texture produces nonsense in a specific and
instructive way: 138 tiles came back "nothing visible", which sounds like 138
blank files. They are not. Checked: 31x30 to 65x55 pixels, 231 to 649 distinct
colours each, zero transparency — ordinary textured tiles. A full-bleed texture
has no backdrop to distinguish from a subject, so the corner flood fill spreads
across the whole low-contrast image and keys it away, and "coverage 0" follows.
The rule assumes a subject sitting on a separable background; a tile is
definitionally the opposite.

The one number that does corroborate something: 1,340 of 2,001 tiles are under
the training-size floor. That matches `curate-training-set.py`'s independent
finding of a median short side of 63px, and it is that script's verdict — not
this one's — that should govern the tile set.

Per-file marks were deliberately **not** kept for these two kinds. A file
containing 1,708 confidently-worded "blocking" verdicts that are mostly category
error is a hazard to whoever finds it later. Regenerate the summary in one
command if it is ever wanted:

```bash
python scripts/audit-entity-refs.py --dir images --kind tile
```

---

## 3. Three findings in the code, independent of the data

### 3a. The trainer composites alpha away — correctly, but nothing replaces it

`cache_inputs` flattens every reference onto grey `#808080`
(the `alpha_composite` in [`train-lora.py`](../../../scripts/train-lora.py)'s image-prep helper). The comment
defends the choice well: compositing onto black would teach a black background.
The choice is right and there is no alternative in SDXL.

But it means the 4 genuinely-transparent references are worth no more to the
trainer than the 330 opaque ones — they simply become entities on grey. The
adapter's notion of "background" is whatever backdrops dominate the set, and
right now that is *every backdrop at once*: white, black, grey, parchment,
gradient, checkerboard. A model taught that backgrounds vary will produce
varied backgrounds, and a varied background is the one thing a corner flood
fill cannot key.

**The fix is not "train on transparency".** It is to make every training image
carry the *same* flat backdrop — grey `#808080`, matching what the compositor
already uses — so the adapter learns one background colour and the cutout stage
has a constant to key against.

**Half of that is now in place, from another session's work.** `prepare_image`
gained a `--fit` mode defaulting to **`pad`**, and the padding is filled with
the *same* `background` colour as the alpha composite. So a non-square reference
is now centred on one continuous grey field rather than cropped. Two
consequences, both good for this goal:

- The uniform-backdrop half of the recommendation above is satisfied for the
  padding; what remains is the images that arrive with a backdrop of their own.
- **A clipping risk that was real is closed.** The previous behaviour
  centre-cropped to square, and the recovered cells are not square (216x226 is
  typical). Cropping a tall character cell to a square takes it off the top or
  bottom — heads and feet — silently. `pad` cannot clip by construction. The
  Celery path inherits this: `train_lora_job` passes `--fit` only when the run
  config sets it, so the default applies.

### 3b. Captions were storage hashes — FIXED while this was being written

Recorded because it shaped the recommendations below, not as an open defect.

`caption_for` used to build `"{trigger} pixel art sprite, {stem}"` where `stem`
was the filename after the `ref core ` prefix strip — which left the 12-hex-char
storage id, so every caption in every run read
`pixelsprite pixel art sprite, 039fd1bc0d72`.

**Another session fixed this during this investigation** (see the working-tree
change to `scripts/train-lora.py` and `.ai/decisions/0009-character-training-dataset.md`).
`caption_for` now takes `label` and `body`, drops a stem that is pure storage
naming, and the call site passes labels through from the manifest. That session
also judged the per-image hash *actively* harmful — a unique handle the text
encoder can hang one image's specifics on — where this note had it as merely
inert. Their reading is the more careful one.

What that fix does **not** yet do is give the set a background vocabulary: the
default body is `"pixel art sprite"`, with nothing saying "on a flat grey
background". That half of recommendation 5 still stands — you cannot prompt for
what was never captioned.

### 3c. The entity path skips the cleanup every other path applies

`generate_raw_task` is what the A1111 facade calls, which is what something2
calls to generate entity images with `cutout: true`. Its cutout is:

```python
img = remove_background(img)          # tasks.py:2374
```

Every other cutout in the codebase is not:

| Call site | What it does |
|---|---|
| `tasks.py:1380` | `remove_background(img, keep_largest=True)` |
| `tasks.py:1593` | `remove_background(img, keep_largest=True)` |
| `tasks.py:1599` | `strip_ground_patch(img)` |
| `tasks.py:2036` | `remove_background(img, keep_largest=True)` |
| **`tasks.py:2374`** | **`remove_background(img)` — neither** |

So the one path built for entity assets is the only one that keeps stray
duplicate blobs and fused ground patches. `generate_raw_task`'s docstring
justifies staying "plain" for the *prompt*, and that reasoning is sound — sprite
styling is wrong for a caller asking for a tileable ground texture. It does not
extend to the cutout, which is already gated behind `if strip_background:`. A
caller that asked for a cutout has asserted the image contains one object.

**`keep_largest=True` is the safe half of this** and belongs under that guard.

**`strip_ground_patch` is NOT safe to add blindly**, and this matters: its own
docstring says "use on rest-pose turnarounds", and its rule is "anything at the
bottom much wider than the shins is not the body". A tree, a bush, a barrel, a
rock or a chest *is* legitimately widest at its base. Applied to something2's
entity vocabulary it would amputate the bottom of a large fraction of props.
It should be opt-in per request, not automatic.

### 3d. Flood fill is the wrong tool for the remaining cases, and the right one is already vendored

`remove_background` handles a flat backdrop well and structurally cannot handle
a gradient, a vignette, a two-tone checkerboard, or a shadow that fades into the
floor. `src/mesh_worker/mesh.py:73` uses **`rembg`** for exactly that job.

**This has already been considered and rejected here, and the rejection is
documented.** `compose/develop/sprite_generator/requirements.txt` removed rembg
with the reasoning spelled out: it ships without `onnxruntime` as a hard
dependency so the import failed at runtime, it was never actually used because
`remove_background` is a border flood fill, and — the part that matters —
"neural matting would also be the wrong tool for pixel art, which needs hard
edges, not soft alpha". The file says to re-add it "as `rembg[cpu]` only if
photographic matting is ever wanted".

So this is **not** a free upgrade sitting unused in the tree; taking it means
reversing a decision that was made deliberately. The only version worth
reopening it for is "use rembg to get the *mask*, then hard-threshold it to 0/255
alpha", which answers the soft-alpha objection but not the onnxruntime weight.
Treat it as the last resort, after the data is fixed — most of these cutouts
fail because the training set taught a textured backdrop, not because the flood
fill is too weak.

---

## 4. What to do, in order

Ordered by cost-to-benefit, not by how interesting each one is.

1. **Do not train an entity adapter on `ref_core` as it stands.** One of 334
   images comes through clean, two if you allow the one whose "strays" are
   really its own drips. This is the finding that matters most; everything else
   is cheaper than the retrain it prevents.
   The live-data version of this, measured by ADR 0009's session against the
   database rather than the filesystem, is sharper still: a default
   `POST /api/training` today reads **435 live trainable rows** (333 core, 102
   sprite) and **230 of them are rejected** by that audit — **all 12
   checkerboarded sheets among them, still `deleted=false, trainable=true`**. So
   this is not a hypothetical about a future run; it describes what the next
   training job would consume if started now.
2. **Train from `images/recovered/cells`, not from `ref_core`.** All 103 cells
   are clean by these rules and none is blocking — ready as they stand, with the
   edge slivers already erased at the split. This supersedes what this note
   originally recommended — "rebuild the set from the ~200 keyable single-subject
   references" — because someone already built a better one by keying and
   splitting the checkerboard sheets.
   Rebuilding from `ref_core` is still worth doing *afterwards*, to grow the set
   past 103: run the keyable single-subject references through
   `remove_background`, re-store as RGBA, split the 98 sheets with
   `scripts/split-sheets.py`, and re-audit the cells. The terrain work
   (`43af8c1`, `28e32a5`) is the template, including its warning that splitting
   alone was not enough.
3. **Delete the 5 baked checkerboards and the 7 unkeyable character sheets.**
   These are not recoverable by any automatic step.
4. **Add `keep_largest=True` at `tasks.py:2374`.** One argument, matches every
   other cutout in the file, already correctly gated.
5. **Give captions a background phrase.** The label half of this is already
   done (see 3b); what remains is appending a constant phrase naming the flat
   backdrop to every caption, so a token exists to prompt with at inference.
6. **Last resort only: rembg-with-hard-threshold** for subjects on non-flat
   backdrops, behind a per-request flag — and only after 1–5, because it means
   reversing the documented decision in `requirements.txt` to drop it.

Steps 1–4 need no new model and no GPU time.

---

## 5. Reproducing this

```bash
# Any container with pillow, numpy, scipy.
python scripts/audit-entity-refs.py --dir images --kind core \
    --json images/audit_core.json \
    --markdown images/_audit/core_marks.md \
    --contact-sheet images/_audit/core

# Write the verdicts back, once the database has reference rows again:
python scripts/audit-entity-refs.py --dir images --kind core --apply
```

`--apply` writes BLOCKING findings to `reference_assets.trainable_why`, which
`ReferenceTab.tsx` already renders under any image marked not trainable, and
records every measurement in `metrics->'entity_audit'`. REVIEW findings are
recorded but never used to exclude an image, because both detectors behind them
have failure modes a human can see and the measurement cannot.

> **CORRECTION.** An earlier version of this note said "the database is
> currently empty — `reference_assets` does not exist, and the Docker images
> were pruned too". **That was wrong, and it was wrong for an embarrassing
> reason: there are two Docker daemons on this machine.** Windows Docker Desktop
> (`desktop-linux` context) has never run this project, so it has no project
> images and no data. The real stack runs in **WSL Ubuntu's own docker**, and it
> was up the whole time — `stats_db`, `sprite_generator`, `sprite_worker`,
> `redis_broker`, all healthy for hours. `llm_monitoring` holds **2,555
> references, 573 trainable** (tile 2001, core 334, map 114, sprite 106).
>
> I queried a database I had started myself, on the wrong daemon, from an empty
> data directory, and reported the emptiness as a fact about the project. The
> lesson is the one this whole note keeps re-learning: `docker ps` answered, so
> it looked like an answer about the project, and it was an answer about a
> different machine. Reach the real stack with `wsl -d Ubuntu docker ...`.
>
> Nothing was damaged — the duplicate container was on a separate daemon with a
> separate filesystem. Verified afterwards: 2,555/573 unchanged, no `audit_smoke`
> on the real instance, and the stray container and network removed.

So **`--apply` can be run for real**, against 573 trainable rows. It has not
been, deliberately: it would flip verdicts, change what `ReferenceTab.tsx`
displays and what the next training run consumes. That is a decision for whoever
owns the dataset, not a step in verifying a script. The verification below used
a throwaway database instead.

One mismatch worth knowing before running it: the DB holds **106** live sprite
rows while there are **149** `ref_sprite_*.png` files on disk. Chased down by
ADR 0009's session: **all 43 are soft-deleted rows** whose files remain on disk —
150 sprite rows total, 106 live and 44 deleted, with zero orphans in either
direction. `core` matches exactly at 334, which is why the gap was easy to miss.

That has a consequence for **this** note too, and it is not cosmetic: this audit
reads the **filesystem**, so its sprite figures count images the user has
already thrown away. Over live references only, the sprite set has *no*
survivors at all — the three that passed here (the tree on a plinth, the hollow
fragment, the pine tree) are all deleted rows. `--apply` was never affected; it
scopes to `deleted = false`. Only the reporting over-counts.

**`--apply` has been verified, not merely written.** It would otherwise have
shipped having never executed once. It was exercised against a throwaway
`audit_smoke` database built from this repo's own migrations (013/014/015),
seeded with rows pointing at real files, and dropped afterwards; no production
database was written to. A scratch database is the right venue regardless of
whether the real one has rows — the point is to exercise the write path without
changing anything anyone depends on. Four behaviours were checked, each chosen
so that it fails if the code is wrong rather than merely passing when it is
right:

| behaviour | evidence |
|---|---|
| a BLOCKING image is excluded, with its reason | two rows went `trainable = false` carrying the multi-subject and checkerboard text |
| a clean image is **not** flipped | the clean row kept `trainable = true` and its prior `trainable_why` |
| `metrics` is MERGED, not replaced | a pre-existing `metrics->>'coverage'` survived alongside the new `entity_audit` |
| `deleted = false` guards the write | a soft-deleted row came through with no `entity_audit` and nothing altered |

The metrics-merge case is the one worth keeping: `||` on a jsonb column is easy
to write as an overwrite by accident, and the failure would be silent
destruction of every measurement `measure.py` had already recorded.

**And a bug that this exact test could not catch, found afterwards.** `--apply`
matched `file_path` against `os.path.join(a.dir, file)`. The database stores
absolute container paths (`/app/images/ref_core_<hex>.png`), while this script is
normally pointed at a relative `--dir images` from the host — so the comparison
was `images/...` against `/app/images/...`, which matches nothing. Zero rows
updated, "applied to 0 rows", exit 0. A silent no-op reported as success.

The scratch test could not have found it, and the reason is worth keeping: the
test **seeds whatever paths the test invocation produces**, so both sides agreed
and the bug was invisible from inside. It surfaced only from reading the real
`file_path` column. A test built from your own assumptions cannot falsify them.

Fixed two ways — match on the `/<basename>` suffix so either form works, and
**report what did not land**, since an audit that quietly updates nothing looks
identical to one that had nothing to change. Re-verified against a scratch
database whose schema was `pg_dump`ed from production and seeded with
production-format paths: 3 of 3 rows matched and flipped correctly, and the 37
findings with no row were named rather than swallowed.

---

## 6. What was measured and then thrown away

Recorded because this repo has a habit of shipping confident detectors that turn
out backwards (see `c967bed`), and the near-misses are worth keeping:

- **"Wider at the bottom than the shins" as a pedestal test, applied to every
  image.** It fired on every icon grid in the set — a grid's bottom row is wider
  than any imagined shin. Now restricted to single-subject images, which dropped
  the count from 58 to 37 and removed every false positive visible in the
  contact sheet.
- **"No subject above 1% of frame means the image is empty."** Reported two
  64-icon atlases as "nothing visible". They are the opposite of empty; they are
  dense. Now a separate `dense atlas` verdict.
- **Checkerboard detection from a corner block.** Wrong twice: the checker is
  often only *behind* the subject with a flat margin around it, and a block big
  enough to hold the ~36px pitch also contains the coloured subject, which fails
  a whole-block greyness test. Now measured on grey pixels across the whole
  image, with the pattern reconstructed and fitted rather than described.
- **My own eyeball read of `ref_core_b88ccc34e525` as a checkerboard.** It is a
  flat grey backdrop. The detector said 0 and was right.
- **Measuring colour on `convert("RGB")` of an RGBA image.** Keying a
  checkerboard away sets alpha to 0 and LEAVES THE OLD RGB IN PLACE, so
  discarding the alpha hands the dead backdrop straight back. The checkerboard
  rule reported "baked in" on 14 of the 103 correctly-keyed cells in
  `images/recovered/cells` — condemning the repaired copies, the worst direction
  for the error to point. Now every stage of that rule is masked to opaque
  pixels; all 8 genuine core checkerboards survive the change.
- **Believing two detectors covered "nothing else in frame".** They did not.
  `pedestal_ratio` only sees what is FUSED to the base and `subject_count` only
  sees blobs over 1% of the frame, so a detached drop shadow at 0.22% was
  invisible to both — in three of the four images this note had called clean.
  Found by `scripts/audit-character-refs.py` flagging one of them for an
  unrelated reason, not by anything here. Hence `stray_blobs`.
- **The assumption that a transparency checker is grey.** It cost two
  detections: `ref_core_cff50aa39462` checks in pink, `ref_core_fe898e015a5d`
  in navy. Editors let people recolour the checker and people do.
- **Counting exact colours to find the two tones.** These are JPEG-sourced
  images with paper texture, so one tone fragments across dozens of near
  values and none is dominant. Now counted on a coarse colour grid.

## 7. Overlap with ADR 0009 and `scripts/audit-character-refs.py`

Both appeared in the working tree during this investigation, from another
session. [`0009-character-training-dataset.md`](../../decisions/0009-character-training-dataset.md)
audits the same 483 references, and its script covers much the same ground.
**Someone should decide whether the two auditors merge**, because two auditors
of one reference set will drift apart.

**They agree, which is the useful part.** Reached independently, by different
measurements:

| | ADR 0009 | this note |
|---|---|---|
| core usable | 2 | 1 (2 by eye) |
| sprite usable | 2 | 1 |
| core checkerboards | 7 | 8 |
| sprite checkerboards | 12 | 12 |

**The "usable" rows are not comparable and should not be read as agreement.**
An earlier draft of this note claimed the two audits converged on one sprite.
They do not, and 0009's author caught it. The counts are close by coincidence
and they are counts of *different files*: 0009 keeps `6f8abf25aa2b` and
`bd7e43ed3cff`; this note keeps `06074972c0ad`, which 0009 **rejects** at 24,268
colours with no pixel grid.

Run this note's rules over 0009's two keeps, which nobody had done:

| file | verdict here |
|---|---|
| `6f8abf25aa2b` | pedestal 5.8x — **false positive**, no pedestal; it is a hollow architectural fragment whose wide bottom rail beats its thin sides |
| `bd7e43ed3cff` | pedestal 2.9x — **true positive**, a grey stone plinth under the trunk, plus one stray mark |

So one of 0009's keeps carries exactly the defect this note exists to find, and
this note's own pedestal rule produces a clean false positive on the other. That
is the cross-reference earning its keep, and it is worth more than a matching
number would have been. The checkerboard rows are the ones that genuinely agree.

`6f8abf25aa2b` also shows a gap in **both** audits: it is a fragment of a
structure with a hollow centre, not a whole entity, and neither of us has a rule
that notices.

Two deltas worth resolving rather than averaging:

- **core checkerboards, 7 vs 8.** The extra is `ref_core_a9a6b26fb952`, a
  clipart-site PNG whose subject touches the border so there is no clean border
  strip to read. Confirmed by eye. The union is now taken here, so this script
  reports 8; 0009's own count is the one that is low.
- **core usable, 2 vs 1 — resolved, and 0009 is right.** Adding the stray-marks
  rule dropped this note's core count to 1. The image we differ on is
  `ref_core_ca0070408096`, a beast whose design includes dripping tendrils; the
  drips detach into separate blobs, so the stray rule fires. Looked at: they are
  **legitimate art, not litter**. This is precisely the false positive the rule
  is documented to produce, which is why it is review-tier and never blocking.
  Counted by eye, core usable is **2**. Counted by machine it is 1, and the
  machine is wrong here.

0009 also measures three things this note does not, and they are worth having:
near-duplicate detection (24 sprite references are hash-duplicates), pixel-grid
and colour-count analysis, and it names a Shutterstock-watermarked frame and a
tileset-store screenshot with browser chrome still in it.

They are not redundant today:

| | `audit-character-refs.py` | `audit-entity-refs.py` (this one) |
|---|---|---|
| Asks | can this teach a style adapter at all? | can this teach an *entity cutout*? |
| Unique to it | pixel-grid detection, palette size, outline width | background keyability, pedestal ratio, centre drift, full bleed |
| Checkerboards | border two-tone runs | ideal-checker model fit |

The checkerboard detectors are genuinely complementary, and rather than pick a
winner this script now **runs both and takes the union** — see
`_sibling_checkerboard`. The border test finds 3 core and 12 sprite references
this one cannot (their checker is nearly white, or their two tones are too
close to separate); the model fit finds `ref_core_a9a6b26fb952`, whose subject
touches the border and so leaves no clean strip to read. Every hit from both
was confirmed by eye before the union was taken.
