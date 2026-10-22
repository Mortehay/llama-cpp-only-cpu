# Sheet quality: per-frame checks, redraws and the frame store

Date: 2026-10-22. Status: implemented; the measurements below are reproducible
from the two evidence jobs, which are still on the box.

The goal this answers: a full character sheet (7 actions x 6 frames x 8
facings = 336 cells) came back **unusable in game**, with four named
symptoms — a character facing the opposite way from the row it is in, cells
zoomed so only the chest shows, background baked in behind the figure, and
identity drifting through attack/cast/use.

| What | Where |
|---|---|
| The checks | [`src/sprite_generator/sheet_qc.py`](../../../src/sprite_generator/sheet_qc.py) |
| Re-score a finished job | [`scripts/score-sheet.py`](../../../scripts/score-sheet.py) |
| Assembly invariants (no GPU) | [`scripts/check-sheet-assembly.py`](../../../scripts/check-sheet-assembly.py) |
| Cost of extra conditioning images | [`scripts/bench-frame-context.py`](../../../scripts/bench-frame-context.py) |
| The conveyor | [`scripts/build-sheet.py`](../../../scripts/build-sheet.py) |
| Evidence job, flat concept | `images/_job_212d56d7-c934-4a76-ae19-3fc87d438205` |
| Evidence job, painterly concept | `images/_job_8d9d164d-32ed-432a-aca1-c228f246d650` |

---

## The one-paragraph answer

Three of the four symptoms were **not** where they looked. The sheet builder
was already applying one scale to the whole sheet, so "each cell is fitted
independently" was never true — what is true is that the scale is the
*minimum* over all cells, so one chest close-up shrinks the other 335. The
baked background was not a prompt problem: the sheet path was the last place
in the codebase still keying cells with the corner flood fill, which
structurally cannot clear a glow, a scene or a painted checkerboard, while
BiRefNet (already on the box, already used for object cutouts) clears 9-39%
of frame on exactly those cells. The spell effects were not a missing
negative prompt — `action_prompts.json` asked for them by name, and the edit
path runs `true_cfg_scale=1.0`, so a negative prompt is inert and "no glow"
in a positive prompt only adds the noun. Only the facing bug was where it
looked, and it is unfixable by prompt: the western rows are now flipped from
the eastern ones.

---

## 1. What was measured, and what it overturned

All on the raw 512px renders of both evidence jobs, which survive on disk.

### 1.1 Height carries no signal; area does

Pre-fit figure height is **478-512 of 512 on all 336 cells** of 8d9d164d,
median 512. The angles LoRA frames every subject edge to edge, so a torso
close-up is 512 tall and so is a full body.

The design brief proposed `body height before fitting > 1.25x the reference`.
That rule passes every close-up in both sheets. Area against the direction's
reference separates cleanly instead:

    clean cells (idle, walk, both sheets)       1.07 - 1.09 x
    8d9d164d attack s f2 / f4 (chest close-up)  1.85 / 1.98 x

### 1.2 One scale already, and that is the problem

`sheet.SheetBuilder.build` has always computed a single scale for the sheet.
It is the **minimum fit scale over every cell**, and it is bound by the
*widest* cell, not the tallest — which is why every figure measured 39-46 px
tall in a 64 px cell. So the close-up does not get "scaled to full height";
it gets rendered slightly larger than its neighbours while a cell with a
keyed-in backdrop blob silently shrinks all 336.

The fix is therefore not to the scale rule. It is to stop producing close-ups
and backdrops, and to measure cells **before** the fit, where the
contamination has not yet spread.

### 1.3 Two masks, because two defects look identical in one

The something2 spike could not tell a close-up from baked backdrop: both are
"lots of opaque pixels", and its backdrop rule fired zero times because those
cells had already been claimed by its close-up rule. Measuring the frame
twice separates them —

    cell                        flood%  figure%  backdrop%
    8d9d164d idle s f0           45.5    42.0      3.6   clean
    8d9d164d walk e f0           44.1    40.9      3.3   clean
    8d9d164d attack se f3        76.8    37.7     39.1   painted checker
    8d9d164d cast s f1           55.0    34.2     22.7   spell glow
    8d9d164d attack w f1         51.0    38.4     12.6   white blob
    8d9d164d attack se f4        92.6    85.3      9.0   checker + close-up
    8d9d164d attack s f2         78.6    77.5      1.1   close-up, no backdrop

— where `flood` is the corner flood fill and `figure` is BiRefNet. The clean
population sits at 3.1-3.6% and every named backdrop cell at 9% or more.

`backdrop%` is the **set difference** `(flood AND NOT figure)`, not the
difference of the two columns. On attack se f4 those are 9.0% and 7.3%, and
the first version of the module used the subtraction and let that cell — one
the owner named — through an 8% threshold.

### 1.4 The flood fill was the background bug

`stage_compose` keyed every cell with `pixelate.key_background`. That is the
one place in the pipeline still doing so: object cutouts moved to BiRefNet in
`.ai/specs/transparent-cutouts/` precisely because a corner flood "cannot
clear a painted transparency checkerboard (two tones), a wall or scene
filling the frame, a floor with a shadow, or a pocket of backdrop enclosed by
the subject". Looked at directly, 8d9d164d attack se f3 is a figure standing
on a **painted transparency checkerboard** — the failure CLAUDE.md already
documents for positive-prompt transparency, reaching the sheet lane.

Switching the sheet path to BiRefNet removes that backdrop with no redraw and
no prompt change. It does **not** fix a close-up (attack s f2 is 77.5% figure
by BiRefNet too — it really is a chest filling the frame) and it does not
remove a glow attached to a hand.

### 1.5 `head_lean` catches the facing flips that silhouette IoU missed

The spike's silhouette IoU "MISSED Cultist:1 attack e f3", the cell the owner
pointed at, because its anchor is the idle frame: a big pose scores low
against a narrow standing figure both ways round and the margin is noise.

`head_lean` asks where the top of the figure sits relative to its hips, as a
fraction of its own width. That is stable under pose change and flips sign
with the facing. Anchored **per direction**, because the value is a property
of the view:

    anchor    dir_e   dir_w   dir_s   dir_n
    212d56d7 -0.118  -0.176  -0.174  +0.177
    8d9d164d -0.096  +0.098  -0.193  +0.208

    212d56d7 attack e  f0..f5  -0.106 -0.119 -0.177 +0.196 -0.016 -0.113
    8d9d164d attack e  f0..f5  -0.090 -0.075 -0.116 +0.101 -0.079 -0.079
    212d56d7 idle   ne f0..f5  +0.116 -0.087 +0.116 -0.088 -0.101 +0.118
    212d56d7 walk   e  f0..f5  -0.078 -0.097 -0.115 -0.104 -0.113 -0.114

`attack e f3` flips sign on **both** sheets, so it is a property of the pose
("arms fully extended forward, strike landed") and not of a seed. `idle ne`
alternates exactly as the brief describes. The two clean walk rows hold their
sign on 12 of 12 frames.

An embedding check (DINO/CLIP) was considered and is not needed:
`openai/clip-vit-large-patch14` is in the cache **tokenizer-only**, with no
vision weights, so it would have meant a new 1.7 GB download and a
`models.toml` stanza to answer a question two numpy functions answer.

### 1.6 se and sw are the front view, and it starts at the turnaround

Mean absolute difference between turnaround views, before any action is
rendered:

    |s-se|  |s-sw|  |s-e|  |s-n|   |se-sw|  |se-mirror(sw)|
    12.8    15.3    52.2   39.0    18.1     43.6             212d56d7
    16.4    28.4    60.4   48.6    33.9     53.7             8d9d164d

se and sw sit 3-4x closer to the front view than e does, and they are near
duplicates of **each other** rather than mirrors. So the front quarters
collapse onto the front view at the turnaround stage, and every action cell
inherits it because each one is conditioned on `dir_<d>.png`.

This is the same class of failure as `left side` (qwen_edit, AZIMUTH_HINTS)
and is **not fixed here**. The mirrored-source trick cannot help: the
collapse is toward a near-symmetric front view, so mirroring maps it onto
itself — which is why `sw`, which already renders from a mirrored source,
still equals `se`. The back quarters are fine: |ne-nw| 49.0/58.1 against
|ne-mirror(nw)| 26.8/33.6, so `nw` is a correct mirror of `ne`.

### 1.7 The effects were requested by name

`action_prompts.json` asked for them: "a small glowing ember gathering in the
hand", "bright light growing around the hand", "a glowing orb held at the
fingertips", "glowing sparks around both hands", and for `use`, "holding an
object out in front". The masked-colour identity check scores the cast row
0.082/0.064 against 0.009 for idle, on both sheets.

The brief proposed adding "no spell effects, no glow, no aura, no energy, no
particles, no background" to every action prompt. In this pipeline that is
two mistakes at once: `denoise_cells` passes `true_cfg_scale=1.0`, so there
is no classifier-free guidance and a negative prompt is inert; and those
nouns in a **positive** prompt land in the conditioning, which is the
documented "no frame, no border, no card" trap.

The poses now name hand and arm positions only. The rule, and the measured
2026-08-25 fix it supersedes, are recorded in the file's own
`_readme.sheet_frames.effects_stay_on_the_body`.

---

## 2. The checks

`sheet_qc.py`, on the raw 512px render, against a `Reference` built from that
direction's `dir_<d>.png`. Thresholds sit in the measured gap and are env
overridable so a finished job can be re-scored at a different cut.

| check | rule | clean | failing |
|---|---|---|---|
| `close_up` | figure area / reference > 1.40 | 1.07-1.09x | 1.85-1.98x |
| `backdrop` | (flood AND NOT figure) > 0.08 | 3.1-3.6% | 9.0-39.1% |
| `border` | left+right edge opacity > 0.25 | ~0 | a scene, or a close-up |
| `facing` | sign(lean) != sign(reference lean), both outside a 0.04 deadband | same sign | 0.04-0.33 the other way |
| `identity` | masked colour chi-square vs the direction view > 0.030 | 0.006-0.010 | 0.021-0.095 |
| `continuity` | redrawn frame vs its neighbours > 0.060 | — | redraws only |

Top and bottom borders are deliberately excluded from `border`: the LoRA
frames every subject edge to edge vertically, so including them would fire on
every cell.

### What the rates look like

Scored over all 336 cells of each job — the only external check these
thresholds have is whether they land where the owner said the sheets were
bad:

    action   212d56d7  8d9d164d      direction  212d56d7  8d9d164d
    idle        5/48      2/48       s              3/42     11/42
    walk        3/48      3/48       se             8/42     20/42
    sway        3/48     10/48       e             12/42     11/42
    use        15/48     25/48       ne            26/42     25/42
    attack     14/48     22/48       n              6/42      3/42
    damage     17/48     21/48       nw            19/42     14/42
    cast       29/48     40/48       w              8/42     23/42
                                     sw             4/42     16/42

idle, walk and sway — the rows nobody complained about — stay near zero.
`close_up` fires 40 times on the painterly sheet and once on the flat one;
`backdrop` 29 and 8. Both splits match the owner's description of the two
sheets. Overall 89/336 (26.5%) and 123/336 (36.6%) of cells are flagged.

**What the conveyor will actually redraw is smaller than that.** 8d9d164d's
`w` row alone is 23 flags, 19 of them facing, and that row is no longer
rendered — it is a flip of `e`. The `cast` row is another 40, most of them
the glow the pose text asked for by name and no longer does. Only the five
rendered directions are redrawable at all: on 8d9d164d those carry 70 of the
123 flags.

### The anti-goal gate

`score-sheet.py --anti-goals` asserts that every cell the owner named still
fires, for the right reason, and exits non-zero otherwise. This is what stops
the checks degenerating into a check that always passes:

    PASS 212d56d7 attack e  f3  want facing   got close_up,border,facing,identity
    PASS 8d9d164d attack s  f2  want close_up got close_up,border,facing
    PASS 8d9d164d attack s  f4  want close_up got close_up,border
    PASS 8d9d164d attack se f3  want backdrop got backdrop,identity
    PASS 8d9d164d attack se f4  want backdrop got close_up,backdrop,border,facing,identity

---

## 3. The conveyor

### 3.1 West rows are flipped, not rendered

`MIRROR_WEST` (default on). `w`, `nw` and `sw` are produced by flipping the
finished `e`, `ne` and `se` cells, so a 336-cell sheet renders 210.

The facing bug cannot be fixed at this stage — prompt wording, dropping the
camera clause, swapping it, and the mirrored-source trick that fixes the
turnaround were all measured and all came back facing east. So the choice is
a flipped row or a wrong one.

The cost is handedness: a one-handed sword changes hands between the `e` row
and the `w` row. That was rejected before on those grounds and is accepted
now because something2 already mirrors `e`/`ne`/`se` in game. Every flipped
row carries `mirrored_from` and `handedness: "swapped"` in the atlas so a
consumer that does **not** mirror can see it.

The flip happens on the **fitted** cell, not the 512px render. `pixelate`
centres with `(cell_w - width) // 2`, so flipping before the fit rounds the
sprite one pixel to the left in both cells and the pair is off by one — in
game, a sprite that jumps sideways when the character turns around.

### 3.2 Redraws are batched, not row-by-row

The brief asks for row-by-row work: draw a row, check it, redraw what failed,
move on. **That shape cannot be built on this hardware.** A cell is checked
from decoded pixels, and `denoise_cells` cannot decode while the transformer
is resident — the VAE does not fit beside it on a 12 GB card, which is why it
renders every latent first and releases the transformer before decoding.
Checking after each row means reloading the transformer per row: ~90 s x 35
rows, about an hour of pure model loading added to a job.

So: render everything, check everything, re-render the failures together.
Same outcome — every frame checked, failures redrawn, retries capped — at
three model loads instead of thirty-five. Two per-frame rounds on fresh
seeds, then one whole-row round for any row with two or more failures left,
which is the case where the frames disagree with each other rather than each
being individually wrong.

The automatic rounds **reuse the embeddings**. An embedding belongs to the
(prompt, image) pair and a seed re-roll changes neither, so the text encoder
does not run again.

A redraw deliberately overrides a caller-pinned seed, for the reason the
entity-cutout path records: a defect that is a property of the seed can never
be fixed by retrying that seed.

### 3.3 A sheet is a folder of frames

Each finished 48x64 RGBA cell is written to
`_job_<id>/frames/<action>_<direction>/<n>.png`, and `sheet.png` is assembled
from those exact objects, so the grid cannot disagree with the files.

The sheet-wide scale is **pinned** in `config.json` after the first compose.
Without it, re-assembling after a one-frame redraw recomputes the minimum
over a set with one new member, and a repaired frame that is a little wider
rescales all 336 cells — so repairing one sprite would move every sprite in
the game. The pin is ignored when the new cells genuinely need a smaller
scale (otherwise `Image.paste` clips the sprite), and that case sets
`atlas["refit"] = true` rather than claiming only one cell changed.

---

## 4. API additions — all additive

    GET  /api/jobs/{id}/frames                 the per-frame PNG index
    GET  /api/jobs/{id}/frames/{row}/{n}.png   one finished cell
    POST /api/jobs/{id}/redraw                 202, a child job id

`POST /api/jobs` body, job status fields, the `/sheet` PNG layout and the
existing `/atlas` fields are unchanged. The atlas gains `quality` (one entry
per frame: attempts, passed, failed_checks, row_redrawn, and `mirrored_from`
on flipped rows), `frames`, `scale`, and `refit` when one happened.

A redraw is its own job row, so the existing `GET /api/jobs/{id}` polling
works unchanged and it can be cancelled like anything else. It writes into
the **parent's** work directory and rewrites the parent's sheet in place, so
a caller holding `/api/jobs/{parent}/sheet` keeps working. It runs three
subprocess stages — `explicit-encode`, `explicit-denoise`, `compose` — and
the first two are separate processes for the same reason every encode/denoise
pair in the conveyor is: the text encoder and the transformer must not be
resident together on this card. Unlike the automatic rounds it DOES re-encode,
because a named frame is given its neighbours as context and an embedding
belongs to the (prompt, image) pair.

An explicitly requested redraw replaces the frame even when it scores worse.
The caller looked at the cell and said it was wrong; a verdict from five
numbers does not overrule that. The new measurement is still recorded, so the
atlas says what the replacement is. `quality_checks:
false` in a job spec skips the redraw rounds; the checks themselves still run,
because they are a by-product of the cutout every cell needs anyway.

`tasks.redraw_frames_job` is registered in `model_gateway.GATED` under the
same label as `build_sheet_job` — it is the same transformer on the card, and
a GPU task missing from `GATED` bypasses the gateway silently.

---

## 5. What this does NOT fix

- **se and sw are still the front view** (1.6). Goal 1 of the brief is only
  partly met: `e`, `ne`, `n` and `s` are real views and the west rows now
  face west, but there is no true three-quarter view on either side. The
  checks flag the rows; a redraw cannot fix them, because the defect is in
  the turnaround view every attempt is conditioned on.
- **Effects attached to a hand survive the cutout.** Removing the effect
  nouns from the pose text removes the ones that were requested; a starburst
  the model invents at the end of an outstretched arm is kept by BiRefNet as
  part of the figure, and `identity` is what flags it.
- **An accepted cell can still be the wrong subject.** The checks measure
  framing, backdrop, facing and colour, not species — the same limit
  `concept.judge` has.
- **The strict sequential redraw** the brief describes ("adjacent marked
  frames redrawn in order, each using the other's NEW version") is not built;
  see `_context_images` for why it costs a text-encoder and a transformer
  load per frame here. Neighbours are read as they stand at the start of the
  pass.
- **Frame context is implemented but OFF** (`SHEET_FRAME_CONTEXT=1` enables
  it), because it does not fit this card. See below.

---

## 5a. What an extra conditioning image costs

Measured 2026-10-22 with `bench-frame-context.py`, one configuration per
process so the measurement is not of a fragmented allocator:

    stage     images  s/cell  peak reserved  min free (driver)
    encode      1      3.38      6180 MB        11191 MB
    encode      2      3.90      6332 MB        11191 MB
    denoise     1     36.69      9388 MB        11187 MB
    denoise     2     78.90      9988 MB        11167 MB

**The encode side is cheap** — +15% time and +152 MB for a second image. The
**denoise side is not**: +115% time and +600 MB peak on a single cell, and
`nvidia-smi` showed 10.7 GB of the 12.3 GB card in use during the run.

Against the adoption criteria set for this (peak leaving >1.5 GB headroom and
per-cell time rising <40%), the time cost alone fails by a factor of three.

It degrades further with the number of cells in a pass, which is the number
that matters: a 4-cell, 2-image denoise was killed after 22 minutes, i.e.
**>330 s/cell against 78.9 s/cell for a single cell** — consistent with
spilling into WSL's shared host memory once the card is full. A production
action pass is ~210 cells.

So frame-to-frame continuity is handled by the checks and redraws, not by
conditioning. `images=3` was not run: the verdict was already decided at two,
and a third pass at that rate would have cost an hour of GPU for a number
nothing depends on.

---

## 5b. The first full sheet on the new conveyor

Job `2ff8af36`, 2026-10-22: the same concept, seed, actions and directions as
`8d9d164d`, so the only variable is the conveyor. The turnaround views came
out **byte-identical** to the old job (|new-old| = 0.00 on every direction),
which is what makes this an A/B of the action stages alone.

    flagged, of the 210 rendered cells
      old sheet                70  (33.3%)
      new, before redraws      61  (29.0%)
      after round 1            38
      final                    27  (12.9%)

    by check   old -> final      by action   old -> final
      backdrop  20 -> 1            cast       25 -> 6
      close_up  23 -> 3            attack     13 -> 5
      border    19 -> 1            use        14 -> 7
      facing    26 -> 8            damage     11 -> 8
      identity  45 -> 22           sway        5 -> 0
                                   walk        2 -> 1
                                   idle        0 -> 0

Two changes did most of the work and did exactly what they were predicted to:
**backdrop 20 -> 1** is the segmenter, **cast 25 -> 6** is the pose rewrite.

Rounds: 61 cells / 41 improved, 38 / 17, then 42 / 14 as seven whole rows
(`attack ne,se`, `cast ne`, `damage ne,se`, `use ne,se`). Total 141 redraw
renders on top of 196 initial. 3 h 07 against roughly 2 h 50 for the old
sheet - the mirroring saving funded the redraws almost exactly.

Against the brief's acceptance criteria: figure height within +-15% of idle s
**passes** (0 of 336 outside, idle s 45 px of 64); cells over 40% opaque went
**54 -> 2** (one cell and its mirror, so a 96% reduction but not zero);
`se`/`sw` visibly different from `s` **fails, unchanged** (|s-se| 5.8 against
|s-e| 11.1, was 5.6/12.0).

**The residual is one defect, not many.** `ne` is 18 of the 27, and identity
is 22 of the failures across all checks. Within the `ne` row identity runs
0.005-0.009 at the clean end and up to 0.059 at the bad end, so it is real
drift in back-three-quarter poses rather than a mis-set threshold - three
rounds of fresh seeds barely moved it. That is the next thing worth
investigating, and a camera hint will not touch it.

## 5c. The front-quarter hint: a candidate, not a fix

`scripts/try-quarter-hints.py`, same subject and seed. Appending
`", halfway between front and profile"` moves `se` from 0.22 to 0.45 on the
front->profile path with framing preserved (1.02x), and the turn is visible
at full size - angled shoulders, face turned, talisman hand set back, where
the control is square-shouldered and straight-on.

Not adopted. It is one subject on one seed, and `sw` does not become a clean
mirror of `se` under it (|se-sw| 54.35 against |se-mirror(sw)| 55.90, a tie).

**A metric warning worth more than the result.** The first scoring here used
`|s-se|` alone, and it is actively misleading: the angles LoRA crops
differently per azimuth, so a wording that merely zooms scores as a
rotation. The wording that scored the biggest `|s-se|` of all - 78.72,
nearly five times the control - did it by framing the subject 1.62x larger.
That is a close-up, and this repo's own `close_up` check would reject it.
Read `|s-se|` beside the figure fraction and the position on the s->e path,
never alone.

## 6. Re-running any of this

    # re-score a finished job; --anti-goals is the regression gate
    docker exec sprite_worker python3 /app/scripts/score-sheet.py \
        212d56d7-c934-4a76-ae19-3fc87d438205 \
        8d9d164d-32ed-432a-aca1-c228f246d650 \
        --json /app/images/_qc/scores.json --anti-goals

    # assembly invariants, seconds, no GPU
    docker exec sprite_generator python3 /app/scripts/check-sheet-assembly.py

    # what an extra conditioning image costs
    docker exec sprite_worker python3 /app/scripts/bench-frame-context.py \
        --work /app/images/_job_<id> --stage encode --images 2

Scoring 672 cells takes about 25 minutes, almost all of it BiRefNet at ~1.8 s
per cell. Do not run it while a sheet job is on the card.
