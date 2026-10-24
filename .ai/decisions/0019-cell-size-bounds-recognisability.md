# 0019 - Sprite recognisability is bounded by the cell, not by the model

Date: 2026-10-24
Status: Proposed. Needs a something2 decision; nothing here changes the
generator.

**Result: at the production 48x64 cell, ~17% of a sprite's interior detail
survives the fit. 96x128 more than doubles it (0.1665 -> 0.3582), costs no
GPU time at all, and can be applied to sheets that already exist.**

## Why this exists

The owner's recurring complaint, across months and several models, is some
form of *"nothing similar with template image"* / *"it does not look at the
origin template"*. Every attempt to answer it reached for the generator - a
better quant, different prompt wording, a different adapter scale - and the
complaint survived all of them.

It survived because the generator was never the cause, and because the one
check that should have caught it structurally cannot. `sheet_qc.identity` is
a **colour histogram** and is spatially blind: it scored 48x64 and 96x128 as
equally far from the concept while the 96px version is obviously more
recognisable (sheet-quality contract, section 7). So there was no number for
"the shape became mush", and the question stayed a matter of eyesight.

## Evidence

### The loss is interior, not silhouette

`scripts/cell-size-cost.py`, all 210 cells of job `2ff8af36`, CPU only - it
reads the alpha already baked into `cut_*.png`, so it runs safely while a
build holds the card.

| cell | silhouette IoU | interior detail vs 512 | sprite px |
|---|---|---|---|
| **48x64** | 0.6121 | **0.1665** | 1499 |
| 64x86 | 0.6035 | 0.2346 | 2745 |
| 96x128 | 0.6025 | 0.3582 | 6216 |
| 128x170 | 0.6028 | 0.4566 | 10960 |

**Silhouette IoU is flat across every size.** That was not the expected
result - the script was built looking for the outline to be the casualty, and
found it intact. Resolution buys nothing for the outline. What a bigger cell
buys is the interior: the face, the hood edge, the belt, the gem.

A `shape` metric (perimeter^2 / area) was measured alongside and **dropped**:
it rose with cell size, but a NEAREST upscale makes the boundary staircase
finer as the cell grows, so it was reading quantisation rather than detail.
Do not reinstate it.

### Confirmed end to end, not only in the metric

`compose` reads `cut_*.png`, so the same sheet was rebuilt at a larger cell
**without re-rendering a single frame**:

    48x64    sheet 288x3584    scale 0.089844    0.5 MB
    96x128   sheet 576x7168    scale 0.183594    1.6 MB

The 48x64 run reproduced the baseline's pinned `sheet_scale` exactly
(0.089844), so the comparison is faithful. 2.04x linear scale, matching the
+315% pixel area the table predicts. Side by side in
`images/_cellsize/compare.png`: at 48x64 the staff is a brown smear and there
is no face; at 96x128 both read.

### Palette was ruled out twice, from two directions

Independently of this, [0016](0016-rig-gate-retest-and-off-card-3d.md)'s
pixel-art concept test concluded *"palette was never the defect"* for the 3D
lane, on the grounds that both pipelines end at exactly 24 opaque colours.
The 2D measurement agrees from the other side: palette size made no
difference to identity at 48x64 across 24, 48 and 96 colours. Two pipelines,
two methods, same answer - **do not spend a third run on colour count.**

That ADR's other finding, flat lighting from a rasteriser that adds none,
is specific to rendering views off a TripoSR mesh. It does not transfer
here: the editor returns already-lit images.

### No quant touches it

Q3_K_S tripled closeness to the concept at 512 - masked colour distance
0.0336 -> 0.0108 - and at 48x64 that became 0.1322 -> 0.1196, which is no
visible change. Q6_K improves framing and borders substantially
(`close_up` 14 -> 1, `border` 9 -> 0 on a 48-cell subset) and is worth having
for that. Neither reaches this.

## Decision

**Treat the cell size as the primary lever on recognisability, and as a
contract question rather than a tuning knob.**

1. **Raise 96x128 with something2.** Their atlas consumer and in-game sprite
   scale both change with it, so this is not a decision this repo can take
   alone. The generator is unaffected either way.
2. **Do not spend GPU chasing this through the model.** Prompt wording,
   adapter scale and quant have all been measured against it and none moves
   it. Spend GPU on the defects a redraw or a prompt genuinely can fix.
3. **A cell-size change is re-composable, not a rebuild.** Any sheet whose
   `images/_job_<id>/` still holds its `cut_*.png` can be recomposed at a new
   cell with no rendering. Strip `sheet_scale` from `config.json` first, or
   the sheet is drawn at the old pinned scale inside bigger cells and the
   change appears to buy nothing.
4. **The per-frame checks are unaffected.** `sheet_qc` scores the RAW 512
   render before the fit, which is exactly why it cannot see this - and also
   why its thresholds need no revisiting if the cell changes.

## What this costs

- **4x the sheet pixels**: 288x3584 -> 576x7168, 0.5 MB -> 1.6 MB per sheet.
  Bandwidth and texture memory on something2's side, not here.
- **A consumer change.** Every atlas coordinate scales; anything hard-coding
  48x64 breaks.
- **Job-directory retention becomes load-bearing.** Point 3 only holds while
  the work dirs survive, which makes `make prune-job-cache` and any future
  cleanup sweep a decision about whether old sheets can be re-packaged later.
  `prune-job-cache` keeps `cut_*.png` and drops `embeds_*.pt`, so it is safe
  for this; a blunter sweep would not be.

## What this does NOT claim

Neither `iou` nor the interior-detail figure says a player recognises a
sprite. They say how much information survived the fit. Read them beside the
rendered cells - the same rule the quarter-view hints carry, where the
wording that scored the largest raw pixel difference turned out to be a zoom.

96x128 was measured because it was the obvious next step, not because it is
optimal. 64x86 recovers +41% of interior detail for 1.8x the pixels and was
not evaluated visually; if something2 finds 4x too expensive, that is the
candidate to look at rather than abandoning the change.
