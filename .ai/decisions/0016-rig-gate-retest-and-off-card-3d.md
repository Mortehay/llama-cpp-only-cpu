# 0016 - The rig gate, re-run with T-pose feedstock: still closed. And the 3D lane runs off-card.

Date: 2026-10-23
Status: **Proposed.** Confirms [0005](0005-back-to-2d-modern-editors.md)'s reversal of
[0004](0004-pivot-to-3d-conveyor.md) by closing 0004's own "Option 2", which
0004 recorded as untested and cheap.

> **Read this before reviving `mesh_worker`/`rig_worker` for a sprite
> conveyor.** The short version: the containers build, the mesh stage works,
> UniRig works, all three run without touching the GPU - and the rigging gate
> still fails on generated reconstructions, 0 of 2, with a synthetic control
> passing cleanly in the same run. Separately, and independently fatal for the
> "3D renders the sprite" plan: a pixelated render of a TripoSR mesh is not
> pixel art.

## Why this exists

0004 pivoted to `TripoSR -> UniRig -> Blender` and died at the rig: **4 of 5
reconstructions could not be rigged**, with no measured input property
predicting which succeeded. It listed three options in evidence order, and
option 2 was:

> **Feed reconstruction a T-pose.** Every source core here is in a rest or
> action pose with limbs near the torso. The control that worked had limbs
> held clear. Untested and NOT the same claim as the rejected "arms at sides"
> hypothesis - that compared arm position within action poses, not a true
> T-pose generated for reconstruction.

Two things made it worth testing now rather than then. First, the editor lane
*spontaneously* produces T-poses - 0005 logged that as a defect ("The T-pose:
the same trap, twice"), which means a deliberate one is well inside what the
stack can render. Second, the host was upgraded: Ryzen 5 5500 (6c/12t),
47.8 GB RAM, so the "recompiling on a 4-thread i3-8100" cost in 0004's revival
warning no longer applies.

The owner also asked a second question - whether the 3D route could make the
*sprites*, not just the geometry - so the taste gate was pulled forward and
answered in the same session. It turned out to be the cheaper of the two
kill-shots.

## Decision

**Do not revive the mesh -> rig conveyor.** 0004's reversal stands. Of 0004's
three options, option 2 is now closed by measurement rather than assumption;
option 1 (replace image-to-3D) narrows to AniGen, untested; option 3
(pre-rigged source models) is unaffected and remains the honest fallback.

**Two directions survive and neither needs generated rigging:**

- **3D as a control signal.** Pose a *stock* rig and use the render only as
  camera/pose conditioning for the 2D editor that already produces shippable
  art. This is what the evidence below actually argues for.
- **Pre-rigged library as a render farm** - 0004's own option 3.

## Gate result: 0 of 2 characters, 1 of 1 control

Run on CPU (see "Off-card" below), `identify_limbs.py` labelling from geometry
alone.

| subject | source | bones | leaves | verdict |
|---|---|---|---|---|
| **control** | `make_test_humanoid.py`, synthetic boxes | 22 | 5 | **clean, no warnings.** head lateral 0.004, arms ±0.918/0.926, legs −0.996/−0.996 |
| zombie T-pose | TripoSR, photoreal concept | 28 | 5 | FAILED - "the two legs are not level: 27% of the figure's height apart" |
| knight T-pose | TripoSR, photoreal concept | 28 | 5 | FAILED - "the two arms are not level: 29% of the figure's height apart" |

**The control is what makes this conclusive.** It rigged perfectly in 0004 on
the GPU and rigs perfectly here on the CPU, in the same run as the two
failures. So neither the rigger nor the CPU path can be blamed; the only
remaining variable is the reconstruction. That is 0004's sentence verbatim -
*"UniRig is not the problem. The reconstructions are."* - now re-measured with
the feedstock 0004 said to try.

### The failures share one signature: the figure collapses on one side

Both produce two leaves at extreme lateral **on the same side**, while the
opposite side never reaches past −0.74:

| | leaf A | leaf B | opposite side's furthest |
|---|---|---|---|
| zombie | +1.113 | +1.082 | −0.777 |
| knight | +1.113 | +1.121 | −0.738 |

For a T-pose, arms at ±max lateral is the one thing that should be
unmissable - the arms are unambiguously clear of the torso in the source (see
the inputs under `images/_rig_gate/`). The consistency across two unrelated
characters suggests something systematic rather than noise, and it is
**unexplained**. Anyone retrying should start here rather than re-running the
same test.

### What was NOT the variable

- **Mesh quality.** All three reconstructions passed `inspect_mesh` as
  "Structurally rig-ready": single component, watertight after
  `repair_for_rigging`, winding consistent, one degenerate face each.

  | mesh | faces | silhouette IoU | aspect vs source |
  |---|---|---|---|
  | zombie T-pose | 70,194 | 0.844 | 1.07 / 1.09 |
  | zombie rest *(control for pose)* | 49,156 | 0.859 | 2.33 / 2.40 |
  | knight T-pose | 69,004 | 0.833 | 1.10 / 1.08 |

  Those IoUs sit inside the band 0004's *successes* occupied (0.844-0.910). So
  the mesh stage reproduces on photoreal T-poses what it did on pixel-art
  cores, and mesh quality remains anti-correlated with skeleton quality.
- **The CPU path.** The control passed on it.
- **Keyed-source scoring.** `inspect_mesh` was given the keyed PNGs, not the
  `*_input.png` grey composites - the trap 0004 fixed, which made two good
  meshes read as 0.383 and 0.412.

### One suggestive, non-load-bearing observation

Same character, same seed, same reconstructor: the T-pose yields **70,194
faces against the rest pose's 49,156**, 43% more geometry - consistent with
arms resolving as separate volumes rather than merging into the torso, which
is the mechanism the T-pose hypothesis rested on. It did not translate into a
riggable skeleton. Face count is not limb separation; do not read this as
support for retrying.

## The taste gate: a pixelated TripoSR render is not pixel art

Independently fatal for "3D renders the sprite", and cheaper to answer than the
rig gate - so **answer it first** next time. Both strips below went through the
project's own `SheetBuilder`: shared palette, shared min-scale fit, common
baseline, 24 colours.

| | 3D render (`strip8_zombie_3d_x5.png`) | 2D production (`strip5_real_2d_x5.png`, job `d3083b62`) |
|---|---|---|
| source | TripoSR vertex colours, flat-shaded | Qwen-Image-Edit + pixel-art LoRA, BiRefNet `cut_*.png` |
| cells | 8 directions × 1 frame | 5 rendered directions × 1 frame |
| sheet scale | 0.205357 | 0.121331 |
| transparent | 80.6% | 55.9% |
| partial alpha | 0 px | 0 px |
| opaque colours | 24 | 24 |
| **reads as** | **a dark blob** | **shippable pixel art** |

Every *numeric* acceptance check passes on both. The 3D cells hit 0005's
`_sheet8.png` criteria exactly - 0 partial alpha, 24 colours - and are still
unusable: near-monochrome dark green, no outlines, no readable features, no
face, arms reduced to 1-2 px stubs. The 2D cells show a readable face, staff,
armour, cloth and boots at the same 48×64.

**This is the clearest available demonstration that the numeric gates cannot
see style.** It is the same blind spot `.ai/specs/sheet-quality/` §7 records
for `identity` (a colour histogram, spatially blind) generalised to the whole
check suite - see "Consequences" below.

Three levers were **not** tried and would have to be, before anyone revisits
this:

1. ~~**The concept was photoreal.** Generated with no pixel-art trigger, so
   photoreal colour baked into the vertices. A pixel-art concept bakes
   pixel-art colour. This is the cheap test and it was not run.~~
   **RUN 2026-10-24, and the premise was false - see "The pixel-art concept
   test" below.** A pixel-art concept bakes MORE distinct colours, not fewer,
   and palette was never the defect.
2. **No toon shader, no outline pass.** The renderer used here is a ~130-line
   z-buffered flat-colour rasteriser written for this spike (no GL, no
   lighting). The production renderer is meant to be `bpy`, where an outline
   pass is standard - and outlines are most of what makes a sprite read at
   48px.
3. **No pixel-art post-processing beyond palette lock** - no contrast boost,
   no hand-tuned ramp.

The gap is "unusable vs shippable", not a 10% delta, so the burden sits with
whoever wants to close it.

### What the 2D strip also showed

The documented front-quarter collapse is **visible by eye**: cells `s` and
`se` are near-duplicates, which is the `|s−se|` 12.8 measurement in CLAUDE.md
rendered at full size. `e`, `n` and `ne` are all correct. So the 2D lane's
direction defect is narrow and specific - and it is exactly what a camera move
fixes for free. The 3D profiles were *genuine* profiles, with the near arm
correctly foreshortened to a stub; nothing collapsed toward the front. That
property is real and is the whole argument for the surviving directions.

## The pixel-art concept test (2026-10-24): palette was never the defect

Lever 1 above, run. It changes the diagnosis rather than the verdict, and it
contradicts what the taste-gate section implies, so read both.

**Method, and a methodological correction.** The gate was re-run on
`concept_clean.png` from job `d3083b62` - the production pixel-art concept
behind the 2D comparison strip. Three advantages over generating a fresh
T-pose, and the third is the correction:

- it is genuinely pixel art, already keyed and ground-stripped, which is
  exactly TripoSR's required input;
- it is **the same character as the 2D strip**, so the comparison becomes one
  character through two pipelines rather than two characters through two;
- **it needs no GPU at all.** The T-pose was only ever required by the RIG
  gate. The taste gate asks whether a pixelated render reads as the character,
  and pose is irrelevant to that. Conflating the two cost a day of waiting on
  a card for no reason - the other session's Q6_K sheet held it for hours.
  **Run the taste gate on an existing core; never queue a generation for it.**

**Result: the premise was wrong.** The pixel-art concept baked **16,561
distinct RGB** into the vertices, against the photoreal zombie's 11,717 -
*more*, not fewer. TripoSR interpolates colour across the surface whatever the
input's palette was, so no concept choice reduces it. And the final cells were
palette-locked to 24 colours either way, so **palette count was never the
defect** - the taste-gate section's framing of "muddy" as a colour problem is
superseded here.

### The defect separates into three causes, only one of which is fatal

| cause | verdict |
|---|---|
| palette too large | **not a cause.** Both pipelines end at exactly 24 opaque colours |
| no lighting | **real, and partly fixable.** Measured luminance range across all 8 views: **1-144 of 255** - the renders used 56% of the range, because TripoSR bakes already-shaded colour and the rasteriser adds no light. A shared stretch over the figure recovered the lit views |
| the unseen side is invented | ~~**dominant, and not fixable at render time.** Missing information, not missing contrast~~ **FIXED 2026-10-24 - see "Colour projection" below. The claim was half right: the invented COLOUR was the defect, the invented geometry was not, and projecting the observed views supplies the colour.** |

The lift must use **one shared mapping across all eight views**, computed over
`alpha > 0` only. Per-view normalisation would make the character change
brightness as it turns - the same class of defect the shared palette and the
pinned min-scale exist to prevent.

## Colour projection: the actual fix (2026-10-24, later the same day)

**Projecting the OBSERVED turnaround views onto the mesh turns both
reconstructors from unusable to usable at 48x64.** It needs no new model, no
GPU and no download, and it is the single change that closed most of the gap
to the 2D lane.

The defect was never the palette and never, mostly, the geometry. It was that
vertex colour for surfaces the reconstructor could not see is EXTRAPOLATED.
The 2D lane already emits a turnaround, so those surfaces are in fact
observed - just not by the reconstructor.

| | before (baked vertex colour) | after (projected from 3 views) |
|---|---|---|
| TripoSR | muddy, unusable at 48px | **legible and clean, no artefacts** |
| Hunyuan3D-2mv | n/a - shape-only model | legible, stray slab in 3 of 8 cells |

Coverage: TripoSR 52,512 of 56,804 verts sampled from views (**92%**),
Hunyuan 197,504 of 223,636 (**88%**). The denser mesh scores WORSE because
more folds and recesses are unreachable by three cameras - more geometry means
more unobserved geometry. Unsampled vertices take the mesh median, never white,
which would read as a hole. Both sheets: 24 opaque colours, 0 partial-alpha.

### Two corrections to claims made EARLIER IN THIS DOCUMENT and in the session

1. **"TripoSR's back was mush" is wrong.** Rendered through an identical
   normal-shaded path, its back carries hood, shoulder cape, robe folds, the
   staff and the talisman. The back GEOMETRY was always adequate; the back
   COLOUR was not. The cause table above is corrected accordingly.
2. **"Multiview solved the dominant defect" is wrong.** Projection colouring
   did. Both claims attributed an improvement to a newly added component
   instead of isolating the variable; the controlled comparison is what caught
   it, and it was only possible because projection is reconstructor-agnostic.

### Hunyuan3D-2mv: it runs, and it did not clearly earn its cost

| | TripoSR | Hunyuan3D-2mv |
|---|---|---|
| weights | 1.68 GB, MIT | 5.36 GB, Tencent licence with EU/UK/KR exclusion |
| runtime | ~45 s, CPU, no card | 67 s load + 105 s reconstruct, on the 12 GB card |
| output | 113,652 faces, watertight after repair | 447,420 faces, **raw and not watertight** |
| coloured result | clean | stray slab visible in 3 of 8 cells |

The strays are the direct cause of its worse showing: Hunyuan's output has had
no `repair_for_rigging` pass, and that function exists precisely to drop
sub-threshold components. A fair rematch has to clean it first. But the margin
to beat is small, because once both are coloured the same way the denser
geometry is not visibly better at 48x64 - which is 0004's original argument
for choosing the smallest model that gets the silhouette right, arriving again
from the other direction.

Setup, for whoever repeats it: `hy3dgen` in a venv inside mesh_worker
(`--system-site-packages`, so torch is shared and TripoSR keeps its pinned
transformers 4.35 in system site-packages). **Pin transformers to the
checkpoint's era** - unpinned installs a version whose DINOv2 attention
refactor renames `query/key/value` to `q_proj/k_proj/v_proj`, and every
projection weight in a 2025 checkpoint then reads as missing. 4.49.0 works.
This stack now carries three transformers versions on purpose: 4.35.0 system,
4.49.0 in the hy3d venv, 4.51.3 in rig_worker. The venv lives in the
CONTAINER, not the image, so it dies on `--force-recreate`.

### Derive the azimuth; do not assume it. This finally works.

The fix proposed earlier in this document is implemented and measured: sweep
candidate azimuths, rasterise a silhouette at each, and assign every observed
view the angle whose silhouette best matches it by IoU. The source views ARE
the ground truth for where the cameras were.

| mesh | up axis | front | left | back | front/back |
|---|---|---|---|---|---|
| Hunyuan3D-2mv | **y** | 0 deg (IoU 0.831) | 270 (0.821) | 180 (0.806) | 180 OK |
| TripoSR | **z** | 270 deg (IoU 0.810) | 20 (0.730) | 100 (0.800) | 170 OK |

**The two meshes disagree on BOTH the up axis and the azimuth origin.** That
is why every hardcoded offset failed: there is no single convention to
hardcode. Four orientation bugs were hit in one session before this went in -
a `+90` calibrated on one mesh, a 180 on another, Z-up assumed for a Y-up
mesh, and inverted winding after a mirror - and each looked like "the model
produced something bad" rather than "the camera was pointed wrong". Detect up
from the tallest extent (what `inspect_mesh` does), derive azimuth by IoU, and
shade two-sided so a flipped winding cannot read as a black mesh.

Known weakness: silhouette IoU is front/back ambiguous for a robed figure. The
separation check catches a gross error but not a clean swap - the giveaway is
a face on the back of the head, so LOOK rather than trust the number.

### What the first quadruped exposed, and most of it is the 2D lane

A wolf was the first non-humanoid through this lane (2026-10-25). Everything it
hit is worth more than the sprite was.

### `concept.judge` refuses anything wider than tall

A clean, isolated, border-clear wolf concept was refused by `POST /api/jobs`:

> `concept does not look like an isolated character: not taller than wide
> (aspect 0.91) (coverage 39%, border 0%, aspect 0.91)`

So **a quadruped cannot enter the 2D conveyor at all** without
`concept_check=false`. The message names the bypass but not the reason, and
the reason is that the gate encodes "a character is taller than wide". That is
true of the humanoids the conveyor was built on and false of most animals.

It also explains something that read as a data accident: scanning twelve jobs
for sources that pass the 3D lane's input gate returned **twelve humanoids**.
Not coincidence - creatures are refused upstream. The prop route
(`txt2img` + cutout) exists because props cannot go through `POST /api/jobs`;
a wolf falls in the same gap, too animal for the character gate and too
animate for the prop route.

### A quadruped's turnaround collapses to two viewpoints

Measured by matching each view's silhouette against the mesh: the wolf's five
colour views resolved to **two distinct angles** - front and left both at 260
degrees, back and q1 both at 80, q2 at 230 grouping with 260. Eight directions
requested, two delivered.

Colour coverage was still 94%, which is why nobody would notice: two opposite
profiles cover most of a laterally symmetric animal. The OUTCOME was fine and
the INPUT was impoverished. This is adjacent to the documented front-quarter
collapse (`|s-se|` 12.8) but worse, and nothing measured it before.

### The vignette was the concept's, not the editor's

Job `f841ec70`'s turnaround renders the wolf inside a cream disc with a dark
rim - the "wolf medal". A fresh concept generated with `plain white
background` and `cutout: true` produced a **clean turnaround with no
vignette**, so this is not systemic editor behaviour. Worth knowing because
the opposite conclusion would have made every turnaround suspect.

The input gate still earns its place: that concept existed, shipped a
medallion to every consumer of its views, and BiRefNet could not rescue it
(cleared the disc on 3 of 5 views, kept it on 2, making the spread worse).

### A quadruped needs a raised camera and a landscape cell

Both measured, both now flags rather than assumptions:

| | wolf | mage |
|---|---|---|
| cell utilisation at 48x64 | 64% (96% of width, 67% of height) | 89% |
| at 64x48 | 73% | n/a |

And at a dead-level camera a quadruped seen along its own length shows head,
chest and two front legs stacked vertically - it reads as a figure standing
upright. The mesh is correct; the camera is wrong for the subject. Every
reference sheet this project was built against uses a raised three-quarter
camera.

### The plinth is Hunyuan3D-2mv's, not the quadruped's

The ground slab was first seen under the wolf and looked like something about
animals. It is not. Validated 2026-10-25 across both body plans and both
reconstructors:

| subject | body plan | reconstructor | plinth |
|---|---|---|---|
| wolf | quadruped | Hunyuan3D-2mv | **yes** - bottom 1.8% of height, 113,102 of 477,840 faces |
| mage | biped | Hunyuan3D-2mv | **yes** - bottom 1.8% of height, 60,607 of 460,790 faces |
| mage | biped | TripoSR | no |
| archer, wide stance + longbow | biped | TripoSR | no - **feet intact** |

So **the multiview model reconstructs the ground its subject stands on**, at
the same 1.8% of height on both, and TripoSR does not. `--cut-pedestal`
compensates for a specific model's behaviour rather than for a subject type.

The fourth row is the one that matters most: a wide-stance biped carrying a
longbow is exactly the false positive `strip_ground_patch`'s old rule produced
("a wide stance looked like ground") and the reason `cut_below.py` exists. It
is untouched. A gate that occasionally amputates feet would be worse than no
gate at all, because the sheet still passes every numeric check afterwards.

`--pedestal-ratio 1.4` therefore rests on two positives and two negatives
rather than the single measurement it was first set from. The measured ratio on
the wolf was 1.51 (footprint 2.489 against 1.65 for the body above it), and 1.3
and 1.4 behave identically, so there is margin on both sides. Still only two
reconstructors and two body plans - a bird, or anything with a wide low mass
that is genuinely part of the subject, remains untested.

Cutting the slab also raised colour coverage 89.1% -> 95.9% on the wolf,
because most of the median-filled vertices WERE the slab. The grey mound and
the plinth were one defect, not two, which is why changing the median-fill
colour would have been the wrong fix.

## The rule this lane keeps re-learning: do not measure against the contaminated region

Four separate times in one session, a check was written to compare a thing
against something the defect had already spoiled, and each time it silently
measured the wrong quantity:

| check | wrong reference | correct reference |
|---|---|---|
| mesh fitness (0004) | the grey-composited `*_input.png`, 100% opaque | the keyed source |
| cell fit | the FITTED sprite's aspect - 46x43, square only because 46 was all the width there was | width-fill against height-fill |
| plinth detection | the bottom band's own median - if the plinth fills the band, the median IS the plinth | the shins, the region just above |
| plinth axis | `argmax(ptp)`, which for a quadruped is nose-to-tail | the derived up axis |

`strip_ground_patch` learned it first and `sheet_qc` exists in its current form
because of it - "the fitted cell cannot answer the question". It is the single
most repeated mistake in this project's history. **Before writing a check, ask
what the defect being hunted would do to the reference.**

A related one, same session: the azimuths were searched as three free
parameters for a problem with ONE. The views come from a single turnaround, so
their relative angles are fixed and only the global offset and handedness are
unknown. Free search let them land in arrangements that cannot physically
exist. **Count the free parameters before searching.**

## Where this leaves the lane

Against the 2D production cells the gap is now a difference of degree, not of
kind: the 2D strip is still crisper with more contrast and a more readable
face, but these are recognisable coloured sprites with **eight structurally
correct directions**, which the 2D lane cannot produce (`se`/`sw` collapse,
west rows mirrored with handedness swapped).

It still does not unblock sprites. Rigging is closed, so there are no action
frames, and the deliverable remains **static 8-direction turnarounds**.

Cheapest next steps, in order:

1. `repair_for_rigging` on the Hunyuan mesh, for a fair rematch.
2. Add `dir_se`/`dir_ne` as extra COLOUR sources. They are rendered and
   genuinely independent, and colour projection does not care about the
   front-quarter collapse that would poison a reconstructor.
3. `bpy` toon render with a key light and an outline pass, for the remaining
   contrast gap.

### Quality degrades with angular distance from the input view

With the front identified correctly (see the azimuth note below), the eight
cells read: front and both front-quarters **usable** - readable face, pale
hood, red sash, staff; profiles middling; back and back-quarters dark and
mushy. That is exactly what single-image reconstruction predicts, and it is
why the invented-back cause dominates.

**So the gap is much smaller than the zombie implied, and the dominant
remaining cause has a fix that already exists in this repo**: the 2D lane
produces eight consistent `dir_*.png` turnaround views for free, which is
precisely the multi-view input that removes invented geometry. Model choice is
the blocker - no confirmed open multi-view-to-mesh was found that fits 12 GB -
so that is a research pass, not a download.

**It does not unblock sprites.** Action frames need posing, posing needs a
rig, and the rig gate is closed. The end-to-end deliverable is still **static
8-direction objects**, which need no rig and which the `txt2img`+cutout prop
route cannot produce.

### The azimuth mapping must be DERIVED, not offset - this is the third miss

`preview_render.py` applied `+90` after the zombie's coverage numbers came out
the wrong way round. On this mesh that offset was **180 degrees wrong**:
`dir_n` held the face, the staff in the figure's right hand and the talisman
on its left, while `dir_s` was the hooded back. TripoSR's output orientation
depends on its input, so **a hardcoded offset is wrong by construction.**

Counting 0005's own "Azimuth mapping: checked, broken, fixed", the zombie
correction, and this, the project has now got this wrong three times. The
principled fix is cheap and already half-written: the concept image **is** the
front view by definition, and `inspect_mesh.silhouette_iou` already scores a
mesh projection against a source image. Score all eight, call the best match
`s`, derive the rest.

### Incidentals

- **The thin staff survived**, which was the stated risk for this subject. Of
  113,652 faces, the 8 dropped stray components (292 faces) were the
  talisman's chain, not the staff - it renders clearly, gem included, in six
  of eight views.
- Warm CPU cost on a 113k-face mesh: load 91.1 s, infer 17.4 s, marching
  cubes 22.8 s.

## Off-card: the finding worth keeping regardless

**Every stage measured here runs with zero VRAM.** On a box where the 12 GB
card is the binding constraint and where a second session's sheet build took
the card mid-run (see "Collisions"), this matters more than the seconds lost.

| stage | GPU (0004, RTX 3060) | CPU (here) | ratio |
|---|---|---|---|
| TripoSR reconstruct | ~24 s end to end | **~45 s warm** (load 7.6 s, infer 16.2 s, mcubes 20.5 s) | ~2x |
| UniRig skeleton | ~13 s | **5 min 08 s** | ~23x |

Cold TripoSR load was 198.3 s, including the 1.68 GB checkpoint download.

### How UniRig runs on CPU

Undocumented upstream - the README states a CUDA GPU with ≥8 GB and the
inference YAML ships `accelerator: gpu`. It works here because **this repo's
image has already removed the real blockers**:

- `flash_attn`: `patch_unirig.py` rewrites the AR config to `sdpa`. (The same
  substitution upstream PR #70 makes for its own CPU fallback.)
- `spconv`: the eager PTv3 import is made optional, and the skeleton task
  selects `michelangelo_encoder`, so PTv3 is never constructed.
- `run.py` contains **no hardcoded `cuda`**. Every `.cuda()` in the tree is
  under `src/model/pointcept/engines/` - PTv3 training/eval code that the
  skeleton path never reaches.

So the change is two lines in a copy of the task config:

```yaml
accelerator: cpu      # was gpu
precision: 32         # was bf16-mixed - which has NO CPU path
```

Saved here as `configs/task/qi_skeleton_ar_256_cpu.yaml` **inside the
container**, which is image state and therefore ephemeral. The log confirms it
took: `GPU available: True (cuda), used: False`.

**Only the skeleton half is needed to answer the gate.** `identify_limbs.py`
reads an armature; skinning weights say nothing about whether limb ends are
labellable. So the 4.4 GB skin model and the 4.9 GB `data/rigxl/processed.7z`
training archive stay unused - as 0004 already warned.

## Corrections to other documents, made while getting here

- **`.ai/project-context.md` is stale on `.wslconfig`.** It says the file
  "still says `processors=4` and `memory=24GB`, with comments describing the
  i3". Actual, read 2026-10-23: `memory=42GB`, `processors=10`, `swap=16GB`,
  updated 2026-09-28 and 2026-10-01. The VM is *not* using a third of the CPU.
- **`mesh.py`'s colour comment was half wrong.** It read "colour is irrelevant
  at sprite scale and the rigging step that follows cares only about
  geometry." The second clause is right; the first is the opposite of 0005's
  measurement, that at 48px a hue shift is what reads as a different character
  - which is why palette lock exists. Corrected in place, and `--bake-texture`
  is now exposed on the CLI (the `bake_texture` parameter existed and `main()`
  never offered it, so a coloured mesh could not be requested at all).
  Vertex colour survives `repair_for_rigging` and a `.glb` write intact:
  `ColorVisuals`, one colour per vertex, ~12k distinct RGB. Export `.ply` or
  `.glb`, not `.obj` - trimesh wrote 3 values per `v` line.
- **0004's revival warning is stale.** Both images build clean on current
  hardware: `mesh-worker` 22.3 GB in 484 s, `rig-worker` 25.2 GB in 688 s,
  `torch_scatter`/`torch_cluster` compiled from source without incident.
  `MAX_JOBS` raised 2 -> 6 in `rig_worker/Dockerfile.cuda`; its justification
  named "a 4-thread i3-8100" and "WSL's 11GB" and both had expired.
- **`UNIRIG_REF` is pinned** at `6793c6640ff01c8fb389f3993434124bb43d2933`
  (2026-06-04, "Update README.md"). `patch_unirig.py` passes against it, which
  is evidence the tree is the shape August's build was patched against. Note
  `--depth 1 --branch` accepts only branch and tag names, never a bare commit,
  so the clone form had to change with the pin.
- **UniRig fetches its own checkpoint.** It restored from
  `/models/hub/models--VAST-AI--UniRig/snapshots/.../model.ckpt` under
  `HF_HOME=/models`. A manual `hf_hub_download(local_dir=/models/unirig)` plus
  a symlink into `/opt/UniRig/experiments/` was **wasted work and a duplicate
  1.44 GB on disk**. Do not repeat it.

## Operational notes learned the hard way

### Collisions: check `docker top` before EACH GPU stage, not once

A second session started `build-sheet.py turnaround-denoise --work
/app/images/_qab_q6k_full` at 15:05, after the preflight check came back clean
at 13:06. The result, mid-run: one `500` reading *"5.9 GiB free of 12.0, needs
9.8 GiB"* and two `504`s at the 285 s facade ceiling - 3 of 6 planned
generations lost. The tripwire in CLAUDE.md is right; checking it once at the
start is not enough.

### A facade 504 does NOT cancel the generation

The timed-out task stayed `running: true` in `/api/model-gateway` for ~12
minutes while `idle_for_s` climbed to 251, and - because the worker is
`--pool=solo` - it blocked `make gpu-health`, which reported *"TIMEOUT after
60s... Inconclusive: solo pool, so a running job blocks the probe."* **It
cleared on its own.** Every remedy considered (revoke `--terminate` on a solo
pool, restarting `sprite_worker`) would have killed the other session's
`build-sheet.py` subprocess, which lives in that container. Doing nothing was
correct.

This is the third instance of one shape, after `remeasure-all` outrunning curl
and a hand-rolled download outliving its client: **a client giving up does not
stop the work.**

Also note `DELETE /api/task/{id}` takes the integer row id, and facade
generations do not appear in `/api/tasks/recent`, so there is no obvious route
to cancel one.

### `vmIdleTimeout` does nothing on Windows 10

`~/.wslconfig` sets `vmIdleTimeout=-1` and `wsl-keepalive.ps1`'s own docs say
the setting is Windows 11 only and silently ignored here. The keepalive
process is the only protection. Also: **`Get-ScheduledTask` is the wrong probe**
for "is the keepalive running" - it only detects `-Install` registration, and
reading its silence as "unprotected" was wrong; the script's own idempotency
check reported one already running.

## Consequences

- **A new 3D lane would inherit a check suite that cannot judge it.**
  `sheet_qc` was calibrated against *diffusion* failure modes - contact
  sheets, close-ups, painted checkerboards, invented props. A 3D render
  produces none of them, so the checks pass trivially while the cells are
  unusable, exactly as measured above. That is the failure
  `scripts/score-sheet.py --anti-goals` exists to prevent, and a 3D lane would
  need its own anti-goals (interpenetration, clipped limbs, wrong facing, dead
  rows) before any of its green lights meant anything.
- **The `cut_*.png` seam held up.** `stage_compose` is pure PIL/numpy over
  `cut_{action}_{direction}_{frame}.png` plus `config.json`, and the 3D cells
  went through it untouched - palette lock, pinned min-scale fit, baseline,
  atlas, something2 packing, all free. Whatever renderer a future lane uses,
  this is where it attaches, and it is not where the risk lives.
- **Two `models.toml` stanzas are owed** if this is ever revived:
  `stabilityai/TripoSR` and `VAST-AI/UniRig`. `make test-models-manifest`
  does not currently demand them only because its scanner defaults to
  `src/sprite_generator` and `scripts` - it never walks `src/mesh_worker` or
  `src/rig_worker`. A driver script placed in `scripts/` would fail that test
  immediately.
- **The spike's tooling is NOT in the repo.** The preview rasteriser and the
  turnaround-sheet assembler were written to a session scratchpad and are
  gone. They are small, and this document describes what they did; rewriting
  them is cheaper than hunting for them.

## Reproducing or extending this

Artefacts are under `images/_rig_gate/`: the T-pose and rest concepts, the
`.obj` meshes, the `_col.glb` coloured meshes, the `_skel.fbx` skeletons (all
three, control included), the 8-azimuth renders in `render_zombie/`, and both
comparison strips. 21 MB in total, and **verified present on 2026-10-24**,
after the reclaim below.

**The two container images are NOT.** `mesh-worker` (22.3 GB) and
`rig-worker` (25.2 GB) were the "two 20-GB-class experiment images" in
[0018](0018-disk-retention-is-a-policy.md) and were deleted on 2026-10-24 -
this spike is most of why C: went from 108.7 GB free to 16.3 GB in a day. So
any re-run starts with **two builds, 484 s and 688 s**, and the ephemeral
pieces went with the containers exactly as warned above: the
`qi_skeleton_ar_256_cpu.yaml` task config and the weights symlink both lived
in image state and no longer exist. The checkpoints survive in `/models`,
and UniRig re-fetches its own regardless.

Budget ~50 GB of disk before starting, and read 0018's ordering rule first -
delete dead images BEFORE `builder prune`, or the prune nets out layers still
shared with them.

If you extend it, the order that wastes least GPU is:

1. ~~**Pixel-art concept through the render path**~~ - **done 2026-10-24**,
   see above. Palette was not the defect; lighting and invented geometry are.
2. **Derive the front azimuth per mesh** rather than offsetting. Cheapest item
   on this list, and it has now been got wrong three times.
3. **Multi-view reconstruction from the existing eight `dir_*.png`.** Attacks
   the dominant remaining render defect using input the 2D lane already
   produces for free. Needs a model that fits 12 GB; none confirmed.
4. **bpy toon render with a key light and an outline pass.** Better justified
   than it was - the 1-144 luminance measurement shows lighting is a real
   cause, and outlines are most of what makes a sprite read at 48px.
5. **Why one side collapses** in the reconstruction-to-skeleton step. It is
   consistent across two characters and unexplained.
6. **AniGen** last - it generates skeleton and geometry jointly
   rather than inferring a skeleton from a reconstruction, which is the one
   mechanism that attacks the measured failure, but it is reported at 18 GB+
   VRAM against a 12 GB card and its licence has not been read.

Do **not** re-run the T-pose test with a fourth character hoping for a better
number. 0004's finding was that no input property predicts which characters
succeed; this document adds that a textbook T-pose is not one either.
