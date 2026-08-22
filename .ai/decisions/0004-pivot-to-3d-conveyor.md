# 0004 - Pivot the sprite conveyor to a 3D intermediate

Date: 2026-08-22
Status: Direction accepted.
  - Stage "mesh" is BUILT AND VALIDATED (2026-08-22) - see "Mesh stage results".
  - Stage "rig" is next and unbuilt. The open question there is skeleton
    CONSISTENCY across characters, not animatability - see "Gate result".

A previous revision of this file said "ROUTE BLOCKED". That was wrong and is
corrected below; it rested on a summarised paraphrase of the UniRig README
rather than the README itself.

Supersedes most of [0003](0003-directional-sprites-and-view-cores.md) in
practice - not because its reasoning was wrong, but because the problem it
solves stops existing.

## Context

Reference sheets were supplied showing the target: RPG-Maker-format character
sheets of roughly 9 columns x 16 rows, about **150 cells per character**, 8
directions x ~15 actions, plus a pre-rendered werewolf sheet.

Their defining property is not art quality. It is that the character is
**pixel-identical in every cell** - same jacket, same hair silhouette, same
palette, across all directions and all actions.

That is precisely the property 2D diffusion failed to deliver, measured
repeatedly on 2026-08-21/22:

| Attempt | Result |
|---|---|
| Derived view cores | Back core returned a thinner, simpler, differently-dressed character. On BOTH All-In-One and Onodofthenorth |
| 3/4 views | COCO-18 exposes nose / 2 eyes / 2 ears, and `_head` has exactly two modes. Hand-built 3/4 collapsed to profile - and 3/4 is 4 of the 8 directions |
| Off-the-shelf turnaround models | Every one sits on a 50GB+ gated base. None fit 12GB |
| Frame-to-frame within one row | Drifts even with a shared seed AND the IP-Adapter holding identity |
| Chaining frame N-1 forward | Improves adjacent frames, wrecks loop closure, compounds VAE artifacts |

The cause is structural, not a tuning gap: **every cell is an independent sample
from a distribution, and the target needs ~150 samples to agree exactly.** No
amount of prompt, strength or adapter tuning changes that.

The reference itself shows the answer. The werewolf sheet was never generated in
2D - it is a 3D model rendered from 8 camera angles. That is how Diablo, Fallout
and Age of Empires produced directional sprites, and it solves consistency by
**never sampling twice**.

## Decision

The conveyor becomes:

```
concept  ->  mesh      ->  rig  ->  clip      ->  render        ->  sheet
diffusion    image-to-3D   rig      animation     8 cameras         compose
                                                  + pixelate
```

Consistency is then structural: every cell is the same mesh under a different
camera, so it cannot drift.

### What survives

**Step 1 survives**, as a *concept* generator. ADR 0002 already records that
"the core does not need to be a pixel-art model - it needs to produce one clean,
well-composed, centred character". That stays true; the pixel style simply moves
from step 2 to the render step.

### What is retired

Everything in step 2 built to fake directional consistency in 2D: the hand
authored skeletons in `poses.py`, `derive_view_core`, the IP-Adapter view
wiring, `VIEW_TRIGGERS`, and the per-action view prompts. 3D supplies all of it
for free.

Keep the code and ADR 0003 rather than deleting them. They record why the 2D
route fails, which is the justification for paying the cost of this one.

### Model selection: TripoSR

| Candidate | Size | Licence | Verdict |
|---|---|---|---|
| `stabilityai/TripoSR` | 1.68 GB | MIT | **Chosen** |
| `microsoft/TRELLIS-image-large` | 3.3 GB weights | MIT | Fallback - reported ~24GB PEAK VRAM for sparse-voxel decoding; weights are not the constraint, runtime is. Untested at 12GB |
| `stabilityai/stable-fast-3d` | 4.02 GB | other | Gated |
| `tencent/Hunyuan3D-2mini` / `-2` | 25.3 / 74.9 GB | other | Out on size |

TripoSR is marked down in reviews for "modest quality, at the cost of textures
and fine detail". **That criticism does not apply here.** The final cell is a
48-128px sprite; TRELLIS's 4096-square PBR and Hunyuan's 8K textures are
destroyed by pixelation. What survives at 128px is silhouette and volume, which
is what TripoSR provides. Choose the smallest model that gets the silhouette
right, not the highest-fidelity one.

## The gate

Model choice was the easy part and is not the risk. The risk is the chain after
it, and it is unverified:

1. **Image-to-3D output is unrigged and messy.** Marching-cubes-style topology:
   non-manifold, no clean edge loops, no skeleton. Auto-rigging such a mesh
   frequently fails or deforms badly.
2. **Auto-riggers expect a T-pose or A-pose humanoid.** The current core is
   generated hunched with arms down, which is close to worst-case input. Step 1's
   prompt may need to change to produce a rigging pose - which would mean the
   core image stops being the thing that looks good in the UI and becomes
   rigging feedstock. STILL UNTESTED: UniRig claims to rig "diverse 3D assets"
   rather than requiring a template pose, so this may not bite. Do not rebuild
   step 1 for it until UniRig has actually rejected a hunched mesh.
3. **Mixamo is an Adobe web service.** Scriptability and ToS unchecked. No
   offline auto-rigger has been evaluated.
4. **Blender is not in this stack at all** - no 3D tooling of any kind is present
   in the repo. Headless rendering would run on 4 CPU threads inside WSL beside a
   12GB GPU already holding diffusion models.
5. **Pixelation method is unspecified.** It is what makes a render read as the
   werewolf sheet rather than as a generic 3D render.

Spike order is mesh -> rig -> animate, not mesh quality. If a generated mesh
cannot be rigged and animated, the honest fallback is sourcing pre-rigged models
and using the pipeline as a render farm - which still hits the reference bar but
stops being an AI character generator.

## Mesh stage results (2026-08-22): BUILT, and it works on existing cores

`compose/develop/mesh_worker/` + `src/mesh_worker/`. Isolated container
(Python 3.11, CUDA 13 devel base, torchmcubes compiled WITH the CUDA kernel),
driven by `mesh.py`; `inspect_mesh.py` judges the result for rigging fitness.

### Step 1 does NOT need to change to concept art

This ADR originally predicted the core would have to be re-generated as clean,
high-res concept art rather than pixel art. **That prediction was wrong.** The
EXISTING pixel-art cores reconstruct fine:

| Core | Faces | Watertight | Aspect | Silhouette IoU | Verdict |
|---|---|---|---|---|---|
| 48, zombie | 47,828 | yes | 2.16 | 0.908 | rig-ready |
| 59, humanoid | 73,056 | yes | 2.42 | 0.884 | rig-ready |
| 82, goblin IN A FRAME | 396,886 | no | 1.02 | 0.84 | FAILS |

~24s per character end to end (5.3s inference, 3.8s marching cubes, rest is
model load). Silhouette IoU ~0.9 means it reconstructs THAT character, not a
generic humanoid - which was the question that mattered.

### The real input requirement: no scenery, not "concept art"

Core 82 is a goblin inside a decorative frame. TripoSR faithfully reconstructed
the FRAME as geometry: 5x the faces, 165 stray components, aspect 1.02 instead
of a humanoid 2-3. The character was fine; the border destroyed it.

So the constraint on step 1 is **isolation**, not style: one character, no
frame, no border, no scenery, keyed background. `NEGATIVE_SINGLE` already fights
grounds and duplicates; frames are not covered and should be added if cores are
generated for this path.

Also note the input must have its ground patch stripped first
(`tasks.strip_ground_patch`). Raw core 48 measures aspect 1.44 because the dirt
widens it, and that dirt would be reconstructed as geometry too.

### Watertight was a real blocker, and is solved

Raw TripoSR output is not watertight - measured 100 boundary edges of 71,768
(0.14%), with boundary vertices in the INTERIOR rather than on the volume faces,
i.e. small holes rather than a mesh clipped by the grid. `repair_for_rigging()`
in `mesh.py` drops sub-threshold components, cleans degeneracies, then fills and
fixes winding: watertight False -> True, euler -26 -> -2, positive computable
volume. Order matters - specks are dropped BEFORE filling, or hole-filling seals
them into a solid pebble inside the rig.

## Gate result, rigging leg (2026-08-22): the chain breaks at ANIMATE

Researched before downloading anything. The best open auto-rigger is
**UniRig** (VAST-AI, SIGGRAPH 2025) - conveniently the same lab as TripoSR:

| | |
|---|---|
| Size | 6.44 GB total; **5.8 GB for inference** (skeleton 1.4 + skin 4.4). The 4.9 GB `data/rigxl/processed.7z` is training data |
| Licence | **MIT** - matters, because the alternative RigNet is GPL3-or-commercial, unusable in a game without buying a licence |
| VRAM | 8 GB minimum - fits, with diffusion unloaded |

**Install burden, not a blocker.** UniRig needs Python 3.11 (`sprite_worker` is
3.10) plus compiled CUDA extensions - `spconv`, `flash-attn`, `torch_scatter`,
`torch_cluster` - and its own README warns flash-attn "may encounter errors".
Building those on a 4-thread i3-8100 is slow. But this needs its own container
regardless, and a `python:3.11` base solves the version question outright. Cost,
not obstacle.

**What UniRig actually outputs.** From the README: `joints`, `parents`, `names`,
and `matrix_local` "aligned to Y-up axis, consistent with Blender" - a valid,
named, Blender-compatible armature with per-vertex skinning weights. It also
notes that "combining UniRig with keyframe animation produces" animated results.
So the output IS animation-ready.

What it does NOT promise is a standard bone-naming convention or a predefined
humanoid template, and it says nothing about Mixamo or retargeting.

### The real open question: skeleton CONSISTENCY

Retargeting a stock animation onto a non-standard but humanoid skeleton is a
solved problem in Blender - Rokoko's retargeter, Auto-Rig Pro's remapper, or a
script matching by hierarchy topology instead of by name. So "can it be
animated" is not in doubt. The question is whether the mapping is authored ONCE
or PER CHARACTER:

- If two humanoid meshes yield the same bone count and hierarchy, one mapping
  serves every character and the conveyor is fully automatic.
- If each mesh gets a differently-shaped skeleton, mapping is per-character
  manual work and the automation argument collapses.

**That is the test to run**, and it is cheap once installed: rig two different
generated characters, compare bone counts and hierarchies. Mesh quality was
never the risk, and neither is animatability - this is.

### Fallback if consistency fails

**Pre-rigged source models** (asset stores, Mixamo characters) skip mesh AND rig,
keep stock-animation compatibility, and still hit the reference bar. Cost:
geometry is no longer generated, so the product becomes a render farm rather
than an AI character generator. This is also the variant that suits a 4-thread
CPU best, since it skips the compile burden entirely.

### Correction notice

An earlier revision recorded this leg as BLOCKED, on the strength of a
summarised paraphrase of the README claiming UniRig "generates custom skeletons
tailored to input geometry rather than standardized humanoid rigs". Reading the
README directly does not support a blocking conclusion. Do not treat summarised
secondary readings as gate evidence.

## Consequences

- The three deliverables in `project-context.md` predate this and need revisiting.
  **Deliverable 3 (LoRA training) may be moot** - it is the 2D answer to
  consistency, solving a problem the pivot removes.
- The something2 provider contract is a txt2img A1111 facade. A 3D conveyor does
  not fit that shape, and something2 is the first external consumer.
- "Core image" changes meaning: today an img2img init, in the conveyor a concept
  reference that is never denoised from. See `domain.md`.

## Gate result (2026-08-22): the rig stage runs, and the meshes fail it

The consistency test above was run. The rig stage works end-to-end - UniRig
produces a skeleton in ~13s on the 3060 - but the answer it gives is not the one
the plan assumed.

### What was measured

Five meshes were rigged and their skeletons labelled from geometry alone
(`identify_limbs.py`: does each limb end land where an arm/leg/head should be).

| mesh | source | mesh quality | skeleton | limb ends labelled |
|---|---|---|---|---|
| control | synthetic box humanoid | watertight | 22 bones, spine + 2 arms + head + 2 legs | **yes, perfect** |
| meshtest | TripoSR, hunched zombie | watertight, IoU 0.908 | 19 bones, 3 branch / 5 ends | yes |
| charB | TripoSR, stocky, fused ground disc | watertight, IoU 0.895 | 6 bones, straight chain | no |
| charB2 | TripoSR, upright, arms at sides | watertight, IoU 0.844 | 10 bones, 3 branch / 5 ends | no - arms 52% of body height apart, same side |
| charB3 | charB with the disc cut away | IoU 0.910, NOT watertight | 21 bones, 2 branch / 5 ends | no - three limb ends converge on one point |
| charD | TripoSR, clean upright figure | watertight, IoU 0.884, **best of the set** | **3 bones, 1 end** | no |

### The finding

**UniRig is not the problem. The reconstructions are.** Given an unambiguous
humanoid - boxes, limbs clear of the torso, perfectly symmetric - it returns a
textbook rig: spine, two level arms at +/-0.93, head above, two level legs at
+/-0.16. On TripoSR reconstructions of pixel-art sprites it succeeded once in
five.

Mesh quality does not predict skeleton quality. charD had the best mesh in the
set by every metric measured and produced a 3-bone vertical line; meshtest was
comparable and produced a correct biped.

### Hypotheses tested and rejected

- **Fused ground patch.** charB3 (patch removed) came out WORSE than charB
  (patch present): 21 bones with three limb ends converging on one point.
- **Arms held against the body.** charD has that posture and failed at 3 bones;
  charB2 has it and produced 10 bones with correct branching. Same posture, two
  unrelated failures - so posture is not the variable.
- **Watertightness.** Four of the five were watertight, including both total
  failures.

No input property measured here predicts which characters rig successfully.

### What this costs the plan

The original goal - author one retarget map, reuse it for every character - is
gone in both of its forms:

- A bone-NAME or hierarchy-POSITION map cannot be authored: where two skeletons
  are both correct (control and meshtest) they still carry different joint
  counts per limb, so there is no one-to-one bone correspondence.
- An IK retarget keyed on the five limb ends WOULD survive differing joint
  counts, and is the reason `identify_limbs.py` exists. But it needs the limb
  ends to be identifiable, and they are identifiable on 1 of 5 characters.

### Options, in order of what the evidence supports

1. **Replace image-to-3D.** The rigger is fine; the reconstruction is what UniRig
   cannot read. A model producing cleaner limb separation (TRELLIS, InstantMesh,
   Hunyuan3D) may move the success rate directly. This is the cheapest test with
   the largest expected effect, and nothing else in the conveyor changes.
2. **Feed reconstruction a T-pose.** Every source core here is in a rest or
   action pose with limbs near the torso. The control that worked had limbs held
   clear. Untested and NOT the same claim as the rejected "arms at sides"
   hypothesis - that compared arm position within action poses, not a true
   T-pose generated for reconstruction.
3. **Pre-rigged source models** (the fallback recorded above). Still available,
   still costs the "AI generates the geometry" property.

Option 1 first. Option 2 is cheap to bolt on and the two are independent.

### Tool corrections made while getting here

Four measurement bugs were found and fixed, all of the same shape - a rule
calibrated on one character, applied to another:

- `inspect_mesh` scored a mesh against `*_input.png`, the grey-composited image
  TripoSR consumes, whose alpha is 100% opaque. Two good meshes read as broken
  (0.383 and 0.412; against the keyed sources, 0.844 and 0.895). It now REFUSES
  an image that is >95% opaque.
- `inspect_mesh` judged aspect against a fixed 1.6-3 "humanoid" band tuned on
  one slim zombie, failing charB at 1.48 - whose SOURCE measures 1.40. It now
  compares against the character's own proportions.
- `identify_limbs` took "up = axis with the largest spread". In a T-pose the arm
  span exceeds the height, so it mislabelled the control - the one skeleton that
  was perfect. It now searches orientations against biped constraints.
- That search then found degenerate orientations, passing charB2 by treating its
  SHORTEST axis (0.468) as up for a figure 2.199 tall. Candidate up axes are now
  restricted to the dominant extents.

`strip_ground_patch` has a documented gap it cannot currently close - see its
docstring in `tasks.py`. It is cosmetic in 2D and structural in 3D.
