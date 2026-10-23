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

1. **The concept was photoreal.** Generated with no pixel-art trigger, so
   photoreal colour baked into the vertices. A pixel-art concept bakes
   pixel-art colour. This is the cheap test and it was not run.
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
`.obj` meshes, the `_col.glb` coloured meshes, the `_skel.fbx` skeletons, the
8-azimuth renders in `render_zombie/`, and both comparison strips.

If you extend it, the order that wastes least GPU is:

1. **Pixel-art concept through the render path** - the cheapest unanswered
   question, and it gates everything else.
2. **Why one side collapses** in the reconstruction-to-skeleton step. It is
   consistent across two characters and unexplained.
3. **AniGen** only after (1) - it generates skeleton and geometry jointly
   rather than inferring a skeleton from a reconstruction, which is the one
   mechanism that attacks the measured failure, but it is reported at 18 GB+
   VRAM against a 12 GB card and its licence has not been read.

Do **not** re-run the T-pose test with a fourth character hoping for a better
number. 0004's finding was that no input property predicts which characters
succeed; this document adds that a textbook T-pose is not one either.
