# 0003 - Directional sprites: view cores, structured actions, and what blocks 8-way

Date: 2026-08-22
Status: SUPERSEDED IN PRACTICE by
[0004](0004-pivot-to-3d-conveyor.md), which moves directional sprites to a 3D
intermediate. Kept, not deleted: this document is the evidence that the 2D route
cannot reach whole-sheet consistency, and is therefore the justification for
paying 0004's cost. The mechanisms below are still live in the code and still
correct for anything that stays 2D.

(Previously: Accepted, 8-way deferred - the model spike found no viable
off-the-shelf route on 12 GB; see "Spike results".)

## Context

Step 2 renders an action by handing the core image to img2img with an OpenPose
skeleton on ControlNet. That works for actions which only change posture. It
does not work for actions that change which way the character FACES, and the
reason is structural rather than a tuning failure:

**img2img cannot rotate a character.** It preserves composition, and facing IS
composition. A profile skeleton over a camera-facing core asks for a turn the
init image forbids, so the model satisfies the init and ignores the skeleton.

Measured, on the zombie core:

| Symptom | Evidence |
|---|---|
| `move right` rendered front-facing | Character faced the camera while a strict-profile skeleton was applied at `controlnet_conditioning_scale` 1.0 |
| `move up` renders identical to `move down` | Both route to `WALK_FRONT`, whose `_head(False)` places nose and BOTH eyes camera-facing, so ControlNet demands a face while the prompt asks for "back of the head, no face". Rendered output shows a full face with red eyes for both |
| Raising strength does not fix it | 0.90+ turns the body but returns a different character; see the sweep in `tasks.py` next to `POSED_STRENGTH` |

This is the same defect class as two others found the same day - `burning`
routed to the side-view `HURT` cycle, and diagonals silently collapsing. In all
three the **skeleton and the prompt described different characters and nothing
complained**. Treat "skeleton facing disagrees with prompt facing" as a named
bug class, not three coincidences.

## Decisions

### 1. Derived view cores

For an action whose skeleton is in profile, derive a profile-facing core once
per sheet and use it as that action's init, instead of fighting the front core.
Implemented as `tasks.derive_side_core`, selected by `poses.is_side_view()`.

`is_side_view` keys off `_build(side=True)` collapsing both shoulders onto
x=0.50, so the two shoulder keypoints being identical is an exact structural
marker. Deliberately NOT a separate flag - a flag is a second source of truth
that drifts from the coordinates.

### 2. The IP-Adapter is what makes rotation possible

Rotation needs a strength high enough to repaint the body (0.90), which alone
destroys identity. The IP-Adapter re-injects the core through cross-attention at
every denoising step, holding identity at that strength.

This is worth stating carefully, because the same adapter was measured and
REJECTED for `burning` in the same session:

- It **suppresses anything the reference does not contain**. Burning flame
  coverage went 15.57% (no adapter) to 0.00% (scale 0.50+). Fatal for actions
  that must ADD an effect.
- A **rotation adds nothing**. The whole character is present in the reference;
  only the viewing angle differs. So the adapter helps here and hurts there.

Swept for the side core: scale 0.0 turns but degrades to a blob, 0.4 gives a
clean profile with the character intact, 0.7 clings to the front core and will
not turn. Chosen: strength 0.90, `ip_adapter_scale` 0.4.

### 3. Mirror, never re-generate, for the left/right pair

A left-facing core is the right-facing core flipped horizontally. Two
independent generations do not agree on the character, and a walk that changes
clothes when it turns around is worse than one that does not turn. Left and
right are the *same pixels*, which is why that pair is the only one with no
cross-view identity problem.

### 4. Actions become structured `{motion, direction}`

Action matching was contiguous-substring over free text, order-dependent.
`"move up right"` contains `"move up"`, so every diagonal was silently swallowed
by a cardinal - measured: `move up left`, `move up right` and `move down right`
all rendered as pure up/down with the horizontal component discarded, and
`walk northeast` fell through to the generic `("move","walk","run")` catch-all
and rendered as "walking toward the camera".

`poses.py` already documented the workaround ("specific entries must come before
general ones"). That footgun has now fired three times. Direction becomes data
rather than something parsed out of prose.

Legacy strings normalise at the boundary rather than being migrated in the DB,
because string handling cannot be deleted anyway: `action_prompts.json`
`_readme.fallback` promises a hand-typed custom action still reaches the model
as its own words. 52 existing spritesheet rows keep replaying through
`/api/task/{id}/retry`, which feeds `requested_actions` straight from the DB.

The something2 provider contract is unaffected - `a1111.py` never constructs
actions.

## Open: 8-way is gated on a model spike

The target is 8 directions with real diagonal art. A feasibility spike deriving
the three view families beyond profile found two blockers:

1. **3/4 views did not render as 3/4.** COCO-18 offers nose, two eyes and two
   ears, and `_head` has exactly two modes: full-face or profile. Hand-built 3/4
   variants (near eye, both ears, nose displaced) collapsed to profile. 3/4 is
   four of the eight directions, so this is half the deliverable, not an edge
   case.
2. **Cross-view identity drift is severe.** The front core is a hunched,
   detailed zombie in a layered jacket; every derived view came back thinner and
   simpler with different clothing. An 8-way set built this way reads as eight
   related characters rather than one character from eight angles - which for a
   game sprite is precisely the thing that has to hold.

The back view (N) DID turn correctly - no face - but at degraded quality (thin
body, malformed arm, a white rectangle artifact between the legs).

So 8-way is not blocked on effort; it is blocked on whether any available model
does consistent turnarounds. Spike before building. Success criteria:

- produces a recognisable 3/4 view, the mode `_head` structurally cannot express
- views of one character are mutually consistent, not cousins
- fits 12 GB alongside ControlNet and the adapter's CLIP image encoder

Candidates: SDXL at 1024 with its openpose ControlNet, a character-turnaround
LoRA, and `runwayml/stable-diffusion-v1-5` as a control for whether the
checkpoint or our wiring limits pose adherence.

**Note for anyone starting from ADR 0002:** its claim that `get_sd_pipeline`
"explicitly refuses ControlNet on SDXL" is now stale. The code swaps in
`thibaud/controlnet-openpose-sdxl-1.0` when the target is SDXL
(`tasks.py`, `is_sdxl and controlnet == OPENPOSE_CONTROLNET`). SDXL is a live
candidate, not an excluded one.

## Spike results (2026-08-22)

### Off-the-shelf turnaround models do not fit this hardware

Every multi-view / turnaround LoRA found is tiny itself (0.06-0.9 GB) and sits
on a base that cannot run on a 12 GB card. Measured via the HF API:

| LoRA | Base | Base size |
|---|---|---|
| `reverentelusarca/kontext-turnaround-sheet-lora-v1` | FLUX.1-Kontext-dev | 57.9 GB, gated |
| `Alissonerdx/CharacterSheet` | krea2 / klein9b | klein-9B 52.9 GB gated; krea-2 401 |
| `matlod/minimax-h3-turnaround` | MiniMax-H3 | 498 GB, image-text-to-video |

The Kontext LoRA is exactly the missing capability - front, 3/4 left, left
profile, back, right profile, 3/4 right - and it is simply out of reach here.
**8-way with real diagonal art is not achievable off-the-shelf on this GPU.**

### Onodofthenorth was removed for a reason that is not true

`templates/index.html` records: *"every trigger it ships is a SHEET trigger
(PixelartFSS = 'Front Sprite Sheet') and it returns four characters in a row."*

The removal was correct FOR STEP 1 and wrong for step 2, and the distinction is
the whole point.

The "four characters in a row" result is real and measured - see the comment on
`CORE_TRIGGERS`, which records it at guidance 7.5 with duplicate-suppression
negatives, concluding "training beats guidance". But that was **txt2img with no
init image**. Step 2 is img2img from a single-character core *plus* a single
ControlNet skeleton, and those two constraints pin the composition: rendering
FSS/BSS/RSS through step 2's path returned single characters every time.

The other half - that these are "SHEET triggers" and therefore not view
selectors - is a misreading. The card documents them as four **directional
views** (`FSS` front, `BSS` back, `LSS` left, `RSS` right) producing individual
characters from specified angles, and advises mirroring left/right from the best
result, which is what `derive_side_core` already does.

It is also genuinely SD1.5 (`cross_attention_dim: 768`, `sample_size: 64`,
`StableDiffusionPipeline`), so it works unchanged with the SD1.5 openpose
ControlNet, the SD1.5 IP-Adapter and every `Y` proportion in `poses.py`.

### But it does not solve the actual blocker

Tested with the model as the only variable - same core, same back-view skeleton,
same strength 0.90 / `ip_adapter_scale` 0.4 recipe as `derive_side_core`:

| Case | Result |
|---|---|
| All-In-One, "back view" prompt | Turned away but malformed - bad arm, white rectangle artifact |
| Onodo `PixelartBSS` | **Genuine faceless back view.** The trained trigger works |
| Onodo `PixelartFSS` | Best-looking of the three; clear zombie, jacket, face |
| Onodo `PixelartRSS` | Did **not** clearly read as a right profile |

So a *trained* back view beats an *invented* one, which was the hypothesis. But
the BSS output is a low-detail generic green humanoid - the core's layered
jacket and distinctive clothing are gone. **Identity collapses across views on
this model too**, exactly as it did on All-In-One.

Conclusion: swapping the checkpoint answers "can a back view exist" and does not
answer "is it the same character". Cross-view identity is the blocker, and it is
not a checkpoint problem. That points at deliverable 3 - LoRA training on
curated outputs - as the only credible route to consistent multi-view sprites on
this hardware.

Caveat: settings were inherited from All-In-One tuning. Onodofthenorth may do
better at a different strength, adapter scale, or as txt2img letting the trigger
carry the view rather than img2img fighting the core. Not swept.

## Consequences

- Up to five derived cores per sheet if 8-way lands; each costs a generation.
  Derivation is lazy and cached per sheet, and falls back to the front core with
  a warning rather than failing the job.
- Left/right are trustworthy; every other cross-view pair carries drift risk
  until cross-view identity is solved - plausibly by LoRA training, which
  `project-context.md` already lists as deliverable 3.
- `attack_melee` uses the profile `ATTACK` cycle, so it now silently receives a
  derived side core. Shipped, never verified.
