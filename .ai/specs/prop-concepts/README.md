# Prop concepts - getting a subject that is not a character

A "prop" here is any step-1 subject that is not a humanoid: a tree, a crate, a
banner, a barrel, a creature that does not stand upright. There are two
supported routes, and one that looks supported and is not.

Related: [0015](../../decisions/0015-edit-endpoint-on-qwen.md) (why editing a
bad concept is not the fix), [0012](../../decisions/0012-qwen-image-default-lane.md),
migration `022_core_prop.sql`.

## The problem this exists to solve

`tasks._core_prompt` wraps every step-1 subject in sprite-specific styling,
and that styling is unconditional:

    a single {your prompt}, one character alone, full body, standing,
    centered, pixel art sprite, 16-bit, sharp focus, plain white background

A CLIP/Qwen text encoder has no way to know that "one character alone, full
body, standing" is scaffolding rather than part of the request, so it applies
it to whatever it is given. **"a tall dense dark green pine tree" comes back
as a tree with a small man standing at its base.** Measured on both SDXL
+nerijs and qwen-image-2512, which is what rules out the checkpoint as the
cause - it is the prompt.

The wording is not an accident and must not simply be deleted: the long
comment in `_core_prompt` records that stating the exclusions *positively*
("one character alone") is what stopped cores coming back as a crowd of five
to nine duplicates, and that it must stay positive because distilled
checkpoints run at guidance 0 where the negative prompt is a silent no-op.

Route 1 below is the measurement of what happens when you delete it anyway.

## Route 1 (REJECTED) - a "prop mode" on `/api/generate_core`

**Built, measured, and reverted on 2026-10-21. Do not rebuild it from the
reasoning alone - the reasoning is sound and the result is worse.**

The obvious fix is to drop the humanoid clause for non-character subjects,
since a crate has no body and does not stand. Three wordings were tried
against the current one. Same subject ("a tall dense dark green pine tree"),
same model (SDXL + nerijs), the same 12 fixed seeds, through
`/sdapi/v1/txt2img` so the prompt reached the model verbatim. Counting frames
that came back as a forest or a contact sheet instead of one subject:

| positive wording | bad / 12 |
|---|---|
| `a single X, one character alone, full body, standing` (current) | **2-3** |
| `a single X alone` | 7 |
| `a single X, one object alone, whole object in frame` | 7 |
| `a single X, one object alone, full body, upright` | 6 |

**The token "character" is itself carrying the isolation.** Every
substitution lost it and the frame filled with trees - the same duplication
failure the long comment in `_core_prompt` records this wording having fixed
in the first place. "object" does not substitute for it, and neither does
repeating "alone".

Two further findings from the same 48 generations:

- **The motivating complaint did not reproduce.** Not one of the 48 images
  contained a person. The bystander is real - it arrived with a sample image
  from the something2 session, on both SDXL and qwen-image-2512 - but it is
  not a reliable consequence of this wording, so it cannot be measured away
  by changing it. Anyone retrying this needs a reproducer first.
- **Seed dominates wording.** At seed 777 all four wordings produced a
  contact sheet; at 4242 all four produced a framed card. A two-seed
  comparison here is noise, which is why the table above uses twelve. This is
  the same lesson `generate_raw_task` records: a contact sheet is a property
  of the seed.

A `prop` flag, its `sprite_images.prop` column and migration 022 were all
written and then removed with the wording. If a future attempt needs the
flag, note that the retry path (`POST /api/task/{id}/retry`) rebuilds its
call from the row, so an unstored flag silently reverts on retry.

## Route 2 - `/sdapi/v1/txt2img`, the one that actually works

The A1111 facade goes to `generate_raw_task`, which does no prompt rewriting
of any kind - it exists precisely because the sprite styling is wrong for
tileable ground textures and other non-character subjects. A caller that
wants to compose its own prompt exactly should use this and not
`/api/generate_core`.

Then feed the result into a sheet build by **filename**:

    POST /api/jobs
      { "concept_image": "raw_abc123.png", "concept_check": false, ... }

**Verified on 2026-10-21**, not read off the source: a prop generated this way
("pixel art, a tall dense dark green pine tree, 16-bit, sharp focus, plain
white background", `cutout: true`, seed 4242) came back as one isolated tree,
transparent, with no person in it. One generation, no edit, no VAE round trip.

**Stop there. A prop is a finished static asset.**

### Do NOT send a prop through `POST /api/jobs`

An earlier draft of this file recommended exactly that, and it is wrong. The
run that proved it: the tree above, `actions: ["idle"]`, one direction, one
frame. The job was accepted, passed the concept gate, ran all five stages for
seven minutes, and produced **a green humanoid person in a dark shirt and
trousers** - `images/sheet_32e0657d-....png`, 48x64.

The sheet conveyor is character-only by construction and not by accident. Its
turnaround and action stages drive Qwen-Image-Edit with camera and pose
instructions ("idle", facing south) from `action_prompts.json`, and the
multiple-angles LoRA pins pose as well as framing (see `qwen_edit.denoise_only`).
Handed a subject with no pose, the model invents a body that can hold one. A
tree has no idle animation, so there is nothing for the conveyor to do but
turn it into something that does.

This is the same requirement ADR 0004 and 0005 record from the 3D side: the
constraint on a concept is isolation **and** that it is a character.

Two further details, for anyone who does need the hand-off for an actual
character:

- **`concept_image` is a basename resolved under `images/`**, not an arbitrary
  path (`build_sheet_job` does `os.path.basename`). There is no upload
  endpoint; anything already in `images/` is addressable, including every
  `core_*.png` and `edit_*.png`.
- **`concept_check: false` is NOT needed for a clean cut-out prop**, which is
  the other thing the earlier draft got wrong. It claimed a tree would be
  refused with "concept does not look like an isolated character". It was not:
  the job passed. `concept.judge` measures *isolation* - coverage, aspect,
  border - not species, and a cleanly keyed prop on a transparent background
  looks exactly like an isolated subject by those numbers. **So the gate will
  not save you from the humanoid above.** Note also that the check runs in the
  WORKER: `POST /api/jobs` returns 202 either way and a refusal arrives later
  as a failed job, never as a 400.

## The route that is NOT supported: generate, then edit the person out

`POST /api/edit` cannot repair a concept, and this was measured rather than
assumed (2026-10-21, Qwen-Image-Edit-2511 Q2_K + Lightning, production path).

It is not a masked inpaint - it re-renders the whole image through a VAE - so
the damage scales with how little of the frame the subject fills:

| source | subject coverage | after | distinct colours | outcome |
|---|---|---|---|---|
| large pine | 39.2% | 38.7% | 40,081 -> 29,187 | clean, framing held |
| banner pair | 14.1% | 43.9% | 26,847 -> 32,713 | invented four black blobs |
| bare tree | 8.3% | 39.0% | 16,094 -> 38,697 | person NOT removed, 2x zoom |

Cropping to the subject's alpha bbox first (tested) removes the hallucination
but not the rest: the cropped banner still went 23.1% -> 54.4% coverage and
**26,847 -> 74,579 colours**, a 2.8x explosion that is the pixel grid being
replaced by smooth gradients. A crisp pixel-art grid does not survive a VAE
round trip at any framing, and Q2_K - the only quant that fits 12 GB - makes
it worse.

Editing stays useful for what it did well: large, frame-filling, organic
subjects, where the softening reads as texture.
