# Domain language

Vocabulary that is ambiguous in this codebase, or that reads as ordinary English
while meaning something specific here. Add a term when a wrong reading would
cause a wrong change, not merely because a word is domain-ish.

> **Pending change.** [decisions/0004](decisions/0004-pivot-to-3d-conveyor.md)
> moves the conveyor to a 3D intermediate. When that lands, "core image" becomes
> a **concept reference** feeding image-to-3D rather than an img2img init, and
> *view family* / *derived view core* below become historical - 3D has no view
> families, only camera angles. Both are kept because they explain why the 2D
> route failed. New stage vocabulary will be needed:
> concept -> mesh -> rig -> clip -> render -> sheet.

## "Core" is overloaded - three meanings

The most dangerous word in the sprite pipeline. It currently carries three
distinct senses, and up to five objects in one task can each be called "the
core".

| Term | Means | Where |
|---|---|---|
| **Core image** | The step-1 output: one character, one view, saved to the DB and pickable in the UI. What a user means by "the core" | `image_type='core'`, `get_core_image_path()` |
| **Derived view core** | A core image ROTATED to another facing, generated at render time from the core image and never persisted. Not user-visible | `derive_side_core()` |
| **`core_img` / `core_frame` / `core_box`** | Local variables inside `generate_spritesheet_task` holding the loaded, composited and measured forms of the core image | `tasks.py` |

**Rule:** say *core image* for the persisted step-1 asset and *derived view
core* for a rotated variant. Never bare "core" in a comment where both are in
scope. If 8-way lands there will be up to five derived view cores per sheet and
bare "core" stops being resolvable at all.

## View family

A group of directions that share one derived view core. Left and right are the
*same* view family - a left-facing core is the right-facing one mirrored, so
they are literally the same pixels. Front, back and the two 3/4 angles are
separate families, each needing its own derivation.

This matters because **identity drift is a between-family problem, not a
per-frame one.** Frames within a family agree; families need not agree with each
other. When someone reports "the sprite changes between rows", check whether the
rows are in different view families before touching per-frame settings.

## Facing vs pose

- **Facing** - which way the character is turned relative to the camera. Set by
  the skeleton's head keypoints and body plan (`_build(side=...)`, `_head`).
  img2img cannot change facing; see `decisions/0003`.
- **Pose** - limb positions within a facing. Set by the skeleton's wrist and
  ankle endpoints.

Kept apart because a defect in one is fixed nowhere near the other. "The walk is
wrong" has meant both "it faces the camera while walking right" (facing) and
"the legs do not move" (pose), which have unrelated causes and unrelated fixes.

## Action

A single row of the sprite sheet - one motion in one direction, four frames.
Historically a free-text string (`"move right"`) matched by contiguous
substring; moving to structured `{motion, direction}` per `decisions/0003`.

Note the collision that motivated the change: `"move up right"` **contains**
`"move up"`, so diagonals were silently swallowed by cardinals.

## "Task" is overloaded - four meanings

Second only to "core", and newer, so it is not yet wrong everywhere - keep it
that way. Four different objects in this system answer to "task":

| Term | Means | Where |
|---|---|---|
| **Task** | a Celery task | `tasks.py`, `@celery.task` |
| **Job** | a row in `jobs` plus its `202`/poll contract | `jobs.py`, `tiles.py`, `GET /api/jobs/{id}` |
| **Ticket** | a Plane work item (`SOMET-*`) | plane.so, workspace `something2` |
| **Action** | one sheet row: motion + direction | see "Action" above |

**Rule:** never say "task" for a Plane item or a `jobs` row. "Add a task to
render tiles" is ambiguous across three of these at once - it has already
caused one planning session to address the wrong machine.

Note that something2 uses "job" for its own in-memory registry too
(`rmt_`-prefixed, `services/remoteImageProvider.js`). Their job is not our job:
theirs is one blocking HTTP call awaiting a response, ours is a queued row a
worker picks up. When both are in scope, say **our job row** and **their remote
job**.

## Strength

`strength` is the img2img parameter, 0..1: how much of the init image the model
may paint over. It is NOT a quality or intensity dial, and it controls three
things at once, which is why one value never suits every action:

1. how much identity survives
2. how much freedom the prompt has to ADD something (flames, a bow)
3. **whether the frame is redrawn at all** - below ~0.70 the output is the init
   image lightly modified, so it keeps the core's painterly rendering and never
   becomes crisp pixel art

Sense 3 is the one that surprises people. "The row looks mushy rather than
wrong" is a strength problem, not a prompt problem.

## Trigger

A checkpoint-specific token that activates a DreamBooth finetune's trained style
(`pixelsprite` for All-In-One). Without it you get the base model wearing a thin
coat of prompt wording. Distinct from the action prompt text, which is the
descriptive fragment in `action_prompts.json` - both end up in the same string,
so do not use "trigger" for both.

## Skeleton / control image - and the second, unrelated "skeleton"

**Skeleton** - the COCO-18 keypoint dict authored in `poses.py`.
**Control image** - that skeleton rendered to RGB for ControlNet.

Worth separating because a bug can be in the keypoints (wrong anatomy) or in the
rendering and fitting (right anatomy, landed off the character).

**A second meaning already exists in this repo and means something else
entirely.** In `src/rig_worker/` - `dump_skeleton.py`, `identify_limbs.py`,
`compare_skeletons.py`, and UniRig's own output - a *skeleton* is a **3D
armature**: named bones with head/tail positions and a parent hierarchy, in an
FBX. It is not a 2D pose, it has no keypoints, and no ControlNet consumes it.

The two are not variants of one idea. The pose sense drives *what a character
is doing* in one image; the armature sense is *how a mesh deforms* across many.
Say **control skeleton** for the `poses.py` one and **armature** for the 3D one
when both could be meant - which is any discussion of the 3D lane
([decisions/0016](decisions/0016-rig-gate-retest-and-off-card-3d.md)).

**Rig** is a third word and a fourth object: mesh + armature + skinning weights
together. An armature alone is not a rig, which is why 0016's gate could report
a usable *armature shape* (28 bones, 5 limb ends, the right figure) on a
character that still could not be rigged.

## Frame - a sheet cell, not a moment in time

**Frame** in this codebase is a **cell index within a row**: the `f` in
`cut_{action}_{direction}_{f}.png`, `--frames N`, `atlas.rows[].motion`, and
`/api/action-catalog`'s `redraw.frame_context` and `.max_frames`.

It is *not* a timestamp, and it is not an animation keyframe. The two coincide
today only because the conveyor renders exactly one image per cell and the
consumer plays cells in order - so nothing has forced the distinction yet.

It would be forced immediately by anything that poses a model over time rather
than sampling an image per cell, because such a lane has **keyframes** (poses at
time offsets, interpolated between) and **clips** (a named sequence of them),
and neither maps one-to-one onto a cell. If that ever gets built, keep `cell`
for the sheet unit and do not let `frame` drift to mean both; the atlas, the
redraw API and something2's packing all read the current sense.

## "Map" is overloaded - five meanings, and it is the worst one yet

Third in the series after "core" and "task", and written down *before* the code
exists rather than after it hurt. Five distinct objects are in scope during a
single map build, and four of them are things a person would casually call
"the map":

| Term | Means | Produced by |
|---|---|---|
| **Biome painting** | The low-res diffusion output. A few dozen pixels square, palette-locked, one colour per terrain. Not art anyone looks at - it is the LAYOUT in visual form | the map LoRA, one denoise |
| **Tilemap** | The quantized terrain grid: `layers[[tile ids]]`. What makes ground walkable | quantizing the biome painting |
| **Region graph** | The LLM's semantic output. Tens of items - regions, roads, landmarks - never coordinates for hundreds of props | CPU llama.cpp |
| **Map picture** | The composited preview PNG. The only one that crosses something2's AI connector | compositing tiles + entity placements |
| **World / level** | something2's own noun for the thing it renders. Not ours | something2 |

**Rule:** never say bare "map" where two are in scope. The dangerous pair is
*biome painting* and *map picture* - both are PNGs, both are "the map image",
and they are at opposite ends of the pipeline at wildly different resolutions.
Confusing them means compositing the layout or quantizing the artwork.

Note the inversion that makes this design work and reads as backwards: the
**biome painting is data** and the **map picture is derived**. A wrong reading
here inverts the dependency and produces a picture the ground does not match -
the exact property the design exists to guarantee.

## Style (of audio) is a roster entry, not a string

In the audio surface a **style** is one entry in a Python roster - a prompt
template with slots and a negative list - the way a checkpoint is one entry in
`core_models.CORE_MODELS`. The LLM *selects* a style and fills its slots; it
never writes the model prompt from nothing. A free-text `prompt` is an explicit
**override** that bypasses the roster, and it is the exception.

Kept apart because "the style" in a request can mean the roster key
(`medieval_fantasy`), the filled template actually sent to the model, or the
caller's free text, and only the first is reproducible across re-rolls.

## Track vs ambience, and loop points

- **Track** - a `music` artefact: a musical bed for one map, minutes long,
  looped endlessly by the player.
- **Ambience** - a stationary texture under it (wind, birds, drips), tens of
  seconds, its own artefact and its own `kind`. Never mixed into the track.
- **Loop points** - `loop_start` / `loop_end` as SAMPLE offsets into the WAV
  master, carried both as OGG tags and in `info`. The **seam** is the place
  they meet, and it is the property under test - "the loop is fine" means the
  seam passed the fixed-set listening bar, not that the file plays twice.

Both are named by the map, like `map:<name>`. Do not say "the map's audio" when
one of the two is meant - they are fetched, looped and re-rolled separately.

## Cue and engine (sfx)

- **Cue** - one `sfx` roster entry: a one-shot sound for a game event
  (`slash`, `hit`, `pickup`). Deliberately NOT called an *action*: **action**
  already means one sprite-sheet row (above), and "the attack action" vs "the
  attack sound" would collide in every plan that touches both.
- **Engine** - how a cue is rendered: `realistic` (neural, Stable Audio Open)
  or `retro` (procedural sfxr-style, no GPU). Applies to `sfx` only; music and
  ambience have one model each and no engine field.

## Audio provider (something2's side)

A new connector kind on something2's side, distinct from their **image
provider**. Their image provider decodes `images[0]` as a PNG and slices it,
so it cannot carry audio; the audio one reads `audio[0]`. When both are in
scope say which, because "the provider" has meant the image one for a month.

## Entity asset vs entity placement

The Entity Generation tab produces **entity assets** - sprites, rows in the
asset list, generated once and reused. A map contains **entity placements** -
`{asset, x, y}` referring to one. One asset, many placements, across many maps.

Kept apart because "generate the entities for this map" is ambiguous across a
30-second reference lookup and five hours of GPU. Bare "entity" is fine in the
generation tab, where only assets exist, and never fine in map code, where both
do.

## Provisional (of a map)

A map whose terrain is final but whose entity placements are not all satisfied:
some referenced asset does not exist yet and has a queued generation job behind
it. Served, not withheld - `complete: false` plus a `pending` list, with
placeholder art standing in.

**Provisional is not "failed" and not "in progress".** The map is complete
enough to walk on and will improve without being re-requested. A caller that
treats it as an error abandons a usable map; a caller that treats it as final
caches a placeholder forever.

## `llm_name` is the image model, not an LLM

A legacy name from the llama.cpp era. In `tasks.py`, `main.py`, the
`sprite_images.llm_name` column and the form fields, **`llm_name` holds a
diffusion checkpoint string** (`"<base>+<lora>"`), never a language model.

| Term | Means | Where |
|---|---|---|
| **Image model** | The diffusion checkpoint that draws the PNG | `llm_name`, `core_models.CORE_MODELS`, `a1111.KNOWN_MODELS` |
| **Brain** | A text or vision language model that plans (region graphs, biome plans, audio styles) or judges images. Never draws | `brain_engine.BRAINS`, reached through `text.run_text` or `brain_engine.complete_in_job`; `decisions/0011`, `0013`. **Not** `llm-server`/`LLM_URL` — that router retired to the `legacy-llm` profile on 2026-10-01 |

**Rule:** say *image model* or *brain*, never bare "the model" or "the LLM",
where both are in scope. The confusion is not theoretical: on 2026-09-27 a
plan to improve sprite quality by upgrading the LLM got as far as choosing a
runtime before it surfaced that no LLM touches the sprite path at all.

It happened again on 2026-09-27 from the other side: a batch of files
described as "LLM images" were all diffusion transformers - *image models*.
A `.gguf` extension says nothing about which kind a file is; read
`general.architecture` and the tensor names (`decisions/0012`).

## "Model" on disk is four kinds, and each is fetched differently

Settled 2026-10-13, while working out how to provision a second machine. The
four live side by side under `MODELS_DIR` (which is also `HF_HUB_CACHE`), and
treating one as another is the failure: `hf download --local-dir` produces a
layout diffusers cannot resolve, and a GGUF one level deep gets served as a
chat model.

| Kind | Lives at | Fetched by | Example |
|---|---|---|---|
| **`hf-cache` repo** | `models--<org>--<name>/` — HF cache layout, resolved by *repo id*, never by path | `snapshot_download`, usually with allow/ignore globs | `stabilityai/stable-diffusion-xl-base-1.0`, `ZhengPeng7/BiRefNet` |
| **`hf-file`** | an explicit subdirectory, **two levels** below `/models` | single-file download to a named destination | `brain-gguf/qwen3.6-35b-a3b/…IQ4_XS.gguf` |
| **`checkpoint-tree`** | its own layout, outside the HF cache | whatever that project expects | `acestep/**` |
| **`local`** | `/models/loras/` | **nothing — trained on this machine** | `something2.safetensors` |

**Rule:** say which *kind* when the fetch or the path is in question. A `local`
adapter cannot be downloaded at all; `core_models` builds `local:<name>` roster
entries from whatever is in `LORA_DIR` and `repos_for` excludes them, so a
machine without them simply offers fewer options — that is not a failure.

**Depth is load-bearing for `hf-file`.** Keep GGUFs two levels down. The
retired `llm_engine` router advertised every `.gguf` one level deep as a *chat*
model, which is how a Qwen-Image transformer once appeared in the chat roster
named "gguf".

## Four answers to "what models does this have", and they disagree

`scripts/list-models.sh` already distinguishes three; the fourth is the one
whose absence let `models.txt` rot unnoticed for months.

| Term | Means |
|---|---|
| **DECLARED** | listed in the provisioning manifest — what a fresh machine would fetch |
| **CODE-REFERENCED** | named by `CORE_MODELS`, `BRAINS`, `GGUF_COMPANION_REPOS`, `cutout.py`, `audio_engine.py`, `qwen_edit.py`, `tasks.py` — what the code can actually ask for |
| **ON DISK** | present under `MODELS_DIR` right now |
| **SERVED** | advertised by a running process this minute |

**Rule:** never say "the models" unqualified in a provisioning or deploy
context. Drift between the first two is the bug class — on 2026-10-13 both
openpose ControlNets and `stabilityai/sdxl-turbo` were CODE-REFERENCED and not
ON DISK (so pose-conditioned step 2 could not run), while a 1.8 GB
`Qwen2.5-3B` GGUF was ON DISK and referenced by nothing.

## Activity, never "Actions"

The UI tab listing every generation (API, job queue, browser) is **Activity**.
Owners call it "the actions tab"; translate, do not rename. `actions.py` and
`action_prompts.json` already mean animation actions (walk, idle, attack), and
`frontend/src/tabs/Activity.tsx` explains the choice. It reads the
`generations` ledger through `activity_v`.

## "Source" on the Activity tab - who asked, not what ran it

Fourth in the series after "core", "task" and "map" - the three this file's own
notes already count as lost arguments - and the one that has produced three
defects rather than one wrong change.

**Source** is the **identity of the caller**: `api` (an API key), `internal`
(work this machine asked itself for - benches, `internal:worlds`,
`internal:regions`, `internal:audio_styles`), `unknown` (rows that predate
attribution and can never acquire it).

**Lane** is a different axis and the one most people mean when they say
"source": whether a `jobs` row owns the work. `job` when `job_id IS NOT NULL`,
else `sync`. This is the **Job** sense from the table above - a `jobs` row plus
its 202/poll contract - so do not invent a third word for it.

**What `source` meant before [decisions/0017](decisions/0017-activity-source-is-who-asked.md),
and still means in any `activity_v` older than it:** which arm of the UNION the
row came from, i.e. **which table owns the artefact**. That is not a property of
the caller, which is why `ui` could not mean "this browser" - the browser holds
an API key, writes a ledger row, and therefore matched the `api` arm. The `ui`
arm matched only `sprite_images` rows with no ledger row, which since migration
018 means pre-018 history: 264 rows, newest 2026-09-04, a count that can never
grow.

**Requested by** is the API key *name*, free text an operator typed when minting
it. It is the only identity that separates callers on this network - the address
cannot, because every external request arrives as the docker bridge gateway
(`generations.describe_caller` records why). So "who asked" is answerable at key
granularity and no finer, and `api_keys` carries no type column: nothing
structurally distinguishes the browser's key from a consumer's.


## "Finding" is already taken - use *claim* for the cross-agent log

Reserved before the cross-agent log is built, because this repo has lost this
argument three times already ("core", "task", "map").

**"Finding" currently means three different things here**, none of them the new
one:

| Sense | Where |
|---|---|
| A discovery recorded in prose | `decisions/0004`, `0005`, `0006`, `0008` - "### The finding", "the real finding" |
| One row of an audit verdict | `audit-character-refs.py` - "40 finding(s) matched no live row" |
| A whole document | `.ai/specs/entity-cutout/findings.md` |

So the shared cross-agent record is a **claim**, never a finding. A claim is an
assertion *with provenance* - author, timestamp, and a status - not an
established fact, and the vocabulary should keep saying so:

- **`measured`** - carries evidence: a command and its output, a log line with a
  timestamp, or a `file:line`. **Evidence is required to write this status.**
- **`believed`** - a lead. Reasonable, unverified, and to be treated as
  something to check rather than something to build on.
- **`retracted`** - withdrawn, superseding an earlier claim by id.

**Why the distinction is load-bearing, and not bureaucracy.** On 2026-09-04 two
sessions debugging the same GPU fault produced four wrong theories between them,
each stated confidently. What made the investigation converge was that claims
arrived with their evidence attached and wrong ones were retracted explicitly -
including two of this project's own, recorded in `project-context.md`. A shared
log without that distinction just distributes confident guesses faster, and one
of the intended writers is an 8B local model.

**Do not say "the claim log found X".** A claim log holds claims; people and
measurements find things.
