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

## Skeleton / control image

**Skeleton** - the COCO-18 keypoint dict authored in `poses.py`.
**Control image** - that skeleton rendered to RGB for ControlNet.

Worth separating because a bug can be in the keypoints (wrong anatomy) or in the
rendering and fitting (right anatomy, landed off the character).

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
