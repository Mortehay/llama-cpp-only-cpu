#!/usr/bin/env python3
"""Concept image -> finished multi-action, multi-direction sprite sheet.

Run as THREE separate invocations, not one:

    build-sheet.py turnaround <concept.png> --work images/_work
    build-sheet.py actions               --work images/_work
    build-sheet.py compose  <out.png>    --work images/_work

WHY THREE PROCESSES AND NOT THREE FUNCTIONS

Loading and releasing the 7.1 GB GGUF transformer twice inside one process
fragments the CUDA allocator past the point of usefulness. Measured 2026-08-23:
the second pass failed on a **2.00 MiB** allocation with **3.42 GiB free** -
there was plenty of memory, just none of it contiguous. Explicitly deleting the
latents, the VAE and the pipeline between passes reduced the failure (72 MiB ->
2 MiB) but did not remove it, because the arena itself is what is fragmented.

A fresh process gets a fresh allocator, and the transformer was being loaded
twice either way, so the only real cost is ~10 s of torch import per stage.

Splitting it also makes each stage resumable and its output inspectable on
disk - which matters when a full character is an hour of GPU time and a crash
in compose should not throw away the generation.
"""

import argparse
import json
import logging
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
for _c in (os.path.join(_here, "..", "src", "sprite_generator"),
           os.path.dirname(_here)):
    if os.path.isfile(os.path.join(_c, "qwen_edit.py")):
        sys.path.insert(0, _c)
        break

from PIL import Image  # noqa: E402

import actions as action_lib  # noqa: E402

CONFIG = "config.json"
log = logging.getLogger("build-sheet")


def _cfg_path(work):
    return os.path.join(work, CONFIG)


def _embeds_path(work, name):
    return os.path.join(work, f"embeds_{name}.pt")


def _action_plan(cfg, actions=None, frames=None):
    """Every (action, direction, frame index, pose) the action stages produce.

    ONE function, called by encode, denoise and compose, because they used to
    derive this list independently and drifted:

        encode   for f_i, pose in enumerate(action_lib.frames(act, d, frames))
        denoise  for f_i in range(cfg["frames"])
        compose  for f_i in range(cfg["frames"])

    `action_lib.frames()` truncates to the poses that exist - four for every
    action today - while `range()` does not. A job asking for six frames
    therefore encoded keys 0..3 and then looked up 0..5, and died eight minutes
    into the GPU pass with `KeyError: 'idle|s|4'`, having already paid for the
    turnaround. Compose would have gone on to emit two empty columns per row.

    Deriving all three from the library also makes a ragged library safe: an
    action with six poses beside one with four now renders six and four,
    instead of over-reading the shorter one.
    """
    actions = cfg["actions"] if actions is None else actions
    frames = cfg["frames"] if frames is None else frames
    return [(act, d, f_i, pose)
            for d in cfg["directions"]
            for act in actions
            for f_i, pose in enumerate(action_lib.frames(act, d, frames))]


def stage_turnaround_encode(a):
    import torch

    import qwen_edit

    os.makedirs(a.work, exist_ok=True)
    directions = [s.strip() for s in a.directions.split(",")]
    concept = Image.open(a.concept)

    # Clean the CONCEPT once, rather than fighting its shadow in every cell.
    #
    # Qwen-Image-Edit preserves what it is given - that is the whole reason it
    # holds identity across a sheet - so a concept with a shadow ellipse baked
    # in propagates that shadow into all ~96 outputs, where the per-cell width
    # rule then has to remove it 96 times and cannot fully (measured on
    # core_21f88cbbe9b4: remnants survive because they are no wider than the
    # character's own legs).
    #
    # One strip on the input is both cheaper and more reliable. It also cannot
    # amputate anything the way a per-cell rule can, because the concept is a
    # single known-good rest pose rather than 96 poses of varying width.
    if a.clean_concept:
        import pixelate
        before = concept.size
        concept = pixelate.strip_ground_patch(
            pixelate.key_background(concept, tolerance=10))
        log.info("cleaned concept %s -> %s", before, concept.size)
        concept.save(os.path.join(a.work, "concept_clean.png"))

    cells = [{"key": d, "image": concept, "prompt": qwen_edit.angle_prompt(d)}
             for d in directions]
    torch.save(qwen_edit.encode_cells(cells), _embeds_path(a.work, "turn"))

    with open(_cfg_path(a.work), "w") as f:
        json.dump({"concept": a.concept, "directions": directions,
                   "size": a.size, "seed": a.seed}, f, indent=2)
    log.info("encoded %d direction prompt(s)", len(cells))


def stage_turnaround_denoise(a):
    import torch

    import qwen_edit

    with open(_cfg_path(a.work)) as f:
        cfg = json.load(f)
    concept = Image.open(cfg["concept"])
    cells = [{"key": d, "image": concept, "prompt": ""}
             for d in cfg["directions"]]

    # angles LoRA at FULL strength here: between directions the body must not
    # move. The action stage deliberately lowers it.
    views = qwen_edit.denoise_cells(
        cells, torch.load(_embeds_path(a.work, "turn")),
        seed=cfg["seed"], size=cfg["size"], with_angles=True, angles_scale=1.0)

    for d, img in views.items():
        img.save(os.path.join(a.work, f"dir_{d}.png"))
    log.info("wrote %d direction image(s) to %s", len(views), a.work)


def stage_actions(a):
    import qwen_edit

    with open(_cfg_path(a.work)) as f:
        cfg = json.load(f)
    directions = cfg["directions"]
    actions = [s.strip() for s in a.actions.split(",")]
    for act in actions:
        if act not in action_lib.ACTIONS:
            raise SystemExit(f"unknown action {act!r}; "
                             f"have {sorted(action_lib.ACTIONS)}")

    # Poses are chosen per VIEW FAMILY: a stride reads in profile and vanishes
    # head-on, where the same walk has to be a knee lift.
    views = {d: Image.open(os.path.join(a.work, f"dir_{d}.png"))
             for d in directions}
    cameras = {d: qwen_edit.angle_prompt(d) for d in directions}
    plan = _action_plan({**cfg, "actions": actions}, frames=a.frames)
    cells = [{
        "key": f"{act}|{d}|{f_i}",
        "image": views[d],
        # The camera prompt is carried into EVERY action prompt. Without it the
        # model reverts to its own framing and the cells stop matching.
        "prompt": f"{cameras[d]}, the character is {pose}",
        "action": act, "direction": d, "frame": f_i,
    } for act, d, f_i, pose in plan]

    import torch
    log.info("actions: %d cell(s)", len(cells))
    torch.save(qwen_edit.encode_cells(cells), _embeds_path(a.work, "act"))

    # Record the count the library ACTUALLY supplied, not the count requested.
    # cfg is what every later stage and the atlas read; storing the request
    # there is what let compose lay out columns nothing ever rendered.
    effective = max((f_i for _, _, f_i, _ in plan), default=-1) + 1
    if effective < a.frames:
        log.warning("requested %d frame(s), pose library supplies %d; "
                    "building at %d", a.frames, effective, effective)
    cfg.update({"actions": actions, "frames": effective,
                "frames_requested": a.frames})
    with open(_cfg_path(a.work), "w") as f:
        json.dump(cfg, f, indent=2)
    log.info("encoded %d action prompt(s)", len(cells))


def stage_actions_denoise(a):
    import torch

    import qwen_edit

    with open(_cfg_path(a.work)) as f:
        cfg = json.load(f)

    views = {d: Image.open(os.path.join(a.work, f"dir_{d}.png"))
             for d in cfg["directions"]}
    plan = _action_plan(cfg)

    # Render each DISTINCT (direction, pose) once and copy the result.
    #
    # A cell is a pure function of (input view, prompt, seed): denoise_cells
    # builds a fresh generator from the same cfg["seed"] for every cell, and
    # the view is per-direction, so two cells whose pose text matches produce
    # byte-identical output. Several actions repeat a pose deliberately - a
    # walk passes through the same contact pose twice, an idle breathes
    # A-B-A-C - and those repeats were being denoised a second time to arrive
    # at pixels already on disk.
    #
    # Measured 2026-08-25 on the shipped library: walk, idle and sway each
    # repeat one pose in four, so a quarter of every such row was paid for
    # twice. On an eight-direction walk that is 8 cells, about 4.4 minutes.
    # attack, damage, use and cast repeat nothing and are unaffected.
    #
    # This dedupes the RENDER, not the sheet: every frame still gets its own
    # file and the grid still has its full width. Frame 3 of a walk is a copy
    # rather than a recomputation.
    groups = {}
    for act, d, f_i, pose in plan:
        groups.setdefault((d, pose), []).append((act, d, f_i))

    # (cell, members) side by side, so the copy step needs no second lookup.
    work = []
    for (d, _pose), members in groups.items():
        act, _, f_i = members[0]
        work.append(({"key": f"{act}|{d}|{f_i}", "image": views[d],
                      "prompt": "", "action": act, "direction": d,
                      "frame": f_i}, members))

    cells = [c for c, _ in work]
    saved = len(plan) - len(cells)
    if saved:
        log.info("%d distinct cell(s) to render; %d repeated pose(s) will be "
                 "copied instead of denoised", len(cells), saved)

    rendered = qwen_edit.denoise_cells(
        cells, torch.load(_embeds_path(a.work, "act")), seed=cfg["seed"],
        size=cfg["size"], with_angles=True, angles_scale=a.angles_scale)

    written = 0
    for c, members in work:
        img = rendered[c["key"]]
        for act, d, f_i in members:
            img.save(os.path.join(a.work, f"cell_{act}_{d}_{f_i}.png"))
            written += 1
    log.info("wrote %d cell(s) to %s", written, a.work)


def stage_compose(a):
    # No GPU, no model - pure PIL/numpy, so this stage can be re-run freely
    # while tuning cell size, palette or keying without regenerating anything.
    import pixelate
    import sheet as sheet_mod

    with open(_cfg_path(a.work)) as f:
        cfg = json.load(f)
    cw, ch = (int(x) for x in a.cell.lower().split("x"))

    plan = _action_plan(cfg)
    # Grid width is the widest row actually rendered. Taking it from the
    # request instead would append empty columns for frames no stage produced,
    # and something2 rejects a sheet whose cells do not fill it.
    grid_frames = max((f_i for _, _, f_i, _ in plan), default=-1) + 1
    b = sheet_mod.SheetBuilder(cell=(cw, ch), frames=grid_frames,
                               directions=cfg["directions"],
                               n_colors=a.colors)
    for act, d, f_i, _ in plan:
        path = os.path.join(a.work, f"cell_{act}_{d}_{f_i}.png")
        if not os.path.isfile(path):
            log.warning("missing %s", path)
            continue
        cell = pixelate.key_background(Image.open(path),
                                       tolerance=a.key_tolerance)
        if a.strip_ground:
            # Safe on action frames now, which it was NOT under the old rule.
            # That compared the patch against the body median, so a wide stance
            # looked like ground; the shin reference moves with the legs, so a
            # wide stance raises the threshold instead of tripping it.
            cell = pixelate.strip_ground_patch(
                cell, width_ratio=a.ground_ratio)
        b.add(act, d, f_i, cell)

    gaps = b.missing()
    if gaps:
        log.warning("%d cell(s) missing, emitting as holes: %s",
                    len(gaps), gaps[:5])

    concept = pixelate.key_background(Image.open(cfg["concept"]),
                                      tolerance=a.key_tolerance)
    out, atlas = b.save(a.dst, concept=concept, allow_missing=True)
    log.info("wrote %s (%dx%d)", a.dst, out.width, out.height)
    print(json.dumps(atlas["something2"], indent=2))
    print(f"rows: {[(r['action'], r['direction']) for r in atlas['rows']]}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="stage", required=True)

    te = sub.add_parser("turnaround-encode")
    te.add_argument("concept")
    te.add_argument("--directions", default="s,e,n,w")
    te.add_argument("--size", type=int, default=512)
    te.add_argument("--seed", type=int, default=0)
    te.add_argument("--no-clean-concept", dest="clean_concept",
                    action="store_false", default=True,
                    help="skip keying and ground-stripping the concept. The "
                         "editor preserves what it is given, so a shadow left "
                         "on the concept propagates into every cell")

    td = sub.add_parser("turnaround-denoise")

    ae = sub.add_parser("actions-encode")
    ae.add_argument("--actions", default="walk")
    ae.add_argument("--frames", type=int, default=4)

    ad = sub.add_parser("actions-denoise")
    ad.add_argument("--angles-scale", type=float, default=0.4,
                    help="1.0 pins the pose as hard as the framing and "
                         "flattens the animation; 0.4 is the measured balance")

    co = sub.add_parser("compose")
    co.add_argument("dst")
    co.add_argument("--cell", default="48x64")
    co.add_argument("--colors", type=int, default=24)
    co.add_argument("--key-tolerance", type=int, default=10)
    co.add_argument("--no-strip-ground", dest="strip_ground",
                    action="store_false", default=True,
                    help="keep the ground/shadow patch under each sprite")
    co.add_argument("--ground-ratio", type=float, default=1.8,
                    help="how much wider than the SHINS a row must be to count "
                         "as ground")

    for sp in (te, td, ae, ad, co):
        sp.add_argument("--work", default="/app/images/_work")

    a = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    {"turnaround-encode": stage_turnaround_encode,
     "turnaround-denoise": stage_turnaround_denoise,
     "actions-encode": stage_actions,
     "actions-denoise": stage_actions_denoise,
     "compose": stage_compose}[a.stage](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
