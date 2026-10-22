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
SCORES = "scores.json"
log = logging.getLogger("build-sheet")

# Build the western rows by flipping the eastern ones instead of rendering
# them. SHEET_MIRROR_WEST=0 renders all eight.
#
# WHY THIS IS ON. The action stage cannot produce a correct west-facing cell:
# `left side` returns the same facing as `right side`, and the mirrored-SOURCE
# trick that fixes the turnaround was measured at this stage and did not work
# (qwen_edit, AZIMUTH_HINTS; all four variants came back facing right). So the
# choice is between a flipped row and a wrong one.
#
# The cost is handedness: a one-handed sword changes hands between the e row
# and the w row. That was rejected before on exactly those grounds and is
# accepted now because something2 already mirrors e/ne/se in game, so for the
# consumer this changes nothing - and the atlas names every flipped row so a
# consumer that does NOT mirror can see it.
#
# It also pays for the checks: 8 directions x 7 actions x 6 frames is 336
# cells, 5 rendered directions is 210, and the ~126 cells saved are roughly
# what three redraw rounds cost.
MIRROR_WEST = os.environ.get("SHEET_MIRROR_WEST", "1").strip() != "0"

# How many times a failing frame may be redrawn on a fresh seed, and then how
# many whole-row redraws. 2 + 1 is the design brief's cap.
REDRAW_ROUNDS = int(os.environ.get("SHEET_REDRAW_ROUNDS", "2"))
ROW_REDRAW_ROUNDS = int(os.environ.get("SHEET_ROW_REDRAW_ROUNDS", "1"))

# A row is re-rendered whole rather than per frame when at least this many of
# its frames are still failing - that is the case where the frames disagree
# with each other rather than each being individually wrong.
ROW_REDRAW_MIN_FAILS = int(os.environ.get("SHEET_ROW_REDRAW_MIN_FAILS", "2"))


def _cfg_path(work):
    return os.path.join(work, CONFIG)


def _scores_path(work):
    return os.path.join(work, SCORES)


def _embeds_path(work, name):
    return os.path.join(work, f"embeds_{name}.pt")


def _cut_path(work, act, d, f):
    """The cut-out cell: what compose actually lays into the grid."""
    return os.path.join(work, f"cut_{act}_{d}_{f}.png")


def _raw_path(work, act, d, f):
    """The raw render, kept for re-scoring and for the flood-fill fallback."""
    return os.path.join(work, f"cell_{act}_{d}_{f}.png")


def _render_directions(directions):
    """The directions actually rendered. The rest are flips of these."""
    import qwen_edit

    if not MIRROR_WEST:
        return list(directions)
    out = []
    for d in directions:
        src = qwen_edit.MIRRORED_AZIMUTHS.get(d, d)
        if src not in out:
            out.append(src)
    return out


def _mirror_plan(directions):
    """{requested direction: the rendered direction it is flipped from}.

    Only for directions the caller actually asked for - a sheet that requests
    `w` alone renders `e` as a source and must not emit an `e` row for it.
    """
    import qwen_edit

    if not MIRROR_WEST:
        return {}
    return {d: qwen_edit.MIRRORED_AZIMUTHS[d] for d in directions
            if d in qwen_edit.MIRRORED_AZIMUTHS}


def _cell_seed(base_seed: int, key: str, attempt: int) -> int:
    """A reproducible per-frame seed for attempt `attempt` of `key`.

    Derived rather than random so a redraw can be replayed. Attempt 0 is the
    job's own seed, so an unchanged conveyor renders exactly what it rendered
    before this file grew a retry loop.

    A caller-pinned seed IS overridden from attempt 1 on, deliberately and for
    the reason the entity cutout path records: a contact sheet is a property
    of the seed, so retrying the same one gives the same rejection forever.
    """
    if attempt == 0:
        return base_seed
    import zlib
    return (base_seed + attempt * 7919
            + zlib.crc32(key.encode())) % (2 ** 31 - 1)


def _action_plan(cfg, actions=None, frames=None, directions=None):
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
    # `render_directions` when the config has one: the stages after the encode
    # must plan over the rows that were actually rendered, not over the rows
    # the caller asked for, or they look up cells that the mirror plan says
    # will be produced by flipping.
    if directions is None:
        directions = cfg.get("render_directions") or cfg["directions"]
    return [(act, d, f_i, pose)
            for d in directions
            for act in actions
            for f_i, pose in enumerate(action_lib.frames(act, d, frames))]


def _key(act, d, f_i):
    return f"{act}|{d}|{f_i}"


def _load_scores(work, cfg=None) -> dict:
    """The per-frame verdict ledger, created on first use.

    One file rather than a column per check in the job row: the shape is
    per-frame and a sheet has hundreds of frames, and compose has to read it
    from a different process than the one that wrote it.
    """
    path = _scores_path(work)
    if os.path.isfile(path):
        with open(path) as fh:
            return json.load(fh)
    return {"seed": (cfg or {}).get("seed", 0), "cells": {},
            "rounds": [], "engine": None}


def _save_scores(work, scores):
    with open(_scores_path(work), "w") as fh:
        json.dump(scores, fh, indent=2)


def _references(work, cfg) -> dict:
    """One Reference per RENDERED direction, from its turnaround view.

    Built once per stage: each costs a BiRefNet pass, and every frame in the
    row is compared against it.
    """
    import sheet_qc

    out = {}
    for d in (cfg.get("render_directions") or cfg["directions"]):
        ref = sheet_qc.Reference.from_path(os.path.join(work, f"dir_{d}.png"))
        if ref is None:
            log.warning("no dir_%s.png; row %s cannot be checked", d, d)
        out[d] = ref
    return out


def _record(scores, act, d, f_i, attempt, seed, metrics):
    """Append one attempt and keep the best one.

    `best` is the attempt with the lowest `sheet_qc.score`, not the last one:
    a redraw can come back worse, and a conveyor that always keeps the newest
    render would then have made the sheet worse by trying.
    """
    import sheet_qc

    k = _key(act, d, f_i)
    cell = scores["cells"].setdefault(
        k, {"action": act, "direction": d, "frame": f_i,
            "attempts": [], "row_redrawn": False})
    cell["attempts"].append({
        "attempt": attempt, "seed": seed,
        "failed_checks": sheet_qc.failures(metrics),
        "score": round(sheet_qc.score(metrics), 3),
        "metrics": {kk: (round(v, 4) if isinstance(v, float) else v)
                    for kk, v in metrics.items()},
    })
    best = min(cell["attempts"], key=lambda at: at["score"])
    cell["best"] = best["attempt"]
    cell["failed_checks"] = best["failed_checks"]
    cell["passed"] = not best["failed_checks"]
    return best["attempt"] == attempt


def _failing(scores) -> list:
    return [c for c in scores["cells"].values() if not c.get("passed")]


def _by_check(scores) -> dict:
    counts = {}
    for c in _failing(scores):
        for k in c.get("failed_checks", []):
            counts[k] = counts.get(k, 0) + 1
    return counts


def stage_turnaround_encode(a):
    import torch

    import qwen_edit

    os.makedirs(a.work, exist_ok=True)
    directions = [s.strip() for s in a.directions.split(",")]

    # Also render whatever the mirror plan will need as a SOURCE.
    #
    # A sheet that asks for `w` but not `e` still builds its west rows by
    # flipping east ones, and the action stage conditions those on
    # `dir_e.png`. Without this the turnaround writes dir_s and dir_w, and
    # the action stage dies on a missing dir_e - after the turnaround has
    # already been paid for.
    turn = list(directions)
    for src in _render_directions(directions):
        if src not in turn:
            turn.append(src)
    if len(turn) > len(directions):
        log.info("also rendering %s as mirror source(s)",
                 ",".join(turn[len(directions):]))

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
    concept_path = a.concept
    if a.clean_concept:
        import pixelate
        before = concept.size
        concept = pixelate.strip_ground_patch(
            pixelate.key_background(concept, tolerance=10))
        log.info("cleaned concept %s -> %s", before, concept.size)
        concept_path = os.path.join(a.work, "concept_clean.png")
        concept.save(concept_path)

    # turnaround_cells, not a comprehension over angle_prompt: the western
    # azimuths are rendered from a MIRRORED concept and flipped back, because
    # `left side` returns the same facing as `right side` (qwen_edit,
    # MIRRORED_AZIMUTHS). The denoise stage must build its cells the same way
    # or the embeddings will not match the images they were encoded against.
    cells = qwen_edit.turnaround_cells(concept, turn)
    torch.save(qwen_edit.encode_cells(cells), _embeds_path(a.work, "turn"))

    with open(_cfg_path(a.work), "w") as f:
        # The CLEANED concept when there is one, not a.concept.
        #
        # This stage encoded against the cleaned image and recorded the raw
        # path, so the denoise stage reloaded the raw one - and an embedding
        # belongs to the (prompt, IMAGE) pair, so every direction was denoised
        # against conditioning its embedding had never seen. Silent: the
        # output still looks like a turnaround, just a worse one. Found while
        # wiring MIRRORED_AZIMUTHS, which depends on the two stages agreeing
        # about the image.
        # `directions` stays what the CALLER asked for - it is what compose
        # emits as rows. `turnaround_directions` is the superset that was
        # actually rendered, and is what the denoise stage must rebuild its
        # cells from: an embedding belongs to the (prompt, image) pair, so
        # denoising a different set than was encoded pairs them wrongly.
        json.dump({"concept": concept_path, "directions": directions,
                   "turnaround_directions": turn,
                   "size": a.size, "seed": a.seed}, f, indent=2)
    log.info("encoded %d direction prompt(s)", len(cells))


def stage_turnaround_denoise(a):
    import torch

    import qwen_edit

    with open(_cfg_path(a.work)) as f:
        cfg = json.load(f)
    concept = Image.open(cfg["concept"])
    # Must mirror the same cells the encode stage did, and the same SET of
    # them - an embedding belongs to the (prompt, IMAGE) pair, so feeding the
    # unmirrored concept here, or a different direction list, would pair an
    # embedding with the wrong image.
    cells = qwen_edit.turnaround_cells(
        concept, cfg.get("turnaround_directions") or cfg["directions"])

    # angles LoRA at FULL strength here: between directions the body must not
    # move. The action stage deliberately lowers it.
    views = qwen_edit.denoise_cells(
        cells, torch.load(_embeds_path(a.work, "turn")),
        seed=cfg["seed"], size=cfg["size"], with_angles=True, angles_scale=1.0)

    # Flip the mirror-sourced directions back exactly once, here, so that
    # everything downstream - the action stage, compose, the atlas - sees a
    # normal turnaround and needs to know nothing about this.
    views = qwen_edit.unmirror_views(views)

    for d, img in views.items():
        img.save(os.path.join(a.work, f"dir_{d}.png"))
    log.info("wrote %d direction image(s) to %s", len(views), a.work)


def _action_cell(qwen_edit, views, act, d, f_i, pose):
    """One action cell.

    THE WESTERN AZIMUTHS ARE NOT RENDERED HERE ANY MORE, and that is how the
    facing bug is answered rather than fixed. `left side` returns the same
    facing as `right side`; the mirrored-SOURCE trick that fixes the
    turnaround was measured at this stage and did NOT work - feeding
    `flip(views[d])` with the eastern camera and `qwen_edit.swap_lr(pose)`,
    then flipping back, still produced a right-facing w cell, as did dropping
    the camera clause and as did swapping it. All four variants came back
    facing right.

    So compose flips the finished e/ne/se rows instead (MIRROR_WEST). The
    cost is handedness, which the atlas names on every flipped row.

    What is still open is WHY. The remaining hypothesis is that the POSE text
    sets the facing: the side family's frame 0 is "left leg forward heel down,
    right arm forward", and the renders put the arm out to screen-right
    whatever the camera says. A front-family pose did not break the facing in
    the same harness. That run also used angles_scale=1.0 by mistake, so it is
    suggestive and not evidence.
    """
    return {
        "key": f"{act}|{d}|{f_i}",
        "image": views[d],
        # The camera prompt is carried into EVERY action prompt. Without it the
        # model reverts to its own framing and the cells stop matching.
        "prompt": f"{qwen_edit.angle_prompt(d)}, the character is {pose}",
        "action": act, "direction": d, "frame": f_i,
    }


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
    render = _render_directions(directions)
    mirrored = _mirror_plan(directions)
    if mirrored:
        log.info("rendering %s; %s will be flipped from %s",
                 ",".join(render),
                 ",".join(sorted(mirrored)),
                 ",".join(mirrored[d] for d in sorted(mirrored)))

    views = {d: Image.open(os.path.join(a.work, f"dir_{d}.png"))
             for d in render}
    plan = _action_plan({**cfg, "actions": actions}, frames=a.frames,
                        directions=render)
    cells = [_action_cell(qwen_edit, views, act, d, f_i, pose)
             for act, d, f_i, pose in plan]

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
                "frames_requested": a.frames,
                "render_directions": render, "mirrored": mirrored})
    with open(_cfg_path(a.work), "w") as f:
        json.dump(cfg, f, indent=2)
    log.info("encoded %d action prompt(s)", len(cells))


def stage_actions_denoise(a):
    import torch

    import qwen_edit
    import sheet_qc

    with open(_cfg_path(a.work)) as f:
        cfg = json.load(f)

    views = {d: Image.open(os.path.join(a.work, f"dir_{d}.png"))
             for d in (cfg.get("render_directions") or cfg["directions"])}
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
    # How much this saves depends entirely on the library, so it is counted at
    # runtime and logged rather than assumed. Against the library as it stands
    # on 2026-08-25, all seven actions across eight directions:
    #
    #     planned 256, rendered 234, copied 22  (~8.6%, about 12 min)
    #
    #     idle   1 repeat in 4, every direction   (A-B-A-C breathing cycle)
    #     sway   1 repeat in 4, every direction   (leans through centre twice)
    #     walk   1 repeat in 6, front and back only; the profile cycle is clean
    #     attack, cast, damage, use   no repeats
    #
    # Do not read those numbers as fixed. An earlier draft of this comment said
    # walk repeated a pose in four and was already wrong by the time it was
    # committed - the walk cycle had been rewritten to six distinct poses in
    # the JSON, which is the whole point of the table being data.
    #
    # This dedupes the RENDER, not the sheet: every frame still gets its own
    # file and the grid still has its full width. A repeated frame is a copy
    # rather than a recomputation.
    groups = {}
    for act, d, f_i, pose in plan:
        groups.setdefault((d, pose), []).append((act, d, f_i))

    # (cell, members) side by side, so the copy step needs no second lookup.
    work = []
    for (d, pose), members in groups.items():
        act, _, f_i = members[0]
        # Rebuilt through _action_cell, not by hand: the western azimuths are
        # conditioned on a MIRRORED view, and an embedding belongs to the
        # (prompt, IMAGE) pair - pairing the encode stage's mirrored embedding
        # with an unmirrored image here would silently denoise against the
        # wrong conditioning.
        work.append((_action_cell(qwen_edit, views, act, d, f_i, pose),
                     members))

    cells = [c for c, _ in work]
    saved = len(plan) - len(cells)
    if saved:
        log.info("%d distinct cell(s) to render; %d repeated pose(s) will be "
                 "copied instead of denoised", len(cells), saved)

    rendered = qwen_edit.denoise_cells(
        cells, torch.load(_embeds_path(a.work, "act")), seed=cfg["seed"],
        size=cfg["size"], with_angles=True, angles_scale=a.angles_scale)

    # No flip here: _action_cell does not mirror. compose flips whole rows.
    scores = _load_scores(a.work, cfg)
    refs = _references(a.work, cfg)
    written = 0
    for c, members in work:
        img = rendered[c["key"]]
        act0, d0, _ = members[0]
        # Scored ONCE per rendered image. A repeated pose is a copy of the
        # same pixels, so scoring each copy would spend ~1.8 s of GPU to
        # arrive at a number already in hand.
        metrics, cut = sheet_qc.analyse(img, refs.get(d0))
        for act, d, f_i in members:
            img.save(_raw_path(a.work, act, d, f_i))
            cut.save(_cut_path(a.work, act, d, f_i))
            _record(scores, act, d, f_i, attempt=0,
                    seed=cfg["seed"], metrics=metrics)
            written += 1
    _save_scores(a.work, scores)

    bad = _failing(scores)
    log.info("wrote %d cell(s) to %s; %d flagged %s", written, a.work,
             len(bad), _by_check(scores) or "")
    # `denoised N/M` is what job_runner parses for the progress bar; the
    # redraw stages print their own, so nothing here has to change for the
    # percentage to keep moving.


def _redraw_targets(scores, cfg, whole_rows: bool):
    """Which (action, direction, frame) to re-render this round.

    Per frame by default. `whole_rows` instead re-renders every frame of any
    row with ROW_REDRAW_MIN_FAILS or more failures still outstanding - the
    case where the frames disagree with EACH OTHER, which re-rolling them
    independently cannot fix because each new frame is rolled against the same
    disagreeing neighbours.
    """
    bad = _failing(scores)
    if not whole_rows:
        return [(c["action"], c["direction"], c["frame"]) for c in bad]

    rows = {}
    for c in bad:
        rows.setdefault((c["action"], c["direction"]), 0)
        rows[(c["action"], c["direction"])] += 1
    hot = {r for r, n in rows.items() if n >= ROW_REDRAW_MIN_FAILS}
    return [(act, d, f) for (act, d) in sorted(hot)
            for f in range(cfg["frames"])]


# Condition each cell on its neighbours as well as on its direction view.
#
# OFF until measured. Qwen-Image-Edit-2511 takes several reference images, so
# this is expressible (qwen_edit._as_sources), but every extra image is extra
# vision tokens in the text encoder's sequence and that sequence is what this
# module's OOM history is about. scripts/bench-frame-context.py reports the
# cost per extra image on this card; turn it on when that says it fits.
FRAME_CONTEXT = os.environ.get("SHEET_FRAME_CONTEXT", "0").strip() != "0"


def _context_images(work, cfg, act, d, f_i, views):
    """The conditioning images for one redrawn frame.

    The direction view first - it is the template the whole row was built on
    and the thing the camera clause holds framing against - then the
    surviving neighbour frames either side, newest on disk.

    NOT the strict sequential version the design brief describes ("adjacent
    marked frames redrawn in order, each using the other's NEW version").
    That cannot be built on this hardware at a sane cost: an embedding belongs
    to the (prompt, image) pair, so a frame conditioned on a neighbour that
    was redrawn in the same pass needs its OWN encode after that neighbour
    exists - and the text encoder and the transformer cannot be resident
    together on 12 GB. Honouring it literally would cost one text-encoder load
    AND one transformer load per frame, about 180 s each, or 18 minutes of
    pure model loading for a six-frame row. Neighbours are therefore read as
    they stand at the start of the pass, which needs one encode and one
    denoise for the whole set.
    """
    imgs = [views[d]]
    if not FRAME_CONTEXT:
        return imgs[0]
    for nb in (f_i - 1, f_i + 1):
        if nb < 0 or nb >= cfg["frames"]:
            continue
        p = _cut_path(work, act, d, nb)
        if os.path.isfile(p):
            imgs.append(Image.open(p).convert("RGB"))
    return imgs if len(imgs) > 1 else imgs[0]


PLAN = "redraw_plan.json"


def _explicit_plan(work, cfg, scores, wanted):
    """Resolve a caller's frame list into a renderable plan.

    Shared by the two explicit-redraw stages so they cannot disagree about
    which frames, which seeds or which context images - an embedding belongs
    to the (prompt, image) pair, and the two stages are separate processes.
    """
    mirrored = cfg.get("mirrored") or {}
    rendered_dirs = cfg.get("render_directions") or cfg["directions"]
    plan = []
    for w in wanted:
        act, d, f_i = w["action"], w["direction"], int(w["frame"])
        # A caller marking a mirrored row means its source: the w row is a
        # flip of e and has no cells of its own to redraw.
        d = mirrored.get(d, d)
        if d not in rendered_dirs:
            log.warning("direction %r was never rendered; skipping", d)
            continue
        poses = action_lib.frames(act, d, cfg["frames"])
        if f_i >= len(poses):
            log.warning("%s %s has no frame %d; skipping", act, d, f_i)
            continue
        key = _key(act, d, f_i)
        attempt = len(scores["cells"].get(key, {}).get("attempts", []))
        plan.append({"action": act, "direction": d, "frame": f_i,
                     "pose": poses[f_i], "attempt": attempt,
                     "seed": _cell_seed(cfg["seed"], key, attempt or 1)})
    return plan


def _explicit_cells(qwen_edit, work, cfg, plan):
    views = {d: Image.open(os.path.join(work, f"dir_{d}.png"))
             for d in (cfg.get("render_directions") or cfg["directions"])}
    cells = []
    for p in plan:
        c = _action_cell(qwen_edit, views, p["action"], p["direction"],
                         p["frame"], p["pose"])
        c["image"] = _context_images(work, cfg, p["action"], p["direction"],
                                     p["frame"], views)
        c["seed"] = p["seed"]
        cells.append(c)
    return cells


def stage_explicit_encode(a):
    """Embed the frames a caller named. Its own process, see below.

    Unlike the automatic rounds this re-encodes, because the conditioning
    changes: a named frame is given its neighbours as context, and an
    embedding belongs to the (prompt, image) pair.

    SEPARATE FROM THE DENOISE for the reason every other encode/denoise pair
    in this file is: the text encoder and the transformer must not be loaded
    in one process on a 12 GB card, and loading-then-releasing a large
    quantised model fragments the allocator enough to fail a later load with
    gigabytes nominally free.
    """
    import torch

    import qwen_edit

    with open(_cfg_path(a.work)) as f:
        cfg = json.load(f)
    with open(a.frames_json) as f:
        wanted = json.load(f)

    plan = _explicit_plan(a.work, cfg, _load_scores(a.work, cfg), wanted)
    if not plan:
        raise SystemExit("no redrawable frames in the request")

    cells = _explicit_cells(qwen_edit, a.work, cfg, plan)
    log.info("explicit redraw: embedding %d cell(s), context=%s",
             len(cells), "on" if FRAME_CONTEXT else "off")
    torch.save(qwen_edit.encode_cells(cells), _embeds_path(a.work, "redraw"))
    with open(os.path.join(a.work, PLAN), "w") as f:
        json.dump(plan, f, indent=2)


def stage_explicit_denoise(a):
    """Render the frames the explicit encode stage embedded.

    Only the named frames' files change. Everything else on disk is untouched,
    which is what makes "replacing one frame and reassembling changes only
    that cell" true.
    """
    import torch

    import qwen_edit
    import sheet_qc

    with open(_cfg_path(a.work)) as f:
        cfg = json.load(f)
    with open(os.path.join(a.work, PLAN)) as f:
        plan = json.load(f)

    scores = _load_scores(a.work, cfg)
    refs = _references(a.work, cfg)
    cells = _explicit_cells(qwen_edit, a.work, cfg, plan)

    rendered = qwen_edit.denoise_cells(
        cells, torch.load(_embeds_path(a.work, "redraw")), seed=cfg["seed"],
        size=cfg["size"], with_angles=True, angles_scale=a.angles_scale)

    for c in cells:
        act, d, f_i = c["action"], c["direction"], c["frame"]
        img = rendered[c["key"]]
        metrics, cut = sheet_qc.analyse(img, refs.get(d))
        attempt = len(scores["cells"].get(_key(act, d, f_i), {})
                      .get("attempts", []))
        # An explicitly requested redraw REPLACES the frame even if it scores
        # worse. The caller looked at the cell and said it was wrong; a
        # verdict from five numbers does not get to overrule that. The score
        # is still recorded, so the atlas says what the new cell measures.
        img.save(_raw_path(a.work, act, d, f_i))
        cut.save(_cut_path(a.work, act, d, f_i))
        _record(scores, act, d, f_i, attempt, c["seed"], metrics)
        cell = scores["cells"][_key(act, d, f_i)]
        cell["best"] = attempt
        cell["requested"] = True
        last = cell["attempts"][-1]
        cell["failed_checks"] = last["failed_checks"]
        cell["passed"] = not last["failed_checks"]

    _save_scores(a.work, scores)
    log.info("explicit redraw done; %d frame(s) still failing overall",
             len(_failing(scores)))


def stage_actions_redraw(a):
    """Re-render the frames that failed, on fresh seeds, and keep the best.

    WHY THIS IS A SEPARATE PASS AND NOT A PER-ROW LOOP

    The design brief asks for row-by-row work: draw a row, check it, redraw
    what failed, move on. That shape cannot be built on this hardware. A cell
    is checked from its DECODED pixels, and `denoise_cells` cannot decode
    while the transformer is resident - the VAE does not fit beside it on a
    12 GB card, which is why it renders every latent first and only then
    releases the transformer to decode. Checking after each row would mean
    reloading the transformer per row: ~90 s x 35 rows, about an hour of pure
    model loading added to a job.

    So the loop is batched instead - render everything, check everything,
    re-render the failures together. The outcome is the same (every frame is
    checked, failures are redrawn, retries are capped) at three model loads
    instead of thirty-five.

    THE EMBEDDINGS ARE REUSED, WHICH IS WHY THIS IS CHEAP

    An embedding belongs to the (prompt, image) pair. A seed re-roll changes
    NEITHER, so the text encoder does not have to run again and this stage
    skips straight to the transformer. That is the difference between a
    redraw round costing ~90 s of loading and costing ~180 s.
    """
    import torch

    import qwen_edit
    import sheet_qc

    with open(_cfg_path(a.work)) as f:
        cfg = json.load(f)
    scores = _load_scores(a.work, cfg)

    targets = _redraw_targets(scores, cfg, whole_rows=a.whole_rows)
    if not targets:
        log.info("redraw round %d: nothing failing, skipping", a.round)
        return

    # Cap attempts per frame. Without this a frame that no seed can fix burns
    # every round, and the rounds exist to be spent on frames that CAN be.
    limit = REDRAW_ROUNDS + ROW_REDRAW_ROUNDS + 1
    targets = [t for t in targets
               if len(scores["cells"].get(_key(*t), {}).get("attempts", []))
               < limit]
    if not targets:
        log.info("redraw round %d: every failing frame is out of attempts",
                 a.round)
        return

    views = {d: Image.open(os.path.join(a.work, f"dir_{d}.png"))
             for d in (cfg.get("render_directions") or cfg["directions"])}
    refs = _references(a.work, cfg)
    embeds = torch.load(_embeds_path(a.work, "act"))

    # One seed for the whole row on a row redraw, a per-frame seed otherwise.
    # A row has to be re-rolled together or its frames come from different
    # draws and disagree with each other exactly as before.
    row_seed = {}
    cells = []
    for act, d, f_i in targets:
        poses = action_lib.frames(act, d, cfg["frames"])
        if f_i >= len(poses):
            continue
        key = _key(act, d, f_i)
        if key not in embeds:
            log.warning("no embedding for %s; it was never encoded", key)
            continue
        if a.whole_rows:
            seed = row_seed.setdefault(
                (act, d), _cell_seed(cfg["seed"], f"{act}|{d}", a.round))
        else:
            seed = _cell_seed(cfg["seed"], key, a.round)
        c = _action_cell(qwen_edit, views, act, d, f_i, poses[f_i])
        c["seed"] = seed
        cells.append(c)

    if not cells:
        log.info("redraw round %d: nothing renderable", a.round)
        return

    log.info("redraw round %d: %d cell(s)%s", a.round, len(cells),
             " (whole rows)" if a.whole_rows else "")
    rendered = qwen_edit.denoise_cells(
        cells, embeds, seed=cfg["seed"], size=cfg["size"],
        with_angles=True, angles_scale=a.angles_scale)

    kept = 0
    for c in cells:
        act, d, f_i = c["action"], c["direction"], c["frame"]
        img = rendered[c["key"]]
        metrics, cut = sheet_qc.analyse(img, refs.get(d))
        attempt = len(scores["cells"].get(_key(act, d, f_i), {})
                      .get("attempts", []))
        # Every attempt is kept on disk under its own name. Without that a
        # redraw that came back worse has already overwritten the better cell
        # by the time the score says so.
        img.save(os.path.join(a.work, f"att{attempt}_{act}_{d}_{f_i}.png"))
        if _record(scores, act, d, f_i, attempt, c["seed"], metrics):
            img.save(_raw_path(a.work, act, d, f_i))
            cut.save(_cut_path(a.work, act, d, f_i))
            kept += 1
        if a.whole_rows:
            scores["cells"][_key(act, d, f_i)]["row_redrawn"] = True

    scores["rounds"].append({"round": a.round, "cells": len(cells),
                             "kept": kept, "whole_rows": a.whole_rows})
    scores["engine"] = sheet_qc.engine()
    _save_scores(a.work, scores)
    log.info("redraw round %d: %d/%d improved; %d frame(s) still failing %s",
             a.round, kept, len(cells), len(_failing(scores)),
             _by_check(scores) or "")


def stage_compose(a):
    # No GPU, no model - pure PIL/numpy, so this stage can be re-run freely
    # while tuning cell size, palette or keying without regenerating anything.
    import pixelate
    import sheet as sheet_mod

    with open(_cfg_path(a.work)) as f:
        cfg = json.load(f)
    cw, ch = (int(x) for x in a.cell.lower().split("x"))

    plan = _action_plan(cfg)
    mirrored = cfg.get("mirrored") or {}
    # Grid width is the widest row actually rendered. Taking it from the
    # request instead would append empty columns for frames no stage produced,
    # and something2 rejects a sheet whose cells do not fill it.
    grid_frames = max((f_i for _, _, f_i, _ in plan), default=-1) + 1
    # Pin the scale the FIRST compose settled on. A redraw re-runs this stage
    # over a set of cells where one has changed, and without the pin a
    # slightly wider repaired frame would rescale all 336 cells - so repairing
    # one sprite would move every sprite in the game.
    b = sheet_mod.SheetBuilder(cell=(cw, ch), frames=grid_frames,
                               directions=cfg["directions"],
                               n_colors=a.colors, mirrored=mirrored,
                               pinned_scale=(None if a.refit
                                             else cfg.get("sheet_scale")))

    rendered = {}
    cut_used = 0
    for act, d, f_i, _ in plan:
        cut = _cut_path(a.work, act, d, f_i)
        raw = _raw_path(a.work, act, d, f_i)
        if os.path.isfile(cut):
            # Already segmented and already hard-alpha. No keying, and no
            # ground strip either: the flood fill left a ground patch because
            # it can only clear what is connected to the border, and the
            # segmenter does not produce one. Running the width rule over a
            # clean cutout can only take feet off.
            cell = Image.open(cut)
            cut_used += 1
        elif os.path.isfile(raw):
            # The pre-segmenter path, kept so an old work directory still
            # composes and so a box without BiRefNet weights still builds.
            cell = pixelate.key_background(Image.open(raw),
                                           tolerance=a.key_tolerance)
            if a.strip_ground:
                # Safe on action frames now, which it was NOT under the old
                # rule. That compared the patch against the body median, so a
                # wide stance looked like ground; the shin reference moves
                # with the legs, so a wide stance raises the threshold instead
                # of tripping it.
                cell = pixelate.strip_ground_patch(
                    cell, width_ratio=a.ground_ratio)
        else:
            log.warning("missing %s", raw)
            continue
        rendered[(act, d, f_i)] = cell
        # Added even when `d` is only a mirror source and not a requested
        # row - SheetBuilder fits it so the flipped row can be derived, and
        # drops it before the grid.
        b.add(act, d, f_i, cell)

    # The flipped rows are NOT added here. SheetBuilder produces them in
    # cell_images, from the FITTED cell, because flipping the 512px render
    # and fitting it separately lands the sprite a pixel off (the centring
    # divides by two and rounds the same way for both) - which in game is a
    # sprite that jumps sideways when the character turns around.

    gaps = b.missing()
    if gaps:
        log.warning("%d cell(s) missing, emitting as holes: %s",
                    len(gaps), gaps[:5])

    concept = pixelate.key_background(Image.open(cfg["concept"]),
                                      tolerance=a.key_tolerance)
    frames_root = os.path.join(a.work, "frames")
    out, atlas = b.save(a.dst, concept=concept, allow_missing=True,
                        frames_root=frames_root,
                        quality=_quality(a.work, cfg, mirrored))
    # Read BEFORE recording, so the first compose does not report its own
    # freshly written value as a pin it honoured.
    was_pinned = bool(cfg.get("sheet_scale")) and not a.refit

    # Record the scale so the next compose over this work directory - which
    # is what a redraw runs - reproduces this sheet's sizing exactly.
    if atlas.get("scale") and not cfg.get("sheet_scale"):
        cfg["sheet_scale"] = atlas["scale"]
        with open(_cfg_path(a.work), "w") as fh:
            json.dump(cfg, fh, indent=2)

    log.info("wrote %s (%dx%d) from %d cut-out and %d keyed cell(s), "
             "scale %.5f%s", a.dst, out.width, out.height, cut_used,
             len(rendered) - cut_used, atlas.get("scale") or 0.0,
             " (pinned)" if was_pinned and not atlas.get("refit") else "")
    if atlas.get("refit"):
        print("refit: a redrawn frame was wider than the original build, so "
              "every cell was rescaled")
    print(json.dumps(atlas["something2"], indent=2))
    print(f"rows: {[(r['action'], r['direction']) for r in atlas['rows']]}")
    q = atlas.get("quality") or []
    failed = [e for e in q if not e["passed"]]
    print(f"quality: {len(q)} frame(s) checked, {len(failed)} still failing")
    for e in failed[:12]:
        print(f"  {e['action']} {e['direction']} f{e['frame']}: "
              f"{'+'.join(e['failed_checks'])} after {e['attempts']} attempt(s)"
              + (f" (mirrored from {e['mirrored_from']})"
                 if e.get("mirrored_from") else ""))


def _quality(work, cfg, mirrored) -> list:
    """The per-frame verdict list that goes into the atlas.

    Includes the flipped rows, carrying the verdict of the row they were
    flipped from. A consumer asking "is attack w f3 good" must get an answer,
    and the honest answer for a mirrored row is whatever its source scored.

    Silence is not an option here: a frame that never passed has to appear in
    the job result, or the conveyor has quietly shipped a defect it knew
    about.
    """
    scores = _load_scores(work, cfg)
    # Only the directions this sheet actually emits as rows. A mirror source
    # can be rendered and scored without being a row - a sheet may ask for
    # `w` and not `e` - and reporting quality for a row the consumer cannot
    # fetch is worse than reporting nothing: it invites a redraw request for
    # a cell that is not in the sheet.
    emitted = set(cfg["directions"])
    out = []
    for c in scores["cells"].values():
        entry = {
            "action": c["action"], "direction": c["direction"],
            "frame": c["frame"], "attempts": len(c.get("attempts", [])),
            "passed": bool(c.get("passed")),
            "failed_checks": c.get("failed_checks", []),
            "row_redrawn": bool(c.get("row_redrawn")),
        }
        if c["direction"] in emitted:
            out.append(entry)
        # `entry`, not out[-1]: today each source maps to exactly one target,
        # so out[-1] happens to be right, and it would silently start copying
        # the previous MIRRORED entry the moment that stopped being true.
        for target, src in mirrored.items():
            if src == c["direction"] and target in emitted:
                out.append({**entry, "direction": target,
                            "mirrored_from": src})
    return sorted(out, key=lambda e: (e["action"], e["direction"],
                                      e["frame"]))


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

    rd = sub.add_parser("actions-redraw")
    rd.add_argument("--round", type=int, default=1)
    rd.add_argument("--whole-rows", action="store_true",
                    help="re-render every frame of a row with several "
                         "failures, on one shared seed, instead of "
                         "re-rolling its frames independently")
    rd.add_argument("--angles-scale", type=float, default=0.4)

    xe = sub.add_parser("explicit-encode")
    xe.add_argument("--frames-json", required=True,
                    help="path to a JSON list of {action, direction, frame}")

    xr = sub.add_parser("explicit-denoise")
    xr.add_argument("--angles-scale", type=float, default=0.4)

    co = sub.add_parser("compose")
    co.add_argument("dst")
    co.add_argument("--cell", default="48x64")
    co.add_argument("--colors", type=int, default=24)
    co.add_argument("--key-tolerance", type=int, default=10)
    co.add_argument("--no-strip-ground", dest="strip_ground",
                    action="store_false", default=True,
                    help="keep the ground/shadow patch under each sprite")
    co.add_argument("--refit", action="store_true",
                    help="recompute the sheet-wide scale instead of reusing "
                         "the one this sheet was first built at. Every cell "
                         "changes size, so only for a deliberate rebuild")
    co.add_argument("--ground-ratio", type=float, default=1.8,
                    help="how much wider than the SHINS a row must be to count "
                         "as ground")

    # EVERY subparser, including the redraw ones. They were added without it
    # and died on "unrecognized arguments: --work" the first time the
    # conveyor reached them - after the whole action pass had been paid for.
    for sp in (te, td, ae, ad, rd, xe, xr, co):
        sp.add_argument("--work", default="/app/images/_work")

    a = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    {"turnaround-encode": stage_turnaround_encode,
     "turnaround-denoise": stage_turnaround_denoise,
     "actions-encode": stage_actions,
     "actions-denoise": stage_actions_denoise,
     "actions-redraw": stage_actions_redraw,
     "explicit-encode": stage_explicit_encode,
     "explicit-denoise": stage_explicit_denoise,
     "compose": stage_compose}[a.stage](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
