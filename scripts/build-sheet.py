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


def stage_turnaround_encode(a):
    import torch

    import qwen_edit

    os.makedirs(a.work, exist_ok=True)
    directions = [s.strip() for s in a.directions.split(",")]
    concept = Image.open(a.concept)

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

    cells = []
    for d in directions:
        view = Image.open(os.path.join(a.work, f"dir_{d}.png"))
        camera = qwen_edit.angle_prompt(d)
        for act in actions:
            # Poses are chosen per VIEW FAMILY: a stride reads in profile and
            # vanishes head-on, where the same walk has to be a knee lift.
            for f_i, pose in enumerate(
                    action_lib.frames(act, d, a.frames)):
                cells.append({
                    "key": f"{act}|{d}|{f_i}",
                    "image": view,
                    # The camera prompt is carried into EVERY action prompt.
                    # Without it the model reverts to its own framing and the
                    # cells stop matching each other.
                    "prompt": f"{camera}, the character is {pose}",
                    "action": act, "direction": d, "frame": f_i,
                })

    import torch
    log.info("actions: %d cell(s)", len(cells))
    torch.save(qwen_edit.encode_cells(cells), _embeds_path(a.work, "act"))

    cfg.update({"actions": actions, "frames": a.frames})
    with open(_cfg_path(a.work), "w") as f:
        json.dump(cfg, f, indent=2)
    log.info("encoded %d action prompt(s)", len(cells))


def stage_actions_denoise(a):
    import torch

    import qwen_edit

    with open(_cfg_path(a.work)) as f:
        cfg = json.load(f)

    cells = []
    for d in cfg["directions"]:
        view = Image.open(os.path.join(a.work, f"dir_{d}.png"))
        for act in cfg["actions"]:
            for f_i in range(cfg["frames"]):
                cells.append({"key": f"{act}|{d}|{f_i}", "image": view,
                              "prompt": "", "action": act,
                              "direction": d, "frame": f_i})

    rendered = qwen_edit.denoise_cells(
        cells, torch.load(_embeds_path(a.work, "act")), seed=cfg["seed"],
        size=cfg["size"], with_angles=True, angles_scale=a.angles_scale)

    for c in cells:
        name = f"cell_{c['action']}_{c['direction']}_{c['frame']}.png"
        rendered[c["key"]].save(os.path.join(a.work, name))
    log.info("wrote %d cell(s) to %s", len(cells), a.work)


def stage_compose(a):
    # No GPU, no model - pure PIL/numpy, so this stage can be re-run freely
    # while tuning cell size, palette or keying without regenerating anything.
    import pixelate
    import sheet as sheet_mod

    with open(_cfg_path(a.work)) as f:
        cfg = json.load(f)
    cw, ch = (int(x) for x in a.cell.lower().split("x"))

    b = sheet_mod.SheetBuilder(cell=(cw, ch), frames=cfg["frames"],
                               directions=cfg["directions"],
                               n_colors=a.colors)
    for act in cfg["actions"]:
        for d in cfg["directions"]:
            for f_i in range(cfg["frames"]):
                path = os.path.join(a.work, f"cell_{act}_{d}_{f_i}.png")
                if not os.path.isfile(path):
                    log.warning("missing %s", path)
                    continue
                b.add(act, d, f_i,
                      pixelate.key_background(Image.open(path),
                                              tolerance=a.key_tolerance))

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
