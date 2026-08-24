#!/usr/bin/env python3
"""Generate one action's frames from a single direction image.

Tests whether Qwen-Image-Edit-2511 can pose a character from a plain
instruction, before paying for a pose-transfer LoRA (see qwen_edit.action_frames
for why that matters).

Prompts are kept SHORT on purpose. Stage 1's bitsandbytes dequant fallback
allocates against sequence length, so a verbose instruction is not free here -
a wordy version of this test OOM'd on a 1.02 GiB buffer.

Usage:
    python action-test.py <source.png> <out.png> [--frames 6] [--action walk]
"""

import argparse
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
for _c in (os.path.join(_here, "..", "src", "sprite_generator"),
           os.path.dirname(_here)):
    if os.path.isfile(os.path.join(_c, "qwen_edit.py")):
        sys.path.insert(0, _c)
        break

from PIL import Image  # noqa: E402

import qwen_edit  # noqa: E402

# One entry per action, listing the per-frame instruction.
#
# Written as POSE descriptions, not as motion verbs: "walking" is a process and
# the model has to guess which instant of it to draw, which is how you get four
# frames of the same stance. Naming the limb positions makes each frame a
# different, checkable target.
ACTIONS = {
    "walk": [
        "left leg forward, right arm forward, mid stride",
        "legs together, passing pose, arms at sides",
        "right leg forward, left arm forward, mid stride",
        "legs together, passing pose, arms at sides",
    ],
    "walk6": [
        "left leg far forward, right arm forward",
        "left leg forward, weight settling",
        "legs together, passing pose",
        "right leg far forward, left arm forward",
        "right leg forward, weight settling",
        "legs together, passing pose",
    ],
    "attack": [
        "arms drawn back, winding up to strike",
        "arms swinging forward, mid strike",
        "arms fully extended forward, strike landed",
        "arms lowering, recovering from the strike",
    ],
    "idle": [
        "standing still, arms at sides",
        "standing still, shoulders slightly raised",
        "standing still, arms at sides",
        "standing still, shoulders slightly lowered",
    ],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--action", default="walk", choices=sorted(ACTIONS))
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prefix", default="the character is",
                   help="prepended to every frame instruction")
    p.add_argument("--cell", type=int, default=None,
                   help="crop this cell out of an N-wide turnaround strip "
                        "first (0=s 1=se 2=e 3=ne 4=n 5=nw 6=w 7=sw)")
    p.add_argument("--cells", type=int, default=8,
                   help="how many cells the source strip holds")
    p.add_argument("--angles-scale", type=float, default=0.5,
                   help="angles LoRA weight for action frames. 1.0 pins the "
                        "pose as hard as the framing and flattens the "
                        "animation; 0 frees the pose and loses the camera")
    p.add_argument("--direction", default=None,
                   help="carry this azimuth's camera prompt into every frame "
                        "(s se e ne n nw w sw). Without it the framing drifts")
    a = p.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    # Keep the CAMERA trigger in front of every action instruction.
    #
    # Measured 2026-08-23: sending the pose instruction alone produced correct
    # poses but threw the framing away - the character came back larger, at eye
    # level instead of the elevated iso camera, and at a different scale in
    # every frame. The angle prompt is not just for turnarounds; it is what
    # pins framing, and an action prompt REPLACES it unless it is carried
    # through explicitly.
    camera = (qwen_edit.angle_prompt(a.direction) + ", ") if a.direction else ""
    frames = [f"{camera}{a.prefix} {f}" for f in ACTIONS[a.action]]
    print(f"action '{a.action}': {len(frames)} frames")
    for f in frames:
        print(f"  {f}")

    src = Image.open(a.src)
    if a.cell is not None:
        cw = src.width // a.cells
        src = src.crop((a.cell * cw, 0, (a.cell + 1) * cw, src.height))
        print(f"cropped cell {a.cell} of {a.cells}: {src.width}x{src.height}")

    # with_angles tracks --direction: the `<sks>` trigger is only meaningful
    # while the angles LoRA is loaded, and loading it otherwise is dead weight.
    rendered = qwen_edit.action_frames(src, frames, seed=a.seed, size=a.size,
                                       with_angles=bool(a.direction),
                                       angles_scale=a.angles_scale)

    strip = Image.new("RGB", (a.size * len(frames), a.size), (0, 0, 0))
    for i, f in enumerate(frames):
        strip.paste(rendered[f], (i * a.size, 0))
    strip.save(a.dst)
    print(f"wrote {a.dst} ({strip.width}x{strip.height})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
