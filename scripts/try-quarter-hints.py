#!/usr/bin/env python3
"""Can a prompt hint make `se` a real three-quarter view instead of the front?

    try-quarter-hints.py <concept.png> --variant 0 --out /app/images/_quarters

ONE VARIANT PER INVOCATION, and a wrapper loop runs them. Each variant is a
text-encoder load followed by a transformer load, and doing several in one
process is what `build-sheet.py` is split into stages to avoid: loading and
releasing a large quantised model fragments the CUDA allocator enough to fail
a later load with gigabytes nominally free.

WHAT IS BROKEN

The front quarters collapse onto the front view at the turnaround stage.
Measured on the two evidence jobs, mean absolute greyscale difference between
turnaround views:

            |s-se|  |s-sw|  |s-e|  |se-sw|  |se-mirror(sw)|
    212d56d7  12.8    15.3   52.2    18.1       43.6
    8d9d164d  16.4    28.4   60.4    33.9       53.7

se sits 3-4x closer to the front view than e does, and se and sw are near
duplicates of EACH OTHER rather than mirrors. Every action cell inherits
this, because each is conditioned on `dir_<d>.png`.

WHY A HINT IS THE THING TO TRY

This is the same shape as the `e` failure. `right side` alone came back
near-frontal, and ", exact side profile, one eye visible" fixed it - that is
what AZIMUTH_HINTS exists for. The quarters have no hint at all today.

The module warns, correctly, NOT to apply the side hint globally: telling the
FRONT view that the character faces sideways is a contradiction and the model
resolves it per seed. A quarter-specific hint is a different thing, and that
is what this measures.

Hints are kept SHORT on purpose. Prompt length is not free here - the first
working version of the side hint was long enough to OOM a 1.02 GiB buffer
once more than one long-hinted direction was in a batch.

WHAT WOULD COUNT AS SUCCESS

`se` should sit roughly BETWEEN the front and the profile, not on top of the
front: |s-se| rising from ~13-16 toward half of |s-e| (~25-30), while |s-e|
itself stays where it is. And `sw` should become the mirror of `se`
(|se-mirror(sw)| falling below |se-sw|) rather than its duplicate.

A hint that moves |s-se| by a point or two has not worked. The `e` fix moved
its pair from 4.70/24.55 to 25.56/6.27 - a real fix here should be that
unambiguous, and a marginal one should be read as noise.

RESULTS, 2026-10-22, core_671281682df7 (the Cultist), seed 1916017834

    variant                       |s-se|  figure vs s   position   verdict
    0  control                     16.42     1.00x        0.22      front copy
    1  three-quarter, shoulders    78.72     1.62x        0.48      CLOSE-UP
    2  turned 45 deg, shoulder     51.60     1.12x        0.52      off the path
    3  halfway front and profile   38.72     1.02x        0.45      WORKS

"position" is |s-se| / (|s-se| + |se-e|): 0 is the front, 1 is the profile,
0.5 is a true three-quarter. The control sits at 0.22 - a fifth of the way
round - which is the bug, measured.

**Variant 3 is the one that works**, confirmed by eye at full size: angled
shoulders, the face turned, a narrower torso and the talisman hand set back,
where the control is square-shouldered and straight-on.

DO NOT TRUST |s-se| ALONE. It was the first metric here and it is wrong:
the angles LoRA crops differently per azimuth, so a wording that merely
ZOOMS scores as if it had rotated. Variant 1 scored the largest |s-se| of
all - 78.72, nearly five times the control - purely by framing the subject
1.62x larger, which is a close-up and a regression that this repo's own
`close_up` check would flag. Always read |s-se| beside the figure fraction
and the position on the s->e path.

STILL UNSETTLED, and the reason variant 3 is not wired into AZIMUTH_HINTS:

  * ONE subject, ONE seed. The `w` investigation shows how far a single-seed
    conclusion can mislead, and 0002 records the same lesson.
  * `sw` does not become a clean mirror of `se` under variant 3 (|se-sw|
    54.35 against |se-mirror(sw)| 55.90 - a tie, so neither duplicate nor
    mirror). Variant 2 mirrors better (67.64 against 46.36) while giving a
    worse `se`. Whatever is adopted has to satisfy both.

Validate over at least two more subjects and two seeds before editing
AZIMUTH_HINTS.
"""

import argparse
import json
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
for _c in (os.path.join(_here, "..", "src", "sprite_generator"),
           os.path.dirname(_here), "/app"):
    if os.path.isfile(os.path.join(_c, "qwen_edit.py")):
        sys.path.insert(0, _c)
        break

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

# The hint is appended to the LoRA's own trigger for that azimuth, exactly as
# AZIMUTH_HINTS does for the sides. Variant 0 is the control: what the
# conveyor does today, which is nothing.
VARIANTS = [
    {"name": "control (no quarter hint)", "se": "", "sw": ""},
    {"name": "three-quarter, shoulders angled",
     "se": ", three-quarter view, shoulders angled",
     "sw": ", three-quarter view, shoulders angled"},
    {"name": "turned 45 degrees, one shoulder forward",
     "se": ", body turned 45 degrees, one shoulder forward",
     "sw": ", body turned 45 degrees, one shoulder forward"},
    {"name": "halfway between front and profile",
     "se": ", halfway between front and profile",
     "sw": ", halfway between front and profile"},
]

# s and e are the anchors every score is read against, so they are rendered
# in every variant rather than reused from another run - a different pass is
# a different allocator and, more importantly, the comparison has to be
# within one seed and one load to mean anything.
DIRECTIONS = ["s", "se", "e", "sw"]


def grey(img):
    return np.asarray(img.convert("L"), dtype=np.float32)


def mad(a, b):
    return float(np.abs(a - b).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("concept")
    p.add_argument("--variant", type=int, required=True)
    p.add_argument("--out", default="/app/images/_quarters")
    p.add_argument("--seed", type=int, default=1916017834)
    p.add_argument("--size", type=int, default=512)
    a = p.parse_args()

    import qwen_edit

    v = VARIANTS[a.variant]
    # Patched before any prompt is built: `angle_prompt` reads this dict at
    # call time, and `turnaround_cells` calls it. The side hints are left
    # alone - `e` is the anchor and is known to work with its own hint.
    qwen_edit.AZIMUTH_HINTS = {**qwen_edit.AZIMUTH_HINTS,
                               **{k: v[k] for k in ("se", "sw") if v[k]}}

    os.makedirs(a.out, exist_ok=True)
    concept = Image.open(a.concept)

    print(f"variant {a.variant}: {v['name']}")
    for d in DIRECTIONS:
        print(f"  {d:2s} -> {qwen_edit.angle_prompt(d)}")

    views = qwen_edit.turnaround_split(concept, seed=a.seed,
                                       directions=DIRECTIONS, size=a.size)
    for d, im in views.items():
        im.save(os.path.join(a.out, f"v{a.variant}_{d}.png"))

    g = {d: grey(im) for d, im in views.items()}
    res = {
        "variant": a.variant,
        "name": v["name"],
        "hint_se": v["se"],
        "s_se": round(mad(g["s"], g["se"]), 2),
        "s_sw": round(mad(g["s"], g["sw"]), 2),
        "s_e": round(mad(g["s"], g["e"]), 2),
        "se_sw": round(mad(g["se"], g["sw"]), 2),
        "se_mirror_sw": round(mad(g["se"], g["sw"][:, ::-1]), 2),
    }
    # The headline: how far se sits from the front, as a fraction of how far
    # the profile does. ~0 is a front copy, ~0.5 is a genuine three-quarter.
    res["se_fraction_of_profile"] = round(res["s_se"] / max(res["s_e"], 1e-6), 3)
    res["sw_is_mirrored"] = res["se_mirror_sw"] < res["se_sw"]

    print(json.dumps(res))
    with open(os.path.join(a.out, f"v{a.variant}.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
