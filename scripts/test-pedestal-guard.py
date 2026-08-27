#!/usr/bin/env python3
"""Does the pedestal cut remove a pedestal without amputating a wide-based prop?

WHY THIS EXISTS

`generate_raw_task` now runs `strip_ground_patch(..., require_legs=True)` on
every entity cutout, because a pedestal is fused to the subject's feet and
therefore invisible to both `remove_background` (not border-connected) and
`_isolate_largest_sprite` (not a separate blob). Geometry is the only stage that
can remove it, and prompting cannot be the backstop: this task takes the
CALLER's negative prompt, and on a distilled checkpoint `resolve_sampling_params`
discards it entirely because guidance 0 turns classifier-free guidance off.

The danger of applying it is the opposite error. something2 asks for barrels,
bushes, rocks, chests and robed figures as readily as for characters, and every
one of those is legitimately widest at its base. Cutting there amputates the
subject, and the result still looks like a plausible sprite.

`require_legs=True` refuses whenever the shins are not clearly narrower than the
body. This test proves it does both halves.

THE FIRST FIXTURE HERE WAS USELESS, and that is worth recording

A gently-tapered barrel was the obvious wide-based prop. It is spared with the
guard AND without it - its taper never exceeds `width_ratio`, so
`strip_ground_patch` declines anyway and the guard is doing nothing. It passed,
and proved nothing.

That is the second failure mode of a check, distinct from provenance: the
fixture was independently built and still could not REACH the failure. So every
"spare" case here asserts that the UNGUARDED call cuts. A spare case where the
unguarded call also spares is reported as unable to reach the failure, not as a
pass.

Usage:
    python test-pedestal-guard.py
"""

import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src", "sprite_generator"))

from pixelate import strip_ground_patch  # noqa: E402


def character_on_pedestal():
    """Narrow shins under a wider torso, standing on a slab. The case the cut
    exists for."""
    a = np.zeros((200, 200, 4), np.uint8)
    a[40:110, 80:120] = (200, 150, 120, 255)     # torso, 40 wide
    a[110:170, 90:110] = (80, 90, 160, 255)      # shins, 20 wide
    a[170:185, 40:160] = (120, 110, 90, 255)     # pedestal, 120 wide
    return Image.fromarray(a, "RGBA")


def robed_figure():
    """A hem that flares past width_ratio - no legs, and the wide base is the
    subject. Unguarded this loses about a third of the figure.

    Not a hypothetical shape: the 103 recovered RPG-Maker cells are robed
    characters, and the audit's precondition line reports the shin/body test
    holding for 0 of 103 of them.
    """
    a = np.zeros((220, 220, 4), np.uint8)
    a[40:120, 95:125] = (210, 180, 150, 255)
    for y in range(120, 190):
        w = 15 + int((y - 120) * 1.15)
        a[y, 110 - w:110 + w] = (60, 70, 130, 255)
    return Image.fromarray(a, "RGBA")


def opaque_px(img):
    return int((np.asarray(img.convert("RGBA"))[..., 3] >= 128).sum())


CASES = [
    ("character on a pedestal", character_on_pedestal, "cut"),
    ("robed figure, flared hem", robed_figure, "spare"),
]


def main():
    failures = 0
    for name, make, expect in CASES:
        src = make()
        before = opaque_px(src)
        guarded = opaque_px(strip_ground_patch(src, require_legs=True))
        plain = opaque_px(strip_ground_patch(src, require_legs=False))

        g = "cut" if guarded < before else "spared"
        u = "cut" if plain < before else "spared"

        if expect == "cut":
            ok = g == "cut"
            why = "" if ok else "the guard blocked a cut it should allow"
        else:
            ok = g == "spared" and u == "cut"
            why = ("" if ok else
                   "the guard spared it but so did the unguarded call - this "
                   "fixture cannot reach the failure and proves nothing"
                   if g == "spared" else
                   "the guard failed to spare a subject with no legs")

        print("%-26s expect %-5s | guarded %-6s %5d -> %5d | unguarded %-6s "
              "%5d -> %5d | %s"
              % (name, expect, g, before, guarded, u, before, plain,
                 "PASS" if ok else "FAIL"))
        if not ok:
            print("      %s" % why)
            failures += 1

    if failures:
        print("\nFAIL: %d of %d" % (failures, len(CASES)))
        return 1
    print("\nPASS: a pedestal is removed, and a wide-based subject that the "
          "unguarded cut would amputate is left alone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
