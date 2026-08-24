#!/usr/bin/env python3
"""Print a sprite's opaque width per row, as a percentage of its height.

Written after two failed guesses at a ground-patch rule. Both were plausible
(body median, then shin band) and both were wrong in ways the image alone did
not reveal - the numbers were needed.

Usage:
    python width-profile.py sprite.png [--rows 24] [--key-tolerance 10]
"""

import argparse
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
for _c in (os.path.join(_here, "..", "src", "sprite_generator"),
           os.path.dirname(_here)):
    if os.path.isfile(os.path.join(_c, "pixelate.py")):
        sys.path.insert(0, _c)
        break

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import pixelate  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("--rows", type=int, default=26, help="sample this many rows")
    p.add_argument("--key", action="store_true", default=True)
    p.add_argument("--key-tolerance", type=int, default=10)
    a = p.parse_args()

    img = Image.open(a.src)
    if a.key:
        img = pixelate.key_background(img, tolerance=a.key_tolerance)

    arr = np.asarray(img.convert("RGBA"))
    opaque = arr[..., 3] >= 128
    rows = np.where(opaque.any(axis=1))[0]
    if rows.size == 0:
        print("nothing opaque")
        return 1

    top, bot = int(rows[0]), int(rows[-1])
    height = bot - top + 1
    widths = opaque.sum(axis=1).astype(float)
    body = widths[top:bot + 1]
    body = body[body > 0]
    peak = float(body.max())

    print(f"{a.src}")
    print(f"  bbox rows {top}..{bot}  height {height}px  peak width {peak:.0f}px")
    print(f"  {'from bottom':>12}  {'width':>6}  {'/peak':>6}")

    step = max(1, height // a.rows)
    for y in range(bot, top - 1, -step):
        frac = (bot - y) / height
        w = widths[y]
        bar = "#" * int(w / peak * 40)
        print(f"  {frac * 100:10.0f}%  {w:6.0f}  {w / peak:6.2f}  {bar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
