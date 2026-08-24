#!/usr/bin/env python3
"""Acceptance check for a finished sprite sheet.

The deliverable is "a pixel-art sheet with a transparent background", and every
part of that claim is measurable. This asserts it rather than eyeballing a
preview, because a viewer composites RGBA over white and an image with NO alpha
at all looks identical to a correctly keyed one.

Usage:
    check-sprite.py <sheet.png> [--grid 4x2]
"""

import sys
from collections import Counter

import numpy as np
from PIL import Image


def report(path: str, cols: int = 1, rows: int = 1) -> bool:
    img = Image.open(path)
    arr = np.asarray(img.convert("RGBA"))
    h, w = arr.shape[:2]
    alpha = arr[..., 3]

    total = alpha.size
    transparent = int((alpha == 0).sum())
    opaque = int((alpha == 255).sum())
    partial = total - transparent - opaque

    # Unique colours among opaque pixels only. Counting transparent pixels here
    # inflates the number with whatever RGB happens to sit under alpha 0.
    opaque_rgb = arr[alpha == 255][:, :3]
    colours = len(np.unique(opaque_rgb, axis=0)) if len(opaque_rgb) else 0

    print(f"{path}")
    print(f"  size          {w}x{h}" + (f"  ({cols}x{rows} cells of "
          f"{w // cols}x{h // rows})" if cols * rows > 1 else ""))
    print(f"  mode          {img.mode}")
    print(f"  transparent   {transparent / total:6.1%}")
    print(f"  opaque        {opaque / total:6.1%}")
    print(f"  partial alpha {partial / total:6.1%}  ({partial} px)")
    print(f"  colours       {colours}")

    ok = True

    if img.mode != "RGBA":
        print("  FAIL  not RGBA - there is no alpha channel at all")
        ok = False
    elif transparent == 0:
        print("  FAIL  nothing is transparent - the background was never keyed")
        ok = False

    if partial:
        print(f"  FAIL  {partial} pixels have partial alpha - these become a "
              f"halo once composited over a game tile")
        ok = False

    # A hand-drawn reference sheet sits at 16-32. Well above that means the
    # palette lock did not run, which is the failure this check exists to catch:
    # the sheet can look right at a glance and still carry 40k colours.
    if colours > 64:
        print(f"  FAIL  {colours} colours - not palette-locked pixel art")
        ok = False

    if cols * rows > 1:
        if w % cols or h % rows:
            print(f"  FAIL  does not divide evenly into {cols}x{rows} - "
                  f"something2 rejects such a sheet")
            ok = False
        else:
            # Per-cell footprint. Cells that differ wildly mean the character
            # changes size between frames, which reads as pulsing in game.
            cw, ch = w // cols, h // rows
            foots = []
            for r in range(rows):
                for c in range(cols):
                    cell = alpha[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
                    ys, xs = np.nonzero(cell)
                    if len(ys) == 0:
                        print(f"  WARN  cell ({c},{r}) is entirely empty")
                        continue
                    foots.append((int(xs.max() - xs.min() + 1),
                                  int(ys.max() - ys.min() + 1),
                                  int(ys.max())))
            if foots:
                hs = [f[1] for f in foots]
                bl = Counter(f[2] for f in foots)
                print(f"  cell height   min {min(hs)} max {max(hs)} "
                      f"(spread {max(hs) - min(hs)}px)")
                print(f"  baseline rows {dict(bl)}")
                if len(bl) > 1:
                    print("  WARN  cells do not share a baseline - the sprite "
                          "will bob vertically during playback")

    print("  RESULT        " + ("PASS" if ok else "FAIL"))
    return ok


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    path = argv[0]
    cols = rows = 1
    if "--grid" in argv:
        cols, rows = (int(x) for x in argv[argv.index("--grid") + 1].split("x"))
    return 0 if report(path, cols, rows) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
