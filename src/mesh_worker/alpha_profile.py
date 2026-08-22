"""Is the ground shadow separable from the sprite by alpha alone?

strip_ground_patch detects by WIDTH, which fails on a character whose stance is
wider than its own shadow (charB: arm outstretched, 346px body vs a narrower
ellipse). This checks the other candidate signal: pixel-art sprites are
hard-edged and fully opaque, while a rendered shadow is soft. If the alpha
histogram is bimodal - a spike at 255 and a spread of partial values - a hard
threshold separates them cleanly and generalises better than a width rule.

Prints the histogram overall and for the bottom band where the shadow lives, so
the two can be compared rather than guessed at.
"""
import sys

import numpy as np
from PIL import Image


def main():
    path = sys.argv[1]
    a = np.asarray(Image.open(path).convert("RGBA"))[..., 3]
    h, w = a.shape

    ys, xs = np.nonzero(a > 16)
    top, bot = ys.min(), ys.max()

    def report(label, arr):
        nz = arr[arr > 0]
        if nz.size == 0:
            print(f"{label:16s} (empty)")
            return
        opaque = (nz >= 250).sum()
        partial = ((nz > 16) & (nz < 250)).sum()
        print(f"{label:16s} n={nz.size:7d}  opaque(>=250)={opaque:7d} "
              f"({100 * opaque / nz.size:5.1f}%)  partial={partial:7d} "
              f"({100 * partial / nz.size:5.1f}%)")

    report("whole image", a)

    # Bottom fifth of the OCCUPIED box - where a ground patch sits.
    band_top = bot - int((bot - top) * 0.20)
    report("bottom 20%", a[band_top:bot + 1, :])

    # Width per row near the bottom, to show why the width rule missed it.
    print("\nrow widths (bottom band, every 8th row):")
    for y in range(band_top, bot + 1, 8):
        row = a[y]
        nzx = np.nonzero(row > 16)[0]
        nzx_hard = np.nonzero(row >= 250)[0]
        wid = (nzx.max() - nzx.min() + 1) if nzx.size else 0
        wid_hard = (nzx_hard.max() - nzx_hard.min() + 1) if nzx_hard.size else 0
        print(f"  y={y:4d}  width(>16)={wid:4d}   width(>=250)={wid_hard:4d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
