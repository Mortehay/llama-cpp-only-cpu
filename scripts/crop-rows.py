#!/usr/bin/env python3
"""Cut a row range out of a sheet and upscale it for inspection.

A 24-row sheet at a readable zoom is 6000px tall, which is unreadable as one
image. This pulls out one action's block of rows at a time.

Usage:
    python crop-rows.py sheet.png out.png --rows 0-7 --cell 48x64 --scale 5
"""

import argparse
import sys

from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--rows", required=True, help="inclusive range, e.g. 0-7")
    p.add_argument("--cell", default="48x64")
    p.add_argument("--scale", type=int, default=5)
    a = p.parse_args()

    cw, ch = (int(x) for x in a.cell.lower().split("x"))
    lo, _, hi = a.rows.partition("-")
    lo, hi = int(lo), int(hi or lo)

    im = Image.open(a.src)
    total_rows = im.height // ch
    if hi >= total_rows:
        print(f"sheet has {total_rows} rows; {hi} is out of range", file=sys.stderr)
        return 1

    crop = im.crop((0, lo * ch, im.width, (hi + 1) * ch))
    # NEAREST so the preview shows the real pixels rather than a smoothed guess.
    out = crop.resize((crop.width * a.scale, crop.height * a.scale),
                      Image.Resampling.NEAREST)

    # Flatten onto white: a viewer composites RGBA over its own background, and
    # for INSPECTION a known ground is what makes alpha errors visible.
    if out.mode == "RGBA":
        bg = Image.new("RGB", out.size, (255, 255, 255))
        bg.paste(out, mask=out.split()[3])
        out = bg

    out.save(a.dst)
    print(f"rows {lo}-{hi} of {total_rows} -> {a.dst} ({out.width}x{out.height})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
