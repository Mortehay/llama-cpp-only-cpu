#!/usr/bin/env python3
"""Re-lay an 8-wide turnaround strip into a labelled 4x2 grid for inspection.

The strip is 4096px wide, so on screen every cell is thumbnail-sized and the
question that matters - which way is this character actually facing - cannot be
answered. This reflows it and burns the requested direction into each cell.

The label is what the LoRA was ASKED for. Comparing it against what the cell
shows is the whole point: `fal/...-Multiple-Angles-LoRA` documents 8 azimuths,
but nothing guarantees its "left side" is the compass direction this project
calls west, and a mislabelled azimuth is wrong in every sheet generated
afterwards.

Usage:
    python label-turnaround.py images/_turnaround8.png images/_labelled.png
"""

import sys

from PIL import Image, ImageDraw

# Same order qwen_edit._selftest writes the strip in.
ORDER = ["s", "se", "e", "ne", "n", "nw", "w", "sw"]

ASKED = {
    "s":  "front view",
    "se": "front-right quarter",
    "e":  "right side",
    "ne": "back-right quarter",
    "n":  "back view",
    "nw": "back-left quarter",
    "w":  "left side",
    "sw": "front-left quarter",
}


def main(argv):
    src_path = argv[0] if argv else "images/_turnaround8.png"
    dst_path = argv[1] if len(argv) > 1 else "images/_labelled.png"

    strip = Image.open(src_path).convert("RGB")
    n = len(ORDER)
    if strip.width % n:
        raise SystemExit(f"{strip.width}px does not divide into {n} cells")
    cw, ch = strip.width // n, strip.height

    cols, rows = 4, 2
    band = 34  # room for the caption under each cell
    out = Image.new("RGB", (cols * cw, rows * (ch + band)), (24, 24, 28))
    draw = ImageDraw.Draw(out)

    for i, key in enumerate(ORDER):
        c, r = i % cols, i // cols
        cell = strip.crop((i * cw, 0, (i + 1) * cw, ch))
        x, y = c * cw, r * (ch + band)
        out.paste(cell, (x, y))
        draw.text((x + 8, y + ch + 10),
                  f"[{i}] {key.upper():<3}  asked for: {ASKED[key]}",
                  fill=(235, 235, 240))

    out.save(dst_path)
    print(f"{src_path} -> {dst_path} ({out.width}x{out.height}, "
          f"{cols}x{rows} cells of {cw}x{ch})")
    print("reading order:", " ".join(ORDER))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
