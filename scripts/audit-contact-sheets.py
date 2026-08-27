#!/usr/bin/env python3
"""Render the audit as contact sheets, each reference marked with its verdict.

WHY A PICTURE AND NOT JUST THE MARKDOWN

`audit-character-refs.py --out` already writes every verdict and reason. That
file is 500 lines and answers "why was this one excluded?" perfectly, and
answers "what IS this dataset?" not at all. The thing that made the problem
obvious was seeing 149 references at once and noticing that almost every tile
was a grid.

So: same verdicts, laid out. Green keeps, amber needs a look, red cannot teach,
with the first reason printed under each one.

Reads the JSON written by `audit-character-refs.py --json`, so the two can never
disagree about a verdict.

Usage:
    python audit-character-refs.py --data ./images --json /tmp/audit.json
    python audit-contact-sheets.py --json /tmp/audit.json --data ./images \
                                   --out /tmp/sheets
"""

import argparse
import json
import os
import sys
import textwrap

from PIL import Image, ImageDraw

CELL = 240
COLS, ROWS = 6, 5

VERDICT_COLOR = {
    "keep": (64, 200, 110),
    "review": (226, 170, 60),
    "reject": (214, 78, 78),
}

# Magenta behind every thumbnail, deliberately.
#
# Transparency has to be VISIBLE here or the sheet lies about the dataset: a
# subject on white and a subject on real alpha look identical against a white
# page, and the difference between them is most of what this audit is about.
ALPHA_BACKDROP = (255, 0, 255)


def short_reason(row: dict) -> str:
    why = (row.get("why") or [""])[0]
    # The reasons are written as "<what> - <why it matters>". The first half is
    # the label; the second is in the markdown for whoever wants it.
    return why.split(" - ")[0][:64]


def build(rows, data_dir, out_dir, kind):
    rows = [r for r in rows if r.get("kind") == kind]
    if not rows:
        return []
    # Worst first: the point of the sheet is to show what the dataset is made
    # of, and sorting by verdict puts the answer in the first screen.
    order = {"reject": 0, "review": 1, "keep": 2}
    rows.sort(key=lambda r: (order.get(r["verdict"], 3), r["file"]))

    os.makedirs(out_dir, exist_ok=True)
    per = COLS * ROWS
    written = []
    for s in range((len(rows) + per - 1) // per):
        chunk = rows[s * per:(s + 1) * per]
        sheet = Image.new("RGB", (COLS * CELL, ROWS * CELL), (28, 28, 34))
        d = ImageDraw.Draw(sheet)
        for i, r in enumerate(chunk):
            x, y = (i % COLS) * CELL, (i // COLS) * CELL
            colour = VERDICT_COLOR.get(r["verdict"], (128, 128, 128))
            path = os.path.join(data_dir, r["file"])
            try:
                im = Image.open(path).convert("RGBA")
                bg = Image.new("RGBA", im.size, ALPHA_BACKDROP + (255,))
                im = Image.alpha_composite(bg, im).convert("RGB")
                im.thumbnail((CELL - 20, CELL - 54), Image.LANCZOS)
                sheet.paste(im, (x + (CELL - im.width) // 2,
                                 y + 16 + (CELL - 54 - im.height) // 2))
            except Exception:
                d.text((x + 10, y + 30), "unreadable", fill=(200, 200, 200))

            d.rectangle([x + 2, y + 2, x + CELL - 3, y + CELL - 3],
                        outline=colour, width=3)
            d.rectangle([x + 2, y + 2, x + CELL - 3, y + 15], fill=colour)
            d.text((x + 6, y + 4), r["verdict"].upper(), fill=(20, 20, 20))
            d.text((x + 70, y + 4),
                   f"{r.get('w', '?')}x{r.get('h', '?')}", fill=(20, 20, 20))
            for j, line in enumerate(
                    textwrap.wrap(short_reason(r), 40)[:3]):
                d.text((x + 8, y + CELL - 38 + j * 11), line,
                       fill=(205, 205, 215))
        p = os.path.join(out_dir, f"audit-{kind}-{s:02d}.png")
        sheet.save(p)
        written.append(p)
        print(p)
    return written


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True,
                   help="the file audit-character-refs.py --json wrote")
    p.add_argument("--data", default="/app/images")
    p.add_argument("--out", required=True)
    p.add_argument("--kind", action="append", choices=["sprite", "core"])
    a = p.parse_args()

    with open(a.json) as fh:
        rows = json.load(fh)
    for kind in (a.kind or ["sprite", "core"]):
        build(rows, a.data, a.out, kind)
    return 0


if __name__ == "__main__":
    sys.exit(main())
