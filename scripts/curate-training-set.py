#!/usr/bin/env python3
"""Exclude training images that cannot teach, and say why for each one.

WHY THIS EXISTS - two bugs found by looking at a retrained adapter's output

Splitting the contact sheets into 1859 single-subject cells did NOT fix the
lattice. The retrained terrain adapter still produced a grid of framed cells,
and asking it for "a single isometric grass tile" produced a grid too. So the
adapter was not confused by the prompt; it had learned grids.

Measuring the dataset explained it:

1. THE CELLS ARE TINY. Median short side 63px, 77% under 128px - and every one
   was being upscaled to 1024x1024 to train. A 16x blow-up of a 63px tile is
   mush, and mush is what most of the dataset was.

   This was self-inflicted. `judge_trainable` carries MIN_TRAIN_SIDE precisely
   to stop this, and cell registration wrote `trainable = true` directly,
   routing around the guard.

2. THE SHARP IMAGES WERE ALL SHEETS. Of the 112 images at 256px or better, 60
   were original uploads the splitter could not segment - median 529px, and
   grids. So the only crisp gradients in the dataset came from grid images,
   while the single-subject cells contributed blur. The model learned the sharp
   thing.

Both are excluded here, with the reason recorded per row so the UI can explain
any image's absence rather than silently dropping it.

WHAT THIS IMPLIES, and it is worth saying plainly: these references are
low-resolution asset packs. The subjects are 40-80px in the source. No amount of
splitting creates detail that was never captured, so the realistic target is a
512px adapter over a few hundred cells, not a 1024px one over thousands.

Usage:
    python curate-training-set.py --kind tile --min-px 128
    python curate-training-set.py --kind tile --min-px 128 --apply
"""

import argparse
import importlib.util
import os
import sys

import psycopg2
from PIL import Image

spec = importlib.util.spec_from_file_location(
    "ss", os.path.join(os.path.dirname(os.path.abspath(__file__)), "split-sheets.py"))
ss = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ss)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", default="tile")
    p.add_argument("--min-px", type=int, default=128,
                   help="short side below this cannot teach at training "
                        "resolution; it is upscaled into mush")
    p.add_argument("--apply", action="store_true",
                   help="write the exclusions; otherwise report only")
    a = p.parse_args()

    db = os.environ.get("DB_URL")
    if not db:
        sys.exit("DB_URL unset - run inside the container")

    conn = psycopg2.connect(db)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, file_path, coalesce(metrics->>'parent_id','') "
        "FROM reference_assets "
        "WHERE kind = %s AND deleted = false AND trainable = true", (a.kind,))
    rows = cur.fetchall()

    too_small, unverifiable, keep = [], [], []
    for rid, path, parent in rows:
        try:
            img = Image.open(path)
        except Exception:
            unverifiable.append((rid, "file unreadable"))
            continue

        if min(img.size) < a.min_px:
            too_small.append((rid, f"{img.size[0]}x{img.size[1]} - short side "
                                   f"below {a.min_px}px, would be upscaled into "
                                   f"mush at training resolution"))
            continue

        # A SOURCE (not a cell) that the splitter could not segment is not known
        # to be single-subject. The sheet of ninety weapons reads as 87%
        # foreground and yields nothing; keeping it means the sharpest images in
        # the set are grids, which is exactly what the model learned last time.
        if not parent:
            try:
                mask, _ = ss.foreground_mask(img)
                fg = float(mask.mean())
            except Exception:
                fg = -1.0
            if not (ss.FG_MIN <= fg <= ss.FG_MAX):
                unverifiable.append(
                    (rid, f"foreground {fg:.0%} - could not be segmented, so it "
                          f"cannot be confirmed single-subject"))
                continue
            if len(ss.find_cells(img)) >= 2:
                unverifiable.append(
                    (rid, "multi-subject sheet that was not split"))
                continue

        keep.append(rid)

    print(f"kind={a.kind}  currently trainable: {len(rows)}")
    print(f"  keep:                       {len(keep)}")
    print(f"  exclude, too small:         {len(too_small)}")
    print(f"  exclude, unverifiable/sheet:{len(unverifiable)}")

    if not a.apply:
        print("\n(report only - pass --apply to write)")
        return 0

    for rid, why in too_small + unverifiable:
        cur.execute("UPDATE reference_assets SET trainable = false, "
                    "trainable_why = %s WHERE id = %s", (why, rid))
    conn.commit()
    conn.close()
    print(f"\napplied: {len(too_small) + len(unverifiable)} excluded, "
          f"{len(keep)} remain trainable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
