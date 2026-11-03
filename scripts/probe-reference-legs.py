#!/usr/bin/env python3
"""Can a CHEAP signal catch a turnaround reference that is cut off at the thigh?

`probe-turnaround-framing.py` measures opaque area and the occupied box's
aspect, and on job `b0e77e2a` it flagged NOTHING: `dir_e` read 1.17x the
median (under that probe's own 1.25 cut) and aspect 0.762 against 0.727 for a
`dir_s` that is a clean full body. The two images are not subtle - `dir_e` is
chin to mid-thigh with no legs at all - so area and aspect are not merely
weak here, they are the wrong measurement. Both are confounded by a held
weapon: the sword widens the box in every view.

This tests a different idea before reaching for a VLM. A figure standing on
its feet TAPERS at the bottom - two legs, or two feet with air between them -
while a figure cut across the thighs ends in one solid block spanning most of
its width. So measure the lowest band of the figure and ask how full it is:

  bottom_fill  mean opaque fraction of the figure's bbox WIDTH over the
               lowest `--band` share of its height
  waist_ratio  that bottom fill divided by the fill at mid-height, which
               normalises away how wide the subject is in general. A thigh
               cut should read near 1.0 (bottom as solid as the waist); a
               standing figure should read well under it.
  parts        connected components in the lowest band, counted per row and
               taken at the median: 2 means two legs with a gap.

Verdict column marks the within-job MAXIMUM of `waist_ratio`, because the
question is always "is one of these eight framed unlike the others" and an
absolute threshold would not transfer across subjects.

Read-only, CPU-only: safe to run while a build holds the card.

    python3 scripts/probe-reference-legs.py <job_id> [<job_id> ...] [--band 0.18]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slice_metrics import alpha_of  # noqa: E402

IMAGES = "/app/images"
DIRECTIONS = ["s", "se", "e", "ne", "n", "nw", "w", "sw"]


def runs_in_row(row: np.ndarray) -> int:
    """How many separate opaque stretches this row has."""
    if not row.any():
        return 0
    edges = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
    return int((edges == 1).sum())


def measure(path: str, band: float) -> dict | None:
    if not os.path.exists(path):
        return None
    # alpha_of, not a raw alpha read: `dir_*.png` is an RGB render on white and
    # has no alpha channel at all, so `convert("RGBA")[..., 3]` returns 255
    # everywhere and every measurement below comes out as exactly 1.000. That
    # is how the first run of this probe "found no difference" on all 4 jobs.
    solid = alpha_of(path) > 0
    ys, xs = np.nonzero(solid)
    if not ys.size:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    box = solid[y0:y1 + 1, x0:x1 + 1]
    h, w = box.shape
    n = max(int(round(h * band)), 1)

    bottom = box[h - n:, :]
    bottom_fill = float(bottom.mean())
    # Mid-height band of the same thickness: the torso/waist, which is the
    # widest solid part of a standing figure and so the right normaliser.
    mid0 = max((h - n) // 2, 0)
    mid = box[mid0:mid0 + n, :]
    mid_fill = float(mid.mean()) or 1e-6
    parts = int(np.median([runs_in_row(r) for r in bottom]))
    return {
        "bottom_fill": round(bottom_fill, 3),
        "mid_fill": round(mid_fill, 3),
        "waist_ratio": round(bottom_fill / mid_fill, 3),
        "parts": parts,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_ids", nargs="+")
    ap.add_argument("--band", type=float, default=0.18,
                    help="share of the figure's height to treat as the bottom")
    a = ap.parse_args(argv)

    for job in a.job_ids:
        work = os.path.join(IMAGES, f"_job_{job}")
        if not os.path.isdir(work):
            print(f"{job}: no work directory", file=sys.stderr)
            continue
        cfg_p = os.path.join(work, "config.json")
        seed = json.load(open(cfg_p)).get("seed") if os.path.exists(cfg_p) else "?"
        found = {d: measure(os.path.join(work, f"dir_{d}.png"), a.band)
                 for d in DIRECTIONS}
        found = {d: m for d, m in found.items() if m}
        if not found:
            print(f"\n=== job {job[:8]}: no dir_*.png")
            continue
        worst = max(found, key=lambda d: found[d]["waist_ratio"])
        print(f"\n=== job {job[:8]}  seed {seed}  band {a.band}")
        print(f"  {'dir':<5} {'bottom':>7} {'mid':>7} {'waist':>7} {'parts':>6}"
              f"   verdict")
        for d in DIRECTIONS:
            m = found.get(d)
            if not m:
                continue
            print(f"  {d:<5} {m['bottom_fill']:>7.3f} {m['mid_fill']:>7.3f} "
                  f"{m['waist_ratio']:>7.3f} {m['parts']:>6}"
                  f"   {'<- widest bottom in job' if d == worst else ''}")
    print("\nA standing figure tapers at the bottom; one cut across the thighs "
          "does not. If `waist_ratio` does not separate the known-bad `e` "
          "references from the known-good ones, geometry is not enough and "
          "the check has to be semantic.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
