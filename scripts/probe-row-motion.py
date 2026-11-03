#!/usr/bin/env python3
"""Does a row actually animate, and which frames are copies of which?

`sheet.row_motion` already reports `mean_step`, `min_step` and `duplicates`
per row into the atlas, but it has two blind spots this probe exists to cover:

  1. **`duplicates` counts CONSECUTIVE pairs only.** A pose list that repeats
     an entry at frames 2 and 5 produces two byte-identical renders - the
     conveyor renders one image per distinct (direction, pose) and copies it -
     and `row_motion` reports `duplicates: 0` because the pair is not
     adjacent. That exact shape was found in `sway side` (frames 1 and 3) and
     is in `walk front` (frames 2 and 5, "both feet flat on the ground,
     standing upright"). The full pairwise matrix is what shows it.
  2. It measures the FITTED cell. The family averages on record (side 19.08,
     front 14.84, back 12.20) are fitted-cell numbers, while the idle and sway
     pose-rewrite verifications were measured on the 512 px `cut_*.png`. Those
     two scales are NOT comparable. This probe reads `cut_*.png` by default so
     a pose rewrite can be scored against those recorded figures, with
     `--fitted` to read the sheet's own cells instead.

Reports, per row, the loop steps (wrapping last->first, as `row_motion` does),
then the closest pair anywhere in the row. A row whose closest pair is far
apart in index but near zero in distance is a repeated pose, not a slow one.

Read-only, CPU-only: safe to run while a build holds the card.

    python3 scripts/probe-row-motion.py <job_id> [--action walk] [--fitted]
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np
from PIL import Image

IMAGES = "/app/images"
DIRECTIONS = ["s", "se", "e", "ne", "n", "nw", "w", "sw"]
# sheet.DUPLICATE_STEP, repeated rather than imported so this stays runnable
# without pulling in the sheet module and its dependencies.
DUPLICATE_STEP = 1.0
VIEW_FAMILY = {"s": "front", "se": "front", "sw": "front",
               "n": "back", "ne": "back", "nw": "back",
               "e": "side", "w": "side"}


def load(path: str):
    if not os.path.exists(path):
        return None
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.float32)


def row_frames(work: str, action: str, d: str, frames: int, prefix: str):
    out = []
    for f in range(frames):
        arr = load(os.path.join(work, f"{prefix}_{action}_{d}_{f}.png"))
        if arr is None:
            return None
        out.append(arr)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id")
    ap.add_argument("--action", default=None,
                    help="only this action (default: every action in config)")
    ap.add_argument("--fitted", action="store_true",
                    help="read the fitted sheet cells instead of the 512px cutouts")
    a = ap.parse_args(argv)

    work = os.path.join(IMAGES, f"_job_{a.job_id}")
    if not os.path.isdir(work):
        print(f"{a.job_id}: no work directory", file=sys.stderr)
        return 2
    cfg = json.load(open(os.path.join(work, "config.json")))
    frames = int(cfg.get("frames") or cfg.get("frames_requested") or 6)
    actions = [a.action] if a.action else list(cfg.get("actions") or [])
    prefix = "cell" if a.fitted else "cut"
    scale = "fitted cells" if a.fitted else "512px cutouts"
    print(f"job {a.job_id[:8]}  seed {cfg.get('seed')}  "
          f"{frames} frames  measured on {scale}")
    print("  family averages on record are FITTED-cell numbers "
          "(side 19.08, front 14.84, back 12.20);")
    print("  the idle/sway rewrite figures are 512px ones. Do not mix them.\n")

    for action in actions:
        print(f"=== {action}")
        print(f"  {'dir':<5} {'family':<6} {'mean':>7} {'min':>7} {'dups':>5}"
              f"   closest pair")
        for d in DIRECTIONS:
            arr = row_frames(work, action, d, frames, prefix)
            if arr is None:
                continue
            steps = [float(np.abs(arr[i] - arr[(i + 1) % len(arr)]).mean())
                     for i in range(len(arr))]
            pairs = {(i, j): float(np.abs(arr[i] - arr[j]).mean())
                     for i, j in itertools.combinations(range(len(arr)), 2)}
            (ci, cj), cv = min(pairs.items(), key=lambda kv: kv[1])
            # A near-zero pair that is NOT adjacent is the one row_motion
            # cannot see, so say so explicitly rather than leaving it to be
            # read off the indices.
            note = ""
            if cv < DUPLICATE_STEP:
                note = ("  <- COPY, non-adjacent: invisible to row_motion"
                        if abs(ci - cj) not in (1, len(arr) - 1) else
                        "  <- copy, adjacent")
            print(f"  {d:<5} {VIEW_FAMILY.get(d, '?'):<6} "
                  f"{np.mean(steps):>7.2f} {min(steps):>7.2f} "
                  f"{sum(1 for s in steps if s < DUPLICATE_STEP):>5}"
                  f"   {ci}-{cj} at {cv:.2f}{note}")
        print()
    print("A row that does not animate needs different POSE TEXT, not another "
          "seed: every frame is re-conditioned on the same turnaround "
          "reference, so a redraw cannot add motion.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
