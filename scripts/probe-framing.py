"""Exercise framing.py end to end on an existing sheet's cells. CPU only.

WHY THIS AND NOT A SHEET BUILD
A build is 3h07 at the Q2_K baseline and holds the card exclusively. The
framing check reads decoded pixels off `cut_*.png` and never touches the
GPU, so every claim in framing.py's docstring can be tested against an
already-rendered sheet - at ~1.18 s/cell, ~4 min for 210 cells - without
queueing a job or disturbing another session.

WHAT IT ANSWERS
1. Does the check run at all in this container (weights, env flag, no GPU)?
2. The `feet_in_frame` distribution over a real 210-cell sheet.
3. How many cells the 0.49 cut would newly flag, and how that overlaps the
   `close_up` flags already in scores.json - close_up is the check framing
   was built to cover for, so the overlap is the interesting number.
4. The turnaround references' own scores, against the 0.9931/0.9932 on
   record. Those are what per-sheet calibration was rejected for.
5. Real seconds per cell on this box.

Run with FRAMING_CHECK=1 in the environment or framing returns None by design,
and run it INSIDE the worker, where /app and /models are mounted:

    docker cp scripts/probe-framing.py sprite_worker:/tmp/
    docker exec -e FRAMING_CHECK=1 sprite_worker \
        python3 -u /tmp/probe-framing.py 2ff8af36

The argument is any substring of a job directory under /app/images. Safe
while another session is working - it is CPU only and opens nothing for
writing inside the job - but do NOT run it against a job whose build is still
on the card, because the cells are rewritten under you as stages finish.
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, "/app")

from PIL import Image  # noqa: E402

import framing  # noqa: E402

def _job_dir(argv):
    """Resolve a job from a substring, or explain what is there instead."""
    want = argv[1] if len(argv) > 1 else ""
    hits = sorted(d for d in glob.glob("/app/images/*%s*" % want)
                  if os.path.isdir(d) and glob.glob(os.path.join(d, "cut_*.png")))
    if not hits:
        print("no job directory with cut_*.png matching %r. Candidates:" % want)
        for d in sorted(glob.glob("/app/images/*")):
            n = len(glob.glob(os.path.join(d, "cut_*.png")))
            if n:
                print("  %-56s %d cells" % (os.path.basename(d), n))
        sys.exit(2)
    if len(hits) > 1:
        print("ambiguous (%r matches %d):" % (want, len(hits)))
        for d in hits:
            print("  " + os.path.basename(d))
        sys.exit(2)
    return hits[0]


JOB = _job_dir(sys.argv)
OUT = "/tmp/framing_probe_%s.json" % os.path.basename(JOB).lstrip("_")[:24]


def main():
    print("framing.ENABLED   :", framing.ENABLED)
    print("framing.FRAMING_MIN:", framing.FRAMING_MIN)
    print("framing.BAND/ZOOM :", framing.BAND, framing.ZOOM)
    print("job               :", os.path.basename(JOB))
    if not framing.ENABLED:
        print("FRAMING_CHECK is off - nothing to measure. Set it and re-run.")
        return 1

    scores = json.load(open(os.path.join(JOB, "scores.json")))
    cells = scores["cells"]

    # The references first: cheap, and if calibration mattered this is where
    # it would show.
    print("\n-- turnaround references --")
    refs = {}
    for p in sorted(glob.glob(os.path.join(JOB, "dir_*.png"))):
        d = os.path.basename(p)[4:-4]
        s = framing.feet_in_frame(Image.open(p))
        refs[d] = s
        print("  dir_%-2s %.4f" % (d, s))

    print("\n-- cells --")
    rows = []
    t0 = time.time()
    for key, rec in cells.items():
        action, direction, frame = key.split("|")
        p = os.path.join(JOB, "cut_%s_%s_%s.png" % (action, direction, frame))
        if not os.path.exists(p):
            continue
        s = framing.feet_in_frame(Image.open(p))
        if s is None:
            print("  framing returned None on", os.path.basename(p))
            continue
        # scores.json keeps every attempt; `best` is the one on disk.
        best = rec["attempts"][rec["best"]]
        rows.append({
            "key": key,
            "feet_in_frame": round(s, 4),
            "framing_flag": framing.is_close_up(s),
            "recorded_failed": rec.get("failed_checks", []),
            "best_failed": best.get("failed_checks", []),
            "figure_ratio": best["metrics"].get("figure_ratio"),
        })
    dt = time.time() - t0

    n = len(rows)
    if not n:
        print("no cells scored")
        return 1

    vals = sorted(r["feet_in_frame"] for r in rows)
    flagged = [r for r in rows if r["framing_flag"]]
    close_up = [r for r in rows if "close_up" in r["recorded_failed"]]
    both = [r for r in flagged if "close_up" in r["recorded_failed"]]
    any_fail = [r for r in rows if r["recorded_failed"]]

    def pct(k):
        return vals[min(int(k * (n - 1)), n - 1)]

    print("\ncells scored      : %d" % n)
    print("seconds total     : %.1f  (%.2f s/cell)" % (dt, dt / n))
    print("feet_in_frame     : min %.4f  p10 %.4f  median %.4f  p90 %.4f  max %.4f"
          % (vals[0], pct(.10), pct(.50), pct(.90), vals[-1]))
    print("below 0.49 (flag) : %d of %d (%.1f%%)" % (len(flagged), n, 100.0 * len(flagged) / n))
    print("recorded close_up : %d" % len(close_up))
    print("  overlap         : %d" % len(both))
    print("recorded any fail : %d" % len(any_fail))

    print("\n-- lowest 12 --")
    for r in sorted(rows, key=lambda r: r["feet_in_frame"])[:12]:
        print("  %-22s %.4f  flag=%-5s ratio=%-7s recorded=%s"
              % (r["key"], r["feet_in_frame"], r["framing_flag"],
                 r["figure_ratio"], r["recorded_failed"] or "-"))

    print("\n-- the recorded close_up cells, as framing sees them --")
    for r in close_up:
        print("  %-22s %.4f  flag=%s" % (r["key"], r["feet_in_frame"], r["framing_flag"]))

    json.dump({"job": os.path.basename(JOB), "refs": refs,
               "seconds_per_cell": round(dt / n, 3), "cells": rows},
              open(OUT, "w"), indent=1)
    print("\nper-cell rows -> %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
