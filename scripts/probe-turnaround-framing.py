#!/usr/bin/env python3
"""Is a row's close-up inherited from its TURNAROUND REFERENCE?

something2 reported (2026-11-03, job 7f9fc8f1) that the e/ne rows of a 96x128
sheet are severe close-ups - chin to mid-thigh - while s/se/n/sw are full
body, and that sheet QA passed every frame.

`sheet_qc.figure_ratio` scored those cells at ~0.93, i.e. SMALLER than their
reference, when a close-up should read larger (the signature on record is
1.85-1.98x). A ratio near 1.0 on a cell that is plainly a close-up has one
obvious explanation: the reference is a close-up too. Every action cell is
re-conditioned on `dir_<d>.png`, so a badly framed turnaround view propagates
to every frame of that row AND hides itself from the one check that should
catch it - the same structural blindness the `facing` check has when its
reference is weak.

This measures the turnaround references against each other instead of against
themselves. Every render is edge-to-edge (pre-fit figure height is 478-512 of
512 on every cell), so HEIGHT cannot discriminate; opaque AREA can, because a
torso filling the frame covers more of it than a full body with air beside the
legs.

Reports each direction's area as a multiple of the MEDIAN direction, so the
comparison needs no absolute threshold and no good reference to exist.

`--baseline <job_id>` takes the median from another job instead, and restricts
it to the directions BOTH jobs rendered. Needed whenever a job renders a
subset: a re-queue of the flagged rows only (se,e,ne,nw,w,sw) has lost `s` and
`n`, which are the full-body anchors, and if `e`/`ne` are close-ups they
inflate the very median they are being judged against. Two jobs compared on
different direction sets are not comparable at all.

Read-only, CPU-only: safe while a build holds the card.

    python3 scripts/probe-turnaround-framing.py <job_id> [<job_id> ...]
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slice_metrics import alpha_of  # noqa: E402

IMAGES = "/app/images"
DIRECTIONS = ["s", "se", "e", "ne", "n", "nw", "w", "sw"]


def describe(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    alpha = alpha_of(path)
    h, w = alpha.shape
    ys, xs = np.nonzero(alpha > 0)
    if not ys.size:
        return None
    return {
        "area": int((alpha > 0).sum()),
        "area_frac": round(float((alpha > 0).mean()), 4),
        "bbox_h": int(ys.max() - ys.min() + 1),
        "bbox_w": int(xs.max() - xs.min() + 1),
        "h_frac": round(float((ys.max() - ys.min() + 1) / h), 3),
        # Aspect of the occupied box. A full standing body is tall and narrow;
        # a torso close-up is squarer. Independent of area, so the two agreeing
        # is worth more than either alone.
        "aspect": round(float((xs.max() - xs.min() + 1)
                              / max(ys.max() - ys.min() + 1, 1)), 3),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_ids", nargs="+")
    ap.add_argument("--baseline", default=None,
                    help="take the median from this job, over the directions "
                         "both jobs rendered")
    a = ap.parse_args(argv)

    base = {}
    if a.baseline:
        bwork = os.path.join(IMAGES, f"_job_{a.baseline}")
        for d in DIRECTIONS:
            m = describe(os.path.join(bwork, f"dir_{d}.png"))
            if m:
                base[d] = m
        if not base:
            print(f"baseline {a.baseline[:8]} has no dir_*.png", file=sys.stderr)
            return 2
        print(f"baseline: {a.baseline[:8]}, {len(base)} direction(s)")

    for job in a.job_ids:
        work = os.path.join(IMAGES, f"_job_{job}")
        if not os.path.isdir(work):
            print(f"{job}: no work directory", file=sys.stderr)
            continue
        cfg_p = os.path.join(work, "config.json")
        seed = json.load(open(cfg_p)).get("seed") if os.path.exists(cfg_p) else "?"
        print(f"\n=== job {job[:8]}  seed {seed}")

        found = {}
        for d in DIRECTIONS:
            m = describe(os.path.join(work, f"dir_{d}.png"))
            if m:
                found[d] = m
        if not found:
            print("  no dir_*.png turnaround references (pruned?)")
            continue

        if base:
            # Only the directions both jobs rendered, or the two medians
            # describe different things and the multiples cannot be compared.
            shared = [d for d in DIRECTIONS if d in found and d in base]
            med = float(np.median([base[d]["area"] for d in shared]))
            print(f"  median from baseline over {len(shared)} shared "
                  f"direction(s): {', '.join(shared)}")
        else:
            med = float(np.median([m["area"] for m in found.values()]))
            if len(found) < len(DIRECTIONS):
                print(f"  WARNING: only {len(found)} of {len(DIRECTIONS)} "
                      f"directions rendered, so this median excludes the "
                      f"missing ones. Pass --baseline <full job> to compare "
                      f"across jobs.")
        print(f"  {'dir':<5} {'area':>7} {'x median':>9} {'h_frac':>7} "
              f"{'aspect':>7}  verdict")
        for d in DIRECTIONS:
            m = found.get(d)
            if not m:
                continue
            mult = m["area"] / med
            # Direction-aware, because a PROFILE legitimately covers less of
            # the frame than a front view - it is the narrowest of the eight.
            # Calling that "small" was an artefact of calibrating on the
            # broken state: at the old `medium` default `e` read 1.17-1.22x
            # the median, i.e. MORE area than a front view, which is
            # physically backwards. Since the `wide` default a correct
            # profile reads 0.68-0.69x and would have been flagged for being
            # right. For e/w the tell is therefore reaching the median at
            # all, not falling below it.
            if d in ("e", "w"):
                flag = "CLOSE-UP? (a profile should be the SMALLEST view)" \
                    if mult >= 1.0 else ""
            else:
                flag = ("CLOSE-UP?" if mult >= 1.25 else
                        "small" if mult <= 0.8 else "")
            print(f"  {d:<5} {m['area']:>7} {mult:>9.2f} {m['h_frac']:>7.3f} "
                  f"{m['aspect']:>7.3f}  {flag}")

        # What the build itself recorded, read at `best` rather than attempt 0:
        # the redraw rounds replace the picture and move `best`, so attempt 0
        # can describe a frame that no longer exists.
        sp = os.path.join(work, "scores.json")
        if os.path.exists(sp):
            cells = json.load(open(sp)).get("cells", {})
            by_dir: dict = {}
            for key, c in cells.items():
                best = c.get("attempts", [{}])[c.get("best", 0)]
                fr = (best.get("metrics") or {}).get("figure_ratio")
                if fr is not None:
                    by_dir.setdefault(c["direction"], []).append(fr)
            if by_dir:
                print("  figure_ratio at attempts[best], per direction "
                      "(1.85-1.98 is the close-up signature on record):")
                for d in DIRECTIONS:
                    v = by_dir.get(d)
                    if v:
                        print(f"    {d:<5} n={len(v)}  min {min(v):.3f}  "
                              f"max {max(v):.3f}  mean {sum(v)/len(v):.3f}")
    print("\nA row whose reference is itself a close-up cannot be caught by a "
          "cell-vs-reference ratio, and a redraw re-conditions on that same "
          "reference - so re-rolling the cell cannot fix it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
