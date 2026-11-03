#!/usr/bin/env python3
"""Where does the straight vertical cut on side-view cells come from?

something2 reported (2026-11-03) that e/w frames come back with one side of
the figure sliced off by a straight vertical line, that redrawing does not fix
it, and that sheet QA passes every one of those frames. Their measurement was
on the FITTED 48x64 cells, which cannot say which stage did it.

This reads the per-frame stages the conveyor leaves in the work directory and
measures the same thing at each one, so the answer is "stage X introduced it"
rather than "it is there by the end":

    cell_*.png   the raw 512px render, before any keying
    cut_*.png    after BiRefNet removed the background
    (fitted)     the sheet cell, reconstructed by the same fit

A straight vertical run means: for N consecutive rows, the leftmost (or
rightmost) opaque pixel sits within `--jitter` px of the same column. A real
limb silhouette wanders; a cut does not.

Reports the run as a FRACTION of the figure's own height, not the cell's, so
one number is comparable across stages at different resolutions - and prints
whether the run sits on the image border, which distinguishes "the subject was
drawn against the edge" from "something sliced through the middle of it".

Read-only and CPU-only: it opens finished PNGs, so it is safe to run while a
build holds the card.

    python3 scripts/probe-vertical-slice.py <job_id> [--action idle]
                                            [--jitter 1] [--json out.json]
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

WORK = "/app/images"
DIRECTIONS = ["s", "se", "e", "ne", "n", "nw", "w", "sw"]

# The metric lives in slice_metrics so probe-slice-fitted.py measures the
# IDENTICAL thing at 48x64 - comparing the two is the point.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slice_metrics import alpha_of, longest_vertical_run  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id")
    ap.add_argument("--action", default=None,
                    help="only this action (default: every action found)")
    ap.add_argument("--jitter", type=int, default=1)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    work = os.path.join(WORK, f"_job_{a.job_id}")
    if not os.path.isdir(work):
        print(f"no work directory for {a.job_id} at {work}", file=sys.stderr)
        return 2

    names = sorted(os.listdir(work))
    rows = []
    for name in names:
        if not name.endswith(".png"):
            continue
        stage = ("cut" if name.startswith("cut_")
                 else "render" if name.startswith("cell_") else None)
        if stage is None:
            continue
        stem = name[4:-4] if stage == "cut" else name[5:-4]
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        action, direction, frame = parts[0], parts[1], parts[-1]
        if a.action and action != a.action:
            continue
        if direction not in DIRECTIONS:
            continue
        alpha = alpha_of(os.path.join(work, name))
        h, w = alpha.shape
        runs = longest_vertical_run(alpha, a.jitter)
        for side, (run, col, fig_rows) in runs.items():
            if not fig_rows:
                continue
            on_border = col is not None and (col <= 1 or col >= w - 2)
            rows.append({
                "action": action, "direction": direction, "frame": frame,
                "stage": stage, "side": side, "run_rows": run,
                "figure_rows": fig_rows,
                "run_frac": round(run / fig_rows, 4),
                "at_col": col, "img_w": w, "img_h": h,
                "on_border": bool(on_border),
            })

    if not rows:
        print("no per-frame stages found - this job's work dir may be pruned")
        return 2

    # Side views are the complaint; everything else is the control.
    def bucket(d):
        return "side (e/w)" if d in ("e", "w") else "other"

    print(f"job {a.job_id}: {len(rows)} measurement(s)\n")
    for stage in ("render", "cut"):
        sub = [r for r in rows if r["stage"] == stage]
        if not sub:
            continue
        print(f"--- {stage} ---")
        for b in ("side (e/w)", "other"):
            vals = [r["run_frac"] for r in sub if bucket(r["direction"]) == b]
            if not vals:
                continue
            border = sum(1 for r in sub
                         if bucket(r["direction"]) == b and r["on_border"])
            print(f"  {b:<12} n={len(vals):<3} max run {max(vals):.3f} "
                  f"of figure height, median {sorted(vals)[len(vals)//2]:.3f}"
                  f"  ({border} of {len(vals)} on the image border)")
        worst = sorted(sub, key=lambda r: -r["run_frac"])[:5]
        print("  worst:")
        for r in worst:
            print(f"    {r['action']:<8} {r['direction']:<3} f{r['frame']:<3} "
                  f"{r['side']:<5} run {r['run_rows']:>4}/{r['figure_rows']:<4}"
                  f" = {r['run_frac']:.3f} at col {r['at_col']} of "
                  f"{r['img_w']}{'  ON BORDER' if r['on_border'] else ''}")
        print()

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
        print(f"per-measurement rows -> {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
