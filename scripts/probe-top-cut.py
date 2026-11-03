#!/usr/bin/env python3
"""Is the head cut flat by the TOP of the 512px render?

something2 reported (2026-11-03) that heads are cut flat across a 96x128 idle
sheet, including on FRONT views, which the profile-close-up story does not
explain. They measured no opaque pixels on a cell's top row - so compose is
not clipping - but a wide flat topmost row sitting 19-33 px below the cell
top, which is what a figure whose head was already cut at 512 looks like once
it is centred in a cell.

THIS IS NOT A PROXY, and that is the point. Every framing metric tried so far
- area, aspect, h_frac, bottom taper, profile-vs-median - is an indirect
reading of "is the figure complete", and all five failed on hand-labelled
references. "The alpha reaches row 0" is an exact geometric fact about the
render. If the head is cut by the frame, the silhouette must touch the top
border; if it touches, something left the frame.

The cutout does NOT hide it: `cutout.cut_out` returns a full-size RGBA with no
crop and no pad, so `cut_*.png` is still 512x512 with the alpha where the
model put it. The centring that produces the observed margin happens later, in
`pixelate.pixelate`, when the bounding box is fitted into the cell. So the
touch is measurable on the cutout and is destroyed by the fit - which is why
measuring the finished sheet found "no opaque pixels on the top row" and
concluded, wrongly, that nothing was cut.

Reports per image:
  top_touch   does any opaque pixel sit on row 0
  top_w       width of the topmost opaque row, as a fraction of the figure's
              own width - a cut skull is a wide flat slice, an uncut head
              starts as a narrow crown
  flat_px     absolute width of that row in pixels, for comparison with
              something2's >=12 px threshold

Read-only, CPU-only: safe while a build holds the card.

    python3 scripts/probe-top-cut.py <job_id> [--refs] [--cells]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slice_metrics import alpha_of  # noqa: E402

IMAGES = "/app/images"


def measure(path: str) -> dict | None:
    solid = alpha_of(path) > 0
    ys, xs = np.nonzero(solid)
    if not ys.size:
        return None
    h = solid.shape[0]
    y0, y1 = int(ys.min()), int(ys.max())
    fig_w = int(xs.max() - xs.min() + 1)
    flat = int(solid[y0].sum())
    bot = int(solid[y1].sum())
    # TOP and BOTTOM are DIFFERENT DEFECTS and must be reported apart. A head
    # cut by the frame touches row 0; legs cut by the frame touch the last
    # row. Job 02cb757f is the case that forces the distinction: its `dir_e`
    # starts at y0=8 - no head cut at all - and is still unusable because the
    # figure runs off the BOTTOM at mid-thigh. Reporting only the top would
    # have called that reference clean.
    return {
        "top_touch": y0 == 0,
        "bot_touch": y1 == h - 1,
        "y0": y0,
        "y1_gap": h - 1 - y1,
        "flat_px": flat,
        "top_w": round(flat / max(fig_w, 1), 3),
        "bot_w": round(bot / max(fig_w, 1), 3),
        "fig_w": fig_w,
    }


def report(label: str, rows: list[tuple[str, dict]]) -> tuple[int, int]:
    if not rows:
        return 0, 0
    print(f"\n=== {label}  ({len(rows)} image(s))")
    print(f"  {'image':<26} {'y0':>4} {'gap':>4} {'TOP':>4} {'BOT':>4} "
          f"{'top_w':>6} {'bot_w':>6}  verdict")
    touching = 0
    for name, m in rows:
        bad = m["top_touch"] or m["bot_touch"]
        touching += bool(bad)
        why = ("head cut" if m["top_touch"] and not m["bot_touch"] else
               "legs cut" if m["bot_touch"] and not m["top_touch"] else
               "BOTH ends cut" if bad else "")
        print(f"  {name:<26} {m['y0']:>4} {m['y1_gap']:>4} "
              f"{('YES' if m['top_touch'] else '-'):>4} "
              f"{('YES' if m['bot_touch'] else '-'):>4} "
              f"{m['top_w']:>6.3f} {m['bot_w']:>6.3f}  {why}")
    print(f"  running off a border: {touching}/{len(rows)}")
    return touching, len(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_ids", nargs="+")
    ap.add_argument("--refs", action="store_true", help="dir_<d>.png only")
    ap.add_argument("--cells", action="store_true", help="cut_*.png only")
    ap.add_argument("--limit", type=int, default=40)
    a = ap.parse_args(argv)
    both = not (a.refs or a.cells)

    grand_t = grand_n = 0
    for job in a.job_ids:
        work = os.path.join(IMAGES, f"_job_{job}")
        if not os.path.isdir(work):
            print(f"{job}: no work directory", file=sys.stderr)
            continue
        cfg_p = os.path.join(work, "config.json")
        seed = json.load(open(cfg_p)).get("seed") if os.path.exists(cfg_p) else "?"
        print(f"\n######## job {job[:8]}  seed {seed}")

        if a.refs or both:
            rows = []
            for p in sorted(glob.glob(os.path.join(work, "dir_*.png"))):
                m = measure(p)
                if m:
                    rows.append((os.path.basename(p), m))
            t, n = report("turnaround references", rows)
            grand_t, grand_n = grand_t + t, grand_n + n

        if a.cells or both:
            cuts = sorted(glob.glob(os.path.join(work, "cut_*.png")))
            rows = []
            for p in cuts[:a.limit]:
                m = measure(p)
                if m:
                    rows.append((os.path.basename(p), m))
            t, n = report(f"cell cutouts (first {len(rows)} of {len(cuts)})", rows)
            grand_t, grand_n = grand_t + t, grand_n + n

    print(f"\nTOTAL touching the top border: {grand_t}/{grand_n}")
    print("A touch means the silhouette runs off the top of the 512px render, "
          "so something left the frame. The fit then centres the bounding box "
          "and gives the cell a margin, which is why the finished sheet shows "
          "no opaque pixel on its top row while the head is still missing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
