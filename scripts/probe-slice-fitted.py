#!/usr/bin/env python3
"""Does the straight vertical cut exist in the RENDER, or only after the fit?

something2 measured straight vertical silhouette runs of 20-26 scanlines on
e/w cells of a 48x64 sheet and read them as the figure being sliced during
generation. `probe-vertical-slice.py` measured the 512 px renders of the same
job and found the side views had the SHORTEST runs of any direction, so the
two measurements disagree about where the defect lives.

This settles it by measuring the same quantity at both resolutions on the same
cells: it slices the finished sheet PNG with its own atlas, measures the
fitted cell, and measures that cell's 512 px source beside it.

The hypothesis worth ruling out first is that the fit CREATES the run. A
~10x nearest-neighbour downscale quantises an edge that wanders by a couple of
pixels at 512 onto a single column for many rows, which looks exactly like a
cut and cannot be fixed by redrawing - which is what they observe.

Read-only, CPU-only, safe while a build holds the card.

    python3 scripts/probe-slice-fitted.py <job_id> [--jitter 1]
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

IMAGES = "/app/images"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slice_metrics import longest_vertical_run  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id")
    ap.add_argument("--jitter", type=int, default=1)
    a = ap.parse_args(argv)

    sheet_p = os.path.join(IMAGES, f"sheet_{a.job_id}.png")
    atlas_p = os.path.join(IMAGES, f"sheet_{a.job_id}.json")
    work = os.path.join(IMAGES, f"_job_{a.job_id}")
    for p in (sheet_p, atlas_p):
        if not os.path.exists(p):
            print(f"missing {p}", file=sys.stderr)
            return 2

    sheet = Image.open(sheet_p).convert("RGBA")
    atlas = json.load(open(atlas_p, encoding="utf-8"))
    # The atlas spells the cell as {"cell": {"w": .., "h": ..}} and names a
    # mirrored row in a top-level {"mirrored": {"w": "e", ...}} map; the
    # per-frame `mirrored_from` lives in `quality`, not in `rows`.
    cell = atlas.get("cell") or {}
    cw, ch = int(cell.get("w", 48)), int(cell.get("h", 64))
    nframes = int(atlas.get("frames_per_action")
                  or (atlas.get("grid") or {}).get("columns") or 1)
    mirrored = atlas.get("mirrored") or {}
    print(f"job {a.job_id}: cell {cw}x{ch}, sheet {sheet.size}, "
          f"{nframes} frame(s)/action, mirrored rows {sorted(mirrored)}")
    print()

    rows = atlas.get("rows") or []
    print(f"{'row':<16} {'f':<3} {'side':<6} {'fitted':<18} {'render(512)':<18}")
    print("-" * 68)
    worst = []
    for r in rows:
        action, direction = r.get("action"), r.get("direction")
        ry = int(r.get("row", 0)) * ch
        # A mirrored row's pixels come from the eastern cell, so the 512 px
        # source to compare against is that one - following the row name would
        # look for a cut_*.png that was never rendered.
        src_dir = mirrored.get(direction, direction)
        for f in range(nframes):
            box = (f * cw, ry, (f + 1) * cw, ry + ch)
            cell_alpha = np.asarray(sheet.crop(box))[:, :, 3]
            fit = longest_vertical_run(cell_alpha, a.jitter)
            cut = os.path.join(work, f"cut_{action}_{src_dir}_{f}.png")
            raw = None
            if os.path.exists(cut):
                raw = longest_vertical_run(
                    np.asarray(Image.open(cut).convert("RGBA"))[:, :, 3],
                    a.jitter)
            for side in ("left", "right"):
                fr, _, fh = fit[side]
                line = (f"{action}_{direction:<10} {f:<3} {side:<6} "
                        f"{fr:>3}/{fh:<3} {fr / max(fh, 1):.2f}      ")
                if raw:
                    rr, _, rh = raw[side]
                    line += f"{rr:>3}/{rh:<3} {rr / max(rh, 1):.2f}"
                    worst.append((fr / max(fh, 1), rr / max(rh, 1),
                                  f"{action} {direction} f{f} {side}"))
                else:
                    line += "   (no 512px source)"
                print(line)

    if worst:
        fits = [w[0] for w in worst]
        raws = [w[1] for w in worst]
        print(f"\n{'':<20}{'fitted':>10}{'render':>10}")
        print(f"{'max run / height':<20}{max(fits):>10.3f}{max(raws):>10.3f}")
        print(f"{'median':<20}{sorted(fits)[len(fits)//2]:>10.3f}"
              f"{sorted(raws)[len(raws)//2]:>10.3f}")
        amp = [f / r for f, r in zip(fits, raws) if r > 0]
        if amp:
            print(f"{'fit amplification':<20}{sum(amp)/len(amp):>10.2f}x "
                  f"(median {sorted(amp)[len(amp)//2]:.2f}x)")
        print("\nIf the fitted column is far larger than the render column, the "
              "straight run is made by the downscale, not by generation - and "
              "no redraw can remove it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
