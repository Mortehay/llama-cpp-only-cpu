#!/usr/bin/env python3
"""How much sprite SHAPE does the 48x64 cell throw away?

    cell-size-cost.py --job /app/images/_job_2ff8af36-...
    cell-size-cost.py --job <dir> --sizes 48x64,96x128 --json out.json

WHY THIS EXISTS

The owner's recurring complaint is that a sprite looks nothing like the
concept it came from. That was traced through the pipeline: the concept, the
turnaround views and the raw 512px cell all keep the identity, and it is the
fit into 48x64 that destroys it. 96x128 looked plainly better by eye.

By eye is the problem. `sheet_qc.identity` is a COLOUR HISTOGRAM and is
spatially blind - it scored 48x64 and 96x128 as equally far from the concept
while the 96px version was obviously more recognisable
(`.ai/specs/sheet-quality/contract.md` section 7). So the project has no
number for "the shape became mush", and the cell size is a CONTRACT question
with something2: their atlas consumer and in-game sprite scale both change
with it. That decision should rest on a measurement, not on my eyesight.

This runs on cells already on disk. No GPU, no BiRefNet - `cut_*.png` carries
the mask in its alpha channel already, so it is safe to run while a sheet
build holds the card.

WHAT IT MEASURES

For each `cut_*.png`, it fits the sprite into each requested cell size with
the real `pixelate()` path, upscales the result back to 512 with NEAREST, and
compares silhouettes against the original 512 mask:

  iou     intersection over union of the two silhouettes.
  px      opaque pixels in the fitted cell. The information budget: whatever
          the sprite has to say about itself, it says in this many pixels.
  grad    internal luminance gradient energy inside the mask, as a fraction
          of the 512 reference's. This is the INTERIOR detail - a face, a
          belt, the edge of a hood. Mush flattens it toward 0.

THE FIRST RESULT CORRECTED THE PREMISE, so do not re-derive it

This was built expecting the 48x64 SILHOUETTE to be the casualty. It is not.
Measured on 20 cells of 2ff8af36, median IoU is **flat across every size** -
0.6044 at 48x64 against 0.5943 at 128x170 - so the outline survives the
downscale intact and resolution buys nothing there. A `shape` metric
(perimeter^2/area) was tried alongside and DROPPED: it rose with cell size,
but a NEAREST upscale makes the boundary staircase finer as the cell grows,
so it was measuring quantisation rather than detail. Do not reinstate it.

What that leaves is the interior, which is why `grad` and `px` exist.

WHAT IT CANNOT TELL YOU

No number here says "a player recognises this". Read them beside the
rendered cells, never instead of them - the same rule the quarter-view hints
carry, where the wording that scored the largest raw pixel difference turned
out to be a zoom.

It also fits each sprite ALONE (`scale=None`) rather than at the sheet-wide
pinned scale. That is deliberate: the production scale is the MINIMUM fit
over every cell, so one bloated cell shrinks all of them, and measuring at
the pinned scale would report that contamination instead of the cost of the
cell size itself.
"""

import argparse
import json
import os
import statistics
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
for _c in (os.path.join(_here, "..", "src", "sprite_generator"),
           os.path.dirname(_here), "/app"):
    if os.path.isfile(os.path.join(_c, "pixelate.py")):
        sys.path.insert(0, _c)
        break

from PIL import Image  # noqa: E402

from pixelate import DEFAULT_ALPHA_THRESHOLD, pixelate  # noqa: E402


def mask_of(img: Image.Image, thr: int = DEFAULT_ALPHA_THRESHOLD) -> np.ndarray:
    return np.asarray(img.convert("RGBA"))[..., 3] >= thr


def luma(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("RGBA")).astype(np.float32)
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def grad_energy(y: np.ndarray, m: np.ndarray) -> float:
    """Mean |gradient| of luminance strictly INSIDE the mask.

    The silhouette edge is excluded on purpose: it is a huge step from the
    subject to transparency and would swamp the interior detail this is
    trying to see. A pair is counted only where both pixels are opaque, so
    the figure answers "how much structure is there within the body" rather
    than "how sharp is the outline".
    """
    dx = np.abs(np.diff(y, axis=1))
    dy = np.abs(np.diff(y, axis=0))
    mx = m[:, :-1] & m[:, 1:]
    my = m[:-1, :] & m[1:, :]
    n = int(mx.sum()) + int(my.sum())
    if n == 0:
        return 0.0
    return float((dx[mx].sum() + dy[my].sum()) / n)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--job", required=True, help="a job/work dir with cut_*.png")
    p.add_argument("--sizes", default="48x64,64x86,96x128,128x170")
    p.add_argument("--limit", type=int, default=0,
                   help="score only the first N cells (0 = all)")
    p.add_argument("--json", help="write per-cell rows here")
    a = p.parse_args()

    sizes = []
    for tok in a.sizes.split(","):
        w, _, h = tok.strip().partition("x")
        sizes.append((int(w), int(h)))

    cuts = sorted(f for f in os.listdir(a.job)
                  if f.startswith("cut_") and f.endswith(".png"))
    if a.limit:
        cuts = cuts[:a.limit]
    if not cuts:
        raise SystemExit("no cut_*.png in %s - a sheet built before BiRefNet "
                         "keying has only cell_*.png, which still has the "
                         "backdrop in it and cannot answer this" % a.job)

    print("job    %s" % os.path.basename(a.job.rstrip("/")))
    print("cells  %d" % len(cuts))

    rows, agg = [], {s: {"iou": [], "grad": [], "px": []} for s in sizes}
    for name in cuts:
        src = Image.open(os.path.join(a.job, name))
        ref = mask_of(src)
        if not ref.any():
            continue
        ref_grad = grad_energy(luma(src), ref)
        row = {"cell": name[4:-4]}
        for (cw, ch) in sizes:
            fitted = pixelate(src, cw, ch)
            px = int(mask_of(fitted).sum())
            back = fitted.resize(src.size, Image.Resampling.NEAREST)
            m = mask_of(back)
            union = int((m | ref).sum())
            iou = int((m & ref).sum()) / union if union else 0.0
            g = grad_energy(luma(back), m)
            grel = (g / ref_grad) if ref_grad else 0.0
            row["%dx%d" % (cw, ch)] = {"iou": round(iou, 4),
                                       "grad_rel": round(grel, 4), "px": px}
            agg[(cw, ch)]["iou"].append(iou)
            agg[(cw, ch)]["grad"].append(grel)
            agg[(cw, ch)]["px"].append(px)
        rows.append(row)

    print("\n%-10s %14s %16s %12s" % ("cell size", "silhouette IoU",
                                      "interior detail", "sprite px"))
    print("%-10s %14s %16s %12s" % ("", "median", "median (1.0=512)",
                                    "median"))
    for s in sizes:
        if not agg[s]["iou"]:
            continue
        print("%-10s %14.4f %16.4f %12d"
              % ("%dx%d" % s, statistics.median(agg[s]["iou"]),
                 statistics.median(agg[s]["grad"]),
                 int(statistics.median(agg[s]["px"]))))

    base = sizes[0]
    bg = statistics.median(agg[base]["grad"]) if agg[base]["grad"] else 0
    bp = statistics.median(agg[base]["px"]) if agg[base]["px"] else 0
    print("\nagainst %dx%d:" % base)
    for s in sizes[1:]:
        if agg[s]["grad"]:
            g = statistics.median(agg[s]["grad"])
            px = statistics.median(agg[s]["px"])
            print("  %-9s interior detail %+6.1f%%   pixels %+6.1f%%"
                  % ("%dx%d" % s, (g / bg - 1) * 100 if bg else 0,
                     (px / bp - 1) * 100 if bp else 0))
    print("\nSilhouette IoU is expected to be FLAT - the outline survives the "
          "downscale at every\nsize (measured, see the docstring). Interior "
          "detail is where the cell size bites.\nNeither number says a player "
          "recognises the sprite; read them beside the cells.")

    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"job": a.job, "sizes": ["%dx%d" % s for s in sizes],
                       "cells": rows}, fh, indent=1)
        print("\nper-cell rows -> %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
