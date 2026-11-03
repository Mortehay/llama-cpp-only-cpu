#!/usr/bin/env python3
"""Does a bigger cell remove the apparent vertical slice? Re-fit, don't re-render.

`probe-slice-fitted.py` established that the straight vertical runs something2
sees are made by the downscale: the 512 px render never exceeds 0.238 of the
figure's height while the fitted 48x64 cell reaches 0.42-0.76 (SOMET-696).

The obvious follow-up is "then build it at 96x128 and look". That needs no GPU.
The defect is introduced by the FIT, so re-fitting the same `cut_*.png` at
several cell sizes isolates it exactly - and because the renders are identical
by construction, this is a cleaner comparison than two builds could be. Same
trick as `cell-size-cost.py`, which measures interior detail this way.

Uses `pixelate.pixelate`, the production fit, rather than a resize of its own:
a restated fit would answer a question about a function nobody runs.

The sheet-wide scale is the MINIMUM fit over every cell, so this computes that
minimum per cell size exactly as compose does, instead of fitting each sprite
alone - fitting alone would flatter every size equally and hide the real
production geometry.

CPU only, read-only: safe while a build holds the card.

    python3 scripts/cell-size-slice.py <job_id> [--cells 48x64,96x128]
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pixelate  # noqa: E402
from slice_metrics import longest_vertical_run  # noqa: E402

IMAGES = "/app/images"
DEFAULT_CELLS = "48x64,64x86,96x128,128x170"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id")
    ap.add_argument("--cells", default=DEFAULT_CELLS)
    ap.add_argument("--jitter", type=int, default=1)
    a = ap.parse_args(argv)

    work = os.path.join(IMAGES, f"_job_{a.job_id}")
    if not os.path.isdir(work):
        print(f"no work directory at {work}", file=sys.stderr)
        return 2
    cuts = sorted(f for f in os.listdir(work)
                  if f.startswith("cut_") and f.endswith(".png"))
    if not cuts:
        print("no cut_*.png in that work directory (pruned?)", file=sys.stderr)
        return 2

    imgs = [(f, Image.open(os.path.join(work, f)).convert("RGBA"))
            for f in cuts]
    print(f"job {a.job_id}: {len(imgs)} rendered cell(s)")
    print(f"{'cell':<10} {'sheet scale':<13} {'max run':<9} {'median':<9} "
          f"{'>=0.31':<8} {'figure px':<11} {'clipped':<8}")
    print("-" * 74)

    for spec in a.cells.split(","):
        cw, ch = (int(x) for x in spec.strip().split("x"))
        # compose pins the MINIMUM fit over every cell (sheet.SheetBuilder,
        # the `fit = min(fit, s)` loop). Computed the same way here, from the
        # same `_alpha_bbox` and the same margin, rather than fitting each
        # sprite alone - fitting alone would flatter every size equally and
        # hide the geometry production actually uses.
        margin = 1
        scale = None
        for _, img in imgs:
            box = pixelate._alpha_bbox(img, pixelate.DEFAULT_ALPHA_THRESHOLD)
            if box is None:
                continue
            w, h = box[2] - box[0], box[3] - box[1]
            s = min((cw - 2 * margin) / w, (ch - 2 * margin) / h)
            scale = s if scale is None else min(scale, s)
        if scale is None:
            continue
        fracs, areas, clipped = [], [], 0
        for _, img in imgs:
            cell = pixelate.pixelate(img, cw, ch, scale=scale)
            alpha = np.asarray(cell)[:, :, 3]
            areas.append(int((alpha > 0).sum()))
            # The named hazard of a cell-size change: `pixelate` pastes at a
            # centred offset and `Image.paste` CLIPS a negative one silently
            # instead of failing, so a sprite can lose its edge rather than be
            # downsized. Opaque pixels on the outermost row or column are the
            # tell, and a cell-size change is exactly when CLAUDE.md says this
            # starts firing - so it is checked here, before anyone switches.
            if (alpha[0].any() or alpha[-1].any()
                    or alpha[:, 0].any() or alpha[:, -1].any()):
                clipped += 1
            runs = longest_vertical_run(alpha, a.jitter)
            for side in ("left", "right"):
                run, _, fig = runs[side]
                if fig:
                    fracs.append(run / fig)
        if not fracs:
            continue
        fracs.sort()
        over = sum(1 for f in fracs if f >= 0.31)
        print(f"{spec:<10} {scale:<13.6f} {max(fracs):<9.3f} "
              f"{fracs[len(fracs)//2]:<9.3f} "
              f"{f'{over}/{len(fracs)}':<8} "
              f"{sum(areas)//len(areas):<11} "
              f"{f'{clipped}/{len(imgs)}':<8}")

    print("\n'>=0.31' is how many measurements would trip the straight-run "
          "check something2 proposed. A threshold that fires on most cells at "
          "the production size is not a defect detector.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
