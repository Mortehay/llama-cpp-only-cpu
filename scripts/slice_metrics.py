"""The straight-vertical-run metric, shared by the two slice probes.

One definition, imported by both, because the whole point of the second probe
is to compare the first probe's number at a different resolution. Two copies
of this function would make a disagreement between them unreadable - and a
hand-restated metric agreeing with the misunderstanding that wrote it is a
mistake this repo has already made once (see
scripts/test-isolate-mirrors-tasks.py).

Importable name (underscores, no hyphens) so the probes can `import
slice_metrics` - the probe scripts themselves cannot be imported.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def longest_vertical_run(alpha: np.ndarray, jitter: int = 1) -> dict:
    """Longest run of consecutive rows whose silhouette edge barely moves.

    Returns {"left": (run_rows, at_column, figure_rows), "right": ...}.

    A real limb silhouette wanders from row to row; a cut does not. `jitter`
    is how many pixels of wander still counts as "the same column" - 1 by
    default, because an anti-aliased edge moves by a pixel without being
    straight.

    Measured against the FIGURE's own row count, not the image's, so the
    number is comparable between a 512 px render and a 48x64 cell. That
    comparison is the only reason this module exists.
    """
    out = {}
    rows = [np.flatnonzero(r) for r in (alpha > 0)]
    present = [i for i, r in enumerate(rows) if r.size]
    if not present:
        return {"left": (0, None, 0), "right": (0, None, 0)}
    fig_rows = len(present)
    for side, pick in (("left", lambda r: r[0]), ("right", lambda r: r[-1])):
        best, best_col, run, anchor = 0, None, 0, None
        for i in present:
            col = int(pick(rows[i]))
            if anchor is not None and abs(col - anchor) <= jitter:
                run += 1
            else:
                run, anchor = 1, col
            if run > best:
                best, best_col = run, anchor
        out[side] = (best, best_col, fig_rows)
    return out


def alpha_of(path: str) -> np.ndarray:
    """The alpha channel, or a crude key for a render that has none.

    A raw `cell_*.png` is RGB: near-white is treated as background so its
    silhouette is comparable to the keyed `cut_*.png` beside it. Crude on
    purpose - this looks for a straight EDGE, it does not grade a cutout.
    """
    img = Image.open(path)
    if img.mode != "RGBA":
        g = np.asarray(img.convert("L"))
        return (g < 245).astype(np.uint8) * 255
    return np.asarray(img)[:, :, 3]
