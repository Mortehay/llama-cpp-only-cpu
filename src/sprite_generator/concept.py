"""Judge whether an image can serve as a CHARACTER concept.

Lives here, beside pixelate, rather than in scripts/ because BOTH the operator
tool (`scripts/check-concept.py`) and the job runner need it, and a file named
`check-concept.py` cannot be imported.

WHY THIS EXISTS

The conveyor accepts anything. Pointed at a landscape it produced twelve
structurally perfect cells of a tree - right column count, 24 colours, no
partial alpha, baselines aligned, `check-sprite` PASS - because check-sprite
validates the SHEET, not the subject. Thirteen minutes of GPU for a sheet of
scenery, and the only tell was that the cells were 31% transparent where a
character runs 70-84%.

ADR 0004 recorded the same requirement from the 3D side, where the cost was
higher: a goblin inside a decorative frame reconstructed the FRAME as geometry.
The constraint on a concept is **isolation**, not style - one character, no
frame, no scenery, keyed background.

Three measurements separate a character from a scene, none needing a model:

    coverage  how much of the frame survives keying. A character occupies a
              minority of its frame; a scene fills it.
    border    how much of the outermost ring survives. A framed or full-bleed
              image keeps its edges; an isolated subject does not.
    aspect    height/width of the opaque bounding box. Humanoids are taller
              than wide.

The thresholds separate the two classes actually observed here rather than
being fitted precisely: characters measured 4-20% coverage with a 0% border,
scenes 54-99% coverage with a 20-94% border. That is a wide gap, so the exact
values matter little - which is the point, after three ground-patch rules that
failed because a constant was tuned on one sample.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

import pixelate

MAX_COVERAGE = 0.60
MAX_BORDER = 0.25
MIN_ASPECT = 1.20


def judge(path_or_image, tolerance: int = 10) -> dict:
    """Measure a concept. Returns a dict with `ok`, `why` and the three numbers.

    Accepts a path or a PIL image, because the job runner has a path and the
    UI may one day have neither.
    """
    img = (Image.open(path_or_image)
           if isinstance(path_or_image, (str, bytes))
           else path_or_image)
    keyed = pixelate.key_background(img, tolerance=tolerance)
    arr = np.asarray(keyed.convert("RGBA"))
    opaque = arr[..., 3] >= 128

    coverage = float(opaque.mean())

    ring = np.zeros_like(opaque)
    ring[0, :] = ring[-1, :] = True
    ring[:, 0] = ring[:, -1] = True
    border = float(opaque[ring].mean())

    ys, xs = np.nonzero(opaque)
    if len(ys) == 0:
        return {"ok": False, "why": "nothing survives keying",
                "coverage": 0.0, "border": 0.0, "aspect": 0.0}

    aspect = (ys.max() - ys.min() + 1) / max(xs.max() - xs.min() + 1, 1)

    reasons = []
    if coverage > MAX_COVERAGE:
        reasons.append(f"fills the frame ({coverage:.0%} opaque)")
    if border > MAX_BORDER:
        reasons.append(f"reaches the border ({border:.0%} of the edge)")
    if aspect < MIN_ASPECT:
        reasons.append(f"not taller than wide (aspect {aspect:.2f})")

    return {"ok": not reasons, "why": "; ".join(reasons) or "ok",
            "coverage": coverage, "border": border, "aspect": float(aspect)}


def describe(verdict: dict) -> str:
    """One line naming the measurement, for an error a caller has to act on."""
    return (f"{verdict['why']} (coverage {verdict['coverage']:.0%}, "
            f"border {verdict['border']:.0%}, aspect {verdict['aspect']:.2f})")
