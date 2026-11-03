"""Did the frame cut the figure off, as an exact fact rather than a proxy?

WHY THIS ONE IS NOT A PROXY
---------------------------
Every other framing signal tried on this problem is an indirect reading of
"is the figure complete", and all of them failed against hand labels: opaque
area, the occupied box's aspect, height fraction, bottom-band taper,
profile-vs-median, the CLIP bottom band and a Qwen3-VL judge. The last two
both scored 8/15 on ten labelled turnaround references - exactly the base
rate (`scripts/compare-framing-detectors.py`).

"The silhouette reaches row 0" is different in kind. If the head is cut by
the frame, the alpha MUST touch the top border; if it touches, something left
the frame. There is no threshold to calibrate and nothing to transfer across
subjects.

WHERE THE EVIDENCE LIVES
------------------------
`cut_*.png`, not the finished sheet. `cutout.cut_out` returns a full-size
RGBA with no crop and no pad, so the cutout keeps the alpha where the model
put it; `pixelate` centres the bounding box into the cell afterwards, which
is why measuring a composed sheet finds no opaque pixel on a cell's top row
and reads as "compose is not cropping". True, and also the one place the
evidence has been destroyed.

THE BOTTOM BORDER IS NOT A DEFECT
----------------------------------
A correctly framed standing figure has its feet ON the last row - the angles
LoRA frames edge to edge - so verified-good references touch the bottom too.
Width does not rescue it either: a verified thigh cut measured 0.402 of the
figure's width on its bottom row against a verified-good 0.416. This module
deliberately offers no bottom check.

A RAISED WEAPON ALSO TOUCHES THE TOP
-------------------------------------
On the first `wide` attack sheet the only three touching cells were `n_0`,
`ne_0` and `se_0` - all frame 0, whose pose is "weapon raised overhead". The
figures are complete and the heads entirely intact; what leaves the frame is
the sword blade. Sword tips measured top_w 0.092-0.158 against genuine mild
crown clips at 0.018-0.210, so WIDTH CANNOT SEPARATE THEM and an
unconditional per-frame check would fail every overhead attack frame forever,
with no redraw able to clear it.

Hence `overhead_pose`: the conveyor knows the pose string for every (action,
direction, frame), so the exemption needs no model. References need no
exemption - a turnaround is a neutral standing pose and nothing is overhead
in any of the eight.

WIDTH IS SEVERITY, NOT THE TEST.
"""
from __future__ import annotations

import os
import re

import numpy as np

# Recording-only by default. The verdict is computed and reported on every
# build; it does not fail a cell into the redraw loop unless this is on.
#
# The reason is a measured fire rate, not caution: on the medium-era idle
# sheet 7f9fc8f1 the top-touch subset is 11 of 30 cells, and `idle` has no
# overhead frames so the exemption saves none of them - a 37% rate on a sheet
# whose QA currently reports clean. On the first `wide` attack sheet it is 3
# of 30 and all three are exempt sword tips. Those are one data point each;
# record the verdict across a few builds before spending GPU on the trigger.
ENFORCE = os.environ.get("BORDER_QC_ENFORCE", "0").strip().lower() in (
    "1", "true", "yes")

# Words that put something above the head on purpose. Matched against the
# frame's pose text; anything here legitimately leaves the top of the frame.
# `behind the head` is here because the frame AFTER an overhead raise keeps
# the weapon up: attack front 1 and back 1 are "weapon at the top of the
# swing, tipped back behind the head" and "weapon tipped back behind the head,
# elbows flared wide". Omitting it left those two frames unexempted, which
# scripts/test-border-qc.py caught - the wind-up frames are exactly as likely
# to put a blade through the top border as the raise itself.
OVERHEAD = re.compile(
    r"\boverhead\b|\babove the head\b|\bover the head\b"
    r"|\bbehind the head\b|\braised high\b|\btop of the swing\b",
    re.I)


def overhead_pose(pose: str | None) -> bool:
    """Does this frame's pose text deliberately put something above the head?"""
    return bool(pose and OVERHEAD.search(pose))


def measure(mask: np.ndarray) -> dict | None:
    """Top-border contact for one figure mask. None when the mask is empty.

    `top_w` is the topmost opaque row's width as a fraction of the figure's
    own width, so it is comparable between subjects and resolutions. It is a
    SEVERITY score: a cut skull is a wide flat slice, an intact crown is
    narrow - but a raised blade is narrow too, so it does not decide anything.
    """
    solid = mask > 0
    ys, xs = np.nonzero(solid)
    if not ys.size:
        return None
    y0 = int(ys.min())
    fig_w = int(xs.max() - xs.min() + 1)
    flat = int(solid[y0].sum())
    return {
        "top_touch": y0 == 0,
        "top_y0": y0,
        "top_px": flat,
        "top_w": round(flat / max(fig_w, 1), 4),
    }


def is_cut(m: dict | None, pose: str | None = None,
           reference: bool = False) -> bool:
    """True when the frame ran off the TOP of the render and should not have.

    `pose` is the frame's pose text; ignored for a reference, which has no
    pose and never raises anything. A missing measurement is not a failure -
    that is the difference between "judged and passed" and "never judged",
    the same distinction `facing` draws on a weak reference.
    """
    if not m or not m.get("top_touch"):
        return False
    if not reference and overhead_pose(pose):
        return False
    return True
