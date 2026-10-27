"""Per-frame quality checks for the sheet conveyor.

WHAT THIS IS FOR

A finished sheet used to be accepted structurally: right column count, 24
colours, no partial alpha, baselines aligned. `check-sprite` validates the
GRID, not the pictures in it, so four defects reached the game unflagged - a
character facing the opposite way from the row it is in, a cell the model
recomposed into a chest close-up, backdrop baked in behind the figure, and
identity drifting across an action.

Every number here was measured on the two evidence jobs 212d56d7 (flat
concept) and 8d9d164d (painterly concept), 336 raw 512px cells each, which
are the sheets the owner called unusable. The thresholds sit in the gap
between the clean population and the named failures, not at a round number.

WHAT IS MEASURED, AND ON WHAT

On the RAW 512px render, before the cell is fitted into 48x64. The fitted
cell cannot answer these questions:

  * height carries no signal at all. Pre-fit figure height is 478-512 of 512
    on all 336 cells of 8d9d164d, median 512 - the LoRA frames every subject
    edge to edge. A torso close-up is 512 tall and so is a full body. The
    first draft of this module compared heights, as the design brief asked,
    and would have passed every close-up in both sheets.
  * the sheet-wide scale is the MINIMUM fit scale over all cells
    (sheet.SheetBuilder.build), so a single bloated cell shrinks all 336.
    Measuring after the fit measures that contamination instead of its cause.

AREA is the signal height is not. Against the direction's reference:

    clean cells (idle, walk, both sheets)       1.07 - 1.09 x
    8d9d164d attack s f2 / f4 (chest close-up)  1.85 / 1.98 x

TWO MASKS, BECAUSE TWO DEFECTS LOOK IDENTICAL IN ONE

The something2 spike could not tell a close-up from baked backdrop: both
present as "lots of opaque pixels", and its backdrop rule fired zero times
because those cells had already been taken by its close-up rule. The split is
to measure the frame twice:

    figure = BiRefNet (cutout.cut_out)   - what is actually the character
    flood  = corner flood fill           - what the old compose kept

and read the DIFFERENCE as backdrop. Measured on the named cells:

    cell                        flood%  figure%  backdrop%
    8d9d164d idle s f0           45.5    42.0      3.6   clean
    8d9d164d walk e f0           44.1    40.9      3.3   clean
    8d9d164d attack se f3        76.8    37.7     39.1   painted checker
    8d9d164d cast s f1           55.0    34.2     22.7   spell glow
    8d9d164d attack w f1         51.0    38.4     12.6   white blob
    8d9d164d attack se f4        92.6    85.3      9.0   checker + close-up
    8d9d164d attack s f2         78.6    77.5      1.1   close-up, no backdrop

The clean population sits at 3.1-3.6% and every named backdrop cell at 9% or
more, so one number separates them and the close-up stays out of it.

`backdrop%` is the SET DIFFERENCE and not the difference of the two columns -
see `measure`. On attack se f4 the two disagree by enough to change the
verdict.

WHAT THE RATES LOOK LIKE ON A WHOLE SHEET

Scored over all 336 cells of each evidence job, the flags land where the
owner said the sheets were bad, which is the only external check this module
has:

    action   212d56d7   8d9d164d        direction  212d56d7  8d9d164d
    idle        5/48       2/48         s              3/42     11/42
    walk        3/48       3/48         se             8/42     20/42
    sway        3/48      10/48         e             12/42     11/42
    use        15/48      25/48         ne            26/42     25/42
    attack     14/48      22/48         n              6/42      3/42
    damage     17/48      21/48         nw            19/42     14/42
    cast       29/48      40/48         w              8/42     23/42
                                        sw             4/42     16/42

    totals: 89/336 (26.5%) and 123/336 (36.6%)

idle, walk and sway - the rows nobody complained about - stay near zero,
while cast, use, attack and damage carry the flags. `ne` is the worst
direction on both sheets, which is the row the brief describes as
alternating between a back view and a side view. `close_up` fires 40 times
on the painterly sheet and once on the flat one; `backdrop` 29 and 8. Both
splits match the owner's description of the two sheets.

8d9d164d's `w` row contributes 23 of its 123 flags, 19 of them facing. That
row is no longer rendered at all - it is a flip of `e` - so those flags
disappear rather than being redrawn.

FACING: `head_lean`, NOT silhouette IoU

The spike's silhouette IoU "MISSED Cultist:1 attack e f3", and that is the
cell the owner pointed at. It misses it because the anchor is the idle frame,
whose silhouette is a narrow standing figure - a big pose scores low against
it both ways round and the margin is noise.

`head_lean` asks a different question: where does the top of the figure sit
relative to its hips? That is stable under pose change and flips sign with
the facing. Anchored per direction, because the value is a property of the
VIEW (a back view leans the other way from a front view), measured:

    anchor    dir_e   dir_w   dir_s   dir_n
    212d56d7 -0.118  -0.176  -0.174  +0.177
    8d9d164d -0.096  +0.098  -0.193  +0.208

    212d56d7 attack e  f0..f5  -0.106 -0.119 -0.177 +0.196 -0.016 -0.113
    8d9d164d attack e  f0..f5  -0.090 -0.075 -0.116 +0.101 -0.079 -0.079
    212d56d7 idle   ne f0..f5  +0.116 -0.087 +0.116 -0.088 -0.101 +0.118
    212d56d7 walk   e  f0..f5  -0.078 -0.097 -0.115 -0.104 -0.113 -0.114

f3 of `attack e` flips sign on BOTH sheets, so that cell is a property of the
pose ("arms fully extended forward, strike landed") rather than of a seed.
`idle ne` alternates exactly as the brief described. The two clean walk rows
hold their sign on 12 of 12 frames.

Note 212d56d7's dir_e and dir_w share a sign: that sheet's west view is the
known duplicate of its east view. The check reports it rather than hiding it.

IDENTITY: masked colour histogram against THAT DIRECTION'S view

Against the front concept a colour histogram "is useless: it fires on every
back view" - correct, and the reason the anchor here is the turnaround view
for the same direction. Restricted to the BiRefNet mask so a backdrop cannot
carry the comparison. Chi-square, measured per row (f0..f5):

    8d9d164d idle s   0.009 0.009 0.009 0.009 0.008 0.008   clean
    8d9d164d walk s   0.007 0.009 0.008 0.010 0.009 0.008   clean
    8d9d164d cast se  0.082 0.024 0.064 0.026 0.021 0.015   glow + drift
    8d9d164d use  se  0.024 0.012 0.095 0.020 0.016 0.019   drift
    212d56d7 cast s   0.092 0.029 0.051 0.017 0.022 0.015   glow + drift

It does not separate "drifted" from "a large bright thing was added", and it
is not meant to: both are defects. The cast row scores high on BOTH sheets,
which is the glow the pose text asks for by name.
"""

from __future__ import annotations

import logging
import os

import numpy as np
from PIL import Image

import framing

logger = logging.getLogger(__name__)


def _f(name: str, default: float) -> float:
    """Env override for a threshold. They were tuned by re-scoring finished
    jobs at different cuts, and that loop needs them settable without a
    rebuild."""
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Clean 1.07-1.09x, close-ups 1.85-1.98x. 1.40 leaves room for a genuinely
# wide pose (arms thrown wide measured 1.0-1.2x) without reaching a close-up.
FIGURE_RATIO_MAX = _f("QC_FIGURE_RATIO_MAX", 1.40)

# Clean 3.1-3.6%, named backdrops 9.0-39.1%.
BACKDROP_MAX = _f("QC_BACKDROP_MAX", 0.08)

# After the cutout the figure should not fill the frame edge to edge
# horizontally. LEFT and RIGHT only - see border_fraction.
BORDER_MAX = _f("QC_BORDER_MAX", 0.25)

# Deadband around zero, applied to BOTH the cell and its reference. A
# near-frontal view has a small lean of either sign and flipping it is not
# evidence of anything.
#
# 0.04 rather than 0.02, from the distribution of the calls it makes. Over
# both evidence sheets the flagged cells' |lean| runs 0.020 to 0.332 with a
# median of 0.065-0.116, so the bottom of that range is cells sitting on the
# line. Raising the deadband drops 212d56d7 from 45 facing calls to ~38 and
# 8d9d164d from 69 to ~50 and keeps every strong one - attack e f3 at 0.196,
# the idle ne alternation at 0.116. A marginal call costs a redraw, and a
# redraw of a cell that was never wrong is pure GPU time.
LEAN_MIN = _f("QC_LEAN_MIN", 0.04)

# Clean 0.006-0.010, drifted 0.021-0.095.
IDENT_MAX = _f("QC_IDENT_MAX", 0.030)

# A redrawn frame against its surviving neighbours. Looser than IDENT_MAX:
# neighbours differ by a pose as well as by whatever drift there is.
CONTINUITY_MAX = _f("QC_CONTINUITY_MAX", 0.060)

ALPHA = 128

CHECKS = ("close_up", "backdrop", "border", "facing", "identity",
          "continuity", "framing")


# --- masks --------------------------------------------------------------

def flood_mask(img: Image.Image, tolerance: int = 10,
               min_enclosed: int = 200) -> np.ndarray:
    """What `pixelate.key_background` would keep, computed with labels.

    Same rule - border-connected near-background, plus enclosed
    near-background regions above a size floor - but vectorised. The original
    walks a Python stack one pixel at a time, which is a measured 20-40x
    slower here and is called once per cell; scoring 336 cells with it took
    minutes.

    Only used as the REFERENCE for `backdrop_frac`. The production cutout is
    BiRefNet (cutout.cut_out); this is what it is being compared against.
    """
    from scipy import ndimage

    rgba = np.asarray(img.convert("RGBA"))
    h, w = rgba.shape[:2]
    rgb = rgba[..., :3].astype(np.int16)
    seeds = [rgb[0, 0], rgb[0, w - 1], rgb[h - 1, 0], rgb[h - 1, w - 1]]

    near = np.zeros((h, w), bool)
    for s in seeds:
        near |= (np.abs(rgb - s).max(axis=2) <= tolerance)

    lab, n = ndimage.label(near)
    if n == 0:
        return np.ones((h, w), bool)
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    border.discard(0)
    sizes = ndimage.sum(near, lab, range(1, n + 1))
    drop = np.zeros(n + 1, bool)
    for i in range(1, n + 1):
        if i in border or (min_enclosed > 0 and sizes[i - 1] >= min_enclosed):
            drop[i] = True
    return ~drop[lab]


def figure_mask(img: Image.Image) -> np.ndarray:
    """The character, per BiRefNet. Falls back to the flood fill without it.

    The fallback is recorded rather than silent: a flood-fill figure mask
    makes `backdrop_frac` structurally zero and the backdrop check
    meaningless, so a caller has to be able to tell which engine ran.
    """
    import cutout

    if cutout.enabled("cuda"):
        try:
            return np.asarray(cutout.cut_out(img))[..., 3] >= ALPHA
        except Exception as e:
            logger.warning("sheet_qc: BiRefNet failed (%s); using flood fill", e)
    return flood_mask(img)


def engine() -> str:
    """Which mask engine `figure_mask` will use, for the report."""
    import cutout

    return "birefnet" if cutout.enabled("cuda") else "floodfill"


# --- primitives ---------------------------------------------------------

def head_lean(mask: np.ndarray) -> float | None:
    """Where the top of the figure sits relative to its hips, as a fraction of
    its own width. Positive means the head leads to screen-right.

    Bands are 0-18% and 35-75% of the figure's height: head and shoulders
    against hips and thighs. Both are measured from the figure's own bbox, so
    the value does not move when the character is drawn larger or smaller.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) < 50:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    span = y1 - y0 + 1
    width = x1 - x0 + 1
    if width < 8:
        return None

    def centre(lo: float, hi: float):
        band = mask[y0 + int(span * lo):y0 + int(span * hi), :]
        _, bx = np.nonzero(band)
        return bx.mean() if len(bx) else None

    top, hips = centre(0.0, 0.18), centre(0.35, 0.75)
    if top is None or hips is None:
        return None
    return float((top - hips) / width)


def colour_hist(img: Image.Image, mask: np.ndarray, bins: int = 6):
    """Normalised RGB histogram over the masked pixels only."""
    arr = np.asarray(img.convert("RGB"))
    if mask.sum() < 50:
        return None
    q = arr[mask] // (256 // bins)
    h, _ = np.histogramdd(q, bins=(bins,) * 3, range=((0, bins),) * 3)
    total = h.sum()
    if total <= 0:
        return None
    return (h / total).ravel()


def chi2(p, q) -> float:
    """Chi-square distance between two normalised histograms."""
    if p is None or q is None:
        return float("nan")
    return float(0.5 * np.sum((p - q) ** 2 / (p + q + 1e-9)))


def border_fraction(mask: np.ndarray) -> float:
    """Opaque share of the LEFT and RIGHT frame edges.

    Not the top and bottom: the angles LoRA frames every subject edge to edge
    vertically (measured, 478-512 of 512 on every cell of both evidence jobs),
    so a full top row is the normal case and including it would make this fire
    on everything.
    """
    return float((mask[:, 0].mean() + mask[:, -1].mean()) / 2.0)


# --- the per-direction anchor -------------------------------------------

class Reference:
    """What every comparison is made against: the turnaround view for that
    direction.

    Not the front concept, and not the sheet as a whole. It is the image the
    action cells were conditioned on, so it is the only thing they can fairly
    be asked to look like.
    """

    def __init__(self, img: Image.Image):
        self.mask = figure_mask(img)
        self.figure_frac = float(self.mask.mean())
        self.lean = head_lean(self.mask)
        self.hist = colour_hist(img, self.mask)

    @classmethod
    def from_path(cls, path: str):
        if not os.path.isfile(path):
            return None
        return cls(Image.open(path))

    def as_dict(self) -> dict:
        return {"figure_frac": self.figure_frac, "lean": self.lean}


# --- one frame ----------------------------------------------------------

def analyse(img: Image.Image, view: Reference | None = None):
    """(metrics, cut_out_RGBA) from ONE pass over the masks.

    The conveyor needs both for every cell and the figure mask costs a
    measured ~1.8 s of GPU each; computing it twice would add ten minutes to a
    336-cell sheet for nothing.

    The alpha is hard 0/255 at the threshold, like `cutout.cut_out`: a
    semi-transparent fringe has no meaning in pixel art and some atlas packers
    bleed it back in.
    """
    fig = figure_mask(img)
    m = measure(img, view, fig=fig)

    rgba = np.asarray(img.convert("RGBA")).copy()
    rgba[..., 3] = np.where(fig, 255, 0)
    # Zero the colour under cleared pixels for the same reason pixelate does.
    rgba[~fig, :3] = 0
    return m, Image.fromarray(rgba, "RGBA")


def measure(img: Image.Image, view: Reference | None = None,
            fig: np.ndarray | None = None) -> dict:
    """Every metric for one raw cell. No thresholds applied."""
    fig = figure_mask(img) if fig is None else fig
    flood = flood_mask(img)
    figure_frac = float(fig.mean())
    flood_frac = float(flood.mean())
    out = {
        "figure_frac": figure_frac,
        "flood_frac": flood_frac,
        # The SET DIFFERENCE, not the difference of the two fractions.
        #
        # Those are not the same number, because BiRefNet also keeps pixels
        # the flood fill dropped, and subtracting fractions nets the two off
        # against each other. Measured on 8d9d164d attack se f4: 92.6% flood,
        # 85.3% figure, so the subtraction says 7.3% and passes an 8%
        # threshold, while the pixels that are actually backdrop-and-not-
        # figure are 9.0% and fail it. That cell is one the owner named, and
        # the first version of this module let it through on exactly that
        # arithmetic.
        "backdrop_frac": float((flood & ~fig).mean()),
        "border_frac": border_fraction(fig),
        "lean": head_lean(fig),
    }
    if view is not None:
        out["figure_ratio"] = (figure_frac / view.figure_frac
                               if view.figure_frac else None)
        out["identity"] = chi2(colour_hist(img, fig), view.hist)
        out["view_lean"] = view.lean

    # Semantic, and the only metric here that is not a measurement of the
    # mask. `figure_ratio` cannot see a moderate close-up - it is
    # non-monotonic in it (1.1) - and neither can any other geometry, because
    # the LoRA frames a torso and a whole body to the same 512px box.
    # None unless FRAMING_CHECK is on and the weights are present.
    feet = framing.feet_in_frame(img)
    if feet is not None:
        out["feet_in_frame"] = round(feet, 4)
    return out


def failures(m: dict, neighbours: list | None = None) -> list:
    """Which checks this frame fails. An empty list means it passes."""
    bad = []
    ratio = m.get("figure_ratio")
    if ratio is not None and ratio > FIGURE_RATIO_MAX:
        bad.append("close_up")
    if m.get("backdrop_frac", 0.0) > BACKDROP_MAX:
        bad.append("backdrop")
    if m.get("border_frac", 0.0) > BORDER_MAX:
        bad.append("border")

    lean, ref = m.get("lean"), m.get("view_lean")
    if (lean is not None and ref is not None
            and abs(lean) > LEAN_MIN and abs(ref) > LEAN_MIN
            and (lean > 0) != (ref > 0)):
        bad.append("facing")

    ident = m.get("identity")
    # `ident == ident` is a NaN guard: chi2 returns NaN when either histogram
    # was empty, and NaN > threshold is False, which would pass silently.
    if ident is not None and ident == ident and ident > IDENT_MAX:
        bad.append("identity")

    if neighbours:
        vals = [v for v in neighbours if v is not None and v == v]
        if vals and min(vals) > CONTINUITY_MAX:
            bad.append("continuity")

    # `framing` is the only check that can see a moderate close-up, and it is
    # absent unless FRAMING_CHECK is on and the weights are present. A missing
    # metric must not fail a cell - that is the difference between "judged and
    # passed" and "never judged", the same distinction `facing` draws when its
    # reference is too weak to read.
    if framing.is_close_up(m.get("feet_in_frame")):
        bad.append("framing")
    return bad


def score(m: dict) -> float:
    """How bad a frame is, for picking the best of several attempts.

    Each term is normalised by its own threshold so they are comparable, and
    the sum is used rather than the count of failures: two attempts that each
    fail one check should be separated by HOW badly, not tied.
    """
    s = 0.0
    ratio = m.get("figure_ratio")
    if ratio:
        s += max(0.0, ratio - 1.0) / FIGURE_RATIO_MAX
    s += m.get("backdrop_frac", 0.0) / BACKDROP_MAX
    s += max(0.0, m.get("border_frac", 0.0)) / BORDER_MAX
    lean, ref = m.get("lean"), m.get("view_lean")
    if (lean is not None and ref is not None
            and abs(lean) > LEAN_MIN and abs(ref) > LEAN_MIN
            and (lean > 0) != (ref > 0)):
        s += abs(lean) / LEAN_MIN
    ident = m.get("identity")
    if ident is not None and ident == ident:
        s += ident / IDENT_MAX

    # Framing has to be in the sum, not only in `failures`. A redraw keeps the
    # best-SCORING attempt, so a term that is missing here lets a round
    # replace a correctly framed cell with a close-up that happens to score
    # better on colour - and the failure list would then flag the cell the
    # loop just chose. Scaled so that a confident close-up outweighs a single
    # borderline check: this is the defect the owner actually reports.
    feet = m.get("feet_in_frame")
    if feet is not None and feet == feet:
        s += 2.0 * max(0.0, framing.FRAMING_MIN - feet) / framing.FRAMING_MIN
    return s
