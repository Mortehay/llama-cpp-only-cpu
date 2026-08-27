#!/usr/bin/env python3
"""Audit reference art for ENTITY generation: is this image only the entity?

WHY THIS IS A DIFFERENT QUESTION FROM `judge_trainable`

`measure.judge_trainable` asks "can a style LoRA learn anything from this?" and
is deliberately permissive - blank, tiny, or extreme-strip only. That is the
right gate for a STYLE adapter, where a JPEG reference board with a grey
backdrop still teaches palette and shading.

It is the wrong gate for an ENTITY adapter. The entity conveyor needs an RGBA
cutout: one subject, centred, nothing under it, nothing behind it. Every pixel
that is not the entity is a pixel the adapter learns to draw, and then
`remove_background` cannot remove it - a pedestal fused to the feet is neither
border-connected nor a separate blob, so both the flood fill and
`_isolate_largest_sprite` walk straight past it (see NEGATIVE_SINGLE in
tasks.py, which exists for exactly this reason).

So: a reference can be perfectly trainable and still be poison for this job.
That is what this script measures, and it says which of the two it means.

WHAT IS MEASURED, AND HOW MUCH IT IS TRUSTED

The repo has already shipped two confident alpha-statistic detectors that were
measured and found wrong in BOTH directions (see commit c967bed). So the
findings here are split by confidence and NOT collapsed into one verdict:

  BLOCKING  - certain from the pixels alone. An image with no alpha and a
              background that will not key cannot become a cutout, whatever it
              depicts. These are safe to act on unseen.

  REVIEW    - a real measurement with a known failure mode, reported WITH its
              number so a human can judge. `pedestal` is the repo's own
              strip_ground_patch rule and inherits its documented gap (a stocky
              character with an arm out reads as its own pedestal).

Nothing is deleted. `--apply` writes `trainable = false` for BLOCKING findings
only, and records every finding in `metrics->'entity_audit'` so the UI can
explain an image's absence instead of silently dropping it.

Usage:
    python audit-entity-refs.py --dir /app/images --glob 'ref_core_*.png'
    python audit-entity-refs.py --kind core --json /app/images/audit.json
    python audit-entity-refs.py --kind core --contact-sheet /app/images/audit
    python audit-entity-refs.py --kind core --apply
"""

import argparse
import glob as globlib
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

# ---------------------------------------------------------------------------
# Thresholds. Each one is named after the failure it catches, not a magic value.
# ---------------------------------------------------------------------------

# Below this an RGBA image is "technically transparent, practically a square" -
# the alpha channel exists but nothing was ever cut out of it.
ALPHA_IN_USE_PCT = 0.02

# A background that will not flood-fill away is not a background, it is a
# scene. 2% matches the floor generate_raw_task already refuses a cutout at.
KEYABLE_PCT = 0.02

# Short side below this is upscaled into mush at training resolution.
# Same value as measure.MIN_TRAIN_SIDE, restated so this script stands alone.
MIN_TRAIN_SIDE = 128

# Subject bbox centre may drift this far from the image centre, as a fraction
# of the image size, before the adapter is being taught an off-centre habit.
MAX_CENTRE_DRIFT = 0.15

# strip_ground_patch's own rule: the thing standing on the ground is legs, so
# anything at the bottom wider than this multiple of the shins is not the body.
PEDESTAL_RATIO = 1.8
SHIN_BAND = (0.10, 0.30)     # fraction of subject height above the bottom
PEDESTAL_BAND = 0.25         # only the bottom quarter can be a pedestal

# A subject that covers most of the frame AND touches most of the border is a
# texture or a scene, not an entity with air around it.
FULL_BLEED_COVERAGE = 0.85
FULL_BLEED_BORDER = 0.75

# A blob under this fraction of the frame is a speck, a stray pixel or JPEG
# noise - not a second subject. Counting those calls every image multi-subject.
MIN_SUBJECT_FRAC = 0.01

# A detached mark beside the subject counts as a stray once it clears BOTH
# floors. Well under MIN_SUBJECT_FRAC on purpose - a stray is not a second
# subject, it is litter, and the whole point is that it is small enough for the
# multi-subject rule to miss.
STRAY_MIN_PX = 24
STRAY_MIN_FRAC = 0.0002

# Colour bin width for finding a checkerboard's two tones. Coarse enough to
# gather one tone's JPEG noise into a single bin, fine enough that the two
# tones of a subtle checker (measured at 73 and 82) still land in different
# bins.
QUANT = 8


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def key_background(img, tolerance=22):
    """What `tasks.remove_background` would actually clear, as a boolean mask.

    MIRRORS THAT FUNCTION DELIBERATELY, RULE FOR RULE. The whole point of this
    audit is to predict what the cutout stage does to an image, and the cutout
    stage is `remove_background` - so any difference between the two is this
    script lying about its own subject.

    It is reimplemented rather than imported only because importing `tasks`
    drags in celery, psycopg2, torch and a CUDA availability check.

    THE RULE, and the first version got it wrong: take the colour MOST CORNERS
    AGREE ON - a majority vote - and key pixels within `tolerance` of that one
    colour which are also reachable from the border. The first version matched
    against ANY of the four corners, which is `pixelate.key_background`'s rule
    and not this one.

    That difference is not academic. Measured over the 462 references where
    keying decides the verdict, the two rules disagree on the BLOCKING verdict
    for 25 files, and always in the dangerous direction: on
    `ref_core_022767411230` the any-corner rule reports 68% of the image as
    removable and `remove_background` actually clears 0.06%. Those 25 have
    corners of different colours - a gradient, a border, a subject running into
    one corner - so "close to any corner" matches most of the frame while
    "close to the majority corner" matches nearly none. Every one of them was
    being passed as a recoverable backdrop when the real cutout cannot key it
    at all.
    """
    img = img.convert("RGBA")
    w, h = img.size
    samples = [img.getpixel(c)[:3] for c in
               [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]]
    bg = np.array(max(set(samples), key=samples.count), dtype=np.int16)

    arr = np.array(img, dtype=np.int16)
    # `< tolerance`, not `<=`: remove_background uses a strict comparison.
    match = np.all(np.abs(arr[:, :, :3] - bg) < tolerance, axis=-1)

    labels, n = ndimage.label(match)
    if n == 0:
        return np.zeros((h, w), dtype=bool)
    border = np.concatenate([labels[0, :], labels[-1, :],
                             labels[:, 0], labels[:, -1]])
    mask = np.isin(labels, np.unique(border[border > 0]))

    # remove_background's safety trigger: clearing over 98% means the corner
    # sample was not background at all, so it keeps the image untouched. An
    # audit that reported 99% removable where the real stage removes nothing
    # would be wrong in the worst direction.
    if mask.mean() > 0.98:
        return np.zeros((h, w), dtype=bool)
    return mask


def subject_mask(img):
    """Boolean "this pixel is the entity", plus how that was decided.

    Alpha wins when it is actually in use. Otherwise the background is keyed
    off the corners, which is the only other thing that generalises - a
    modal-colour threshold eats matching pixels inside the subject.
    """
    a = np.asarray(img.convert("RGBA"))
    alpha = a[..., 3]
    clear = float((alpha < 128).mean())
    if clear > ALPHA_IN_USE_PCT:
        return alpha >= 128, "alpha", clear, clear

    keyed = key_background(img)
    return ~keyed, "keyed", clear, float(keyed.mean())


def pedestal_ratio(mask):
    """(bottom-band width / shin width, shin width / body median).

    The first number is strip_ground_patch's rule reduced to what it turns on.
    The second is ITS PRECONDITION, and it is returned because a documented
    caveat does not fire.

    strip_ground_patch assumes a body whose reference width is dominated by the
    torso and whose feet are clearly narrower - a humanoid in a rest pose, which
    is what it was written for. Where that does not hold, the shin reference is
    inflated, the ratio comes out low, and a real pedestal passes in silence.
    Reporting only the first number hides exactly the failure the docstring
    warns about.

    Measured over the 245 single-subject `core` references, the precondition
    mostly DOES NOT HOLD here: shin/body runs p25 0.92, median **1.00**, p90
    1.45. Shins are typically as wide as the body, because this set is mostly
    creatures, props and items rather than standing humanoids. So the pedestal
    count is a floor, its misses are unquantifiable, and the run-level summary
    says so with the number recomputed each time rather than trusting this
    comment to be read.

    A per-image flag was considered and NOT shipped: at any useful cut it fires
    on 151-210 of 245 images, which is a rule that cries wolf rather than a
    signal.
    """
    rows = np.where(mask.any(axis=1))[0]
    if rows.size == 0:
        return 0.0, 0.0
    top, bot = int(rows[0]), int(rows[-1])
    height = bot - top + 1
    if height < 8:
        return 0.0, 0.0

    widths = mask.sum(axis=1).astype(float)
    lo = bot - int(height * SHIN_BAND[1])
    hi = bot - int(height * SHIN_BAND[0])
    shins = widths[max(lo, top):max(hi, top + 1)]
    shins = shins[shins > 0]
    if shins.size == 0:
        return 0.0, 0.0
    reference = float(np.median(shins))
    if reference <= 0:
        return 0.0, 0.0

    body = widths[top:bot + 1]
    body = body[body > 0]
    body_med = float(np.median(body)) if body.size else 0.0
    precondition = (reference / body_med) if body_med > 0 else 0.0

    band = widths[max(bot - int(height * PEDESTAL_BAND), top):bot + 1]
    if band.size == 0:
        return 0.0, precondition
    return float(band.max() / reference), precondition


def subject_count(mask):
    """Separable subjects above a size floor.

    Closure first, so a sword whose pommel is a separate blob stays one subject
    rather than becoming two.

    Returns 0 for BOTH "empty frame" and "every blob is under the size floor".
    Callers must not conflate them: the second case is a dense icon atlas, which
    is very much not empty. `measure` keeps `coverage` alongside so the two can
    be told apart, because the first version of this script reported two
    64-icon atlases as "nothing visible" - checked by eye, and wrong.
    """
    h, w = mask.shape
    r = max(1, int(round(min(h, w) / 256)))
    closed = ndimage.binary_closing(mask, structure=np.ones((2 * r + 1,) * 2))
    labels, n = ndimage.label(closed, structure=np.ones((3, 3)))
    if n == 0:
        return 0
    sizes = ndimage.sum(closed, labels, range(1, n + 1))
    return int((sizes > MIN_SUBJECT_FRAC * h * w).sum())


def checkerboard_score(img):
    """Is the transparency checkerboard PAINTED INTO the pixels? 0.0 or 1.0.

    The nastiest defect in this set, because it survives review: the image looks
    like a clean cutout in any viewer that draws a checkerboard behind
    transparency, and it is a fully opaque square of squares. An adapter trained
    on it learns to draw the checker as part of the subject - a backdrop no
    amount of background keying will remove, because it is two colours and a
    flood fill only ever takes one.

    HOW IT WORKS: find the two dominant flat tones of the WHOLE IMAGE, read the
    checker's pitch off the longest unbroken run of those tones, then
    reconstruct the ideal checker ((x//p + y//p) % 2) and require it to
    reproduce the backdrop. Four conditions, and a photograph, a gradient, a
    flat backdrop and dithered artwork each fail one:

      two tones  - two quantised colours dominate, in near-equal measure
      backdrop   - together they cover at least 15% of the frame
      square     - horizontal and vertical pitch agree to within a pixel
      predicted  - the phase-fitted ideal checker reproduces >88% of them

    That last one is what makes it safe to treat as blocking. Two earlier
    alpha-statistic detectors in this repo were confidently wrong (commit
    c967bed) because they described a symptom; this reconstructs the pattern and
    checks that it matches.

    THREE FRAMINGS THAT FAILED, each worth keeping so nobody rebuilds them:

      a corner block - the checker is often only BEHIND the subject, with a flat
      margin around it (b88ccc34e525, bf85cd390051 both have a uniform corner),
      so a corner test scored them 0.

      any fixed block - a block large enough to hold the ~36px pitch these
      references use also contains part of the coloured subject, which fails a
      whole-block flatness test. Working on the dominant tones instead lets the
      subject drop out of the measurement rather than poison it.

      GREY tones only - the assumption that a transparency checker is grey.
      Caught by diffing against scripts/audit-character-refs.py, which finds
      these by a different route: ref_core_cff50aa39462 checks in PINK and
      ref_core_fe898e015a5d in NAVY. A saturation gate scored both 0. Editors
      let people recolour the checker, and people do.

    Still misses cases the border test in that sibling script catches, which is
    why `measure` runs both and takes the union - see `_sibling_checkerboard`.
    """
    rgba = np.asarray(img.convert("RGBA"))
    a = rgba[..., :3].astype(np.int16)
    h, w = a.shape[:2]
    if min(h, w) < 64:
        return 0.0

    # TRANSPARENT PIXELS ARE NOT MEASURED, and this is load-bearing rather than
    # tidiness. Keying a checkerboard away sets alpha to 0 but LEAVES THE OLD
    # RGB IN PLACE, and `convert("RGB")` discards the alpha and hands that dead
    # backdrop straight back. So the first version of this reported
    # "checkerboard baked in" on 14 of the 103 cells in images/recovered/cells -
    # which are correctly keyed, 66% transparent, and precisely the images
    # somebody had just fixed. Condemning the repaired copies is the worst
    # direction for this error to point.
    opaque = rgba[..., 3] >= 128
    n_opaque = int(opaque.sum())
    if n_opaque < 0.15 * h * w:
        return 0.0

    # Colour counts on a COARSE grid, packed so np.unique runs on one integer
    # array. Quantised, not exact: these are JPEG-sourced references with paper
    # texture over the checker, so an exact count fragments one tone across
    # dozens of near values and no single colour is dominant. Counting exact
    # colours lost two checkerboards that the coarse count finds.
    q = (a // QUANT) * QUANT
    packed = (q[..., 0].astype(np.int32) << 16 | q[..., 1].astype(np.int32) << 8
              | q[..., 2].astype(np.int32))
    vals, counts = np.unique(packed[opaque], return_counts=True)
    if vals.size < 2:
        return 0.0
    order = np.argsort(counts)[::-1][:12]

    def rgb(v):
        return np.array([(int(v) >> 16) & 255, (int(v) >> 8) & 255,
                         int(v) & 255], dtype=np.int16)

    c1 = rgb(vals[order[0]])
    # The SECOND tone must be a separate colour, not the first one's own
    # dither. Taking the plain top two picked 130 and 131 off a flat grey
    # backdrop (ref_core_b88ccc34e525), which is one tone, not two.
    second = None
    for j in order[1:]:
        c = rgb(vals[j])
        d = float(np.abs(c - c1).max())
        if 8 <= d <= 200:
            second = (c, int(counts[j]))
            break
    if second is None:
        return 0.0
    c2, n2 = second
    n1 = int(counts[order[0]])

    # Both tones common, and together a real part of the frame. Below this it is
    # a two-tone prop, not a backdrop.
    if min(n1, n2) < 0.3 * (n1 + n2):
        return 0.0
    # Measured against the OPAQUE area, not the frame: on a keyed image most of
    # the frame is nothing at all, and a fraction of it is not a fraction of
    # anything the viewer sees.
    if (n1 + n2) < 0.15 * n_opaque:
        return 0.0

    # Distance to each tone. `near` is deliberately loose: this gate only has to
    # establish that these two tones are the backdrop story of the image.
    # Whether the pattern is a CHECKERBOARD is decided by the model fit below,
    # which is the strict test.
    d1 = np.abs(a - c1).max(-1)
    d2 = np.abs(a - c2).max(-1)
    # ANDed with `opaque` throughout, so a keyed-away checkerboard still sitting
    # in the RGB channels cannot contribute to the pitch, the phase or the fit.
    near = ((d1 <= 12) | (d2 <= 12)) & opaque
    if near.sum() < 0.15 * n_opaque:
        return 0.0
    grey = near                       # "a visible pixel of one backdrop tone"

    # Periodic, confirmed by reconstructing the pattern. Pitch is read off the
    # longest unbroken backdrop run in the most-backdrop row and column - never
    # a mean across an axis, because a checkerboard column is half one tone and
    # half the other, and averaging flattens it to 0.5, leaving only noise.
    dark = d1 <= d2                   # this pixel belongs to tone 1
    px, x0 = _pitch_in_run(dark[int(np.argmax(grey.sum(1)))],
                           grey[int(np.argmax(grey.sum(1)))])
    py, y0 = _pitch_in_run(dark[:, int(np.argmax(grey.sum(0)))],
                           grey[:, int(np.argmax(grey.sum(0)))])
    if px is None or py is None or abs(px - py) > 1:
        return 0.0

    g = grey & near
    # Enough of the VISIBLE image to be a backdrop, not a two-tone prop that
    # happens to fit.
    if g.sum() < 0.15 * n_opaque:
        return 0.0

    # Refine the phase before scoring. The first flip of one scanline lands
    # within a few pixels of the true origin, and on an 80px pitch "a few
    # pixels" is enough to cost several points of agreement - measured on
    # ref_core_039fd1bc0d72, where the estimated phase scored 0.87 and the best
    # phase scored 0.88 against a 0.90 bar. Swept on every 4th pixel so the
    # sweep costs a sixteenth of the full grid; the winner is then scored in
    # full.
    yy, xx = np.mgrid[0:h, 0:w]
    # ~20 candidate offsets per axis. Coarser (px//6) walks straight past the
    # best phase: on the 80px pitch of ref_core_039fd1bc0d72 the optimum sits at
    # (40, 44) and a 13px sweep never lands near enough to clear the bar.
    step = max(1, px // 20)
    sub = (slice(None, None, 4), slice(None, None, 4))
    gs, ds = g[sub], dark[sub]
    best, best_phase = -1.0, (x0, y0)
    for dx in range(0, px, step):
        for dy in range(0, px, step):
            m = (((xx[sub] - dx) // px) + ((yy[sub] - dy) // px)) % 2 == 0
            score = max((m[gs] == ds[gs]).mean(), (m[gs] != ds[gs]).mean())
            if score > best:
                best, best_phase = score, (dx, dy)

    dx, dy = best_phase
    model = (((xx - dx) // px) + ((yy - dy) // px)) % 2 == 0
    # Scored over grey pixels only, and in either polarity - which of the two
    # greys is "dark" is arbitrary.
    agree = max((model[g] == dark[g]).mean(), (model[g] != dark[g]).mean())

    # 0.88, not 0.90. These are lossy JPEG-sourced references with paper
    # texture over the checker, so a perfect fit is not available: the best
    # phase on a confirmed checkerboard measures 0.885. Still a hard bar - a
    # flat backdrop, a gradient and dithered artwork all score far below it.
    return 1.0 if agree > 0.88 else 0.0


# The loaded `baked_checkerboard`, or None. `_SIBLING_TRIED` is separate so
# "not loaded yet" and "tried and failed" stay distinguishable - collapsing them
# into one sentinel is what made the failure silent in the first place.
_SIBLING = None
_SIBLING_TRIED = False


def sibling_available():
    """Did the sibling checkerboard detector load? Forces the attempt first.

    Exists so a run can SAY it degraded. `_sibling_checkerboard` returns False
    when the sibling is missing, which is indistinguishable from "checked, and
    it is not a checkerboard" - a silent loss of the 3 core + 12 sprite files
    only the border test catches.
    """
    _sibling_checkerboard(Image.new("RGBA", (1, 1)))
    return _SIBLING is not None


def _sibling_checkerboard(img):
    """`baked_checkerboard` from audit-character-refs.py, if that script is here.

    THE TWO DETECTORS ARE COMPLEMENTARY, WHICH IS WHY BOTH RUN

    That script finds a checker by asking whether the BORDER alternates between
    two flat tones in equal runs along both axes. This one fits an ideal
    checker to the two dominant tones of the whole image. Measured across all
    483 references, each catches cases the other cannot:

      only the border test - 3 core and 12 sprite references, mostly RPG-Maker
      character sheets whose checker is nearly white, and one whose two tones
      are too close to survive quantisation into separate bins.

      only the model fit - ref_core_a9a6b26fb952, a clipart-site PNG whose
      subject touches the border, leaving no clean border strip to read.

    Every hit from both was confirmed by eye, so the union is taken rather than
    a winner picked. Loaded by path, the way curate-training-set.py loads
    split-sheets.py, because a hyphenated filename is not importable. Absent or
    broken, this degrades to the local detector alone rather than failing.
    """
    global _SIBLING, _SIBLING_TRIED
    if not _SIBLING_TRIED:
        _SIBLING_TRIED = True
        try:
            import importlib.util
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "audit-character-refs.py")
            spec = (importlib.util.spec_from_file_location("acr", path)
                    if os.path.exists(path) else None)
            # Both halves really can be None - spec_from_file_location returns
            # None for a path it cannot make a loader for, and a spec's loader
            # is optional. The broad `except` below would swallow the
            # AttributeError either way, but silently treating "the sibling
            # script is malformed" as "the sibling script is absent" is exactly
            # the kind of quiet degradation this audit exists to avoid.
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                _SIBLING = getattr(mod, "baked_checkerboard", None)
        except Exception:
            _SIBLING = None
    if _SIBLING is None:
        return False
    try:
        return bool(_SIBLING(img.convert("RGBA")))
    except Exception:
        return False


def _pitch_in_run(line, grey_line):
    """Checker pitch and phase, measured inside the longest grey run of a line.

    Restricted to the run because the subject interrupts the pattern, and a
    flip caused by the subject's edge is not a checker flip.
    """
    idx = np.flatnonzero(grey_line)
    if idx.size < 16:
        return None, None
    # Longest contiguous grey stretch.
    splits = np.flatnonzero(np.diff(idx) != 1)
    starts = np.r_[idx[0], idx[splits + 1]]
    ends = np.r_[idx[splits], idx[-1]]
    k = int(np.argmax(ends - starts))
    s, e = int(starts[k]), int(ends[k]) + 1
    seg = line[s:e]
    if seg.size < 16:
        return None, None

    flips = np.flatnonzero(np.diff(seg.astype(np.int8)) != 0)
    if flips.size < 2:
        return None, None
    runs = np.diff(flips)
    p = round(float(np.median(runs)))
    if p < 2 or p > seg.size // 3:
        return None, None
    return p, s + int(flips[0]) + 1


def stray_blobs(mask):
    """(count, largest as a fraction of the frame) of detached marks beside the
    subject. The gap between the other two detectors, and it is a real one.

    A DETACHED DROP SHADOW falls through both of them. `pedestal_ratio` only
    sees something FUSED to the base - it measures width, so a shadow floating
    clear of the trunk is invisible to it. `subject_count` only counts blobs
    over MIN_SUBJECT_FRAC of the frame, which is the right floor for "is this a
    multi-subject sheet" and far too coarse here: the shadow under
    ref_sprite_790879270b49 is 0.22% of the frame, so that image measured as one
    clean subject when it has a grey ellipse sitting next to the trunk.

    Found by cross-checking with scripts/audit-character-refs.py, whose
    subject-counting fix flagged the same file for a different reason.

    Sized against a FLOOR IN BOTH UNITS. An absolute pixel floor alone would
    scale wrongly across a set that runs from 128px to 1200px; a fraction alone
    would let a 24px speck through on a large image and flag antialiasing on a
    small one.

    REVIEW, never blocking: a detached blob can be legitimate art - floating
    leaves, sparks off a torch, a gleam beside a blade. What it must not be is
    invisible.
    """
    h, w = mask.shape
    r = max(1, int(round(min(h, w) / 256)))
    closed = ndimage.binary_closing(mask, structure=np.ones((2 * r + 1,) * 2))
    labels, n = ndimage.label(closed, structure=np.ones((3, 3)))
    if n <= 1:
        return 0, 0.0
    sizes = ndimage.sum(closed, labels, range(1, n + 1))
    main = float(sizes.max())
    floor = max(STRAY_MIN_PX, STRAY_MIN_FRAC * h * w)
    strays = sizes[(sizes < main) & (sizes >= floor)]
    if strays.size == 0:
        return 0, 0.0
    return int(strays.size), float(strays.max() / (h * w))


def measure(path):
    img = Image.open(path)
    w, h = img.size
    mask, how, alpha_clear, bg_clear = subject_mask(img)

    m = {"file": os.path.basename(path), "w": w, "h": h,
         "background": how,
         "alpha_transparent_pct": round(alpha_clear, 4),
         "removable_bg_pct": round(bg_clear, 4),
         "checkerboard": bool(checkerboard_score(img)
                              or _sibling_checkerboard(img))}

    if not mask.any():
        m.update(coverage=0.0, border_pct=0.0, subjects=0,
                 pedestal_ratio=0.0, centre_drift=0.0, bbox_frac=0.0,
                 edges_touched=0, strays=0, stray_frac=0.0, shin_body=0.0)
        return m

    ring = np.zeros_like(mask)
    ring[0, :] = ring[-1, :] = True
    ring[:, 0] = ring[:, -1] = True

    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    n_strays, stray_frac = stray_blobs(mask)
    ped, shin_body = pedestal_ratio(mask)

    m.update(
        coverage=round(float(mask.mean()), 4),
        border_pct=round(float(mask[ring].mean()), 4),
        subjects=subject_count(mask),
        pedestal_ratio=round(ped, 2),
        shin_body=round(shin_body, 2),
        strays=n_strays,
        stray_frac=round(stray_frac, 5),
        # Drift measured per axis against that axis's own length, so a tall
        # thin image is not called off-centre for an offset that is small in
        # absolute pixels.
        centre_drift=round(float(max(abs(cy - h / 2) / h,
                                     abs(cx - w / 2) / w)), 3),
        bbox_frac=round(float((y1 - y0 + 1) * (x1 - x0 + 1)) / (w * h), 4),
        edges_touched=sum([bool(mask[0].any()), bool(mask[-1].any()),
                           bool(mask[:, 0].any()), bool(mask[:, -1].any())]),
    )
    return m


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def judge(m):
    """(blocking, review) - lists of reasons this image cannot teach an entity."""
    blocking, review = [], []

    if m["coverage"] == 0.0:
        blocking.append("nothing visible - the whole frame keys away as "
                        "background, so there is no entity in here to learn")
        return blocking, review

    # subjects == 0 with coverage > 0 does NOT mean empty. It means every blob
    # is under the size floor, which is what a densely packed icon atlas looks
    # like. The first version of this script called two 64-icon atlases
    # "nothing visible"; they were checked by eye and they are the opposite.
    if m["subjects"] == 0:
        blocking.append(
            "dense atlas - the frame is {:.0%} covered but no single subject "
            "reaches {:.0%} of it, which is a packed sheet of many small icons. "
            "There is no entity here to isolate; split it into cells first "
            "(scripts/split-sheets.py) and audit the cells".format(
                m["coverage"], MIN_SUBJECT_FRAC))

    # Worse than an opaque background, because it survives review: the image
    # reads as a clean cutout in any viewer that draws a checkerboard behind
    # alpha, and it is a solid square of grey squares.
    if m["checkerboard"]:
        blocking.append(
            "checkerboard baked in - the transparency grid is PAINTED into the "
            "pixels, so this looks like a cutout and is a fully opaque square. "
            "Keying cannot remove it (a flood fill takes one colour and this is "
            "two), and an adapter trained on it learns to draw the grey checker "
            "pattern around every entity it generates")

    if min(m["w"], m["h"]) < MIN_TRAIN_SIDE:
        blocking.append(
            "too small - {}x{}, short side under {}px, so training upscales it "
            "and learns invented detail rather than the entity".format(
                m["w"], m["h"], MIN_TRAIN_SIDE))

    # The defining defect for this job. No alpha AND nothing keys away means the
    # entity is welded to its backdrop: there is no cutout in here to be had,
    # and an adapter trained on it draws the backdrop back in every time.
    if m["removable_bg_pct"] < KEYABLE_PCT:
        blocking.append(
            "opaque background - only {:.1%} of this image is transparent or "
            "keyable, so the entity cannot be separated from what is behind "
            "it. Training on it teaches the model to fill the frame, which is "
            "the exact opposite of a cutout".format(m["removable_bg_pct"]))

    if m["subjects"] >= 2:
        blocking.append(
            "multi-subject - {} separate subjects, so this is a sheet or a "
            "scene rather than one entity. It teaches the model to lay several "
            "objects out, and no negative prompt reliably undoes that (see "
            "CORE_TRIGGERS in tasks.py: training beats guidance)".format(
                m["subjects"]))

    # --- review: real, measured, with a known failure mode ------------------

    # ONLY on a single subject. "Wider at the bottom than the shins" is a
    # statement about a body standing on something, and it is meaningless on a
    # grid - the bottom ROW of an icon sheet is wider than any imagined shin, so
    # this fired on every asset board in the set until it was checked by eye.
    # A multi-subject image is already blocked above for a better reason.
    if m["subjects"] == 1 and m["pedestal_ratio"] >= PEDESTAL_RATIO:
        review.append(
            "pedestal or ground patch - the bottom band is {:.1f}x wider than "
            "the shins. Whatever it is (base, plinth, dirt, contact shadow) it "
            "is FUSED to the feet, so it survives both remove_background and "
            "_isolate_largest_sprite, and every sprite derived from it "
            "inherits it. Known gap: a stocky subject with a limb held out can "
            "score this way with no pedestal - confirm by eye".format(
                m["pedestal_ratio"]))

    # Only on a single subject. On a sheet the "strays" are just the smaller
    # items, which the multi-subject verdict above already covers better.
    if m["subjects"] == 1 and m["strays"]:
        review.append(
            "stray marks beside the subject - {} detached blob(s) clear of the "
            "entity, the largest {:.2%} of the frame. A drop shadow floating "
            "free of the feet lands here, and it is missed by BOTH other "
            "rules: pedestal_ratio measures width so it cannot see a shadow "
            "that does not touch, and the multi-subject rule ignores anything "
            "under {:.0%} of the frame. `_isolate_largest_sprite` deletes these "
            "at generation time, so this matters for TRAINING, where the "
            "adapter simply learns to draw the litter. Can be legitimate "
            "art - floating leaves, sparks - so look before cropping".format(
                m["strays"], m["stray_frac"], MIN_SUBJECT_FRAC))

    if (m["coverage"] > FULL_BLEED_COVERAGE
            and m["border_pct"] > FULL_BLEED_BORDER):
        review.append(
            "full bleed - {:.0%} of the frame is subject and {:.0%} of the "
            "border is too. That is a texture or a scene, not an entity with "
            "air around it; an entity adapter learns from it to run to the "
            "edges, and a subject touching every edge is also the case "
            "remove_background provably cannot cut".format(
                m["coverage"], m["border_pct"]))

    if m["centre_drift"] > MAX_CENTRE_DRIFT and m["edges_touched"] < 3:
        review.append(
            "off centre - the subject sits {:.0%} of the frame away from the "
            "middle. The conveyor composites entity assets by their image box, "
            "so a learned offset becomes a placement error in every map that "
            "uses the asset".format(m["centre_drift"]))

    # RGBA that never got cut out. Not blocking on its own - it keys fine - but
    # it matters which references arrived pre-cut and which merely could be.
    if (m["background"] == "keyed"
            and m["alpha_transparent_pct"] < ALPHA_IN_USE_PCT
            and m["removable_bg_pct"] >= KEYABLE_PCT):
        review.append(
            "background is flat, not transparent - {:.0%} of it keys away "
            "cleanly, so this image is recoverable, but as stored it still "
            "teaches a backdrop. Cut it out before training, do not train on "
            "it as it is".format(m["removable_bg_pct"]))

    return blocking, review


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def contact_sheet(paths, out, cols=8, cell=128):
    """A page of thumbnails on magenta, so transparency is visible at a glance.

    Magenta rather than the usual checkerboard: the question these sheets exist
    to answer is "what is opaque that should not be", and a flat alarm colour
    makes a leftover backdrop or a pedestal read instantly, where a
    checkerboard reads as texture and hides small opaque patches inside it.
    """
    from PIL import ImageDraw
    rows = (len(paths) + cols - 1) // cols
    pad = 14
    sheet = Image.new("RGB", (cols * cell, max(rows, 1) * (cell + pad)),
                      (255, 0, 255))
    d = ImageDraw.Draw(sheet)
    for i, p in enumerate(paths):
        try:
            im = Image.open(p).convert("RGBA")
        except Exception:
            continue
        im.thumbnail((cell, cell), Image.Resampling.LANCZOS)
        x = (i % cols) * cell + (cell - im.width) // 2
        y = (i // cols) * (cell + pad)
        sheet.paste(im, (x, y), im)
        d.text(((i % cols) * cell + 2, y + cell + 1),
               os.path.basename(p)[-16:], fill=(0, 0, 0))
    sheet.save(out)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="/app/images")
    p.add_argument("--glob", default=None,
                   help="filename pattern; defaults from --kind")
    p.add_argument("--kind", default="core",
                   help="reference kind: core | sprite | tile | map")
    p.add_argument("--json", default=None, help="write full findings here")
    p.add_argument("--markdown", default=None,
                   help="write the per-file marks as readable Markdown. The "
                        "durable form of --apply when there is no database to "
                        "write to")
    p.add_argument("--contact-sheet", default=None,
                   help="prefix for contact sheet PNGs, one per finding bucket")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--apply", action="store_true",
                   help="write BLOCKING findings to reference_assets.trainable")
    a = p.parse_args()

    pattern = a.glob or "ref_{}_*.png".format(a.kind)
    paths = sorted(q for q in globlib.glob(os.path.join(a.dir, pattern))
                   if not os.path.basename(q).startswith("thumb_"))
    if a.limit:
        paths = paths[:a.limit]
    if not paths:
        sys.exit("no images matched {} in {}".format(pattern, a.dir))

    findings, buckets = [], {}
    for i, path in enumerate(paths):
        try:
            m = measure(path)
        except Exception as e:
            findings.append({"file": os.path.basename(path),
                             "blocking": ["unreadable: {}".format(e)],
                             "review": []})
            buckets.setdefault("unreadable", []).append(path)
            continue
        blocking, review = judge(m)
        m["blocking"], m["review"] = blocking, review
        findings.append(m)

        for r in blocking + review:
            buckets.setdefault(r.split(" - ")[0], []).append(path)
        if not blocking and not review:
            buckets.setdefault("clean", []).append(path)
        if (i + 1) % 250 == 0:
            print("  ... {}/{}".format(i + 1, len(paths)), file=sys.stderr)

    nblock = sum(1 for f in findings if f.get("blocking"))
    nreview = sum(1 for f in findings if f.get("review") and not f.get("blocking"))
    print("kind={}  images: {}".format(a.kind, len(findings)))
    print("  fit for entity training:    {}".format(len(findings) - nblock - nreview))
    print("  BLOCKING (certain):         {}".format(nblock))
    print("  REVIEW (measured, judge):   {}".format(nreview))
    print()
    for name, group in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        print("  {:5d}  {}".format(len(group), name))

    # THE PEDESTAL RULE'S PRECONDITION, recomputed every run.
    #
    # strip_ground_patch assumes shins clearly narrower than the body. Where
    # that fails the shin reference inflates and a real pedestal scores under
    # threshold in silence - so the pedestal count is a floor, and how much of a
    # floor depends on THIS set rather than on the comment in pedestal_ratio().
    # Printed rather than documented because a caveat in a docstring does not
    # fire when the population changes underneath it.
    singles = [f for f in findings if f.get("subjects") == 1 and f.get("shin_body")]
    if singles:
        holds = sum(1 for f in singles if f["shin_body"] < 0.8)
        print("\n  pedestal rule precondition (shins < 0.8x body) holds for "
              "{} of {} single-subject images".format(holds, len(singles)))
        if holds < 0.5 * len(singles):
            print("  -> it does NOT hold for most of this set, so the pedestal "
                  "count above is a floor and its misses are unquantifiable")

    # Degrading to one checkerboard detector is a real loss of coverage, and it
    # was previously silent by design ("graceful degradation"). Graceful and
    # invisible are different things.
    if not sibling_available():
        print("\n  NOTE: scripts/audit-character-refs.py was not loadable, so "
              "checkerboard detection ran on this script's rule alone. That "
              "loses the cases only the border test finds "
              "(3 core + 12 sprite when last measured).")

    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"kind": a.kind, "dir": a.dir, "findings": findings},
                      fh, indent=1)
        print("\nwrote {}".format(a.json))

    if a.markdown:
        with open(a.markdown, "w", encoding="utf-8") as fh:
            fh.write("# Entity-fitness marks: `{}` references\n\n".format(a.kind))
            fh.write("Generated by `scripts/audit-entity-refs.py`. One entry "
                     "per image that cannot teach an entity cutout, with the "
                     "measurement that says so.\n\n")
            fh.write("- **{}** images audited\n- **{}** blocking\n"
                     "- **{}** for review\n- **{}** fit as they stand\n\n".format(
                         len(findings), nblock, nreview,
                         len(findings) - nblock - nreview))
            for tier, key in (("Blocking", "blocking"), ("Review", "review")):
                rows = [f for f in findings
                        if f.get(key) and (key == "blocking"
                                           or not f.get("blocking"))]
                fh.write("\n## {} ({})\n\n".format(tier, len(rows)))
                for f in sorted(rows, key=lambda r: r["file"]):
                    fh.write("### `{}`\n\n".format(f["file"]))
                    if "w" in f:
                        fh.write("`{}x{}`, background read via {}, "
                                 "{:.0%} removable, {} subject(s), "
                                 "pedestal {:.1f}x\n\n".format(
                                     f["w"], f["h"], f["background"],
                                     f["removable_bg_pct"], f["subjects"],
                                     f["pedestal_ratio"]))
                    for r in f[key]:
                        fh.write("- {}\n".format(r))
                    fh.write("\n")
        print("\nwrote {}".format(a.markdown))

    if a.contact_sheet:
        for name, group in buckets.items():
            slug = name.replace(" ", "-").replace("/", "-")[:40]
            pages = (min(len(group), 128) + 63) // 64
            for pg in range(pages):
                out = "{}_{}_{}.png".format(a.contact_sheet, slug, pg)
                contact_sheet(group[pg * 64:(pg + 1) * 64], out)
                print("wrote {}  ({} in bucket)".format(out, len(group)))

    if a.apply:
        import psycopg2
        db = os.environ.get("DB_URL")
        if not db:
            sys.exit("DB_URL unset - run inside the container")
        conn = psycopg2.connect(db)
        cur = conn.cursor()
        marked = touched = missed = ambiguous = 0

        # MATCH ON THE BASENAME, NOT THE PATH THIS RUN HAPPENED TO USE.
        #
        # The database stores absolute container paths - `/app/images/ref_core_
        # <hex>.png` - because that is where the API wrote them. This script is
        # normally pointed at a relative `--dir images` from the host, so
        # `os.path.join(a.dir, file)` builds `images/ref_core_<hex>.png`, which
        # equals nothing.
        #
        # That failure was SILENT and reported as success: zero rows updated,
        # "applied to 0 rows", exit 0. Worse, a scratch-database test cannot
        # catch it, because the test seeds whatever paths the test invocation
        # produces - so both sides agree and the bug survives. It was found by
        # reading the real `file_path` column instead of the test's.
        #
        # A suffix match on `/<basename>` works from either side. Basenames are
        # `ref_<kind>_<12 hex>.png` and unique by construction, but that is a
        # property of the current naming rather than a guarantee, so a finding
        # that hits more than one row is counted and reported rather than
        # trusted.
        #
        # `right(file_path, n) = tail` AND NOT `LIKE '%' || tail`, which is what
        # this first shipped as. In SQL `LIKE`, an underscore is a
        # single-character wildcard - and every filename here is
        # `ref_core_<hex>.png`, so the pattern also matched `refXcoreY<hex>.png`
        # and anything else of that shape. Against this data it was harmless by
        # luck rather than by design, and a wrong-row update would have been
        # silent. `right()` is a byte comparison with no pattern semantics.
        for f in findings:
            path = os.path.join(a.dir, f["file"])
            tail = "/" + f["file"]
            audit = {k: v for k, v in f.items() if k != "file"}
            where = ("WHERE (file_path = %s OR right(file_path, %s) = %s) "
                     "AND deleted = false")
            if f.get("blocking"):
                cur.execute(
                    "UPDATE reference_assets SET trainable = false, "
                    "trainable_why = %s, "
                    "metrics = metrics || jsonb_build_object('entity_audit', %s::jsonb) "
                    + where,
                    ("; ".join(f["blocking"]), json.dumps(audit), path, len(tail), tail))
                marked += cur.rowcount
            else:
                # NOT an exclusion. REVIEW findings are recorded so the UI can
                # show them; they never silently drop an image, because both
                # detectors behind them have a failure mode a human can see and
                # the measurement cannot.
                cur.execute(
                    "UPDATE reference_assets SET "
                    "metrics = metrics || jsonb_build_object('entity_audit', %s::jsonb) "
                    + where,
                    (json.dumps(audit), path, len(tail), tail))
            if cur.rowcount == 0:
                missed += 1
            elif cur.rowcount > 1:
                ambiguous += 1
            touched += cur.rowcount
        conn.commit()
        conn.close()
        print("\napplied to {} rows ({} marked not trainable)".format(
            touched, marked))
        # Say what did NOT land. An audit that quietly updates nothing looks
        # exactly like an audit that had nothing to change.
        if missed:
            print("  {} of {} findings matched no row - the image is on disk "
                  "but not registered in reference_assets (or is deleted)"
                  .format(missed, len(findings)))
        if ambiguous:
            print("  {} findings matched MORE THAN ONE row - two references "
                  "share a basename; check before trusting these"
                  .format(ambiguous))

    return 0


if __name__ == "__main__":
    sys.exit(main())
