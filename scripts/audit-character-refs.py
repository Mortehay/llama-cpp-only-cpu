#!/usr/bin/env python3
"""Judge every character reference against what a training run can learn from.

WHY A SECOND CURATOR

`curate-training-set.py` did this for `tile`, and doing it there is what
explained why the terrain adapter drew lattices: the cells were 63px mush and
every sharp image in the set was a grid. The character side - `sprite` and
`core`, the two kinds `POST /api/training` trains on by default - never had the
same pass. `judge_trainable` in measure.py is the only gate it goes through,
and that gate rejects exactly three things: a fully transparent image, a short
side under 160px, and an aspect beyond 3:1. It is documented as deliberately
permissive, and against this reference set it lets nearly everything through.

WHAT IT LETS THROUGH, measured on the 483 references in /app/images:

  * 125 of 149 sprite references hold two or more separable subjects, 116 of
    them five or more. They are asset packs - RPG Maker walk-cycle sheets,
    weapon-icon boards, explosion strips. This is the SAME failure the tile
    adapter had, in the kind nobody re-measured.
  * 12 sprite references and 7 core references have a transparency
    CHECKERBOARD baked into their pixels. A model trained on those learns to
    draw the checker, and no prompt removes it.
  * 141 of 149 sprite references have no detectable pixel grid, with a median
    of 62k distinct colours. The `sprite` tab means "finished sprite art" and
    is measured for palette, cell grid and outline width - these are JPEG
    reference boards, 3D renders and painted concept sheets.

NOT THE SAME QUESTION AS `audit-entity-refs.py`

That script asks "is this image ONLY the entity?" - the cutout question, where
a pedestal fused to the feet is poison because `remove_background` cannot lift
it. This one asks "can a style adapter learn from this at all?", where a
subject on a grey backdrop is fine.

An image can pass here and fail there, and the reverse - both directions have
been measured rather than assumed:

  * passes here, fails there - any JPEG concept board on flat grey. Perfectly
    good style material; useless as a cutout.
  * fails here for a reason that would not trouble a cutout -
    `ref_sprite_06074972c0ad`: one subject, 78% transparent, a flat backdrop,
    and rejected here on the pixel-art test alone (24,268 colours, no grid).
    Eleven more sprite references fail on that test and nothing else. They are
    cleanly cut out and simply are not pixel art, which matters for THIS
    question and not for that one.

The two scripts therefore report different totals for the same folder, and
neither number is wrong.

Run both; they gate different stages.

Where they DO overlap, they agree from independent code: both find 12 sprite
checkerboards and 2 core references usable as-is. That agreement, between two
detectors sharing no implementation, is the strongest evidence either has that
those counts are real.

IT READS THE FILESYSTEM, AND THE FILESYSTEM IS NOT THE DATASET

Deliberate - it runs with no database, which is how the whole first pass was
done. But the counts it REPORTS are over files, and training reads a manifest
built from `reference_assets`. Those differ: 43 of the 149 `ref_sprite_*.png`
files on disk belong to references that have been SOFT-DELETED, so every sprite
figure here is inflated by images nobody will ever train on. Core happens to
agree exactly, 334 files to 334 live rows, which is what made the discrepancy
easy to miss.

`--apply` is not affected - it scopes its write to `deleted = false`. Only the
reporting over-counts. If a rowcount from `--apply` looks lower than the
findings, that is why, and it is not a failure.

WHAT THIS SCRIPT WILL NOT TELL YOU

Whether the subject is a CHARACTER, or even a whole object. `keep` means "can
teach", never "is what you wanted". Both sprite references that pass every test
here turned out to be props: `ref_sprite_bd7e43ed3cff` is a tree on a stone
plinth, and `ref_sprite_6f8abf25aa2b` is a hollow architectural fragment - one
connected subject on clean alpha, and a piece of a larger tileset rather than a
thing. `audit-entity-refs.py` misses both for the same reason.

Neither tool should try to fix this. "Is this a piece of a thing?" has no
non-circular test - a hollow centre is correct for an archway or a barrel hoop -
and "is this a character?" is semantics, not measurement. Known blind spot,
recorded rather than papered over with a rule that would misfire.

The camera. Whether a character is drawn in the game's isometric projection is
not readable from a single character image - there is no ground plane in it to
measure against, which is precisely why `measure_tile` reads the projection off
a TILE instead. Every rule here is about whether an image can teach at all.
Which VIEW it teaches is a human call, and the report says so rather than
implying a verdict it did not make.

Usage:
    python audit-character-refs.py --data ./images
    python audit-character-refs.py --kind sprite --out audit.md
    python audit-character-refs.py --apply          # needs DB_URL
"""

import argparse
import glob
import importlib.util
import json
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "ss", os.path.join(_here, "split-sheets.py"))
ss = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ss)

# Mirrors measure.MIN_TRAIN_SIDE / MAX_TRAIN_ASPECT rather than importing them:
# this script runs from /app/scripts, mounted read-only and separate from the
# /app package, so measure.py is not importable from here.
MIN_TRAIN_SIDE = 160
MAX_TRAIN_ASPECT = 3.0

# Two images this close in perceptual hash are the same picture. A repeat is
# extra WEIGHT on one example, not extra data - 24 of the 149 sprite references
# are repeats of another one in the same set.
DUP_HAMMING = 4

# Above this, with no pixel grid, the image is a render or a painting. Harmless
# for `core`, which is concept art on purpose; wrong for `sprite`, which the UI
# measures for palette, cell grid and outline width.
MAX_SPRITE_COLORS = 1024


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def dhash(img: Image.Image, size: int = 8) -> int:
    g = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(g).astype(np.int16)
    v = 0
    for bit in (a[:, 1:] > a[:, :-1]).flatten():
        v = (v << 1) | int(bit)
    return v


def _two_tone_runs(sig: np.ndarray) -> tuple[bool, int]:
    """Does this 1-D border signal alternate between two flat levels?"""
    if sig.size < 24:
        return False, 0
    lo, hi = float(sig.min()), float(sig.max())
    sep = hi - lo
    if sep < 6:
        return False, 0                    # a flat backdrop, which is fine
    lab = sig > (lo + hi) / 2
    a, b = sig[lab], sig[~lab]
    if a.size < 6 or b.size < 6:
        return False, 0
    if max(float(a.std()), float(b.std())) > sep * 0.25:
        return False, 0                    # a gradient or a photo, not two tones
    edges = np.nonzero(np.diff(lab))[0]
    if edges.size < 3:
        return False, 0
    runs = np.diff(edges)
    med = float(np.median(runs))
    if runs.size < 2 or med < 3:
        return False, 0
    regular = float(np.mean(np.abs(runs - med) <= max(1, 0.2 * med)))
    return regular >= 0.75, int(round(med))


def baked_checkerboard(img: Image.Image, strip: int = 8) -> int:
    """Square size of a transparency checker painted INTO the pixels, else 0.

    Two earlier versions are recorded here so nobody rebuilds them:

      * Sampling an aligned block grid found 0 of 9 known cases. A checker's
        phase does not align with an arbitrary grid, so every sample averaged
        two half-squares and the alternation disappeared entirely.
      * Maximising autocorrelation over lag flagged 192 of 483, nearly all at
        lag 4 - any smooth or noisy signal correlates with itself at short lag.
        A statistic whose maximum is trivially near 1 separates nothing.

    What actually identifies a checker: the border alternates between exactly
    two flat tones, in runs of equal length, in BOTH axes. Requiring both axes
    is what distinguishes it from a striped or gradient backdrop.

    KNOWN MISS, and it is not worth "fixing"

    Both axes must pass, so a subject that runs to two opposite edges hides the
    checker on that axis and the image reads as clean. One real case in 483:
    `ref_core_a9a6b26fb952` - its left and right strips two-tone perfectly at
    period 15, its top and bottom are the creature, so this returns 0. It is a
    genuine baked checker.

    The fix would be to find the subject first and sample around it - but the
    checker is precisely what breaks segmentation, which is why it is detected
    before segmentation rather than after. Relaxing to one axis instead would
    admit every striped and gradient backdrop, and this function's precision is
    its entire value. So: it is documented, not patched.

    DEPENDENCY - `scripts/audit-entity-refs.py` importlib-loads this module and
    calls this function, taking the union with its own detector (which catches
    the case above). It degrades SILENTLY to local-only detection if the name
    or signature changes, so tell that script's owner before renaming it.
    """
    a = np.asarray(img)
    if (a[..., 3] < 128).mean() > 0.5:
        return 0                           # honestly transparent, not baked
    g = np.asarray(img.convert("L"), dtype=np.float32)
    h, w = g.shape
    if h < 64 or w < 64:
        return 0
    xs = [_two_tone_runs(g[:strip].mean(axis=0)),
          _two_tone_runs(g[-strip:].mean(axis=0))]
    ys = [_two_tone_runs(g[:, :strip].mean(axis=1)),
          _two_tone_runs(g[:, -strip:].mean(axis=1))]
    ok_x = [c for c in xs if c[0]]
    ok_y = [c for c in ys if c[0]]
    if not ok_x or not ok_y:
        return 0
    px, py = ok_x[0][1], ok_y[0][1]
    if abs(px - py) > max(2, 0.3 * max(px, py)):
        return 0                           # squares, not rectangles
    return px


def pixel_scale(arr: np.ndarray, max_scale: int = 16) -> int:
    """Screen pixels per art pixel, or 1 when there is no grid to find.

    Same method as measure.pixel_scale, kept local for the same reason the
    constants are: /app/scripts cannot import the /app package.
    """
    both = np.concatenate([arr[..., :3].astype(np.int16),
                           arr[..., 3:4].astype(np.int16)], axis=2)
    xs = np.nonzero(np.any(both[:, 1:, :] != both[:, :-1, :], axis=(0, 2)))[0] + 1
    ys = np.nonzero(np.any(both[1:, :, :] != both[:-1, :, :], axis=(1, 2)))[0] + 1
    coords = np.concatenate([xs, ys])
    if coords.size == 0:
        return 1
    for s in range(max_scale, 1, -1):
        if float(np.mean(coords % s == 0)) >= 0.98:
            return s
    return 1


def grid_round_trip_error(img: Image.Image, max_scale: int = 12) -> float:
    """How far the image is from being flat NxN blocks, at its best N.

    `pixel_scale` above answers "is there a grid, and how big". This answers a
    narrower question that was missing: "does this image carry noise inside its
    apparent blocks". An image can look like pixel art at thumbnail size and be
    noise all the way down. Collapsing each block to its mean is lossless for
    the genuine article and expensive for the imitation.

    ONE DIRECTION ONLY. High is a real finding. Low is not evidence of pixel
    art - a smooth gradient scores 0.42, since a soft ramp also varies little
    between neighbours. Used here inside the `scale == 1 and colors > MAX`
    branch, where the only question left is WHICH kind of not-pixel-art the
    image is, and both answers are already a rejection.

    Measured over this repo, in 0-255 units of mean absolute error:

        hand-made 64x96 references    0.00   (p10 and p90 also 0.00)
        1024x1024 sprite references   7.73
        the 103 recovered cells      10.79

    Duplicated in train-lora.py rather than shared, for the same reason
    everything else here is: /app/scripts cannot import the /app package, and
    the trainer and the audit disagreeing about what pixel art is was how this
    function came to be written in the first place. If you change one, change
    both - test-train-prep.py covers the trainer's copy.
    """
    rgb = np.asarray(img.convert("RGB")).astype(np.float32)
    h, w = rgb.shape[:2]
    best = float("inf")
    for n in range(2, max_scale + 1):
        if h // n < 8 or w // n < 8:
            break
        hh, ww = (h // n) * n, (w // n) * n
        blocks = rgb[:hh, :ww].reshape(hh // n, n, ww // n, n, 3)
        err = float(np.abs(
            blocks - blocks.mean(axis=(1, 3), keepdims=True)).mean())
        best = min(best, err)
    return 0.0 if best == float("inf") else best


def grid_by_profile(img: Image.Image, max_scale: int = 8) -> tuple[int, float]:
    """Grid factor read from the SHAPE of the block-error curve, or (1, err).

    The companion `pixel_scale` needs, and this one does not.

    `pixel_scale` asks whether 98% of edge coordinates fall on a multiple of s.
    That is exact and unforgiving: it reads a pristine 3x upscale perfectly and
    collapses to 1 on the same art after one lossy re-save, because the damage
    puts a difference between nearly every adjacent pair of lines. Measured on
    two files of the same art, one damaged (fraction of edges divisible by s):

        ref_core_08e39eb3c931   s=2 0.502  s=3 1.000  s=4 0.255  s=5 0.200
        ref_core_ca0070408096   s=2 0.503  s=3 0.358  s=4 0.252  s=5 0.198

    The second row is almost exactly 1/s - the signature of EVERY row and column
    registering as an edge. No s can clear 98%, so the detector says "no grid".
    Its block-error profile says otherwise, loudly: 5.15 at 2, 0.28 at 3, 10.32
    at 4. An 18x minimum.

    So this reads the minimum's DEPTH against the shallowest other factor - the
    hardest available comparison, so a merely smooth image cannot qualify by
    being flat everywhere.

    THE BLIND SPOTS ARE OPPOSITE, WHICH IS THE POINT. This one misses the
    cleanest art in the repo: palette-locked pixel art is flat at several
    factors at once, so its minimum has no depth to measure and eight 64x96
    references score a ratio near 1. `pixel_scale` reads all eight perfectly.
    Use both - `looks_like_pixel_art` below - and neither alone.

    Opaque pixels only. The transparent margin is flat and dilutes every factor
    equally, which flatters exactly the files that deserve it least.
    """
    arr = np.asarray(img.convert("RGBA"))
    rgb = np.asarray(img.convert("RGB")).astype(np.float32)
    h, w = rgb.shape[:2]
    prof: dict[int, float] = {}
    for n in range(2, max_scale + 1):
        if h // n < 8 or w // n < 8:
            break
        hh, ww = (h // n) * n, (w // n) * n
        b = rgb[:hh, :ww].reshape(hh // n, n, ww // n, n, 3)
        dev = np.abs(b - b.mean(axis=(1, 3), keepdims=True)).mean(axis=4)
        m = (arr[:hh, :ww, 3] >= 128).reshape(hh // n, n, ww // n, n)
        if m.any():
            prof[n] = float(dev[m].mean())
    if len(prof) < 3:
        return 1, float("inf")
    best = min(prof, key=lambda k: prof[k])
    second = min(v for k, v in prof.items() if k != best)
    deep = (second + 1e-6) / (prof[best] + 1e-6) >= MIN_GRID_PROFILE_DEPTH
    if deep and prof[best] <= MAX_GRID_ROUND_TRIP_ERROR:
        return best, prof[best]
    return 1, prof[best]


def looks_like_pixel_art(img: Image.Image, arr: np.ndarray) -> bool:
    """Either detector firing is evidence; they miss different things."""
    return pixel_scale(arr) > 1 or grid_by_profile(img)[0] > 1


# How much deeper the minimum must be than the shallowest other factor.
#
# SET AT THE BOTTOM OF THE OBSERVED RANGE, BECAUSE THERE IS NO GAP TO SIT IN.
# The first draft used 4.0 and called it measured. It was not: the ranked list
# over 586 images runs
#
#     ... 5.6  4.9  4.6 | 3.8  3.7  3.7  3.1  2.4  2.0  1.8   then nothing
#
# with the bar drawn at the pipe for no reason. Ten of those sixteen were
# opened and looked at, spanning the whole range - a spirit, a serpent, a
# minotaur, a dragon, a lich, an icon board, a skeleton turnaround, a 32-frame
# cat sheet at 3.8, a tree tutorial board at 3.7, a framed tree at 2.0. Every
# one is genuine pixel art. No false positive has been seen at ANY depth.
#
# Which means the boundary has not been found, not that there is none. The test
# is extremely specific either way - 16 hits out of 586 - so it is doing its
# work through specificity, not through this number.
#
# A NOTE ON A REJECTED ALTERNATIVE. A sibling session proposed `palette <= 300`
# as the second gate, having measured a clean gap from 246 colours to 4,455 in
# its own hit list. That gap does not exist in this one: the minotaur, dragon,
# lich and icon board above carry 15k-20k colours and are unambiguously pixel
# art by eye. Palette size identifies pixel art that was EXPORTED cleanly, and
# these are heavily shaded, lossily re-saved, or both - which is exactly the
# population this detector was added to catch. Adopting it would have discarded
# four confirmed files to close a gap that was an artefact of a smaller sample.
MIN_GRID_PROFILE_DEPTH = 2.0

# ABOVE this there is real noise inside the blocks. Read it in that direction
# only: below it means "not noisy", NOT "pixel art" - a smooth gradient scores
# 0.42 here. The gap on the populations that matter is enormous (0.00 against
# 7.73), so the exact value matters far less than their being disjoint.
MAX_GRID_ROUND_TRIP_ERROR = 1.0


def distinct_colors(arr: np.ndarray) -> int:
    opaque = arr[..., 3] >= 128
    if not opaque.any():
        return 0
    px = arr[opaque][:, :3].astype(np.uint32)
    return int(np.unique((px[:, 0] << 16) | (px[:, 1] << 8) | px[:, 2]).size)


def border_flatness(arr: np.ndarray) -> float:
    """Fraction of the border ring within tolerance of one colour."""
    rgb = arr[..., :3].astype(np.int16)
    ring = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    bg = np.median(ring, axis=0)
    return float((np.abs(ring - bg).sum(axis=1) <= 24).mean())


# A component bigger than this IS the background, or a frame around everything.
# Deliberately far above split-sheets' MAX_AREA_FRAC of 0.55 - see below.
SUBJECT_MAX_AREA_FRAC = 0.95

# A component smaller than this FRACTION OF THE LARGEST one is a signature, a
# watermark, a caption or a stray speck - not a co-equal subject.
#
# Measured, not guessed. Across the 55 references that segmented into 2-4
# components, the ratio of second-largest to largest is sharply bimodal: 36 sit
# at or below 0.078 and 19 at or above 0.151, with NOTHING in between. The
# low group is signatures and titles ("ID EV GAHIN", "KING 2023.5"); the high
# group is genuine turnarounds and multi-subject sheets. 0.12 sits in the empty
# gap, so the threshold is chosen by the data rather than by taste.
MINOR_COMPONENT_RATIO = 0.12


def count_subjects(img: Image.Image) -> tuple[int, str, int]:
    """How many subjects are in this image, and how confident that count is.

    WHY NOT JUST `len(ss.find_cells(img))`

    Because that answers a different question, and using it here reproduced the
    exact bug migration 014 was written to undo.

    `find_cells` asks "what can I safely CROP OUT of this?", so it discards any
    piece larger than 55% of the frame - correctly, since on a sheet such a
    piece is the backdrop or a border, not a subject. Asked "how many subjects
    are here?" that same rule throws away the answer: a creature centred on
    white, cropped tightly, is one component covering 60% of its frame, so
    `find_cells` returns nothing and the image reads as unsegmentable.

    Measured: of 169 core references `find_cells` returned nothing for, 141
    were exactly this - one correctly-framed subject. Only 25 genuinely could
    not be segmented. Migration 014 records `usable` rejecting 60 of 90
    characters for being "cropped tightly, which is what good reference art
    looks like"; this was the same mistake with a different rule.

    So the cap here is 0.95 - big enough to admit a tight crop, small enough to
    still exclude a component that is the whole image.

    Returns `(subjects, confidence, minor)`. `minor` counts the components too
    small RELATIVE to the largest to be a second subject - signatures, title
    bars, detached drop shadows, and loose specks of the art. Worth reporting,
    never a subject. A count that could not be established is never silently
    reported as 1.

    THE COUNT IS BADLY WRONG ON PACKED SHEETS, AND DELIBERATELY LEFT THAT WAY.

    `binary_closing` below joins a figure's own anti-aliased gaps so an arm
    separated by one transparent pixel is not a second subject. On a densely
    packed sheet the same closing bridges the GUTTERS BETWEEN sprites.
    `ref_sprite_939803c0ff65` is a 384px RPG Maker sheet with 1,082 connected
    components; this function reports 2.

    Why it is not fixed: the verdict is unaffected, and the fix is worse than
    the bug. 2 is already multi-subject, so the file is rejected either way.
    Checked across the whole set rather than assumed - of 139 kept references,
    ZERO have three or more substantial raw components, so no sheet has ever
    fused far enough to be waved through as a single subject. That was the only
    failure that would have changed an answer, and it does not occur here.

    Removing or shrinking the closing would trade this for the opposite error,
    on the case that is already the shakiest: a single creature with detached
    limbs. `ref_core_93f55ceabeae` is one treant that this function already
    calls 2 subjects WITH the closing. Without it, that class gets worse.

    So: read `subjects` as "at least this many, and at least 2 means a sheet",
    never as a quantity. If you need the real number on a sheet, `find_cells`
    gives it - it returned exactly 96 on a store page whose own banner reads
    "96 tiles".
    """
    from scipy import ndimage

    mask, _how = ss.foreground_mask(img)
    fg = float(mask.mean())
    if not (ss.FG_MIN <= fg <= ss.FG_MAX):
        # The honest refusal `split-sheets.py` documents: a sheet of ninety
        # weapons on near-black measures 87% "foreground" because the backdrop
        # is not quite uniform. Guessing harder is how bad cells get made.
        return 0, "unverifiable", 0

    h, w = mask.shape
    area = h * w
    r = max(1, int(round(min(h, w) / 256)))
    closed = ndimage.binary_closing(mask, structure=np.ones((2 * r + 1,) * 2))
    labels, n = ndimage.label(closed, structure=np.ones((3, 3)))
    if n == 0:
        return 0, "unverifiable", 0

    boxes = []
    for sl_y, sl_x in ndimage.find_objects(labels):
        bh, bw = sl_y.stop - sl_y.start, sl_x.stop - sl_x.start
        frac = (bh * bw) / area
        if frac < ss.MIN_AREA_FRAC or frac > SUBJECT_MAX_AREA_FRAC:
            continue
        if min(bh, bw) < ss.MIN_CELL_PX:
            continue
        boxes.append(bh * bw)

    if not boxes:
        return 0, "unverifiable", 0

    largest = max(boxes)
    big = sum(1 for b in boxes if b >= largest * MINOR_COMPONENT_RATIO)
    minor = len(boxes) - big

    # A sheet whose items TOUCH fuses into one blob, and only subdivision can
    # separate them - the packed-atlas case `find_cells` was written for. Ask
    # it too, and take whichever count is higher.
    #
    # Order matters: this runs AFTER the component pass, not instead of it.
    # `find_cells` discards any piece over 55% of the frame, so on a tightly
    # cropped single subject it returns nothing at all - which is the bug this
    # whole function exists to avoid re-introducing.
    cells = len(ss.find_cells(img))
    if cells >= 2:
        return max(cells, big), "counted", minor
    return big, "counted", minor


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def judge(path: str, kind: str, seen: list) -> dict:
    """Three outcomes, not two.

    `reject` is for what provably cannot teach. `review` is for what a human
    has to look at - an image that could not be segmented, or a subject sitting
    in a scene. Collapsing those two into one verdict is how the tile side
    ended up excluding 100 of 106 real sprites for having too many colours.
    """
    img = Image.open(path).convert("RGBA")
    arr = np.asarray(img)
    w, h = img.size
    alpha = arr[..., 3]

    subjects, confidence, minor = count_subjects(img)
    colors = distinct_colors(arr)
    scale = pixel_scale(arr)
    # Two detectors with opposite blind spots - see `grid_by_profile`. Using
    # only `pixel_scale` here read six files as having no grid that plainly do,
    # four of them single creatures on clean alpha: the best material in the
    # set, rejected for not being the thing it is.
    # Deferred, not computed here: it reshapes the whole image once per factor
    # and only the sprite branch below ever asks. Computing it eagerly for all
    # 483 references - two thirds of them `core`, many at 1200x1200 - took the
    # full audit from about two minutes to over seven.
    def profile_scale() -> int:
        return grid_by_profile(img)[0]
    checker = baked_checkerboard(img)
    flat = border_flatness(arr)
    transparent = float((alpha < 128).mean())
    h9 = dhash(img)

    reject, review = [], []

    if not (alpha >= 128).any():
        reject.append("nothing visible - fully transparent")
    if min(w, h) < MIN_TRAIN_SIDE:
        reject.append(f"{w}x{h} - below {MIN_TRAIN_SIDE}px, so training upscales "
                      f"it and invents the detail it is meant to learn")
    ratio = max(w, h) / max(min(w, h), 1)
    if ratio > MAX_TRAIN_ASPECT:
        reject.append(f"{ratio:.1f}:1 strip - the square centre crop discards "
                      f"most of it, and a strip this long is usually a sheet")
    if subjects >= 2:
        reject.append(f"{subjects} separable subjects - a contact sheet. "
                      f"Training on it teaches a grid of framed cells, which "
                      f"is exactly what both earlier adapters produced. Run "
                      f"split-sheets.py on it rather than dropping it")
    if checker:
        reject.append(f"transparency checkerboard baked into the pixels "
                      f"({checker}px squares) - the model learns to draw the "
                      f"checker and no prompt removes it. Key it out to real "
                      f"alpha and the image is fine")
    if (kind == "sprite" and scale == 1 and profile_scale() == 1
            and colors > MAX_SPRITE_COLORS):
        # The round-trip separates the two ways of failing this test, and they
        # want opposite things done about them. An honestly painted image
        # belongs under 'core'. An IMITATION of pixel art - blocky at thumbnail
        # size, per-pixel noise underneath - belongs nowhere: it is what
        # generated art looks like, and 013 already says not to train on that.
        #
        # This distinction is here because the old wording ("a render or a
        # painting") sent a reader looking for brush strokes, found none, and
        # concluded the gate was wrong. It was not. See 0009's recovery section.
        err = grid_round_trip_error(img)
        if err > MAX_GRID_ROUND_TRIP_ERROR:
            reject.append(
                f"{colors} colours, no pixel grid, and it does not survive a "
                f"round-trip through its own blocks (error {err:.1f}; hand-made "
                f"pixel art here scores 0.0) - blocky to look at, noise "
                f"underneath. Either art imitating pixel art, or real pixel "
                f"art that has been through JPEG; the measurement cannot tell "
                f"those apart and does not try. Both teach the noise")
        else:
            reject.append(
                f"{colors} colours and no pixel grid - a render or a painting, "
                f"not the finished pixel art this tab means. It may still "
                f"belong under 'core'")

    dup = next((f for hv, f in seen if bin(hv ^ h9).count("1") <= DUP_HAMMING), None)
    if dup and not reject:
        reject.append(f"near-duplicate of {dup} - a repeat is extra weight on "
                      f"one example, not extra data")

    if confidence == "unverifiable":
        review.append("foreground could not be told from background, so the "
                      "subject count is unknown - look at it before trusting it")
    if minor:
        # Reported, never rejected, and deliberately vague about WHAT the mark
        # is - because the first version of this message guessed, and guessed
        # wrong. It said "usually a signature or a title bar". Measured, the
        # marks are at least four different things: an artist's signature
        # (`ref_core_..` "ID EV GAHIN"), a title bar (the Mesgard sheets), a
        # DETACHED DROP SHADOW sitting clear of the subject, and loose specks
        # of the subject's own art - `ref_sprite_790879270b49` is a shadow plus
        # stray leaves, not a signature.
        #
        # Three of those four should be cropped out and the fourth is the art.
        # Nothing here can tell them apart, so it says what was measured and
        # leaves the call to a person.
        review.append(f"{minor} detached mark(s) beside the subject - a "
                      f"signature, a title bar, a drop shadow clear of the "
                      f"feet, or loose specks of the art itself. The first "
                      f"three train as part of the style; look before "
                      f"deciding")
    if flat < 0.9:
        review.append(f"only {flat:.0%} of the border is one colour - the "
                      f"subject sits in a scene, so the backdrop trains too")

    # NOT a flag: an opaque background.
    #
    # It was one, and it fired on 203 of 334 core references - i.e. on the
    # normal case, which is a subject on flat white. That is what concept art
    # looks like, the conveyor cuts it out downstream, and a flag that matches
    # most of the set sorts nothing. The backdrop that actually costs something
    # is a non-flat one, and `flat` above already catches that.

    verdict = "reject" if reject else ("review" if review else "keep")
    if verdict != "reject":
        seen.append((h9, os.path.basename(path)))

    return {
        "file": os.path.basename(path), "kind": kind, "verdict": verdict,
        "w": w, "h": h, "subjects": subjects, "subject_count": confidence,
        "minor_marks": minor,
        "colors": colors, "pixel_scale": scale, "checker_px": checker,
        "transparent_pct": round(transparent * 100, 1),
        "border_flat": round(flat, 3),
        "why": reject or review or ["single subject, flat backdrop, sane size"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", action="append", choices=["sprite", "core"],
                   help="repeatable; defaults to both")
    p.add_argument("--data", default="/app/images")
    p.add_argument("--out", default=None, help="write a markdown report here")
    p.add_argument("--json", default=None, help="write the raw rows here")
    p.add_argument("--apply", action="store_true",
                   help="write trainable=false + trainable_why for every "
                        "rejection. Needs DB_URL. 'review' rows are never "
                        "written - those are a human's call, not this script's")
    p.add_argument("--pattern", default=None,
                   help="glob for files that are not registered references, "
                        "e.g. 'cell_sprite_*.png' for split-sheets output. "
                        "Report only: these have no row to write back to")
    a = p.parse_args()
    kinds = a.kind or ["sprite", "core"]

    # --apply matches findings to reference_assets rows by filename. A custom
    # pattern means files that are not references at all, so every finding
    # would land nowhere - and the run would print `applied: 0` and exit 0,
    # which is the exact silent no-op the write path was fixed to stop being.
    if a.apply and a.pattern:
        sys.exit("--apply needs registered references; --pattern selects files "
                 "that have no reference_assets row. Drop one of the two.")

    rows = []
    for kind in kinds:
        pat = a.pattern or f"ref_{kind}_*.png"
        files = sorted(glob.glob(os.path.join(a.data, pat)))
        files = [f for f in files
                 if not os.path.basename(f).startswith("thumb_")]
        seen: list = []
        for i, f in enumerate(files):
            try:
                rows.append(judge(f, kind, seen))
            except Exception as e:
                rows.append({"file": os.path.basename(f), "kind": kind,
                             "verdict": "reject", "why": [f"unreadable: {e}"]})
            if (i + 1) % 25 == 0:
                print(f"  {kind} {i + 1}/{len(files)}", file=sys.stderr, flush=True)

    for kind in kinds:
        k = [r for r in rows if r["kind"] == kind]
        c = Counter(r["verdict"] for r in k)
        print(f"{kind:7s} n={len(k):<4d} keep={c['keep']:<4d} "
              f"review={c['review']:<4d} reject={c['reject']}")

    if a.json:
        with open(a.json, "w") as fh:
            json.dump(rows, fh, indent=1)
        print(f"wrote {a.json}")

    if a.out:
        with open(a.out, "w") as fh:
            fh.write("# Character reference audit\n")
            fh.write("\nGenerated by `scripts/audit-character-refs.py`. "
                     "This judges whether an image CAN teach, not which camera "
                     "it teaches - see the module docstring.\n")
            for kind in kinds:
                k = [r for r in rows if r["kind"] == kind]
                c = Counter(r["verdict"] for r in k)
                fh.write(f"\n## {kind} - {len(k)} references: {c['keep']} keep, "
                         f"{c['review']} review, {c['reject']} reject\n")
                for v in ("keep", "review", "reject"):
                    sel = [r for r in k if r["verdict"] == v]
                    if not sel:
                        continue
                    fh.write(f"\n### {v} ({len(sel)})\n\n")
                    for r in sel:
                        fh.write(f"- `{r['file']}` "
                                 f"{r.get('w', '?')}x{r.get('h', '?')} - "
                                 f"{'; '.join(r['why'])}\n")
        print(f"wrote {a.out}")

    if a.apply:
        db = os.environ.get("DB_URL")
        if not db:
            sys.exit("DB_URL unset - run this inside the container to --apply")
        import psycopg2
        conn = psycopg2.connect(db)
        cur = conn.cursor()
        n, missed = 0, []
        for r in rows:
            if r["verdict"] != "reject":
                continue
            # An EXACT suffix, not `LIKE '%name'`.
            #
            # Two reasons the LIKE version was wrong. The audit runs against a
            # local `--data ./images` while production stores
            # `/app/images/<name>`, so a whole-path comparison matches nothing -
            # and every filename here contains UNDERSCORES, which LIKE treats as
            # single-character wildcards. Neither would have failed loudly;
            # `right(...)` compares bytes and has no pattern semantics.
            needle = "/" + r["file"]
            cur.execute(
                "UPDATE reference_assets SET trainable = false, "
                "trainable_why = %s "
                "WHERE deleted = false "
                "  AND (file_path = %s OR right(file_path, %s) = %s)",
                ("; ".join(r["why"]), r["file"], len(needle), needle))
            if cur.rowcount:
                n += cur.rowcount
            else:
                missed.append(r["file"])
        conn.commit()
        conn.close()
        print(f"applied: {n} reference(s) marked not trainable")
        # "applied: 0" and "nothing needed changing" used to be the same
        # sentence. They are opposite outcomes: one is a clean no-op, the other
        # is every finding failing to match a row - a silent no-op reported as
        # success, which is the failure shape this whole audit exists to catch.
        if missed:
            print(f"NOT applied: {len(missed)} finding(s) matched no live row. "
                  f"Expected for references that are soft-deleted or were never "
                  f"registered; alarming if it is most of them.")
            for f in missed[:5]:
                print(f"    no row: {f}")
            if len(missed) > 5:
                print(f"    ... and {len(missed) - 5} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
