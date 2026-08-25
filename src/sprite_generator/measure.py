"""Measure reference art, so style is constrained by numbers rather than adjectives.

WHY THIS EXISTS

Style has been chased through prompt text for the whole life of this project,
and ADR 0005 recorded why that fails: the things that make art look like *this
game* - the palette, the pixel grid, the camera - are not things a text encoder
is reliable about. They are, however, directly measurable from a single example.

So the reference tabs do not just store examples. They measure them, and the
measurements become hard constraints on the conveyor.

Three kinds, three different questions:

    tile    What camera does this world use?  A ground tile's diamond is a
            direct readout of the projection angle. This is the highest-value
            measurement in the system and needs exactly one example.

    sprite  What is a pixel here, how many colours, is there an outline?
            Reproduces the target's grid rather than guessing at it.

    core    Is this a usable character concept at all? Delegated to
            `concept.judge`, which already answers exactly that.

WHAT THIS DELIBERATELY DOES NOT MEASURE

Line quality, shading idiom, anatomy. Those are style in the sense that only
training captures, and pretending a number describes them would be the same
mistake as the prompt-text era. Every metric here is one a human could verify
with a ruler.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
from PIL import Image

import concept as concept_lib

# The camera vocabulary Qwen was trained on, in degrees. Measurement produces a
# continuous angle; generation needs one of these four words, so the angle is
# snapped to the nearest. Kept in sync with qwen_edit.ELEVATIONS by name.
ELEVATION_DEGREES = {"low": -30.0, "eye": 0.0, "elevated": 30.0, "high": 60.0}

# Below this fraction of transparent pixels a "tile" is treated as a full-bleed
# rectangle rather than a diamond, and its aspect is measured from the image.
DIAMOND_MIN_TRANSPARENT = 0.08


def _rgba(path_or_image) -> Image.Image:
    img = (Image.open(path_or_image) if isinstance(path_or_image, (str, bytes))
           else path_or_image)
    return img.convert("RGBA")


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------

def pixel_scale(arr: np.ndarray, max_scale: int = 16) -> dict:
    """How many screen pixels one *art* pixel occupies.

    A 32x32 sprite exported at 128x128 has a scale of 4, and every colour
    boundary in it falls on a multiple of 4. So: collect the coordinates where
    colour changes, then find the largest scale that explains almost all of
    them.

    Returns the scale and the fraction of boundaries it explains, because a
    photo or a smooth render has no honest answer here and the caller needs to
    know that rather than receive a confident 1.
    """
    rgb = arr[..., :3].astype(np.int16)
    alpha = arr[..., 3:4].astype(np.int16)
    both = np.concatenate([rgb, alpha], axis=2)

    # Boundary columns: any x where column x differs from column x-1, and the
    # same for rows. Note the axes differ - the column test collapses rows
    # (0) and channels (2); the row test collapses columns (1) and channels.
    # Using (0, 2) for both measures columns twice and calls half of it rows,
    # which fills `coords` with every index 1..H and buries the real grid.
    col_change = np.any(both[:, 1:, :] != both[:, :-1, :], axis=(0, 2))
    row_change = np.any(both[1:, :, :] != both[:-1, :, :], axis=(1, 2))
    xs = np.nonzero(col_change)[0] + 1
    ys = np.nonzero(row_change)[0] + 1

    coords = np.concatenate([xs, ys])
    if coords.size == 0:
        return {"scale": None, "confidence": 0.0,
                "scale_why": "no colour boundaries - image is a flat fill"}

    best, best_frac = 1, 1.0
    for s in range(max_scale, 1, -1):
        frac = float(np.mean(coords % s == 0))
        # 0.98 rather than 1.0: a single stray anti-aliased pixel should not
        # veto an otherwise perfect grid.
        if frac >= 0.98:
            best, best_frac = s, frac
            break
    else:
        best_frac = float(np.mean(coords % 1 == 0))

    return {"scale": best, "confidence": round(best_frac, 3),
            "scale_why": ("clean pixel grid" if best > 1 else
                    "no upscaling detected - 1 art pixel per image pixel")}


def palette_of(arr: np.ndarray, alpha_threshold: int = 128) -> dict:
    """Exact distinct colours among opaque pixels, most common first.

    Exact, not median-cut: the question here is "how many colours does the
    target actually use", and quantising first would answer a question nobody
    asked.
    """
    opaque = arr[..., 3] >= alpha_threshold
    if not opaque.any():
        return {"colors": 0, "palette": [], "palette_why": "fully transparent"}

    px = arr[opaque][:, :3].astype(np.uint32)

    # Pack RGB into one integer and let numpy do the counting.
    #
    # This was `Counter(map(tuple, px.tolist()))`, which is a Python-level tuple
    # allocation per pixel. On a 1200x3610 reference board - 4.3 million pixels
    # - that took tens of seconds, and re-measuring 227 references ran for over
    # seven minutes before this replaced it. Same exact answer, vectorised.
    packed = (px[:, 0] << 16) | (px[:, 1] << 8) | px[:, 2]
    values, counts = np.unique(packed, return_counts=True)
    order = np.argsort(-counts)[:64]
    ordered = [((int(v) >> 16) & 0xFF, (int(v) >> 8) & 0xFF, int(v) & 0xFF)
               for v in values[order]]

    return {
        "colors": int(values.size),
        # Cap the stored list: a photo would otherwise write 200k entries into
        # JSONB. The count above is still exact.
        "palette": [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in ordered],
        "palette_why": ("bounded palette" if len(ordered) <= 64 else
                f"{len(ordered)} distinct colours - not pixel art, or not "
                f"palette-locked"),
    }


def alpha_profile(arr: np.ndarray) -> dict:
    """Whether alpha is binary. Pixel art wants hard edges; a soft edge means
    the source was resampled and its grid can no longer be trusted."""
    a = arr[..., 3]
    total = a.size
    partial = int(np.count_nonzero((a > 8) & (a < 248)))
    return {
        "transparent_pct": round(float(np.mean(a < 128)) * 100, 1),
        "partial_alpha_px": partial,
        "binary_alpha": partial <= total * 0.005,
    }


def outline_of(arr: np.ndarray, alpha_threshold: int = 128) -> dict:
    """Detect a dark outline around the silhouette.

    Looks only at opaque pixels that touch transparency, and asks whether one
    colour dominates that rim. A selective or absent outline gives a low
    dominance, which is reported rather than rounded up to "yes".
    """
    opaque = arr[..., 3] >= alpha_threshold
    if not opaque.any():
        return {"has_outline": False, "outline_why": "nothing opaque"}

    pad = np.pad(opaque, 1, constant_values=False)
    neighbours = (pad[:-2, 1:-1] & pad[2:, 1:-1] &
                  pad[1:-1, :-2] & pad[1:-1, 2:])
    rim = opaque & ~neighbours
    if rim.sum() < 16:
        return {"has_outline": False, "outline_why": "silhouette too small to judge"}

    rim_px = arr[rim][:, :3]
    counts = Counter(map(tuple, rim_px.tolist()))
    (top, n), = counts.most_common(1)
    dominance = n / rim.sum()
    luma = 0.2126 * top[0] + 0.7152 * top[1] + 0.0722 * top[2]

    has = dominance >= 0.5 and luma < 96
    return {
        "has_outline": bool(has),
        "outline_color": f"#{top[0]:02x}{top[1]:02x}{top[2]:02x}",
        "outline_dominance": round(float(dominance), 3),
        "outline_why": ("consistent dark outline" if has else
                f"no single outline colour (most common rim colour covers "
                f"{dominance:.0%}{', and is not dark' if luma >= 96 else ''})"),
    }


# ---------------------------------------------------------------------------
# Trainability - a DIFFERENT question from measurability
# ---------------------------------------------------------------------------
#
# WHY THIS IS SEPARATE, learned the hard way on 2026-08-25
#
# `usable` originally gated both deriving a style profile AND training. Against
# 227 real references that rejected 100 of 106 sprites and 60 of 90 characters,
# and it was wrong nearly every time:
#
#   * sprites were rejected for "32,000 distinct colours" - they were JPEG
#     reference boards, not palette-locked art. You cannot read a PALETTE from
#     them, which is a real limitation. You can absolutely TRAIN a style on
#     them; SDXL was trained on photographs.
#   * characters were rejected for "fills the frame; reaches the border" with
#     an average aspect of 1.36 - i.e. they were correctly character-shaped and
#     simply CROPPED TIGHTLY, which is what good reference art looks like. The
#     isolation rule came from `concept.judge`, which answers "did the
#     GENERATOR produce a scene instead of a character?" - the opposite
#     expectation from a reference a human chose.
#
# So the two questions are asked separately now. Measurement stays strict,
# because a palette read off a JPEG collage would be garbage that silently
# poisons every sheet. Training is permissive, because it can afford to be.

# Below this the image is upscaled to reach the training resolution, which
# invents detail rather than teaching any. The smallest real reference measured
# here was 236x208.
MIN_TRAIN_SIDE = 160

# Training centre-crops to a square. Beyond roughly 3:1 that throws most of the
# image away, and a strip that long is usually a CONTACT SHEET of many subjects
# - which teaches the model to draw contact sheets.
MAX_TRAIN_ASPECT = 3.0


def judge_trainable(img: Image.Image, arr: np.ndarray) -> dict:
    """Can a style LoRA learn from this image? Deliberately permissive.

    Rejects only what genuinely cannot teach: blank images, images too small to
    survive the training resolution, and extreme strips that are nearly always
    multi-subject sheets.
    """
    w, h = img.size
    reasons = []

    if arr.shape[2] == 4:
        opaque = arr[..., 3] >= 128
        if not opaque.any():
            reasons.append("nothing visible - fully transparent")

    short = min(w, h)
    if short < MIN_TRAIN_SIDE:
        reasons.append(f"only {w}x{h} - below {MIN_TRAIN_SIDE}px the training "
                       f"resolution has to upscale it, which invents detail "
                       f"rather than learning any")

    long_side, short_side = max(w, h), max(min(w, h), 1)
    ratio = long_side / short_side
    if ratio > MAX_TRAIN_ASPECT:
        reasons.append(f"{ratio:.1f}:1 strip - training centre-crops to a "
                       f"square, so most of this would be discarded, and a "
                       f"strip this long is usually a sheet of several "
                       f"subjects. Split it into individual images first")

    return {"trainable": not reasons,
            "trainable_why": "; ".join(reasons) or "fine to train on"}


# ---------------------------------------------------------------------------
# Per-kind measurement
# ---------------------------------------------------------------------------

def measure_tile(path_or_image) -> dict:
    """Read the world's projection off a ground tile.

    An isometric ground tile is a rhombus inscribed in its image. The rhombus's
    width:height ratio IS the projection, and the camera elevation that
    produces it is atan(height / width):

        2 : 1   -> 26.6 deg   classic dimetric, most 2D "isometric" games
        1.73:1  -> 30.0 deg   true isometric
        1 : 1   -> 45.0 deg   top-down-ish

    This is the measurement worth the whole feature. `QWEN_ISO_ELEVATION`
    currently defaults to "eye" - 0 degrees, a flat side-on camera - chosen in
    ADR 0005 because "elevated" cropped legs. That fixed a framing symptom with
    a camera change, and no tile had ever been consulted.
    """
    img = _rgba(path_or_image)
    arr = np.asarray(img)
    prof = alpha_profile(arr)

    opaque = arr[..., 3] >= 128
    if not opaque.any():
        return {"usable": False, "why": "tile is fully transparent",
                "metrics": {}, "trainable": False,
                "trainable_why": "nothing visible - fully transparent"}

    ys, xs = np.nonzero(opaque)
    w = int(xs.max() - xs.min() + 1)
    h = int(ys.max() - ys.min() + 1)

    transparent = np.mean(arr[..., 3] < 128)
    diamond = transparent >= DIAMOND_MIN_TRANSPARENT

    ratio = w / max(h, 1)
    elevation_deg = math.degrees(math.atan(h / max(w, 1)))

    # The vocabulary is four discrete words, so a measured angle has to snap.
    # A 45 deg tile sits exactly between 'elevated' (30) and 'high' (60), and
    # picking one silently would hide a genuine ambiguity behind a confident
    # answer - so ties are reported.
    by_distance = sorted(ELEVATION_DEGREES,
                         key=lambda k: abs(ELEVATION_DEGREES[k] - elevation_deg))
    nearest = by_distance[0]
    runner_up = by_distance[1]
    tied = abs(abs(ELEVATION_DEGREES[nearest] - elevation_deg)
               - abs(ELEVATION_DEGREES[runner_up] - elevation_deg)) < 1e-6

    metrics = {
        "diamond": bool(diamond),
        "tile_w": w, "tile_h": h,
        "projection_ratio": round(float(ratio), 3),
        "elevation_deg": round(float(elevation_deg), 1),
        "elevation": nearest,
        "elevation_ambiguous": bool(tied),
        "elevation_runner_up": runner_up,
        **prof,
        **pixel_scale(arr),
        **palette_of(arr),
    }

    if not diamond:
        # Still returns the ratio - a rectangular tile sheet is a legitimate
        # thing to upload - but says the number is weaker evidence.
        return {"usable": True, "metrics": metrics,
                **judge_trainable(img, arr),
                "why": (f"no transparent corners ({transparent:.0%} "
                        f"transparent), so this is measured as a rectangle, "
                        f"not a diamond. The angle is a guess; upload a tile "
                        f"with transparent corners for a real reading.")}

    note = (f"{ratio:.2f}:1 diamond -> {elevation_deg:.1f} deg camera "
            f"-> nearest supported elevation '{nearest}'")
    if tied:
        note += (f" (exactly between '{nearest}' and '{runner_up}' - the "
                 f"measurement does not choose; pick one deliberately)")
    return {"usable": True, "metrics": metrics, "why": note,
            **judge_trainable(img, arr)}


def measure_sprite(path_or_image) -> dict:
    """Read the pixel grid, palette and outline off finished sprite art."""
    img = _rgba(path_or_image)
    arr = np.asarray(img)

    scale = pixel_scale(arr)
    pal = palette_of(arr)
    prof = alpha_profile(arr)
    out = outline_of(arr)

    s = scale["scale"] or 1
    metrics = {
        "image_w": img.width, "image_h": img.height,
        "art_w": img.width // s, "art_h": img.height // s,
        **scale, **pal, **prof, **out,
    }

    reasons = []
    if pal["colors"] > 256:
        reasons.append(f"{pal['colors']} distinct colours - this looks like a "
                       f"render, not palette-locked pixel art")
    if not prof["binary_alpha"]:
        reasons.append(f"{prof['partial_alpha_px']} semi-transparent pixels - "
                       f"edges are anti-aliased, so the pixel grid is unreliable")

    return {"usable": not reasons, "metrics": metrics,
            "why": "; ".join(reasons) or
                   (f"{metrics['art_w']}x{metrics['art_h']} art pixels at "
                    f"scale {s}, {pal['colors']} colours"),
            **judge_trainable(img, arr)}


def measure_core(path_or_image) -> dict:
    """Is this a usable character concept? Delegated to the existing judge."""
    v = concept_lib.judge(path_or_image)
    img = _rgba(path_or_image)
    arr = np.asarray(img)
    return {
        "usable": bool(v["ok"]),
        "why": concept_lib.describe(v),
        **judge_trainable(img, arr),
        "metrics": {"coverage": round(v["coverage"], 4),
                    "border": round(v["border"], 4),
                    "aspect": round(v["aspect"], 3),
                    **palette_of(arr), **alpha_profile(arr)},
    }


MEASURERS = {"tile": measure_tile, "sprite": measure_sprite, "core": measure_core}


def measure(kind: str, path_or_image) -> dict:
    if kind not in MEASURERS:
        raise ValueError(f"unknown reference kind {kind!r}; expected one of "
                         f"{', '.join(sorted(MEASURERS))}")
    return MEASURERS[kind](path_or_image)
