#!/usr/bin/env python3
"""Turn a painted-on transparency checkerboard back into real alpha.

WHY THIS IS WORTH A SCRIPT

Twelve sprite references share one property and one problem. The property: they
are the only stylistically consistent character art in the whole reference set -
one artist, one palette, eight directions per character, hard edges, a real
pixel grid. The problem: whoever exported them screenshotted the editor, so the
transparency CHECKERBOARD is painted into the pixels.

That single fact disqualifies them twice over:

  * trained on directly, the adapter learns to draw the checker, and no prompt
    removes it;
  * `split-sheets.py` cannot split them either, because the checker is
    foreground as far as `foreground_mask` is concerned - it measures ~85%
    foreground, outside the FG_MIN..FG_MAX band, so it correctly refuses.

Key the checker out and both problems go at once: 12 unusable files become 131
single-character cells at a median 190px short side, which is above
`measure.MIN_TRAIN_SIDE` and is the only training-grade character material here.

WHY FLOOD FILL AND NOT A COLOUR MATCH

The first version matched every pixel within a tolerance of the two checker
tones, anywhere in the image. It removed the checker - and also punched holes
through white armour and pale skin, because those genuinely are the same
colour as the light square. You cannot tell them apart by colour, because they
ARE the same colour.

Connectivity tells them apart. The checker touches the border and the armour
does not, so a flood fill from the edges takes the backdrop and stops at the
sprite. Same tolerance, no holes.

Usage:
    python key-checkerboard.py --data ./images --dry-run
    python key-checkerboard.py --data ./images --out ./images/keyed --write
"""

import argparse
import glob
import os
import sys
from collections import deque

import numpy as np
from PIL import Image

# How far a pixel may sit from a checker tone and still count as backdrop.
# Measured against these twelve: the checker squares are flat, but PNG
# rescaling left a one-pixel soft seam between them, and a tolerance under ~24
# leaves that seam behind as a grid of thin lines - which is the artefact this
# script exists to remove, only fainter.
TOLERANCE = 36

# A checker whose two tones differ by less than this is not a checker, it is a
# flat backdrop with noise. Refuse rather than key a solid background out.
MIN_TONE_SEPARATION = 6


def _two_tone_runs(sig: np.ndarray) -> tuple[bool, int]:
    """Does this 1-D border signal alternate between two flat levels?"""
    if sig.size < 24:
        return False, 0
    lo, hi = float(sig.min()), float(sig.max())
    sep = hi - lo
    if sep < MIN_TONE_SEPARATION:
        return False, 0
    lab = sig > (lo + hi) / 2
    a, b = sig[lab], sig[~lab]
    if a.size < 6 or b.size < 6:
        return False, 0
    if max(float(a.std()), float(b.std())) > sep * 0.25:
        return False, 0
    edges = np.nonzero(np.diff(lab))[0]
    if edges.size < 3:
        return False, 0
    runs = np.diff(edges)
    med = float(np.median(runs))
    if runs.size < 2 or med < 3:
        return False, 0
    return float(np.mean(np.abs(runs - med) <= max(1, 0.2 * med))) >= 0.75, int(round(med))


def detect(img: Image.Image, strip: int = 8) -> int:
    """Checker square size in pixels, or 0 if this image has no baked checker.

    A checker repeats in BOTH axes at the same period. Requiring both is what
    separates it from a striped backdrop or a gradient, and requiring equal
    periods is what keeps it square.
    """
    a = np.asarray(img)
    if (a[..., 3] < 128).mean() > 0.5:
        return 0                          # already honestly transparent
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
        return 0
    return px


def _border_tones(rgb: np.ndarray) -> list[tuple[int, int, int]]:
    """The two most common exact colours around the border ring."""
    ring = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]).astype(np.uint32)
    packed = (ring[:, 0] << 16) | (ring[:, 1] << 8) | ring[:, 2]
    vals, counts = np.unique(packed, return_counts=True)
    return [((int(v) >> 16) & 255, (int(v) >> 8) & 255, int(v) & 255)
            for v in vals[np.argsort(-counts)[:2]]]


def key(img: Image.Image, tolerance: int = TOLERANCE) -> tuple[Image.Image, float]:
    """Replace the border-connected checker with alpha 0. Returns (image, removed)."""
    a = np.asarray(img).copy()
    rgb = a[..., :3].astype(np.int16)
    tones = _border_tones(a[..., :3])
    if len(tones) < 2:
        return img, 0.0

    # Candidates: any pixel close to either tone. This over-selects on purpose -
    # white armour looks exactly like the light square - and the flood fill
    # below is what narrows it back down to the actual backdrop.
    cand = np.zeros(rgb.shape[:2], bool)
    for t in tones:
        cand |= np.abs(rgb - np.array(t, np.int16)).sum(axis=2) <= tolerance

    h, w = cand.shape
    bg = np.zeros_like(cand)
    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if cand[y, x] and not bg[y, x]:
                bg[y, x] = True
                q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if cand[y, x] and not bg[y, x]:
                bg[y, x] = True
                q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and cand[ny, nx] and not bg[ny, nx]:
                bg[ny, nx] = True
                q.append((ny, nx))

    a[..., 3] = np.where(bg, 0, 255)
    return Image.fromarray(a), float(bg.mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/app/images")
    p.add_argument("--pattern", default="ref_sprite_*.png,ref_core_*.png")
    p.add_argument("--out", default=None,
                   help="directory for the keyed copies; required with --write")
    p.add_argument("--tolerance", type=int, default=TOLERANCE)
    p.add_argument("--write", action="store_true",
                   help="write the keyed copies. Originals are never touched - "
                        "a bad key is recoverable, an overwritten source is not")
    p.add_argument("--dry-run", action="store_true", help="the default; report only")
    a = p.parse_args()

    files: list[str] = []
    for pat in a.pattern.split(","):
        files.extend(sorted(glob.glob(os.path.join(a.data, pat.strip()))))
    files = [f for f in files if not os.path.basename(f).startswith("thumb_")]

    if a.write and not a.out:
        sys.exit("--write needs --out; this never overwrites the originals")
    if a.write:
        os.makedirs(a.out, exist_ok=True)

    hits = 0
    for f in files:
        img = Image.open(f).convert("RGBA")
        period = detect(img)
        if not period:
            continue
        hits += 1
        keyed, removed = key(img, a.tolerance)
        note = ""
        if removed < 0.2:
            note = "  <-- little removed; look at this one before trusting it"
        print(f"{os.path.basename(f):40s} checker={period:3d}px  "
              f"removed={removed:5.1%}{note}")
        if a.write:
            dest = os.path.join(a.out, os.path.basename(f))
            keyed.save(dest)

    print(f"\n{hits} of {len(files)} reference(s) carry a baked checkerboard")
    if hits and not a.write:
        print("(report only - pass --out DIR --write to produce keyed copies, "
              "then run split-sheets.py on them)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
