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
    if kind == "sprite" and scale == 1 and colors > MAX_SPRITE_COLORS:
        reject.append(f"{colors} colours and no pixel grid - a render or a "
                      f"painting, not the finished pixel art this tab means. "
                      f"It may still belong under 'core'")

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
    a = p.parse_args()
    kinds = a.kind or ["sprite", "core"]

    rows = []
    for kind in kinds:
        files = sorted(glob.glob(os.path.join(a.data, f"ref_{kind}_*.png")))
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
