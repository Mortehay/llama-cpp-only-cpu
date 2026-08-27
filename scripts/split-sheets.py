#!/usr/bin/env python3
"""Split contact sheets into single-subject cells.

WHY THIS EXISTS

Both trained adapters produced a lattice of framed cells instead of the subject
they were asked for. The cause was the dataset: the references are asset packs -
"Tile set" grids, packed sprite atlases, a sheet of ninety weapon icons - so the
most consistent feature in the data was "a regular grid of bordered cells", and
that is what the adapters learned.

The fix is not to reject those images. A sheet of forty icons is forty perfectly
good training images that happen to share a file, so splitting turns the dataset's
biggest liability into its biggest source of material.

WHY SEGMENTATION AND NOT GRID DETECTION

Two attempts at a *classifier* ("is this a sheet?") were written and thrown away;
both misclassified sheets that were obviously sheets by eye. Grid detection has
the same weakness - it assumes even spacing, which packed atlases do not have.

Connected components need no such assumption, and they make the question moot:
run every image through this, and whatever comes out is single-subject BY
CONSTRUCTION. One component means it was already a single subject. Forty means
it was a sheet. Nothing has to classify anything, which is the property that
makes this trustworthy where the detectors were not.

WHAT IT WILL NOT HANDLE

A subject on a photographic or textured background - there is no separable
foreground, so it yields one component and the image passes through unchanged.
That is the safe failure: it declines to split rather than splitting wrongly.

Usage:
    python split-sheets.py --kind tile --dry-run
    python split-sheets.py --kind tile --out /app/images/cells --write
"""

import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

# A component smaller than this fraction of the image is a speck: JPEG noise, a
# caption's letter, a stray antialiased pixel.
MIN_AREA_FRAC = 0.0004

# ...and one larger than this is the background itself, or a frame around the
# whole sheet, not a subject in it.
MAX_AREA_FRAC = 0.55

# Below this a crop cannot teach anything at training resolution.
MIN_CELL_PX = 24

# A sheet with more cells than this is almost certainly being over-segmented.
MAX_CELLS = 400


def foreground_mask(img: Image.Image) -> tuple[np.ndarray, str]:
    """Boolean mask of "not background", plus how the background was decided."""
    a = np.asarray(img.convert("RGBA"))
    alpha = a[..., 3]
    if (alpha < 128).mean() > 0.02:
        return alpha >= 128, "alpha"

    # No usable transparency: treat the modal border colour as the backdrop.
    # Asset sheets are overwhelmingly flat black, white or mid-grey behind the
    # items, which is exactly what the border samples.
    rgb = a[..., :3].astype(np.int16)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    bg = np.median(border, axis=0)
    return np.abs(rgb - bg).sum(axis=2) > 60, "border-colour"


def _valleys(profile: np.ndarray, min_run: int) -> list[int]:
    """Cut points where the foreground profile dips to near-nothing.

    Connected components alone cannot separate items that TOUCH, and packed
    atlases are full of them - the first version returned a whole row of eight
    tiles as one component, and a sheet of ninety weapons as zero (everything
    merged into one blob larger than the size cap, so it was discarded).

    Where items abut they still leave a thin gutter, so a column-sum profile
    across the component dips sharply between them. Cutting at those dips
    separates what closure fused.
    """
    if profile.size < 2 * min_run:
        return []
    # "Near-nothing" relative to this component, not an absolute: a row of dark
    # tiles and a row of bright icons have very different profile magnitudes.
    floor = profile.max() * 0.18
    low = profile <= floor

    cuts, run_start = [], None
    for i, is_low in enumerate(low):
        if is_low and run_start is None:
            run_start = i
        elif not is_low and run_start is not None:
            if i - run_start >= 1:
                cuts.append((run_start + i) // 2)
            run_start = None
    # Drop cuts that would produce a sliver rather than a subject.
    return [c for c in cuts if min_run <= c <= profile.size - min_run]


def _subdivide(mask: np.ndarray, box, depth: int = 0):
    """Split a box on foreground valleys, recursively, longest axis first."""
    x0, y0, x1, y1 = box
    sub = mask[y0:y1, x0:x1]
    h, w = sub.shape
    if depth >= 3 or min(h, w) < 2 * MIN_CELL_PX:
        return [box]

    # Only attempt a split on an elongated box: a roughly square component is
    # far more likely to be one subject than several in a line.
    if w >= h:
        if w / max(h, 1) < 1.5:
            return [box]
        cuts = _valleys(sub.sum(axis=0), MIN_CELL_PX)
        if not cuts:
            return [box]
        out, prev = [], 0
        for c in cuts + [w]:
            if c - prev >= MIN_CELL_PX:
                out.extend(_subdivide(mask, (x0 + prev, y0, x0 + c, y1), depth + 1))
            prev = c
        return out

    if h / max(w, 1) < 1.5:
        return [box]
    cuts = _valleys(sub.sum(axis=1), MIN_CELL_PX)
    if not cuts:
        return [box]
    out, prev = [], 0
    for c in cuts + [h]:
        if c - prev >= MIN_CELL_PX:
            out.extend(_subdivide(mask, (x0, y0 + prev, x1, y0 + c), depth + 1))
        prev = c
    return out


# Foreground fraction outside this band means the background was not found.
#
# Measured on a real failure: a 1152x2048 sheet of ~90 weapon icons on near-black
# came back 87% "foreground", because the backdrop is not quite uniform and the
# border median missed it. Everything then fused into one blob with no valleys
# to cut, and the sheet yielded zero cells.
#
# Rather than guess harder at the backdrop, refuse. An image whose foreground is
# 87% has not been segmented, and cropping it would produce 90 bad cells instead
# of one honest skip.
FG_MIN, FG_MAX = 0.01, 0.75

# A cell far longer than it is tall is a merged ROW of subjects, not a subject.
# This is the packed-atlas failure: adjacent tiles touch, so closure fuses them
# and there is no gutter to cut at. Dropping them keeps the bad cells out of the
# dataset at the cost of yield, which is the right way round - the whole reason
# this script exists is that bad training data is expensive and silent.
MAX_CELL_ASPECT = 3.0


def find_cells(img: Image.Image) -> list[tuple[int, int, int, int]]:
    """Bounding boxes of the separable subjects, largest first.

    Returns [] when the image cannot be segmented CONFIDENTLY. Callers treat
    that as "leave this one alone", never as "it has no subjects".
    """
    mask, _ = foreground_mask(img)
    if not (FG_MIN <= mask.mean() <= FG_MAX):
        return []
    h, w = mask.shape
    area = h * w

    # Close small gaps so one object does not fragment. A sword with a gap
    # between blade and pommel, or a tile whose highlight is a separate blob,
    # should stay one cell. Radius scales with the image so a 2048px sheet is
    # not treated more finely than a 256px one.
    r = max(1, int(round(min(h, w) / 256)))
    closed = ndimage.binary_closing(mask, structure=np.ones((2 * r + 1,) * 2))

    labels, n = ndimage.label(closed, structure=np.ones((3, 3)))
    if n == 0:
        return []

    boxes = []
    for sl_y, sl_x in ndimage.find_objects(labels):
        bh, bw = sl_y.stop - sl_y.start, sl_x.stop - sl_x.start
        if bh < MIN_CELL_PX or bw < MIN_CELL_PX:
            continue
        if (bh * bw) / area < MIN_AREA_FRAC:
            continue
        # A large component is NOT discarded here any more. On a packed atlas
        # everything fuses into one near-full-frame blob, and dropping it
        # returned zero cells for a sheet of ninety weapons. Subdivide first
        # and judge the PIECES.
        boxes.extend(_subdivide(closed, (sl_x.start, sl_y.start,
                                         sl_x.stop, sl_y.stop)))

    # Now apply the "this is the background, not a subject" cap, to the pieces.
    kept = []
    for b in boxes:
        bw, bh = b[2] - b[0], b[3] - b[1]
        if bw < MIN_CELL_PX or bh < MIN_CELL_PX:
            continue
        if (bw * bh) / area > MAX_AREA_FRAC:
            continue
        if max(bw, bh) / max(min(bw, bh), 1) > MAX_CELL_ASPECT:
            continue
        kept.append(b)

    kept.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return kept[:MAX_CELLS]


def cell_is_junk(cell: Image.Image) -> str | None:
    """Reason to discard a cell, or None to keep it.

    Segmentation is indifferent to what a region MEANS. On the character
    references it happily cropped letters out of titles ("GANIN", "EV"), flat
    blue rectangles, and swatches of solid colour - all perfectly good
    connected components, all useless or harmful as training images.

    These two tests catch the cheap cases. They do NOT catch text on a card,
    which stays a known contaminant - see the note in the module docstring.
    """
    a = np.asarray(cell.convert("RGBA"))
    opaque = a[..., 3] >= 128
    if opaque.mean() < 0.15:
        return "almost empty"

    rgb = a[..., :3][opaque]
    if rgb.size == 0:
        return "nothing opaque"

    # A flat swatch has almost no distinct colours once quantised coarsely.
    packed = ((rgb[:, 0] >> 4) << 8) | ((rgb[:, 1] >> 4) << 4) | (rgb[:, 2] >> 4)
    if np.unique(packed).size < 6:
        return "flat colour swatch"

    # ...and almost no internal structure. Gradient magnitude on the luma plane
    # separates "a picture of something" from "a rectangle".
    grey = np.asarray(cell.convert("L"), dtype=np.float32)
    if grey.size and (np.abs(np.diff(grey, axis=0)).mean()
                      + np.abs(np.diff(grey, axis=1)).mean()) < 2.0:
        return "no internal detail"
    return None


def crop_cell(img: Image.Image, box, pad: int = 2) -> Image.Image:
    x0, y0, x1, y1 = box
    w, h = img.size
    return img.convert("RGBA").crop((max(0, x0 - pad), max(0, y0 - pad),
                                     min(w, x1 + pad), min(h, y1 + pad)))


# A component this small next to the cell's main subject, AND touching the crop
# edge, is a piece of the NEIGHBOURING cell rather than part of this one.
EDGE_SLIVER_MAX_RATIO = 0.12


def drop_edge_slivers(cell: Image.Image) -> tuple[Image.Image, int]:
    """Erase fragments of the neighbouring subject that the crop rectangle caught.

    WHY A RECTANGLE IS NOT ENOUGH

    `find_cells` returns a bounding BOX per subject, and subjects on a sheet
    are not box-shaped. A wizard's staff leans up and to the left, so it hangs
    over the corner of the box belonging to the character beside him, and the
    crop takes it. The cell is single-subject by component analysis and still
    has a stray sliver of somebody else in the corner.

    Found on 9 of 103 recovered character cells - all ~0.1% of the frame, all a
    neighbour's staff tip. Cheap to remove here and annoying to remove later.

    BOTH CONDITIONS ARE REQUIRED, and the second is what makes this safe:

      * SMALL relative to the main subject - so a genuine companion survives.
        Several cells in that set are a dwarf WITH a wolf standing beside him,
        two large components, both wanted.
      * TOUCHING THE CROP EDGE - so a subject's own detached parts survive. A
        floating spark or a dropped accessory sits inside the cell; a piece of
        the neighbour is necessarily clipped by the boundary it came across.

    Returns the cleaned cell and how many fragments were erased.
    """
    a = np.asarray(cell.convert("RGBA")).copy()
    opaque = a[..., 3] >= 128
    if not opaque.any():
        return cell, 0

    labels, n = ndimage.label(opaque, structure=np.ones((3, 3)))
    if n < 2:
        return cell, 0

    sizes = ndimage.sum(opaque, labels, range(1, n + 1))
    largest = float(sizes.max())
    h, w = opaque.shape

    removed = 0
    for i, size in enumerate(sizes, start=1):
        if size >= largest * EDGE_SLIVER_MAX_RATIO:
            continue
        blob = labels == i
        touches = (blob[0].any() or blob[-1].any()
                   or blob[:, 0].any() or blob[:, -1].any())
        if not touches:
            continue                      # the subject's own, keep it
        a[..., 3][blob] = 0
        removed += 1

    return (Image.fromarray(a), removed) if removed else (cell, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", default="tile", help="tile, core or sprite")
    p.add_argument("--images", default="/app/images")
    p.add_argument("--out", default="/app/images/cells")
    p.add_argument("--write", action="store_true",
                   help="actually write cells; otherwise report only")
    p.add_argument("--limit", type=int, default=0,
                   help="only process this many source images")
    p.add_argument("--sample", type=int, default=0,
                   help="write only this many cells, for eyeballing")
    p.add_argument("--min-side", type=int, default=0,
                   help="skip cells whose short side is below this. 0 keeps "
                        "the old behaviour of writing every cell. Set it to "
                        "measure.MIN_TRAIN_SIDE (160) when the output is going "
                        "to a TRAINING set: the tile side learned that a 63px "
                        "median cell upscaled to 1024 is mush, and mush with a "
                        "filename looks exactly like data")
    p.add_argument("--drop-edge-slivers", action="store_true",
                   help="erase fragments of the NEIGHBOURING subject that the "
                        "crop rectangle caught. A leaning staff overhangs the "
                        "box of the character beside it, so a cell can be "
                        "single-subject and still carry a sliver of someone "
                        "else. Off by default; it edits pixels")
    p.add_argument("--max-aspect", type=float, default=0.0,
                   help="skip cells more elongated than this (long/short). 0 "
                        "keeps every cell, subject to the 3.0 the segmenter "
                        "already applies. 2.0 is the useful value on character "
                        "sheets: it drops the TITLE BANNER, which is a "
                        "perfectly good component and reads as a subject. "
                        "Measured on the recovered set - the four banners sit "
                        "at 2.83-2.97 and the most elongated real character at "
                        "1.43, so anything in that gap works")
    p.add_argument("--register", action="store_true",
                   help="write cells AND record them as references, retiring "
                        "the sheets they came from")
    a = p.parse_args()

    if a.register:
        db = os.environ.get("DB_URL")
        if not db:
            sys.exit("--register needs DB_URL; run this inside the container")
        register_cells(a.kind, a.images, db)
        return 0

    files = sorted(glob.glob(os.path.join(a.images, f"ref_{a.kind}_*.png")))
    files = [f for f in files if not os.path.basename(f).startswith("thumb_")]
    if a.limit:
        files = files[:a.limit]
    if not files:
        sys.exit(f"no ref_{a.kind}_*.png under {a.images}")

    if a.write or a.sample:
        os.makedirs(a.out, exist_ok=True)

    singles = sheets = written = too_small = too_long = slivers = 0
    hist = {}
    for f in files:
        try:
            img = Image.open(f)
        except Exception as e:
            print(f"  unreadable {os.path.basename(f)}: {e}")
            continue

        boxes = find_cells(img)
        n = len(boxes)
        hist[n] = hist.get(n, 0) + 1
        if n <= 1:
            singles += 1
        else:
            sheets += 1

        if not (a.write or a.sample):
            continue
        if a.sample and written >= a.sample:
            continue

        stem = os.path.splitext(os.path.basename(f))[0]
        for i, box in enumerate(boxes):
            if a.sample and written >= a.sample:
                break
            cell = crop_cell(img, box)
            cw, ch = cell.size
            if a.min_side and min(cw, ch) < a.min_side:
                too_small += 1
                continue
            if a.max_aspect and max(cw, ch) / max(min(cw, ch), 1) > a.max_aspect:
                too_long += 1
                continue
            if a.drop_edge_slivers:
                cell, gone = drop_edge_slivers(cell)
                slivers += gone
            if cell_is_junk(cell):
                continue
            cell.save(os.path.join(a.out, f"cell_{a.kind}_{stem[4:]}_{i:03d}.png"))
            written += 1

    print(f"kind={a.kind}  sources={len(files)}")
    print(f"  would split (>=2 subjects): {sheets}")
    print(f"  single subject already:     {singles}")
    total_cells = sum(k * v for k, v in hist.items() if k >= 2)
    print(f"  cells the sheets would yield: {total_cells}")
    print("  subjects-per-image distribution (count: images):")
    for k in sorted(hist)[:14]:
        print(f"    {k:4}: {hist[k]}")
    if a.write or a.sample:
        print(f"  wrote {written} cell(s) to {a.out}")
        # Say what was dropped. A count of what was written, on its own, reads
        # as "this is everything" - and the whole tile-side lesson was that the
        # cells nobody looked at were the problem.
        if too_small:
            print(f"  skipped {too_small} cell(s) under {a.min_side}px")
        if too_long:
            print(f"  skipped {too_long} cell(s) longer than "
                  f"{a.max_aspect}:1 (usually a title banner)")
        if slivers:
            print(f"  erased {slivers} edge sliver(s) belonging to the "
                  f"neighbouring subject")
    return 0



def register_cells(kind, images_dir, db_url):
    """Write cells as first-class references, and retire the sheets they came from.

    The cells become reference_assets rows so everything downstream - the
    trainable count, the manifest the queue freezes, the UI grid - sees them
    without another special case. The PARENT sheet is marked not-trainable in
    the same transaction: leaving it eligible would train on both the sheet and
    its own cells, which is the original bug plus duplication.
    """
    import json
    import uuid as _uuid
    import psycopg2
    import psycopg2.extras

    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute("SELECT id, file_path FROM reference_assets "
                "WHERE kind = %s AND deleted = false AND trainable = true "
                "  AND coalesce(metrics->>'parent_id','') = ''", (kind,))
    parents = cur.fetchall()

    made = retired = 0
    for pid, path in parents:
        if not os.path.exists(path):
            continue
        try:
            img = Image.open(path)
            boxes = find_cells(img)
        except Exception as e:
            print(f"  skip {os.path.basename(path)}: {e}")
            continue
        if len(boxes) < 2:
            continue  # already a single subject - leave it exactly as it is

        stem = os.path.splitext(os.path.basename(path))[0]
        kept = 0
        for i, box in enumerate(boxes):
            cell = crop_cell(img, box)
            if cell_is_junk(cell):
                continue
            cid = _uuid.uuid4()
            out = os.path.join(images_dir, f"ref_{kind}_{cid.hex[:12]}.png")
            cell.save(out, "PNG")
            # Thumbnail alongside, named so the trainer's globs cannot match it.
            t = cell.copy()
            t.thumbnail((320, 320), Image.LANCZOS)
            t.save(os.path.join(images_dir, "thumb_" + os.path.basename(out)), "PNG")

            cur.execute(
                "INSERT INTO reference_assets "
                "  (id, kind, file_path, label, metrics, usable, why, "
                "   trainable, trainable_why) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (str(cid), kind, out, f"cell {i:03d} of {stem}",
                 json.dumps({"parent_id": str(pid), "cell_index": i,
                             "image_w": cell.width, "image_h": cell.height}),
                 None, "extracted cell - not measured",
                 True, "single subject extracted from a contact sheet"))
            made += 1
            kept += 1

        if kept:
            cur.execute(
                "UPDATE reference_assets SET trainable = false, "
                "  trainable_why = %s WHERE id = %s",
                (f"contact sheet - split into {kept} single-subject cells, "
                 f"which are trained on instead", pid))
            retired += 1

    conn.commit()
    conn.close()
    print(f"  registered {made} cell(s) from {retired} sheet(s)")
    return made, retired


if __name__ == "__main__":
    sys.exit(main())
