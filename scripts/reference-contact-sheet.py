#!/usr/bin/env python3
"""Lay every reference of one kind out on paginated sheets, for a human pass.

WHY THIS EXISTS

The character references are not all character art. Splitting them into cells
showed roughly a quarter of the result was TEXT - letters cropped out of titles,
and UI cards where the words are the subject ("Golden Order Fundamentalism",
"ELECTRO / ECHO"). Training on that would swap a lattice artefact for a text
artefact: the same failure in a new shape, invisible until a 308-image batch
finished.

The obvious move is a text detector. Deliberately not written: two detectors
have already been thrown away on this exact dataset for being confidently wrong,
and one that misses puts letterforms into the catalogue silently. Ten minutes of
human eyes beats a third heuristic.

So this makes looking cheap. Each thumbnail is labelled with the short id the
answer comes back as, so a verdict is a list of ids rather than "third row,
sixth along", and the order is stable (oldest first) so the grid does not
reshuffle between passes.

Usage:
    python reference-contact-sheet.py --kind core
    python reference-contact-sheet.py --kind sprite --per-page 80
"""

import argparse
import os
import sys

import psycopg2
from PIL import Image, ImageDraw, ImageFont

CELL = 190          # thumbnail box
LABEL_H = 22        # strip under each thumbnail
PAD = 6
BG = (24, 24, 38)
FG = (226, 224, 240)
MUTED = (136, 132, 168)
SHEET_TAG = (252, 211, 77)   # amber: this one is a multi-subject sheet


def _font(size: int):
    """A legible font, whatever this image happens to have.

    The container ships no TrueType fonts, so the default bitmap face is the
    realistic option; Pillow >= 10.1 can at least scale it.
    """
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def fetch(kind: str, db_url: str):
    with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, file_path, coalesce(label,''), trainable, "
            "       coalesce(metrics->>'parent_id','') "
            "FROM reference_assets "
            "WHERE kind = %s AND deleted = false "
            "ORDER BY created_at", (kind,))
        return cur.fetchall()


def thumb_for(path: str) -> str:
    d, name = os.path.split(path)
    t = os.path.join(d, "thumb_" + name)
    return t if os.path.exists(t) else path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", default="core")
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--per-page", type=int, default=96)
    p.add_argument("--out", default="/app/images")
    p.add_argument("--include-cells", action="store_true",
                   help="also show cells extracted from sheets (off by "
                        "default: the point is to judge the SOURCES)")
    a = p.parse_args()

    db = os.environ.get("DB_URL")
    if not db:
        sys.exit("DB_URL unset - run this inside the container")

    rows = fetch(a.kind, db)
    if not a.include_cells:
        rows = [r for r in rows if not r[4]]     # drop extracted cells
    if not rows:
        sys.exit(f"no {a.kind} references")

    # Note which sources are multi-subject, since that is information the
    # reviewer would otherwise have to infer by eye.
    sys.path.insert(0, "/app/scripts")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ss", "/app/scripts/split-sheets.py")
    ss = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ss)

    label_f, head_f = _font(13), _font(20)
    pages, made = (len(rows) + a.per_page - 1) // a.per_page, []

    for pg in range(pages):
        chunk = rows[pg * a.per_page:(pg + 1) * a.per_page]
        cols = a.cols
        nrows = (len(chunk) + cols - 1) // cols
        W = cols * (CELL + PAD) + PAD
        H = nrows * (CELL + LABEL_H + PAD) + PAD + 34
        sheet = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(sheet)
        d.text((PAD, 8),
               f"{a.kind} references - page {pg + 1}/{pages} - "
               f"{len(rows)} total - amber id = multi-subject sheet",
               font=head_f, fill=FG)

        for i, (rid, path, label, trainable, _parent) in enumerate(chunk):
            cx = PAD + (i % cols) * (CELL + PAD)
            cy = 34 + PAD + (i // cols) * (CELL + LABEL_H + PAD)
            d.rectangle([cx, cy, cx + CELL, cy + CELL], fill=(16, 16, 26))

            n_cells = 0
            try:
                im = Image.open(thumb_for(path)).convert("RGBA")
                try:
                    n_cells = len(ss.find_cells(Image.open(path)))
                except Exception:
                    n_cells = 0
                im.thumbnail((CELL - 8, CELL - 8), Image.LANCZOS)
                bg = Image.new("RGB", im.size, (16, 16, 26))
                bg.paste(im, (0, 0), im)
                sheet.paste(bg, (cx + (CELL - im.width) // 2,
                                 cy + (CELL - im.height) // 2))
            except Exception:
                d.text((cx + 8, cy + 8), "unreadable", font=label_f, fill=MUTED)

            short = str(rid)[:8]
            is_sheet = n_cells >= 2
            d.text((cx + 2, cy + CELL + 3),
                   f"{short}  {'SHEET x' + str(n_cells) if is_sheet else 'single'}",
                   font=label_f, fill=SHEET_TAG if is_sheet else MUTED)
            if not trainable:
                d.text((cx + 2, cy + CELL + 12), "excluded",
                       font=label_f, fill=MUTED)

        out = os.path.join(a.out, f"_refsheet_{a.kind}_{pg + 1}.png")
        sheet.save(out)
        made.append(out)
        print(f"  page {pg + 1}/{pages}: {len(chunk)} refs -> {out}")

    print(f"{len(rows)} {a.kind} references across {pages} page(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
