#!/usr/bin/env python3
"""Does a sheet still equal its per-frame PNGs? Synthetic cells, no GPU.

    check-sheet-assembly.py

Four properties, all of which the redraw endpoint depends on and none of
which a sheet shows you by looking at it:

  1. sheet.png is byte-for-byte the frame files laid out in the grid.
  2. a mirrored row is the EXACT horizontal flip of the row it came from.
  3. with the scale pinned, replacing one frame changes only that cell
     (and its mirror).
  4. a replacement wider than the sheet's widest cell forces a refit, and the
     atlas says `refit: true` rather than quietly moving all 336 sprites.

(2) is the one that caught a real defect. Flipping the 512px render and
fitting it separately does NOT produce a mirror: `pixelate` crops to the
alpha bbox and centres with `(cell_w - width) // 2`, so an odd remainder
rounds the sprite one pixel to the left in BOTH cells. Every frame of the
flipped row differed from the mirror of its source, which in game is a sprite
that jumps a pixel sideways when the character turns around. The fix is to
flip the FITTED cell, which is a pure transpose.

(4) exists because the honest answer to "does a redraw change only that cell"
is "usually". The sheet-wide scale is the minimum fit over all cells, so a
repaired frame that is wider than anything in the original build cannot be
honoured at the pinned scale without `Image.paste` clipping it. The sheet is
refitted instead and the atlas records it.

Runs in a few seconds against stdlib + PIL + numpy. No model, no work
directory, no GPU - so it is cheap enough to run after any change to
sheet.py or pixelate.py.
"""

import os
import sys
import tempfile

_here = os.path.dirname(os.path.abspath(__file__))
for _c in (os.path.join(_here, "..", "src", "sprite_generator"),
           os.path.dirname(_here), "/app"):
    if os.path.isfile(os.path.join(_c, "sheet.py")):
        sys.path.insert(0, _c)
        break

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import sheet as sheet_mod  # noqa: E402

CW, CH = 48, 64
FRAMES = 3
DIRS = ["s", "e", "w"]
ACTS = ["idle", "walk"]
TARGET = ("walk", "e", 1)


def fake_cell(seed, width=110, extra=None):
    """An asymmetric blob, so a flip is detectable.

    `width` drives the bbox width, which is what sets the sheet-wide scale.
    """
    rng = np.random.default_rng(seed)
    a = np.zeros((200, 200, 4), np.uint8)
    a[40:190, 50:50 + width] = [60, 90, 140, 255]      # body
    a[20:60, 60:100] = [200, 170, 140, 255]            # head
    a[70:95, 110:150] = [220, 60, 60, 255]             # arm, to the RIGHT
    a[150:190, 40:70] = [int(rng.integers(80, 200)), 120, 60, 255]
    if extra is not None:
        a[95:130, extra[0]:extra[1]] = [240, 240, 60, 255]
    return Image.fromarray(a, "RGBA")


def build(tmp, swap=None, pin=None):
    b = sheet_mod.SheetBuilder(cell=(CW, CH), frames=FRAMES, directions=DIRS,
                               n_colors=16, mirrored={"w": "e"},
                               pinned_scale=pin)
    for act in ACTS:
        for d in ["s", "e"]:                  # `w` is produced by mirroring
            for f in range(FRAMES):
                key = (act, d, f)
                img = (swap[1] if swap and swap[0] == key
                       else fake_cell(abs(hash(key)) % 9999))
                b.add(act, d, f, img)
    png = os.path.join(tmp, "sheet.png")
    _, atlas = b.save(png, concept=fake_cell(1), allow_missing=True,
                      frames_root=os.path.join(tmp, "frames"))
    return png, atlas


def arr(p):
    return np.asarray(Image.open(p).convert("RGBA"))


def differing_cells(p1, a1, p2):
    s1, s2 = arr(p1), arr(p2)
    out = []
    for row in a1["rows"]:
        for f in range(FRAMES):
            y, x = row["row"] * CH, f * CW
            if not np.array_equal(s1[y:y + CH, x:x + CW],
                                  s2[y:y + CH, x:x + CW]):
                out.append(f"{row['action']}_{row['direction']}/{f}")
    return sorted(out)


def main():
    fails = 0

    with tempfile.TemporaryDirectory() as tmp:
        png, atlas = build(tmp)
        sheet = arr(png)
        print(f"sheet {sheet.shape[1]}x{sheet.shape[0]}, "
              f"{atlas['grid']['rows']} rows x {atlas['grid']['columns']} "
              f"cols, scale {atlas['scale']}")

        bad = []
        for row in atlas["rows"]:
            for f in range(FRAMES):
                fp = os.path.join(tmp, "frames", row["frames_dir"], f"{f}.png")
                if not os.path.isfile(fp):
                    bad.append(f"missing {row['frames_dir']}/{f}.png")
                    continue
                y, x = row["row"] * CH, f * CW
                if not np.array_equal(sheet[y:y + CH, x:x + CW], arr(fp)):
                    bad.append(f"{row['frames_dir']}/{f} differs from the grid")
        print(f"1. sheet == frames in grid: "
              f"{'PASS' if not bad else 'FAIL ' + str(bad[:4])}")
        fails += bool(bad)

        mism = []
        for row in atlas["rows"]:
            src = row.get("mirrored_from")
            if not src:
                continue
            srow = next(r for r in atlas["rows"]
                        if r["action"] == row["action"]
                        and r["direction"] == src)
            for f in range(FRAMES):
                a = sheet[row["row"] * CH:(row["row"] + 1) * CH,
                          f * CW:(f + 1) * CW]
                s = sheet[srow["row"] * CH:(srow["row"] + 1) * CH,
                          f * CW:(f + 1) * CW]
                if not np.array_equal(a, s[:, ::-1]):
                    mism.append(f"{row['action']} {row['direction']} f{f}")
        print(f"2. mirrored rows are exact flips: "
              f"{'PASS' if not mism else 'FAIL ' + str(mism[:4])}")
        fails += bool(mism)

    with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
        p1, a1 = build(t1)
        p2, a2 = build(t2, swap=(TARGET, fake_cell(4242, 110, (60, 100))),
                       pin=a1["scale"])
        changed = differing_cells(p1, a1, p2)
        ok = changed == ["walk_e/1", "walk_w/1"] and not a2.get("refit")
        print(f"3. in-width redraw, scale pinned -> only that cell: "
              f"{'PASS' if ok else 'FAIL'}   changed {changed}")
        fails += (not ok)

    with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
        p1, a1 = build(t1)
        p2, a2 = build(t2, swap=(TARGET, fake_cell(4242, 145, (10, 50))),
                       pin=a1["scale"])
        changed = differing_cells(p1, a1, p2)
        ok = (a2.get("refit") is True and a2["scale"] < a1["scale"]
              and len(changed) > 2)
        print(f"4. wider redraw -> refit reported, not hidden: "
              f"{'PASS' if ok else 'FAIL'}   refit={a2.get('refit')}, "
              f"scale {a1['scale']} -> {a2['scale']}, {len(changed)} cells moved")
        fails += (not ok)

    # 5. A sheet can ask for `w` WITHOUT asking for `e`. The source is
    # rendered, fitted and flipped, but never becomes a row of its own.
    with tempfile.TemporaryDirectory() as tmp:
        b = sheet_mod.SheetBuilder(cell=(CW, CH), frames=FRAMES,
                                   directions=["w"], n_colors=16,
                                   mirrored={"w": "e"})
        for f in range(FRAMES):
            b.add("walk", "e", f, fake_cell(100 + f))
        png = os.path.join(tmp, "sheet.png")
        _, atlas = b.save(png, concept=fake_cell(1), allow_missing=True,
                          frames_root=os.path.join(tmp, "frames"))
        sheet = arr(png)
        rows = [(r["action"], r["direction"]) for r in atlas["rows"]]
        opaque = [bool((sheet[0:CH, f * CW:(f + 1) * CW, 3] > 0).any())
                  for f in range(FRAMES)]
        emitted = sorted(os.listdir(os.path.join(tmp, "frames")))
        ok = (rows == [("walk", "w")] and all(opaque)
              and emitted == ["walk_w"])
        print(f"5. mirror source need not be a row: "
              f"{'PASS' if ok else 'FAIL'}   rows={rows}, "
              f"frames dirs={emitted}, cells drawn={opaque}")
        fails += (not ok)

    print(f"\n{'ALL PASS' if not fails else str(fails) + ' CHECK(S) FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
