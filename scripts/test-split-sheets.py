#!/usr/bin/env python3
"""Tests for split-sheets.py's cell cleanup. No GPU, no weights, no database.

    python scripts/test-split-sheets.py

WHY THIS EXISTS

`drop_edge_slivers` EDITS PIXELS in training images, and it was written and
applied to 103 recovered character cells in one go. A function that silently
erases part of a dataset is the worst possible thing to leave untested: the
failure mode is not a crash, it is a slightly worse adapter weeks later, which
is precisely the class of bug this whole audit exists to stop.

Both of its conditions are load-bearing and each has a test that fails if the
other is dropped:

  * relative-only would erase a dwarf's wolf companion - a second large subject
    that belongs in the cell;
  * edge-only would erase a subject's own detached parts - a spark, a dropped
    item, a floating staff-tip that sits inside the crop.

That claim was CHECKED rather than asserted. Each condition was mutated out of
`drop_edge_slivers` in a scratch copy and the suite re-run: removing the size
test reddens "large companion kept" and "just above the ratio is kept";
removing the edge test reddens "interior speck kept" and the multi-sliver
count. A test that passes against a broken function is worse than no test,
because it is evidence that something has been verified when nothing has.

The real defect it fixes: `find_cells` returns a bounding BOX, and characters
are not box-shaped. A wizard's staff leans up and to the left, overhanging the
box of the character beside him, so a cell is genuinely single-subject by
component analysis AND contains a fragment of somebody else.
"""

import importlib.util
import os
import sys

import numpy as np
from PIL import Image

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "split_sheets", os.path.join(_here, "split-sheets.py"))
ss = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ss)

fails: list[str] = []


def check(name, got, want):
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got!r}"
          + ("" if ok else f"  != {want!r}"))
    if not ok:
        fails.append(name)


def cell(*blobs, size=(200, 300)):
    """RGBA canvas, transparent, with opaque rectangles painted in."""
    a = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    for (x0, y0, x1, y1) in blobs:
        a[y0:y1, x0:x1] = (200, 120, 80, 255)
    return Image.fromarray(a)


def opaque_count(img):
    return int((np.asarray(img.convert("RGBA"))[..., 3] >= 128).sum())


SUBJECT = (60, 40, 140, 260)          # the character: 80x220, the big one


# --- the defect it was written for -----------------------------------------

img = cell(SUBJECT, (0, 0, 10, 14))   # neighbour's staff tip, touching x=0
out, removed = ss.drop_edge_slivers(img)
check("edge sliver erased", removed, 1)
check("only the sliver went",
      opaque_count(img) - opaque_count(out), 10 * 14)

# --- condition 1: relative size. A companion must survive -------------------

# A wolf beside the ranger: large, and touching the crop edge as it happens.
img = cell(SUBJECT, (0, 150, 55, 250))
out, removed = ss.drop_edge_slivers(img)
check("large companion kept even though it touches the edge", removed, 0)
check("companion pixels untouched", opaque_count(out), opaque_count(img))

# --- condition 2: touching the edge. Own detached parts must survive --------

# A spark floating INSIDE the cell, small, clear of every border.
img = cell(SUBJECT, (150, 100, 162, 112))
out, removed = ss.drop_edge_slivers(img)
check("interior speck kept", removed, 0)
check("speck pixels untouched", opaque_count(out), opaque_count(img))

# --- several at once, and the other three borders ---------------------------

img = cell(SUBJECT,
           (0, 0, 8, 8),               # left/top corner
           (192, 280, 200, 300),       # right/bottom corner
           (95, 0, 105, 6),            # top edge
           (150, 100, 160, 110))       # interior - must survive
out, removed = ss.drop_edge_slivers(img)
check("three edge slivers erased, interior kept", removed, 3)
check("interior speck still there",
      opaque_count(img) - opaque_count(out), 8 * 8 + 8 * 20 + 10 * 6)

# --- things it must not touch ----------------------------------------------

img = cell(SUBJECT)
out, removed = ss.drop_edge_slivers(img)
check("single subject untouched", removed, 0)
check("same object returned when nothing changed", out is img, True)

blank = Image.fromarray(np.zeros((80, 80, 4), dtype=np.uint8))
out, removed = ss.drop_edge_slivers(blank)
check("fully transparent is a no-op", removed, 0)

# A cell whose subject is exactly at the threshold must be KEPT, not erased.
# 12% of the subject's 17,600px is 2,112; this blob is 2,400 and touches x=0.
img = cell(SUBJECT, (0, 0, 40, 60))
out, removed = ss.drop_edge_slivers(img)
check("just above the ratio is kept", removed, 0)


print("\nFAILURES:", ", ".join(fails) if fails else "none")
sys.exit(1 if fails else 0)
