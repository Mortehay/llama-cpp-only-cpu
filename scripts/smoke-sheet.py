#!/usr/bin/env python3
"""Exercise the sheet composer without needing any model.

Slices an existing generated sheet back into cells, feeds them to SheetBuilder
as if they had come from the conveyor, and checks that what comes out is a
valid, evenly-dividing, palette-locked, hard-alpha sheet with a matching atlas.

The point is to have the composition and pixelation half of the pipeline under
test BEFORE the generation half exists, so that when a real turnaround lands
there is only one new thing that can be wrong.

Usage:
    python scripts/smoke-sheet.py images/sheet_3025b822691a.png
"""

import json
import os
import sys

# The sprite modules sit in two different places depending on where this runs:
# `src/sprite_generator/` in a checkout, and `/app` inside the container (the
# compose file mounts the package there and this script at /app/scripts). Python
# puts the SCRIPT's directory on sys.path, not the working directory, so /app is
# not importable by default even when it is the CWD.
_here = os.path.dirname(os.path.abspath(__file__))
for _candidate in (os.path.join(_here, "..", "src", "sprite_generator"),
                   os.path.dirname(_here)):
    if os.path.isfile(os.path.join(_candidate, "pixelate.py")):
        sys.path.insert(0, _candidate)
        break
else:
    raise SystemExit(
        "cannot find pixelate.py next to this script; looked in "
        "../src/sprite_generator and the parent directory"
    )

from PIL import Image  # noqa: E402

import pixelate  # noqa: E402
import sheet as sheet_mod  # noqa: E402


def main(argv):
    src_path = argv[0] if argv else "images/sheet_3025b822691a.png"
    out_png = argv[1] if len(argv) > 1 else "images/_smoke_sheet.png"

    src = Image.open(src_path)
    # The source has a white background, not alpha. Key it once, up front - the
    # conveyor will hand SheetBuilder already-keyed cells.
    src = pixelate.key_background(src, tolerance=24)

    cols, rows = 4, 2
    cells = [sub for _, _, sub in pixelate._slice_grid(src, cols, rows)]
    print(f"{src_path}: {src.width}x{src.height} -> {len(cells)} cells")

    # 1 action x 4 directions x 2 frames = the 8 cells available.
    b = sheet_mod.SheetBuilder(cell=(48, 64), frames=2,
                               directions=sheet_mod.DIRECTIONS_4)
    for i, cell in enumerate(cells):
        d = sheet_mod.DIRECTIONS_4[i % 4]
        f = i // 4
        b.add("walk", d, f, cell)

    assert not b.missing(), f"unexpected gaps: {b.missing()}"

    out, atlas = b.save(out_png)
    print(f"-> {out_png} {out.width}x{out.height}")
    print(json.dumps(atlas["something2"], indent=2))
    print(f"palette: {len(atlas['palette'])} colours, "
          f"rows: {[(r['action'], r['direction']) for r in atlas['rows']]}")

    # The composer promises an evenly dividing grid; something2 rejects
    # anything else outright, so assert it here rather than discovering it in
    # their admin panel.
    g = atlas["grid"]
    assert out.width % g["columns"] == 0 and out.height % g["rows"] == 0, \
        "sheet does not divide evenly into its own declared grid"
    assert out.width // g["columns"] == atlas["cell"]["w"]
    assert out.height // g["rows"] == atlas["cell"]["h"]

    # And that the gap check actually fires, since it is the guard standing
    # between a hole in a sheet and a character vanishing for one frame.
    b2 = sheet_mod.SheetBuilder(cell=(48, 64), frames=2,
                                directions=sheet_mod.DIRECTIONS_4)
    b2.add("walk", "s", 0, cells[0])
    try:
        b2.build()
    except ValueError as e:
        print(f"gap check fires as expected: {str(e)[:60]}...")
    else:
        raise AssertionError("missing cells did NOT raise")

    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
