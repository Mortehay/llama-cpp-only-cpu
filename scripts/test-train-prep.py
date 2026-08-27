#!/usr/bin/env python3
"""Tests for the parts of train-lora.py that need no GPU.

WHY THESE EXIST AND WHY THEY ARE THIS SHAPE

ADR 0006 records that the pipeline builders could not be extracted safely
because they "have no tests", and the training script was in the same position:
its only validation was that one run finished. That is exactly the condition
under which the caption bug survived - a caption ending in a filename hash
produces a perfectly normal-looking loss curve.

Every check is on a pure function, so this runs anywhere Pillow and numpy are
installed - no CUDA, no weights, no /models cache:

    python scripts/test-train-prep.py

The preparation checks assert on what the CROP actually lost, not on sizes and
offsets alone. A test that only compared dimensions would have passed against
the old centre-crop, which returned a perfectly square image with the
character's head removed.
"""

import importlib.util
import os
import sys

import numpy as np
from PIL import Image

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "train_lora", os.path.join(_here, "train-lora.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

fails: list[str] = []


def check(name, got, want):
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got!r}"
          + ("" if ok else f"  != {want!r}"))
    if not ok:
        fails.append(name)


# ---------------------------------------------------------------------------
# Captions: the storage hash must never reach the text encoder
# ---------------------------------------------------------------------------

T = "<x-style>"
check("sprite hash dropped", m.caption_for("/i/ref_sprite_004228080a22.png", T),
      "<x-style> pixel art sprite")
check("core hash dropped", m.caption_for("/i/ref_core_00e81fce7f73.png", T),
      "<x-style> pixel art sprite")
# ref_map_ was missing from the strip list entirely, so these read "ref map ...".
check("map hash dropped", m.caption_for("/i/ref_map_112233445566.png", T),
      "<x-style> pixel art sprite")
check("real label kept",
      m.caption_for("/i/ref_sprite_ab.png", T, "dwarf ranger, walking south"),
      "<x-style> pixel art sprite, dwarf ranger, walking south")
# split-sheets writes `cell_<kind>_<kind>_<hash>_<index>`. A prefix list cannot
# strip that, and the first version of the fix let all 103 through with the
# hash intact - which is the original bug, on new filenames.
check("split-sheets cell name stripped",
      m.caption_for("/i/cell_sprite_sprite_291203136719_000.png", T),
      "<x-style> pixel art sprite")
check("custom body",
      m.caption_for("/i/ref_core_00e81fce7f73.png", T, None, "isometric character"),
      "<x-style> isometric character")
# Pure-hex WORDS are labels a person could type; the digit test keeps them.
check("hex word kept as label", m.caption_for("/i/ref_core_beef.png", T),
      "<x-style> pixel art sprite, beef")
check("short hex hash dropped", m.caption_for("/i/ref_core_00e8.png", T),
      "<x-style> pixel art sprite")


# ---------------------------------------------------------------------------
# Preparation: pad keeps the subject, crop is what threw it away
# ---------------------------------------------------------------------------

tall = Image.new("RGBA", (200, 400), (0, 0, 0, 0))
tall.paste((255, 0, 0, 255), (0, 0, 200, 20))        # red band at the very top
tall.paste((0, 255, 0, 255), (0, 380, 200, 400))     # green at the very bottom

img, orig, crop, _ = m.prepare_image(tall, 256, fit="pad")
tones = {c[1] for c in img.convert("RGB").getcolors(99999)}
check("pad: true original size reported", orig, (200, 400))
check("pad: offset is negative (a frame, not a crop)", crop, (-100, 0))
check("pad: top band survives",
      any(r > 220 and g < 40 for r, g, b in tones), True)
check("pad: bottom band survives",
      any(g > 220 and r < 40 for r, g, b in tones), True)
check("pad: square at the target resolution", img.size, (256, 256))

img2, _, crop2, _ = m.prepare_image(tall, 256, fit="crop")
tones2 = {c[1] for c in img2.convert("RGB").getcolors(99999)}
check("crop: offset is positive", crop2, (0, 100))
check("crop: top band LOST - this is the old default",
      any(r > 220 and g < 40 for r, g, b in tones2), False)


# ---------------------------------------------------------------------------
# Resampling: hard-edged art must not be enlarged with a smooth filter
# ---------------------------------------------------------------------------

tiny = Image.new("RGBA", (32, 32), (0, 0, 0, 255))
tiny.paste((255, 255, 255, 255), (0, 0, 16, 16))
check("tiny pixel art enlarged -> nearest",
      m.prepare_image(tiny, 512)[3], "nearest")
check("explicit override respected",
      m.prepare_image(tiny, 512, resample="lanczos")[3], "lanczos")

rng = np.random.default_rng(0)
noise = Image.fromarray(
    rng.integers(0, 255, (800, 800, 3), dtype=np.uint8)).convert("RGBA")
check("photographic and downscaled -> lanczos",
      m.prepare_image(noise, 512)[3], "lanczos")

# HD pixel art: hard-edged but NOT palette-locked and with no pixel grid, so
# the colour-count and grid tests both miss it. This is the recovered Mesgard
# set's shape, and routing it to LANCZOS defeated the whole change.
#
# It has to be built carefully. A first attempt used 1px-wide bands stepping by
# a few levels each - which is a fine GRADIENT, not steps, and it passed only
# because the palette test caught it. Flat blocks with large jumps between
# them, and more than PIXEL_ART_MAX_COLORS of them, is the real shape.
BLOCK, GRID = 5, 40                       # 1600 flat blocks, 200x200 px
rng2 = np.random.default_rng(7)
blocks = rng2.integers(0, 256, (GRID, GRID, 3), dtype=np.uint8)
hd = Image.fromarray(
    np.repeat(np.repeat(blocks, BLOCK, axis=0), BLOCK, axis=1)).convert("RGBA")
check("fixture really is high-colour",
      hd.convert("RGB").getcolors(m.PIXEL_ART_MAX_COLORS) is None, True)
check("hard-edged, high-colour, enlarged -> nearest",
      m.prepare_image(hd, 512)[3], "nearest")

ramp = np.zeros((200, 200, 3), dtype=np.uint8)
yy, xx = np.mgrid[0:200, 0:200]           # smooth gradient: painted
ramp[..., 0] = (xx * 255 // 199).astype(np.uint8)
ramp[..., 1] = (yy * 255 // 199).astype(np.uint8)
ramp[..., 2] = 128
painted = Image.fromarray(ramp).convert("RGBA")
check("smooth gradient, enlarged -> lanczos",
      m.prepare_image(painted, 512)[3], "lanczos")
soft_hd, soft_painted = m.edge_softness(hd), m.edge_softness(painted)
check("edge_softness separates stepped from ramped",
      soft_hd is not None and soft_painted is not None
      and soft_hd < m.MAX_PIXEL_ART_SOFTNESS <= soft_painted, True)

grid = Image.new("RGB", (32, 32))
for y in range(32):
    for x in range(32):
        grid.putpixel((x, y), (x * 8 % 256, y * 8 % 256, 0))
check("8x upscale detected", m.has_pixel_grid(grid.resize((256, 256), Image.NEAREST)), 8)
check("no grid claimed at 1:1", m.has_pixel_grid(grid), 1)


print("\nFAILURES:", ", ".join(fails) if fails else "none")
sys.exit(1 if fails else 0)
