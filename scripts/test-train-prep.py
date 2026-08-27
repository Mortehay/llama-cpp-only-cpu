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
# The round-trip test, and the fixture that matters is the THIRD one.
#
# `hd` above is flat blocks - a real grid. `painted` is a smooth ramp. Any test
# separates those. The interesting case is art that looks blocky at thumbnail
# size and is not: take the same flat blocks and add per-pixel noise. That is
# what the 103 recovered cells turned out to be (block error 10.79 against a
# hand-made control's 0.00, and 985 distinct colours in a 32x32 patch), and it
# is what fooled me into writing a whole extra signal to route them to NEAREST.
rng3 = np.random.default_rng(11)
noisy = np.repeat(np.repeat(blocks, BLOCK, axis=0), BLOCK, axis=1).astype(np.int16)
noisy += rng3.integers(-12, 13, noisy.shape)
imitation = Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8)).convert("RGBA")

check("real grid survives its own round-trip",
      m.grid_round_trip_error(hd) <= m.MAX_GRID_ROUND_TRIP_ERROR, True)
check("imitation blocks do not",
      m.grid_round_trip_error(imitation) > m.MAX_GRID_ROUND_TRIP_ERROR, True)
# The two must be far apart, not merely on opposite sides of the constant. On
# the real data the gap is 0.00 against 7.73, so a fixture that squeaks past
# would mean the fixture, not the measure, is doing the work.
check("and by a wide margin, not a squeak",
      m.grid_round_trip_error(imitation) > 4 * m.MAX_GRID_ROUND_TRIP_ERROR, True)
# The noise is invisible to the colour-count test: it still reads as
# high-colour, exactly like the flat-block fixture. That is precisely why
# colour count could not tell them apart and why a third signal seemed needed.
check("imitation is high-colour too, so colour count cannot separate them",
      imitation.convert("RGB").getcolors(m.PIXEL_ART_MAX_COLORS) is None, True)
# The behaviour that changed. NEAREST protects a grid; there is no grid here,
# so it would only magnify the noise on the 4-6x upscale to 1024.
check("imitation, enlarged -> lanczos", m.prepare_image(imitation, 512)[3],
      "lanczos")

# The two copies must agree, and this is the only thing that makes them.
#
# `grid_round_trip_error` lives in both train-lora.py and
# audit-character-refs.py, because /app/scripts cannot import the /app package
# and there is nowhere shared to put it. Duplication is tolerable; SILENT
# divergence is not - the trainer and the audit disagreeing about what pixel
# art is, each with its own passing tests, is exactly how the signal this
# replaced came to be written.
_aspec = importlib.util.spec_from_file_location(
    "audit_refs", os.path.join(_here, "audit-character-refs.py"))
audit = importlib.util.module_from_spec(_aspec)
_aspec.loader.exec_module(audit)
check("the audit's copy of the measure agrees, image by image",
      [round(audit.grid_round_trip_error(i), 6) for i in (hd, imitation, painted)],
      [round(m.grid_round_trip_error(i), 6) for i in (hd, imitation, painted)])
check("and so does the threshold",
      audit.MAX_GRID_ROUND_TRIP_ERROR, m.MAX_GRID_ROUND_TRIP_ERROR)

grid = Image.new("RGB", (32, 32))
for y in range(32):
    for x in range(32):
        grid.putpixel((x, y), (x * 8 % 256, y * 8 % 256, 0))
check("8x upscale detected", m.has_pixel_grid(grid.resize((256, 256), Image.NEAREST)), 8)
check("no grid claimed at 1:1", m.has_pixel_grid(grid), 1)


print("\nFAILURES:", ", ".join(fails) if fails else "none")
sys.exit(1 if fails else 0)
