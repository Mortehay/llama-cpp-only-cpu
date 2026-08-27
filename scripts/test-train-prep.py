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

# The copy that has an ORIGINAL, checked against the original itself.
#
# The check above compares two copies to each other. That catches them drifting
# apart and misses them drifting together, which is the failure mode that
# matters when there is a source of truth - and for `pixel_scale` there is one:
# `measure.pixel_scale`, which the audit's docstring says it mirrors and which
# nothing verified. A sibling session shipped exactly this and its copy had
# been transcribed from a DIFFERENT function entirely, flipping 25 verdicts.
#
# measure.py cannot be imported here - it pulls the `concept` package - so the
# original's source is extracted and executed on its own. That is the point:
# the test reads the original rather than restating what it is believed to do,
# so a change there fails HERE instead of quietly making the copy wrong.
#
# TWO LAYOUTS, because this suite runs in both and the first version only
# worked in one. On the host the repo is laid out as `<repo>/scripts` and
# `<repo>/src/sprite_generator`. In the worker container `src/sprite_generator`
# is mounted AS `/app` and the scripts at `/app/scripts`, so the same relative
# walk lands on `/app/src/sprite_generator/measure.py`, which does not exist.
#
# It passed locally and crashed under `make test-train-prep`, which is the only
# way it is ever run by anyone but me. Found by executing it in the container
# rather than by reading it.
_measure_candidates = [
    os.path.join(os.path.dirname(_here), "src", "sprite_generator", "measure.py"),
    os.path.join(os.path.dirname(_here), "measure.py"),
]
_measure_path = next((p for p in _measure_candidates if os.path.isfile(p)), None)
if _measure_path is None:
    sys.exit("cannot find measure.py; looked in:\n  "
             + "\n  ".join(_measure_candidates))
_measure_src = open(_measure_path, encoding="utf-8").read()
_start = _measure_src.index("def pixel_scale(")
_end = _measure_src.index("\ndef ", _start)
_ns: dict = {"np": np}
exec(compile(_measure_src[_start:_end], "measure.pixel_scale", "exec"), _ns)
original = _ns["pixel_scale"]

# Fixtures spanning what the two can disagree about: a real 8x grid, art at
# 1:1, and a smooth image with no honest answer. Built here rather than reusing
# the `grid` fixture below, so this block does not depend on statement order in
# a file that is appended to.
_base = Image.new("RGB", (32, 32))
for _y in range(32):
    for _x in range(32):
        _base.putpixel((_x, _y), (_x * 8 % 256, _y * 8 % 256, 0))

# A fixture where the 0.98 THRESHOLD is load-bearing, which the three above are
# not. Added because the first version of this guard was decoration: mutating
# the copy's threshold from 0.98 to 0.50 left every check green. Clean fixtures
# give the same answer under any threshold, so they cannot detect a change to
# it - a guard that only sees unambiguous cases proves nothing about the rule.
#
# Vertical bands only, so `coords` is the x boundaries alone. Ten fall on a
# multiple of 16, five more only on a multiple of 8:
#     s=16 explains 10/15 = 0.67   s=8 explains 15/15 = 1.00
# At 0.98 the answer is 8. At anything below 0.67 it is 16.
_amb = Image.new("RGB", (256, 64), (20, 20, 20))
_cuts = sorted([16 * i for i in range(1, 11)] + [8, 24, 40, 56, 72])
_px, _on = _amb.load(), False
for _x in range(256):
    if _x in _cuts:
        _on = not _on
    for _y in range(64):
        _px[_x, _y] = (210, 40, 90) if _on else (20, 20, 20)

_probe = [("8x upscale", _base.resize((256, 256), Image.NEAREST)),
          ("1:1 art", _base),
          ("no grid at all", painted),
          ("threshold is load-bearing", _amb)]
for _label, _img in _probe:
    _arr = np.asarray(_img.convert("RGBA"))
    check(f"audit mirrors measure.pixel_scale: {_label}",
          audit.pixel_scale(_arr), original(_arr)["scale"])

# The fusion detector, on a sheet built to trigger it.
#
# `count_subjects` closes the mask before labelling, which bridges the gutters
# of a packed sheet and can report a hundred sprites as one subject. Over the
# 483 references here it never fires on anything that would otherwise be KEPT -
# so the bug is real, dormant, and invisible. A detector for it that has never
# been seen to fire is the same object as the unfailable guard above, so it
# gets a fixture that makes it fire.
#
# The items are RINGS, not filled squares, and that is load-bearing rather
# than decorative. A filled grid cannot reproduce this: `foreground_mask`
# refuses anything over 75% coverage, so filled items need wide gutters to stay
# under it - and wide gutters are exactly what the closing does not bridge. The
# first version of this fixture was filled squares, measured 77% foreground,
# and came back "unverifiable" instead of fused. It agreed with the check while
# testing nothing.
#
# Real sprites are outlines with hollow interiors, so their coverage is low
# while their gutters stay narrow. Rings reproduce that; solid blocks do not.
# 64 rings on a 64px pitch, 4px apart, in a 512px frame: the closing radius
# there is 2, so a 4px gutter bridges and 64 components become one or two.
_sheet = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
for _r in range(8):
    for _c in range(8):
        _ox, _oy = _c * 64 + 2, _r * 64 + 2
        for _dy in range(60):
            for _dx in range(60):
                if 5 <= _dx < 55 and 5 <= _dy < 55:
                    continue                      # hollow centre
                _sheet.putpixel((_ox + _dx, _oy + _dy), (200, 60, 60, 255))
_subjects, _conf, _ = audit.count_subjects(_sheet)
check("packed sheet is reported as few subjects", _subjects <= 2, True)
check("and the fusion is DECLARED rather than silent",
      _conf.startswith("fused-from-"), True)

# The negative, which is what stops the flag being noise: one figure whose
# parts are separated by a hairline is exactly what the closing exists for, and
# must NOT be flagged. Without this the detector could fire on everything and
# still pass the check above.
_fig = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
for _dy in range(120):
    for _dx in range(60):
        _fig.putpixel((70 + _dx, 40 + _dy), (60, 120, 200, 255))
for _dy in range(30):                       # a limb one pixel clear of it
    for _dx in range(20):
        _fig.putpixel((131 + _dx, 60 + _dy), (60, 120, 200, 255))
_s2, _c2, _ = audit.count_subjects(_fig)
check("one figure with a detached part is NOT flagged as fused",
      _c2.startswith("fused-from-"), False)

# The one place they deliberately differ, asserted so it stays deliberate.
# `measure` returns scale=None for a flat fill - "no honest answer" - while the
# audit returns 1, because its callers branch on `scale == 1` and None would
# read as a grid. Documented divergence, not drift.
_flat = np.asarray(Image.new("RGBA", (64, 64), (7, 9, 11, 255)))
check("flat fill: audit says 1 where measure says None",
      (audit.pixel_scale(_flat), original(_flat)["scale"]), (1, None))

grid = Image.new("RGB", (32, 32))
for y in range(32):
    for x in range(32):
        grid.putpixel((x, y), (x * 8 % 256, y * 8 % 256, 0))
check("8x upscale detected", m.has_pixel_grid(grid.resize((256, 256), Image.NEAREST)), 8)
check("no grid claimed at 1:1", m.has_pixel_grid(grid), 1)


print("\nFAILURES:", ", ".join(fails) if fails else "none")
sys.exit(1 if fails else 0)
