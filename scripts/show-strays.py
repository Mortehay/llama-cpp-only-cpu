"""Show exactly which pixels `_isolate_largest_sprite` would delete.

The recovery deliberately does NOT pass keep_largest=True on a reference: at
generation time dropping every blob but the biggest is right, and on a reference
a detached part may BE the subject - the findings note records
`ref_core_ca0070408096`, whose "strays" are the detached drips of a dripping
beast and are the art.

So the question "is this stray litter or art" cannot be answered by a count. It
is answered by looking. This paints the kept blob normally and the doomed pixels
in solid red, so the two are separable at a glance.

Usage:
    python3 scripts/show-strays.py <file-list> <src-dir> <out.png>
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

names = [l.strip() for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
src_dir, out_path = sys.argv[2], sys.argv[3]

cell = int(sys.argv[4]) if len(sys.argv) > 4 else 340
cols = int(sys.argv[5]) if len(sys.argv) > 5 else 3
rows = (len(names) + cols - 1) // cols
sheet = Image.new("RGB", (cols * cell, rows * (cell + 18)), (255, 0, 255))
d = ImageDraw.Draw(sheet)

for i, n in enumerate(names):
    arr = np.array(Image.open(os.path.join(src_dir, n)).convert("RGBA"))
    opaque = arr[:, :, 3] > 0
    labels, count = ndimage.label(opaque, structure=np.ones((3, 3), dtype=bool))
    dropped_px = 0
    if count > 1:
        sizes = ndimage.sum(opaque, labels, range(1, count + 1))
        keep = int(np.argmax(sizes)) + 1
        doomed = (labels != keep) & opaque
        dropped_px = int(doomed.sum())
        arr[doomed] = (255, 0, 0, 255)

    im = Image.fromarray(arr, "RGBA")
    a = np.asarray(im)[..., 3]
    ys, xs = np.nonzero(a >= 128)
    im = im.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1,
                  int(ys.max()) + 1))
    # Scale UP as well as down. thumbnail() only shrinks, so a 60px subject in
    # a 336px frame stayed 60px on the sheet and could not be judged at all -
    # which is the whole point of the sheet.
    k = min((cell - 8) / im.width, (cell - 8) / im.height)
    im = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))),
                   Image.Resampling.NEAREST)
    x = (i % cols) * cell + (cell - im.width) // 2
    y = (i // cols) * (cell + 18)
    sheet.paste(im, (x, y), im)
    d.text(((i % cols) * cell + 2, y + cell + 2),
           "%s  %d blobs, %d px red" % (n[9:-4], count, dropped_px),
           fill=(0, 0, 0))

sheet.save(out_path)
print(len(names), "tiles ->", out_path)
