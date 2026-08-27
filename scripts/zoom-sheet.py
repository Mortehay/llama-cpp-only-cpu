"""Throwaway: magnify a named list of survivors so they can be judged by eye."""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

names = [l.strip() for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
dirs = sys.argv[2].split(",")
out_path = sys.argv[3]

cell, cols = 340, 3
rows = (len(names) + cols - 1) // cols
out = Image.new("RGB", (cols * cell, rows * (cell + 16)), (255, 0, 255))
d = ImageDraw.Draw(out)
for i, n in enumerate(names):
    p = next(q for q in (os.path.join(x, n) for x in dirs) if os.path.exists(q))
    im = Image.open(p).convert("RGBA")
    a = np.asarray(im)[..., 3]
    ys, xs = np.nonzero(a >= 128)
    im = im.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1,
                  int(ys.max()) + 1))
    im.thumbnail((cell - 8, cell - 8), Image.Resampling.NEAREST)
    x = (i % cols) * cell + (cell - im.width) // 2
    y = (i // cols) * (cell + 16)
    out.paste(im, (x, y), im)
    d.text(((i % cols) * cell + 2, y + cell + 2), n[9:-4], fill=(0, 0, 0))
out.save(out_path)
print(len(names), "tiles ->", out_path)
