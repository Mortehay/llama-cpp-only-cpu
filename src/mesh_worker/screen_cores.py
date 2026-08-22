"""Which core images are clean enough to feed TripoSR, and which are humanoid?

Screening beats eyeballing here: an unkeyed background or a baked-in scenery
frame gets reconstructed into the mesh as geometry, and that corrupts the rig
test downstream. Two cheap signals:

  corner_alpha  - mean alpha of the four corner patches. A properly keyed sprite
                  is ~0. Anything above a few counts means a halo, a frame or a
                  solid backdrop survived.
  aspect        - height/width of the opaque bounding box. Bipeds come out tall
                  (>1.4 here); trees and splayed shapes come out square or wide.

Neither is a verdict, they just order the candidates worth opening.
"""
import pathlib
import sys

from PIL import Image
import numpy as np


def stats(path):
    im = Image.open(path).convert("RGBA")
    a = np.asarray(im)[..., 3].astype(np.float32)
    h, w = a.shape
    k = max(4, min(h, w) // 16)
    corners = np.concatenate([a[:k, :k].ravel(), a[:k, -k:].ravel(),
                              a[-k:, :k].ravel(), a[-k:, -k:].ravel()])
    ys, xs = np.nonzero(a > 16)
    if len(ys) == 0:
        return None
    bh, bw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    return {
        "corner_alpha": float(corners.mean()),
        "aspect": float(bh) / float(bw),
        "coverage": float((a > 16).mean()),
    }


def main():
    d = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/app/images")
    rows = []
    for p in sorted(d.glob("core_*.png")):
        s = stats(p)
        if s:
            rows.append((p.name, s))
    rows.sort(key=lambda r: (r[1]["corner_alpha"], -r[1]["aspect"]))
    print(f"{'file':30s} {'corner_a':>9s} {'aspect':>7s} {'cover':>7s}  guess")
    for name, s in rows:
        guess = ("clean+tall" if s["corner_alpha"] < 2 and s["aspect"] > 1.4
                 else "clean" if s["corner_alpha"] < 2
                 else "DIRTY")
        print(f"{name:30s} {s['corner_alpha']:9.2f} {s['aspect']:7.2f} "
              f"{s['coverage']:7.3f}  {guess}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
