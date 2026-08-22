"""Zero alpha below a HAND-PICKED row. A diagnostic, not a pipeline step.

strip_ground_patch cannot see charB's ground patch (see its docstring: the
figure's raised arm inflates the reference width, and its boots sit just above
the walk-up threshold, so the patch is missed and lowering the threshold
amputates the feet instead). A gradient detector failed too - the patch ramps in
over ~12 rows rather than stepping.

That leaves a confounded experiment: charB's skeleton came out as a 6-bone chain
while carrying BOTH a fused ground disc and a raised arm, so neither can be
blamed. This exists purely to remove one of the two variables by cutting at a
row read off the width profile by hand:

    y=462  count 206   <- boots
    y=466  count 207   <- boots
    y=470  count 285   <- patch begins

Hand-picking a row does not generalise and is not meant to. It buys ONE
controlled comparison; the real detector is still an open problem.
"""
import argparse

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("row", type=int, help="first row to clear (inclusive)")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    arr = np.array(Image.open(a.image).convert("RGBA"))
    before = int((arr[:, :, 3] > 0).sum())
    arr[a.row:, :, 3] = 0
    after = int((arr[:, :, 3] > 0).sum())
    Image.fromarray(arr).save(a.out)
    print(f"cleared rows {a.row}.. : {before - after} px removed "
          f"({before} -> {after})")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    raise SystemExit(main())
