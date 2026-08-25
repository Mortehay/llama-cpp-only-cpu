#!/usr/bin/env python3
"""Rank images by whether they can serve as a CHARACTER concept.

The measurements and thresholds live in `sprite_generator/concept.py`, which the
job runner also imports - a file named `check-concept.py` cannot be imported, and
two copies of a rule is how they drift. This is the operator front-end: it walks
a directory and prints a table.

Usage:
    python check-concept.py images/core_x.png
    python check-concept.py --all images/          # rank every core_*.png
"""

import argparse
import glob
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
for _c in (os.path.join(_here, "..", "src", "sprite_generator"),
           os.path.dirname(_here)):
    if os.path.isfile(os.path.join(_c, "concept.py")):
        sys.path.insert(0, _c)
        break

import concept as concept_lib  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("target")
    p.add_argument("--all", action="store_true",
                   help="treat target as a directory and rank core_*.png in it")
    p.add_argument("--key-tolerance", type=int, default=10)
    a = p.parse_args()

    paths = (sorted(glob.glob(os.path.join(a.target, "core_*.png")))
             if a.all else [a.target])
    if not paths:
        print("no core_*.png found", file=sys.stderr)
        return 2

    results = []
    for path in paths:
        v = concept_lib.judge(path, tolerance=a.key_tolerance)
        v["path"] = path
        results.append(v)

    # Usable first, then by how little of the frame they occupy - the most
    # isolated subject makes the best concept.
    results.sort(key=lambda r: (not r["ok"], r["coverage"]))

    print(f"{'concept':32} {'cover':>6} {'border':>7} {'aspect':>7}  verdict")
    for r in results:
        mark = "OK  " if r["ok"] else "NO  "
        print(f"{os.path.basename(r['path']):32} {r['coverage']:6.0%} "
              f"{r['border']:7.0%} {r['aspect']:7.2f}  {mark}{r['why']}")

    usable = [r for r in results if r["ok"]]
    print(f"\n{len(usable)}/{len(results)} usable as a character concept")
    return 0 if usable else 1


if __name__ == "__main__":
    sys.exit(main())
