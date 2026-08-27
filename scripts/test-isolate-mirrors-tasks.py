#!/usr/bin/env python3
"""Does the recovery's `isolate_largest` match `_isolate_largest_sprite`?

WHY THIS EXISTS

`recover-grid-refs.py` restates `_isolate_largest_sprite`'s rule instead of
calling it: the original mutates a numpy array in place and logs, while the
recovery needs a PIL round trip and a dropped-pixel count. Four lines, easy to
restate, and a restatement is exactly what already went wrong once in this
investigation - `audit-entity-refs.key_background` was written as a twin of the
wrong function and mis-scored 300 of 462 references before anything compared
them.

That copy decides which pixels get deleted from a training reference. It is
worth a test.

The original's source is lifted out of `tasks.py` with `ast` and executed here,
rather than transcribed a second time. A transcription written from the same
misunderstanding that produced a bug agrees with the bug.

MUTATION RESULT, so nobody has to trust that this test can fail

Changing `isolate_largest` to 4-connectivity (`ndimage.generate_binary_
structure(2, 1)`) instead of the original's 8-connectivity 3x3 block:

    8-connectivity (correct)   0 disagreements over 22 files (13 multi-blob)
    4-connectivity (mutant)   14 disagreements, worst 584 px

A diagonal-only join is common in pixel art, so this is the mutation that
matters: it severs limbs and outlines from their own bodies.

Usage:
    python3 scripts/test-isolate-mirrors-tasks.py --data images/recovered/entity
"""

import argparse
import ast
import glob
import importlib.util
import os
import sys
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
# Two layouts - see test-audit-mirrors-cutout.py. On the host the package is
# `<repo>/src/sprite_generator`; in the container that directory IS `/app` with
# the scripts at `/app/scripts`, so the relative walk lands on a path that does
# not exist and the Makefile target dies where a host shell passes.
_TASKS_CANDIDATES = [
    os.path.join(HERE, "..", "src", "sprite_generator", "tasks.py"),
    os.path.join(HERE, "..", "tasks.py"),
]
TASKS = next((p for p in _TASKS_CANDIDATES if os.path.isfile(p)), None)
if TASKS is None:
    sys.exit("cannot find tasks.py; looked in: "
             + ", ".join(os.path.abspath(p) for p in _TASKS_CANDIDATES))


class _Logger:
    def __getattr__(self, _):
        return lambda *a, **kw: None


def load_original(path=TASKS) -> Any:
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef)
               and n.name == "_isolate_largest_sprite"), None)
    if fn is None:
        sys.exit("_isolate_largest_sprite not found in %s - it was renamed or "
                 "moved, which is exactly the drift this test exists to catch"
                 % path)
    body = ast.get_source_segment(src, fn)
    if body is None:
        sys.exit("could not recover _isolate_largest_sprite's source from %s. "
                 "This test is worthless without it." % path)
    ns = {"np": np, "ndimage": ndimage, "logger": _Logger()}
    exec(compile(body, path, "exec"), ns)
    return ns["_isolate_largest_sprite"]


def load_copy() -> Any:
    path = os.path.join(HERE, "recover-grid-refs.py")
    spec = importlib.util.spec_from_file_location("recover_grid_refs", path)
    if spec is None or spec.loader is None:
        sys.exit("cannot load %s" % path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.isolate_largest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="images/recovered/entity")
    a = p.parse_args()

    original, copy = load_original(), load_copy()
    files = sorted(glob.glob(os.path.join(a.data, "*.png")))
    if not files:
        sys.exit("no images under %s - run `make recover-entity-refs` first. "
                 "A comparison over zero files passes and proves nothing."
                 % a.data)

    bad = 0
    multi = 0
    for f in files:
        img = Image.open(f).convert("RGBA")
        want = original(np.array(img))
        got = np.array(copy(img)[0].convert("RGBA"))

        # Only images with more than one blob exercise the rule at all. Count
        # them, so a run over 22 single-blob images cannot report a green pass.
        opaque = np.array(img)[:, :, 3] > 0
        _, n = ndimage.label(opaque, structure=np.ones((3, 3), dtype=bool))
        if n > 1:
            multi += 1

        delta = int((want[:, :, 3] != got[:, :, 3]).sum())
        if delta:
            print("DISAGREE %-30s %d px" % (os.path.basename(f), delta))
            bad += 1

    print()
    print("compared %d file(s), %d of them with more than one blob" %
          (len(files), multi))
    if not multi:
        print("FAIL: every file had a single blob, so the rule was never "
              "exercised and this run proves nothing.")
        return 1
    if bad:
        print("FAIL: %d disagreement(s)" % bad)
        return 1
    print("PASS: isolate_largest matches _isolate_largest_sprite exactly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
