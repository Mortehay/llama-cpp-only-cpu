#!/usr/bin/env python3
"""Does the audit still predict what the cutout stage actually does?

WHY THIS EXISTS

`audit-entity-refs.key_background` is a copy of `tasks.remove_background`'s
rule. The audit's entire claim - "this backdrop is recoverable", "this one is
not" - is only true while the copy matches. It did not: it was written as a twin
of `pixelate.key_background`, a different function with a different rule, and
mis-scored 300 of 462 references before anyone compared them. 25 of those
crossed the blocking threshold and were reported as recoverable backdrops that
`remove_background` cannot key at all.

Nothing would have caught that. This does.

IT READS THE ORIGINAL'S SOURCE, NOT A COPY OF ITS BEHAVIOUR

`remove_background` is lifted out of `tasks.py` with `ast` and executed here.
Importing `tasks` would drag in celery, psycopg2, torch and a CUDA check, so the
tempting shortcut is to hand-transcribe the rule into the test - and that is
worthless: a transcription written from the same misunderstanding that produced
the bug agrees with the bug. The source text is the only thing that cannot
inherit the author's belief about what it says.

Borrowed from a something2 session by way of the ADR 0009 session, both of whom
hit the same class of problem in the same afternoon.

THE MUTATION RESULT, so nobody has to trust that this test can fail

Reverting `key_background` to the old any-corner rule and re-running:

    correct rule   0 disagreements over 462 files, worst delta 0.00000
    any-corner     300 disagreements,              worst delta 0.80253

Checked 2026-08-27. A test that has never been shown to go red is decoration -
three separate sessions shipped one on this codebase in a single day.

Usage:
    python test-audit-mirrors-cutout.py --data ./images
"""

import argparse
import ast
import glob
import importlib.util
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = os.path.join(HERE, "..", "src", "sprite_generator", "tasks.py")

# Above this fraction of transparent pixels an image already has usable alpha,
# so `subject_mask` never consults `key_background` and the comparison would be
# vacuous. Same constant the audit uses.
ALPHA_IN_USE_PCT = 0.02

# Agreement is expected to be EXACT. The tolerance exists so a future
# vectorisation that changes a rounding edge does not fail the suite over one
# pixel; it is not room for a different rule.
MAX_DELTA = 0.001


class _Logger:
    def __getattr__(self, _):
        return lambda *a, **kw: None


def load_remove_background(path=TASKS):
    """Execute the real `remove_background` out of tasks.py, by source text."""
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == "remove_background"),
              None)
    if fn is None:
        sys.exit("remove_background not found in %s - it was renamed or moved, "
                 "which is exactly the drift this test exists to catch" % path)
    body = ast.get_source_segment(src, fn)
    if body is None:
        sys.exit("could not recover remove_background's source text from %s. "
                 "This test is worthless without it - a hand-transcribed rule "
                 "would agree with whatever misunderstanding wrote it." % path)
    ns = {"Image": Image, "np": np, "ndimage": ndimage, "logger": _Logger(),
          # Only reached when keep_largest=True, which this comparison never sets.
          "_isolate_largest_sprite": lambda arr: arr}
    exec(compile(body, path, "exec"), ns)
    return ns["remove_background"], len(body.splitlines())


def load_audit():
    path = os.path.join(HERE, "audit-entity-refs.py")
    spec = importlib.util.spec_from_file_location("audit_entity_refs", path)
    if spec is None or spec.loader is None:
        sys.exit("cannot load %s" % path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="./images")
    p.add_argument("--limit", type=int, default=0,
                   help="check only the first N eligible images")
    a = p.parse_args()

    real, nlines = load_remove_background()
    audit = load_audit()
    print("loaded remove_background from tasks.py (%d lines of source)" % nlines)

    files = [q for q in sorted(glob.glob(os.path.join(a.data, "ref_*.png")))
             if not os.path.basename(q).startswith("thumb_")]
    if not files:
        sys.exit("no ref_*.png under %s" % a.data)

    checked = 0
    fails = []
    worst = (0.0, None)
    for q in files:
        img = Image.open(q)
        arr = np.asarray(img.convert("RGBA"))
        if (arr[..., 3] < 128).mean() > ALPHA_IN_USE_PCT:
            continue                      # alpha in use; key_background unused
        checked += 1

        mine = float(audit.key_background(img).mean())
        out = np.asarray(real(img).convert("RGBA"))
        theirs = float((out[..., 3] < 128).mean())

        delta = abs(mine - theirs)
        if delta > worst[0]:
            worst = (delta, os.path.basename(q))
        if delta > MAX_DELTA:
            fails.append((os.path.basename(q), mine, theirs))
        if a.limit and checked >= a.limit:
            break

    print("compared on %d image(s) where key_background decides" % checked)
    print("worst delta: %.5f%s" % (worst[0], " on " + worst[1] if worst[1] else ""))

    if not checked:
        sys.exit("FAIL: nothing was comparable - every image had alpha in use, "
                 "so this run proved nothing")
    if fails:
        print("\nFAIL: %d image(s) where the audit and the cutout stage disagree"
              % len(fails))
        for n, mi, th in fails[:15]:
            print("  %-34s audit %.4f  cutout %.4f" % (n, mi, th))
        print("\nThe audit is no longer predicting what remove_background does. "
              "Either the cutout rule changed and key_background must follow, or "
              "key_background drifted.")
        return 1

    print("\nPASS: the audit mirrors the cutout stage exactly on all %d." % checked)
    return 0


if __name__ == "__main__":
    sys.exit(main())
