#!/usr/bin/env python3
"""Report the frame ceiling per action, and for a chosen combination.

`max_frames()` is a MINIMUM across the selected axes, so one short action drags
the whole sheet down. That is correct - a sheet cannot have ragged rows - but it
makes the number surprising: extending walk to six poses does nothing for a
selection that also includes a four-pose action.

This makes the arithmetic visible, so "why is it still 4?" has an answer that
does not require reading the library.

Usage:
    python frames-ceiling.py
    python frames-ceiling.py --actions walk,attack
"""

import argparse
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
for _c in (os.path.join(_here, "..", "src", "sprite_generator"),
           os.path.dirname(_here)):
    if os.path.isfile(os.path.join(_c, "actions.py")):
        sys.path.insert(0, _c)
        break

import actions as action_lib  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--actions", default=None,
                   help="comma-separated; default is every known action")
    a = p.parse_args()

    known = sorted(action_lib.ACTIONS)
    families = sorted(action_lib.FAMILIES)
    # One row per action: the per-family counts and the action's own ceiling.
    print(f"{'action':10} " + " ".join(f"{f:>7}" for f in
                                       sorted(set(action_lib.FAMILIES.values())))
          + "   ceiling")
    for act in known:
        per = {}
        for d in families:
            fam = action_lib.family(d)
            per[fam] = action_lib.available_frames(act, d)
        cols = " ".join(f"{per.get(f, 0):>7}" for f in sorted(per))
        print(f"{act:10} {cols}   {action_lib.max_frames([act]):>7}")

    print()
    chosen = ([s.strip() for s in a.actions.split(",")] if a.actions else known)
    ceiling = action_lib.max_frames(chosen)
    print(f"selection: {', '.join(chosen)}")
    print(f"ceiling:   {ceiling}")
    if a.actions is None:
        print("\nNote: this is the ALL-ACTIONS ceiling. Selecting a subset of")
        print("actions that all have longer cycles raises it - try")
        print("  --actions walk,attack")
    return 0


if __name__ == "__main__":
    sys.exit(main())
