#!/usr/bin/env python3
"""Do the sheet pose lists obey the rules their own `_readme` sets out?

WHY THIS EXISTS

`action_prompts.json` records hard-won rules in `_readme.sheet_frames` and
nothing enforced any of them, so one of the rules was violated by the very
file that states it for over a year. `back_is_not_front` says, in as many
words, that the old walk list named "both feet flat on the ground, standing
upright" at frames 2 AND 5 and that this was fixed on 2026-10-22. It was
fixed for `back` and never for `front`, where the repeat survived until
2026-11-03 - producing two byte-identical renders per front row, measured
0.00 apart on both `s` and `se`.

It stayed invisible because `sheet.row_motion` counts CONSECUTIVE duplicate
pairs and 2-5 is not adjacent, so the atlas reported `duplicates: 0` for
every row of a sheet with two frozen frames in each of three rows.

These checks are structural: they read the JSON and need no GPU, no model and
no render. They cannot judge whether a pose is GOOD - only measurement can do
that, and the `sway` case is on record as a list that looked broken and
measured fine. What they can do is make the mechanical mistakes impossible to
reintroduce in silence.

FIVE CHECKS

  1. no pose string repeats inside one family list (the 2/5 bug)
  2. `back` is not a verbatim copy of `front` (the 2026-10-22 bug)
  3. every action with `frames` declares all three families
  4. every family list is the same length within one action
  5. no frame names a bare arm direction with no height qualifier

Check 5 is the one that is a heuristic rather than an invariant, so it reports
as an ADVISORY and does not fail the run - the wording rule is real but the
phrasings that satisfy it are open-ended.

    docker compose run --rm --entrypoint python sprite-worker \\
        /app/scripts/test-action-poses.py [path-to-a-candidate-file]

MUTATION RESULT (how this test was proved able to fail, 2026-11-03). Written
to a copy at /tmp/mutated.json and run with the optional path argument, so the
file the service reads was never touched:

    walk front 2 and 5 set back to the old repeated string   -> check 1 fails
    attack `back` replaced with a copy of `front`            -> check 2 fails
    exit code                        mutated 1, real file 0

WHAT THESE CHECKS DELIBERATELY DO NOT DO is judge whether a pose is any good.
`sway side` carries a repeat ON PURPOSE and is in ALLOWED_REPEATS with the
measurement that justifies it; `idle` and `sway` have no `front` list on
purpose and resolve through a documented fallback. A check that fired on
those would be a check nobody could ever get to pass, which this repo already
has one cautionary tale about (the >= 0.31 straight-run proposal).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [
    # The package is flattened onto /app in the image, so the file sits at the
    # root there rather than under a sprite_generator/ directory.
    "/app/action_prompts.json",
    "/app/sprite_generator/action_prompts.json",
    os.path.join(HERE, "..", "src", "sprite_generator", "action_prompts.json"),
]
FAMILIES = ("front", "side", "back")

# A direction word that places a limb but says nothing about HEIGHT. Measured
# 2026-11-03: every frame reported as a "pointing pose" was one of these with
# no height word anywhere in the string.
BARE_DIRECTIONS = ("arm forward", "arms forward", "arms raised high",
                   "arm raised high", "arms extended forward",
                   "arm extended forward")

# Repeats that are DELIBERATE, with the measurement that justifies them. A
# sway is a pendulum and its centre frames should look alike, so `sway side`
# carries one on purpose: rewriting that list to remove it measured WORSE
# (`sway e` 16.84 -> 14.01) and was reverted, keeping only the added `back`.
# Anything not listed here is the walk-cycle case, where a repeat freezes a
# third of the animation. Keep this list short and cite a measurement.
ALLOWED_REPEATS = {
    ("sway", "side"): "pendulum centre frames; rewriting measured worse "
                      "(sway e 16.84 -> 14.01, reverted 2026-10-26)",
}

# Actions whose pose is genuinely view-independent, where `actions.frames`
# falling back to the `side` list for a missing family is correct rather than
# an oversight - the fallback is documented in that function's docstring.
VIEW_INDEPENDENT = {"idle", "sway"}
HEIGHT_WORDS = ("low", "hip", "thigh", "waist", "knee", "shoulder height",
                "chest height", "overhead", "above the head", "ground",
                "side", "behind")


def load():
    # An explicit path first, so a candidate file - or a deliberately mutated
    # copy, to prove these checks can fail - can be run without touching the
    # one the service reads.
    for c in (sys.argv[1:2] or []) + CANDIDATES:
        if os.path.exists(c):
            return os.path.abspath(c), json.load(open(c, encoding="utf-8"))
    print("action_prompts.json not found; looked in:")
    for c in CANDIDATES:
        print("   ", os.path.abspath(c))
    sys.exit(1)


def sheet_actions(doc):
    """Only actions that carry sheet `frames`; the rest are the ControlNet path."""
    return [a for a in doc.get("actions", []) if a.get("frames")]


def check_no_repeats(actions, fails, advisories):
    for a in actions:
        sid = a.get("sheet_id", a["id"])
        for fam, lst in a["frames"].items():
            seen = {}
            for i, pose in enumerate(lst):
                if pose in seen:
                    why = ALLOWED_REPEATS.get((sid, fam))
                    where = (f"{sid}/{fam}: frames {seen[pose]} and {i} are "
                             f"the SAME STRING, so the conveyor renders one "
                             f"image and copies it -- {pose!r}")
                    (advisories if why else fails).append(
                        f"{where}{'  [allowed: ' + why + ']' if why else ''}")
                seen[pose] = i
        counts = ", ".join(
            "{}={}/{}".format(fam, len(set(lst)), len(lst))
            for fam, lst in a["frames"].items())
        print(f"  {a.get('sheet_id', a['id']):<8} {counts}")


def check_back_not_front(actions, fails, advisories):
    for a in actions:
        front, back = a["frames"].get("front"), a["frames"].get("back")
        if front and back and front == back:
            fails.append(
                f"{a.get('sheet_id', a['id'])}: `back` is a verbatim copy of "
                f"`front`. From behind the torso occludes anything held in "
                f"front of it, so those frames have no silhouette to change.")
        elif front and back:
            # An individual shared string is not the 2026-10-22 bug, which was
            # a whole verbatim copy. A recovery frame like "arms lowering to
            # the sides" is legitimately visible from behind. Worth seeing,
            # not worth failing on.
            shared = set(front) & set(back)
            if shared:
                advisories.append(
                    f"{a.get('sheet_id', a['id'])}: {len(shared)} pose "
                    f"string(s) appear in BOTH `front` and `back`; check each "
                    f"is actually visible from behind -- "
                    f"{sorted(shared)[0]!r}")


def check_families(actions, fails, advisories):
    for a in actions:
        sid = a.get("sheet_id", a["id"])
        if "side" not in a["frames"]:
            # `actions.frames` falls back to `by_family["side"]`, so a missing
            # side list is not a degraded render - it is a KeyError at build
            # time for every direction that has no list of its own.
            fails.append(
                f"{sid}: no `side` list, which is the fallback every other "
                f"family resolves to -- actions.frames raises KeyError")
        missing = [f for f in FAMILIES if f not in a["frames"] and f != "side"]
        if not missing:
            continue
        if sid in VIEW_INDEPENDENT:
            advisories.append(
                f"{sid}: no {', '.join(missing)} list, falling back to "
                f"`side`. Documented as correct for a view-independent "
                f"action, and NOT measured from those cameras.")
        else:
            fails.append(
                f"{sid}: no {', '.join(missing)} list, so those directions "
                f"are driven by another family's camera (see "
                f"`_readme.sheet_frames.back_is_not_front`)")


def check_lengths(actions, fails):
    for a in actions:
        lens = {f: len(v) for f, v in a["frames"].items()}
        if len(set(lens.values())) > 1:
            fails.append(
                f"{a.get('sheet_id', a['id'])}: families disagree on frame "
                f"count {lens}; a sheet asks every row for the same count")


def check_arm_heights(actions, advisories):
    for a in actions:
        for fam, lst in a["frames"].items():
            for i, pose in enumerate(lst):
                low = pose.lower()
                for bare in BARE_DIRECTIONS:
                    if bare in low and not any(h in low for h in HEIGHT_WORDS):
                        advisories.append(
                            f"{a.get('sheet_id', a['id'])}/{fam}/{i}: "
                            f"{bare!r} with no height word -- a direction "
                            f"alone renders as a pointing pose "
                            f"-- {pose!r}")


def main() -> int:
    path, doc = load()
    print("pose file:", path)
    actions = sheet_actions(doc)
    print(f"{len(actions)} action(s) carry sheet frames\n")

    fails, advisories = [], []
    print("== 1. no pose string repeats inside one family list "
          "(distinct/total) ==")
    check_no_repeats(actions, fails, advisories)
    print("\n== 2. `back` is not a copy of `front` ==")
    check_back_not_front(actions, fails, advisories)
    print("\n== 3. every action declares all three families ==")
    check_families(actions, fails, advisories)
    print("\n== 4. families agree on frame count ==")
    check_lengths(actions, fails)
    print("\n== 5. an arm direction carries a height (ADVISORY) ==")
    check_arm_heights(actions, advisories)

    print("\nADVISORIES:", "none" if not advisories else "")
    for a in advisories:
        print("  -", a)
    print("\nFAILURES:", "none" if not fails else "")
    for f in fails:
        print("  -", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
