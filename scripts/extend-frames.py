#!/usr/bin/env python3
"""Extend an action's frames block from 4 poses to 6, in action_prompts.json.

Done as a script rather than by hand-editing the JSON because the file is now
the single source of truth for BOTH conveyors (see actions.py), it is ~450
lines, and a stray comma in it takes the action table down for the whole
service. This parses, edits and re-serialises, so the result is valid or the
script fails.

WHY SIX AND NOT MORE POSES OF THE SAME KIND

A 4-frame walk is contact/passing/contact/passing - it reads, but the character
snaps between extremes. Six adds the DOWN phase either side, which is what makes
a walk look weighted rather than glided:

    contact -> down -> passing -> contact -> down -> passing

The same logic applies to a strike: 4 frames give wind-up/mid/hit/recover, and
the two extra frames buy anticipation before the hit and follow-through after
it, which is where the weight reads.

Every added pose still obeys the two rules in actions.py's docstring: it names
limb positions rather than a verb, and it changes the silhouette as seen from
THAT camera - so the front/back variants move vertically (knee height, arm
height) where the profile variants move sagittally (stride length, reach).

Usage:
    python extend-frames.py [--file action_prompts.json] [--dry-run]
"""

import argparse
import json
import os
import sys

# sheet_id -> family -> the full six-pose cycle that REPLACES the four.
SIX = {
    "walk": {
        "side": [
            "left leg forward heel down, right arm forward, contact",
            "weight settling onto the left leg, body at its lowest",
            "legs together, right leg passing, body at its highest",
            "right leg forward heel down, left arm forward, contact",
            "weight settling onto the right leg, body at its lowest",
            "legs together, left leg passing, body at its highest",
        ],
        # Vertical, not horizontal: head-on a stride is foreshortened to
        # nothing, so the cycle has to be carried by knee and body height.
        "front": [
            "left knee raised high, left foot well off the ground",
            "left foot lowering toward the ground, knee half bent",
            "both feet flat on the ground, standing upright",
            "right knee raised high, right foot well off the ground",
            "right foot lowering toward the ground, knee half bent",
            "both feet flat on the ground, standing upright",
        ],
        "back": [
            "left knee raised high, left foot well off the ground",
            "left foot lowering toward the ground, knee half bent",
            "both feet flat on the ground, standing upright",
            "right knee raised high, right foot well off the ground",
            "right foot lowering toward the ground, knee half bent",
            "both feet flat on the ground, standing upright",
        ],
    },
    "attack": {
        "side": [
            "arms drawn back behind the body, weight on the back foot",
            "arms coiled, shoulders turned, about to release",
            "arms swinging forward, weight shifting to the front foot",
            "arms fully extended forward, strike landed",
            "arms past full extension, body carried forward",
            "arms lowering back to the sides, weight settling",
        ],
        # An overhead swing, because a forward thrust cannot be seen head-on -
        # it came back as a literal T-pose in six of eight directions.
        "front": [
            "both arms raised high overhead, leaning back",
            "arms at their highest, body coiled and still",
            "arms coming down, elbows bent, weight shifting forward",
            "arms swung down low in front, body hunched forward",
            "arms low and loose, body still bent forward",
            "arms returning to the sides, standing upright",
        ],
        "back": [
            "both arms raised high overhead, leaning back",
            "arms at their highest, body coiled and still",
            "arms coming down, elbows bent, weight shifting forward",
            "arms swung down low in front, body hunched forward",
            "arms low and loose, body still bent forward",
            "arms returning to the sides, standing upright",
        ],
    },
    # idle is deliberately NOT extended. It is a breathing cycle - two
    # positions and their returns - and six frames of it would be four frames
    # of standing still with extra steps.
}


def main():
    p = argparse.ArgumentParser()
    # Two layouts, same as every other script here: `src/sprite_generator/` in a
    # checkout, and `/app` in the container (which mounts scripts at
    # /app/scripts, so `..` is /app itself, not a src tree).
    here = os.path.dirname(os.path.abspath(__file__))
    default = None
    for candidate in (os.path.join(here, "..", "src", "sprite_generator",
                                   "action_prompts.json"),
                      os.path.join(os.path.dirname(here),
                                   "action_prompts.json")):
        if os.path.isfile(candidate):
            default = candidate
            break
    p.add_argument("--file", default=default, required=default is None)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    with open(a.file, encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("actions", [])
    if not entries:
        print("no `actions` list in the file", file=sys.stderr)
        return 1

    changed = 0
    for entry in entries:
        sid = entry.get("sheet_id")
        if sid not in SIX or "frames" not in entry:
            continue
        for family, poses in SIX[sid].items():
            if family not in entry["frames"]:
                continue
            was = len(entry["frames"][family])
            entry["frames"][family] = poses
            print(f"  {sid:8} {family:6} {was} -> {len(poses)} poses")
            changed += 1

    if not changed:
        print("nothing to change")
        return 0

    if a.dry_run:
        print(f"\n--dry-run: {changed} block(s) would change")
        return 0

    # Re-serialise rather than patch text: the file carries a large _readme and
    # a lot of prose, and a regex edit of it is how a comma goes missing.
    with open(a.file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\nupdated {changed} block(s) in {a.file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
