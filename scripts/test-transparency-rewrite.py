#!/usr/bin/env python3
"""tasks.rewrite_transparency: swaps "transparent background" for a keyable one.

Why the phrase is swapped at all: .ai/specs/transparent-cutouts/ - asked for a
transparent background the model PAINTS the checkerboard, and the cutout cannot
clear it (9/9 opaque), while "plain white background" cut out 12/12.

The cases that must NOT fire matter as much as the ones that must: a
transparent crystal, clear water or a translucent slime is the SUBJECT, and
rewriting it would change what gets drawn.

Run: make test-transparency-rewrite
"""
import sys

sys.path.insert(0, "/app")
from tasks import (rewrite_transparency, split_negations,  # noqa: E402
                   TRANSPARENCY_REPLACEMENT, TRANSPARENCY_NEGATIVE)

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(name)


S2_NEG = "picture frame, sprite sheet, multiple objects"

# Must fire: something2's real prompts, and the common wordings.
for prompt in [
    "pixel art, a translucent green slime blob, single object, with transparent background, only slime",
    "pixel art, a translucent green slime blob, single object, solid transparent background, only slime",
    "pixel art, a wooden chest, no background",
    "pixel art, a sword, transparent bg",
    "pixel art, a potion, png with transparency",
    "pixel art, a helmet on a transparent backdrop",
    "pixel art, a torch, transparent png",
]:
    pos, neg, asked = rewrite_transparency(prompt, S2_NEG)
    check(f"fires: {prompt[-45:]!r}", asked and TRANSPARENCY_REPLACEMENT in pos
          and "transparen" not in pos.lower().replace("translucent", "")
          and TRANSPARENCY_NEGATIVE in neg and neg.startswith(S2_NEG), pos)

# Said three ways -> replaced once, no comma debris.
pos, _, asked = rewrite_transparency(
    "pixel art, a ring, transparent background, no background, with alpha channel")
check("three wordings -> one replacement", asked and pos.count(TRANSPARENCY_REPLACEMENT) == 1
      and ",," not in pos and not pos.endswith(","), pos)

# Must NOT fire: transparency of the SUBJECT, or a background that is not empty.
for prompt in [
    "pixel art, a transparent crystal sword, plain background",
    "pixel art, a glass of clear water on a wooden table",
    "pixel art, a translucent green slime monster",
    "pixel art, a knight, forest background",
    "pixel art, a ghost, semi-transparent body",
    "",
]:
    pos, neg, asked = rewrite_transparency(prompt, "x")
    check(f"leaves alone: {prompt[-45:]!r}", not asked and pos == prompt and neg == "x")

# Order matters: after the rewrite, split_negations must not strip "background".
pos, neg, _ = rewrite_transparency("pixel art, a chest, no background, no frame", "")
pos2, neg2 = split_negations(pos, neg)
check("rewrite then split_negations keeps the white backdrop",
      TRANSPARENCY_REPLACEMENT in pos2 and "frame" in neg2 and "background," not in neg2.split("checkered")[0],
      f"{pos2!r} | {neg2!r}")

if failures:
    print(f"\n{len(failures)} failure(s)")
    sys.exit(1)
print("\nall transparency-rewrite checks passed")
