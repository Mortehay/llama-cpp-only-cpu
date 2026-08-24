#!/usr/bin/env python3
"""Assert every action resolves to a self-consistent skeleton + prompt + view.

WHY THIS EXISTS

Three separate bugs shipped in one day, and they were the same bug:

  * `burning` routed to HURT, a SIDE-view recoil cycle, while its prompt asked
    for a front view.
  * `move up` routed to WALK_FRONT, whose head keypoints place nose and both
    eyes facing camera, while its prompt asked for "back of the head, no face".
    It rendered identically to `move down`.
  * Every diagonal silently resolved to a cardinal and discarded its horizontal
    component, because matching is contiguous-substring and "move up right"
    contains "move up".

In all three the skeleton and the prompt described DIFFERENT CHARACTERS, and
nothing complained. Each was found by eye, after wasting a generation. The cost
of the class is not any one bug - it is that the failure mode is silent, so it
recurs.

This is the cheap check that makes it loud. It runs no model and needs no GPU:
it only asks whether the routing tables agree with each other.

    docker exec sprite_worker python /app/../scripts/validate-actions.py
    # or, from the repo, inside any container with the sources on PYTHONPATH:
    python scripts/validate-actions.py

Exit code is non-zero when anything disagrees, so it can gate a commit.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src", "sprite_generator"))
sys.path.insert(0, "/app")

import actions as action_lib  # noqa: E402
import poses  # noqa: E402
import tasks  # noqa: E402

# Every action the UI can send, plus the diagonals it cannot but callers can.
UI_ACTIONS = [
    "move right", "move left", "move down", "move up",
    "close melee attack", "distant attack with bow",
    "distant attack with sling", "close magic attack",
    "distant magic attack", "got damage", "idle", "burning",
]
DIAGONALS = [
    "move up right", "move down right", "move up left", "move down left",
]

# What the skeleton's own geometry says about facing, independent of any label.
# `is_side_view` keys off both shoulders collapsing onto x=0.50, which is a
# structural fact of the cycle rather than an annotation that can drift.
def skeleton_facing(cycle):
    if cycle is None:
        return None
    return "side" if poses.is_side_view(cycle) else "front-or-back"


def check(action, expect_view=None):
    problems = []
    entry = tasks.action_entry(action)
    cycle = poses.cycle_for(action)
    view = entry.get("view")

    if cycle is None:
        problems.append("no pose cycle - falls back to unposed img2img")
    if entry["prompt"] == action:
        problems.append("no expanded prompt - raw action text sent to the model")
    if not view:
        problems.append("no 'view' declared in action_prompts.json")

    # The core assertion: a side-view skeleton must not carry a front/back
    # prompt, and vice versa. This is exactly what burning and move_up violated.
    if cycle is not None and view:
        geom = skeleton_facing(cycle)
        if view in ("left", "right") and geom != "side":
            problems.append(f"view '{view}' wants a profile skeleton, "
                            f"but the cycle is {geom}")
        if view in ("front", "back") and geom == "side":
            problems.append(f"view '{view}' wants a front/back skeleton, "
                            f"but the cycle is a profile")

    if expect_view and view != expect_view:
        problems.append(f"expected view '{expect_view}', got '{view}'")

    return entry, cycle, view, problems


def check_sheet_frames():
    """Assert the conveyor's `frames` blocks in action_prompts.json are usable.

    The ControlNet checks above cover the prompt/skeleton/view tables. Nothing
    covered the conveyor's animation table, and its failure mode is the same
    shape: silent until a job is minutes into the GPU. Two that have bitten:

      * a frame count that is not actually available. encode truncates to the
        poses that exist, denoise used to iterate the REQUESTED count, and the
        run died with KeyError: 'idle|s|4' after the turnaround had been paid
        for. Both derive from _action_plan() now; this asserts the data those
        stages read is well-formed in the first place.

      * a `frames` block with no `side` family. `side` is the documented
        fallback for every family an action does not list, so without it the
        first profile direction raises KeyError deep in the plan. actions.py
        drops such an entry at load with an error - which is safe, but means
        the action silently disappears from the UI instead of failing here.

    Runs no model and needs no GPU.
    """
    problems = []
    known = action_lib.ACTIONS

    if not known:
        return ["no sheet actions loaded from action_prompts.json at all"]

    # Read the file DIRECTLY, not through actions.ACTIONS.
    #
    # actions._parse drops an entry whose frames block has no `side` family -
    # correct at runtime, because `side` is the fallback and without it the
    # first profile direction would raise. But it means the malformed action is
    # simply absent from ACTIONS, so checking ACTIONS can never see it: the
    # action disappears from the UI and every table here still agrees. The
    # declared-versus-loaded comparison below is the only place that catches it.
    with open(action_lib.ACTION_PROMPTS_PATH, encoding="utf-8") as fh:
        raw = json.load(fh)
    declared = {}
    for entry in raw.get("actions", []):
        if entry.get("frames"):
            declared[entry.get("sheet_id") or entry.get("id")] = entry["frames"]

    for name in sorted(declared):
        if name not in known:
            fams = sorted(declared[name])
            problems.append(
                f"{name}: declares a frames block but did not load - "
                f"families {fams}"
                + ("; no 'side', which is the required fallback"
                   if "side" not in declared[name] else ""))

    for name in sorted(known):
        families = known[name]
        for fam, poses_ in sorted(families.items()):
            if fam not in ("front", "back", "side"):
                problems.append(f"{name}: unknown view family {fam!r}; "
                                f"expected front, back or side")
            if not poses_:
                problems.append(f"{name}/{fam}: empty pose list")
            for i, pose in enumerate(poses_):
                if not isinstance(pose, str) or not pose.strip():
                    problems.append(f"{name}/{fam}[{i}]: empty pose")

        # Every direction must resolve, and max_frames must be honest about it:
        # it is what /api/jobs validates against and what the UI caps its input
        # to, so a disagreement here is a request that passes validation and
        # then cannot be built.
        limit = action_lib.max_frames([name])
        for d in sorted(action_lib.FAMILIES):
            try:
                got = len(action_lib.frames(name, d, limit))
            except Exception as e:
                problems.append(f"{name}/{d}: frames() raised {e!r}")
                continue
            if got != limit:
                problems.append(f"{name}/{d}: max_frames says {limit} but "
                                f"frames() returns {got}")

    return problems


def main():
    failures = 0
    print(f"{'action':28s} {'view':6s} {'skeleton':14s} status")
    print("-" * 74)

    for action in UI_ACTIONS:
        entry, cycle, view, problems = check(action)
        geom = skeleton_facing(cycle) or "NONE"
        status = "ok" if not problems else "FAIL"
        print(f"{action:28s} {str(view):6s} {geom:14s} {status}")
        for p in problems:
            print(f"    - {p}")
            failures += 1

    # Diagonals must alias to the HORIZONTAL cardinal. Before they did, they
    # resolved to the back view and dropped the horizontal component silently.
    print()
    for action in DIAGONALS:
        expect = "right" if "right" in action else "left"
        entry, cycle, view, problems = check(action, expect_view=expect)
        status = "ok" if not problems else "FAIL"
        print(f"{action:28s} {str(view):6s} {'aliased':14s} {status}")
        for p in problems:
            print(f"    - {p}")
            failures += 1

    # The conveyor's animation table. Separate from everything above: those
    # check the ControlNet img2img path, this checks the 2D sheet path, and
    # they read different fields of the same file.
    print()
    sheet_problems = check_sheet_frames()
    names = sorted(action_lib.ACTIONS)
    print(f"sheet actions ({len(names)}): {', '.join(names) or 'NONE'}")
    for p in sheet_problems:
        print(f"    - {p}")
        failures += 1
    if not sheet_problems:
        print(f"frames blocks: ok, ceiling {action_lib.max_frames()}")

    # Prompt budget. CLIP silently discards past 77 tokens, so an over-long
    # action prompt is absent from every generation while looking present.
    print()
    try:
        from transformers import CLIPTokenizer
        tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14",
                                            cache_dir="/models")

        # Refuse a degenerate tokenizer instead of counting with it.
        #
        # When the CLIP weights are archived off /models, from_pretrained does
        # NOT raise - it returns a tokenizer with an EMPTY vocab (vocab_size 2),
        # which then encodes roughly one token per character. Measured
        # 2026-08-25: every action reported ~220/77 and this check failed 12
        # times with numbers that were pure fiction, while looking exactly like
        # a real budget failure. A tokenizer that cannot tokenise must be an
        # error about the tokenizer, not a verdict about the prompts.
        if tok.vocab_size < 1000:
            raise RuntimeError(
                f"CLIP tokenizer loaded with vocab_size={tok.vocab_size}; the "
                f"weights are not really on /models. Restore them with "
                f"./scripts/archive-models.sh restore "
                f"models--openai--clip-vit-large-patch14")

        base = ("green zombie, tattered clothes, solid transparent background, "
                "only zombie, high quality pixel art, sharp focus")
        worst = 0
        for action in UI_ACTIONS:
            trig = tasks.action_entry(action)["prompt"]
            full = f"{trig}, {base}, single character, full body, centered"
            n = len(tok(full)["input_ids"])
            worst = max(worst, n)
            if n > 77:
                print(f"OVER BUDGET  {action}: {n}/77 tokens - tail discarded")
                failures += 1
        print(f"prompt budget: worst case {worst}/77")
    except Exception as e:
        print(f"prompt budget: SKIPPED ({type(e).__name__}: {e})")

    print()
    if failures:
        print(f"FAILED: {failures} disagreement(s) between skeleton, prompt "
              "and declared view.")
    else:
        print("All actions self-consistent.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
