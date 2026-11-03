#!/usr/bin/env python3
"""Does the VLM reference check actually separate the hand-labelled set?

SOMET-696. Before this is wired into the conveyor it has to beat the five
geometric proxies that failed, and the only honest way to know is to run it
against references whose truth was established by OPENING THE FILE.

The labelled set, all `dir_e` unless stated, verified by eye on 2026-11-03.
Every one of these is CUT at mid-thigh - that is the finding, and it is also
why this harness needs the `wide` renders: a check that answers "cut" to
everything would score 6/6 here and be worthless.

    --labelled   the six medium-era references, all cut
    --wide       the `wide`-default renders, verified complete by eye

A useful check gets both groups right. Scoring only the cut group measures
nothing but the model's willingness to agree.

Needs the card free: the brain cannot share it with a transformer (0013).
Check `docker top sprite_worker | grep build-sheet` first.

    python3 scripts/judge-references.py --labelled --wide
    python3 scripts/judge-references.py <job_id> [<job_id> ...]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, "/app")

import brain_engine  # noqa: E402
import reference_qc  # noqa: E402

IMAGES = "/app/images"

# (job, direction, expected_ok, note). Truth from opening each PNG, not from a
# metric - the metrics are what this is being measured against.
LABELLED = [
    ("7f9fc8f1-cfa7-4f2d-9220-7809e7a629b1", "e", False, "cut at thigh"),
    ("02cb757f-23fb-431b-a152-efe2a3f740d5", "e", False, "cut; scored MOST normal geometrically"),
    ("b0e77e2a-0e58-4c97-a8e9-72595a5b83c8", "e", False, "cut at thigh"),
    ("b0e77e2a-0e58-4c97-a8e9-72595a5b83c8", "w", False, "cut; both profiles are"),
    ("67e00bd5-9516-4b2a-904a-dee207fb1302", "e", False, "cut; was called full body"),
    ("12b6be3a-ff7b-4ec5-aa47-4b5e06526dc8", "e", False, "cut at thigh"),
    ("f11a661e-c8cc-46e0-a3b8-9bb799ed156b", "e", False, "cut; the 0.729-aspect counter-example"),
    # The control: good framing at `medium`, which exists for the front views.
    ("b0e77e2a-0e58-4c97-a8e9-72595a5b83c8", "s", True, "full body, feet visible"),
    ("b0e77e2a-0e58-4c97-a8e9-72595a5b83c8", "n", True, "full body"),
    ("02cb757f-23fb-431b-a152-efe2a3f740d5", "s", True, "full body"),
]

# Rendered with ISO_DISTANCE=wide and verified complete by eye.
WIDE = [
    ("distwide8-0000-0000-0000-000000000000", "e", True, "wide: head to boots"),
    ("distwide8-0000-0000-0000-000000000000", "w", True, "wide: head to boots"),
    ("distwide8-0000-0000-0000-000000000000", "s", True, "wide: unchanged control"),
    ("distwide8-0000-0000-0000-000000000000", "se", True, "wide: unchanged control"),
    ("distwide8-0000-0000-0000-000000000000", "ne", True, "wide: unchanged control"),
]


def run_cases(cases):
    rows, right = [], 0
    for job, d, expect, note in cases:
        p = os.path.join(IMAGES, f"_job_{job}", f"dir_{d}.png")
        if not os.path.exists(p):
            rows.append((job, d, expect, None, "MISSING", note))
            continue
        try:
            v = reference_qc.judge_one(p)
        except Exception as e:  # noqa: BLE001 - a harness, report and continue
            rows.append((job, d, expect, None, f"ERROR {e}"[:60], note))
            continue
        hit = v["ok"] == expect
        right += hit
        rows.append((job, d, expect, v["ok"],
                     f"{v.get('lowest_visible')}/head={v.get('head_complete')}",
                     note))
    return rows, right


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_ids", nargs="*")
    ap.add_argument("--labelled", action="store_true")
    ap.add_argument("--wide", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    if not brain_engine.has_vision(reference_qc.BRAIN):
        print(f"brain {reference_qc.BRAIN} has NO vision projector on disk.\n"
              f"Run: make provision   (stanza brain-vl-8b-mmproj, 1.08 GiB)\n"
              f"Refusing to run - the language half would answer about an "
              f"imagined image and score well doing it.", file=sys.stderr)
        return 2

    cases = []
    if a.labelled:
        cases += LABELLED
    if a.wide:
        cases += WIDE
    for job in a.job_ids:
        work = os.path.join(IMAGES, f"_job_{job}")
        for d in ["s", "se", "e", "ne", "n", "nw", "w", "sw"]:
            if os.path.exists(os.path.join(work, f"dir_{d}.png")):
                cases.append((job, d, None, "unlabelled"))
    if not cases:
        print("nothing to judge; pass --labelled, --wide or a job id",
              file=sys.stderr)
        return 2

    graded = [c for c in cases if c[2] is not None]
    rows, right = run_cases(graded)
    ungraded, _ = run_cases([(j, d, True, n) for j, d, e, n in cases
                             if e is None]) if len(cases) > len(graded) else ([], 0)

    print(f"\n{'job':<10} {'dir':<4} {'expect':<7} {'got':<7} "
          f"{'model said':<26} note")
    for job, d, expect, got, said, note in rows:
        mark = "" if got is None else ("  ok" if got == expect else "  <-- WRONG")
        print(f"{job[:8]:<10} {d:<4} "
              f"{('whole' if expect else 'cut'):<7} "
              f"{('-' if got is None else ('whole' if got else 'cut')):<7} "
              f"{said:<26} {note}{mark}")
    for job, d, _, got, said, note in ungraded:
        print(f"{job[:8]:<10} {d:<4} {'?':<7} "
              f"{('-' if got is None else ('whole' if got else 'cut')):<7} "
              f"{said:<26} {note}")

    n = len(rows)
    print(f"\n{right}/{n} correct on the hand-labelled set.")
    cut_rows = [r for r in rows if r[2] is False]
    whole_rows = [r for r in rows if r[2] is True]
    print(f"  cut references caught:   "
          f"{sum(1 for r in cut_rows if r[3] is False)}/{len(cut_rows)}")
    print(f"  whole references passed: "
          f"{sum(1 for r in whole_rows if r[3] is True)}/{len(whole_rows)}")
    print("\nBoth numbers matter. Catching every cut reference while failing "
          "the whole ones is a check that always says 'cut', which would "
          "re-render every reference forever.")

    if a.json:
        with open(a.json, "w") as f:
            json.dump([{"job": j, "dir": d, "expect": e, "got": g,
                        "said": s, "note": nt}
                       for j, d, e, g, s, nt in rows + list(ungraded)], f,
                      indent=1)
        print("wrote", a.json)
    return 0 if right == n else 1


if __name__ == "__main__":
    sys.exit(main())
