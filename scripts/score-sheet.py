#!/usr/bin/env python3
"""Score the raw cells of a finished sheet job against sheet_qc.

    score-sheet.py <job-id-or-work-dir> [--json out.json] [--quiet]

Reads `_job_<id>/cell_<action>_<direction>_<frame>.png` - the RAW 512px
renders, before the fit into 48x64 - and the `dir_<d>.png` turnaround views
that each row was conditioned on, and reports which frames fail which check.

WHY THIS IS A SCRIPT AND NOT A TEST

It is the only way to tune a threshold honestly. Every number in sheet_qc was
set by running this over the two evidence jobs at different cuts and looking
at where the clean population ends. A unit test with a frozen expectation
would have locked in the first guess.

It is also the regression gate: the checks must still flag the four cells the
owner pointed at. `--anti-goals` asserts exactly that and exits non-zero if
any of them has stopped firing, which is what stops this becoming a check
that always passes.

Per the repo convention the Markdown stays summarised and the per-row data
goes to the --json sibling.
"""

import argparse
import json
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
for _c in (os.path.join(_here, "..", "src", "sprite_generator"),
           os.path.dirname(_here), "/app"):
    if os.path.isfile(os.path.join(_c, "sheet_qc.py")):
        sys.path.insert(0, _c)
        break

from PIL import Image  # noqa: E402

import sheet_qc  # noqa: E402

IMAGES = os.environ.get("IMAGES_DIR", "/app/images")

# The cells the owner named, per the design brief's anti-goals. If the checks
# stop flagging these, the checks are worthless however green they look.
ANTI_GOALS = [
    ("212d56d7", "attack", "e", 3, "facing"),
    ("8d9d164d", "attack", "s", 2, "close_up"),
    ("8d9d164d", "attack", "s", 4, "close_up"),
    ("8d9d164d", "attack", "se", 3, "backdrop"),
    ("8d9d164d", "attack", "se", 4, "backdrop"),
]


def work_dir(target: str) -> str:
    if os.path.isdir(target):
        return target
    cand = os.path.join(IMAGES, f"_job_{target}")
    if os.path.isdir(cand):
        return cand
    raise SystemExit(f"no work directory for {target!r} (looked in {cand})")


def score_job(work: str, quiet: bool = False) -> dict:
    with open(os.path.join(work, "config.json")) as fh:
        cfg = json.load(fh)

    directions = cfg["directions"]
    actions = cfg.get("actions") or []
    frames = cfg.get("frames", 0)

    refs = {}
    for d in directions:
        r = sheet_qc.Reference.from_path(os.path.join(work, f"dir_{d}.png"))
        if r is None and not quiet:
            print(f"  warning: no dir_{d}.png; row {d} cannot be scored")
        refs[d] = r

    rows = []
    for act in actions:
        for d in directions:
            for f in range(frames):
                path = os.path.join(work, f"cell_{act}_{d}_{f}.png")
                if not os.path.isfile(path):
                    continue
                m = sheet_qc.measure(Image.open(path), refs.get(d))
                rows.append({
                    "action": act, "direction": d, "frame": f,
                    "failed_checks": sheet_qc.failures(m),
                    "score": round(sheet_qc.score(m), 3),
                    "metrics": {k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in m.items()},
                })
                if not quiet and len(rows) % 48 == 0:
                    print(f"  scored {len(rows)} cells", file=sys.stderr)

    return {
        "work": work,
        "engine": sheet_qc.engine(),
        "directions": directions,
        "actions": actions,
        "frames": frames,
        "thresholds": {
            "figure_ratio_max": sheet_qc.FIGURE_RATIO_MAX,
            "backdrop_max": sheet_qc.BACKDROP_MAX,
            "border_max": sheet_qc.BORDER_MAX,
            "lean_min": sheet_qc.LEAN_MIN,
            "ident_max": sheet_qc.IDENT_MAX,
        },
        "reference": {d: (r.as_dict() if r else None) for d, r in refs.items()},
        "cells": rows,
    }


def summarise(rep: dict) -> None:
    cells = rep["cells"]
    bad = [c for c in cells if c["failed_checks"]]
    print(f"\n{os.path.basename(rep['work'])}: {len(cells)} cells, "
          f"{len(bad)} flagged ({len(bad) / max(len(cells), 1) * 100:.1f}%), "
          f"mask engine {rep['engine']}")

    counts = {}
    for c in bad:
        for k in c["failed_checks"]:
            counts[k] = counts.get(k, 0) + 1
    if counts:
        print("  by check: " + ", ".join(f"{k} {v}"
                                         for k, v in sorted(counts.items())))

    by_row = {}
    for c in bad:
        by_row.setdefault((c["action"], c["direction"]), []).append(c)
    print("  worst rows:")
    for (act, d), cs in sorted(by_row.items(),
                               key=lambda kv: -len(kv[1]))[:8]:
        fr = ",".join(f"f{c['frame']}" for c in sorted(cs,
                                                       key=lambda c: c["frame"]))
        checks = sorted({k for c in cs for k in c["failed_checks"]})
        print(f"    {act:7s} {d:2s}  {len(cs)}/{rep['frames']}  {fr:20s} "
              f"{'+'.join(checks)}")


def check_anti_goals(reports: dict) -> int:
    """Every cell the owner named must still be flagged, for the right reason."""
    print("\nanti-goals (cells the owner named; each must still fire):")
    missed = 0
    for job, act, d, f, want in ANTI_GOALS:
        rep = next((r for k, r in reports.items() if k.startswith(job)), None)
        if rep is None:
            print(f"  SKIP {job} {act} {d} f{f}: job not scored")
            continue
        cell = next((c for c in rep["cells"]
                     if c["action"] == act and c["direction"] == d
                     and c["frame"] == f), None)
        if cell is None:
            print(f"  MISS {job} {act} {d} f{f}: cell not on disk")
            missed += 1
            continue
        got = cell["failed_checks"]
        ok = want in got
        missed += 0 if ok else 1
        print(f"  {'PASS' if ok else 'MISS'} {job} {act:6s} {d:2s} f{f}  "
              f"want {want:8s} got {','.join(got) or '(none)'}")
    return missed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("targets", nargs="+", help="job id(s) or work dir(s)")
    p.add_argument("--json", help="write the per-cell data here")
    p.add_argument("--anti-goals", action="store_true",
                   help="assert the owner's named cells still fire; "
                        "exit 1 if any stopped")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()

    reports = {}
    for t in a.targets:
        w = work_dir(t)
        if not a.quiet:
            print(f"scoring {w} ...", file=sys.stderr)
        rep = score_job(w, a.quiet)
        reports[os.path.basename(w).replace("_job_", "")] = rep
        summarise(rep)

    if a.json:
        with open(a.json, "w") as fh:
            json.dump(reports, fh, indent=2)
        print(f"\nper-cell data -> {a.json}")

    if a.anti_goals:
        missed = check_anti_goals(reports)
        if missed:
            print(f"\n{missed} anti-goal(s) no longer fire. The checks have "
                  f"stopped catching what they exist to catch.")
            return 1
        print("\nall anti-goals still fire.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
