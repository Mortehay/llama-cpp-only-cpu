#!/usr/bin/env python3
"""Is a bigger editor quant actually better? Answer it on a SUBSET.

    quant-ab.py --ref /app/images/_job_2ff8af36-... \
                --gguf /models/image-gguf/transformers/qwen-image-edit-2511-Q6_K.gguf \
                --offload --actions attack,cast --directions s,se,e,ne

WHY THIS EXISTS

Q3_K_S was evaluated by building a whole 7-action sheet with it: 210 cells,
four hours of GPU, and the verdict was "worse" (27 -> 45 flagged, because
identity improved and facing regressed). That verdict was readable from a
couple of dozen cells. The four hours bought nothing the subset would not
have shown, and the card was unavailable for anything else meanwhile.

So: render the SAME (action, direction, frame) cells at the SAME seed from
the SAME concept as a finished reference job, under a different quant, and
compare first-attempt scores cell for cell.

WHAT MAKES IT COMPARABLE

  concept   taken from the reference job's own config.json, and re-encoded
            with --no-clean-concept, because `concept_clean.png` has already
            been through the cleaner and a second pass is a different image
  seed      taken from the reference too; `_cell_seed()` derives each cell's
            seed from it, so cell N here is cell N there
  attempt   `attempts[0]` on both sides. The reference has redraw rounds in
            it; this does not render any, and comparing a redrawn cell
            against a first attempt would measure the redraw, not the quant

WHAT IT CANNOT TELL YOU

The sheet-wide scale is the MINIMUM fit over ALL cells, so a subset cannot
predict the final sprite size - and `identity` is a colour histogram and is
spatially blind (sheet-quality contract 7). A win here earns a full sheet;
it is not itself the adoption decision.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

BUILD = "/app/scripts/build-sheet.py"

_here = os.path.dirname(os.path.abspath(__file__))
for _c in (os.path.join(_here, "..", "src", "sprite_generator"),
           os.path.dirname(_here), "/app"):
    if os.path.isfile(os.path.join(_c, "sheet_qc.py")):
        sys.path.insert(0, _c)
        break

# Imported, not restated: a check added to sheet_qc and not to a hardcoded
# list here would simply never appear in the comparison.
from sheet_qc import CHECKS  # noqa: E402


def run(stage: str, work: str, env: dict, *extra) -> None:
    cmd = [sys.executable, BUILD, stage, *extra, "--work", work]
    t0 = time.time()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    tail = (p.stdout or p.stderr or "").strip().splitlines()[-1:]
    print("  %-18s %6.0fs  %s" % (stage, time.time() - t0,
                                  tail[0] if tail else ""), flush=True)
    if p.returncode != 0:
        print(p.stdout[-3000:])
        print(p.stderr[-3000:])
        raise SystemExit("stage %s failed (%d)" % (stage, p.returncode))


def first(cell: dict) -> dict:
    """The first attempt's metrics and failures, never a redraw's."""
    att = cell.get("attempts") or []
    return att[0] if att else {}


def summarise(cells: dict, keys: list) -> dict:
    fails = {c: 0 for c in CHECKS}
    ident, flagged = [], 0
    for k in keys:
        a = first(cells[k])
        m = a.get("metrics") or {}
        fc = a.get("failed_checks") or []
        if fc:
            flagged += 1
        for c in fc:
            fails[c] = fails.get(c, 0) + 1
        if "identity" in m:
            ident.append(m["identity"])
    return {"n": len(keys), "flagged": flagged, "fails": fails,
            "identity_median": round(statistics.median(ident), 5) if ident
            else None,
            "identity_max": round(max(ident), 5) if ident else None}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True,
                   help="a finished job work dir to compare against")
    p.add_argument("--gguf", required=True,
                   help="ABSOLUTE path to the quant under test")
    p.add_argument("--offload", action="store_true",
                   help="stream the transformer from host RAM")
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--actions", default="attack,cast")
    p.add_argument("--directions", default="s,se,e,ne")
    p.add_argument("--frames", type=int, default=6)
    p.add_argument("--work", default="/app/images/_qab")
    p.add_argument("--angles-scale", type=float, default=0.4)
    p.add_argument("--skip-render", action="store_true",
                   help="only re-compare an existing --work")
    a = p.parse_args()

    ref_cfg = json.load(open(os.path.join(a.ref, "config.json")))
    ref_scores = json.load(open(os.path.join(a.ref, "scores.json")))
    concept = ref_cfg["concept"]
    seed = ref_cfg["seed"]
    if not os.path.isfile(concept):
        # the reference recorded its own absolute path; it may have moved
        concept = os.path.join(a.ref, os.path.basename(concept))
    if not os.path.isabs(a.gguf):
        raise SystemExit(
            "--gguf must be an ABSOLUTE path. A bare filename is sent to the "
            "Hub and fails offline with a message that reads like a missing "
            "download.")

    env = dict(os.environ)
    env["QWEN_EDIT_GGUF_FILE"] = a.gguf
    env["QWEN_GROUP_OFFLOAD"] = "1" if a.offload else "0"
    env["QWEN_GROUP_OFFLOAD_BLOCKS"] = str(a.blocks)

    print("quant   %s" % os.path.basename(a.gguf))
    print("offload %s%s" % (a.offload, " blocks=%d" % a.blocks if a.offload
                            else ""))
    print("ref     %s  seed %d" % (os.path.basename(a.ref), seed))
    print("subset  actions=%s directions=%s frames=%d"
          % (a.actions, a.directions, a.frames))

    t0 = time.time()
    if not a.skip_render:
        os.makedirs(a.work, exist_ok=True)
        run("turnaround-encode", a.work, env, concept,
            "--directions", a.directions, "--seed", str(seed),
            "--no-clean-concept")
        run("turnaround-denoise", a.work, env)
        run("actions-encode", a.work, env, "--actions", a.actions,
            "--frames", str(a.frames))
        run("actions-denoise", a.work, env,
            "--angles-scale", str(a.angles_scale))
    wall = time.time() - t0

    new = json.load(open(os.path.join(a.work, "scores.json")))
    keys = sorted(set(new["cells"]) & set(ref_scores["cells"]))
    missing = sorted(set(new["cells"]) - set(ref_scores["cells"]))
    if not keys:
        raise SystemExit("no cells in common with the reference - check that "
                         "--actions and --frames match how it was built")

    old_s = summarise(ref_scores["cells"], keys)
    new_s = summarise(new["cells"], keys)

    print("\n%d cells compared%s" % (len(keys),
                                     ", %d not in the reference (ignored)"
                                     % len(missing) if missing else ""))
    print("%-18s %10s %10s" % ("", "reference", "candidate"))
    print("%-18s %10d %10d" % ("flagged", old_s["flagged"], new_s["flagged"]))
    for c in CHECKS:
        o, n = old_s["fails"].get(c, 0), new_s["fails"].get(c, 0)
        if o or n:
            print("%-18s %10d %10d" % ("  " + c, o, n))
    for k in ("identity_median", "identity_max"):
        print("%-18s %10s %10s" % (k, old_s[k], new_s[k]))
    if not a.skip_render:
        print("%-18s %10s %10.0fs" % ("wall clock", "-", wall))
        print("%-18s %10s %10.1fs" % ("per cell", "-", wall / len(keys)))

    # The point of the gate: say which way it went, in one line, rather than
    # leaving a table for someone to read hopefully.
    d = new_s["flagged"] - old_s["flagged"]
    verdict = ("BETTER" if d < 0 else "WORSE" if d > 0 else "NO CHANGE")
    print("\n%s on flagged cells (%+d). Identity %s." % (
        verdict, d,
        "better" if (new_s["identity_median"] or 1) <
        (old_s["identity_median"] or 1) else "not better"))
    print("A win here earns a full sheet; it is not the adoption decision "
          "(subset cannot set the sheet-wide scale).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
