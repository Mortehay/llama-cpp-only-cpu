#!/usr/bin/env python3
"""Can the grid-locked references be repaired into entity training data?

WHY THIS EXISTS

`audit-entity-refs.py` says that of 483 references, 36 are real pixel art and 23
of those are not blocked as cutouts - and that 21 of the 23 fail on ONE defect,
"background is flat, not transparent", which is the single defect
`remove_background` exists to fix. That is a claim about what a repair WOULD do.
This runs the repair and reports what it actually did.

The difference matters. "Recoverable in principle" was asserted in the findings
note three times before anything keyed a single file. The audit measures the
BACKDROP - how much of the frame a flood fill would take - and never looks at
what is left standing afterwards. An image can be 60% keyable background and
still be useless once keyed: the subject can be border-connected and leave with
it, a fused pedestal stays, a drop shadow survives as a stray.

So every file here is re-measured AFTER the repair by the same `measure` and
`judge` the audit uses, and the verdict that counts is the one on the repaired
file.

IT USES THE PIPELINE'S OWN FUNCTIONS, NOT COPIES

`remove_background` is lifted out of `tasks.py` by source text with `ast` - the
same trick as test-audit-mirrors-cutout.py and for the same reason, that a
hand-transcribed copy agrees with whatever misunderstanding wrote it - and
`strip_ground_patch` is imported from `pixelate`. A private reimplementation
would repair the training data to a standard the pipeline does not hold at
inference, which is the mismatch this whole investigation is about.

TWO STAGES, REPORTED SEPARATELY

    stage 1   remove_background       the flat backdrop becomes alpha
    stage 2   + strip_ground_patch    the pedestal, fused to the subject's feet
                                      and therefore invisible to stage 1, is cut

Stage 2 is geometry surgery and it can amputate a legitimately wide base, which
is why `require_legs=True` is passed and why the two are never collapsed into
one number. A file already clean after stage 1 is never taken to stage 2, and a
stage-2 cut that does not reduce the finding count is rolled back rather than
kept as a silent amputation.

NEITHER STAGE MAKES A FILE FIT TO TRAIN ON BY ITSELF. This produces a candidate
set to LOOK AT, and writes a contact sheet for exactly that. `measure` cannot
see a subject that was keyed in half, because half a subject is still a centred
opaque blob.

Usage:
    python3 scripts/recover-grid-refs.py \
        --dir images --grid-list images/_audit/grid483.txt \
        --out images/recovered/entity \
        --contact-sheet images/_audit/recovered_entity \
        --markdown images/_audit/recovered_entity.md
"""

import argparse
import ast
import importlib.util
import os
import re
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = os.path.join(HERE, "..", "src", "sprite_generator", "tasks.py")
PIXELATE_DIR = os.path.join(HERE, "..", "src", "sprite_generator")


# ACCEPTANCE TEST ON THE REPAIRED FILE, and the reason the audit needs one.
#
# `judge` passed two repaired files as clean that still carry most of a solid
# rectangular backdrop - `ref_core_17e21a84ac41` (a bush on a grey slab) and
# `ref_core_e618100f0726` (a figure on a green card with "KING" lettered into
# it). Both were caught by eye on the contact sheet, by nothing else.
#
# They pass because the audit's thresholds were set for references that have
# NOT been keyed yet. `ALPHA_IN_USE_PCT` is 0.02, so keying a 2% margin off the
# edges is enough for an image to read as "has alpha in use"; the full-bleed
# rule then needs coverage > 0.85 AND border > 0.75, and a backdrop covering
# two thirds of the frame clears neither. Nothing else looks at the border.
#
# After a SUCCESSFUL cutout the border must be empty, so the border is the
# discriminator here even though it is not one on an un-keyed reference. The
# measured split over the 22 repaired files is not a close call:
#
#     0.000 x16,  0.005,  0.008,  0.051   |  0.359,  0.395,  0.550
#
# A threshold anywhere in 0.10-0.30 gives the same answer, which is why 0.10 is
# a reading of the data and not a fit to the two files that prompted it.
#
# A high value has two possible causes and both disqualify: the backdrop
# survived the flood fill, or the subject genuinely runs off the frame - and an
# entity with no air around it is not what this training set is for.
BORDER_AFTER_KEY = 0.10


class _Logger:
    def __getattr__(self, _):
        return lambda *a, **kw: None


def backdrop_survived(m):
    """Reason this repaired file is not a cutout, or None."""
    if m["border_pct"] > BORDER_AFTER_KEY:
        return ("backdrop survived the cut - {:.0%} of the border is still "
                "opaque after keying, so what is left is the subject ON its "
                "panel, not the subject".format(m["border_pct"]))
    return None


def load_remove_background(path=TASKS):
    """Execute the real `remove_background` out of tasks.py, by source text."""
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef)
               and n.name == "remove_background"), None)
    if fn is None:
        sys.exit("remove_background not found in %s - it was renamed or moved"
                 % path)
    body = ast.get_source_segment(src, fn)
    if body is None:
        sys.exit("could not recover remove_background's source text from %s"
                 % path)
    ns = {"Image": Image, "np": np, "ndimage": ndimage, "logger": _Logger(),
          # Only reached with keep_largest=True, which this repair never sets:
          # dropping every blob but the biggest is right at generation time and
          # wrong on a reference, where a detached part may be the subject.
          "_isolate_largest_sprite": lambda arr: arr}
    exec(compile(body, path, "exec"), ns)
    return ns["remove_background"]


def load_audit():
    path = os.path.join(HERE, "audit-entity-refs.py")
    spec = importlib.util.spec_from_file_location("audit_entity_refs", path)
    if spec is None or spec.loader is None:
        sys.exit("cannot load %s" % path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GRID_LINE = re.compile(
    r"^\s*(?P<file>ref_\w+\.png)\s+(?P<w>\d+)x(?P<h>\d+)\s+grid\s+(?P<grid>\d+)"
    r"\s+err\s+(?P<err>[\d.]+)\s+(?P<colours>\d+)\s+colours")


def read_grid_list(path):
    """Parse the block-error sweep's own output rather than a retyped list."""
    rows = []
    for line in open(path, encoding="utf-8"):
        mo = GRID_LINE.match(line)
        if mo:
            rows.append({"file": mo.group("file"),
                         "grid": int(mo.group("grid")),
                         "err": float(mo.group("err")),
                         "colours": int(mo.group("colours"))})
    if not rows:
        sys.exit("no grid rows parsed from %s - the sweep's output format "
                 "changed, and silently recovering zero files would look like "
                 "a result" % path)
    return rows


def read_rejects(path):
    """By-eye rejections, as a reviewable input rather than a claim in prose.

    Three of the nine files that passed every automatic check are unusable, and
    all three were caught only by magnifying them: a figure on a black bar, a
    dragon on rock slabs with white background ENCLOSED by its own silhouette,
    a frog fused to a ground blob. `measure` sees one centred opaque blob in
    each case and has no way to ask how much of that blob is the entity.

    The enclosed-background one is not a threshold that could be tightened.
    `remove_background` takes border-connected pixels only, so a hole surrounded
    by the subject is unreachable by construction, and no flood fill will ever
    find it.

    Keeping these in a file means the judgement is dated, attributable and
    re-checkable, and the file count does not quietly change when someone
    disagrees with one line of it.
    """
    if not path or not os.path.exists(path):
        return {}
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, reason = line.partition(" ")
        out[name.strip()] = reason.strip() or "rejected by eye, no reason given"
    return out


def short(reasons):
    """First clause of each reason, for a table cell."""
    return "; ".join(r.split(" - ")[0] for r in reasons) if reasons else "clean"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="./images")
    p.add_argument("--grid-list", default="./images/_audit/grid483.txt")
    p.add_argument("--out", default="./images/recovered/entity")
    p.add_argument("--contact-sheet", default="")
    p.add_argument("--markdown", default="")
    # Under .ai/specs, not images/_audit, because `images/*` is gitignored and
    # this is a curation decision rather than an artifact. An ignored rejects
    # file would vanish on a fresh clone and the count would silently go back up
    # to nine with no record that three files had ever been rejected.
    p.add_argument("--rejects",
                   default=os.path.join(HERE, "..", ".ai", "specs",
                                        "entity-cutout", "entity_rejects.txt"),
                   help="by-eye rejections applied after the automatic checks")
    a = p.parse_args()
    rejects = read_rejects(a.rejects)

    remove_background = load_remove_background()
    audit = load_audit()
    sys.path.insert(0, PIXELATE_DIR)
    from pixelate import strip_ground_patch  # noqa: E402

    rows = read_grid_list(a.grid_list)
    print("grid list: %d files" % len(rows))
    os.makedirs(a.out, exist_ok=True)

    results, kept = [], []
    for r in rows:
        src = os.path.join(a.dir, r["file"])
        if not os.path.exists(src):
            results.append(dict(r, before="FILE MISSING", after="-",
                                stage="-", out=""))
            continue

        m0 = audit.measure(src)
        # BOTH lists, and the review one is the point. "background is flat, not
        # transparent" - the single defect this repair exists to fix - is a
        # REVIEW finding, not a blocking one, because a keyable backdrop is
        # recoverable and so was never a reason to reject an image outright.
        # Gating the repair on `blocking` alone made it a no-op on all 23 files
        # it was written for, and printed "clean already: 23" while nothing had
        # been keyed. A file is training-ready only when both lists are empty.
        b0, v0 = audit.judge(m0)
        eye = rejects.get(r["file"])

        if not b0 and not v0 and not backdrop_survived(m0) and not eye:
            results.append(dict(r, before="clean", after="clean",
                                stage="none needed", out=src,
                                removable=m0["removable_bg_pct"],
                                alpha_after=m0["alpha_transparent_pct"]))
            kept.append(src)
            continue

        # Nothing keying can do about a 64x96 sprite or a sheet of six icons.
        # Say so instead of writing a repaired copy that scores the same.
        if b0:
            results.append(dict(r, before=short(b0 + v0), after=short(b0 + v0),
                                stage="not repairable", out="",
                                removable=m0["removable_bg_pct"],
                                alpha_after=m0["alpha_transparent_pct"]))
            continue

        img = Image.open(src)
        out_path = os.path.join(a.out, r["file"])

        keyed = remove_background(img)
        keyed.save(out_path)
        m1 = audit.measure(out_path)
        b1, v1 = audit.judge(m1)
        stage, mN, fN = "remove_background", m1, b1 + v1

        if fN:
            cut = strip_ground_patch(keyed, require_legs=True)
            cut.save(out_path)
            m2 = audit.measure(out_path)
            b2, v2 = audit.judge(m2)
            if len(b2 + v2) < len(fN):
                stage, mN, fN = "+ strip_ground_patch", m2, b2 + v2
            else:
                # The cut bought nothing. Leave stage 1 on disk rather than a
                # file that was amputated for no gain.
                keyed.save(out_path)

        # Applied AFTER both stages, because it is a statement about the
        # repaired file and there is nothing to say about it before the cut.
        left = backdrop_survived(mN)
        if left:
            fN = fN + [left]
        if eye:
            fN = fN + [eye]

        if not fN:
            kept.append(out_path)

        results.append(dict(r, before=short(b0 + v0), after=short(fN),
                            stage=stage, out=out_path,
                            removable=m0["removable_bg_pct"],
                            alpha_after=mN["alpha_transparent_pct"]))

    width = max(len(r["file"]) for r in results)
    print()
    print("%-*s  %-36s  %-20s  %s"
          % (width, "file", "before", "stage", "after"))
    print("-" * (width + 82))
    for r in results:
        print("%-*s  %-36s  %-20s  %s"
              % (width, r["file"], r["before"][:36],
                 (r.get("stage") or "-")[:20], r["after"][:40]))

    already = [r for r in results if r.get("stage") == "none needed"]
    repaired = [r for r in results
                if r["after"] == "clean" and r.get("stage") in
                ("remove_background", "+ strip_ground_patch")]
    unrepairable = [r for r in results if r.get("stage") == "not repairable"]
    partial = [r for r in results
               if r["after"] != "clean" and r.get("stage") in
               ("remove_background", "+ strip_ground_patch")]
    print()
    # A rejects file that names something the sweep no longer produces has gone
    # stale, and a stale line silently shrinks nothing - it just stops applying.
    # Say so rather than let the count drift.
    stale = sorted(set(rejects) - {r["file"] for r in results})
    if stale:
        print()
        print("STALE in %s - not in the grid list, so these lines do nothing:"
              % a.rejects)
        for n in stale:
            print("  %s" % n)

    print()
    print("rejected by eye:            %d" % len(
        [r for r in results if r["file"] in rejects]))
    print("clean already:              %d" % len(already))
    print("clean after repair:         %d" % len(repaired))
    print("repaired, still not clean:  %d" % len(partial))
    print("not repairable by keying:   %d" % len(unrepairable))
    print("TOTAL USABLE:               %d" % len(kept))
    print()
    print("NOT YET TRAINING DATA. Every recovered file needs an eye on it - "
          "`measure` cannot tell a subject from half a subject, and a subject "
          "that was border-connected leaves with its own backdrop.")

    if a.contact_sheet and kept:
        out = a.contact_sheet
        if not os.path.splitext(out)[1]:
            out += ".png"
        audit.contact_sheet(kept, out)
        print("contact sheet: %s" % out)

    if a.markdown:
        with open(a.markdown, "w", encoding="utf-8") as fh:
            fh.write("# Grid-locked references after repair\n\n")
            fh.write("Written by `scripts/recover-grid-refs.py`. The `after` "
                     "column is a re-measurement of the REPAIRED file, not a "
                     "prediction.\n\n")
            fh.write("| file | grid err | before | stage | after |\n")
            fh.write("|---|---|---|---|---|\n")
            for r in results:
                fh.write("| `%s` | %.2f | %s | %s | %s |\n"
                         % (r["file"], r["err"], r["before"],
                            r.get("stage") or "-", r["after"]))
        print("markdown: %s" % a.markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
