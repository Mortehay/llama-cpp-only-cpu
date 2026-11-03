#!/usr/bin/env python3
"""Does the border check fire on cut heads and stay quiet on raised weapons?

WHY THIS EXISTS

This is the only framing signal on this problem that works, and it has one
way to become worthless: firing on frames that are correct. On the first
`wide` attack sheet the only three top-touching cells were `n_0`, `ne_0` and
`se_0` - all "weapon raised overhead", all complete figures with intact heads,
the sword blade leaving the frame. An unconditional check fails those forever
and no redraw can clear them.

Width cannot separate the two cases: sword tips measured top_w 0.092-0.158
against genuine mild crown clips at 0.018-0.210. So the exemption has to come
from the pose text, and these checks exist to keep it correct.

SEVEN CHECKS, all CPU-only on synthetic masks plus the real labelled images
when they are present. No card, no model.

WHAT IS DEMONSTRATED RATHER THAN ASSERTED (2026-11-03). Checks 2 and 4 call
the real `is_cut` on the SAME mask and differ only in the pose argument,
returning True then False - so the exemption is load-bearing by construction,
and a version of `is_cut` that ignored `pose` would redden check 4. Check 5
does the same for `reference`. No file mutation is needed and none is claimed:
an earlier test in this session asserted against its own reconstruction of the
production code and would have stayed green with the guard deleted, so these
checks deliberately call the shipped function and nothing else.

This suite found one real defect when first run: `"weapon tipped back behind
the head, elbows flared wide"` - attack front 1 and back 1, where the weapon
is STILL elevated after the overhead raise - was not matched by OVERHEAD, so
those two frames would have been reported cut. Fixed by adding `behind the
head` and `top of the swing`.

There is no bottom-border rule to test, by design; check 7 asserts the module
exposes none.

    docker compose run --rm --entrypoint python sprite-worker \
        /app/scripts/test-border-qc.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import border_qc  # noqa: E402

FAILS: list[str] = []
IMAGES = "/app/images"


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if detail:
        print(f"        {detail}")
    if not ok:
        FAILS.append(label)


def mask(top: int, width: int, flat: int = 20, h: int = 512, w: int = 512):
    """A crude figure: `flat` px wide at its top row, widening below."""
    m = np.zeros((h, w), dtype=np.uint8)
    cx = w // 2
    m[top:top + 40, cx - flat // 2:cx + flat // 2] = 255
    m[top + 40:h - 10, cx - width // 2:cx + width // 2] = 255
    return m


def main() -> int:
    print("\n== 1. a figure clear of the top border is not cut ==")
    m = border_qc.measure(mask(top=12, width=200))
    check(m is not None and not m["top_touch"], "top_touch is False at y0=12",
          f"{m}")
    check(not border_qc.is_cut(m), "is_cut is False")

    print("\n== 2. a figure reaching row 0 is cut ==")
    m0 = border_qc.measure(mask(top=0, width=200, flat=90))
    check(m0["top_touch"] and m0["top_y0"] == 0, "top_touch is True at y0=0")
    check(border_qc.is_cut(m0), "is_cut is True with no pose text")

    print("\n== 3. an empty mask is 'not judged', not 'passed' ==")
    check(border_qc.measure(np.zeros((32, 32), dtype=np.uint8)) is None,
          "measure returns None for an empty mask")
    check(not border_qc.is_cut(None), "is_cut(None) is False, never raises")

    print("\n== 4. an overhead pose is exempt on a CELL ==")
    overhead = "weapon raised overhead in both hands, held vertical, leaning back"
    check(border_qc.overhead_pose(overhead), "overhead_pose matches 'overhead'")
    check(not border_qc.is_cut(m0, pose=overhead),
          "a touching cell with an overhead pose is NOT reported cut",
          "this is the sword tip on attack frame 0")
    for p in ("weapon tipped back behind the head, elbows flared wide",
              "both arms raised high to shield the head",
              "sword held above the head"):
        check(border_qc.overhead_pose(p), f"matches {p[:38]!r}")

    print("\n== 5. a non-overhead pose is NOT exempt, and a REFERENCE never is ==")
    walk = "left leg forward heel down, right arm swung forward low beside the hip"
    check(not border_qc.overhead_pose(walk), "a walk pose is not overhead")
    check(border_qc.is_cut(m0, pose=walk),
          "a touching walk cell IS reported cut")
    check(border_qc.is_cut(m0, pose=overhead, reference=True),
          "a touching REFERENCE is cut even with overhead-looking text",
          "a turnaround has no pose; nothing is ever overhead in one")

    print("\n== 6. top_w is a severity score and is subject-relative ==")
    narrow = border_qc.measure(mask(top=0, width=200, flat=20))
    wide = border_qc.measure(mask(top=0, width=200, flat=120))
    check(narrow["top_w"] < wide["top_w"],
          "a wider top row scores higher",
          f"{narrow['top_w']} < {wide['top_w']}")
    check(border_qc.is_cut(narrow) and border_qc.is_cut(wide),
          "but BOTH are cut - width does not decide",
          "sword tips 0.092-0.158 overlap crown clips 0.018-0.210")

    print("\n== 7. there is no bottom-border rule, deliberately ==")
    src = open(border_qc.__file__, encoding="utf-8").read()
    check("bot_touch" not in src and "bottom_touch" not in src,
          "the module exposes no bottom-border verdict",
          "a correct standing figure has its feet ON the last row")
    check(border_qc.ENFORCE is False or
          os.environ.get("BORDER_QC_ENFORCE", "0") not in ("0", ""),
          "ENFORCE defaults off unless explicitly set",
          f"ENFORCE={border_qc.ENFORCE}")

    print("\n== 8. the real labelled images, when present ==")
    cases = [
        ("30d6823c-061a-440d-be7e-e4bcea83d76c", "cut_attack_ne_0.png",
         overhead, False, "sword tip, head intact - must NOT fire"),
        ("7f9fc8f1-cfa7-4f2d-9220-7809e7a629b1", "cut_idle_s_0.png",
         "standing still, arms at sides", True, "front crown clip - must fire"),
        ("7f9fc8f1-cfa7-4f2d-9220-7809e7a629b1", "cut_idle_n_0.png",
         "standing still, arms at sides", False, "intact - must not fire"),
    ]
    try:
        from PIL import Image
        for job, name, pose, expect, note in cases:
            p = os.path.join(IMAGES, f"_job_{job}", name)
            if not os.path.exists(p):
                print(f"  SKIP  {name} (not on this box)")
                continue
            a = np.asarray(Image.open(p).convert("RGBA"))[..., 3]
            mm = border_qc.measure(a)
            got = border_qc.is_cut(mm, pose=pose)
            check(got == expect, f"{name}: {note}",
                  f"y0={mm['top_y0']} top_w={mm['top_w']} -> cut={got}")
    except ImportError:
        print("  SKIP  PIL unavailable")

    print("\nFAILURES:", "none" if not FAILS else "")
    for f in FAILS:
        print("  -", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
