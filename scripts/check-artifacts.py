#!/usr/bin/env python3
"""Is what we SERVE what this code would produce?

    docker exec sprite_generator python /app/scripts/check-artifacts.py

WHY THIS EXISTS

On 2026-08-27 a level_band fix was committed, its 33 smoke cases passed, and
the region something2 downloads still had three level-1 worlds. The code was
right, the tests were right, and the artifact was wrong - because the process
serving it had not been restarted and was still running the pre-fix module.

Nothing in the suite could have caught that. Every smoke script is run through
`docker exec python /app/scripts/...`, which starts a fresh interpreter and
imports from disk, so it always tests the code as written and never the code as
served. The suite and the service can disagree indefinitely and both look
green. See `.ai/decisions/0008` D12.

WHAT THIS CHECKS

Each stored region is REGENERATED from its own stored parameters and compared
to the artifact on disk. A mismatch means the artifact predates the current
code: someone downloading it gets something this repository would no longer
produce, and no amount of passing tests says otherwise.

This is deliberately not a smoke test. It has no fixtures and asserts nothing
about correctness - it asserts only that the thing on disk and the thing in the
code are the same thing. Run it after deploying, and before telling anyone a
fix has landed.
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, "/app")

import world_gen  # noqa: E402
import worlds  # noqa: E402


def diff_summary(stale: dict, fresh: dict) -> list:
    """What actually differs, in terms a person can act on."""
    out = []
    sw, fw = stale.get("worlds", []), fresh.get("worlds", [])

    if len(sw) != len(fw):
        out.append(f"world count {len(sw)} -> {len(fw)}")
        return out

    for a, b in zip(sw, fw):
        keys = set(a) | set(b)
        for k in sorted(keys):
            if a.get(k) != b.get(k):
                out.append(f"{a.get('key', '?')}.{k}: "
                           f"{a.get(k, '(absent)')!r} -> {b.get(k, '(absent)')!r}")
    if stale.get("links") != fresh.get("links"):
        out.append("links differ")
    return out


def main() -> int:
    pattern = os.path.join(worlds.WORLDS_DIR, "*.map.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"no regions in {worlds.WORLDS_DIR} - nothing to check")
        return 0

    stale_count = 0
    for path in paths:
        name = os.path.basename(path)[: -len(".map.json")]
        gen_path = os.path.join(worlds.WORLDS_DIR, f"{name}.gen.json")

        if not os.path.exists(gen_path):
            # Generated before the sidecar existed, so it cannot be regenerated
            # and cannot be compared. Said out loud rather than skipped
            # silently, because "not checked" is not "fine".
            print(f"  ?     {name}  - no stored params, cannot verify")
            continue

        with open(path, "r", encoding="utf-8") as fh:
            served = json.load(fh)
        with open(gen_path, "r", encoding="utf-8") as fh:
            params = json.load(fh)

        try:
            fresh = world_gen.plan_region(name, **worlds._plan_kwargs(params))
        except Exception as e:
            stale_count += 1
            print(f"  FAIL  {name}  - cannot regenerate: {type(e).__name__}: {e}")
            continue

        if served == fresh:
            print(f"  ok    {name}  - artifact matches the current code")
            continue

        stale_count += 1
        diffs = diff_summary(served, fresh)
        print(f"  STALE {name}  - the served artifact is NOT what this code "
              f"produces ({len(diffs)} difference(s))")
        for d in diffs[:12]:
            print(f"          {d}")
        if len(diffs) > 12:
            print(f"          ... and {len(diffs) - 12} more")

    print()
    if stale_count:
        print(f"{stale_count} region(s) stale. Regenerate them - a passing test "
              f"suite says nothing about what is being served.")
    else:
        print(f"{len(paths)} region(s) checked, all match the current code.")
    return 1 if stale_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
