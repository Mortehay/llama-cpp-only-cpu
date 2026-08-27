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

IMAGES_DIR = "/app/images"


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


def check_maps() -> tuple[int, int]:
    """Are the served MAPS still whole?

    Maps get a weaker guarantee than regions and the difference is worth being
    explicit about. A region is regenerated from its stored params and diffed,
    which proves the artifact matches the current code. A map cannot be - its
    painting and its tiles cost GPU, so regenerating one to check it would cost
    more than the thing being checked.

    So this asks a different question: is the artifact INTERNALLY WHOLE? The
    tilemap now names its terrain tiles, a promise the contract made from the
    start and the build only began keeping recently. Nothing verified those
    files still exist, and a map whose tiles were swept is a tilemap something2
    downloads and cannot draw.

    That is the same class as the region check without the same strength, and
    calling it 'verified' would be the overclaim this whole file exists to stop.

    Returns (broken, checked). Both, because they are different questions:
    zero broken out of zero checked is not a clean bill of health, and this
    function used to let `main` report the one as the other.
    """
    import json as _json

    import psycopg2
    import psycopg2.extras

    db = os.environ.get("DB_URL")
    if not db:
        print("  ?     maps - no DB_URL, cannot check")
        return 0, 0

    with psycopg2.connect(db) as conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT id, spec->>'name' AS name, sheet_path, atlas_path "
                    "FROM jobs WHERE kind = 'map' AND deleted = false "
                    "  AND status = 'done' ORDER BY created_at")
        rows = cur.fetchall()

    if not rows:
        print("  --    no finished maps to check")
        return 0, 0

    broken = 0
    for r in rows:
        name = r["name"] or str(r["id"])[:8]
        faults = []

        if not r["sheet_path"] or not os.path.exists(r["sheet_path"]):
            faults.append("picture missing")
        if not r["atlas_path"] or not os.path.exists(r["atlas_path"]):
            faults.append("tilemap missing")
            broken += 1
            print(f"  BROKEN {name}  - {', '.join(faults)}")
            continue

        try:
            with open(r["atlas_path"], "r", encoding="utf-8") as fh:
                tm = _json.load(fh)
        except Exception as e:
            broken += 1
            print(f"  BROKEN {name}  - tilemap will not parse: {e}")
            continue

        terrains = tm.get("terrains") or []
        for t in terrains:
            # The contract promises these; something2 draws the grid from them.
            url = t.get("tile")
            if not url:
                faults.append(f"terrain {t.get('name')!r} names no tile")
            elif not os.path.exists(os.path.join(IMAGES_DIR,
                                                 os.path.basename(url))):
                faults.append(f"tile for {t.get('name')!r} is gone ({url})")

        road = tm.get("road_tile")
        if road and not os.path.exists(os.path.join(IMAGES_DIR,
                                                    os.path.basename(road))):
            faults.append(f"road tile is gone ({road})")

        grid = tm.get("layers", {}).get("terrain") or []
        if grid:
            top = max(max(row) for row in grid)
            if top >= len(terrains):
                faults.append(f"grid names terrain id {top} but only "
                              f"{len(terrains)} are declared")
            h, w = len(grid), len(grid[0])
            for e in tm.get("entities") or []:
                if not (0 <= e.get("x", -1) < w and 0 <= e.get("y", -1) < h):
                    faults.append(f"entity at ({e.get('x')},{e.get('y')}) is "
                                  f"outside the {w}x{h} grid")
                    break

        # `complete` and the placements have to agree, or a consumer trusts the
        # wrong one. This is the pair that disagreed for two days elsewhere.
        still_pending = {e.get("want") for e in (tm.get("entities") or [])
                         if e.get("status") == "pending" and e.get("want")}
        if bool(tm.get("complete")) and still_pending:
            faults.append(f"claims complete while {len(still_pending)} "
                          f"placement(s) are still pending")
        if not tm.get("complete") and not still_pending:
            faults.append("claims incomplete but nothing is pending")

        if faults:
            broken += 1
            print(f"  BROKEN {name}  ({len(faults)} fault(s))")
            for f in faults[:8]:
                print(f"          {f}")
        else:
            print(f"  ok    {name}  - {len(terrains)} terrains, all tiles "
                  f"present, grid and entities consistent")

    return broken, len(rows)


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

    print()
    broken, checked = check_maps()
    print()
    if broken:
        print(f"{broken} map(s) are not whole. Rebuild them - a tilemap whose "
              f"tiles are gone is one something2 downloads and cannot draw.")
    elif checked:
        print(f"{checked} map(s) checked, all whole.")
    else:
        # Nothing was checked, so nothing is known. Said plainly, because the
        # cheerful version of this line - "maps checked, all whole" over an
        # empty set - is the exact overclaim this file exists to prevent, and
        # it sat here reporting a pass over nothing.
        print("NO maps were checked. This says nothing about maps.")

    return 1 if (stale_count or broken) else 0


if __name__ == "__main__":
    raise SystemExit(main())
