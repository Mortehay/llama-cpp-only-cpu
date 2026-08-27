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

try:
    import world_gen  # noqa: E402
    import worlds  # noqa: E402
except ModuleNotFoundError as e:
    # This is container-only: it imports the service's modules from /app and
    # needs DB_URL and the images volume. Run from a host shell it died on
    # `ModuleNotFoundError: world_gen`, which is true and tells nobody why.
    #
    # Said out loud because a peer's equivalent check SKIPPED politely on a
    # path where it could have run, and an honest skip reads like a check that
    # worked. A crash is better than that and a clear crash is better still.
    raise SystemExit("\n".join([
        f"check-artifacts runs inside the service container, not on the "
        f"host ({e}).",
        "    make check-artifacts",
        "  or",
        "    docker exec sprite_generator python "
        "/app/scripts/check-artifacts.py",
    ]))

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


def check_served_models() -> tuple[int, int]:
    """Does the RUNNING api describe the models this code defines?

    The rest of this file compares artifacts on disk to the code. This asks the
    live process directly, which is a different question and the one nothing
    here could answer: `/openapi.json` is the server describing its own models,
    so it reports what is SERVED rather than what is importable.

    Why it exists: `Terrain` gained a `walkable` field, 24 cases passed, and the
    served schema did not have it. Every suite here runs in a fresh interpreter
    and imports from disk, so not one of them could have noticed - and neither
    could the two halves above, which read files.

    REPORT THE DIFFERENCE; DO NOT NAME A MECHANISM FOR IT. When I hit this I
    concluded the process was stale, restarted it, and the disagreement went
    away - which felt like confirmation and was not. `uvicorn` reloads here in
    about seven seconds (StatReload polls mtimes; drvfs defeating inotify does
    not defeat polling), and afterwards I could not reproduce the staleness by
    any route. The reading was right and the cause I attached to it was
    invented. So this prints what differs and stops.

    Retries once before skipping. There is a ~7s window after any edit to a
    watched file where the API is genuinely unreachable rather than slow, and
    that is exactly when someone runs this - a skip that reads like a pass is
    the failure this whole file is about.

    HOW FAR THIS IS VERIFIED, since the rest of this file is about not
    overclaiming:

      - the SKIP branch fired for real. The first version asked localhost,
        which is the API under `docker exec` and is the throwaway container
        itself under `docker compose run` - it skipped honestly instead of
        passing, and that is how the missing host was found.
      - the DIFFERS branch HAS NEVER FIRED AND WE COULD NOT MAKE IT. Creating a
        served-vs-code gap means editing a model and beating the reload, and
        the reload wins - by the time `docker exec` has started a python and
        imported, the API already serves the new field. TWO sessions tried
        independently, by different routes, and neither opened the window.

    Phrased that way on purpose. "Unproven" invites the reader to assume it
    works; "we tried twice and could not make it fail" tells them what they are
    actually relying on - set arithmetic over two field lists, readable and
    never exercised in anger.

    It may be that on this box the window cannot be opened by a file edit at
    all, which is a claim about StatReload's poll interval against process
    startup time and is testable if anyone ever needs it. That same failure to
    reproduce is what leaves the incident behind this check unexplained.
    """
    import json as _json
    import time
    import urllib.error
    import urllib.request

    # Two hosts, because this file is run two ways and the first attempt at it
    # only worked one of them. `docker exec sprite_generator` shares the API's
    # network namespace, so localhost answers. `docker compose run --rm` - what
    # the Makefile does - starts a SIBLING container where localhost is itself,
    # and the API is reachable only by its name on the compose network.
    #
    # It skipped honestly rather than passing, which is the point of the skip.
    # It was also useless, which is the point of trying both.
    hosts = ([os.environ["API_URL"]] if os.environ.get("API_URL") else
             ["http://sprite_generator:8001", "http://localhost:8001"])

    served, used = None, None
    for attempt in (1, 2):
        for host in hosts:
            try:
                with urllib.request.urlopen(host + "/openapi.json",
                                            timeout=8) as fh:
                    served, used = _json.load(fh), host
                break
            except (urllib.error.URLError, OSError, ValueError):
                continue
        if served is not None:
            if attempt == 2:
                print("  --    api reachable on retry - that was a reload, "
                      "not a dead API")
            break
        if attempt == 1:
            # A reload takes about 7s and the API is genuinely unreachable for
            # it, not slow. That window is exactly when someone runs this.
            time.sleep(9)

    if served is None:
        print(f"  SKIP  no api answered at {' or '.join(hosts)} after a retry. "
              f"NOT CHECKED - this says nothing about what is served.")
        return 0, 0
    print(f"  --    asking {used}")

    schemas = (served.get("components") or {}).get("schemas") or {}

    import maps  # noqa: E402

    pairs = [("MapSpec", maps.MapSpec), ("Terrain", maps.Terrain),
             ("Scatter", maps.Scatter)]

    bad = 0
    for name, model in pairs:
        if name not in schemas:
            bad += 1
            print(f"  BROKEN {name} - the running api serves no such schema")
            continue
        live = set((schemas[name].get("properties") or {}))
        here = set(model.model_fields)
        missing, extra = sorted(here - live), sorted(live - here)
        if missing or extra:
            bad += 1
            print(f"  DIFFERS {name} - served and code disagree")
            if missing:
                print(f"          this code defines, the api does not serve: "
                      f"{missing}")
            if extra:
                print(f"          the api serves, this code does not define: "
                      f"{extra}")
        else:
            print(f"  ok    {name}  - {len(here)} field(s), served matches code")

    return bad, len(pairs)


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

    print()
    served_bad, served_n = check_served_models()
    print()
    if served_bad:
        print(f"{served_bad} model(s) differ between this code and the running "
              f"api. Something is serving different code - do not guess which; "
              f"the difference above is the finding.")
    elif served_n:
        print(f"{served_n} model(s) checked against the running api, all match.")
    else:
        print("NO models were checked against the api. This says nothing "
              "about what is served.")

    return 1 if (stale_count or broken or served_bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
