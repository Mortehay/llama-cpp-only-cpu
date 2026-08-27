#!/usr/bin/env python3
"""The world API's handlers: create, edit, list, download, delete.

    docker exec sprite_generator python /app/scripts/smoke-world-api.py

`smoke-world-gen.py` covers the generator, which is pure. This covers the
ROUTER - the part with files, sidecars and lifecycle - because that is where an
edit can silently lose a previous edit or leave a stale sidecar behind for the
next region of the same name to inherit.

`auth.require` is stubbed. These assert what the handlers DO, not that they are
guarded; the guarding is one `auth.require` line per route and is visible by
reading them. Nothing here needs a token, so nothing here needs the user's.

Regions are written under a `_smoke-` prefix and removed at the end, including
after a failure.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, "/app")

import auth  # noqa: E402

auth.require = lambda *a, **k: None  # noqa: E731

import worlds  # noqa: E402
from worlds import WorldEdit, WorldSpec  # noqa: E402

# No leading underscore: `_slug` strips it, so a region named "_smoke-region"
# is stored and listed as "smoke-region" and every lookup by the original name
# would miss.
NAME = "smoke-region"
CASES = []


def case(name):
    def wrap(fn):
        CASES.append((name, fn))
        return fn
    return wrap


def make(**kw):
    args = dict(name=NAME, worlds=4, target_per_screen=4.0, size=128,
                author="rules", overwrite=True)
    args.update(kw)
    return worlds.create_world(WorldSpec(**args), None)


@case("create writes spec, preview and the parameter sidecar")
def _create():
    res = make()
    map_path, png_path, gen_path = worlds._paths(NAME)
    for p in (map_path, png_path, gen_path):
        assert os.path.exists(p), f"missing {os.path.basename(p)}"
    assert res["report"]["totals"]["worlds"] == 4
    assert res["seed_with"].endswith(NAME)
    return "3 files written"


@case("creating twice without overwrite is a 409, not a silent replace")
def _conflict():
    make()
    try:
        worlds.create_world(WorldSpec(name=NAME, worlds=4, target_per_screen=4.0,
                                      size=128, author="rules"), None)
    except Exception as e:
        assert getattr(e, "status_code", None) == 409, e
        return "409, and it points at PATCH"
    raise AssertionError("an existing region was silently replaced")


@case("editing the density target does NOT re-roll the biomes")
def _edit_keeps_biomes():
    before = make(target_per_screen=4.0)
    biomes_before = [w["biomes"] for w in before["spec"]["worlds"]]
    after = worlds.edit_world(NAME, WorldEdit(target_per_screen=12.0), None)
    biomes_after = [w["biomes"] for w in after["spec"]["worlds"]]

    assert biomes_before == biomes_after, "an edit redrew the region's character"
    assert (after["report"]["totals"]["max_per_screen"]
            > before["report"]["totals"]["max_per_screen"])
    assert after["changed"] == ["target_per_screen"], after["changed"]
    return (f"{before['report']['totals']['max_per_screen']} -> "
            f"{after['report']['totals']['max_per_screen']}/screen, biomes intact")


@case("edits accumulate instead of resetting each other")
def _edits_accumulate():
    make(target_per_screen=4.0)
    worlds.edit_world(NAME, WorldEdit(target_per_screen=12.0), None)
    grown = worlds.edit_world(NAME, WorldEdit(worlds=7), None)
    assert grown["report"]["totals"]["worlds"] == 7
    assert grown["params"]["target_per_screen"] == 12.0, \
        "growing the region forgot the earlier density edit"
    return "target survived a later resize"


@case("an edit naming nothing changes nothing")
def _empty_edit():
    made = make()
    same = worlds.edit_world(NAME, WorldEdit(), None)
    assert same["changed"] == [], same["changed"]
    assert same["spec"] == made["spec"], "an empty PATCH altered the region"
    return "no-op PATCH is a no-op"


@case("the listing says whether a region can be edited at all")
def _listing():
    make()
    row = [i for i in worlds.list_worlds(None)["items"] if i["name"] == NAME][0]
    assert row["editable"] is True and row["params"]["worlds"] == 4
    assert row["preview_url"].endswith("/preview.png")

    # A region from before sidecars existed: listable, downloadable, not
    # editable - and the PATCH must say so rather than 500.
    _, _, gen_path = worlds._paths(NAME)
    os.remove(gen_path)
    row = [i for i in worlds.list_worlds(None)["items"] if i["name"] == NAME][0]
    assert row["editable"] is False and row["params"] is None
    try:
        worlds.edit_world(NAME, WorldEdit(size=64), None)
    except Exception as e:
        assert getattr(e, "status_code", None) == 409, e
    else:
        raise AssertionError("patched a region with no stored parameters")
    return "editable flag tracks the sidecar; legacy regions 409"


@case("missing regions 404 on every route")
def _missing():
    for call in (lambda: worlds.get_world("_smoke-absent", False, None),
                 lambda: worlds.get_report("_smoke-absent", None),
                 lambda: worlds.edit_world("_smoke-absent", WorldEdit(size=64), None),
                 lambda: worlds.delete_world("_smoke-absent", None)):
        try:
            call()
        except Exception as e:
            assert getattr(e, "status_code", None) == 404, e
        else:
            raise AssertionError("a missing region did not 404")
    return "4 routes, all 404"


@case("a name that looks like a path is refused, not sanitised")
def _bad_name():
    # `_slug` alone would turn "../escape" into "escape" - safe, in that it
    # cannot leave WORLDS_DIR, but a lookup silently resolving to a DIFFERENT
    # region is its own bug. Both are refused.
    for bad in ("../escape", "a/b", "..", "x\\y"):
        try:
            worlds._paths(bad)
        except Exception as e:
            assert getattr(e, "status_code", None) == 400, (bad, e)
        else:
            raise AssertionError(f"{bad!r} was accepted as a region name")

    # An empty name is not path-like; it slugs to the "region" fallback. That
    # is a POST-body concern (min_length=2), not a filesystem one.
    root = os.path.realpath(worlds.WORLDS_DIR)
    for ok in ("Emerald Reach", "a b c", ""):
        for path in worlds._paths(ok):
            assert os.path.realpath(path).startswith(root), path
    return "path-like names 400; every slug stays inside WORLDS_DIR"


@case("delete removes the sidecar too")
def _delete():
    make()
    removed = worlds.delete_world(NAME, None)["deleted"]
    assert any(f.endswith(".gen.json") for f in removed), removed
    assert any(f.endswith(".map.json") for f in removed), removed
    for p in worlds._paths(NAME):
        assert not os.path.exists(p), f"{p} survived delete"
    return f"{len(removed)} files"


def cleanup():
    try:
        for p in worlds._paths(NAME):
            if os.path.exists(p):
                os.remove(p)
    except Exception:
        pass


def main() -> int:
    failed = 0
    try:
        for name, fn in CASES:
            try:
                print(f"  ok    {name}  ({fn()})")
            except AssertionError as e:
                failed += 1
                print(f"  FAIL  {name}\n        {e}")
            except Exception as e:
                failed += 1
                print(f"  ERROR {name}\n        {type(e).__name__}: {e}")
    finally:
        cleanup()
    print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
