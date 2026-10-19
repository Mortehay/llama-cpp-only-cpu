#!/usr/bin/env python3
"""The world API's handlers: create, edit, list, download, delete.

    docker exec sprite_generator python /app/scripts/smoke-world-api.py

`smoke-world-gen.py` covers the generator, which is pure. This covers the
ROUTER - the part with files, sidecars and lifecycle - because that is where an
edit can silently lose a previous edit or leave a stale sidecar behind for the
next region of the same name to inherit.

Since 2026-10-19 that includes the OTHER direction: a spec hand-authored in
something2's World Map Editor and PUT back whole. The cases that matter there
are the ones about losing work - a stale `If-Match`, a refused write that wrote
anyway, a regeneration over an admin's map - because every one of them destroys
something a human made and none of them raises anything on its own.

`auth.require` is stubbed. These assert what the handlers DO, not that they are
guarded; the guarding is one `auth.require` line per route and is visible by
reading them. Nothing here needs a token, so nothing here needs the user's.

Regions are written under a `_smoke-` prefix and removed at the end, including
after a failure.
"""

from __future__ import annotations

import json
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
    """A fresh region, whatever the last case left behind.

    `discard_hand_edits` is in the DEFAULTS because the fixture's whole job is
    to start from a known generated region, and a case that PUT a hand-edited
    spec would otherwise poison every case after it with a 409 - which is the
    guard working, not a failure. The guard itself is proved by passing
    `discard_hand_edits=False` explicitly.
    """
    args = dict(name=NAME, worlds=4, target_per_screen=4.0, size=128,
                author="rules", overwrite=True, discard_hand_edits=True)
    args.update(kw)
    return worlds.create_world(WorldSpec(**args), None)


def body_of(res):
    """A handler that has to set a header returns a Response, not a dict."""
    return json.loads(res.body)


def spec_and_etag():
    """What a client editing a region sees: the spec, and what to send back."""
    res = worlds.get_world(NAME, False, None)
    return body_of(res), res.headers["etag"]


def copy_of(spec):
    return json.loads(json.dumps(spec))


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
    absent = {"name": "smoke-absent", "worlds": [{"key": "k"}], "links": []}
    for call in (lambda: worlds.get_world("_smoke-absent", False, None),
                 lambda: worlds.get_report("_smoke-absent", None),
                 lambda: worlds.edit_world("_smoke-absent", WorldEdit(size=64), None),
                 lambda: worlds.delete_world("_smoke-absent", None),
                 # Before If-Match, so an unknown region reads as 404 and not
                 # as a precondition the caller could go and fix.
                 lambda: worlds.put_spec("_smoke-absent", absent,
                                         '"' + "0" * 64 + '"', None)):
        try:
            call()
        except Exception as e:
            assert getattr(e, "status_code", None) == 404, e
        else:
            raise AssertionError("a missing region did not 404")
    return "5 routes, all 404"


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


@case("GET hands back the spec's sha256 as an ETag")
def _etag_header():
    make()
    spec, etag = spec_and_etag()
    map_path, _, _ = worlds._paths(NAME)
    assert etag == f'"{worlds._sha256_file(map_path)}"', etag
    assert spec["name"] == NAME, spec.get("name")
    # The download form carries it too. Without that, a client that TAKES the
    # file has no version to edit against and can only write blind.
    taken = worlds.get_world(NAME, True, None)
    assert taken.headers["etag"] == etag, taken.headers.get("etag")
    return f"{etag[:14]}...\" on both forms"


@case("a PUT of an unchanged spec keeps the bytes, the body and the ETag")
def _roundtrip():
    make()
    spec, etag = spec_and_etag()
    out = body_of(worlds.put_spec(NAME, spec, etag, None))
    assert out["name"] == NAME, out
    assert out["sha256"] == etag.strip('"'), out

    again, etag_again = spec_and_etag()
    assert again == spec, "a no-op PUT altered the spec"
    assert etag_again == etag, (etag, etag_again)
    return "byte-identical; ETag unchanged"


@case("a stale If-Match is a 409 and the stored spec is untouched")
def _stale_if_match():
    make()
    spec, etag = spec_and_etag()
    edited = copy_of(spec)
    edited["worlds"][0]["density"] = "swarm"

    res = worlds.put_spec(NAME, edited, '"' + "0" * 64 + '"', None)
    assert res.status_code == 409, res.status_code
    out = body_of(res)
    assert out["detail"] == "spec changed on the box", out
    # The CURRENT sha, not the one that was sent - a caller that cannot see
    # what it lost to has to GET again to find out.
    assert out["sha256"] == etag.strip('"'), out

    after, etag_after = spec_and_etag()
    assert after == spec, "a refused PUT wrote anyway"
    assert etag_after == etag, (etag, etag_after)
    return "409 carries the current sha256; the file is unchanged"


@case("a PUT with no If-Match is refused rather than written blind")
def _no_if_match():
    make()
    spec, _ = spec_and_etag()
    for header in (None, "", '""'):
        try:
            worlds.put_spec(NAME, spec, header, None)
        except Exception as e:
            assert getattr(e, "status_code", None) == 428, (header, e)
        else:
            raise AssertionError(f"If-Match {header!r} was accepted")
    assert worlds._hand_edit(NAME)["hand_edited"] is False
    return "428 for absent, empty and empty-quoted"


@case("the editor's graph, kind=edge and to_name survive a round trip")
def _editor_fields():
    make()
    spec, etag = spec_and_etag()
    keys = [w["key"] for w in spec["worlds"]]

    edited = copy_of(spec)
    # A dragged graph position, which wins over grid x 220 on their side and
    # means nothing at all here - which is the point: it must come back.
    edited["worlds"][0]["graph"] = [413, 77]
    # A compass doorway with an explicit edge and no grid-delta rule.
    edited["links"].append({"from": keys[0], "edge": "W", "kind": "edge",
                            "to": keys[1]})
    # And one that leaves the region entirely, by world NAME.
    edited["links"].append({"from": keys[1], "edge": "N", "kind": "edge",
                            "to_name": "Vale Crossing"})
    # An unknown key at the top level, standing in for the next field they add.
    edited["authored_by"] = "world-map-editor"

    out = body_of(worlds.put_spec(NAME, edited, etag, None))
    stored, _ = spec_and_etag()
    assert stored == edited, "a hand edit was altered on the way through"
    assert stored["worlds"][0]["graph"] == [413, 77], stored["worlds"][0]
    assert stored["authored_by"] == "world-map-editor"

    # And the box can still REPORT on it. A cross-region link has no `to`, so
    # before `report` read links with .get this was a KeyError - the endpoint
    # refusing exactly the specs it exists to accept.
    rep = worlds.get_report(NAME, None)
    ghosts = [p for p in rep["problems"] if "does not exist" in p]
    assert not ghosts, ghosts
    return (f"3 optional fields plus an unknown key, "
            f"{len(out['problems'])} problem(s) reported")


@case("a hand-edited region refuses a regeneration nobody asked for")
def _guard_hand_edits():
    make()
    spec, etag = spec_and_etag()
    worlds.put_spec(NAME, spec, etag, None)

    # BOTH doors. A guard on PATCH alone is not a guard: POST with overwrite
    # rebuilds the same region from the same parameters.
    for call in (lambda: worlds.edit_world(NAME, WorldEdit(size=64), None),
                 lambda: make(discard_hand_edits=False)):
        try:
            call()
        except Exception as e:
            assert getattr(e, "status_code", None) == 409, e
            assert "discard_hand_edits" in str(getattr(e, "detail", "")), e.detail
        else:
            raise AssertionError("a hand-edited region was regenerated")

    # Asked for by name, it rebuilds and the flag goes with it.
    out = worlds.edit_world(NAME, WorldEdit(size=64, discard_hand_edits=True),
                            None)
    assert out["spec"]["worlds"][0]["width"] == 64, out["spec"]["worlds"][0]
    assert worlds._hand_edit(NAME)["hand_edited"] is False
    assert not os.path.exists(worlds._edit_path(NAME))
    # The flag is not a generation parameter and must not be reported as one,
    # or it lands in the sidecar and the next PATCH "changes" it again.
    assert out["changed"] == ["size"], out["changed"]
    assert "discard_hand_edits" not in out["params"], out["params"]
    return "PATCH and overwrite both 409; discard rebuilds and clears"


@case("the listing and the report both say whether a human drew it")
def _flag_exposed():
    make()
    row = [i for i in worlds.list_worlds(None)["items"] if i["name"] == NAME][0]
    assert row["hand_edited"] is False and row["edited_at"] is None, row
    assert worlds.get_report(NAME, None)["hand_edited"] is False

    spec, etag = spec_and_etag()
    stamp = body_of(worlds.put_spec(NAME, spec, etag, None))["edited_at"]

    row = [i for i in worlds.list_worlds(None)["items"] if i["name"] == NAME][0]
    rep = worlds.get_report(NAME, None)
    assert row["hand_edited"] is True and row["edited_at"] == stamp, row
    assert rep["hand_edited"] is True and rep["edited_at"] == stamp, rep["edited_at"]
    # Still regenerable - `editable` tracks the sidecar, which a hand edit
    # leaves exactly where it was.
    assert row["editable"] is True, row
    return f"edited_at {stamp}"


@case("a body that is not this region is 422, and nothing is written")
def _validation():
    make()
    spec, etag = spec_and_etag()

    wrong_region = copy_of(spec)
    wrong_region["name"] = "some-other-region"
    no_worlds = {"name": NAME, "worlds": [], "links": []}
    keyless = {"name": NAME, "worlds": [{"biomes": ["Meadow"]}], "links": []}
    ghost_link = copy_of(spec)
    ghost_link["links"].append({"from": spec["worlds"][0]["key"], "edge": "N",
                                "to": "a_world_that_is_not_here"})
    duplicate = copy_of(spec)
    duplicate["worlds"].append(copy_of(spec["worlds"][0]))
    edge_without_edge = copy_of(spec)
    edge_without_edge["links"].append({"from": spec["worlds"][0]["key"],
                                       "kind": "edge",
                                       "to": spec["worlds"][1]["key"]})

    for bad in (wrong_region, no_worlds, keyless, ghost_link, duplicate,
                edge_without_edge, ["not", "a", "spec"]):
        try:
            worlds.put_spec(NAME, bad, etag, None)
        except Exception as e:
            assert getattr(e, "status_code", None) == 422, (e, type(bad))
        else:
            raise AssertionError("an unusable spec was stored")

    after, etag_after = spec_and_etag()
    assert after == spec and etag_after == etag, "a 422 wrote anyway"
    assert worlds._hand_edit(NAME)["hand_edited"] is False, \
        "a refused PUT marked the region hand-edited"
    return "7 shapes refused, region untouched, flag unset"


@case("a region stored under a slug accepts its own titled spec back")
def _titled_name():
    # Created as "Smoke Region", so the file is `smoke-region.map.json` while
    # the spec inside says "Smoke Region". Comparing the body's name to the
    # path LITERALLY would refuse this region its own spec - which is the
    # round trip the endpoint exists for - so the comparison is by slug.
    make(name="Smoke Region")
    spec, etag = spec_and_etag()
    assert spec["name"] == "Smoke Region", spec["name"]
    out = body_of(worlds.put_spec(NAME, spec, etag, None))
    assert out["name"] == NAME, out
    return "title in the body, slug in the path"


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
        for p in worlds._paths(NAME) + (worlds._edit_path(NAME),):
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
