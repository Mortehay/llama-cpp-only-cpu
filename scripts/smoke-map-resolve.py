#!/usr/bin/env python3
"""The prop resolver under test. No model, no GPU, no database.

Run inside the API container:

    docker exec sprite_generator python /app/scripts/smoke-map-resolve.py

`resolve_map_props` has two halves. The half that spends the GPU is one call
and cannot be tested without a card. The half that decides what a map SAYS -
which placements found art, what stays pending, whether `complete` flips, and
whether anything moved while it happened - is pure and is all of the risk.

The assertion that matters most is `entities_do_not_move`. Re-compositing after
a prop resolves must reuse the placements the build already made; a map whose
trees shuffled when a windmill finished would be a genuinely baffling bug to be
handed, and re-running the scatter would be deterministic from the seed yet
still able to drift if a grid round-tripped through JSON differently.

The second is `partial_stays_incomplete`: one want that will not generate must
not discard the ones that did, and must not let the map claim to be finished.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, "/app")

import map_tasks as mt  # noqa: E402
import map_geometry as mg  # noqa: E402

TERRAINS = [
    {"id": 0, "name": "grass", "color": "#4a7c3f", "walkable": True},
    {"id": 1, "name": "water", "color": "#2850c8", "walkable": False},
]

TILE_W, TILE_H = 32, 16
GRID = 8

CASES: list = []


def case(name):
    def wrap(fn):
        CASES.append((name, fn))
        return fn
    return wrap


class Images:
    """Point map_tasks at a scratch images directory for one test.

    The module reads IMAGES_DIR at call time, so swapping the global is enough
    and nothing has to be threaded through. Restored on the way out even when
    the test raises, or one failure would silently redirect every later case at
    the real /app/images.
    """

    def __enter__(self):
        self.dir = tempfile.mkdtemp(prefix="smoke-map-")
        self.was = mt.IMAGES_DIR
        mt.IMAGES_DIR = self.dir
        return self.dir

    def __exit__(self, *exc):
        mt.IMAGES_DIR = self.was
        shutil.rmtree(self.dir, ignore_errors=True)
        return False


def art(w=16, h=24, opaque=True) -> Image.Image:
    """A stand-in sprite: solid, with a transparent margin so it is a cutout."""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    if opaque:
        img.paste((200, 40, 40, 255), (2, 2, w - 2, h - 2))
    return img


def write_map(images_dir: str, job_id: str, entities, roads=False) -> dict:
    """A tilemap on disk exactly as `build_map_job` leaves one."""
    grid = np.zeros((GRID, GRID), dtype=int)
    grid[:, GRID // 2:] = 1

    for i, t in enumerate(TERRAINS):
        tile = Image.new("RGBA", (TILE_W, TILE_H),
                         (*mg.parse_color(t["color"]), 255))
        tile.putalpha(__import__("tile_geometry").diamond_mask(TILE_W, TILE_H))
        tile.save(mt._tile_path(job_id, i))

    road_grid = np.zeros((GRID, GRID), dtype=int)
    if roads:
        road_grid[2, :] = 1
        rt = Image.new("RGBA", (TILE_W, TILE_H), (180, 160, 120, 255))
        rt.putalpha(__import__("tile_geometry").diamond_mask(TILE_W, TILE_H))
        rt.save(mt._road_tile_path(job_id))

    tilemap = {
        "id": job_id,
        "complete": False,
        "pending": sorted({e["want"] for e in entities if e.get("want")}),
        "size": {"w": GRID, "h": GRID},
        "tile": {"w": TILE_W, "h": TILE_H},
        "terrains": [dict(t, tile=f"/images/map_{job_id[:12]}_t{i}.png")
                     for i, t in enumerate(TERRAINS)],
        "road_tile": (f"/images/map_{job_id[:12]}_road.png" if roads else None),
        "layers": {"terrain": grid.tolist(), "roads": road_grid.tolist()},
        "entities": entities,
        "props_job": "11111111-1111-1111-1111-111111111111",
    }
    mt._write_tilemap(mt._tilemap_path(job_id), tilemap)
    # A picture must already exist, because the resolver replaces it.
    mg.composite(grid, [Image.open(mt._tile_path(job_id, i)) for i in (0, 1)]
                 ).save(mt._picture_path(job_id))
    return tilemap


# --- the filing key -------------------------------------------------------


@case("a want files under a stable, restricted slug")
def _slug_shape():
    assert mt._slug("Old Windmill") == "old_windmill"
    assert mt._slug("  oak tree!! ") == "oak_tree"
    assert mt._slug("wolf/../../etc/passwd") == "wolf_etc_passwd"
    assert mt._slug("") == "prop"
    assert mt._slug("!!!") == "prop"
    assert len(mt._slug("x" * 200)) == 40
    return "punctuation, separators and emptiness all land somewhere safe"


@case("a slug cannot escape the images directory")
def _slug_contained():
    with Images() as d:
        for hostile in ("../../etc/passwd", "..", "a/b/c", "\\\\share\\x"):
            path = mt._library_path(hostile)
            assert os.path.dirname(path) == d, path
    return "a name is a filename here, never a path"


@case("two maps wanting the same thing share one file")
def _library_shared():
    with Images():
        assert mt._library_path("Oak Tree") == mt._library_path("oak tree")
    return "the second map pays no GPU"


# --- resolving one placement ---------------------------------------------


@case("a named asset is used before anything is generated")
def _asset_first():
    with Images() as d:
        art().save(os.path.join(d, "my_oak.png"))
        e = {"asset": "my_oak.png", "want": "oak tree", "x": 0, "y": 0,
             "layer": "props"}
        img, asset, status = mt._resolve_art(e, (TILE_W, TILE_H * 2))
        assert status == "placed" and asset == "my_oak.png", (asset, status)
    return "library first, ADR 0007 D6"


@case("a want falls back to the prop library")
def _library_second():
    with Images():
        art().save(mt._library_path("oak tree"))
        e = {"asset": None, "want": "oak tree", "x": 0, "y": 0, "layer": "props"}
        img, asset, status = mt._resolve_art(e, (TILE_W, TILE_H * 2))
        assert status == "placed", status
        assert asset == os.path.basename(mt._library_path("oak tree")), asset
    return f"resolved to {asset}"


@case("an asset that is named but missing does not silently vanish")
def _missing_asset_is_pending():
    with Images():
        e = {"asset": "not_on_disk.png", "want": "oak tree", "x": 0, "y": 0,
             "layer": "props"}
        img, asset, status = mt._resolve_art(e, (TILE_W, TILE_H * 2))
        assert status == "pending", status
        assert asset is None, asset
    return "placed with a placeholder, and the tilemap stops claiming the asset"


@case("a gap gets the deliberately ugly placeholder")
def _placeholder_used():
    with Images():
        e = {"asset": None, "want": "windmill", "x": 0, "y": 0, "layer": "props"}
        img, asset, status = mt._resolve_art(e, (TILE_W, TILE_H * 2))
        assert status == "pending" and asset is None
        px = np.asarray(img.convert("RGBA"))
        assert (px[..., :3] == [255, 0, 200]).all(axis=-1).any(), "not magenta"
    return "magenta, so it can never be cached downstream as finished"


# --- dressing a whole map -------------------------------------------------


@case("pending lists each want once, whatever the placement count")
def _pending_dedup():
    with Images():
        entities = [{"asset": None, "want": w, "x": i, "y": 0, "layer": "props"}
                    for i, w in enumerate(["windmill", "shrine", "windmill"])]
        _, pending = mt._dress(entities, TILE_W, TILE_H)
        assert pending == ["windmill", "shrine"], pending
    return "2 wants over 3 placements"


@case("dressing is idempotent")
def _dress_idempotent():
    with Images():
        art().save(mt._library_path("oak"))
        entities = [{"asset": None, "want": "oak", "x": 1, "y": 1,
                     "layer": "props"}]
        mt._dress(entities, TILE_W, TILE_H)
        first = json.dumps(entities, sort_keys=True)
        mt._dress(entities, TILE_W, TILE_H)
        assert json.dumps(entities, sort_keys=True) == first
    return "re-running the resolver cannot churn the tilemap"


# --- writing over a file someone may be reading ---------------------------


@case("a rewrite is atomic")
def _atomic():
    with Images() as d:
        path = os.path.join(d, "x.json")
        with open(path, "w") as fh:
            fh.write('{"v": 1}')

        seen = {}

        def slow(target):
            with open(target, "w") as fh:
                fh.write('{"v": 2, "hal')
            # Mid-write, the reader must still get the WHOLE old file.
            with open(path) as fh:
                seen["mid"] = json.load(fh)
            with open(target, "w") as fh:
                fh.write('{"v": 2}')

        mt._replace_atomically(path, slow)
        assert seen["mid"] == {"v": 1}, seen
        with open(path) as fh:
            assert json.load(fh) == {"v": 2}
    return "no reader ever sees half a tilemap"


@case("a failed rewrite leaves the original and no litter")
def _atomic_failure():
    with Images() as d:
        path = os.path.join(d, "x.json")
        with open(path, "w") as fh:
            fh.write('{"v": 1}')

        def boom(target):
            with open(target, "w") as fh:
                fh.write("partial")
            raise ValueError("disk full")

        try:
            mt._replace_atomically(path, boom)
        except ValueError:
            pass
        else:
            raise AssertionError("the write error was swallowed")

        with open(path) as fh:
            assert json.load(fh) == {"v": 1}
        assert not os.path.exists(path + ".tmp"), "temp file left behind"
    return "raised, original intact, nothing left over"


# --- re-compositing the map ----------------------------------------------


@case("a resolved prop flips the map to complete")
def _resolve_completes():
    job = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    with Images():
        write_map(mt.IMAGES_DIR, job, [
            {"asset": None, "want": "windmill", "x": 1, "y": 1,
             "layer": "props", "status": "pending"},
        ])
        art().save(mt._library_path("windmill"))
        out = mt._rerender_map(job)
        assert out["complete"] is True, out["complete"]
        assert out["pending"] == [], out["pending"]
        assert out["entities"][0]["status"] == "placed"
        assert out["entities"][0]["asset"] == "prop_windmill.png"
        on_disk = json.load(open(mt._tilemap_path(job)))
        assert on_disk["complete"] is True, "the served file was not updated"
    return "picture and tilemap both rewritten"


@case("a partial resolve stays incomplete and keeps what worked")
def _partial_stays_incomplete():
    job = "aaaaaaaa-bbbb-cccc-dddd-ffffffffffff"
    with Images():
        write_map(mt.IMAGES_DIR, job, [
            {"asset": None, "want": "windmill", "x": 1, "y": 1,
             "layer": "props", "status": "pending"},
            {"asset": None, "want": "shrine", "x": 3, "y": 3,
             "layer": "props", "status": "pending"},
        ])
        art().save(mt._library_path("windmill"))  # shrine failed to generate
        out = mt._rerender_map(job)
        assert out["complete"] is False
        assert out["pending"] == ["shrine"], out["pending"]
        placed = [e for e in out["entities"] if e["status"] == "placed"]
        assert len(placed) == 1 and placed[0]["want"] == "windmill"
    return "one failure costs one prop, not the other four"


@case("entities do not move when a prop resolves")
def _entities_do_not_move():
    job = "aaaaaaaa-bbbb-cccc-dddd-111111111111"
    with Images():
        entities = [
            {"asset": None, "want": "windmill", "x": 1, "y": 6,
             "layer": "props", "status": "pending"},
            {"asset": None, "want": "wolf", "x": 5, "y": 2,
             "layer": "creatures", "status": "pending"},
        ]
        write_map(mt.IMAGES_DIR, job, entities)
        before = [(e["x"], e["y"], e["layer"]) for e in entities]
        art().save(mt._library_path("windmill"))
        out = mt._rerender_map(job)
        after = [(e["x"], e["y"], e["layer"]) for e in out["entities"]]
        assert after == before, f"{before} -> {after}"
    return "the resolver re-draws, it does not re-scatter"


@case("re-compositing costs no GPU")
def _no_gpu():
    job = "aaaaaaaa-bbbb-cccc-dddd-222222222222"
    with Images():
        write_map(mt.IMAGES_DIR, job, [
            {"asset": None, "want": "windmill", "x": 2, "y": 2,
             "layer": "props", "status": "pending"}])
        art().save(mt._library_path("windmill"))

        def refuse(*a, **k):
            raise AssertionError("the resolver reached for the GPU")

        import tasks
        was = tasks.get_sd_pipeline
        tasks.get_sd_pipeline = refuse
        try:
            mt._rerender_map(job)
        finally:
            tasks.get_sd_pipeline = was
    return "terrain tiles are reloaded from disk, never regenerated"


@case("roads survive the re-composite")
def _roads_survive():
    job = "aaaaaaaa-bbbb-cccc-dddd-333333333333"
    with Images():
        write_map(mt.IMAGES_DIR, job, [
            {"asset": None, "want": "windmill", "x": 4, "y": 5,
             "layer": "props", "status": "pending"}], roads=True)
        art().save(mt._library_path("windmill"))
        out = mt._rerender_map(job)
        assert any(any(r) for r in out["layers"]["roads"]), "roads were dropped"
        assert out["road_tile"], "the road tile reference was dropped"
    return "layer 2 is reloaded, not re-derived"


@case("a missing terrain tile fails loudly rather than drawing a hole")
def _missing_tile_raises():
    job = "aaaaaaaa-bbbb-cccc-dddd-444444444444"
    with Images():
        write_map(mt.IMAGES_DIR, job, [
            {"asset": None, "want": "windmill", "x": 2, "y": 2,
             "layer": "props", "status": "pending"}])
        os.unlink(mt._tile_path(job, 1))
        try:
            mt._rerender_map(job)
        except Exception as e:
            assert isinstance(e, (FileNotFoundError, OSError)), type(e)
            on_disk = json.load(open(mt._tilemap_path(job)))
            assert on_disk["complete"] is False, "claimed complete after failing"
            return "raised, and the served map was left as it was"
    raise AssertionError("re-rendered a map whose tiles are gone")


@case("the picture is actually replaced")
def _picture_replaced():
    job = "aaaaaaaa-bbbb-cccc-dddd-555555555555"
    with Images():
        write_map(mt.IMAGES_DIR, job, [
            {"asset": None, "want": "windmill", "x": 3, "y": 3,
             "layer": "props", "status": "pending"}])
        before = open(mt._picture_path(job), "rb").read()
        art().save(mt._library_path("windmill"))
        mt._rerender_map(job)
        after = open(mt._picture_path(job), "rb").read()
        assert before != after, "the picture still shows the placeholder"
        assert not os.path.exists(mt._picture_path(job) + ".tmp")
    return f"{len(before)} -> {len(after)} bytes"


# --- what a caller is told while the map is provisional -------------------


class FakeJobs:
    """Stand in for the one query `_with_props_status` makes.

    Worth faking rather than skipping: this function decides whether a caller
    WAITS or gives up, and getting it wrong is invisible - the map looks fine
    and the consumer simply never stops polling.
    """

    def __init__(self, row):
        self.row = row

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self, **kw):
        return self

    def execute(self, *a):
        pass

    def fetchone(self):
        return self.row


def props_status(tilemap, row):
    import maps as maps_api
    was = maps_api._db
    maps_api._db = lambda: FakeJobs(row)
    try:
        return maps_api._with_props_status(dict(tilemap)).get("props_status")
    finally:
        maps_api._db = was


PROVISIONAL = {"complete": False, "pending": ["wolf"], "props_job": "x"}


@case("a running resolver tells a caller to keep waiting")
def _status_working():
    for status in ("queued", "running"):
        s = props_status(PROVISIONAL, {"status": status, "progress_pct": 40,
                                       "progress_msg": "prop 2/5", "error": None})
        assert s["state"] == "working", s
        assert s["final"] is False, s
    return "working, final=false"


@case("a finished resolver on an incomplete map reads as partial, not done")
def _status_partial():
    s = props_status(PROVISIONAL, {"status": "done", "progress_pct": 100,
                                   "progress_msg": "resolved 1/2", "error": None})
    # "done" here would send a caller back to wait for art that is not coming.
    assert s["state"] == "partial", s
    assert s["final"] is True, s
    return "partial, final=true"


@case("a reaped resolver surfaces as failed, with its reason")
def _status_reaped():
    s = props_status(PROVISIONAL, {
        "status": "failed", "progress_pct": 12, "progress_msg": "stranded",
        "error": "worker died before this job finished; resubmit to retry"})
    assert s["state"] == "failed", s
    assert s["final"] is True and "resubmit" in s["detail"], s
    return "the strand becomes a message instead of an infinite wait"


@case("a vanished resolver is not mistaken for one still working")
def _status_lost():
    s = props_status(PROVISIONAL, None)
    assert s["state"] == "lost", s
    return "rebuild the map to try again"


@case("a complete map is not asked about at all")
def _status_skipped():
    import maps as maps_api
    was = maps_api._db

    def boom():
        raise AssertionError("queried the resolver for a finished map")

    maps_api._db = boom
    try:
        out = maps_api._with_props_status({"complete": True, "props_job": "x"})
        assert "props_status" not in out
    finally:
        maps_api._db = was
    return "no query, no key"


def main() -> int:
    failed = 0
    for name, fn in CASES:
        try:
            print(f"  ok    {name}  ({fn()})")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}\n        {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {name}\n        {type(e).__name__}: {e}")

    print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
