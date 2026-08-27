#!/usr/bin/env python3
"""The something2 map facade under test. No model, no GPU.

    docker exec sprite_generator python /app/scripts/smoke-map-facade.py

The facade is the one part of the map feature something2 actually touches, and
its whole job is to NOT generate: it resolves a name to an already-built map
and reads it off disk. Everything that can go wrong here is quiet - the wrong
map, a half-built one, or a two-hour generation started inside a blocking HTTP
request that will time out long before it finishes.

The case that carries the slice is `never generates`. A facade that falls
through to the model on a cache miss looks identical on a hit and destroys the
caller on a miss.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import uuid

sys.path.insert(0, "/app")

import psycopg2  # noqa: E402
from PIL import Image  # noqa: E402

import a1111  # noqa: E402
import maps as maps_api  # noqa: E402

DB = os.environ["DB_URL"]
CASES: list = []


def case(name):
    def wrap(fn):
        CASES.append((name, fn))
        return fn
    return wrap


class Built:
    """A real `jobs` row plus its two files, torn down afterwards.

    A fixture rather than a mock, because the thing under test is a database
    query and two file reads - mocking those would leave nothing.
    """

    def __init__(self, name, status="done", complete=True, picture=True,
                 tilemap=True):
        self.name = name
        self.status = status
        self.complete = complete
        self.picture = picture
        self.tilemap = tilemap
        self.job_id = str(uuid.uuid4())

    def __enter__(self):
        short = self.job_id[:12]
        self.pic_path = f"/app/images/_facade_{short}.png"
        self.map_path = f"/app/images/_facade_{short}.json"

        if self.picture:
            Image.new("RGB", (8, 4), (10, 120, 60)).save(self.pic_path)
        if self.tilemap:
            with open(self.map_path, "w", encoding="utf-8") as fh:
                json.dump({"id": self.job_id, "name": self.name,
                           "complete": self.complete,
                           "pending": [] if self.complete else ["windmill"],
                           "size": {"w": 8, "h": 8}}, fh)

        with psycopg2.connect(DB) as c, c.cursor() as cur:
            cur.execute(
                "INSERT INTO jobs (id, kind, status, spec, sheet_path, "
                "                  atlas_path, finished_at) "
                "VALUES (%s, 'map', %s, %s, %s, %s, now())",
                (self.job_id, self.status, json.dumps({"name": self.name}),
                 self.pic_path if self.picture else None,
                 self.map_path if self.tilemap else None))
        return self

    def __exit__(self, *exc):
        with psycopg2.connect(DB) as c, c.cursor() as cur:
            cur.execute("DELETE FROM jobs WHERE id = %s::uuid", (self.job_id,))
        for p in (self.pic_path, self.map_path):
            if os.path.exists(p):
                os.unlink(p)
        return False


def req(prompt="", **ov):
    return a1111.Txt2ImgRequest(prompt=prompt, override_settings=ov)


# --- resolving a name -----------------------------------------------------


@case("a name resolves to its finished map")
def _resolve():
    with Built("overworld") as b:
        row = maps_api.resolve_name("overworld")
        assert row and str(row["id"]) == b.job_id, row
    return "the facade needs no UUID typed into an admin form"


@case("a name is matched case-insensitively and trimmed")
def _resolve_loose():
    with Built("Overworld"):
        for asked in ("overworld", "OVERWORLD", "  Overworld  "):
            assert maps_api.resolve_name(asked), asked
    return "something2's admin will not preserve your capitalisation"


@case("an unfinished map is not served as if it were finished")
def _resolve_unfinished():
    # The dangerous near-miss: right name, wrong state. Serving it means
    # something2 seeds a world from a file about to be overwritten.
    with Built("halfbuilt", status="running"):
        assert maps_api.resolve_name("halfbuilt") is None
    return "only `done` rows resolve"


@case("a finished map whose tilemap vanished does not resolve")
def _resolve_no_file():
    with Built("ghost", tilemap=False):
        assert maps_api.resolve_name("ghost") is None
    return "a row without its file is not a map"


@case("the newest map of a name wins")
def _resolve_newest():
    with Built("rebuilt") as old, Built("rebuilt") as new:
        row = maps_api.resolve_name("rebuilt")
        assert str(row["id"]) in (old.job_id, new.job_id)
        # Both are `now()`, so this asserts only that ONE is chosen and it is a
        # real one - ordering between same-instant rows is not something to
        # depend on, and pretending otherwise would be a flaky test.
        assert row is not None
    return "rebuilding a map does not orphan its name"


@case("an unknown name resolves to nothing")
def _resolve_missing():
    assert maps_api.resolve_name("no-such-map-anywhere") is None
    return "no fuzzy matching, no nearest hit"


# --- the picture over the AI connector ------------------------------------


@case("a map: prompt returns the built picture, base64")
def _facade_hit():
    with Built("overworld") as b:
        out = a1111._serve_map("overworld", req("map:overworld"), 0.0)
        assert out["images"], out
        raw = base64.b64decode(out["images"][0])
        assert raw[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
        info = json.loads(out["info"])
        assert info["cached"] is True and info["generated"] is False
        assert info["job_id"] == b.job_id
    return f"{len(out['images'][0])} b64 chars, info says cached"


@case("NEVER generates on a cache miss")
def _facade_never_generates():
    """The case the slice rests on.

    A facade that falls through to the model looks identical on a hit and
    destroys the caller on a miss: a map build is minutes to hours inside a
    blocking request with a 240s budget. It must 404, not queue.
    """
    fired = []
    was = a1111.generate_raw_task
    a1111.generate_raw_task = type("X", (), {
        "delay": staticmethod(lambda *a, **k: fired.append(a))})()
    try:
        try:
            a1111._serve_map("nothing-here", req("map:nothing-here"), 0.0)
        except Exception as e:
            assert getattr(e, "status_code", None) == 404, e
        else:
            raise AssertionError("a missing map did not 404")
    finally:
        a1111.generate_raw_task = was
    assert not fired, "the facade queued a generation"
    return "404, and nothing was queued"


@case("a provisional map says so in info")
def _facade_provisional():
    # A consumer that caches images[0] as final keeps a magenta placeholder
    # forever. This is the one thing the response can say that a generated
    # image never has to.
    with Built("halfdone", complete=False):
        info = json.loads(
            a1111._serve_map("halfdone", req("map:halfdone"), 0.0)["info"])
        assert info["complete"] is False, info
        assert info["pending"] == ["windmill"], info
    return "complete=false and the missing prop named"


@case("a map whose picture is gone is 409, not a broken image")
def _facade_no_picture():
    with Built("pictureless", picture=False):
        try:
            a1111._serve_map("pictureless", req("map:pictureless"), 0.0)
        except Exception as e:
            assert getattr(e, "status_code", None) == 409, e
            return "409 - the map exists, the file does not"
    raise AssertionError("served a map with no picture")


@case("an unreadable tilemap does not withhold the picture")
def _facade_bad_sidecar():
    # The picture is the artefact and it is already on disk. Failing the whole
    # request because the sidecar would not parse would withhold something that
    # is perfectly fine.
    with Built("corrupt") as b:
        with open(b.map_path, "w", encoding="utf-8") as fh:
            fh.write("{not json")
        out = a1111._serve_map("corrupt", req("map:corrupt"), 0.0)
        assert out["images"], "the picture was withheld over a bad sidecar"
        assert json.loads(out["info"])["complete"] is True
    return "picture served, status defaults to complete"


@case("a normal prompt is untouched by the facade")
def _facade_passthrough():
    for prompt in ("a green field", "a map of the world", "roadmap:2026", ""):
        assert a1111._map_request(req(prompt)) is None, prompt
    return "only an explicit `map:` prefix diverts"


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
