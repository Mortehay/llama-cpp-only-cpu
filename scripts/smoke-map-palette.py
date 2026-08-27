#!/usr/bin/env python3
"""Terrain colours read off a reference. No model, no GPU.

    docker exec sprite_generator python /app/scripts/smoke-map-palette.py

Two failures were MEASURED on real references, and both are silent - the map
builds, looks plausible, and is missing a third of what was asked for:

    a declared colour that is not in the art costs you the terrain
    a near-neutral terrain swallows everything desaturated

`validate_terrains` cannot see either. It checks the declared colours are far
APART, which says nothing about whether any of them is in the painting. The
build reports coverage afterwards - the right place for the last word, the
wrong place for the first, because by then the GPU is spent.

The case that carries this is `a guessed palette loses a terrain`. It is the
real failure, reproduced from synthetic art whose colours are known exactly, so
it cannot pass for the wrong reason.
"""

from __future__ import annotations

import os
import sys
import uuid

sys.path.insert(0, "/app")

import numpy as np  # noqa: E402
import psycopg2  # noqa: E402
from PIL import Image  # noqa: E402

import map_geometry as mg  # noqa: E402
import maps as maps_api  # noqa: E402
import pixelate  # noqa: E402

DB = os.environ["DB_URL"]
CASES: list = []


def case(name):
    def wrap(fn):
        CASES.append((name, fn))
        return fn
    return wrap


# A painting whose real colours are known: a muted-blue sea, a green shore, a
# sandy strip. Muted deliberately - a saturated sea would snap to almost any
# blue and the test would pass without the fix.
ART = [(91, 131, 161), (74, 124, 63), (201, 181, 138)]


def painting(w=96, h=96) -> Image.Image:
    a = np.zeros((h, w, 3), dtype=np.uint8)
    a[:, : w // 2] = ART[0]
    a[:, w // 2: w * 3 // 4] = ART[1]
    a[:, w * 3 // 4:] = ART[2]
    return Image.fromarray(a, "RGB")


def as_terrains(colors) -> list:
    return [{"name": f"t{i}", "color": "#%02x%02x%02x" % tuple(c),
             "walkable": True} for i, c in enumerate(colors)]


class Reference:
    """A real `reference_assets` row and its file, torn down afterwards."""

    def __init__(self, image):
        self.image = image
        self.ref_id = str(uuid.uuid4())

    def __enter__(self):
        self.path = f"/app/images/_pal_{self.ref_id[:12]}.png"
        self.image.save(self.path)
        with psycopg2.connect(DB) as c, c.cursor() as cur:
            cur.execute(
                "INSERT INTO reference_assets (id, kind, label, file_path) "
                "VALUES (%s::uuid, 'map', '_palette-fixture', %s)",
                (self.ref_id, self.path))
        return self

    def __exit__(self, *exc):
        with psycopg2.connect(DB) as c, c.cursor() as cur:
            cur.execute("DELETE FROM reference_assets WHERE id = %s::uuid",
                        (self.ref_id,))
        if os.path.exists(self.path):
            os.unlink(self.path)
        return False


maps_api.auth.require = lambda *a, **k: {"open": True}


# --- the failures this exists to prevent ----------------------------------


@case("a guessed palette loses a terrain")
def _guessed_loses():
    """The measured failure, reproduced. A muted-blue sea declared against a
    navy #2850c8 came back at 2.4% on a real reference; against its own colour,
    49%. Here the art is synthetic so the numbers are exact."""
    img = painting()
    guessed = as_terrains([(40, 80, 200), (74, 124, 63), (201, 181, 138)])
    cover = mg.coverage(mg.quantize(img, guessed, (64, 64)), guessed)
    sea = cover["t0"]
    assert sea < 0.05, f"the guess did not lose the sea ({sea:.0%}) - the "\
                       f"fixture is too forgiving to prove anything"
    return f"navy declared against a muted sea captured {sea:.1%} of it"


@case("a measured palette keeps every terrain")
def _measured_keeps():
    img = painting()
    pal = pixelate.extract_palette(img.convert("RGBA"), 3)
    measured = as_terrains(pal)
    cover = mg.coverage(mg.quantize(img, measured, (64, 64)), measured)
    assert min(cover.values()) > 0.15, cover
    assert abs(sum(cover.values()) - 1.0) < 0.01, cover
    return ", ".join(f"{k} {v:.0%}" for k, v in sorted(cover.items()))


@case("a near-neutral terrain is flagged before it eats the map")
def _neutral_sink():
    """Dropping a grey `stone` took a real reference from 2.4% to 58% water
    without the water colour changing. Grey sits near the middle of Lab and is
    the nearest match for everything desaturated."""
    grey = np.array([[110, 110, 115]], dtype=np.uint8)
    lab = pixelate.srgb_to_lab(grey).reshape(-1, 3)
    chroma = float(np.sqrt(lab[0, 1] ** 2 + lab[0, 2] ** 2))
    assert chroma < maps_api.NEUTRAL_CHROMA, chroma

    # And a colour that is genuinely coloured must NOT be flagged, or the
    # warning becomes noise everyone learns to skip.
    for vivid in ([40, 80, 200], [74, 124, 63], [201, 181, 138]):
        v = pixelate.srgb_to_lab(np.array([vivid], dtype=np.uint8)).reshape(-1, 3)
        assert float(np.sqrt(v[0, 1] ** 2 + v[0, 2] ** 2)) >= maps_api.NEUTRAL_CHROMA, vivid
    return f"grey chroma {chroma:.1f} < {maps_api.NEUTRAL_CHROMA}, the three real ones above it"


# --- the route ------------------------------------------------------------


@case("the route returns colours that are actually in the art")
def _route_colors():
    with Reference(painting()) as r:
        out = maps_api.palette_from_reference(r.ref_id, terrains=3)
        got = [mg.parse_color(t["color"]) for t in out["terrains"]]
        for real in ART:
            near = min(sum((a - b) ** 2 for a, b in zip(real, g)) for g in got)
            assert near < 900, f"{real} has no suggestion within 30/channel: {got}"
    return f"3 suggestions, each within 30/channel of a colour in the art"


@case("the route says what share each colour would take")
def _route_coverage():
    with Reference(painting()) as r:
        out = maps_api.palette_from_reference(r.ref_id, terrains=3)
        cov = [t["coverage"] for t in out["terrains"]]
        assert abs(sum(cov) - 1.0) < 0.01, cov
        assert min(cov) > 0.15, cov
        # The art is half sea, so one of them must be about half.
        assert max(cov) > 0.4, cov
    return f"coverage {[round(c, 2) for c in cov]}, sums to 1"


@case("the route warns about a near-neutral suggestion")
def _route_warns_neutral():
    # A deliberately washed-out painting, so median-cut has to return greys.
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    a[:, :32] = (118, 116, 120)
    a[:, 32:] = (150, 148, 152)
    with Reference(Image.fromarray(a, "RGB")) as r:
        out = maps_api.palette_from_reference(r.ref_id, terrains=2)
        assert any(t["near_neutral"] for t in out["terrains"]), out["terrains"]
        assert any("near-neutral" in w for w in out["warnings"]), out["warnings"]
    return "greys are named as sinks rather than offered silently"


@case("a colourful reference is not nagged")
def _route_quiet():
    with Reference(painting()) as r:
        out = maps_api.palette_from_reference(r.ref_id, terrains=3)
        assert not out["warnings"], out["warnings"]
    return "no warning when nothing is wrong"


@case("asking for more terrains than the art has is said out loud")
def _route_short():
    """This case was written expecting the surplus to come back badly
    separated. It does not - `extract_palette` caps at the number of distinct
    colours in the art, so eight asked of a three-colour painting returns
    three, silently. The wrong premise found a real gap: a caller who asked for
    eight is building a form with eight rows."""
    with Reference(painting()) as r:
        out = maps_api.palette_from_reference(r.ref_id, terrains=8)
        assert out["asked_for"] == 8
        assert len(out["terrains"]) == 3, out["terrains"]
        assert any("only supports 3" in w for w in out["warnings"]), out["warnings"]
        # And the three it did return are still a usable set.
        assert out["separation"] >= 12.0, out["separation"]
        mg.validate_terrains(out["terrains"])
    return "8 asked of a 3-colour painting -> 3 returned, and it says so"


@case("the suggestions are a valid terrain set when they say they are")
def _route_valid():
    with Reference(painting()) as r:
        out = maps_api.palette_from_reference(r.ref_id, terrains=3)
        assert not out["warnings"], out["warnings"]
        # The claim being tested: no warnings means the build will accept it.
        mg.validate_terrains(out["terrains"])
    return "validate_terrains accepts an unwarned suggestion"


@case("an unknown reference is a 404")
def _route_missing():
    try:
        maps_api.palette_from_reference(str(uuid.uuid4()), terrains=3)
    except Exception as e:
        assert getattr(e, "status_code", None) == 404, e
        return "404 rather than an empty palette"
    raise AssertionError("returned a palette for a reference that does not exist")


@case("an absurd terrain count is refused")
def _route_bounds():
    with Reference(painting()) as r:
        for n in (1, 0, 99):
            try:
                maps_api.palette_from_reference(r.ref_id, terrains=n)
            except Exception as e:
                assert getattr(e, "status_code", None) == 400, (n, e)
            else:
                raise AssertionError(f"accepted terrains={n}")
    return "2..16, refused outside"


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
