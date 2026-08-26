#!/usr/bin/env python3
"""Quantisation and isometric compositing under test. No model, no GPU.

Run inside the API container:

    docker exec sprite_generator python /app/scripts/smoke-map-build.py

This covers the deterministic half of a map build - the half that decides
whether the picture and the walkable ground can disagree. The painting normally
comes from the map adapter; here it is hand-made, so the whole path runs on CPU
in under a second and can be checked without an adapter existing.

The assertion that matters is `no_seams`. A field of rhombi that are one pixel
short leaves a transparent hairline down two sides of every tile - invisible on
one tile and obvious across a map, which is exactly the bug `tile_geometry`
guards against with its width-1/height-1 polygon.
"""

from __future__ import annotations

import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/app")

import map_geometry as mg  # noqa: E402
import tile_geometry as tg  # noqa: E402

TERRAINS = [
    {"id": 0, "name": "grass", "color": "#4a7c3f", "walkable": True},
    {"id": 1, "name": "water", "color": "#2850c8", "walkable": False},
    {"id": 2, "name": "sand", "color": "#c8be78", "walkable": True},
    {"id": 3, "name": "stone", "color": "#6e6e73", "walkable": True},
]

TILE_W, TILE_H = 32, 16
GRID = 16


def painting_from(ids: np.ndarray) -> Image.Image:
    """Render a terrain id array back into a biome painting, exactly."""
    pal = mg.palette_array(TERRAINS)
    return Image.fromarray(pal[ids].astype(np.uint8), "RGB")


def quadrants(n: int) -> np.ndarray:
    ids = np.zeros((n, n), dtype=np.int16)
    ids[: n // 2, n // 2:] = 1
    ids[n // 2:, : n // 2] = 2
    ids[n // 2:, n // 2:] = 3
    return ids


def solid_tiles() -> list[Image.Image]:
    """One rhombus per terrain, through the real tile mask."""
    out = []
    for t in TERRAINS:
        img = Image.new("RGBA", (TILE_W, TILE_H), (*mg.parse_color(t["color"]), 255))
        img.putalpha(tg.diamond_mask(TILE_W, TILE_H))
        out.append(img)
    return out


CASES = []


def case(name):
    def wrap(fn):
        CASES.append((name, fn))
        return fn
    return wrap


@case("quantise is exact for declared colours")
def _exact():
    want = quadrants(GRID)
    got = mg.quantize(painting_from(want), TERRAINS, (GRID, GRID))
    assert np.array_equal(got, want), "a painting of exact terrain colours did " \
                                      "not round-trip to the same ids"
    return f"{GRID}x{GRID} round-tripped"


@case("quantise snaps a noisy painting to the right terrain")
def _noisy():
    want = quadrants(GRID)
    pal = mg.palette_array(TERRAINS)
    rng = np.random.default_rng(7)
    noisy = pal[want].astype(np.int16) + rng.integers(-9, 10, (GRID, GRID, 3))
    img = Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8), "RGB")
    got = mg.quantize(img, TERRAINS, (GRID, GRID))
    assert np.array_equal(got, want), "noise moved pixels to the wrong terrain"
    return "+-9 per channel, still exact"


@case("picture size scales with the SUM of the grid")
def _size():
    assert mg.picture_size(16, 16, 32, 16) == (512, 256)
    assert mg.picture_size(64, 64, 64, 32) == (4096, 2048)
    # Doubling the grid quadruples the pixels - the number worth checking
    # before queueing a large map.
    small = mg.picture_size(32, 32, 64, 32)
    big = mg.picture_size(64, 64, 64, 32)
    assert (big[0] * big[1]) == 4 * (small[0] * small[1])
    return "64x64 @64px -> 4096x2048"


@case("the tile mask can tessellate at all")
def _mask_tiles():
    """Rows half a tile apart must together span exactly one tile.

    Checked directly on the mask rather than only through a composited field,
    because this is the property that fails, and a picture-level assertion says
    "there are holes" without saying where they come from.
    """
    for tw, th in [(32, 16), (64, 32), (48, 24), (64, 64)]:
        a = np.asarray(tg.diamond_mask(tw, th))
        widths = [int((a[r] > 0).sum()) for r in range(th)]
        assert sum(widths) == tw * th // 2, (
            f"{tw}x{th}: {sum(widths)} opaque px, want {tw * th // 2}")
        for y in range(th // 2):
            pair = widths[y] + widths[y + th // 2]
            assert pair == tw, (
                f"{tw}x{th}: rows {y} and {y + th // 2} span {pair}, want {tw}")
        assert widths == widths[::-1], f"{tw}x{th}: mask is not symmetric"
    return "4 sizes, rows pair to exactly one tile"


@case("composite: no seams across the interior")
def _no_seams():
    grid = quadrants(GRID)
    pic = mg.composite(grid, solid_tiles())
    assert pic.size == mg.picture_size(GRID, GRID, TILE_W, TILE_H), pic.size

    alpha = np.asarray(pic)[..., 3]
    h, w = alpha.shape
    # A conservative box around the centre, well inside the map's own rhombus.
    y0, y1 = int(h * 0.35), int(h * 0.65)
    x0, x1 = int(w * 0.35), int(w * 0.65)
    holes = int((alpha[y0:y1, x0:x1] < 255).sum())
    assert holes == 0, f"{holes} non-opaque pixels inside the map - hairline seams"
    return f"{(y1 - y0) * (x1 - x0)} interior px all opaque"


@case("composite: picture agrees with the grid")
def _agree():
    grid = quadrants(GRID)
    pic = mg.composite(grid, solid_tiles()).convert("RGB")
    arr = np.asarray(pic)
    pal = mg.palette_array(TERRAINS)

    # Sample the centre of one cell per quadrant and check the colour is the
    # terrain the grid claims. This is "they must agree", measured.
    origin_x = (GRID - 1) * TILE_W // 2
    for (x, y) in [(3, 3), (12, 3), (3, 12), (12, 12)]:
        sx = origin_x + (x - y) * TILE_W // 2 + TILE_W // 2
        sy = (x + y) * TILE_H // 2 + TILE_H // 2
        want = pal[grid[y, x]]
        got = arr[sy, sx]
        assert tuple(got) == tuple(want), (
            f"cell ({x},{y}) says terrain {grid[y, x]} {tuple(want)} but the "
            f"picture shows {tuple(got)}")
    return "4 cells sampled, all agree"


@case("coverage adds up")
def _coverage():
    cov = mg.coverage(quadrants(GRID), TERRAINS)
    assert abs(sum(cov.values()) - 1.0) < 1e-6, cov
    assert all(abs(v - 0.25) < 1e-6 for v in cov.values()), cov
    return ", ".join(f"{k} {v:.0%}" for k, v in cov.items())


@case("terrains too close in Lab are rejected before any GPU")
def _too_close():
    bad = [{"id": 0, "name": "a", "color": "#808080"},
           {"id": 1, "name": "b", "color": "#858585"}]
    try:
        mg.validate_terrains(bad)
    except ValueError as e:
        assert "quantise into one tile" in str(e), str(e)
        return str(e).split(" - ")[0]
    raise AssertionError("two near-identical terrains were accepted")


@case("duplicate terrain names are rejected")
def _dupes():
    bad = [{"id": 0, "name": "grass", "color": "#4a7c3f"},
           {"id": 1, "name": "grass", "color": "#2850c8"}]
    try:
        mg.validate_terrains(bad)
    except ValueError as e:
        assert "duplicate" in str(e)
        return "rejected"
    raise AssertionError("duplicate names were accepted")


@case("a terrain id with no tile fails loudly")
def _missing_tile():
    grid = quadrants(GRID)
    try:
        mg.composite(grid, solid_tiles()[:2])
    except ValueError as e:
        assert "terrain id" in str(e), str(e)
        return "raised rather than drawing a hole"
    raise AssertionError("composited a grid referencing tiles that do not exist")


@case("the real terrain set validates")
def _real():
    mg.validate_terrains(TERRAINS)
    return f"{mg.separation(mg.palette_array(TERRAINS)):.1f} Lab apart at closest"


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
