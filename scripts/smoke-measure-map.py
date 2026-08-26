#!/usr/bin/env python3
"""`measure.measure_map` under test. No model, no GPU, no network.

Run inside the API container, where PIL/numpy and the module live:

    docker exec sprite_generator python /app/scripts/smoke-measure-map.py

The case that matters most is `painted_map`. A map reference must NOT be
rejected for having thousands of colours - a painted map legitimately does, and
gating on colour count is the mistake migration 014 had to undo for sprites,
where it rejected 100 of 106 real references. What a map IS rejected for is
terrain SEPARATION: two candidates too close in Lab collapse into one tile id.
"""

from __future__ import annotations

import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/app")

import measure  # noqa: E402


def solid(color, size=(64, 64)) -> Image.Image:
    return Image.new("RGBA", size, (*color, 255))


def bands(colors, size=(64, 64)) -> Image.Image:
    """A vertical band per colour - the shape of a quantisable biome layout."""
    img = Image.new("RGBA", size, (0, 0, 0, 255))
    arr = np.asarray(img).copy()
    step = max(1, size[0] // len(colors))
    for i, c in enumerate(colors):
        arr[:, i * step:(i + 1) * step, :3] = c
    return Image.fromarray(arr, "RGBA")


def painted(size=(64, 64)) -> Image.Image:
    """A smooth gradient: thousands of distinct colours, still a usable map."""
    x = np.linspace(0, 255, size[0], dtype=np.uint8)
    y = np.linspace(0, 255, size[1], dtype=np.uint8)
    gx, gy = np.meshgrid(x, y)
    arr = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    arr[..., 0] = gx
    arr[..., 1] = gy
    arr[..., 2] = (gx.astype(int) + gy.astype(int)) // 2
    arr[..., 3] = 255
    return Image.fromarray(arr, "RGBA")


CASES = []


def case(name):
    def wrap(fn):
        CASES.append((name, fn))
        return fn
    return wrap


@case("four separated terrains -> usable")
def _four():
    v = measure.measure_map(bands([(60, 130, 60), (40, 80, 200),
                                   (200, 190, 120), (110, 110, 115)]))
    assert v["usable"] is True, v["why"]
    assert v["metrics"]["terrains"] == 4, v["metrics"]["terrains"]
    assert v["metrics"]["terrain_separation"] >= measure.TERRAIN_MIN_LAB_SEPARATION
    assert len(v["metrics"]["terrain_palette"]) == 4
    return f"{v['metrics']['terrains']} terrains, sep {v['metrics']['terrain_separation']}"


@case("painted map, thousands of colours -> STILL usable")
def _painted():
    v = measure.measure_map(painted())
    assert v["metrics"]["colors"] > 500, v["metrics"]["colors"]
    assert v["usable"] is True, (
        f"a painted map was rejected for colour count - this is the "
        f"migration-014 mistake: {v['why']}")
    return f"{v['metrics']['colors']} colours -> {v['metrics']['terrains']} terrains"


@case("one colour -> not usable")
def _one():
    v = measure.measure_map(solid((60, 130, 60)))
    assert v["usable"] is False
    assert "one colour" in v["why"], v["why"]
    return v["why"]


@case("two near-identical terrains -> not usable")
def _near():
    v = measure.measure_map(bands([(128, 128, 128), (133, 133, 133)]))
    assert v["usable"] is False, v["why"]
    assert "quantise into one tile" in v["why"], v["why"]
    return v["why"]


@case("fully transparent -> not usable, no crash")
def _empty():
    v = measure.measure_map(Image.new("RGBA", (64, 64), (0, 0, 0, 0)))
    assert v["usable"] is False
    assert v["metrics"] == {}
    assert v["trainable"] is False
    return v["why"]


@case("dispatch: measure('map', ...) resolves")
def _dispatch():
    assert "map" in measure.MEASURERS
    v = measure.measure("map", bands([(60, 130, 60), (40, 80, 200)]))
    assert "terrain_palette" in v["metrics"]
    return "registered"


def main() -> int:
    failed = 0
    for name, fn in CASES:
        try:
            note = fn()
            print(f"  ok    {name}  ({note})")
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
