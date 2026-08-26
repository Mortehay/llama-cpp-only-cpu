"""Biome painting -> tilemap -> picture. numpy and PIL only.

Split out of the map router for the reason `tile_geometry.py` records: the
WORKER must import this without dragging in a FastAPI router, pydantic models
and the auth module. The API process must never import `tasks.py`, and this is
the same boundary from the other side.

THE INVERSION THAT MAKES THIS WORK

The biome painting is DATA, not art. It is one pixel per tile, palette-locked
to the declared terrain colours, and nobody ever looks at it. The picture is
derived from the same array the tilemap is derived from, which is what makes
"the menu map and the walkable ground agree" structural rather than a property
to be tested - see `.ai/decisions/0007`.

So quantisation is a LOOKUP, not a nearest-match against whatever the model
happened to paint. The terrain set is declared first and the painting is forced
to it; a pixel that is not already a terrain colour is snapped to the nearest
one in Lab, and there is no third outcome.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

import pixelate

# Below this CIELAB distance two terrains cannot be quantised apart, so they
# would share a tile id. Kept in step with measure.TERRAIN_MIN_LAB_SEPARATION.
MIN_LAB_SEPARATION = 12.0


def parse_color(value: str) -> tuple[int, int, int]:
    """`#4a7c3f` -> (74, 124, 63)."""
    s = value.lstrip("#")
    if len(s) != 6:
        raise ValueError(f"expected #rrggbb, got {value!r}")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def palette_array(terrains) -> np.ndarray:
    """The declared terrain colours as an (n, 3) uint8 array, in id order."""
    if not terrains:
        raise ValueError("a map needs at least one terrain")
    return np.array([parse_color(t["color"]) for t in terrains], dtype=np.uint8)


def separation(palette: np.ndarray) -> float:
    """Smallest pairwise CIELAB distance; inf for a single entry."""
    if len(palette) < 2:
        return float("inf")
    lab = pixelate.srgb_to_lab(palette)
    d = lab[:, None, :] - lab[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", d, d))
    np.fill_diagonal(dist, np.inf)
    return float(dist.min())


def validate_terrains(terrains) -> None:
    """Raise if two terrains would quantise into one tile id.

    Checked before any GPU is spent: two indistinguishable terrains produce a
    map that is silently missing one of them, and the reference art gives no
    hint which.
    """
    names = [t["name"] for t in terrains]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate terrain names: {names}")

    pal = palette_array(terrains)
    sep = separation(pal)
    if sep < MIN_LAB_SEPARATION:
        lab = pixelate.srgb_to_lab(pal)
        d = lab[:, None, :] - lab[None, :, :]
        dist = np.sqrt(np.einsum("ijk,ijk->ij", d, d))
        np.fill_diagonal(dist, np.inf)
        i, j = np.unravel_index(dist.argmin(), dist.shape)
        raise ValueError(
            f"terrains {names[i]!r} and {names[j]!r} are {sep:.1f} apart in "
            f"Lab, under {MIN_LAB_SEPARATION} - they would quantise into one "
            f"tile. Move one of their colours further away.")


def quantize(painting: Image.Image, terrains, size: tuple[int, int]) -> np.ndarray:
    """Biome painting -> (h, w) array of terrain ids.

    NEAREST on the way down, never LANCZOS: interpolating a biome painting
    invents colours between two terrains, and those land on whichever third
    terrain happens to sit between them. A blurred coast becomes a strip of
    desert.
    """
    w, h = size
    img = painting.convert("RGB")
    if img.size != (w, h):
        img = img.resize((w, h), Image.NEAREST)

    pal = palette_array(terrains)
    px = np.asarray(img).reshape(-1, 3)

    lab_px = pixelate.srgb_to_lab(px)
    lab_pal = pixelate.srgb_to_lab(pal)
    d = lab_px[:, None, :] - lab_pal[None, :, :]
    ids = np.einsum("ijk,ijk->ij", d, d).argmin(axis=1)

    return ids.reshape(h, w).astype(np.int16)


def picture_size(grid_w: int, grid_h: int,
                 tile_w: int, tile_h: int) -> tuple[int, int]:
    """Canvas an isometric grid needs.

    Both axes scale with the SUM of the grid dimensions, not either one - a
    64x64 map at 64px tiles is 4096px wide. Doubling the grid quadruples the
    pixels, so this is the number to check before queueing a large map.
    """
    return ((grid_w + grid_h) * tile_w // 2,
            (grid_w + grid_h) * tile_h // 2)


def composite(grid: np.ndarray, tiles: list[Image.Image]) -> Image.Image:
    """Draw a terrain id grid as an isometric picture.

    Back to front by depth (x + y), so a later tile with any overhang covers
    the one behind it. Ground rhombi tessellate and do not overlap, but entity
    placements will sit in this same order and do.
    """
    if not tiles:
        raise ValueError("no tiles to composite")

    tile_w, tile_h = tiles[0].size
    if any(t.size != (tile_w, tile_h) for t in tiles):
        raise ValueError("every terrain tile must be the same size")

    h, w = grid.shape
    out_w, out_h = picture_size(w, h, tile_w, tile_h)
    out = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))

    # Screen x is (x - y) and can go negative; this shifts the leftmost column
    # back onto the canvas.
    origin_x = (h - 1) * tile_w // 2

    for depth in range(w + h - 1):
        for x in range(max(0, depth - h + 1), min(w, depth + 1)):
            y = depth - x
            tid = int(grid[y, x])
            if tid < 0 or tid >= len(tiles):
                raise ValueError(
                    f"cell ({x},{y}) has terrain id {tid}, but only "
                    f"{len(tiles)} tiles were supplied")
            sx = origin_x + (x - y) * tile_w // 2
            sy = (x + y) * tile_h // 2
            out.alpha_composite(tiles[tid], (sx, sy))

    return out


def coverage(grid: np.ndarray, terrains) -> dict:
    """How much of the map each terrain got, by id. Diagnostic, not a gate.

    A terrain at 0% means the painting never produced it - usually a colour the
    adapter will not paint, which is a terrain-set problem rather than a bug.
    """
    total = grid.size
    counts = np.bincount(grid.ravel(), minlength=len(terrains))
    return {terrains[i]["name"]: round(float(counts[i]) / total, 4)
            for i in range(len(terrains))}
