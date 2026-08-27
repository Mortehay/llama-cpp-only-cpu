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


def road_layer(shape, paths, walkable=None, terrain=None) -> np.ndarray:
    """A road grid from a list of `[(x, y), (x, y)]` segments.

    Roads are axis-aligned in grid space, which is what makes them read as
    built rather than as a wandering trail, and what lets a renderer pick a
    straight/corner tile later.

    `walkable` + `terrain` refuse to lay road across a terrain nobody can walk
    on. A road that crosses a lake is not a road, and the alternative - letting
    it through and looking wrong - is the failure this argument exists to stop.
    """
    h, w = shape
    out = np.zeros((h, w), dtype=np.int16)

    for (x0, y0), (x1, y1) in paths:
        if x0 != x1 and y0 != y1:
            raise ValueError(
                f"road segment ({x0},{y0})->({x1},{y1}) is diagonal; roads are "
                f"axis-aligned in grid space")
        for x, y in _walk(x0, y0, x1, y1):
            if not (0 <= x < w and 0 <= y < h):
                raise ValueError(
                    f"road segment ({x0},{y0})->({x1},{y1}) leaves the "
                    f"{w}x{h} grid at ({x},{y})")
            if terrain is not None and walkable is not None:
                tid = int(terrain[y, x])
                if tid < len(walkable) and not walkable[tid]:
                    raise ValueError(
                        f"road crosses unwalkable terrain {tid} at ({x},{y})")
            out[y, x] = 1
    return out


def _walk(x0, y0, x1, y1):
    if x0 == x1:
        step = 1 if y1 >= y0 else -1
        return [(x0, y) for y in range(y0, y1 + step, step)]
    step = 1 if x1 >= x0 else -1
    return [(x, y0) for x in range(x0, x1 + step, step)]


def scatter(grid: np.ndarray, rules, seed: int = 0,
            roads: np.ndarray | None = None) -> list:
    """Place objects on the terrain they belong to, deterministically.

    `rules` are `{layer, asset, terrain, density, spacing}`:

        layer    "props" or "creatures"
        terrain  terrain ids this may stand on
        density  expected objects per tile of matching terrain
        spacing  minimum tiles between two objects of the same rule

    WHY SPACING RATHER THAN PURE RANDOM. Uniform sampling clumps - it puts
    three trees on adjacent tiles and leaves a bare stretch beside them, which
    reads as a mistake rather than as nature. Rejecting a candidate that lands
    within `spacing` of an earlier one costs almost nothing and is the
    difference between scattered and blotchy.

    Deterministic from `seed`: the same map regenerates identically, which is
    what lets a map be re-rendered without its contents moving.

    Roads are avoided when given. A tree in the middle of the road is not a
    charming detail - `safe_road_radius` exists to keep that corridor clear.
    """
    h, w = grid.shape
    placed, out = [], []
    rng = _rng(seed)

    for index, rule in enumerate(rules):
        layer = rule.get("layer", "props")
        if layer not in OBJECT_LAYERS:
            raise ValueError(f"scatter rule {index} has layer {layer!r}, not "
                             f"one of {', '.join(OBJECT_LAYERS)}")
        terrains = set(rule.get("terrain", []))
        density = float(rule.get("density", 0.0))
        spacing = int(rule.get("spacing", 2))
        if not 0.0 <= density <= 1.0:
            raise ValueError(f"scatter rule {index} density {density} is not "
                             f"a fraction of tiles")

        eligible = [(x, y) for y in range(h) for x in range(w)
                    if int(grid[y, x]) in terrains
                    and (roads is None or int(roads[y, x]) == 0)]
        target = int(round(len(eligible) * density))

        mine = []
        for (x, y) in _shuffled(eligible, rng):
            if len(mine) >= target:
                break
            if any(abs(x - px) < spacing and abs(y - py) < spacing
                   for (px, py) in mine):
                continue
            mine.append((x, y))
            out.append({"x": x, "y": y, "layer": layer,
                        "asset": rule.get("asset"), "want": rule.get("want")})
        placed.append(len(mine))

    return out


def _rng(seed: int):
    """A small deterministic PRNG. Not `random`, so a caller's global seeding
    cannot change what a map contains."""
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(n):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (state >> 17) % max(n, 1)
    return nxt


def _shuffled(items, rng):
    """Fisher-Yates with the supplied PRNG."""
    out = list(items)
    for i in range(len(out) - 1, 0, -1):
        j = rng(i + 1)
        out[i], out[j] = out[j], out[i]
    return out


def picture_size(grid_w: int, grid_h: int,
                 tile_w: int, tile_h: int) -> tuple[int, int]:
    """Canvas an isometric grid needs.

    Both axes scale with the SUM of the grid dimensions, not either one - a
    64x64 map at 64px tiles is 4096px wide. Doubling the grid quadruples the
    pixels, so this is the number to check before queueing a large map.
    """
    return ((grid_w + grid_h) * tile_w // 2,
            (grid_w + grid_h) * tile_h // 2)


# The authored layers, in the order they are written and read.
#
# TWO of these are grids and TWO are object lists, and that split is the whole
# design. A grid layer is one value per tile and can only ever be flat. An
# object layer holds things that STAND on the ground and therefore have to
# occlude each other by position.
#
# LAYERS 3 AND 4 ARE SEPARATE DATA AND ONE DRAW PASS. This is not a shortcut,
# it is the only correct option in an isometric view: a tree one tile in front
# of a creature must cover it, and the same creature one tile further forward
# must cover the tree. Two flat passes cannot express that - whichever is drawn
# second always wins - so `composite` sorts creatures and props together by
# depth. something2's own renderer says the same thing about its Pass B:
# "ground items must join the same sort rather than being drawn in a later
# pass, or they would render on top of entities they are actually behind".
#
# What is NOT here: players and projectiles. Both are runtime, not authored
# map data - their renderer draws blasts and VFX after the depth-sorted pass,
# and a player is never in a map file at all. A `-1` under-ground layer is
# likewise unused, and left unallocated rather than reserved.
LAYERS = ("terrain", "roads", "creatures", "props")
GRID_LAYERS = ("terrain", "roads")
OBJECT_LAYERS = ("creatures", "props")

# Draw order WITHIN one depth. Props settle behind creatures standing on the
# same tile, so a character in a doorway reads as being in front of it.
LAYER_ORDER = {"props": 0, "creatures": 1}


def screen_pos(x: int, y: int, tile_w: int, tile_h: int,
               origin_x: int) -> tuple[int, int]:
    """Grid cell -> the top-left pixel of its tile diamond."""
    return origin_x + (x - y) * tile_w // 2, (x + y) * tile_h // 2


def composite(grid: np.ndarray, tiles: list[Image.Image],
              roads: np.ndarray | None = None,
              road_tiles: list[Image.Image] | None = None,
              objects: list | None = None) -> Image.Image:
    """Draw the layers as one isometric picture.

    Pass A is the flat ground: terrain, then the road overlay on top of it.
    Neither can occlude anything, so both are painted cell by cell.

    Pass B is everything that stands up - creatures and props together, sorted
    by depth. See LAYERS for why together and not in two passes.

    `objects` are `{"x", "y", "image", "layer"}`; each image is anchored by its
    BOTTOM CENTRE to the middle of its tile, because a sprite stands on the
    ground rather than being pasted into the diamond.
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

    if roads is not None and roads.shape != grid.shape:
        raise ValueError(f"roads layer is {roads.shape}, terrain is {grid.shape}")

    # --- Pass A: the flat ground ------------------------------------------
    for depth in range(w + h - 1):
        for x in range(max(0, depth - h + 1), min(w, depth + 1)):
            y = depth - x
            tid = int(grid[y, x])
            if tid < 0 or tid >= len(tiles):
                raise ValueError(
                    f"cell ({x},{y}) has terrain id {tid}, but only "
                    f"{len(tiles)} tiles were supplied")
            sx, sy = screen_pos(x, y, tile_w, tile_h, origin_x)
            out.alpha_composite(tiles[tid], (sx, sy))

            # Roads paint over the terrain in the same cell rather than in a
            # later pass: a road is ground, not something standing on it, so it
            # must not be able to draw over a creature.
            if roads is not None:
                rid = int(roads[y, x])
                if rid > 0:
                    if not road_tiles or rid > len(road_tiles):
                        raise ValueError(
                            f"cell ({x},{y}) has road id {rid}, but "
                            f"{len(road_tiles or [])} road tile(s) were supplied")
                    out.alpha_composite(road_tiles[rid - 1], (sx, sy))

    # --- Pass B: everything that stands up, in ONE sort -------------------
    for obj in _sorted_objects(objects or [], w, h):
        img = obj["image"]
        sx, sy = screen_pos(obj["x"], obj["y"], tile_w, tile_h, origin_x)
        # Bottom-centre of the sprite onto the centre of the tile diamond.
        px = sx + tile_w // 2 - img.width // 2
        py = sy + tile_h // 2 - img.height
        out.alpha_composite(img.convert("RGBA"), (px, py))

    return out


def _sorted_objects(objects, w: int, h: int) -> list:
    """Creatures and props in one back-to-front order.

    Sorted by depth first, then by layer within a depth so a prop settles
    behind a creature sharing its tile. Out-of-bounds placements are refused
    rather than clamped: a prop at (-1, 5) is a generation bug, and drawing it
    at the edge would hide that.
    """
    for o in objects:
        if not (0 <= o["x"] < w and 0 <= o["y"] < h):
            raise ValueError(
                f"{o.get('layer', 'object')} at ({o['x']},{o['y']}) is outside "
                f"the {w}x{h} grid")
        if o.get("layer") not in OBJECT_LAYERS:
            raise ValueError(
                f"object layer {o.get('layer')!r} is not one of "
                f"{', '.join(OBJECT_LAYERS)}")
    return sorted(objects,
                  key=lambda o: (o["x"] + o["y"], LAYER_ORDER[o["layer"]]))


def coverage(grid: np.ndarray, terrains) -> dict:
    """How much of the map each terrain got, by id. Diagnostic, not a gate.

    A terrain at 0% means the painting never produced it - usually a colour the
    adapter will not paint, which is a terrain-set problem rather than a bug.
    """
    total = grid.size
    counts = np.bincount(grid.ravel(), minlength=len(terrains))
    return {terrains[i]["name"]: round(float(counts[i]) / total, 4)
            for i in range(len(terrains))}
