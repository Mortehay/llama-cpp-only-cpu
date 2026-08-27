"""The region graph: what the places on a map ARE, and what connects them.

A tilemap says a tile is grass. It cannot say that the grass at (40, 12) is
the fields outside Ashford, that Ashford is a port, or that a road runs from
it to the ruin on the ridge. That semantic layer is this module.

WHY THE GRAPH IS READ OFF THE TERRAIN, NOT AUTHORED BEFORE IT

The pipeline in `specs/maps/contract.md` puts the region graph first, driving
the painting. That ordering cannot work, and the reason is specific: a landmark
has to sit on real ground, and which ground is real is not known until the
painting has been QUANTISED. A graph authored first can only be validated
afterwards, and "validate afterwards" means either rejecting a perfectly good
graph over one town in a lake, or silently moving it - at which point the graph
and the terrain disagree and the graph was never authoritative anyway.

So the graph is derived from the finished terrain instead. A place is proposed
by kind and by rough compass position; this module finds it an actual tile that
satisfies both. The graph then CANNOT disagree with the terrain, because the
terrain is its input. Re-rolling the graph changes the semantics and leaves the
ground exactly where it was, which is the property Slice 5 is measured on.

WHY THE LLM IS OPTIONAL

`propose_rules` places and connects landmarks with no model at all. The LLM
replaces the naming and the choice of kinds - the part it is actually good at -
and everything downstream runs identically either way. A slow or absent model
must cost a few seconds and a note, never the map.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import requests

import map_geometry

logger = logging.getLogger(__name__)

LLM_URL = os.environ.get("LLM_URL", "http://llm-server:8080")
LLM_TIMEOUT = float(os.environ.get("REGION_LLM_TIMEOUT", "90"))

# What a place can be. Each kind carries the two things placement needs: where
# that sort of place belongs, and how much room it wants.
#
# `shore` is the interesting one. A port that is not on the water is not a
# port, and it is the acceptance criterion for this slice - "a town sits at a
# river mouth and a road reaches it" - so it is a placement CONSTRAINT rather
# than a naming convention.
KINDS = {
    "port":    {"shore": True,  "gap": 6, "flora": False},
    "village": {"shore": None,  "gap": 5, "flora": True},
    "town":    {"shore": None,  "gap": 7, "flora": True},
    "farm":    {"shore": False, "gap": 4, "flora": True},
    "fort":    {"shore": None,  "gap": 6, "flora": False},
    "ruin":    {"shore": None,  "gap": 4, "flora": True},
    "shrine":  {"shore": None,  "gap": 3, "flora": True},
    "camp":    {"shore": None,  "gap": 3, "flora": True},
    "cave":    {"shore": False, "gap": 4, "flora": False},
}

# Compass hints, as fractions of the grid. Cardinals span the full other axis
# so "north" means the top third rather than the top-middle ninth - a hint
# should narrow the search, not corner it.
SECTORS = {
    "north":     ((0.0, 1.0), (0.0, 0.34)),
    "south":     ((0.0, 1.0), (0.66, 1.0)),
    "west":      ((0.0, 0.34), (0.0, 1.0)),
    "east":      ((0.66, 1.0), (0.0, 1.0)),
    "northwest": ((0.0, 0.4), (0.0, 0.4)),
    "northeast": ((0.6, 1.0), (0.0, 0.4)),
    "southwest": ((0.0, 0.4), (0.6, 1.0)),
    "southeast": ((0.6, 1.0), (0.6, 1.0)),
    "centre":    ((0.3, 0.7), (0.3, 0.7)),
}


# ---------------------------------------------------------------------------
# Reading the terrain
# ---------------------------------------------------------------------------


def walkable_mask(grid: np.ndarray, terrains) -> np.ndarray:
    walk = np.array([bool(t.get("walkable", True)) for t in terrains])
    # A grid can name an id past the end of the terrain list only if something
    # upstream is already broken; treat it as unwalkable rather than crashing
    # the whole build over it.
    safe = np.zeros(max(len(walk), int(grid.max()) + 1), dtype=bool)
    safe[:len(walk)] = walk
    return safe[grid]


def shore_mask(grid: np.ndarray, terrains) -> np.ndarray:
    """Walkable tiles with unwalkable ground orthogonally adjacent.

    This is the coastline, the river bank and the lake edge all at once, which
    is right: what a port needs is water it can reach, not a particular biome.
    """
    walk = walkable_mask(grid, terrains)
    wet = ~walk
    touch = np.zeros_like(wet)
    touch[1:, :] |= wet[:-1, :]
    touch[:-1, :] |= wet[1:, :]
    touch[:, 1:] |= wet[:, :-1]
    touch[:, :-1] |= wet[:, 1:]
    return walk & touch


def describe(grid: np.ndarray, terrains) -> dict:
    """A compact description of the map, for a model that cannot see it.

    Coverage plus where each terrain actually sits. Without the second part a
    model asked for "a port in the east" has no idea whether the east is water,
    and will confidently place towns in the sea for this module to reject.
    """
    h, w = grid.shape
    walk = walkable_mask(grid, terrains)
    shore = shore_mask(grid, terrains)

    out = []
    for i, t in enumerate(terrains):
        cells = np.argwhere(grid == i)
        if not len(cells):
            continue
        ys, xs = cells[:, 0], cells[:, 1]
        out.append({
            "name": t["name"],
            "walkable": bool(t.get("walkable", True)),
            "coverage": round(float(len(cells)) / (h * w), 3),
            # Where its middle is, in words rather than coordinates.
            "lies": _compass(float(xs.mean()) / w, float(ys.mean()) / h),
        })

    return {"size": {"w": int(w), "h": int(h)},
            "terrains": out,
            "walkable_fraction": round(float(walk.mean()), 3),
            "shore_tiles": int(shore.sum())}


def _compass(fx: float, fy: float) -> str:
    ns = "north" if fy < 0.38 else "south" if fy > 0.62 else ""
    ew = "west" if fx < 0.38 else "east" if fx > 0.62 else ""
    return (ns + ew) or "centre"


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


def place(grid: np.ndarray, terrains, wanted, seed: int = 0) -> tuple[list, list]:
    """Give every proposed place a real tile. Returns (placed, dropped).

    A place is DROPPED rather than moved somewhere unsuitable when its terrain
    is not on this map at all. Everything softer than that - a sector with no
    room, a shore that does not exist - relaxes one constraint at a time and
    records what it gave up, because a graph that silently ignores its own
    hints is worse than one that admits it could not honour them.
    """
    by_name = {t["name"]: i for i, t in enumerate(terrains)}
    walk = walkable_mask(grid, terrains)
    shore = shore_mask(grid, terrains)
    rng = map_geometry._rng(seed)

    placed, dropped = [], []
    for want in wanted:
        kind = want.get("kind", "village")
        rules = KINDS.get(kind, KINDS["village"])
        tid = by_name.get(want.get("terrain"))

        if tid is None:
            dropped.append({"what": want.get("name"),
                            "why": f"terrain {want.get('terrain')!r} is not on "
                                   f"this map"})
            continue

        spot, gave_up = _find(grid, tid, want.get("where"), rules, walk, shore,
                             placed, rng)
        if spot is None:
            dropped.append({"what": want.get("name"),
                            "why": f"no free {want.get('terrain')} tile left "
                                   f"for it"})
            continue

        x, y = spot
        placed.append({"name": want.get("name") or f"{kind} {len(placed) + 1}",
                       "kind": kind, "terrain": want.get("terrain"),
                       "x": int(x), "y": int(y),
                       "where": _compass(x / grid.shape[1], y / grid.shape[0]),
                       "note": want.get("note") or None,
                       # Said out loud so a reader can see the hint was not
                       # honoured, rather than wondering why the port is inland.
                       "relaxed": gave_up or None})

    return placed, dropped


def _find(grid, tid, where, rules, walk, shore, placed, rng):
    """A tile for one place, relaxing constraints in order of how much they
    matter. Returns (spot, list of constraints given up)."""
    h, w = grid.shape
    base = (grid == tid) & walk
    if not base.any():
        return None, []

    want_shore = rules.get("shore")
    gap = int(rules.get("gap", 4))

    # Tried hardest first. Each entry drops one thing: the sector, then the
    # shore preference, then the spacing from other landmarks.
    attempts = [
        ({"sector": True, "shore": True, "gap": gap}, []),
        ({"sector": False, "shore": True, "gap": gap}, ["compass hint"]),
        ({"sector": True, "shore": False, "gap": gap}, ["shoreline"]),
        ({"sector": False, "shore": False, "gap": gap}, ["compass hint", "shoreline"]),
        ({"sector": False, "shore": False, "gap": 1},
         ["compass hint", "shoreline", "distance from other places"]),
    ]

    for opts, gave_up in attempts:
        mask = base.copy()

        if opts["sector"] and where in SECTORS:
            (x0, x1), (y0, y1) = SECTORS[where]
            box = np.zeros_like(mask)
            box[int(y0 * h):max(int(y1 * h), 1), int(x0 * w):max(int(x1 * w), 1)] = True
            mask &= box

        if opts["shore"] and want_shore is True:
            mask &= shore
        elif opts["shore"] and want_shore is False:
            mask &= ~shore

        candidates = [(int(x), int(y)) for y, x in np.argwhere(mask)]
        if not candidates:
            continue

        for x, y in map_geometry._shuffled(candidates, rng):
            if all(abs(x - p["x"]) >= opts["gap"] or abs(y - p["y"]) >= opts["gap"]
                   for p in placed):
                return (x, y), gave_up

    return None, []


# ---------------------------------------------------------------------------
# Roads
# ---------------------------------------------------------------------------


def connect(grid: np.ndarray, terrains, placed, pairs) -> tuple[list, list]:
    """Axis-aligned routes between named places. Returns (roads, dropped).

    An L first, both ways round, then a Z through a handful of midpoints. Not a
    full pathfinder on purpose: roads here are meant to read as BUILT, and a
    shortest path around a coastline produces a staircase that reads as a
    goat track. A pair that needs one is dropped and said so, which is the
    honest answer - the two places are not connected by road.
    """
    walk = walkable_mask(grid, terrains)
    at = {p["name"]: (p["x"], p["y"]) for p in placed}

    roads, dropped = [], []
    for pair in pairs:
        if len(pair) != 2 or pair[0] not in at or pair[1] not in at:
            dropped.append({"what": " - ".join(str(p) for p in pair),
                            "why": "names a place that was not placed"})
            continue
        if pair[0] == pair[1]:
            continue

        segs = _route(at[pair[0]], at[pair[1]], walk)
        if segs is None:
            dropped.append({"what": f"{pair[0]} - {pair[1]}",
                            "why": "no walkable straight route between them"})
            continue
        roads.append({"from": pair[0], "to": pair[1], "segments": segs})

    return roads, dropped


def _clear(x0, y0, x1, y1, walk) -> bool:
    h, w = walk.shape
    for x, y in map_geometry._walk(x0, y0, x1, y1):
        if not (0 <= x < w and 0 <= y < h) or not walk[y, x]:
            return False
    return True


def _route(a, b, walk):
    (ax, ay), (bx, by) = a, b

    for cx, cy in ((bx, ay), (ax, by)):
        if _clear(ax, ay, cx, cy, walk) and _clear(cx, cy, bx, by, walk):
            return [[[ax, ay], [cx, cy]], [[cx, cy], [bx, by]]]

    # A Z: out to a column or row, along it, then in. Enough to get past a bay
    # without becoming a maze solver.
    #
    # The search covers the WHOLE grid, not just the span between the two
    # places, and that is the point. Two towns on the same side of a bay have
    # no column between them that is dry - the way out is to go the other way
    # first. Ordering by total detour keeps the shortest route that works, so
    # scanning wide costs nothing when a near one is fine.
    h, w = walk.shape

    for mx in sorted(range(w), key=lambda m: abs(m - ax) + abs(m - bx)):
        if (_clear(ax, ay, mx, ay, walk) and _clear(mx, ay, mx, by, walk)
                and _clear(mx, by, bx, by, walk)):
            return [[[ax, ay], [mx, ay]], [[mx, ay], [mx, by]],
                    [[mx, by], [bx, by]]]

    for my in sorted(range(h), key=lambda m: abs(m - ay) + abs(m - by)):
        if (_clear(ax, ay, ax, my, walk) and _clear(ax, my, bx, my, walk)
                and _clear(bx, my, bx, by, walk)):
            return [[[ax, ay], [ax, my]], [[ax, my], [bx, my]],
                    [[bx, my], [bx, by]]]

    return None


def segments(roads) -> list:
    """Every road's segments, flattened for `map_geometry.road_layer`."""
    return [seg for r in roads for seg in r["segments"]]


# ---------------------------------------------------------------------------
# Proposing what the places are
# ---------------------------------------------------------------------------


def propose_rules(summary: dict, count: int, seed: int = 0) -> list:
    """A graph with no model involved. The floor, not the fallback.

    Spread over the walkable terrains, cycling kinds and sectors so the result
    is varied without being random. The names are frankly placeholder - naming
    is the part the LLM earns its keep on - but the SHAPE is identical, so
    everything downstream is exercised whether or not a model answered.
    """
    walkable = [t["name"] for t in summary["terrains"] if t["walkable"]]
    if not walkable:
        return []

    # Same filter the grammar gets, so the two paths cannot propose different
    # things about the same map.
    can = set(possible_kinds(summary))
    order = [k for k in ["town", "village", "port", "ruin", "farm", "shrine",
                         "fort", "camp", "cave"] if k in can]
    where = ["centre", "north", "southeast", "west", "northeast", "south",
             "northwest", "east", "southwest"]
    rng = map_geometry._rng(seed)

    out = []
    for i in range(count):
        kind = order[i % len(order)]
        out.append({"name": f"{kind.title()} {i + 1}",
                    "kind": kind,
                    "terrain": walkable[rng(len(walkable))],
                    "where": where[i % len(where)]})
    return out


def possible_kinds(summary: dict) -> list:
    """The kinds this particular map can actually support.

    A map with no water cannot have a port, and the fix belongs HERE rather
    than in a later check. Told in prose that "ports must be somewhere with
    shoreline", a 3B model cheerfully names three ports on a landlocked map and
    placement then has to relax the constraint for all of them - which is
    honest but useless. Removing `port` from the enum makes it unsayable.

    That is the whole point of driving the grammar from the map (plan Q5): the
    prompt is a hint, the schema is the rule.
    """
    if summary.get("shore_tiles"):
        return sorted(KINDS)
    return sorted(k for k, v in KINDS.items() if v.get("shore") is not True)


def _schema(terrain_names, kinds, count: int) -> dict:
    """The answer's shape, as JSON Schema.

    llama.cpp compiles this to a GBNF grammar itself, so the model CANNOT emit
    a kind that does not exist or a terrain this map does not have. That is the
    reliability mechanism (plan Q5) - not the prompt wording, which a 3B model
    will cheerfully ignore.
    """
    return {
        "type": "object",
        "properties": {
            "regions": {
                "type": "array", "minItems": 1, "maxItems": count,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "kind": {"type": "string", "enum": list(kinds)},
                        "terrain": {"type": "string", "enum": list(terrain_names)},
                        "where": {"type": "string", "enum": sorted(SECTORS)},
                        "note": {"type": "string"},
                    },
                    "required": ["name", "kind", "terrain", "where"],
                },
            },
            "roads": {
                "type": "array",
                "items": {"type": "array", "minItems": 2, "maxItems": 2,
                          "items": {"type": "string"}},
            },
        },
        "required": ["regions", "roads"],
    }


def _model() -> str | None:
    """Which model to route to. llama.cpp is in ROUTER mode, so a request with
    no `model` is a 400 rather than a default - the same trap `worlds.py` hit."""
    r = requests.get(f"{LLM_URL}/v1/models", timeout=10)
    data = r.json().get("data") or []
    return data[0]["id"] if data else None


def propose(summary: dict, theme: str | None, count: int,
            seed: int = 0) -> tuple[list, list, str]:
    """Ask the model what the places are. Returns (regions, road pairs, note).

    Never raises. Every failure falls back to `propose_rules`, because the map
    is worth more than the names on it.
    """
    terrain_names = [t["name"] for t in summary["terrains"] if t["walkable"]]
    if not terrain_names:
        return [], [], "no walkable terrain - nowhere to put anything"

    kinds = possible_kinds(summary)

    ground = "; ".join(
        f"{t['name']} covers {int(t['coverage'] * 100)}% and lies "
        f"{t['lies']}" + ("" if t["walkable"] else " (water, impassable)")
        for t in summary["terrains"])

    prompt = (
        f"You are naming the places on a {summary['size']['w']}x"
        f"{summary['size']['h']} fantasy map"
        + (f" for: {theme}." if theme else ".") + "\n\n"
        f"The ground is already fixed: {ground}. "
        + (f"There are {summary['shore_tiles']} tiles of shoreline.\n\n"
           if summary["shore_tiles"] else
           "There is no water on this map at all.\n\n")
        + f"Name up to {count} places, choosing from: {', '.join(kinds)}. "
        f"Put each on a terrain it belongs on and in the part of the map that "
        f"suits it. Then list the roads between them as pairs of names; "
        f"connect them into one network rather than every place to every "
        f"other. Use the names you invented, spelled identically.")

    try:
        model = _model()
        if not model:
            return propose_rules(summary, count, seed), [], (
                "no text model loaded in llama.cpp - places named by rule")

        body = {"model": model, "temperature": 0.8, "max_tokens": 900,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "region_graph", "strict": True,
                                    "schema": _schema(terrain_names, kinds, count)}}}

        # Twice, because a cold model can return an empty first completion
        # while it loads - the same behaviour `worlds.py` had to retry around.
        for attempt in range(2):
            r = requests.post(f"{LLM_URL}/v1/chat/completions", json=body,
                              timeout=LLM_TIMEOUT)
            if r.status_code != 200:
                logger.warning("region LLM %s: %s", r.status_code, r.text[:200])
                continue
            content = ((r.json().get("choices") or [{}])[0]
                       .get("message", {}).get("content") or "").strip()
            if content:
                break
        else:
            return propose_rules(summary, count, seed), [], (
                "llama.cpp did not answer - places named by rule")

        data = json.loads(content)
    except Exception as e:
        logger.warning("region graph fell back to rules: %s", e)
        return propose_rules(summary, count, seed), [], (
            f"LLM unavailable ({type(e).__name__}) - places named by rule")

    regions = [r for r in (data.get("regions") or []) if r.get("name")][:count]
    if not regions:
        return propose_rules(summary, count, seed), [], (
            "LLM named no places - named by rule instead")

    # The grammar guarantees the SHAPE, never the sense. A model told to reuse
    # its own names still invents a road to somewhere it never mentioned, so
    # pairs are filtered against the names that actually exist.
    known = {r["name"] for r in regions}
    pairs = [p for p in (data.get("roads") or [])
             if isinstance(p, list) and len(p) == 2
             and p[0] in known and p[1] in known and p[0] != p[1]]

    note = f"{len(regions)} place(s) named by llama.cpp"
    return regions, pairs, note


# ---------------------------------------------------------------------------
# The whole thing
# ---------------------------------------------------------------------------


def build(grid: np.ndarray, terrains, count: int = 6,
          theme: str | None = None, seed: int = 0,
          use_llm: bool = True) -> dict:
    """Terrain in, region graph out.

    The one entry point a caller needs. Everything it reports as dropped was
    proposed and could not be honoured - that list is the difference between a
    graph that fits the map and one that merely claims to.
    """
    summary = describe(grid, terrains)

    if use_llm:
        wanted, pairs, note = propose(summary, theme, count, seed)
        source = "rules" if "by rule" in note else "llm"
    else:
        wanted, pairs, note = propose_rules(summary, count, seed), [], (
            "places named by rule (LLM not requested)")
        source = "rules"

    placed, dropped = place(grid, terrains, wanted, seed)

    # Nothing proposed any roads - the rule path never does - so string the
    # places together in placement order. A map whose towns are unreachable is
    # not better than one with a slightly arbitrary road.
    if not pairs and len(placed) > 1:
        pairs = [[placed[i]["name"], placed[i + 1]["name"]]
                 for i in range(len(placed) - 1)]

    roads, road_drops = connect(grid, terrains, placed, pairs)

    return {"source": source, "note": note,
            "regions": placed, "roads": roads,
            "dropped": dropped + road_drops}
