#!/usr/bin/env python3
"""The region graph under test. No model, no GPU, no database.

    docker exec sprite_generator python /app/scripts/smoke-regions.py

The graph is the layer that says a map has PLACES on it rather than just
ground, and every failure it can have is quiet. A town in a lake still renders.
A road across a bay still draws. A graph that silently ignored the hints it was
given still looks like a graph. So the cases here are mostly about whether what
the graph CLAIMS is true of the terrain it was built from.

The two that carry the slice:

  `a port sits on the shore`      - the acceptance criterion, half of it.
  `a road actually reaches it`    - the other half, and the one that fails
                                    silently if `connect` is allowed to route
                                    over water.

`the graph does not move the ground` is the third: re-rolling the semantics
must leave the terrain untouched, which is the whole reason the graph is read
off the terrain rather than authored before it.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/app")

import map_geometry as mg  # noqa: E402
import regions as rg  # noqa: E402

TERRAINS = [
    {"name": "grass", "color": "#4a7c3f", "walkable": True},
    {"name": "water", "color": "#2850c8", "walkable": False},
    {"name": "sand", "color": "#c8be78", "walkable": True},
    {"name": "stone", "color": "#6e6e73", "walkable": True},
]

CASES: list = []


def case(name):
    def wrap(fn):
        CASES.append((name, fn))
        return fn
    return wrap


def island(n=40) -> np.ndarray:
    """Grass with a stone spine, a sand rim, and sea around it.

    Deliberately not a blob: the sea reaches in from the east as a bay, so
    there are pairs of places an L-shaped road genuinely cannot join.
    """
    g = np.ones((n, n), dtype=int)          # water
    g[3:n - 3, 3:n - 3] = 2                 # sand rim
    g[5:n - 5, 5:n - 5] = 0                 # grass
    g[n // 2 - 2:n // 2 + 2, 6:n - 6] = 3   # stone spine
    g[n // 2 - 3:n // 2 + 3, n // 2:n - 4] = 1  # a bay biting in from the east
    return g


def lake_only() -> np.ndarray:
    return np.ones((12, 12), dtype=int)


# --- reading the terrain --------------------------------------------------


@case("the shoreline is every walkable tile touching water")
def _shore():
    g = island()
    shore = rg.shore_mask(g, TERRAINS)
    walk = rg.walkable_mask(g, TERRAINS)
    assert (shore & ~walk).sum() == 0, "shore claimed an unwalkable tile"
    ys, xs = np.where(shore)
    for y, x in list(zip(ys, xs))[:200]:
        touching = [g[y + dy, x + dx] for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    if 0 <= y + dy < g.shape[0] and 0 <= x + dx < g.shape[1]]
        assert 1 in touching, f"({x},{y}) is 'shore' with no water beside it"
    return f"{int(shore.sum())} shore tiles, all of them really are"


@case("a grid id past the end of the terrain list is not a crash")
def _short_terrains():
    g = np.array([[0, 9], [0, 0]])
    walk = rg.walkable_mask(g, TERRAINS[:1])
    assert walk[0][0] and not walk[0][1], walk
    return "treated as unwalkable, not an IndexError mid-build"


@case("the description says where each terrain lies")
def _describe():
    g = np.zeros((20, 20), dtype=int)
    g[:, 15:] = 1
    d = rg.describe(g, TERRAINS)
    water = [t for t in d["terrains"] if t["name"] == "water"][0]
    assert water["lies"] == "east", water
    assert not water["walkable"]
    # A terrain that is not on the map at all must not be described as if it is.
    assert {t["name"] for t in d["terrains"]} == {"grass", "water"}
    return f"water lies {water['lies']}, {int(water['coverage'] * 100)}% cover"


# --- placement ------------------------------------------------------------


@case("a port sits on the shore")
def _port_on_shore():
    g = island()
    shore = rg.shore_mask(g, TERRAINS)
    placed, dropped = rg.place(g, TERRAINS, [
        {"name": "Saltmere", "kind": "port", "terrain": "sand", "where": "north"},
    ], seed=5)
    assert placed, dropped
    p = placed[0]
    assert shore[p["y"], p["x"]], f"port at ({p['x']},{p['y']}) is inland"
    assert not p["relaxed"], p["relaxed"]
    return f"Saltmere at ({p['x']},{p['y']}), on the water"

@case("a place is put on the terrain it asked for")
def _on_its_terrain():
    g = island()
    placed, _ = rg.place(g, TERRAINS, [
        {"name": "Highwatch", "kind": "fort", "terrain": "stone", "where": "centre"},
    ], seed=3)
    assert placed, "nothing placed"
    p = placed[0]
    assert int(g[p["y"], p["x"]]) == 3, f"asked for stone, got id {g[p['y'], p['x']]}"
    return f"on stone at ({p['x']},{p['y']})"


@case("nothing is ever placed in water")
def _never_wet():
    g = island()
    wanted = rg.propose_rules(rg.describe(g, TERRAINS), 9, seed=11)
    placed, _ = rg.place(g, TERRAINS, wanted, seed=11)
    walk = rg.walkable_mask(g, TERRAINS)
    wet = [p for p in placed if not walk[p["y"], p["x"]]]
    assert not wet, wet
    return f"{len(placed)} places, none of them afloat"


@case("nothing is placed out of bounds")
def _in_bounds():
    g = island(16)
    wanted = rg.propose_rules(rg.describe(g, TERRAINS), 9, seed=2)
    placed, _ = rg.place(g, TERRAINS, wanted, seed=2)
    for p in placed:
        assert 0 <= p["x"] < 16 and 0 <= p["y"] < 16, p
    return f"{len(placed)} places inside a 16x16"


@case("a terrain this map does not have is dropped, not relocated")
def _unknown_terrain():
    g = island()
    placed, dropped = rg.place(g, TERRAINS, [
        {"name": "Ashfall", "kind": "ruin", "terrain": "lava", "where": "north"},
    ], seed=1)
    assert not placed and len(dropped) == 1, (placed, dropped)
    assert "lava" in dropped[0]["why"], dropped
    return "dropped with a reason, rather than quietly put somewhere else"


@case("an impossible shore preference is relaxed, and says so")
def _relaxed_is_reported():
    # No water anywhere, so a port cannot be on a shore. It must still be
    # placed - and must not pretend the hint was honoured.
    g = np.zeros((20, 20), dtype=int)
    placed, _ = rg.place(g, TERRAINS, [
        {"name": "Drywell", "kind": "port", "terrain": "grass", "where": "centre"},
    ], seed=1)
    assert placed, "a port on a map with no sea was dropped entirely"
    assert placed[0]["relaxed"] and "shoreline" in placed[0]["relaxed"]
    return f"placed, relaxed: {', '.join(placed[0]['relaxed'])}"


@case("places do not stack on top of each other")
def _spacing():
    g = island()
    wanted = rg.propose_rules(rg.describe(g, TERRAINS), 6, seed=8)
    placed, _ = rg.place(g, TERRAINS, wanted, seed=8)
    for i, a in enumerate(placed):
        for b in placed[i + 1:]:
            assert (a["x"], a["y"]) != (b["x"], b["y"]), (a, b)
    return f"{len(placed)} distinct tiles"


@case("a map with no land places nothing and does not raise")
def _all_water():
    g = lake_only()
    graph = rg.build(g, TERRAINS, count=4, seed=1, use_llm=False)
    assert graph["regions"] == [], graph["regions"]
    assert graph["roads"] == []
    return "empty graph, no exception"


# --- roads ----------------------------------------------------------------


@case("a road actually reaches it")
def _road_reaches():
    g = island()
    graph = rg.build(g, TERRAINS, count=5, seed=4, use_llm=False)
    assert graph["roads"], "no roads at all"
    at = {r["name"]: (r["x"], r["y"]) for r in graph["regions"]}
    for road in graph["roads"]:
        segs = road["segments"]
        assert tuple(segs[0][0]) == at[road["from"]], road
        assert tuple(segs[-1][1]) == at[road["to"]], road
        # Contiguous: each segment starts where the last one ended.
        for a, b in zip(segs, segs[1:]):
            assert a[1] == b[0], road
    return f"{len(graph['roads'])} roads, each ending at both its places"


@case("a road never crosses water")
def _road_dry():
    g = island()
    graph = rg.build(g, TERRAINS, count=7, seed=6, use_llm=False)
    walk = rg.walkable_mask(g, TERRAINS)
    for road in graph["roads"]:
        for (x0, y0), (x1, y1) in road["segments"]:
            for x, y in mg._walk(x0, y0, x1, y1):
                assert walk[y, x], f"{road['from']}->{road['to']} fords ({x},{y})"
    return "every tile of every road is walkable"


@case("the road layer accepts what connect() produced")
def _road_layer_agrees():
    # `road_layer` RAISES on unwalkable ground, so this is the same claim as
    # above made by the code that will actually run - the useful version of it.
    g = island()
    graph = rg.build(g, TERRAINS, count=7, seed=6, use_llm=False)
    walkable = [t["walkable"] for t in TERRAINS]
    grid = mg.road_layer(g.shape, [[tuple(p) for p in seg]
                                   for seg in rg.segments(graph["roads"])],
                         walkable=walkable, terrain=g)
    assert grid.sum() > 0
    return f"{int(grid.sum())} road tiles laid without a refusal"


@case("an unroutable pair is dropped, not forced")
def _unroutable_dropped():
    # A moat splits the map in two. No axis-aligned route exists, and inventing
    # one would put a road across water - the failure `road_layer` refuses.
    g = np.zeros((20, 20), dtype=int)
    g[:, 10] = 1
    placed = [{"name": "West", "kind": "town", "x": 4, "y": 10},
              {"name": "East", "kind": "town", "x": 16, "y": 10}]
    roads, dropped = rg.connect(g, TERRAINS, placed, [["West", "East"]])
    assert not roads and len(dropped) == 1, (roads, dropped)
    assert "walkable" in dropped[0]["why"]
    return "the honest answer: they are not connected by road"


@case("a road to a place that was never placed is dropped")
def _road_to_nowhere():
    g = island()
    placed = [{"name": "Real", "kind": "town", "x": 10, "y": 10}]
    roads, dropped = rg.connect(g, TERRAINS, placed, [["Real", "Imaginary"]])
    assert not roads and dropped, (roads, dropped)
    return "the grammar guarantees the shape, never the sense"


@case("a bay is routed around rather than through")
def _z_route():
    g = np.zeros((24, 24), dtype=int)
    g[8:16, 12:24] = 1                       # a bay open to the east
    placed = [{"name": "N", "kind": "town", "x": 20, "y": 4},
              {"name": "S", "kind": "town", "x": 20, "y": 20}]
    roads, dropped = rg.connect(g, TERRAINS, placed, [["N", "S"]])
    assert roads, dropped
    assert len(roads[0]["segments"]) == 3, "took an L through the water"
    return "three segments, around the head of the bay"


# --- the whole graph ------------------------------------------------------


@case("the graph does not move the ground")
def _terrain_untouched():
    g = island()
    before = g.copy()
    for seed in (1, 2, 3):
        graph = rg.build(g, TERRAINS, count=6, seed=seed, use_llm=False)
        assert graph["regions"], seed
    assert np.array_equal(g, before), "build() mutated the terrain"
    return "re-rolled three times, terrain byte-identical"


@case("re-rolling changes the semantics")
def _reroll_differs():
    g = island()
    a = rg.build(g, TERRAINS, count=6, seed=1, use_llm=False)
    b = rg.build(g, TERRAINS, count=6, seed=99, use_llm=False)
    pa = [(r["x"], r["y"]) for r in a["regions"]]
    pb = [(r["x"], r["y"]) for r in b["regions"]]
    assert pa != pb, "two seeds produced the same places"
    return "different places on the same ground"


@case("the same seed gives the same graph")
def _deterministic():
    g = island()
    a = rg.build(g, TERRAINS, count=6, seed=42, use_llm=False)
    b = rg.build(g, TERRAINS, count=6, seed=42, use_llm=False)
    assert a == b, "same seed, different graph"
    return "re-rendering a map cannot move its towns"


@case("everything proposed is either placed or explained")
def _nothing_vanishes():
    g = island()
    summary = rg.describe(g, TERRAINS)
    wanted = rg.propose_rules(summary, 9, seed=7)
    placed, dropped = rg.place(g, TERRAINS, wanted, seed=7)
    assert len(placed) + len(dropped) == len(wanted), (
        f"{len(wanted)} proposed, {len(placed)} placed, {len(dropped)} dropped")
    assert all(d["why"] for d in dropped)
    return f"{len(wanted)} proposed = {len(placed)} placed + {len(dropped)} explained"


@case("the rule path needs no model and still connects the map")
def _rules_only():
    g = island()
    graph = rg.build(g, TERRAINS, count=6, seed=13, use_llm=False)
    assert graph["source"] == "rules"
    assert len(graph["regions"]) >= 4, graph["regions"]
    reached = {r["from"] for r in graph["roads"]} | {r["to"] for r in graph["roads"]}
    assert len(reached) >= 2, graph["roads"]
    return f"{len(graph['regions'])} places, {len(graph['roads'])} roads, no LLM"


@case("a port is never proposed on a map with no shoreline")
def _no_shore_no_port():
    g = np.zeros((20, 20), dtype=int)
    summary = rg.describe(g, TERRAINS)
    assert summary["shore_tiles"] == 0
    wanted = rg.propose_rules(summary, 9, seed=1)
    assert not any(x["kind"] == "port" for x in wanted), wanted
    return "asked for something the map can actually have"


@case("the grammar makes a landlocked port unsayable")
def _grammar_excludes_port():
    # Found by running it: told in prose that ports need shoreline, the 3B
    # model named three ports on a map with no water. Placement relaxed the
    # constraint for all three and said so - honest, and useless. The enum is
    # the fix, because the prompt is a hint and the schema is a rule.
    dry = rg.describe(np.zeros((20, 20), dtype=int), TERRAINS)
    wet = rg.describe(island(), TERRAINS)

    assert "port" not in rg.possible_kinds(dry), rg.possible_kinds(dry)
    assert "port" in rg.possible_kinds(wet)

    enum = rg._schema(["grass"], rg.possible_kinds(dry), 6)[
        "properties"]["regions"]["items"]["properties"]["kind"]["enum"]
    assert "port" not in enum, enum
    return f"{len(enum)} kinds offered when there is no water, none of them ports"


@case("both proposal paths agree on what this map can have")
def _paths_agree():
    # The rule path and the grammar must not offer different things, or turning
    # the LLM off would silently change what a map can contain.
    for g in (island(), np.zeros((20, 20), dtype=int)):
        summary = rg.describe(g, TERRAINS)
        can = set(rg.possible_kinds(summary))
        assert {x["kind"] for x in rg.propose_rules(summary, 9, seed=3)} <= can
    return "the rule path proposes only what the grammar would allow"


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
