#!/usr/bin/env python3
"""World spec generation under test. No model, no GPU, no network.

    docker exec sprite_generator python /app/scripts/smoke-world-gen.py

The assertions that matter are the density ones. something2's own numbers, read
from their repo on 2026-08-26:

    tier     per1000   per screen (x0.225)
    sparse         9                    2.0
    normal        18                    4.1
    dense         36                    8.1
    horde         62                   14.0
    swarm         89                   20.0

and `biomes.creature_density` multiplies that - Meadow 0.5, Mire 2.0. A
Meadow world at 'normal' therefore holds ~2 creatures per screen, which is the
emptiness this generator exists to remove.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/app")

import world_gen as wg  # noqa: E402
import world_preview as wp  # noqa: E402

CASES = []


def case(name):
    def wrap(fn):
        CASES.append((name, fn))
        return fn
    return wrap


@case("their tier table, reproduced per screen")
def _tiers():
    want = {"sparse": 2.0, "normal": 4.1, "dense": 8.1, "horde": 14.0, "swarm": 20.0}
    for tier, expect in want.items():
        got = wg.DENSITY_TIERS[tier] * wg.TILES_PER_SCREEN / 1000.0
        assert abs(got - expect) < 0.1, f"{tier}: {got:.2f} vs {expect}"
    return ", ".join(f"{k} {v}" for k, v in want.items())


@case("the Meadow trap: 'normal' really is ~2 per screen")
def _meadow():
    ps = wg.per_screen("normal", ["Meadow"])
    assert ps < wg.EMPTY_BELOW_PER_SCREEN, ps
    return (f"normal x Meadow(0.5) = {ps:.1f}/screen - under the "
            f"{wg.EMPTY_BELOW_PER_SCREEN} floor, i.e. empty")


@case("solving for a target beats naming a tier")
def _solve():
    # The same target through a thin biome and a thick one must not produce
    # the same tier - that is the entire fix.
    thin = wg.choose_density(6.0, ["Meadow"])
    thick = wg.choose_density(6.0, ["Mire"])
    assert thin != thick, f"both chose {thin}"
    assert wg.per_screen(thin, ["Meadow"]) >= wg.EMPTY_BELOW_PER_SCREEN
    assert wg.per_screen(thick, ["Mire"]) >= wg.EMPTY_BELOW_PER_SCREEN
    return (f"Meadow -> {thin} ({wg.per_screen(thin, ['Meadow']):.1f}/screen), "
            f"Mire -> {thick} ({wg.per_screen(thick, ['Mire']):.1f}/screen)")


@case("a generated region has no empty worlds")
def _no_empty():
    spec = wg.plan_region("Emerald Reach", world_count=8, target_per_screen=6.0)
    rep = wg.report(spec)
    empty = [r["key"] for r in rep["worlds"] if r["verdict"] == "EMPTY"]
    assert not empty, f"empty worlds: {empty}"
    assert rep["totals"]["min_per_screen"] >= wg.EMPTY_BELOW_PER_SCREEN
    return (f"{rep['totals']['worlds']} worlds, "
            f"{rep['totals']['min_per_screen']}-{rep['totals']['max_per_screen']}"
            f"/screen, {rep['totals']['creatures']} creatures")


@case("asking for emptiness still reports it")
def _honest():
    spec = wg.plan_region("Hollow", world_count=3, target_per_screen=0.5)
    rep = wg.report(spec)
    assert rep["totals"]["empty_worlds"] > 0, "a 0.5/screen region claimed to be fine"
    assert not rep["ok"]
    return f"{rep['totals']['empty_worlds']} world(s) flagged EMPTY, as asked for"


@case("every world is reachable and there is exactly one entry")
def _graph():
    spec = wg.plan_region("Linkage", world_count=9, target_per_screen=6.0)
    rep = wg.report(spec)
    bad = [p for p in rep["problems"] if "unreachable" in p or "entry worlds" in p]
    assert not bad, bad
    keys = {w["key"] for w in spec["worlds"]}
    for l in spec["links"]:
        assert l["from"] in keys and l["to"] in keys, l
    return f"{len(spec['links'])} links over {len(keys)} worlds, 1 entry"


@case("links are symmetric across the grid")
def _symmetry():
    spec = wg.plan_region("Mirror", world_count=9, target_per_screen=6.0)
    have = {(l["from"], l["edge"], l["to"]) for l in spec["links"]}
    for f, e, t in have:
        back = (t, wg.OPPOSITE[e], f)
        assert back in have, f"{f} -{e}-> {t} has no return link"
    return f"{len(have)} links, all reciprocal"


@case("thin flora is repaired, not just reported")
def _flora():
    # Catacombs and Deepvault carry one flora type each, and the same one.
    thin = ["Catacombs", "Deepvault"]
    assert len(wg.flora_types(thin)) < wg.THIN_FLORA_BELOW, wg.flora_types(thin)
    fixed = wg.enrich_flora(thin)
    assert len(wg.flora_types(fixed)) >= wg.THIN_FLORA_BELOW, fixed
    assert len(fixed) == 3, "repair should add exactly one biome"

    spec = wg.plan_region("Bare", world_count=1, target_per_screen=6.0,
                          entry_biomes=("Arid Dunes",))
    rep = wg.report(spec)
    assert not any("flora" in p for p in rep["problems"]), rep["problems"]
    return f"{thin} -> {fixed}, and a bare region self-repairs"


@case("deliberately desolate biomes are exempt from the flora floor")
def _desolate():
    """Seven biomes carry no flora BY DESIGN.

    Applying the floor uniformly would fire on them forever and the repair
    would chase something unrepairable - the floor overruling the author.
    """
    assert len(wg.FLORA_OPTIONAL) == 7, sorted(wg.FLORA_OPTIONAL)
    for name in wg.FLORA_OPTIONAL:
        assert wg.enrich_flora([name]) == [name], \
            f"repair added a biome to {name}, which is meant to be bare"
        rep = wg.report({"worlds": [{
            "key": "w", "name": "w", "biomes": [name], "density": "normal",
            "width": 128, "height": 128, "chunk_size": 32, "is_entry": True,
            "allowed_creature_types": wg.spawnable([name])}], "links": []})
        assert not any("read bare" in p for p in rep["problems"]), (name, rep["problems"])
    return f"{len(wg.FLORA_OPTIONAL)} desolate biomes exempt: {sorted(wg.FLORA_OPTIONAL)[:3]}…"


@case("a world still short of flora is reported, not hidden")
def _flora_reported():
    # Creatures must be ones Arid Dunes actually admits, or the ZERO-creatures
    # check fires first and the flora problem never gets evaluated.
    thin = wg.report({"worlds": [{
        "key": "w", "name": "w", "biomes": ["Arid Dunes"], "density": "dense",
        "width": 128, "height": 128, "is_entry": True,
        "allowed_creature_types": wg.spawnable(["Arid Dunes"])}], "links": []})
    assert any("read bare" in p for p in thin["problems"]), thin["problems"]
    return "a hand-authored thin spec is still flagged"


@case("creatures come from the BIOME table, never from entity_types")
def _spawnable():
    """The regression that shipped: real names that can never spawn.

    something2 gates spawning on `biomes.creature_types`, not on a creature
    existing in `entity_types`. Selecting from the P4 bestiary produced names
    that were real, passed their unknown-creature check, and seeded zero.
    """
    spec = wg.plan_region("Variety", world_count=6, target_per_screen=6.0)
    rep = wg.report(spec)
    for r in rep["worlds"]:
        allowed = set(r["creature_types"])
        admits = set(wg.spawnable(r["biomes"]))
        assert allowed, f"{r['key']} allows nothing"
        assert allowed <= admits, (
            f"{r['key']} allows {allowed - admits}, which its biomes "
            f"{r['biomes']} cannot spawn")
    return f"{rep['totals']['creature_types']} types, all spawnable by their biomes"


@case("a biome/creature mismatch is reported as EMPTY, not as healthy")
def _mismatch():
    """The verdict that lied. A hand-built world with P4 names and a starter
    biome must read as zero, not as whatever its density tier budgeted.

    The example had to change on 2026-08-27: this used `Beast Swarm`, which the
    widening put INTO Meadow, so the world stopped being empty and the test
    failed - correctly, by noticing its own premise had expired. Meadow now
    admits the Beast line, so the unspawnable names have to come from a family
    it does not have.
    """
    unspawnable = ["Woodland Line", "Desert Swarm"]
    assert not set(unspawnable) & set(wg.BIOME_CREATURES["Meadow"]), (
        f"the example is stale again: Meadow now admits {unspawnable}")

    rep = wg.report({"worlds": [{
        "key": "w", "name": "w", "biomes": ["Meadow"], "density": "swarm",
        "width": 128, "height": 128, "is_entry": True,
        # Real rows in entity_types; not in Meadow's creature_types.
        "allowed_creature_types": unspawnable}],
        "links": []})
    row = rep["worlds"][0]
    assert row["verdict"] == "EMPTY", row["verdict"]
    assert row["creatures"] == 0 and row["per_screen"] == 0.0, row
    assert any("ZERO creatures" in p for p in rep["problems"]), rep["problems"]
    assert rep["totals"]["empty_worlds"] == 1
    return "swarm tier + unspawnable names -> 0, flagged"


@case("the biome table matches something2's, all 32 of them")
def _biome_table():
    assert len(wg.BIOME_CREATURES) == 32, len(wg.BIOME_CREATURES)
    # Every biome this generator can CHOOSE must be in the spawn table, or a
    # world could be built whose creatures cannot be derived.
    missing = [b for b in wg.BIOMES if b not in wg.BIOME_CREATURES]
    assert not missing, missing
    # Only Swarm/Skirmisher/Line roles appear in any biome list; the other six
    # P4 roles exist in entity_types and can never spawn from a biome.
    roles = {t.rsplit(" ", 1)[-1] for v in wg.BIOME_CREATURES.values()
             for t in v if " " in t}
    assert roles <= {"Swarm", "Skirmisher", "Line"}, roles
    return f"32 biomes, {len(wg.BIOMES)} selectable, roles {sorted(roles)}"


@case("leaders are rationed, because auras are O(leaders x population)")
def _leaders():
    spec = wg.plan_region("Leaders", world_count=9, target_per_screen=8.0)
    rep = wg.report(spec)
    for r in rep["worlds"]:
        assert r["leaders"] <= wg.MAX_LEADER_ROLES_PER_WORLD, (r["key"], r["leaders"])
    assert not any("leader roles" in p for p in rep["problems"])
    return (f"{rep['totals']['leaders']} leader role(s) across "
            f"{rep['totals']['worlds']} worlds, cap "
            f"{wg.MAX_LEADER_ROLES_PER_WORLD}/world")


@case("the four-creature ceiling is gone, and regions now meet the floor")
def _variety_repair():
    """This test previously asserted a CEILING that no longer exists.

    With only the five surface biomes, every combination admitted four
    creatures - Slime, Wolf, Bat, Skeleton - so a variety floor of five was
    unreachable and the honest behaviour was to report the shortfall. The full
    32-biome table lifted that, and the 2026-08-27 widening lifted it again:
    Mire ALONE now meets the floor, so `enrich_variety` correctly declines to
    add anything to it. This asserted that the repair fired, which is no longer
    a thing a healthy biome should make it do.
    """
    ceiling = len({c for b in wg.BIOMES for c in wg.BIOMES[b]["creatures"]})
    assert ceiling >= wg.THIN_VARIETY_BELOW, ceiling

    # A biome that is already varied enough is left alone. Adding a second one
    # would be churn, and on the surface it would also break the descent.
    assert len(wg.spawnable(["Mire"])) >= wg.THIN_VARIETY_BELOW
    assert wg.enrich_variety(["Mire"], [1, 5]) == ["Mire"], "repaired a healthy biome"

    # A genuinely thin one still gets help. Every biome now carries at least
    # its own line family, so thinness has to be constructed to be tested.
    thin = min(wg.BIOMES, key=lambda b: len(wg.BIOMES[b]["creatures"]))
    fixed = wg.enrich_variety([thin], [1, 5])
    assert len(wg.spawnable(fixed)) >= len(wg.spawnable([thin])), (thin, fixed)
    assert len(fixed) <= 2, "repair should add at most one biome"

    rep = wg.report(wg.plan_region("Kinds", world_count=9, target_per_screen=6.0))
    assert not any("read repetitive" in p for p in rep["problems"]), rep["problems"]
    return (f"catalogue admits {ceiling} creatures; thinnest biome {thin!r} has "
            f"{len(wg.BIOMES[thin]['creatures'])}; a 9-world region reports "
            f"{rep['totals']['creature_types']} kinds and no repetition")


@case("the five original biomes carry their P4 lines")
def _widening():
    """The widening something2 applied on 2026-08-27, mirrored here.

    Those five shipped with legacy creatures only, which capped a SURFACE-ONLY
    region at four kinds however many biomes it used. The P3 biomes were caught
    because they shipped empty; these were populated with legacy names, so
    nothing flagged them.

    Nineteen is the checksum on the transcription, and it is why this asserts a
    number rather than a shape: a single mistyped or duplicated creature name
    would not sum to 19. The Frozen Waste row in ADR 0008 was wrong once
    already, from hand-copying exactly this table.
    """
    five = ["Meadow", "Deep Forest", "Arid Dunes", "Frozen Waste", "Mire"]
    kinds = wg.spawnable(five)
    assert len(kinds) == 19, f"{len(kinds)} kinds, expected 19: {sorted(kinds)}"

    # Every one of the five gained its own line family; none is still legacy-only.
    legacy = {"Slime", "Wolf", "Bat", "Skeleton"}
    for b in five:
        gained = set(wg.BIOMES[b]["creatures"]) - legacy
        assert len(gained) == 3, f"{b} has {sorted(gained)}, expected 3 P4 lines"

    # And a surface-only region is no longer thin.
    assert len(kinds) >= wg.THIN_VARIETY_BELOW, len(kinds)
    return f"4 -> {len(kinds)} kinds without leaving the surface"


@case("a region descends: surface at the entry, deep further out")
def _descent():
    """Coherence and variety are the same fix.

    Scattering deep biomes among surface ones would validate and read as
    nonsense - Catacombs beside a meadow. Descending into them is the ordinary
    shape of a region, and it is what reaches the P4 families at all.
    """
    n = 9
    spec = wg.plan_region("Descent", world_count=n, target_per_screen=6.0)
    # By INDEX, not by grid distance. The layout is a spiral, so distance from
    # the origin says nothing about position in the journey - asserting on it
    # is what let bands and biomes disagree in the first place.
    for i, w in enumerate(spec["worlds"]):
        depths = {wg.BIOMES[b]["depth"] for b in w["biomes"]}
        if i <= max(1, n // 3):
            assert depths == {"surface"}, (w["key"], i, w["biomes"])
        else:
            assert "deep" in depths, (w["key"], i, w["biomes"])

    entry = spec["worlds"][0]
    assert entry["is_entry"] and "level_band" not in entry
    last = spec["worlds"][-1]
    assert "deep" in {wg.BIOMES[b]["depth"] for b in last["biomes"]}
    return (f"entry {entry['biomes']} -> last {last['biomes']}, "
            f"{len(wg.SURFACE)} surface / {len(wg.DEEP)} deep")


@case("repairs stay on their side of the surface/deep line")
def _repair_depth():
    """A repair must not undo the descent.

    Infernal Gate admits three creatures, under the variety floor, so it gets
    a companion biome - and the first version reached across the line and put
    a Meadow inside it. Same-depth candidates are tried first now.
    """
    for name in ("Infernal Gate", "Abyssal Rift", "The Maw", "Catacombs"):
        fixed = wg.enrich_variety([name], None)
        depths = {wg.BIOMES[b]["depth"] for b in fixed}
        assert depths == {"deep"}, (name, fixed, depths)

    spec = wg.plan_region("Coherent", world_count=9, target_per_screen=6.0)
    for w in spec["worlds"]:
        depths = {wg.BIOMES[b]["depth"] for b in w["biomes"]}
        assert len(depths) == 1, (w["key"], w["biomes"], depths)
    return "deep worlds repaired with deep biomes; no mixed-depth world"


@case("level bands ramp monotonically, and surface worlds carry none")
def _bands():
    """something2 found these oscillating 3-7 / 5-9, with the deepest world
    landing back at the same band as world 1 - a player fights through six
    worlds and arrives somewhere no harder than they started.

    The cause: depth was driven by grid distance, but the layout is a SPIRAL,
    so distance and position-in-the-journey are unrelated. The LLM authors
    biomes by index, so biomes descended while bands ramped by distance.
    """
    for n in (4, 8, 12):
        spec = wg.plan_region("Bands", world_count=n, target_per_screen=6.0)
        bands, surface_with_band = [], []
        for w in spec["worlds"]:
            depths = {wg.BIOMES[b]["depth"] for b in w["biomes"]}
            if "level_band" in w:
                bands.append(w["level_band"])
                if depths == {"surface"}:
                    surface_with_band.append(w["key"])
            else:
                assert depths == {"surface"}, (w["key"], w["biomes"])

        assert not surface_with_band, surface_with_band
        # Monotonic, and strictly so - equal consecutive bands would still read
        # as a flat stretch.
        for a, b in zip(bands, bands[1:]):
            assert b[0] > a[0] and b[1] > a[1], (n, a, b)
        for lo, hi in bands:
            assert hi > lo, (lo, hi)
    return f"strictly increasing over 4/8/12-world regions, none on surface"


@case("bandwidth is projected, so a dense big map is a visible decision")
def _bandwidth():
    """something2's surviving invariant: ~920 B/s per creature, per socket.

    Their tick loop held ~20 Hz in every world measured, so this is a network
    cost and not a CPU one - and it should not be reachable by accident from a
    theme string.
    """
    quiet = wg.socket_kib_s(4.0, 32)
    loud = wg.socket_kib_s(20.0, 32)
    assert loud > quiet * 4, (quiet, loud)

    rep = wg.report({"worlds": [{
        "key": "w", "name": "w", "biomes": ["Infernal Gate"], "density": "swarm",
        "width": 224, "height": 224, "chunk_size": 32, "is_entry": True,
        "allowed_creature_types": wg.spawnable(["Infernal Gate"])}], "links": []})
    assert any("KiB/s" in p for p in rep["problems"]), rep["problems"]
    assert rep["worlds"][0]["socket_kib_s"] > wg.SOCKET_WARN_KIB_S
    return f"normal {quiet:.0f} KiB/s, swarm {loud:.0f} KiB/s, warned above {wg.SOCKET_WARN_KIB_S}"


@case("variety is reported when a world would read repetitive")
def _variety():
    thin = wg.report({"worlds": [{
        "key": "w", "name": "w", "biomes": ["Meadow"], "density": "dense",
        "width": 128, "height": 128, "is_entry": True,
        "allowed_creature_types": ["Slime", "Wolf"]}], "links": []})
    assert any("read repetitive" in p for p in thin["problems"]), thin["problems"]
    return "2 creature types flagged as repetitive"


@case("roads stop short of the wall ring")
def _wall_ring():
    """Their mapSpec.js:516 rejects any road point outside 1..size-2.

    The outermost ring is wall, and `stampBounds` overwrites anything authored
    there - so a road drawn to the edge is never rendered while still widening
    the safe corridor. 14 errors, every world, found by their validator.
    """
    for n in (1, 6, 9):
        spec = wg.plan_region("Ring", world_count=n, target_per_screen=6.0)
        for w in spec["worlds"]:
            size = w["width"]
            for line in w.get("roads", []):
                for row, col in line:
                    assert 1 <= row <= size - 2, (w["key"], "row", row, size)
                    assert 1 <= col <= size - 2, (w["key"], "col", col, size)
    return "1..size-2 on every road point, 3 region sizes"


@case("roads reach every exit, and stay inside the world")
def _roads():
    spec = wg.plan_region("Roads", world_count=9, target_per_screen=6.0)
    exits = {}
    for l in spec["links"]:
        exits.setdefault(l["from"], []).append(l["edge"])
    for w in spec["worlds"]:
        want = exits.get(w["key"], [])
        roads = w.get("roads", [])
        assert len(roads) == len(want), (w["key"], len(roads), want)
        assert w.get("safe_road_radius") == wg.SAFE_ROAD_RADIUS
        size = w["width"]
        for line in roads:
            assert len(line) == 2, line
            for row, col in line:
                assert 0 <= row < size and 0 <= col < size, (w["key"], row, col)
            # Every road starts at the middle, which is where the waypoint is.
            assert line[0] == [size // 2, size // 2], line[0]
    total = sum(len(w.get("roads", [])) for w in spec["worlds"])
    return f"{total} roads over 9 worlds, all in bounds and joined at the middle"


@case("one waypoint per world, in pixels not tiles")
def _waypoints():
    spec = wg.plan_region("Waypoints", world_count=6, target_per_screen=6.0)
    for w in spec["worlds"]:
        wps = w.get("waypoints", [])
        # Their one_waypoint_per_world migration makes this a constraint.
        assert len(wps) <= 1, (w["key"], len(wps))
        for p in wps:
            mid = w["width"] // 2
            assert p["x"] == wg.to_px(mid) and p["y"] == wg.to_px(mid), p
            assert p["name"]
    return f"{sum(len(w.get('waypoints', [])) for w in spec['worlds'])} waypoints, all at {wg.TILE_PX}px/tile"


@case("the village contains its own spawn, and its gate is a real exit")
def _village():
    for n in (1, 4, 9):
        spec = wg.plan_region("Village", world_count=n, target_per_screen=6.0)
        exits = {}
        for l in spec["links"]:
            exits.setdefault(l["from"], []).append(l["edge"])
        entry = [w for w in spec["worlds"] if w["is_entry"]][0]
        v = entry["village"]

        # Every village in vale-region contains its own spawn point.
        col = (v["spawn_x"] - wg.TILE_PX // 2) // wg.TILE_PX
        row = (v["spawn_y"] - wg.TILE_PX // 2) // wg.TILE_PX
        assert v["min_row"] <= row < v["min_row"] + v["height"], (n, row, v)
        assert v["min_col"] <= col < v["min_col"] + v["width"], (n, col, v)

        # A gate onto an edge with no link opens onto the map boundary.
        mine = exits.get(entry["key"], [])
        if mine:
            assert v["gate_edge"] in mine, (n, v["gate_edge"], mine)
    return "spawn inside the walls and gate on a real exit for 1, 4 and 9 worlds"


@case("tiles-vs-pixels is CAUGHT, not merely documented")
def _units():
    # The trap: tile 64 of a 128-wide world is also a legal pixel coordinate
    # (64 < 12800), so no range check finds it. Only the tile-centre offset
    # does - a real pixel position is always n*100+50.
    assert wg.is_tile_centre(6450) and not wg.is_tile_centre(64)
    assert not wg.is_tile_centre(6400), "6400 is a tile EDGE, not a centre"

    good = wg.plan_region("Units", world_count=4, target_per_screen=6.0)
    assert not [p for p in wg.report(good)["problems"] if "PIXELS" in p]

    broken = wg.plan_region("Units", world_count=4, target_per_screen=6.0)
    entry = [w for w in broken["worlds"] if w["is_entry"]][0]
    mid = entry["width"] // 2
    entry["waypoints"] = [{"x": mid, "y": mid, "name": "wrong units"}]
    entry["entry_spawn"] = {"x": mid, "y": mid}
    problems = wg.report(broken)["problems"]
    caught = [p for p in problems if "PIXELS" in p]
    assert len(caught) == 4, f"expected 4 unit problems, got {len(caught)}"

    # And the mirror: a road written in pixels is out of the tile grid.
    entry["roads"] = [[[mid, mid], [wg.to_px(mid), wg.to_px(mid)]]]
    assert any("roads are TILES" in p for p in wg.report(broken)["problems"])
    return "tile-index positions and pixel-valued roads both caught"


@case("the spec matches something2's shape")
def _shape():
    spec = wg.plan_region("Shape", world_count=4, target_per_screen=6.0)
    assert set(spec) == {"name", "topology", "worlds", "links"}, set(spec)
    assert spec["topology"] == "region"
    # level_band is deliberately absent on surface worlds - see _bands.
    required = {"key", "name", "grid", "seed", "width", "height", "chunk_size",
                "biomes", "biome_cell", "allowed_creature_types", "density",
                "is_entry", "allows_fast_travel"}
    for w in spec["worlds"]:
        missing = required - set(w)
        assert not missing, f"{w['key']} missing {missing}"
        assert w["density"] in wg.DENSITY_ORDER, w["density"]
        assert all(b in wg.BIOMES for b in w["biomes"]), w["biomes"]
    entry = [w for w in spec["worlds"] if w["is_entry"]][0]
    assert "village" in entry and "entry_spawn" in entry
    for l in spec["links"]:
        assert set(l) == {"from", "edge", "to"} and l["edge"] in wg.EDGES
    return "worlds[] and links[] match vale-region.map.json"


@case("generation is deterministic")
def _determinism():
    a = wg.plan_region("Same", world_count=5, target_per_screen=6.0)
    b = wg.plan_region("Same", world_count=5, target_per_screen=6.0)
    assert a == b, "two identical requests produced different regions"
    c = wg.plan_region("Other", world_count=5, target_per_screen=6.0)
    assert [w["seed"] for w in a["worlds"]] != [w["seed"] for w in c["worlds"]]
    return "same name -> same seeds; different name -> different seeds"


@case("preview renders and scales with the region")
def _preview():
    spec = wg.plan_region("Picture", world_count=6, target_per_screen=6.0)
    img = wp.render(spec)
    assert img.width > 200 and img.height > 200, img.size
    big = wp.render(wg.plan_region("Picture", world_count=12,
                                   target_per_screen=6.0))
    assert big.height >= img.height, (img.size, big.size)
    return f"6 worlds -> {img.size}, 12 worlds -> {big.size}"


@case("a big swarm world is clamped and says so")
def _clamp():
    r = wg.resolve_density("swarm", 224, 224)
    assert r["scatter"] <= wg.MAX_WORLD_CREATURES
    crowded = wg.report({"worlds": [{
        "key": "w", "name": "w", "biomes": ["Mire"], "density": "swarm",
        "width": 224, "height": 224, "is_entry": True,
        "allowed_creature_types": ["Slime"]}], "links": []})
    assert any("socket cost" in p for p in crowded["problems"])
    return f"swarm 224x224 -> {r['scatter']} scatter, flagged crowded"


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
