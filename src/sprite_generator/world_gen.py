"""Generate something2 map specs, and prove they are not empty.

WHY THIS IS NOT A MAP JOB

A `*.map.json` spec is arithmetic, not art: worlds, their biomes, their density
tier, and the links between them. No diffusion, no GPU, no queue - which means
it is the ONE thing this service can hand something2 **synchronously**, inside
the blocking POST their provider system is limited to. Tiles and sheets cannot;
this can.

THE PROBLEM THIS EXISTS TO FIX, AND THE HALF OF IT THAT WAS NEVER REAL

something2's own numbers, read out of their repo on 2026-08-26:

  * `DENSITY_TIERS` is per 1000 tiles, and one screen is ~225 tiles, so
    creatures-per-screen is `perThousand * 0.225`.
  * Their `world_density` migration says 'normal' on a 64x64 map is "~12
    scattered creatures", and their own comment notes every checked-in spec
    uses only sparse/normal/dense - the top tiers are "theoretical".

**The founding claim of this module was wrong.** It said the starter world was
thin TWICE over - the default tier, times a Meadow biome that halves it - and
that solving for a target "after the biome multiplier" was the fix. Corrected
2026-08-27, on structural evidence from something2:

  `resolveDensity(tier, width, height)` takes three arguments. No biome
  parameter, so a world's total cannot depend on `creature_density`.

  `creatureDensityField` is purely redistributive and normalises to mean 1.0
  over interior tiles, so in a single-biome world the multiplier CANCELS. It
  is a relative weight between biomes inside one world, and nothing else.

So the Meadow trap did not exist: `normal` in Meadow is 4.1 creatures per
screen, exactly like `normal` anywhere. Worse, compensating for it made the
entry world of every region over-populated - a 128x128 Meadow world came out
`horde`, 1016 creatures, 14 per screen against a requested 6.

What survives is the smaller, true version: the tiers are coarse (2.0 / 4.1 /
8.1 / 14.0 / 20.0 per screen), so asking for a felt number and being handed the
nearest tier still beats naming one and hoping. It just does not vary by biome,
because nothing does. The report says what each world will actually feel like
before it is seeded, which is the part that was always the point.

The wrong version is kept above rather than deleted. It was believed for two
days by two systems, and it is exactly the shape of error worth recognising
again: internally consistent, arithmetically checkable, describing a mechanism
nobody had read.

Numbers here mirror something2's; they are not invented. Where a number was not
visible in their source it is left unmodelled and said so rather than guessed -
see PACKS_NOT_MODELLED.
"""

from __future__ import annotations

import hashlib
import json
import os

# --- something2's constants, mirrored -------------------------------------

# services/densityTiers.js. Rates are per 1000 tiles and are the world's MEAN;
# their creatureDensityField redistributes around it, so a world holds quiet
# stretches and thick pockets while averaging this.
DENSITY_TIERS = {
    "dead": 0.0,
    "sparse": 9.0,
    "normal": 18.0,
    "dense": 36.0,
    "horde": 62.0,
    "swarm": 89.0,
}

# Their CHECK constraint order (migration 1714440070000), weakest first.
DENSITY_ORDER = ["dead", "sparse", "normal", "dense", "horde", "swarm"]

# Their canvas is a fixed 1280x720 with no zoom and a tile projects to a
# 128x64 iso diamond, so one screen is ~225 tiles. This is the number their
# tier table was tuned against, and the only one a human can feel.
TILES_PER_SCREEN = 225

# services/densityTiers.js. Scatter + packs may never exceed this.
MAX_WORLD_CREATURES = 5000

# Their tiers also carry packCount/packSizeMin/packSizeMax. Only the largest
# tier's shape was visible in the source read ("6 packs of at most 12"), so the
# rest are NOT modelled here rather than invented. Scatter is the dominant term
# and the one that decides whether a world reads empty; packs add to it, so
# every estimate below is a floor, never an overstatement.
PACKS_NOT_MODELLED = True

# The live `biomes` table, all 32 rows, read from something2's DB 2026-08-27
# via something2-54. Colour, creature_density, path_tile and flora_types are
# theirs; `depth` is the only column this module adds - see SURFACE below.
#
# `creature_density` spans 0.5 to 2.5 and the SURFACE biomes are the sparse
# end. Moving a region outward into P4 territory raises variety and density
# together, which is why the descent in `plan_region` fixes both at once.
def _b(color, density, path, flora, creatures, depth):
    return {"color": color, "creature_density": density, "path_tile": path,
            "flora": flora, "creatures": creatures, "depth": depth}


BIOMES = {
    # --- surface: the five starter biomes ---------------------------------
    #
    # WIDENED 2026-08-27, applied on the something2 side and mirrored here.
    # These five shipped with legacy creatures only, which capped a
    # surface-only region at FOUR kinds no matter how many biomes it used -
    # the ceiling this generator kept hitting. The P3 biomes were caught
    # because they shipped EMPTY; these were populated with legacy names, so
    # nothing flagged them.
    #
    # The pairings are not a choice made here or there: `scripts/bestiary/
    # template.js` has a LINES table where every P4 line already declares its
    # home biome, and these five biomes are exactly the five it names.
    #
    # Copied from the applied lists rather than retyped from memory - the
    # Frozen Waste row in ADR 0008 was wrong for exactly that reason once
    # already. The check that it is right is arithmetic: these five together
    # must field 19 kinds, and `smoke-world-gen` asserts it.
    "Meadow": _b("#5aa84f", 0.5, "road_dirt",
                 ["bush", "rose_bush", "Tree", "Stone"],
                 ["Slime", "Wolf", "Beast Swarm", "Beast Skirmisher",
                  "Beast Line"], "surface"),
    "Deep Forest": _b("#2f6b3a", 1.0, "road_dirt",
                      ["Tree", "pine_tree", "dead_tree", "bush", "Stone"],
                      ["Wolf", "Bat", "Skeleton", "Woodland Swarm",
                       "Woodland Skirmisher", "Woodland Line"], "surface"),
    "Arid Dunes": _b("#c9a227", 0.8, "road_sand", ["dead_tree", "Stone"],
                     ["Skeleton", "Bat", "Desert Swarm", "Desert Skirmisher",
                      "Desert Line"], "surface"),
    "Frozen Waste": _b("#8fb8d6", 1.1, "road_snow", ["IceRock", "pine_tree"],
                       ["Bat", "Skeleton", "Tundra Swarm", "Tundra Skirmisher",
                        "Tundra Line"], "surface"),
    "Mire": _b("#4d6b41", 2.0, "road_dirt", ["dead_tree", "bush", "Stone"],
               ["Slime", "Bat", "Swamp Swarm", "Swamp Skirmisher",
                "Swamp Line"], "surface"),
    # Highlands and Storm Coast are P4-creature biomes but read as outdoors,
    # so they are the bridge between the surface ring and the deep ones.
    "Highlands": _b("#7d8471", 0.9, "road_stone", ["Stone", "pine_tree"],
                    ["Highland Swarm", "Highland Skirmisher", "Highland Line"], "surface"),
    "Storm Coast": _b("#6b7280", 0.6, "road_sand", ["Stone", "dead_tree"],
                      ["Storm Swarm", "Storm Skirmisher", "Storm Line"], "surface"),
    "Verdant Jungle": _b("#1f6b2e", 1.6, "road_dirt", ["Tree", "bush", "rose_bush"],
                         ["Jungle Swarm", "Jungle Skirmisher", "Jungle Line"], "surface"),
    # --- deep: dungeon-flavoured, P4 creatures, denser --------------------
    "Abyssal Rift": _b("#1c1a24", 2.4, "road_stone", [],
                       ["Void Swarm", "Void Skirmisher", "Void Line"], "deep"),
    # Surface on authored evidence, not on flavour: vale-region pairs it with
    # Arid Dunes on a surface world (vale_dunes).
    "Ashfields": _b("#4a4038", 1.7, "road_ash", ["dead_tree", "Stone"],
                    ["Volcanic Swarm", "Volcanic Skirmisher", "Volcanic Line"],
                    "surface"),
    "Blightworks": _b("#5e6b3a", 2.0, "road_ash", ["dead_tree", "Stone"],
                      ["Blight Swarm", "Blight Skirmisher", "Blight Line"], "deep"),
    "Catacombs": _b("#55504a", 2.3, "road_stone", ["Stone"],
                    ["Undead Swarm", "Undead Skirmisher", "Undead Line"], "deep"),
    "Cavern": _b("#5a5148", 1.9, "road_dirt", ["Stone"],
                 ["Cave Swarm", "Cave Skirmisher", "Cave Line"], "deep"),
    "Crystal Hollows": _b("#6fa8c9", 1.6, "road_snow", ["IceRock", "Stone"],
                          ["Crystal Swarm", "Crystal Skirmisher", "Crystal Line"], "deep"),
    "Deepvault": _b("#4f5560", 1.8, "road_stone", ["Stone"],
                    ["Construct Swarm", "Construct Skirmisher", "Construct Line"], "deep"),
    "Dreaming Dark": _b("#4a3f6b", 2.3, "road_stone", [],
                        ["Nightmare Swarm", "Nightmare Skirmisher", "Nightmare Line"], "deep"),
    "Emberdepths": _b("#7a3b22", 2.2, "road_ash", ["Stone"],
                      ["Ember Swarm", "Ember Skirmisher", "Ember Line"], "deep"),
    "Fallen Sanctum": _b("#b8a97a", 2.2, "road_stone", [],
                         ["Fallen Swarm", "Fallen Skirmisher", "Fallen Line"], "deep"),
    "Frostvault": _b("#a8c6d6", 1.8, "road_snow", ["IceRock"],
                     ["Rime Swarm", "Rime Skirmisher", "Rime Line"], "deep"),
    "Fungal Deep": _b("#6b7f3a", 2.1, "road_dirt", ["bush", "Stone"],
                      ["Fungal Swarm", "Fungal Skirmisher", "Fungal Line"], "deep"),
    "Gloomfen": _b("#4a5a4a", 1.9, "road_dirt", ["dead_tree", "bush"],
                   ["Gloom Swarm", "Gloom Skirmisher", "Gloom Line"], "deep"),
    "Grave of Titans": _b("#7a7266", 2.1, "road_stone", ["Stone"],
                          ["Titan Swarm", "Titan Skirmisher", "Titan Line"], "deep"),
    "Hive Warrens": _b("#8a6a2f", 2.2, "road_dirt", ["Stone"],
                       ["Hive Swarm", "Hive Skirmisher", "Hive Line"], "deep"),
    "Infernal Gate": _b("#8c3a1e", 2.5, "road_ash", [],
                        ["Demonic Swarm", "Demonic Skirmisher", "Demonic Line"], "deep"),
    "Ossuary": _b("#c9c2ad", 2.4, "road_stone", ["Stone"],
                  ["Bonelord Swarm", "Bonelord Skirmisher", "Bonelord Line"], "deep"),
    "Pestilent Deep": _b("#6b6b33", 2.4, "road_ash", [],
                         ["Plague Swarm", "Plague Skirmisher", "Plague Line"], "deep"),
    "Shattered Vault": _b("#6b2f6b", 2.3, "road_stone", [],
                          ["Chaos Swarm", "Chaos Skirmisher", "Chaos Line"], "deep"),
    "Sunken Cistern": _b("#3f5a63", 1.8, "road_stone", ["Stone"],
                         ["Drowned Swarm", "Drowned Skirmisher", "Drowned Line"], "deep"),
    "Sunken Foundry": _b("#6a5a48", 1.8, "road_stone", ["Stone"],
                         ["Stoneborn Swarm", "Stoneborn Skirmisher", "Stoneborn Line"], "deep"),
    # Likewise: vale_frozen pairs it with Frozen Waste on the surface ring.
    "Sunken Ruins": _b("#8a8577", 1.4, "road_stone", ["Stone", "dead_tree", "bush"],
                       ["Ruin Swarm", "Ruin Skirmisher", "Ruin Line"], "surface"),
    "The Maw": _b("#3d1f22", 2.5, "road_ash", [],
                  ["Eldritch Swarm", "Eldritch Skirmisher", "Eldritch Line"], "deep"),
    "Umbral Warren": _b("#2e2a35", 2.2, "road_stone", ["Stone"],
                        ["Umbral Swarm", "Umbral Skirmisher", "Umbral Line"], "deep"),
}

SURFACE = [n for n, b in BIOMES.items() if b["depth"] == "surface"]
DEEP = [n for n, b in BIOMES.items() if b["depth"] == "deep"]

# Seven biomes carry NO flora, and that is authored intent - they are the
# deepest, most desolate places. A flora floor applied uniformly would fire on
# them forever and the repair step would chase something unrepairable.
FLORA_OPTIONAL = frozenset(n for n, b in BIOMES.items() if not b["flora"])

# Their SOMET-311 measurement, reduced to the one invariant that survived a
# rate re-scale: ~920 B/s per creature in the active neighbourhood, per socket.
# Their own rows were measured at old rates and are kept only for the SHAPE of
# the cost; this number is the part that still holds.
BYTES_PER_CREATURE_PER_SEC = 920

# The authority activates and broadcasts a radius-1 chunk neighbourhood: 9
# chunks. That is the set a parked player pays for.
NEIGHBOURHOOD_CHUNKS = 9

# Calibrated against what something2 actually cautioned about, not against a
# round number. They said targeting ~6 per screen "is fine" and that
# horde/swarm on 192+ maps "is a bandwidth decision, not a flavour one". So the
# threshold has to sit ABOVE dense and below horde, or it fires on the
# configuration they blessed and stops being read:
#
#   dense  8.1/screen -> ~300 KiB/s     fine, no warning
#   horde   14/screen -> ~515 KiB/s     warn
#   swarm   20/screen -> ~736 KiB/s     warn
#
# A first pass set this to 250 and warned on a 6.8/screen world, which is
# exactly the crying-wolf failure that gets a warning ignored.
SOCKET_WARN_KIB_S = 400

# --- what a biome can actually spawn --------------------------------------
#
# READ THIS BEFORE TOUCHING CREATURE SELECTION. It is the correction of a real
# and expensive mistake.
#
# `entity_types` holds 293 creature rows, including the whole P4 bestiary
# (`Beast Swarm`, `Drowned Caster`, ...). Those names are REAL - which is why
# something2's "unknown creature type" check never fired on them. But existing
# is not the same as being spawnable: spawning is gated by a SECOND catalog,
# `biomes.creature_types`, and a world seeds nothing unless its
# `allowed_creature_types` INTERSECTS the union of its biomes' lists
# (their SOMET-315).
#
# The first version of this module selected from the bestiary by family and
# level band. Every name it produced was real, none of them were spawnable, and
# something2's validator reported all six worlds of a "0 empty worlds, 3376
# creatures" region as seeding ZERO. The report was green about a number the
# game could never produce.
#
# The evidence that misled: their own `pens` entries use P4 names, so a P4 name
# in a spec looked correct. Pens are exactly where those names ARE valid - they
# are authored placements, not biome spawning.
#
# So `allowed_creature_types` is now DERIVED from the chosen biomes. The
# intersection is non-empty by construction and this class of error cannot come
# back.
#
# Read from something2's live `biomes` table 2026-08-27, via something2-54.

def spawnable(biomes) -> list:
    """Creatures these biomes can actually spawn. The union of their lists.

    This IS `allowed_creature_types`. Deriving it rather than choosing it is
    what makes the biome/creature intersection non-empty by construction.
    """
    seen = []
    for b in biomes:
        for c in BIOMES.get(b, {}).get("creatures", []):
            if c not in seen:
                seen.append(c)
    return seen


# Kept as a name because the correction in ADR 0008 D2 refers to it, and
# because a reader looking for "the spawn table" should land somewhere.
BIOME_CREATURES = {n: b["creatures"] for n, b in BIOMES.items()}


# --- the P4 bestiary ------------------------------------------------------
#
# 288 creatures, 32 families x 9 roles, all present in their seeders. Every one
# has an EMPTY `spawn_tiles`, so which creatures a world can hold is decided
# entirely by `allowed_creature_types` and the biome's `creature_types` - there
# is no terrain gate to fall back on.
#
# This matters more than the density tier for how a world FEELS. Density fixes
# how many; this fixes how many KINDS, and a world can be at `swarm` and still
# read as monotonous.
#
# Until 2026-08-27 the five starter biomes referenced only the four legacy
# creatures (Slime, Wolf, Bat, Skeleton), so a surface-only region saw the same
# four things forever whatever it did. The widening gave each of the five its
# own P4 line family - see BIOMES - and took that number from 4 to 19.
_BESTIARY_PATH = os.path.join(os.path.dirname(__file__), "bestiary_p4.json")
with open(_BESTIARY_PATH, "r", encoding="utf-8") as _fh:
    BESTIARY = json.load(_fh)

FAMILIES = BESTIARY["families"]
ROLES = BESTIARY["roles"]

# Champion carries aura_radius 260 and is the only aura-carrying entry in their
# catalog; Apex is the other end of the same ramp. Their measurement is blunt:
# computeAuras is O(leaders x all) over the WHOLE population every tick, so
# 4500 creatures at 50 leaders already spends most of the 8 ms half-budget and
# 200 leaders blows past the whole 16 ms frame. Leaders are therefore rationed
# per world rather than included because they are available.
LEADER_ROLES = set(BESTIARY["leader_roles"])
MAX_LEADER_ROLES_PER_WORLD = 1

# Which bestiary families suit which biome. Thematic, and the only editorial
# judgement in this module - the level ramp and the role choice below are
# mechanical once the family is picked.
BIOME_FAMILIES = {
    "Meadow": ["Beast", "Woodland"],
    "Deep Forest": ["Woodland", "Fungal", "Gloom"],
    "Arid Dunes": ["Desert", "Bonelord", "Ruin"],
    "Frozen Waste": ["Tundra", "Rime"],
    "Mire": ["Swamp", "Drowned", "Blight"],
}

# Fewer distinct creature types than this and a world reads repetitive however
# many are on screen. Four is the number the starter biomes ship with, and the
# complaint that motivated this.
THIN_VARIETY_BELOW = 5


def families_for(biomes) -> list:
    out = []
    for b in biomes:
        for f in BIOME_FAMILIES.get(b, []):
            if f in FAMILIES and f not in out:
                out.append(f)
    return out


def bestiary_types(biomes, level_band, limit: int = 12) -> list:
    """Creature names for a world, matched to its families and level band.

    Roles are taken in catalog order, which is also the difficulty ramp, and
    only where the role's own level range overlaps the world's band - so an
    entry world gets Swarm and Skirmisher, not an Apex it cannot survive.

    Leaders are capped by `MAX_LEADER_ROLES_PER_WORLD` for the runtime reason
    recorded above, not for game-design reasons.
    """
    lo, hi = (level_band or [1, 3])[:2]
    fams = families_for(biomes)
    if not fams:
        return []

    picked, leaders = [], 0
    for role in ROLES:
        for fam in fams:
            band = FAMILIES.get(fam, {}).get(role)
            if not band:
                continue
            # Overlap, not containment: a role spanning 8-12 belongs in a world
            # banded 10-14 even though neither contains the other.
            if band[1] < lo or band[0] > hi:
                continue
            if role in LEADER_ROLES:
                if leaders >= MAX_LEADER_ROLES_PER_WORLD:
                    continue
                leaders += 1
            name = f"{fam} {role}"
            if name not in picked:
                picked.append(name)
            if len(picked) >= limit:
                return picked
    return picked


def enrich_variety(biomes, level_band) -> list:
    """Add a biome until the world can field enough KINDS of creature.

    A single-biome world reaches only its own families, and within a five-level
    band that is often four creatures - which reads repetitive at any density.
    Adding a biome adds families, and families are what the bestiary is indexed
    by.

    One addition at most, and only if it actually helps: the same bound as
    `enrich_flora`, for the same reason. A world that is still thin afterwards
    is reported rather than padded further.
    """
    if len(spawnable(biomes)) >= THIN_VARIETY_BELOW:
        return biomes

    # Candidates of the SAME depth first. Repairing variety by reaching across
    # the surface/deep line puts a meadow inside the Infernal Gate - which is
    # the incoherence the descent exists to prevent, reintroduced by the repair
    # meant to help. Cross-depth is kept only as a last resort, because a
    # slightly odd pairing still beats a world with three creatures in it.
    here = {BIOMES[b]["depth"] for b in biomes if b in BIOMES}
    same = [n for n in BIOMES if BIOMES[n]["depth"] in here]
    other = [n for n in BIOMES if BIOMES[n]["depth"] not in here]

    best, best_n = biomes, len(spawnable(biomes))
    for name in same + other:
        if name in biomes:
            continue
        candidate = list(biomes) + [name]
        n = len(spawnable(candidate))
        if n > best_n:
            best, best_n = candidate, n
        if best_n >= THIN_VARIETY_BELOW:
            break
    return best


def leader_count(types) -> int:
    return sum(1 for t in types if t.rsplit(" ", 1)[-1] in LEADER_ROLES)


# Below this a world reads as empty space rather than as a quiet stretch. Their
# re-scale comment puts the pre-scale game at 0.7-2.7 per screen and calls that
# the thing being fixed, so the floor sits just above it.
# The band for a surface world, and the reason every world now carries one.
#
# `scripts/seed-map.js` writes a world's level range as:
#
#     w.level_band ? w.level_band[0] : 1,
#     w.level_band ? w.level_band[1] : 1,
#
# AN OMITTED BAND IS NOT "unspecified" OR "derive it". It becomes level 1-1 -
# a valid-looking band that pins the world to level-1 creatures and is
# indistinguishable afterwards from a world that asked for exactly that. The
# same trap as `Number('') === 0`: a default that is itself a legal value, so
# nothing downstream can tell "unset" from "the lowest setting".
#
# This generator used to omit the band on surface worlds, copying vale-region's
# convention. That convention is correct FOR VALE-REGION, whose surface ring is
# the level-1 starting area. It does not transfer to a region that DESCENDS: it
# left emerald-reach's first three worlds at 1-1 and then jumping to 3-5.
#
# [1, 2] rather than [1, 1]: it keeps the entry a level-1 area, which is what
# omitting it was always meant to convey, and it leads into the deep ramp's
# first band of [3, 5] with neither a gap nor an overlap. The deep formula
# cannot be reused here - its lowest rung is 3, which would make a starting
# meadow a level 3-5 world and that is a design change, not a bug fix.
SURFACE_BAND = [1, 2]

EMPTY_BELOW_PER_SCREEN = 3.0

# Above this a screen is a crowd, and the wire cost bites: their measurement
# has swarm on 224x224 at ~940 KiB/s down a single socket.
CROWDED_ABOVE_PER_SCREEN = 22.0

# Fewer distinct flora types than this and the ground reads bare no matter how
# many creatures are on it - trees and stones are what fill space between them.
THIN_FLORA_BELOW = 3

EDGES = ["N", "E", "S", "W"]
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
DELTA = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}

# something2 authors world positions in PIXELS and world layout in TILES, at a
# fixed 100 px per tile: vale_hub's village sits at min_row 44 with spawn_y
# 4550, i.e. row 45 -> 45*100+50. Every x/y this module emits goes through
# `to_px` so the two never drift.
TILE_PX = 100

# Tiles either side of a road that stay safe. Their `world_safe_region`
# migration and vale-region both use 2.
SAFE_ROAD_RADIUS = 2


def to_px(tile: int) -> int:
    """Tile index -> the pixel centre of that tile."""
    return tile * TILE_PX + TILE_PX // 2


def is_tile_centre(px) -> bool:
    """Is this a pixel coordinate, or a tile index someone forgot to convert?

    A range check cannot answer that. Tile 64 of a 128-wide world is a perfectly
    legal PIXEL coordinate too - 64 px is inside the world's 12800 - so the
    mistake survives every bounds test and only shows up as an entity huddled in
    the top-left corner after the map is seeded.

    What separates them exactly is the offset: every position this module emits
    is the CENTRE of a tile, so a real pixel coordinate is always
    `n * 100 + 50`. A tile index almost never is, and 64 certainly is not.
    """
    return isinstance(px, int) and not isinstance(px, bool) \
        and px % TILE_PX == TILE_PX // 2


# Fields something2 reads as PIXELS. Everything else in a world - roads,
# village bounds, width/height, biome_cell - is TILES. Keeping the list here
# rather than in the checker means adding a positioned field to a spec makes
# the checker cover it, instead of quietly not covering it.
PIXEL_FIELDS = (
    ("entry_spawn", ("x", "y")),
    ("village", ("spawn_x", "spawn_y")),
    ("chest", ("x", "y")),
)
PIXEL_LIST_FIELDS = (
    ("waypoints", ("x", "y")),
)


def check_units(world: dict) -> list:
    """Every pixel field in a world, checked for being pixels.

    Run on every generated spec rather than left as a rule to remember, because
    this is the failure that costs a seed-and-look to notice.
    """
    problems = []
    size = int(world.get("width", 0) or 0)
    limit = size * TILE_PX

    def check(where, value, key):
        if value is None:
            return
        if not is_tile_centre(value):
            problems.append(
                f"{world.get('key', '?')}.{where}.{key} = {value} is not a tile "
                f"centre - looks like TILE coordinates where something2 reads "
                f"PIXELS ({TILE_PX}px per tile, so tile n is n*{TILE_PX}+"
                f"{TILE_PX // 2})")
        elif limit and not 0 <= value < limit:
            problems.append(
                f"{world.get('key', '?')}.{where}.{key} = {value} is outside "
                f"the world's {limit}px extent")

    for field, keys in PIXEL_FIELDS:
        obj = world.get(field)
        if isinstance(obj, dict):
            for k in keys:
                check(field, obj.get(k), k)

    for field, keys in PIXEL_LIST_FIELDS:
        for i, obj in enumerate(world.get(field) or []):
            if isinstance(obj, dict):
                for k in keys:
                    check(f"{field}[{i}]", obj.get(k), k)

    # And the mirror: roads are TILES, so a pixel value there is the same
    # mistake in the other direction.
    for i, line in enumerate(world.get("roads") or []):
        for point in line:
            for v in point:
                if size and not 0 <= v < size:
                    problems.append(
                        f"{world.get('key', '?')}.roads[{i}] point {point} is "
                        f"outside the {size}-tile grid - roads are TILES, not "
                        f"pixels")
    return problems


# --- density, solved rather than named ------------------------------------

def biome_multiplier(biomes) -> float:
    """Mean `creature_density` across a world's biomes.

    IT DOES NOT SCALE ANYTHING. Reported because it describes how creatures are
    distributed WITHIN a world, and for nothing else - see `per_screen` below
    for the two days that cost.
    """
    known = [BIOMES[b]["creature_density"] for b in biomes if b in BIOMES]
    return sum(known) / len(known) if known else 1.0


def per_screen(tier: str, biomes=None, width: int = 0, height: int = 0) -> float:
    """Creatures on one screen. The felt number.

    THE BIOME MULTIPLIER IS NOT IN HERE, and removing it on 2026-08-27 was the
    correction this module most needed.

    Two structural facts from something2, neither of which was checked when
    this was written:

      `resolveDensity(tier, width, height)` takes THREE arguments. There is no
      biome parameter, so a world's total cannot depend on the multiplier - it
      is not in scope where the total is computed.

      `creatureDensityField` is purely redistributive and normalises to mean
      1.0 over interior tiles, so in a SINGLE-BIOME world a uniform weight
      divides out completely. Meadow at 0.5 alone behaves exactly like a biome
      at 2.5 alone. It is a relative weight BETWEEN biomes inside one world,
      clamped to [0.15, 1.5], and nothing more.

    So the average over a world is just its population over its area, whatever
    its biomes. Derived from the RESOLVED count rather than the tier, which
    also makes a world clamped at MAX_WORLD_CREATURES report the density it
    will actually have instead of the one its tier asked for.

    What this cost: `choose_density` was compensating for a multiplier that
    does not apply, so the entry world of every region was over-populated - a
    128x128 Meadow world came out at `horde`, 1016 creatures, 14/screen against
    a requested 6. And the "Meadow trap" this module was written to fix, a
    starter world at ~2 creatures per screen, WAS NEVER REAL. `normal` in
    Meadow is 4.1 per screen, like `normal` anywhere else.
    """
    area = int(width) * int(height)
    if area <= 0:
        # No world to measure - fall back to the tier's own rate, which is what
        # a caller asking without dimensions can possibly mean.
        return DENSITY_TIERS.get(tier, 0.0) * (TILES_PER_SCREEN / 1000.0)
    return resolve_density(tier, width, height)["scatter"] * TILES_PER_SCREEN / area


def resolve_density(tier: str, width: int, height: int) -> dict:
    """Mirror of their resolveDensity, scatter only. See PACKS_NOT_MODELLED."""
    area = int(width) * int(height)
    target = round(DENSITY_TIERS.get(tier, 0.0) * area / 1000.0)
    scatter = max(0, min(target, MAX_WORLD_CREATURES))
    return {"scatter": scatter, "clamped": target > MAX_WORLD_CREATURES,
            "area": area}


def choose_density(target_per_screen: float, biomes=None) -> str:
    """The tier landing closest to `target_per_screen`.

    `biomes` is accepted and IGNORED. It used to multiply the target by the
    biome density, which was the module's headline feature - "a thin biome is
    compensated with a heavier tier automatically" - and it was compensating
    for something that does not happen. See `per_screen`.

    The correction makes this function much less clever and much more correct.
    It still earns its place: the tiers are coarse (9, 18, 36, 62, 89 per
    thousand, i.e. 2.0 / 4.1 / 8.1 / 14.0 / 20.0 per screen), so asking for a
    felt number and being given the nearest tier still beats naming one and
    hoping. It simply no longer varies by biome, because nothing does.

    The parameter is kept rather than removed so a stored spec or an older
    caller does not break on a signature change; it is documented as ignored
    rather than quietly accepted.
    """
    best, best_err = "normal", float("inf")
    for name in DENSITY_ORDER:
        if name == "dead":
            continue
        err = abs(DENSITY_TIERS[name] * (TILES_PER_SCREEN / 1000.0)
                  - target_per_screen)
        if err < best_err:
            best, best_err = name, err
    return best


def _depth_of(biomes) -> str | None:
    """`surface`, `deep`, or `mixed` for a transition world."""
    depths = {BIOMES[b]["depth"] for b in biomes if b in BIOMES}
    if not depths:
        return None
    return depths.pop() if len(depths) == 1 else "mixed"


def socket_kib_s(per_screen_count: float, chunk_size: int) -> float:
    """JSON per second down ONE socket for a player parked in the thick of it.

    From something2's SOMET-311 measurement, reduced to its surviving
    invariant: ~920 B/s per creature in the active radius-1 chunk
    neighbourhood. Their absolute rows predate a rate re-scale and are not
    reused; the per-creature figure is.

    This is a BANDWIDTH cost, not a tick-budget one - their tick loop held
    ~20 Hz in every world measured. It is reported because a `horde` region on
    a 224 map is a network decision, and nothing should be able to reach one by
    accident through a theme string.
    """
    tiles = NEIGHBOURHOOD_CHUNKS * chunk_size * chunk_size
    creatures = per_screen_count * tiles / TILES_PER_SCREEN
    return creatures * BYTES_PER_CREATURE_PER_SEC / 1024.0


def flora_types(biomes) -> list:
    """Distinct decoration types a world can spawn - its trees and stones.

    something2 has no per-world decoration density: `entity_types.chance` is
    catalog-global (services/decorationDefs.js). Biome CHOICE is therefore the
    only lever a map spec has over how furnished the ground looks, which is why
    this is reported rather than tuned.
    """
    seen = []
    for b in biomes:
        for f in BIOMES.get(b, {}).get("flora", []):
            if f not in seen:
                seen.append(f)
    return seen


def enrich_flora(biomes) -> list:
    """Add a biome until the ground has enough plant life to not read bare.

    The repair half of "the model authors, the code guarantees". A 3B asked to
    avoid pairing two 2-flora biomes mostly complies, and mostly is not a
    guarantee - so any world still short of `THIN_FLORA_BELOW` gains the
    richest biome it is not already using.

    Kept to at most one addition: a third biome band starts to muddy what the
    world IS, and a world that needs two additions was a bad pairing that the
    report should surface rather than something to paper over.
    """
    # Seven biomes carry no flora by design. A world that is deliberately
    # desolate is not a world to repair, and adding a bush-bearing biome to The
    # Maw to satisfy a floor would be the floor overruling the author.
    if any(b in FLORA_OPTIONAL for b in biomes):
        return biomes
    if len(flora_types(biomes)) >= THIN_FLORA_BELOW:
        return biomes

    # Same-depth first, for the reason `enrich_variety` documents: a repair
    # that crosses the surface/deep line undoes the descent. A deep pair can
    # reach the floor on its own - Crystal Hollows and Gloomfen make four
    # between them - so this is rarely a real constraint.
    here = {BIOMES[b]["depth"] for b in biomes if b in BIOMES}
    order = sorted(BIOMES, key=lambda b: (BIOMES[b]["depth"] not in here,
                                          -len(BIOMES[b]["flora"])))
    for name in order:
        if name in biomes:
            continue
        candidate = list(biomes) + [name]
        if len(flora_types(candidate)) >= THIN_FLORA_BELOW:
            return candidate
    return biomes


def creature_types(biomes) -> list:
    seen = []
    for b in biomes:
        for c in BIOMES.get(b, {}).get("creatures", []):
            if c not in seen:
                seen.append(c)
    return seen


# --- the region graph -----------------------------------------------------

def _seed_for(region: str, key: str) -> int:
    h = hashlib.sha256(f"{region}/{key}".encode()).hexdigest()
    return 1000 + int(h[:6], 16) % 60000


def _spiral(n: int):
    """Grid positions outward from the origin, so a region grows compactly.

    A compact region matters: links are only made between orthogonal grid
    neighbours, and a straggling layout produces worlds with one exit and long
    dead-end corridors.
    """
    out, x, y, dx, dy = [(0, 0)], 0, 0, 1, 0
    steps, run = 1, 0
    while len(out) < n:
        for _ in range(steps):
            x, y = x + dx, y + dy
            out.append((x, y))
            if len(out) >= n:
                return out
        dx, dy = -dy, dx
        run += 1
        if run % 2 == 0:
            steps += 1
    return out[:n]


def plan_region(name: str, world_count: int = 6,
                target_per_screen: float = 6.0,
                size: int = 128, chunk_size: int = 32, biome_cell: int = 32,
                biome_plan=None, entry_biomes=("Meadow",),
                bestiary: bool = True) -> dict:
    """A complete, seedable something2 map spec.

    `biome_plan` is the semantic half - which biome each world is, in graph
    order - and is exactly what an LLM should author: a handful of decisions
    with meaning, not thousands of coordinates. Everything numeric below is
    solved deterministically afterwards, because a model cannot be trusted to
    hit a creatures-per-screen target and does not need to be.
    """
    if world_count < 1:
        raise ValueError("a region needs at least one world")

    names = list(BIOMES)
    cells = _spiral(world_count)
    worlds, by_cell = [], {}
    # How many deep worlds have been placed so far. The band ramps on THIS,
    # not on the loop index, so the ramp is monotonic through the descent even
    # when surface worlds are interleaved.
    depth_rank = 0

    for i, (gx, gy) in enumerate(cells):
        if biome_plan and i < len(biome_plan):
            biomes = [b for b in biome_plan[i] if b in BIOMES] or [names[i % len(names)]]
        elif i == 0:
            biomes = [b for b in entry_biomes if b in BIOMES] or ["Meadow"]
        else:
            # A DESCENT, not a shuffle. Surface biomes near the entry, deep
            # ones further out, matching the level band that already ramps with
            # distance.
            #
            # This is what answers something2's coherence question and what
            # lifts the four-creature ceiling, in one move. The five surface
            # biomes admit only Slime/Wolf/Bat/Skeleton between them; the deep
            # ones carry the P4 families. Reaching for variety by scattering
            # deep biomes among surface ones would validate and read as
            # nonsense - Catacombs next to a meadow - whereas descending into
            # them is the ordinary shape of a region.
            # Progression is the INDEX, not the grid distance.
            #
            # The layout is a spiral, so distance from the origin and position
            # in the journey are unrelated - world 7 of 8 can sit one cell from
            # the entry. Driving depth from distance made the eighth world as
            # shallow as the second, and because the LLM authors biomes in
            # INDEX order, biomes descended while bands ramped by distance and
            # the two disagreed. something2 caught it: bands oscillating
            # 3-7 / 5-9 with the deepest world back at [3,7].
            pool = SURFACE if i <= max(1, world_count // 3) else DEEP
            biomes = [pool[i % len(pool)], pool[(i + 2) % len(pool)]]
            biomes = list(dict.fromkeys(biomes))

        # Both repairs apply to every author, not just the LLM: the
        # deterministic pairing can land two thin biomes together too.
        biomes = enrich_flora(biomes)
        if bestiary:
            biomes = enrich_variety(biomes, None)

        # EVERY world gets a band, including surface ones. Omitting it was a
        # bug - see SURFACE_BAND.
        #
        # The deep ramp is computed AFTER the biomes, because where a world
        # sits in the descent depends on what it is made of, and it ramps on
        # depth_rank rather than the loop index so the ramp stays monotonic
        # when surface worlds are interleaved.
        if all(BIOMES.get(b, {}).get("depth") == "surface" for b in biomes):
            band = list(SURFACE_BAND)
        else:
            lo = 3 + depth_rank * 2
            band = [lo, lo + 2 + depth_rank // 2]
            depth_rank += 1

        key = f"{_slug(name)}_{i}" if i else f"{_slug(name)}_hub"
        tier = choose_density(target_per_screen, biomes)

        w = {
            "key": key,
            "name": _title(name, i),
            "allows_fast_travel": i == 0,
            "grid": [gx, gy],
            "seed": _seed_for(name, key),
            "width": size,
            "height": size,
            "chunk_size": chunk_size,
            "biomes": biomes,
            "biome_cell": biome_cell,
            # The P4 bestiary, not the four legacy names the starter biomes
            # carry. `bestiary=False` keeps the old behaviour for a caller that
            # has not seeded P4.
            # Derived, never chosen. See BIOME_CREATURES.
            "allowed_creature_types": spawnable(biomes),
            "density": tier,
            "is_entry": i == 0,
        }
        # ALWAYS written, and never as null. An absent key and a null are the
        # same thing to the seeder's ternary, and both mean level 1-1.
        w["level_band"] = band
        if i == 0:
            centre = to_px(size // 2)
            w["entry_spawn"] = {"x": centre, "y": centre}
            # Centred ON the spawn tile, not offset from it. Every village in
            # vale-region contains its own spawn point; an offset that put the
            # spawn outside the walls would drop a new player next to the
            # village rather than in it. Width 6 x height 4 around (mid, mid)
            # means cols mid-3..mid+2 and rows mid-2..mid+1, both of which
            # include mid.
            #
            # gate_edge is filled in later, once the links are known - see
            # _add_roads. Their own specs use E, S, S, S, S, so it is not a
            # constant.
            mid = size // 2
            w["village"] = {
                # Hyphens, not underscores: world keys are `vale_hub` but
                # village keys are `vale-crossing` in their own spec.
                "key": f"{_slug(name).replace('_', '-')}-crossing",
                "min_row": mid - 2, "min_col": mid - 3,
                "width": 6, "height": 4, "gate_edge": "E",
                "spawn_x": centre, "spawn_y": centre,
            }
        worlds.append(w)
        by_cell[(gx, gy)] = w

    links = []
    for w in worlds:
        gx, gy = w["grid"]
        for edge in EDGES:
            dx, dy = DELTA[edge]
            nb = by_cell.get((gx + dx, gy + dy))
            if nb:
                links.append({"from": w["key"], "edge": edge, "to": nb["key"]})

    # Roads and waypoints need to know which edges a world actually exits by,
    # so they are a second pass rather than part of the loop above.
    exits = {}
    for l in links:
        exits.setdefault(l["from"], []).append(l["edge"])
    for w in worlds:
        _add_roads(w, exits.get(w["key"], []))

    return {"name": name, "topology": "region", "worlds": worlds, "links": links}


def _add_roads(world: dict, edges) -> None:
    """A road from the middle of a world to every edge it exits by.

    Structure is the other half of "not empty". A world with creatures scattered
    at random still reads as undifferentiated ground; a road gives it a spine,
    tells a player where the exits are without a map, and - through
    `safe_road_radius` - carves a corridor they can move along without being
    swarmed at the density these regions now ask for.

    Roads are `[[row, col], ...]` polylines in TILE coordinates, matching
    vale-region. Waypoints and spawns beside them are in PIXELS, which is why
    both go through `to_px`.
    """
    if not edges:
        return

    size = int(world.get("width", 128))
    mid = size // 2
    last = size - 1

    # 1..size-2, NOT 0..size-1. something2's wall ring occupies the outermost
    # cells, and `stampBounds` overwrites anything authored there - so a road
    # drawn to the edge is a tile that is never rendered while STILL widening
    # the safe corridor, which is the two halves of the feature disagreeing
    # (their mapSpec.js:516). Found by their validator, 14 errors, every world.
    ends = {"N": (1, mid), "S": (last - 1, mid),
            "W": (mid, 1), "E": (mid, last - 1)}
    roads = [[[mid, mid], list(ends[e])] for e in edges if e in ends]
    if not roads:
        return

    world["roads"] = roads
    world["safe_road_radius"] = SAFE_ROAD_RADIUS

    # The village gate must open onto an edge this world actually exits by, or
    # it opens onto the map boundary. Prefer E to match their entry world, but
    # only when E is genuinely an exit.
    village = world.get("village")
    if isinstance(village, dict):
        village["gate_edge"] = "E" if "E" in edges else edges[0]
    # One per world - their `one_waypoint_per_world` migration makes that a
    # constraint, not a preference. It sits where the roads meet, which is the
    # one point in the world every road already reaches.
    world["waypoints"] = [{
        "x": to_px(mid), "y": to_px(mid),
        "name": f"{world.get('name', world['key'])} Waystone",
    }]


def _slug(s: str) -> str:
    out = "".join(c.lower() if c.isalnum() else "_" for c in s).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "region"


def _title(region: str, i: int) -> str:
    return f"{region} Crossing" if i == 0 else f"{region} {i}"


# --- the report: does this world read as empty? ---------------------------

def report(spec: dict, target_per_screen: float | None = None) -> dict:
    """Per-world verdicts, plus every structural check worth failing on.

    Written as a report rather than an exception because a spec with one thin
    world is still seedable, and the useful thing is to say WHICH world and by
    how much - not to refuse the whole region.

    `target_per_screen` is what the caller ASKED for, if known. Passing it is
    what lets the report say "you asked for 6.0 and the tiers can only give you
    4.1" - see the caveat below. It is a parameter rather than a spec key
    because something2's WORLD_KEYS rejects keys it does not know, and a spec
    is not the place to record what someone wanted.
    """
    worlds = spec.get("worlds", [])
    keys = {w["key"] for w in worlds}
    rows, problems = [], []

    for w in worlds:
        biomes = w.get("biomes", [])
        tier = w.get("density", "normal")
        res = resolve_density(tier, w.get("width", 0), w.get("height", 0))
        ps = per_screen(tier, biomes, w.get("width", 0), w.get("height", 0))
        flora = flora_types(biomes)

        # THE CHECK THAT WAS MISSING, and the reason a region could report
        # "0 empty worlds, 3376 creatures" while seeding nothing at all.
        #
        # The tier arithmetic below says how many creatures a world is BUDGETED
        # for. It says nothing about whether any of them can spawn - that needs
        # `allowed_creature_types` to intersect the biomes' own lists. When it
        # does not, the true count is zero however healthy the tier looks, so
        # the verdict must come from the intersection first and the arithmetic
        # second.
        types = w.get("allowed_creature_types", [])
        admitted = [t for t in types if t in spawnable(biomes)]
        if not admitted:
            rows.append({
                "key": w["key"], "name": w.get("name"), "biomes": biomes,
                "density": tier, "per_screen": 0.0, "creatures": 0,
                # Zero under either reading of the multiplier, but the key must
                # be present: a row that omits it makes every consumer branch
                # on whether this world was empty.

                "area": res["area"],
                "biome_multiplier": round(biome_multiplier(biomes), 2),
                "flora": flora, "creature_types": types, "variety": 0,
                "leaders": 0, "verdict": "EMPTY",
            })
            problems.append(
                f"{w['key']}: would seed ZERO creatures. Its biomes "
                f"({', '.join(biomes) or 'none'}) admit "
                f"{', '.join(spawnable(biomes)) or 'nothing'}, but the world "
                f"allows {', '.join(types) or 'nothing'} - the two do not "
                f"intersect. A creature existing in entity_types is not the "
                f"same as a biome being able to spawn it.")
            continue

        verdict = "ok"
        if ps < EMPTY_BELOW_PER_SCREEN:
            verdict = "EMPTY"
            problems.append(
                f"{w['key']}: {ps:.1f} creatures/screen - reads as empty space. "
                f"{tier} over {res['area']} tiles. "
                f"Raise the tier - the biome does not change this.")
        elif ps > CROWDED_ABOVE_PER_SCREEN:
            verdict = "CROWDED"
            problems.append(
                f"{w['key']}: {ps:.1f} creatures/screen and "
                f"{res['scatter']} creatures in the world - past the point "
                f"their own measurement shows the socket cost biting.")

        kib = socket_kib_s(ps, int(w.get("chunk_size", 32)))
        if kib > SOCKET_WARN_KIB_S:
            problems.append(
                f"{w['key']}: ~{kib:.0f} KiB/s down a single socket for one "
                f"parked player ({ps:.1f}/screen). Their measurement puts the "
                f"cost at ~{BYTES_PER_CREATURE_PER_SEC} B/s per creature in "
                f"the active neighbourhood, so this is a bandwidth decision "
                f"rather than a flavour one.")

        types = w.get("allowed_creature_types", [])
        leaders = leader_count(types)
        if len(types) < THIN_VARIETY_BELOW:
            # The honest cause, which is NOT "pick better biomes". The five
            # biomes this generator can select admit only Slime, Wolf, Bat and
            # Skeleton between them - four, total, across every combination. A
            # floor of five is unreachable until the other 27 biomes in
            # something2's table become selectable, and they cannot be until
            # their colour, creature_density and flora_types are known here.
            problems.append(
                f"{w['key']}: only {len(types)} creature type(s) "
                f"({', '.join(types) or 'none'}) - the world will read "
                f"repetitive however many are on screen. The "
                f"{len(BIOMES)} selectable biomes admit "
                f"{len(set(sum((BIOME_CREATURES[b] for b in BIOMES), [])))} "
                f"creatures in total; the other "
                f"{len(BIOME_CREATURES) - len(BIOMES)} biomes in something2's "
                f"table are P4 and would fix this, but their metadata is not "
                f"known here yet.")
        if leaders > MAX_LEADER_ROLES_PER_WORLD:
            problems.append(
                f"{w['key']}: {leaders} leader roles. computeAuras is "
                f"O(leaders x population) every tick over the WHOLE world - "
                f"their measurement has 50 leaders eating most of the 8 ms "
                f"half-budget.")

        desolate = [b for b in biomes if b in FLORA_OPTIONAL]
        if desolate:
            pass  # Bare ground is the point of these; not a defect.
        elif len(flora) < THIN_FLORA_BELOW:
            problems.append(
                f"{w['key']}: only {len(flora)} flora type(s) "
                f"({', '.join(flora) or 'none'}) - the ground will read bare "
                f"between creatures. something2 has no per-world decoration "
                f"density, so the only fix is a biome with richer flora.")

        if res["clamped"]:
            problems.append(f"{w['key']}: clamped to {MAX_WORLD_CREATURES}.")

        # Tiles-vs-pixels, checked rather than remembered. See check_units.
        problems.extend(check_units(w))

        rows.append({
            "key": w["key"], "name": w.get("name"), "biomes": biomes,
            "density": tier, "per_screen": round(ps, 1),
            "creatures": res["scatter"], "area": res["area"],
            # THESE TWO NUMBERS DISAGREE, AND IT IS NOT KNOWN WHICH IS RIGHT.
            #
            # `creatures` mirrors their `resolveDensity`: tier x area, with no
            # biome weighting - which is CORRECT, confirmed 2026-08-27 from
            # `resolveDensity`'s arity: it has no biome parameter, so a total
            # cannot depend on one. `per_screen` is derived from this same
            # count rather than from the tier, so the two can no longer
            # disagree the way they did for two days.
            "biome_multiplier": round(biome_multiplier(biomes), 2),
            "flora": flora, "creature_types": types,
            "variety": len(types), "leaders": leaders,
            # The WORLD's depth, not its first biome's. A transition world
            # genuinely mixes - Verdant Jungle with Blightworks is a reasonable
            # thing for an author to write - and reporting that as "surface"
            # because the surface one happened to be listed first is the same
            # class of quiet inaccuracy as the creature counter was.
            "depth": _depth_of(biomes),
            "socket_kib_s": round(kib, 1),
            "verdict": verdict,
        })

    for l in spec.get("links", []):
        if l["from"] not in keys or l["to"] not in keys:
            problems.append(f"link {l['from']} -{l['edge']}-> {l['to']} "
                            f"names a world that does not exist")

    linked = {k for l in spec.get("links", []) for k in (l["from"], l["to"])}
    for w in worlds:
        if w["key"] not in linked and len(worlds) > 1:
            problems.append(f"{w['key']} has no links - it is unreachable")

    entries = [w for w in worlds if w.get("is_entry")]
    if len(entries) != 1:
        problems.append(f"{len(entries)} entry worlds - something2 expects one")

    per = [r["per_screen"] for r in rows] or [0]

    # Kept for things the report cannot settle from here, and deliberately NOT
    # in `problems`: `ok` gates something2's validator, and a question about
    # what a number means is not a defect in the spec.
    #
    # It held the creatures-vs-per_screen contradiction until 2026-08-27, when
    # something2 read `resolveDensity` and settled it. Kept empty rather than
    # removed - the next such question should have somewhere to go that does
    # not fail a region.
    caveats = []
    raw = sum(r["creatures"] for r in rows)

    # THE TIERS ARE COARSE AND THE GAP IS SILENT WITHOUT THIS.
    #
    # 2.0, 4.1, 8.1, 14.0, 20.0 per screen. A target of 6.0 sits almost exactly
    # between `normal` and `dense`, and the nearest is `normal` at 4.1 - a 32%
    # shortfall the caller never asked for and could not see. Before the
    # multiplier was removed this was hidden: the compensation happened to push
    # thin biomes up a tier, so the number looked close for the wrong reason.
    if target_per_screen and per:
        got = sum(per) / len(per)
        if abs(got - target_per_screen) > 0.15 * max(target_per_screen, 0.1):
            nearer = min((t for t in DENSITY_ORDER if t != "dead"),
                         key=lambda t: abs(DENSITY_TIERS[t]
                                           * (TILES_PER_SCREEN / 1000.0)
                                           - target_per_screen))
            caveats.append(
                f"asked for {target_per_screen:.1f} creatures/screen, this "
                f"region averages {got:.1f}. The tiers are coarse - "
                + ", ".join(f"{t} {DENSITY_TIERS[t] * TILES_PER_SCREEN / 1000:.1f}"
                            for t in DENSITY_ORDER if t != "dead")
                + f" - and {nearer} is the closest one to your target. Ask for "
                  f"a number nearer a tier if the gap matters.")

    return {
        "worlds": rows,
        "problems": problems,
        "caveats": caveats,
        "ok": not problems,
        "totals": {
            "worlds": len(worlds),
            "creatures": raw,

            "mean_per_screen": round(sum(per) / len(per), 1),
            "min_per_screen": round(min(per), 1),
            "max_per_screen": round(max(per), 1),
            "empty_worlds": sum(1 for r in rows if r["verdict"] == "EMPTY"),
            "creature_types": len({t for r in rows for t in r["creature_types"]}),
            "leaders": sum(r["leaders"] for r in rows),
        },
        "notes": [
            "Creature counts are scatter only; packs are not modelled, so "
            "every figure is a floor. See PACKS_NOT_MODELLED.",
            f"One screen is ~{TILES_PER_SCREEN} tiles at something2's fixed "
            f"1280x720 canvas.",
        ],
    }


def to_json(spec: dict) -> str:
    """The file `make seed-map SPEC=<name>` reads, formatted like theirs."""
    return json.dumps(spec, indent=2) + "\n"
