"""Draw a region so emptiness is visible before anything is seeded.

WHAT EACH CELL IS, AND WHY

Every world is drawn as ONE SCREENFUL, not as a shrunken whole world. A whole
world scaled into a thumbnail says nothing useful - 4000 dots and 40 dots both
turn into grey mush - whereas a screen is the unit a player actually
experiences, and it is the unit something2's own density table was tuned
against ("perThousand * 0.225 is creatures per screen").

So the dot count in a cell IS the creatures-per-screen figure. A cell with two
dots looks empty on this page because it will feel empty in the game. That is
the entire point: the judgement should be available before `make seed-map`, not
after walking around.

Colours are something2's own biome colours, so the preview cannot flatter a
palette this project invented.
"""

from __future__ import annotations

import hashlib

from PIL import Image, ImageDraw

import world_gen

CELL = 190          # one world, drawn as one screen
PAD = 26            # gutter between cells, where links are drawn
LABEL_H = 34
BG = (18, 18, 30)
INK = (232, 232, 240)
MUTED = (150, 150, 168)
LINK = (92, 92, 120)


def _rng(seed: str):
    """Deterministic positions - the same spec previews identically."""
    h = hashlib.sha256(seed.encode()).digest()
    state = int.from_bytes(h[:8], "big") or 1

    def nxt(n):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (state >> 17) % n
    return nxt


def _hex(c: str):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _band(draw, x, y, w, h, biomes):
    """Biome bands, the way their biome_cell bands terrain across a world."""
    if not biomes:
        draw.rectangle([x, y, x + w, y + h], fill=(40, 40, 52))
        return
    step = h / len(biomes)
    for i, b in enumerate(biomes):
        col = _hex(world_gen.BIOMES.get(b, {}).get("color", "#333344"))
        draw.rectangle([x, int(y + i * step), x + w, int(y + (i + 1) * step)],
                       fill=col)


def render(spec: dict, rep: dict | None = None) -> Image.Image:
    rep = rep or world_gen.report(spec)
    rows = {r["key"]: r for r in rep["worlds"]}
    worlds = spec.get("worlds", [])
    if not worlds:
        raise ValueError("nothing to preview - the region has no worlds")

    xs = [w["grid"][0] for w in worlds]
    ys = [w["grid"][1] for w in worlds]
    cols, rws = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
    ox, oy = min(xs), min(ys)

    step = CELL + PAD
    W = cols * step + PAD
    H = rws * (step + LABEL_H) + PAD + 46

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    d.text((PAD, 14), f"{spec.get('name', 'region')} - each cell is ONE SCREEN "
                      f"({world_gen.TILES_PER_SCREEN} tiles)", fill=INK)

    def cell_xy(w):
        gx, gy = w["grid"]
        return (PAD + (gx - ox) * step, 46 + (gy - oy) * (step + LABEL_H))

    # Links first, so cells sit on top of them.
    pos = {w["key"]: cell_xy(w) for w in worlds}
    for l in spec.get("links", []):
        a, b = pos.get(l["from"]), pos.get(l["to"])
        if not a or not b:
            continue
        d.line([a[0] + CELL // 2, a[1] + CELL // 2,
                b[0] + CELL // 2, b[1] + CELL // 2], fill=LINK, width=3)

    for w in worlds:
        x, y = cell_xy(w)
        r = rows.get(w["key"], {})
        biomes = w.get("biomes", [])
        _band(d, x, y, CELL, CELL, biomes)

        # Roads, drawn as bands from the centre toward each exit.
        #
        # Honest at this scale for one reason: the cell is the world's CENTRAL
        # screen, which is precisely where the roads meet. A road running to a
        # far edge genuinely does leave this screen in that direction.
        roads = w.get("roads", [])
        if roads:
            size = w.get("width", 128)
            band = max(4, CELL // 16)
            cx, cy = x + CELL // 2, y + CELL // 2
            road_col = (150, 126, 92)
            for line in roads:
                r1, c1 = line[-1]
                if c1 <= 0:                       # W
                    d.rectangle([x, cy - band // 2, cx, cy + band // 2], fill=road_col)
                elif c1 >= size - 1:              # E
                    d.rectangle([cx, cy - band // 2, x + CELL, cy + band // 2], fill=road_col)
                elif r1 <= 0:                     # N
                    d.rectangle([cx - band // 2, y, cx + band // 2, cy], fill=road_col)
                elif r1 >= size - 1:              # S
                    d.rectangle([cx - band // 2, cy, cx + band // 2, y + CELL], fill=road_col)

        rnd = _rng(w["key"])
        flora = r.get("flora", [])

        # Flora first - trees and stones are what fill the ground BETWEEN
        # creatures, and their absence is a different complaint from a low
        # creature count. Shown as small dark marks, one cluster per type.
        if flora:
            for i in range(len(flora) * 7):
                fx, fy = x + 6 + rnd(CELL - 12), y + 6 + rnd(CELL - 12)
                s = 2 + (i % 2)
                d.ellipse([fx, fy, fx + s * 2, fy + s * 2],
                          fill=(28, 44, 30) if i % 3 else (70, 66, 60))

        # Creatures: exactly per_screen of them, rounded. This is the number
        # under judgement, drawn rather than described.
        n = int(round(r.get("per_screen", 0)))
        for _ in range(n):
            cx, cy = x + 10 + rnd(CELL - 20), y + 10 + rnd(CELL - 20)
            d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4],
                      fill=(226, 74, 74), outline=(20, 12, 12))

        verdict = r.get("verdict", "ok")
        border = {"EMPTY": (226, 74, 74), "CROWDED": (226, 176, 74)}.get(
            verdict, (70, 70, 92))
        d.rectangle([x, y, x + CELL, y + CELL], outline=border, width=3)

        if w.get("is_entry"):
            d.rectangle([x + 6, y + 6, x + 74, y + 24], fill=(20, 20, 34))
            d.text((x + 12, y + 10), "ENTRY", fill=(120, 220, 140))

        d.text((x, y + CELL + 4),
               f"{w.get('name', w['key'])}  [{w.get('density')}]", fill=INK)
        line = (f"{r.get('per_screen', 0)}/screen  "
                f"{r.get('creatures', 0)} total  x{r.get('biome_multiplier', 1)}")
        d.text((x, y + CELL + 17), line,
               fill=(226, 74, 74) if verdict == "EMPTY" else MUTED)

    return img
