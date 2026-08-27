"""The map build, as a Celery task. Deliberately NOT in tasks.py.

`tasks.py` is 150 KB and `project-context.md` asks that it be split before more
paths are added to it. This does not fix that, but it declines to make it
worse: the map path lives here and is registered through the Celery app's
`include`, so `tasks.py` gains one name in a list rather than five hundred
lines.

STAGES

    painting   one low-res image, one pixel per tile      (GPU, or reused)
    quantise   painting -> terrain ids                    (CPU, instant)
    tiles      one rhombus per terrain                    (GPU, or reused)
    roads      axis-aligned segments -> a grid layer      (CPU)
    composite  layers -> the picture                      (CPU)

Only the two marked GPU cost anything, and both can be skipped entirely: a map
whose terrains all name existing tiles and whose layout comes from an uploaded
reference builds in seconds without touching the card.

LAYERS

`map_geometry.LAYERS` is the authoritative list. Two are grids - terrain and
roads, one value per tile, flat by definition - and two are object lists,
creatures and props, which STAND on the ground and are therefore composited in
a single shared depth sort rather than one after the other. Players and
projectiles are runtime, not authored, and appear in neither.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import psycopg2
import psycopg2.extras
from PIL import Image, ImageDraw

import map_geometry
import pixelate
import tile_geometry
from job_runner import job_update, now
from tasks import celery_app

logger = logging.getLogger(__name__)

DB_URL = os.environ.get("DB_URL")
IMAGES_DIR = "/app/images"


def _spec(job_id: str) -> dict | None:
    with psycopg2.connect(DB_URL) as conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT spec FROM jobs WHERE id = %s::uuid", (job_id,))
        row = cur.fetchone()
    return dict(row["spec"] or {}) if row else None


def _reference_path(ref_id: str) -> str | None:
    with psycopg2.connect(DB_URL) as conn, conn.cursor() as cur:
        cur.execute("SELECT file_path FROM reference_assets "
                    "WHERE id = %s::uuid AND deleted = false", (ref_id,))
        row = cur.fetchone()
    return row[0] if row else None


def _paint(spec: dict) -> Image.Image:
    """The biome painting: one pixel per tile, and never shown to anyone.

    Reused from a map reference when `painting_from` is set, which is what lets
    a map be built before the map adapter exists. Otherwise generated, then
    forced to the terrain palette by `quantize` regardless of what came back.
    """
    ref_id = spec.get("painting_from")
    if ref_id:
        path = _reference_path(ref_id)
        if not path or not os.path.exists(path):
            raise ValueError(f"painting_from reference {ref_id} has no file")
        return Image.open(path).convert("RGB")

    from tasks import default_model, get_sd_pipeline
    import torch

    # Generated large and shrunk by quantize(), because a diffusion model asked
    # for a 64px image produces noise. The prompt asks for a MAP, not a scene:
    # what matters is that regions of ground read as distinct flat colours.
    brief = (f"{spec.get('prompt', '')}, top-down world map, flat regions of "
             f"colour, distinct biomes, no text, no labels, no border, "
             f"no shading, no perspective")
    pipe = get_sd_pipeline(spec.get("llm_name") or default_model())
    gen = torch.Generator(device=pipe.device).manual_seed(int(spec.get("seed", 0)))
    return pipe(prompt=brief, num_inference_steps=25, guidance_scale=7.5,
                generator=gen).images[0].convert("RGB")


def _make_tile(terrain: dict, w: int, h: int, colors: int,
               seed: int, llm_name: str | None) -> Image.Image:
    """One rhombus for one terrain. Reuses a named tile when given one.

    The generated path mirrors the tile job: the model paints a texture and the
    outline is applied afterwards, because a tile the model outlined does not
    tessellate.
    """
    named = terrain.get("tile")
    if named:
        path = named if os.path.isabs(named) else os.path.join(IMAGES_DIR,
                                                               os.path.basename(named))
        if not os.path.exists(path):
            raise ValueError(f"terrain {terrain['name']!r} names tile "
                             f"{named!r}, which does not exist")
        return tile_geometry.cut_tile(Image.open(path).convert("RGBA"), w, h)

    from tasks import default_model, get_sd_pipeline
    import torch

    full = (f"{terrain.get('prompt') or terrain['name']}, seamless tiling "
            f"ground texture, top-down view, pixel art, flat lighting, "
            f"no shadows, no objects")
    pipe = get_sd_pipeline(llm_name or default_model())
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    image = pipe(prompt=full, num_inference_steps=25, guidance_scale=7.5,
                 generator=gen).images[0]

    tile = tile_geometry.cut_tile(image, w, h)
    arr = np.asarray(tile.convert("RGBA")).copy()
    if (arr[..., 3] >= 128).any():
        palette = pixelate.extract_palette(tile, colors)
        arr[..., :3] = pixelate.snap_to_palette(arr[..., :3], palette)
    tile = Image.fromarray(arr, "RGBA")
    # Re-applied because quantising in Lab can nudge edge pixels either side of
    # the alpha threshold.
    tile.putalpha(tile_geometry.diamond_mask(w, h))
    return tile


def _placeholder(w: int, h: int) -> Image.Image:
    """Stand-in art for a prop that does not exist yet.

    Deliberately ugly. A placeholder that reads as finished is worse than a
    missing one - it gets cached downstream as the real thing, which is the
    failure ADR 0007 D7 names. Magenta and a cross say "not done" in a way no
    generated sprite ever will.
    """
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w - 1, h - 1], outline=(255, 0, 200, 255), width=2)
    d.line([0, 0, w - 1, h - 1], fill=(255, 0, 200, 255), width=2)
    d.line([0, h - 1, w - 1, 0], fill=(255, 0, 200, 255), width=2)
    return img


def _asset_image(name: str) -> Image.Image | None:
    """An existing asset by filename, or None if it is not on disk."""
    if not name:
        return None
    path = name if os.path.isabs(name) else os.path.join(
        IMAGES_DIR, os.path.basename(name))
    if not os.path.exists(path):
        return None
    return Image.open(path).convert("RGBA")


def _populate(spec: dict, grid, roads_grid, terrains, tile_w: int, tile_h: int):
    """Layers 3 and 4: scatter, then resolve each placement to art.

    Library first (ADR 0007 D6). A rule naming an `asset` costs no GPU; one
    naming only a `want` is a gap - it is placed anyway, with placeholder art
    and `status: pending`, so the map is usable now and improves later rather
    than being withheld until every prop exists.
    """
    rules = spec.get("scatter") or []
    if not rules:
        return [], [], []

    by_name = {t["name"]: i for i, t in enumerate(terrains)}
    resolved = [{**r, "terrain": [by_name[n] for n in r["terrain"]
                                  if n in by_name]} for r in rules]

    placements = map_geometry.scatter(grid, resolved,
                                      seed=int(spec.get("seed", 0)),
                                      roads=roads_grid)

    # Sprites are scaled to the tile so a tree is not the size of a world.
    target = (max(8, tile_w), max(8, tile_h * 2))

    entities, objects, pending = [], [], []
    for p in placements:
        img = _asset_image(p.get("asset")) if p.get("asset") else None
        if img is not None:
            img.thumbnail(target, Image.LANCZOS)
            status = "placed"
        else:
            img = _placeholder(target[0] // 2, target[1] // 2)
            status = "pending"
            if p.get("want") and p["want"] not in pending:
                pending.append(p["want"])

        entities.append({"asset": p.get("asset"), "want": p.get("want"),
                         "x": p["x"], "y": p["y"], "layer": p["layer"],
                         "status": status})
        objects.append({"x": p["x"], "y": p["y"], "layer": p["layer"],
                        "image": img})

    return entities, objects, pending


@celery_app.task(bind=True, name="maps.build_map_job")
def build_map_job(self, job_id: str):
    spec = _spec(job_id)
    if spec is None:
        logger.error("map job %s vanished before it started", job_id)
        return {"error": "no such job"}

    size = int(spec.get("size", 64))
    tile_w = int(spec.get("tile_w", 64))
    tile_h = int(spec.get("tile_h", tile_w // 2))
    terrains = spec.get("terrains", [])

    job_update(job_id, status="running", started_at=now(), progress_pct=5,
               progress_msg=f"painting a {size}x{size} biome layout")

    try:
        painting = _paint(spec)

        job_update(job_id, progress_pct=25, progress_msg="quantising to terrain")
        grid = map_geometry.quantize(painting, terrains, (size, size))
        coverage = map_geometry.coverage(grid, terrains)

        tiles = []
        for i, t in enumerate(terrains):
            job_update(job_id,
                       progress_pct=25 + int(45 * i / max(len(terrains), 1)),
                       progress_msg=f"tile {i + 1}/{len(terrains)}: {t['name']}")
            tiles.append(_make_tile(t, tile_w, tile_h,
                                    int(spec.get("colors", 16)),
                                    int(spec.get("seed", 0)) + i,
                                    spec.get("llm_name")))

        # Layer 2. Refused rather than drawn wrong if a road would cross
        # terrain nobody can walk on - a road across a lake is not a road.
        roads_grid, road_tiles = None, None
        if spec.get("roads"):
            job_update(job_id, progress_pct=70, progress_msg="laying roads")
            walkable = [t.get("walkable", True) for t in terrains]
            roads_grid = map_geometry.road_layer(
                grid.shape,
                [[tuple(p) for p in seg] for seg in spec["roads"]],
                walkable=walkable, terrain=grid)
            road_tiles = [_make_tile({"name": "road", "tile": spec["road_tile"]},
                                     tile_w, tile_h, int(spec.get("colors", 16)),
                                     int(spec.get("seed", 0)), None)]

        # Layers 3 and 4.
        job_update(job_id, progress_pct=72, progress_msg="placing entities")
        entities, objects, pending = _populate(
            spec, grid, roads_grid, terrains, tile_w, tile_h)

        job_update(job_id, progress_pct=75, progress_msg="compositing")
        picture = map_geometry.composite(grid, tiles, roads=roads_grid,
                                         road_tiles=road_tiles,
                                         objects=objects)

        pic_path = os.path.join(IMAGES_DIR, f"map_{job_id[:12]}.png")
        picture.save(pic_path)

        tilemap = {
            "id": job_id,
            "name": spec.get("name"),
            # No entity placements yet - that is Slice 4. The key is present
            # and empty so a consumer written against this shape does not have
            # to branch when it arrives.
            # Provisional is a SERVED state, not an error (ADR 0007 D7). A map
            # with unresolved props is walkable now and improves without being
            # re-requested; withholding it would let one missing prop block the
            # whole map.
            "complete": not pending,
            "pending": pending,
            "size": {"w": size, "h": size},
            "tile": {"w": tile_w, "h": tile_h},
            "projection_ratio": spec.get("ratio"),
            "terrains": [
                {"id": i, "name": t["name"], "color": t["color"],
                 "walkable": t.get("walkable", True),
                 "coverage": coverage.get(t["name"], 0.0)}
                for i, t in enumerate(terrains)
            ],
            # Grid layers are one value per tile and can only ever be flat.
            # `roads` is present and empty rather than absent when unused, so a
            # consumer never has to branch on whether the key exists.
            "layers": {
                "terrain": grid.tolist(),
                "roads": (roads_grid.tolist() if roads_grid is not None
                          else [[0] * size for _ in range(size)]),
            },
            # Standing things: creatures and props in ONE list, because they
            # must be depth-sorted TOGETHER when drawn. `layer` says which
            # authored layer each came from, so they can still be filtered or
            # re-authored separately.
            "entities": entities,
            "picture_url": f"/api/jobs/{job_id}/sheet",
        }
        map_path = os.path.join(IMAGES_DIR, f"map_{job_id[:12]}.json")
        with open(map_path, "w", encoding="utf-8") as fh:
            json.dump(tilemap, fh)

    except Exception as e:
        logger.exception("map job %s failed", job_id)
        job_update(job_id, status="failed", finished_at=now(), error=str(e))
        return {"error": str(e)}

    job_update(job_id, status="done", finished_at=now(), progress_pct=100,
               progress_msg=f"done, {picture.width}x{picture.height}",
               sheet_path=pic_path, atlas_path=map_path)
    logger.info("map job %s -> %s", job_id, pic_path)
    return {"picture": pic_path, "map": map_path, "coverage": coverage}
