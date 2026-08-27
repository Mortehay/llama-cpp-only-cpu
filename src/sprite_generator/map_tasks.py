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

RESOLVING THE GAPS

`build_map_job` finishes as soon as the TERRAIN is final, even when props are
still missing - that provisional state is the point of ADR 0007 D7. A second
task, `resolve_map_props`, then generates the missing art and rewrites the
picture in place.

It is a SEPARATE JOB ROW (`kind='map_props'`) rather than a continuation of the
map job, for one reason that matters: `job_runner.fail_stranded_jobs` already
sweeps every row in `jobs`, so a resolver whose worker died is reaped for free
and `GET /api/maps/{id}` can say so. Without a row there is nothing to reap,
and a lost Celery message leaves the map at `complete: false` forever with no
way to tell that from "still working" - which is exactly the strand the plan
named.

ONE resolver row per map, not one per prop. N rows would each have to
re-composite and rewrite the SAME picture file, racing on it, while the solo
pool serialises them anyway.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid

import numpy as np
import psycopg2
import psycopg2.extras
from PIL import Image, ImageDraw

import map_geometry
import pixelate
import regions
import tile_geometry
from job_runner import job_update, now
from tasks import celery_app

logger = logging.getLogger(__name__)

DB_URL = os.environ.get("DB_URL")
IMAGES_DIR = "/app/images"

# A generated prop is filed under a name derived from what was asked for, so
# the SECOND map that wants a windmill finds the first one's art on disk
# instead of spending the GPU again. Library-first (ADR 0007 D6) is only worth
# anything if the library actually grows.
PROP_PREFIX = "prop_"


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


def _tile_path(map_job_id: str, i: int) -> str:
    """Where a terrain's CUT tile is kept.

    Saved rather than held in memory because the resolver has to composite the
    same map again later, and regenerating four terrain tiles on the GPU to
    replace one prop would cost more than the prop did. It also closes a hole
    in the wire format: `terrains[].tile` is documented in the contract and the
    built tilemap was not emitting it, so nothing downstream could draw the
    grid itself.
    """
    return os.path.join(IMAGES_DIR, f"map_{map_job_id[:12]}_t{i}.png")


def _road_tile_path(map_job_id: str) -> str:
    return os.path.join(IMAGES_DIR, f"map_{map_job_id[:12]}_road.png")


def _picture_path(map_job_id: str) -> str:
    return os.path.join(IMAGES_DIR, f"map_{map_job_id[:12]}.png")


def _tilemap_path(map_job_id: str) -> str:
    return os.path.join(IMAGES_DIR, f"map_{map_job_id[:12]}.json")


def _replace_atomically(path: str, write) -> None:
    """Write via a neighbour and rename over the target.

    Both the picture and the tilemap are REWRITTEN IN PLACE by the resolver
    while `GET /api/maps/{id}` may be reading them. A plain overwrite gives a
    reader half a file - a truncated JSON parse error on a map that is
    perfectly fine, which would be a maddening bug to chase. `os.replace` is
    atomic within a filesystem, and the temp file is a sibling so it always is
    one.
    """
    # The temp name keeps NO usable extension, deliberately. A sibling called
    # `map_x.tmp.png` is indistinguishable from a real asset to anything that
    # scans the images directory, and this one is garbage by construction. The
    # cost is that PIL can no longer infer the format, so image callers pass it
    # explicitly - which is the better habit anyway.
    tmp = f"{path}.tmp"
    try:
        write(tmp)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _write_tilemap(path: str, tilemap: dict) -> None:
    def _write(target):
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(tilemap, fh)
    _replace_atomically(path, _write)


def _slug(want: str) -> str:
    """`"old windmill"` -> `"old_windmill"`. The library filing key.

    Deliberately lossy and deliberately stable: two maps asking for the same
    thing in the same words must land on the same file, or the library never
    gets a second hit. Nothing here reaches a shell or a SQL clause, but it is
    still restricted to a known alphabet because it becomes a filename.
    """
    s = re.sub(r"[^a-z0-9]+", "_", (want or "").lower()).strip("_")
    return s[:40] or "prop"


def _library_path(want: str) -> str:
    return os.path.join(IMAGES_DIR, f"{PROP_PREFIX}{_slug(want)}.png")


def _resolve_art(entity: dict, target: tuple[int, int]):
    """Art for one placement, and whether it is real.

    Three places are tried in the order they cost: the named asset, the prop
    library under the `want`, and finally the placeholder. The library lookup
    is what makes a resolved prop visible to EVERY later map rather than only
    the one that paid for it.
    """
    img = _asset_image(entity.get("asset")) if entity.get("asset") else None
    asset = entity.get("asset") if img is not None else None

    if img is None and entity.get("want"):
        path = _library_path(entity["want"])
        img = _asset_image(path)
        if img is not None:
            asset = os.path.basename(path)

    if img is None:
        return _placeholder(target[0] // 2, target[1] // 2), None, "pending"

    img.thumbnail(target, Image.LANCZOS)
    return img, asset, "placed"


def _dress(entities, tile_w: int, tile_h: int):
    """Placements -> draw objects, updating each entity's status in place.

    Shared by the build and by the resolver, which is the point: after the
    resolver has written a prop into the library, re-running exactly this turns
    the same placements into real art with no second scatter and therefore no
    chance of the entities moving between the two renders.
    """
    # Sprites are scaled to the tile so a tree is not the size of a world.
    target = (max(8, tile_w), max(8, tile_h * 2))

    objects, pending = [], []
    for e in entities:
        img, asset, status = _resolve_art(e, target)
        e["asset"], e["status"] = asset, status
        if status == "pending" and e.get("want") and e["want"] not in pending:
            pending.append(e["want"])
        objects.append({"x": e["x"], "y": e["y"], "layer": e["layer"],
                        "image": img})

    return objects, pending


def _coverage_warnings(coverage: dict, terrains) -> list:
    """Terrains that were declared and did not survive quantisation.

    `validate_terrains` exists because "a map silently missing a terrain gives
    no hint which one it lost". It can only check that the declared colours are
    far enough APART, which is not the same thing - a terrain can be perfectly
    separable and still capture nothing, or capture everything.

    Two ways that happens, both measured on real references:

      The declared colour does not match the art. A map whose sea is a muted
      blue, quantised against a navy `#2850c8`, produced 2.4% water. Against
      the sea's actual colour, 49%.

      A NEAR-NEUTRAL terrain is a sink. Grey sits close to the middle of Lab
      space, so it is the nearest match for anything desaturated. Dropping a
      grey `stone` from that same map took water from 2.4% to 58% WITHOUT
      touching the water colour.

    Neither is detectable before the painting exists, and both are silent: the
    map builds, looks plausible, and is missing a third of what was asked for.
    This is the one moment it is knowable, so it is said here.
    """
    out = []
    for t in terrains:
        got = coverage.get(t["name"], 0.0)
        if got < 0.005:
            out.append(
                f"terrain {t['name']!r} ({t['color']}) covers {got:.1%} of this "
                f"map - the painting has nothing that colour, or a more "
                f"neutral terrain captured it")
        elif got > 0.85 and len(terrains) > 2:
            out.append(
                f"terrain {t['name']!r} ({t['color']}) covers {got:.0%} of this "
                f"map - if that is not what you wanted, it is probably the "
                f"closest match in Lab to everything muddy in the painting")
    return out


def _landmarks(graph: dict | None) -> list:
    """A placed region becomes a prop, so the graph is VISIBLE on the picture.

    `want` is the KIND rather than the place's name, deliberately. A library
    keyed on "Saltmere" is a library of one; keyed on "port" it is reused by
    every map that ever has a port. The name still travels, on the entity, so
    a consumer can label it.

    They are ordinary pending props, which means Slice 4's resolver fills them
    in and Slice 4's reaper covers them. Nothing new had to be built for this.
    """
    if not graph:
        return []
    return [{"asset": None, "want": r["kind"], "x": r["x"], "y": r["y"],
             "layer": "props", "status": "pending",
             "region": r["name"]} for r in graph.get("regions", [])]


def _populate(spec: dict, grid, roads_grid, terrains, tile_w: int, tile_h: int,
              graph: dict | None = None):
    """Layers 3 and 4: scatter, then resolve each placement to art.

    Library first (ADR 0007 D6). A rule naming an `asset` costs no GPU; one
    naming only a `want` is a gap - it is placed anyway, with placeholder art
    and `status: pending`, so the map is usable now and improves later rather
    than being withheld until every prop exists.
    """
    landmarks = _landmarks(graph)
    rules = spec.get("scatter") or []
    if not rules and not landmarks:
        return [], [], []

    placements = []
    if rules:
        by_name = {t["name"]: i for i, t in enumerate(terrains)}
        resolved = [{**r, "terrain": [by_name[n] for n in r["terrain"]
                                      if n in by_name]} for r in rules]
        placements = map_geometry.scatter(grid, resolved,
                                          seed=int(spec.get("seed", 0)),
                                          roads=roads_grid)

    # A tree drawn on top of a town reads as an art bug rather than as a
    # generation one, which is the failure `scatter` avoids roads for. Scatter
    # cannot know about landmarks, so they are subtracted here.
    taken = {(m["x"], m["y"]) for m in landmarks}
    placements = [p for p in placements if (p["x"], p["y"]) not in taken]

    entities = landmarks + [
        {"asset": p.get("asset"), "want": p.get("want"),
         "x": p["x"], "y": p["y"], "layer": p["layer"],
         "status": "pending"} for p in placements]

    objects, pending = _dress(entities, tile_w, tile_h)
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
        warnings = _coverage_warnings(coverage, terrains)
        for warning in warnings:
            logger.warning("map %s: %s", job_id, warning)

        tiles = []
        for i, t in enumerate(terrains):
            job_update(job_id,
                       progress_pct=25 + int(45 * i / max(len(terrains), 1)),
                       progress_msg=f"tile {i + 1}/{len(terrains)}: {t['name']}")
            tile = _make_tile(t, tile_w, tile_h,
                              int(spec.get("colors", 16)),
                              int(spec.get("seed", 0)) + i,
                              spec.get("llm_name"))
            # Kept on disk so the resolver can composite again without paying
            # for the GPU twice, and so the served tilemap can name its tiles.
            tile.save(_tile_path(job_id, i))
            tiles.append(tile)

        # The region graph, if asked for. Built here - AFTER quantisation and
        # before anything is drawn - because a landmark has to sit on real
        # ground, and which ground is real is not known until now.
        graph, road_segments = None, [[tuple(p) for p in seg]
                                      for seg in spec.get("roads") or []]
        if spec.get("regions"):
            job_update(job_id, progress_pct=68,
                       progress_msg=f"naming {spec['regions']} place(s)")
            graph = regions.build(
                grid, terrains, count=int(spec["regions"]),
                theme=spec.get("theme") or spec.get("prompt"),
                seed=int(spec.get("seed", 0)),
                use_llm=bool(spec.get("region_llm", True)))
            road_segments = [[tuple(p) for p in seg]
                             for seg in regions.segments(graph["roads"])]
            logger.info("map %s graph: %s", job_id, graph["note"])

        # Layer 2. Refused rather than drawn wrong if a road would cross
        # terrain nobody can walk on - a road across a lake is not a road.
        roads_grid, road_tiles = None, None
        if road_segments and spec.get("road_tile"):
            job_update(job_id, progress_pct=70, progress_msg="laying roads")
            walkable = [t.get("walkable", True) for t in terrains]
            roads_grid = map_geometry.road_layer(
                grid.shape, road_segments, walkable=walkable, terrain=grid)
            road_tiles = [_make_tile({"name": "road", "tile": spec["road_tile"]},
                                     tile_w, tile_h, int(spec.get("colors", 16)),
                                     int(spec.get("seed", 0)), None)]
            road_tiles[0].save(_road_tile_path(job_id))

        # Layers 3 and 4.
        job_update(job_id, progress_pct=72, progress_msg="placing entities")
        entities, objects, pending = _populate(
            spec, grid, roads_grid, terrains, tile_w, tile_h, graph=graph)

        job_update(job_id, progress_pct=75, progress_msg="compositing")
        picture = map_geometry.composite(grid, tiles, roads=roads_grid,
                                         road_tiles=road_tiles,
                                         objects=objects)

        pic_path = _picture_path(job_id)
        picture.save(pic_path)

        tilemap = {
            "id": job_id,
            "name": spec.get("name"),
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
                 # The cut tile itself, so a consumer can draw the grid rather
                 # than only look at the picture. Browser path, not the
                 # container path - `/images` is what is mounted.
                 "tile": "/images/" + os.path.basename(_tile_path(job_id, i)),
                 "coverage": coverage.get(t["name"], 0.0)}
                for i, t in enumerate(terrains)
            ],
            "road_tile": ("/images/" + os.path.basename(_road_tile_path(job_id))
                          if road_tiles else None),
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
            # What the places on this map ARE. Read off the terrain above, so
            # every coordinate in here is a tile that exists and is walkable.
            # `dropped` is the honest half: everything proposed that this
            # terrain could not accommodate, and why.
            "region_graph": graph,
            # Terrains that were declared and did not survive quantisation, or
            # swallowed everything. The map is fine; what was asked for is not
            # what arrived, and this is the only moment that is knowable.
            "warnings": warnings,
            "picture_url": f"/api/jobs/{job_id}/sheet",
        }
        # Queued BEFORE the tilemap is written, so `props_job` is in the very
        # first version a consumer can read. A map that says `complete: false`
        # without saying what is going to fix it is the strand all over again.
        if pending:
            tilemap["props_job"] = _queue_props_job(job_id, spec, pending)

        map_path = _tilemap_path(job_id)
        _write_tilemap(map_path, tilemap)

    except Exception as e:
        logger.exception("map job %s failed", job_id)
        job_update(job_id, status="failed", finished_at=now(), error=str(e))
        return {"error": str(e)}

    job_update(job_id, status="done", finished_at=now(), progress_pct=100,
               progress_msg=f"done, {picture.width}x{picture.height}",
               sheet_path=pic_path, atlas_path=map_path)
    logger.info("map job %s -> %s", job_id, pic_path)
    return {"picture": pic_path, "map": map_path, "coverage": coverage}


# --- Resolving the gaps ---------------------------------------------------


def _queue_props_job(map_job_id: str, spec: dict, wants: list) -> str:
    """A job row for the prop generation, and the Celery message to run it.

    The row is the whole point. `job_runner.fail_stranded_jobs` sweeps `jobs`
    without caring about `kind`, so a resolver whose Celery message was lost -
    a purged queue, a worker killed mid-generate - gets marked failed at the
    next worker boot exactly like a sheet job would, and `GET /api/maps/{id}`
    can then tell a caller "this is not coming" instead of leaving it to wait
    on `complete: false` forever.

    That is the failure `aece983` fixed for training runs, and the one the
    Slice 4 notes flagged as still open here.
    """
    props_id = str(uuid.uuid4())
    payload = {
        "kind": "map_props",
        "map_job": map_job_id,
        "name": spec.get("name"),
        "wants": list(wants),
        "llm_name": spec.get("llm_name"),
        "seed": int(spec.get("seed", 0)),
    }
    with psycopg2.connect(DB_URL) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO jobs (id, kind, status, spec, progress_msg) "
            "VALUES (%s, 'map_props', 'queued', %s, %s)",
            (props_id, json.dumps(payload),
             f"queued, {len(wants)} prop(s) for map {map_job_id[:8]}"))

    async_result = celery_app.send_task("maps.resolve_map_props",
                                        args=[props_id])
    with psycopg2.connect(DB_URL) as conn, conn.cursor() as cur:
        cur.execute("UPDATE jobs SET celery_task_id = %s WHERE id = %s::uuid",
                    (async_result.id, props_id))

    logger.info("map %s queued props job %s for %s", map_job_id, props_id,
                ", ".join(wants))
    return props_id


def _release_vram() -> None:
    """Hand the caching allocator's spare blocks back to the driver.

    MEASURED on this box after one prop run, with the queue empty and
    llama.cpp asleep:

        allocated  6.63 GB      what the pipeline is really using
        reserved  11.32 GB      what PyTorch is sitting on
        total     12.00 GB

    So nearly 5 GB was held and free. `nvidia-smi` reports the RESERVED
    figure, which is why the card read as full with nothing running, and why
    the next big allocation failed - first `CUDA driver error: device not
    ready`, then a `CUDACachingAllocator` internal assert. That is the
    fragmented-allocator failure ADR 0005 already describes.

    Called once when the whole resolve is finished, not between props: freeing
    and re-requesting around every generation would trade the fragmentation for
    latency and get neither.

    This does NOT unload the pipeline - `get_sd_pipeline` caches it on purpose,
    and the 6.63 GB stays. It only stops this task from leaving the other 5 GB
    looking occupied to everything else on the card.
    """
    try:
        import torch
        if torch.cuda.is_available():
            before = torch.cuda.memory_reserved() / 2**30
            torch.cuda.empty_cache()
            after = torch.cuda.memory_reserved() / 2**30
            logger.info("released %.2f GB of reserved VRAM (%.2f -> %.2f)",
                        before - after, before, after)
    except Exception as e:
        # Never fatal. The props are generated and the map is about to be
        # redrawn; failing that over a memory hint would be absurd.
        logger.warning("could not release VRAM: %s", e)


def _register_prop(want: str, path: str, llm_name: str | None) -> None:
    """Make the generated prop visible in the gallery.

    Writing the file is not enough: nothing lists the images directory, so an
    asset that is not in `sprite_images` exists only for the map that happened
    to want it. `assets_v` unions this table with finished jobs, so one INSERT
    is the whole of "it appears in the library".
    """
    try:
        with psycopg2.connect(DB_URL) as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sprite_images (prompt, file_path, image_type, "
                "                           progress_pct, progress_msg, llm_name) "
                "VALUES (%s, %s, 'prop', 100, 'done', %s)",
                (want, path, llm_name))
    except Exception as e:
        # The art is on disk and the map will find it. Failing the whole
        # resolve because a gallery row did not write would throw away work
        # that actually succeeded.
        logger.error("prop %r generated but not registered: %s", want, e)


def _generate_prop(want: str, llm_name: str | None, seed: int) -> str:
    """One prop, cut out, filed in the library. Returns its path.

    Asked for as a lone object on a plain background, because
    `remove_background` flood-fills from the border - a prop generated inside a
    scene has no background to fill from and comes back as an opaque square.
    """
    from tasks import default_model, get_sd_pipeline, remove_background
    import torch

    model = llm_name or default_model()
    brief = (f"{want}, single isolated object, centered, full view, "
             f"pixel art, top-down three-quarter view, flat lighting, "
             f"plain flat white background, no ground, no scene, no shadow")
    negative = "background, scenery, landscape, multiple, text, watermark, frame"

    pipe = get_sd_pipeline(model)
    if not pipe:
        raise ValueError(f"model {model!r} failed to load")

    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    img = pipe(prompt=brief, negative_prompt=negative, num_inference_steps=25,
               guidance_scale=7.5, generator=gen).images[0]
    img = remove_background(img).convert("RGBA")

    # REFUSE ONLY WHAT IS CERTAINLY BROKEN, as the entity cutout path settled
    # on in c967bed. A prop is composited over terrain, so an opaque one is a
    # white rectangle sitting in a field and nobody notices until they look at
    # the map. Zero transparent pixels means the flood fill did nothing at all,
    # which cannot be a correct cutout. Subtler thresholds were measured on the
    # entity path and were wrong in both directions.
    alpha = np.asarray(img)[..., 3]
    if not bool((alpha < 128).any()):
        raise ValueError(
            f"cutout for {want!r} left no transparency; it would composite as "
            f"an opaque block over the terrain")

    path = _library_path(want)
    _replace_atomically(path, lambda t: img.save(t, format="PNG"))
    _register_prop(want, path, model)
    return path


def _rerender_map(map_job_id: str) -> dict:
    """Composite the map again with whatever art now exists.

    Everything is reloaded from the tilemap and the saved tiles, so this costs
    no GPU and - the part that matters - does NOT re-scatter. Re-scattering
    would be deterministic from the seed and still wrong: it would depend on
    the grids round-tripping through JSON identically, and a map whose trees
    moved when a windmill resolved would be a baffling bug to be handed.
    """
    map_path = _tilemap_path(map_job_id)
    with open(map_path, "r", encoding="utf-8") as fh:
        tilemap = json.load(fh)

    grid = np.array(tilemap["layers"]["terrain"], dtype=int)
    tiles = [Image.open(_tile_path(map_job_id, t["id"])).convert("RGBA")
             for t in tilemap["terrains"]]

    roads_grid, road_tiles = None, None
    if tilemap.get("road_tile"):
        roads_grid = np.array(tilemap["layers"]["roads"], dtype=int)
        road_tiles = [Image.open(_road_tile_path(map_job_id)).convert("RGBA")]

    tile_w = int(tilemap["tile"]["w"])
    tile_h = int(tilemap["tile"]["h"])
    objects, pending = _dress(tilemap["entities"], tile_w, tile_h)

    picture = map_geometry.composite(grid, tiles, roads=roads_grid,
                                     road_tiles=road_tiles, objects=objects)
    _replace_atomically(_picture_path(map_job_id),
                        lambda t: picture.save(t, format="PNG"))

    tilemap["pending"] = pending
    tilemap["complete"] = not pending
    _write_tilemap(map_path, tilemap)
    return tilemap


@celery_app.task(bind=True, name="maps.resolve_map_props")
def resolve_map_props(self, props_job_id: str):
    """Generate the props a map is missing, then draw it again.

    PARTIAL SUCCESS IS THE NORMAL OUTCOME and is recorded as success. One want
    that will not cut out must not discard the four that did: the map goes from
    eight placeholders to one, stays `complete: false`, and names what is still
    open. Failing the whole job would leave all five as placeholders and make
    the map worse for having tried.

    The job fails only when NOTHING could be resolved, which is the case worth
    surfacing - a dead model, an unreadable images directory.
    """
    spec = _spec(props_job_id)
    if spec is None:
        logger.error("props job %s vanished before it started", props_job_id)
        return {"error": "no such job"}

    map_job_id = spec.get("map_job")
    if not map_job_id:
        job_update(props_job_id, status="failed", finished_at=now(),
                   error="props job has no map to resolve")
        return {"error": "no map_job in spec"}

    wants = list(spec.get("wants") or [])
    seed = int(spec.get("seed", 0))

    job_update(props_job_id, status="running", started_at=now(), progress_pct=2,
               progress_msg=f"{len(wants)} prop(s) to generate")

    resolved, failed = [], {}
    for i, want in enumerate(wants):
        job_update(props_job_id,
                   progress_pct=2 + int(88 * i / max(len(wants), 1)),
                   progress_msg=f"prop {i + 1}/{len(wants)}: {want}")
        try:
            # Library first even here: another map may have generated this
            # exact prop while this job sat in the queue.
            if os.path.exists(_library_path(want)):
                logger.info("prop %r already in the library", want)
            else:
                _generate_prop(want, spec.get("llm_name"), seed + i)
            resolved.append(want)
        except Exception as e:
            logger.exception("prop %r failed for map %s", want, map_job_id)
            failed[want] = str(e)

    _release_vram()

    try:
        job_update(props_job_id, progress_pct=92,
                   progress_msg="compositing the map again")
        tilemap = _rerender_map(map_job_id)
    except Exception as e:
        # The props may well have generated. The MAP is what could not be
        # updated, and that is a real failure - its placeholders are still on
        # it, so a caller told "done" would be told a lie.
        logger.exception("re-render of map %s failed", map_job_id)
        job_update(props_job_id, status="failed", finished_at=now(),
                   error=f"props generated but the map could not be "
                         f"redrawn: {e}")
        return {"error": str(e), "resolved": resolved}

    if resolved:
        summary = f"resolved {len(resolved)}/{len(wants)}"
        if failed:
            summary += f", still missing: {', '.join(sorted(failed))}"
        job_update(props_job_id, status="done", finished_at=now(),
                   progress_pct=100, progress_msg=summary)
    else:
        job_update(props_job_id, status="failed", finished_at=now(),
                   error="; ".join(f"{w}: {e}" for w, e in failed.items())
                         or "nothing to resolve")

    logger.info("map %s props: %d resolved, %d failed, complete=%s",
                map_job_id, len(resolved), len(failed), tilemap["complete"])
    return {"resolved": resolved, "failed": failed,
            "complete": tilemap["complete"], "pending": tilemap["pending"]}
