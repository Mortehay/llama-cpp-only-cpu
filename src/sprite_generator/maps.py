"""World maps: a terrain set in, a tilemap and a picture out.

A map is not a big tile and not a picture of a place. It is a BIOME PAINTING
quantised to a declared terrain set, composited from the tiles that terrain set
names - so the walkable grid and the picture are the same artefact read twice
and cannot disagree. See `.ai/decisions/0007` and `.ai/specs/maps/plan.md`.

WHY THE TERRAIN SET IS DECLARED RATHER THAN DISCOVERED

Quantisation is a LOOKUP. The terrains are authored first, the painting is
forced to their colours, and a tile id is an index rather than a nearest-match
that might be quietly wrong. Extracting colours from the painting and labelling
them afterwards would make naming a per-map chore forever.

The endpoint lives here rather than in jobs.py for the reason tiles.py gives:
a map spec has nothing in common with a sheet spec. What they share is the
queue and the polling contract - a map is a row in `jobs` with `kind='map'`, so
something2 polls it through the same `GET /api/jobs/{id}`.

Storage reuses the columns a sheet job already has, which is why this needs no
migration: the PICTURE goes in `sheet_path` (so `/api/jobs/{id}/sheet` serves it
unchanged) and the TILEMAP JSON goes in `atlas_path`.
"""

from __future__ import annotations

import json
import logging
import os
import uuid

import psycopg2
import psycopg2.extras
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

import auth
import map_geometry
import tile_geometry

logger = logging.getLogger(__name__)
router = APIRouter()

DB_URL = os.environ.get("DB_URL")
IMAGES_DIR = "/app/images"

# Grid bounds. Both axes of the picture scale with the SUM of the dimensions,
# so 256x256 at 64px tiles is a 16384px canvas - past this a map stops being an
# image and starts being a memory problem.
MIN_SIZE, MAX_SIZE, DEFAULT_SIZE = 8, 256, 64

# More ground types than this and the forced palette stops separating cleanly
# in Lab, which `validate_terrains` would reject anyway - this is the friendlier
# error, raised before the colours are even parsed.
MAX_TERRAINS = 16


class Terrain(BaseModel):
    name: str = Field(..., min_length=1, max_length=40)
    color: str = Field(..., description="#rrggbb the painting is forced to")
    prompt: str | None = Field(
        None, description="what this ground is, if its tile must be generated")
    # Library first: a terrain that names a tile costs no GPU at all. Only the
    # gaps are generated, which is the same trade entity placements make.
    tile: str | None = Field(
        None, description="filename of an existing tile to use instead")

    @field_validator("color")
    @classmethod
    def _hex(cls, v):
        try:
            map_geometry.parse_color(v)
        except ValueError as e:
            raise ValueError(str(e))
        return v.lower()


class Scatter(BaseModel):
    """One rule for populating layers 3 and 4.

    Terrain is named, not indexed: ids are positional and a caller should not
    have to count the terrain list to say "on grass".
    """
    layer: str = Field(..., pattern="^(props|creatures)$")
    terrain: list[str] = Field(..., min_length=1,
                               description="terrain NAMES this may stand on")
    # Library first, exactly as ADR 0007 D6 has it: an `asset` costs no GPU.
    # `want` is the gap - it queues generation and stands in with a placeholder
    # until it resolves.
    asset: str | None = Field(None, description="existing asset to place")
    want: str | None = Field(None, description="what to generate if no asset")
    density: float = Field(0.1, ge=0.0, le=1.0,
                           description="fraction of matching tiles to occupy")
    spacing: int = Field(2, ge=1, le=16,
                         description="minimum tiles between two of these")


class MapSpec(BaseModel):
    # The facade key. Maps are authored here and collected by name, because
    # prompt-keying is fragile and id-keying means copying UUIDs by hand.
    name: str = Field(..., min_length=1, max_length=60)
    terrains: list[Terrain] = Field(..., min_length=2, max_length=MAX_TERRAINS)
    size: int = Field(DEFAULT_SIZE, ge=MIN_SIZE, le=MAX_SIZE)
    prompt: str | None = Field(
        None, description="the biome brief the painting is generated from")
    # Use an uploaded map reference as the painting instead of generating one.
    # This is what makes a map buildable before the map adapter is trained.
    painting_from: str | None = Field(
        None, description="reference id to use as the biome painting")
    # Layer 2. Axis-aligned segments in GRID coordinates, painted over the
    # terrain in the same pass - a road is ground, not something standing on
    # it, so it must never be able to draw over a creature.
    roads: list[list[list[int]]] = Field(
        default_factory=list,
        description="[[[x0,y0],[x1,y1]], ...] axis-aligned road segments")
    road_tile: str | None = Field(
        None, description="filename of an existing tile to use as road surface")

    # Layers 3 and 4. Separate rules in the SPEC, one depth-sorted pass when
    # drawn - see map_geometry.LAYERS for why those are not the same thing.
    scatter: list[Scatter] = Field(default_factory=list)

    tile_w: int = Field(64, ge=8, le=256)
    style_profile: str | None = None
    colors: int = Field(16, ge=2, le=64)
    seed: int = Field(0)
    llm_name: str | None = None

    @field_validator("terrains")
    @classmethod
    def _separable(cls, v):
        # Raised here rather than in the worker: two terrains that cannot be
        # told apart produce a map silently missing one of them, and finding
        # that out after minutes of GPU is the expensive way to learn it.
        map_geometry.validate_terrains([t.model_dump() for t in v])
        return v


def _db():
    return psycopg2.connect(DB_URL)


@router.post("/api/maps", status_code=202)
def create_map(spec: MapSpec, authorization: str | None = Header(None)):
    """Queue a world map. Returns a job id pollable at /api/jobs/{id}."""
    auth.require(authorization, "generate")

    if not spec.prompt and not spec.painting_from:
        raise HTTPException(
            status_code=400,
            detail="give either a prompt to paint the biome layout from, or "
                   "painting_from to use an uploaded map reference as the "
                   "layout")

    ungenerable = [t.name for t in spec.terrains if not t.tile and not t.prompt]
    if ungenerable:
        raise HTTPException(
            status_code=400,
            detail=f"terrain(s) {', '.join(ungenerable)} have neither a tile "
                   f"to reuse nor a prompt to generate one from")

    # Terrain names are resolved here, not in the worker: a typo should be a
    # 400 naming the valid options, not a scatter rule that silently matches
    # nothing and produces an empty layer.
    known = {t.name for t in spec.terrains}
    for i, s in enumerate(spec.scatter):
        unknown = [t for t in s.terrain if t not in known]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"scatter[{i}] names terrain {', '.join(unknown)}, "
                       f"which this map does not have "
                       f"({', '.join(sorted(known))})")
        if not s.asset and not s.want:
            raise HTTPException(
                status_code=400,
                detail=f"scatter[{i}] has neither an asset to place nor a "
                       f"`want` to generate one from")

    if spec.roads and not spec.road_tile:
        raise HTTPException(
            status_code=400,
            detail="roads were given but no road_tile to draw them with")

    # Validated here rather than in the worker: a diagonal or out-of-bounds
    # segment is a caller mistake, and finding it after the terrain has been
    # painted wastes the expensive half of the job.
    for seg in spec.roads:
        if len(seg) != 2 or any(len(p) != 2 for p in seg):
            raise HTTPException(
                status_code=400,
                detail=f"road segment {seg} must be [[x0,y0],[x1,y1]]")
        (x0, y0), (x1, y1) = seg
        if x0 != x1 and y0 != y1:
            raise HTTPException(
                status_code=400,
                detail=f"road segment {seg} is diagonal; roads are axis-aligned")
        if not all(0 <= v < spec.size for v in (x0, y0, x1, y1)):
            raise HTTPException(
                status_code=400,
                detail=f"road segment {seg} leaves the {spec.size}x{spec.size} grid")

    ratio = None
    if spec.style_profile:
        with _db() as conn, conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT projection_ratio FROM style_profiles "
                        "WHERE name = %s", (spec.style_profile,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(
                    status_code=404,
                    detail=f"no style profile {spec.style_profile!r}")
            ratio = row["projection_ratio"]

    measured = ratio is not None
    if ratio is None:
        ratio = tile_geometry.DEFAULT_RATIO

    # Same sizing the tile stage uses, including the even-height rule - an odd
    # height puts the rhombus's side vertices half a pixel off centre and the
    # tiles stop tessellating.
    tile_w, tile_h = tile_geometry.tile_size_for(ratio, spec.tile_w)

    pic_w, pic_h = map_geometry.picture_size(spec.size, spec.size,
                                             tile_w, tile_h)

    job_id = uuid.uuid4()
    payload = {**spec.model_dump(), "kind": "map", "ratio": ratio,
               "tile_w": tile_w, "tile_h": tile_h,
               "picture_w": pic_w, "picture_h": pic_h}

    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO jobs (id, kind, status, spec, progress_msg) "
            "VALUES (%s, 'map', 'queued', %s, %s)",
            (str(job_id), json.dumps(payload),
             f"queued, {spec.size}x{spec.size} map"))

    from tasks import celery_app
    async_result = celery_app.send_task("maps.build_map_job",
                                        args=[str(job_id)])
    with _db() as conn, conn.cursor() as cur:
        cur.execute("UPDATE jobs SET celery_task_id = %s WHERE id = %s",
                    (async_result.id, str(job_id)))

    reused = sum(1 for t in spec.terrains if t.tile)
    return {
        "job_id": str(job_id),
        "status": "queued",
        "name": spec.name,
        "grid": {"w": spec.size, "h": spec.size},
        "tile": {"w": tile_w, "h": tile_h, "ratio": round(ratio, 3)},
        "picture": {"w": pic_w, "h": pic_h},
        # Say how much of this is actually going to cost GPU time. A map whose
        # terrains all name existing tiles is seconds; one that generates four
        # is minutes.
        "tiles": {"reused": reused,
                  "to_generate": len(spec.terrains) - reused},
        "projection": ("measured from your reference tiles" if measured else
                       f"ASSUMED {ratio}:1 - upload a ground tile to the "
                       f"reference-tile tab to measure it instead"),
        "poll": f"/api/jobs/{job_id}",
        "map": f"/api/maps/{job_id}",
    }


@router.get("/api/maps/{job_id}")
def get_map(job_id: str, authorization: str | None = Header(None)):
    """The tilemap and its placements. 409 until the terrain is final.

    409 rather than 404 for the same reason the sheet route uses it: the map
    exists, it is simply not finished, and a client that conflates those
    abandons live jobs.
    """
    auth.require(authorization, "read")

    with _db() as conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT status, atlas_path, error FROM jobs "
                    "WHERE id = %s::uuid AND kind = 'map'", (job_id,))
        row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="no such map")
    if row["status"] == "failed":
        raise HTTPException(status_code=409,
                            detail=f"map failed: {row['error']}")
    if not row["atlas_path"] or not os.path.exists(row["atlas_path"]):
        raise HTTPException(
            status_code=409,
            detail=f"map is {row['status']}, not ready yet - poll "
                   f"/api/jobs/{job_id}")

    with open(row["atlas_path"], "r", encoding="utf-8") as fh:
        return json.load(fh)


@router.get("/api/maps")
def list_maps(limit: int = 50, authorization: str | None = Header(None)):
    """Named maps, newest first. The facade resolves a name through this."""
    auth.require(authorization, "read")

    with _db() as conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT id, status, spec->>'name' AS name, spec, sheet_path, "
            "       atlas_path, created_at "
            "FROM jobs WHERE kind = 'map' AND deleted = false "
            "ORDER BY created_at DESC LIMIT %s", (limit,))
        rows = cur.fetchall()

    items = []
    for r in rows:
        spec = r["spec"] or {}
        items.append({
            "job_id": str(r["id"]),
            "name": r["name"],
            "status": r["status"],
            "size": spec.get("size"),
            "terrains": [t["name"] for t in spec.get("terrains", [])],
            "picture_url": (f"/api/jobs/{r['id']}/sheet"
                            if r["sheet_path"] else None),
            "map_url": f"/api/maps/{r['id']}" if r["atlas_path"] else None,
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        })
    return {"items": items, "total": len(items)}
