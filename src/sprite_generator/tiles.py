"""Isometric ground tiles: geometry, and the queue endpoint that makes them.

A tile is not a small sprite. A sprite is a subject with a silhouette; a tile is
a SHAPE - a rhombus that has to tessellate with its neighbours exactly, or the
ground shows seams. So the model never decides the outline: it paints texture,
and the rhombus is applied afterwards as a mask, at the exact ratio the world
uses.

That ratio comes from `measure.py` when a style profile exists (a reference tile
IS a readout of the projection), and defaults to the classic 2:1 dimetric
otherwise. Getting it from the profile is the whole reason the reference-tile
tab exists.

The endpoint lives here rather than in jobs.py because a tile spec has nothing
in common with a sheet spec - no actions, no directions, no frames. What they
DO share is the queue and the polling contract: a tile job is a row in `jobs`
with `kind='tile'`, so something2 polls it through exactly the same
`GET /api/jobs/{id}` it already uses. One polling contract, several job shapes.
"""

from __future__ import annotations

import json
import logging
import os
import uuid

import psycopg2
import psycopg2.extras
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import auth
import tile_geometry

logger = logging.getLogger(__name__)
router = APIRouter()

DB_URL = os.environ.get("DB_URL")
IMAGES_DIR = "/app/images"

# The geometry lives in tile_geometry so the worker can import it without
# pulling a FastAPI router into the Celery process. Re-exported here because
# this module is the one callers name.
DEFAULT_RATIO = tile_geometry.DEFAULT_RATIO
DEFAULT_TILE_W = tile_geometry.DEFAULT_TILE_W
diamond_mask = tile_geometry.diamond_mask
cut_tile = tile_geometry.cut_tile
tile_size_for = tile_geometry.tile_size_for


class TileSpec(BaseModel):
    prompt: str = Field(..., min_length=1,
                        description="what the ground is: grass, stone, sand")
    # The handle the facade addresses this tile by. Optional, because the Tiles
    # tab makes one-off tiles nobody looks up again; required in practice for
    # anything something2 will ask for, since a NAME is the only thing it can
    # carry - see `resolve_name`.
    name: str | None = Field(
        None, max_length=120,
        description="address this tile by name later: 'road_sand'")
    style_profile: str | None = Field(
        None, description="take the projection ratio and palette from here")
    tile_w: int = Field(DEFAULT_TILE_W, ge=8, le=512)
    ratio: float | None = Field(
        None, description="width:height; overrides the profile. 2.0 = 2:1")
    colors: int = Field(16, ge=2, le=64)
    seed: int = Field(0)
    llm_name: str | None = None


def _db():
    return psycopg2.connect(DB_URL)


@router.post("/api/tiles", status_code=202)
def create_tile(spec: TileSpec, authorization: str | None = Header(None)):
    """Queue a ground tile. Returns a job id pollable at /api/jobs/{id}."""
    auth.require(authorization, "generate")
    return queue_tile(spec)


def queue_tile(spec: TileSpec) -> dict:
    """Queue a tile build and return its envelope. NO auth check - callers do it.

    Split out of `create_tile` so the A1111 facade queues a tile through exactly
    this path rather than a parallel one. Two ways to start the same job is how
    the two drift, and a tile built down a second path would differ in the one
    thing tiles cannot differ in: its geometry.
    """
    ratio = spec.ratio
    profile_name = spec.style_profile
    if ratio is None and profile_name:
        with _db() as conn, conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT projection_ratio FROM style_profiles "
                        "WHERE name = %s", (profile_name,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404,
                                    detail=f"no style profile {profile_name!r}")
            ratio = row["projection_ratio"]

    measured = ratio is not None
    if ratio is None:
        ratio = DEFAULT_RATIO

    w, h = tile_size_for(ratio, spec.tile_w)

    job_id = uuid.uuid4()
    payload = {**spec.model_dump(), "ratio": ratio, "tile_h": h, "tile_w": w,
               "kind": "tile"}
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO jobs (id, kind, status, spec, progress_msg) "
            "VALUES (%s, 'tile', 'queued', %s, %s)",
            (str(job_id), json.dumps(payload), f"queued, {w}x{h} tile"))

    from tasks import build_tile_job
    async_result = build_tile_job.delay(str(job_id))
    with _db() as conn, conn.cursor() as cur:
        cur.execute("UPDATE jobs SET celery_task_id = %s WHERE id = %s",
                    (async_result.id, str(job_id)))

    return {
        "job_id": str(job_id),
        # The facade blocks on this. `job_id` is our row; the Celery id is the
        # only handle that can be waited on or revoked.
        "celery_task_id": async_result.id,
        "status": "queued",
        "tile": {"w": w, "h": h, "ratio": round(ratio, 3)},
        # Say whether the projection was measured or assumed. A tile at the
        # wrong angle looks fine alone and wrong the moment it is tiled, so the
        # caller should know which they are getting.
        "projection": ("measured from your reference tiles"
                       if measured else
                       f"ASSUMED {DEFAULT_RATIO}:1 - upload a ground tile to "
                       f"the reference-tile tab to measure it instead"),
        "poll": f"/api/jobs/{job_id}",
    }


def resolve_name(name: str) -> dict | None:
    """The newest FINISHED tile with this name, or None.

    Mirrors `maps.resolve_name`, and for the same reason: something2 knows tile
    NAMES - its `tile_types` rows are `road_sand`, `road_snow`, `road_ash` - not
    job UUIDs, and asking an admin to paste a UUID out of this machine's
    database into their form is a design that leaks.

    Names are deliberately not unique. Re-rolling a tile makes a second job with
    the same name and the newest finished one wins, so a better result takes
    effect without anyone touching configuration on the other machine.
    """
    with _db() as conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT id, sheet_path, spec, finished_at "
            "FROM jobs "
            "WHERE kind = 'tile' AND deleted = false AND status = 'done' "
            "  AND lower(spec->>'name') = lower(%s) "
            "ORDER BY COALESCE(finished_at, updated_at) DESC LIMIT 1",
            (name.strip(),))
        row = cur.fetchone()

    # A row whose PNG has been swept off disk is not a usable tile. Returning it
    # anyway would make the facade answer 200 and then fail on the file read,
    # which is the same bug as `resolve_name` in maps guards against.
    if not row or not row["sheet_path"] or not os.path.exists(row["sheet_path"]):
        return None
    return dict(row)


@router.get("/api/tiles")
def list_tiles(authorization: str | None = Header(None)):
    """Every NAMED tile this service can serve, newest first.

    Exists so the facade is operable. something2 addresses a tile by name; with
    no listing, the only way to discover whether a name resolves is to ask for
    it and read the 404. That is a debugging session per typo.

    Unnamed tiles are omitted rather than shown with a blank: they cannot be
    addressed, so listing them here would only invite someone to try.
    """
    auth.require(authorization, "read")
    with _db() as conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT DISTINCT ON (lower(spec->>'name')) "
            "       spec->>'name' AS name, id, spec, finished_at "
            "FROM jobs "
            "WHERE kind = 'tile' AND deleted = false AND status = 'done' "
            "  AND COALESCE(spec->>'name', '') <> '' "
            "ORDER BY lower(spec->>'name'), "
            "         COALESCE(finished_at, updated_at) DESC")
        rows = cur.fetchall()

    out = []
    for r in rows:
        spec = r["spec"] or {}
        out.append({
            "name": r["name"],
            "job_id": str(r["id"]),
            "prompt": spec.get("prompt"),
            "tile": {"w": spec.get("tile_w"), "h": spec.get("tile_h"),
                     "ratio": spec.get("ratio")},
            "colors": spec.get("colors"),
            "style_profile": spec.get("style_profile"),
            "finished_at": (r["finished_at"].isoformat()
                            if r["finished_at"] else None),
        })
    out.sort(key=lambda t: t["finished_at"] or "", reverse=True)
    return {"tiles": out, "count": len(out)}


@router.get("/api/tiles/by-name/{name}")
def get_tile_by_name(name: str, authorization: str | None = Header(None)):
    """A finished tile PNG, by name. A cache READ - it never builds.

    The blocking, build-if-missing behaviour lives on the A1111 facade instead,
    because that is the surface something2 actually calls. This route is the
    plain way to check what that facade would serve, without going through a
    provider registration to find out.
    """
    auth.require(authorization, "read")

    row = resolve_name(name)
    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"no finished tile named {name!r}. GET /api/tiles lists what "
                   f"exists; POST /api/tiles builds one.")
    return FileResponse(row["sheet_path"], media_type="image/png")
