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
