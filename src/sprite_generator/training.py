"""Queue and inspect LoRA training runs.

Training goes through the same queue as generation because there is one GPU
with 12 GB on it. A training run started while a sheet build is denoising would
not share politely - it would OOM one of them, most likely the one that had
already spent forty minutes. Celery runs one task at a time on this worker, so
enqueuing IS the lease.

The endpoints deliberately refuse more than they accept:

  * fewer than `min_images` usable references -> 400 naming the count. A LoRA
    trained on three images memorises them, and the failure appears hours later
    as output that ignores the prompt.
  * a run already queued or running -> 409. Two concurrent runs would fight for
    the card and both lose.
"""

from __future__ import annotations

import logging
import os
import uuid

import psycopg2
import psycopg2.extras
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

import auth

logger = logging.getLogger(__name__)
router = APIRouter()

DB_URL = os.environ.get("DB_URL")
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

# Below this, a style LoRA memorises rather than generalises. Measurement needs
# three examples; training needs an order of magnitude more, and saying so up
# front is cheaper than discovering it after an hour of GPU time.
MIN_IMAGES = 8


def _db():
    return psycopg2.connect(DB_URL)


class TrainRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=60,
                      description="adapter name; also the filename")
    profile: str | None = Field(None, description="style profile to attach to")
    steps: int = Field(1000, ge=50, le=20000)
    rank: int = Field(32, ge=4, le=128)
    lr: float = Field(1e-4, gt=0, le=1e-2)
    resolution: int = Field(1024, ge=256, le=1536)
    trigger: str | None = None
    # Which references to train on. Sprites and cores by default; tiles are
    # excluded because a tile teaches the model to draw tiles.
    pattern: str = "ref_sprite_*.png,ref_core_*.png"


def _usable_reference_count(pattern: str) -> int:
    """How many usable references the pattern would actually match.

    Counted from the DB rather than the filesystem so that an example marked
    unusable - a 200-colour render, an anti-aliased sprite - is not silently
    trained on.
    """
    kinds = []
    if "ref_sprite_" in pattern:
        kinds.append("sprite")
    if "ref_core_" in pattern:
        kinds.append("core")
    if "ref_tile_" in pattern:
        kinds.append("tile")
    if not kinds:
        return 0
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM reference_assets "
            "WHERE deleted = false AND usable = true AND kind = ANY(%s)",
            (kinds,))
        return int(cur.fetchone()[0])


@router.post("/api/training", status_code=202)
def start_training(body: TrainRequest, authorization: str | None = Header(None)):
    """Queue a training run. Returns immediately with a run id."""
    auth.require(authorization, "generate")

    with _db() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM training_runs "
                    "WHERE status IN ('queued', 'running')")
        if cur.fetchone()[0]:
            raise HTTPException(
                status_code=409,
                detail="a training run is already queued or running - there is "
                       "one GPU, so they cannot overlap")

    usable = _usable_reference_count(body.pattern)
    if usable < MIN_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"only {usable} usable reference(s) match {body.pattern}; "
                   f"need at least {MIN_IMAGES}. Measurement works from three "
                   f"examples, training does not - a LoRA trained on a handful "
                   f"memorises them.")

    profile_id = None
    if body.profile:
        with _db() as conn, conn.cursor() as cur:
            cur.execute("SELECT id FROM style_profiles WHERE name = %s",
                        (body.profile,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404,
                                    detail=f"no style profile {body.profile!r}")
            profile_id = row[0]

    run_id = uuid.uuid4()
    config = {"name": body.name, "steps": body.steps, "rank": body.rank,
              "lr": body.lr, "resolution": body.resolution,
              "pattern": body.pattern, "trigger": body.trigger,
              "min_images": MIN_IMAGES}

    import json
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO training_runs (id, profile_id, base_model, config, "
            "                           dataset_size, steps_total) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (str(run_id), profile_id, BASE_MODEL, json.dumps(config),
             usable, body.steps))

    # Imported here, not at module scope: tasks.py pulls in torch, and the web
    # process must never do that.
    from tasks import train_lora_job
    async_result = train_lora_job.delay(str(run_id))

    with _db() as conn, conn.cursor() as cur:
        cur.execute("UPDATE training_runs SET config = config || %s::jsonb "
                    "WHERE id = %s::uuid",
                    (json.dumps({"celery_task_id": async_result.id}),
                     str(run_id)))

    return {"run_id": str(run_id), "status": "queued",
            "dataset_size": usable, "steps": body.steps,
            "trigger": body.trigger or f"<{body.name}-style>"}


@router.get("/api/training")
def list_runs(limit: int = 25, authorization: str | None = Header(None)):
    auth.require(authorization, "read")
    with _db() as conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT id, base_model, config, dataset_size, status, steps_done, "
            "       steps_total, loss, output_path, error, created_at, "
            "       started_at, finished_at "
            "FROM training_runs ORDER BY created_at DESC LIMIT %s", (limit,))
        out = []
        for r in cur.fetchall():
            d = dict(r)
            d["id"] = str(d["id"])
            for f in ("created_at", "started_at", "finished_at"):
                d[f] = d[f].isoformat() if d[f] else None
            out.append(d)
    return {"items": out, "total": len(out), "min_images": MIN_IMAGES}


@router.get("/api/training/readiness")
def readiness(authorization: str | None = Header(None)):
    """Can training start, and if not, what is missing?

    Exists so the UI can disable the button with a reason rather than offering
    an action that will 400.
    """
    auth.require(authorization, "read")
    usable = _usable_reference_count("ref_sprite_*.png,ref_core_*.png")
    with _db() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM training_runs "
                    "WHERE status IN ('queued', 'running')")
        busy = int(cur.fetchone()[0]) > 0
    reasons = []
    if usable < MIN_IMAGES:
        reasons.append(f"{usable}/{MIN_IMAGES} usable references")
    if busy:
        reasons.append("a run is already queued or running")
    return {"ready": not reasons, "usable_references": usable,
            "min_images": MIN_IMAGES, "busy": busy,
            "why": "; ".join(reasons) or "ready to train"}


@router.delete("/api/training/{run_id}")
def delete_run(run_id: str, authorization: str | None = Header(None)):
    """Forget a run record. Does NOT delete the adapter it produced.

    A trained adapter is the most expensive artefact on this machine - hours of
    GPU time - so removing the bookkeeping never removes the weights.
    """
    auth.require(authorization, "generate")
    with _db() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM training_runs WHERE id = %s::uuid "
                    "AND status NOT IN ('queued', 'running')", (run_id,))
        if cur.rowcount == 0:
            raise HTTPException(
                status_code=404,
                detail="no such finished run (a queued or running one cannot "
                       "be deleted)")
    return {"deleted": run_id}
