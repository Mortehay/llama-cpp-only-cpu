"""Async sheet jobs for something2.

The contract, per the consumer's own description:

    something2 puts a task on a queue and does NOT wait. All it needs back
    immediately is a task number. Later it asks for status - either for one
    task, or for all the tasks it has sent.

This supersedes what `.ai/specs/something2-provider/contract.md` says. That
document was written from their published `docs/ai-providers.md`, which
describes a synchronous txt2img facade and states that submit/poll is
unsupported (their SOMET-334) - **and it carried an explicit caveat that their
actual calling code had never been read.** The caveat was the accurate part.

Why this is not the A1111 facade with a longer timeout: a full character is
~2 hours of GPU time on this hardware (measured, ADR 0005). No HTTP timeout
makes that synchronous. The facade in `a1111.py` stays for single-image
txt2img, which genuinely does fit in one request.

Endpoints:

    POST   /api/jobs              -> 202 {"job_id": ..., "status": "queued"}
    GET    /api/jobs/{job_id}     -> one job
    GET    /api/jobs?ids=a,b,c    -> those jobs        (poll a known set)
    GET    /api/jobs?since=<ts>   -> jobs updated since (poll for changes)
    GET    /api/jobs/{job_id}/sheet  -> the finished PNG
    GET    /api/jobs/{job_id}/atlas  -> the atlas JSON
    DELETE /api/jobs/{job_id}     -> request cancellation
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras
from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

# The pose library. stdlib-only, so the API process can consult it without
# pulling in the Celery/torch import chain that tasks.py brings.
import actions as action_lib

logger = logging.getLogger(__name__)
router = APIRouter()

DB_URL = os.environ.get("DB_URL")
API_TOKEN = os.environ.get("SPRITE_API_TOKEN", "").strip()
IMAGES_DIR = "/app/images"

# Terminal states. A client that sees one of these can stop polling.
TERMINAL = {"done", "failed", "cancelled"}


def _db():
    return psycopg2.connect(DB_URL)


def _require_auth(authorization: str | None):
    if not API_TOKEN:
        return
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


class JobSpec(BaseModel):
    """What to build. Every field has a default so a minimal POST works."""

    prompt: str = Field("", description="concept prompt, if generating a concept")
    concept_image: str | None = Field(
        None, description="path under images/ to use instead of generating one")
    actions: list[str] = Field(default_factory=lambda: ["walk"])
    directions: list[str] = Field(
        default_factory=lambda: ["s", "se", "e", "ne", "n", "nw", "w", "sw"])
    frames: int = 4

    @field_validator("directions")
    @classmethod
    def _empty_means_front(cls, v):
        """An explicitly EMPTY direction list means front only.

        Not the same as omitting the field, which keeps the documented full
        turnaround for existing API consumers. The UI always sends the array,
        so unticking every direction arrives here as [] - and a direction is
        not something a caller should be forced to think about to get a sheet.
        Front is the cheapest possible answer (one row) and the canonical
        facing, so an empty choice costs 8 minutes rather than 24.
        """
        return v if v else ["s"]
    cell: str = "48x64"
    colors: int = 24
    seed: int = 0

    def cells(self) -> int:
        """Cell count, which is what actually predicts runtime."""
        return len(self.actions) * len(self.directions) * self.frames


def _row_to_job(row) -> dict:
    """Shape one DB row into the response body.

    `sheet_url` is present only when the job is done, so a client can treat its
    presence as "ready" without parsing status strings.
    """
    out = {
        "job_id": str(row["id"]),
        "status": row["status"],
        "stage": row["stage"],
        "progress_pct": row["progress_pct"],
        "progress_msg": row["progress_msg"],
        "spec": row["spec"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        "finished_at": row["finished_at"].isoformat() if row["finished_at"] else None,
        "error": row["error"],
    }
    if row["status"] == "done" and row["sheet_path"]:
        out["sheet_url"] = f"/api/jobs/{row['id']}/sheet"
        out["atlas_url"] = f"/api/jobs/{row['id']}/atlas"
    return out


@router.post("/api/jobs", status_code=202)
def create_job(spec: JobSpec, authorization: str | None = Header(None)):
    """Enqueue a sheet build. Returns immediately with an id.

    202 rather than 200 on purpose: nothing has been produced yet, and the
    status code should say so rather than relying on the caller reading the
    body carefully.
    """
    _require_auth(authorization)

    # Validate the spec against the pose library before spending any GPU time.
    #
    # These used to fail deep inside the worker, minutes in and after the
    # turnaround pass had already been paid for: an unknown action as a
    # SystemExit from stage_actions, and too many frames as
    # `KeyError: 'idle|s|4'` halfway through the denoise pass. Both are
    # answerable here, instantly, from the request alone.
    unknown = [a for a in spec.actions if a not in action_lib.ACTIONS]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown action(s): {', '.join(unknown)}. "
                   f"Available: {', '.join(sorted(action_lib.ACTIONS))}")

    limit = action_lib.max_frames(spec.actions, spec.directions)
    if spec.frames > limit:
        raise HTTPException(
            status_code=400,
            detail=f"{spec.frames} frames requested, but the pose library "
                   f"defines only {limit} for {', '.join(spec.actions)}. "
                   f"A frame is a named pose, not an interpolation - asking "
                   f"for more cannot produce more motion. Use {limit} or "
                   f"fewer, or add poses to action_prompts.json.")

    job_id = uuid.uuid4()
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO jobs (id, status, spec, progress_msg) "
            "VALUES (%s, 'queued', %s, %s)",
            (str(job_id), json.dumps(spec.model_dump()),
             f"queued, {spec.cells()} cells"),
        )

    # Imported here, not at module scope: main.py imports this router in the
    # API process, which must not pull in the Celery/torch import chain.
    from tasks import build_sheet_job

    async_result = build_sheet_job.delay(str(job_id))
    with _db() as conn, conn.cursor() as cur:
        cur.execute("UPDATE jobs SET celery_task_id = %s WHERE id = %s",
                    (async_result.id, str(job_id)))

    logger.info("job %s queued: %d cells (%s x %s x %d frames)", job_id,
                spec.cells(), spec.actions, spec.directions, spec.frames)
    return {
        "job_id": str(job_id),
        "status": "queued",
        "cells": spec.cells(),
        # An honest estimate beats a client guessing. ~33s/cell measured, plus
        # roughly 4 model loads at ~90s across the five stages.
        "estimated_seconds": spec.cells() * 33 + 360,
        "poll": f"/api/jobs/{job_id}",
    }


@router.get("/api/jobs")
def list_jobs(ids: str | None = Query(None, description="comma-separated job ids"),
              since: str | None = Query(None, description="ISO timestamp"),
              status: str | None = Query(None),
              limit: int = Query(100, le=1000),
              authorization: str | None = Header(None)):
    """Poll many jobs at once.

    Three ways to ask, matching how a consumer actually polls:
      - `ids=`    the set it submitted and is tracking
      - `since=`  anything that changed since it last asked
      - `status=` everything currently queued/running/done
    """
    _require_auth(authorization)

    where, params = [], []
    if ids:
        wanted = [s.strip() for s in ids.split(",") if s.strip()]
        if not wanted:
            return {"jobs": []}
        try:
            wanted = [str(uuid.UUID(w)) for w in wanted]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"bad job id: {e}")
        # ::uuid[] is required, not decorative. psycopg2 sends a Python list of
        # strings as text[], and Postgres has no uuid = text operator, so the
        # query fails with "operator does not exist: uuid = text" rather than
        # returning nothing.
        where.append("id = ANY(%s::uuid[])")
        params.append(wanted)
    if since:
        try:
            datetime.fromisoformat(since.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400,
                                detail="`since` must be an ISO timestamp")
        where.append("updated_at > %s")
        params.append(since)
    if status:
        where.append("status = %s")
        params.append(status)

    sql = "SELECT * FROM jobs"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY updated_at DESC LIMIT %s"
    params.append(limit)

    with _db() as conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    jobs = [_row_to_job(r) for r in rows]
    # `server_time` so a client can use it as the next `since` without trusting
    # its own clock to agree with the database's.
    return {"jobs": jobs, "count": len(jobs),
            "server_time": datetime.now(timezone.utc).isoformat()}


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str, authorization: str | None = Header(None)):
    _require_auth(authorization)
    with _db() as conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM jobs WHERE id = %s", (job_id,))
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="no such job")
    return _row_to_job(row)


def _finished_file(job_id: str, column: str, media_type: str):
    with _db() as conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT status, %s AS path FROM jobs WHERE id = %%s"
                    % column, (job_id,))
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="no such job")
    if row["status"] != "done":
        # 409 rather than 404: the job exists, it is just not ready. A client
        # that cannot tell those apart will give up on a job that is still
        # running.
        raise HTTPException(status_code=409,
                            detail=f"job is {row['status']}, not done")
    if not row["path"] or not os.path.isfile(row["path"]):
        raise HTTPException(status_code=410,
                            detail="output no longer on disk")
    return FileResponse(row["path"], media_type=media_type)


@router.get("/api/jobs/{job_id}/sheet")
def get_sheet(job_id: str, authorization: str | None = Header(None)):
    _require_auth(authorization)
    return _finished_file(job_id, "sheet_path", "image/png")


@router.get("/api/jobs/{job_id}/atlas")
def get_atlas(job_id: str, authorization: str | None = Header(None)):
    _require_auth(authorization)
    return _finished_file(job_id, "atlas_path", "application/json")


@router.delete("/api/jobs/{job_id}")
def cancel_job(job_id: str, authorization: str | None = Header(None)):
    """Ask a job to stop.

    Cooperative, not immediate: a stage in the middle of a denoise finishes its
    current cell first. Already-terminal jobs are left alone rather than
    reported as an error - cancelling something that already finished is a
    race, not a mistake.
    """
    _require_auth(authorization)
    with _db() as conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT status, celery_task_id FROM jobs WHERE id = %s",
                    (job_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="no such job")
        if row["status"] in TERMINAL:
            return {"job_id": job_id, "status": row["status"],
                    "note": "already finished; nothing to cancel"}
        cur.execute("UPDATE jobs SET status = 'cancelled', "
                    "progress_msg = 'cancelled by request', finished_at = now() "
                    "WHERE id = %s", (job_id,))

    if row["celery_task_id"]:
        from tasks import set_cancel_flag
        set_cancel_flag(row["celery_task_id"])

    return {"job_id": job_id, "status": "cancelled"}


@router.get("/api/action-catalog")
def action_catalog():
    """What the pose library can build, for the UI to constrain its inputs.

    Served rather than hardcoded in the template for the same reason the core
    model roster is: the frames input offered up to 8 while every action
    defined 4 poses, and the only feedback was a job that died mid-render.
    """
    cat = action_lib.catalog()
    return {
        # [{id, label, max_frames}], so the UI renders the checkboxes from the
        # library instead of hardcoding three of them in the template - which
        # is what kept the four newest actions invisible until this existed.
        "actions": cat["actions"],
        "directions": [
            {"id": d, "family": action_lib.family(d)}
            for d in ["s", "se", "e", "ne", "n", "nw", "w", "sw"]
        ],
        "max_frames": cat["max_frames"],
        "frames_by_action": {a["id"]: a["max_frames"] for a in cat["actions"]},
        # Empty directions is a valid request, not an error. Named here so the
        # UI can say what it will build rather than guessing.
        "default_direction": "s",
    }


@router.get("/api/jobs-health")
def jobs_health():
    """Queue depth, so an operator can see a backlog without reading logs."""
    with _db() as conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT status, count(*) AS n FROM jobs GROUP BY status")
        counts = {r["status"]: r["n"] for r in cur.fetchall()}
    return JSONResponse({"counts": counts})
