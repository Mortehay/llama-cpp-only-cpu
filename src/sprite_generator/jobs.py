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

    GET    /api/jobs/{job_id}/frames               -> the per-frame PNG index
    GET    /api/jobs/{job_id}/frames/{row}/{n}.png -> one finished cell
    POST   /api/jobs/{job_id}/redraw               -> 202, a child job id

A sheet is a folder of per-frame images and sheet.png is assembled from them,
so one cell can be re-rendered without touching the other 335. The redraw
endpoint returns a NEW job id to poll through the same GET above rather than
inventing a second status shape, and it rewrites the PARENT's sheet in place -
a caller holding /api/jobs/{parent}/sheet keeps working.

Every addition here is additive. The POST /api/jobs body, the job status
fields, the /sheet PNG layout and the existing /atlas fields are unchanged;
the atlas gains `quality`, `frames` and, on a flipped row, `mirrored_from`.
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
import auth

logger = logging.getLogger(__name__)
router = APIRouter()

DB_URL = os.environ.get("DB_URL")
API_TOKEN = os.environ.get("SPRITE_API_TOKEN", "").strip()
IMAGES_DIR = "/app/images"

# Terminal states. A client that sees one of these can stop polling.
TERMINAL = {"done", "failed", "cancelled"}


def _db():
    return psycopg2.connect(DB_URL)


def _require_auth(authorization: str | None, scope: str = "read"):
    """Delegates to the shared key system.

    This used to be a local check against one env var that began
    `if not API_TOKEN: return`. Leaving it in place after migration 013 would
    have been worse than never adding keys at all: creating a key would secure
    /api/assets and /api/references while /api/jobs - the endpoint that spends
    GPU hours - stayed open, and the UI would have said "API secured".

    The legacy token still works; `auth` honours it. See auth.py.
    """
    return auth.require(authorization, scope)


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

    # Name of a style profile derived from reference art. When set, its
    # MEASURED constraints win over the defaults above - most importantly the
    # camera elevation, which is otherwise an unverified guess.
    #
    # Opt-in by name rather than "use the newest profile": a job whose output
    # depends on which profile happened to be derived last is not reproducible.
    style_profile: str | None = None

    # Declared because `build_sheet_job` reads `spec.get("concept_check", True)`
    # and tells the caller, in the failure message, to "Pass concept_check=false
    # to build anyway". Pydantic drops undeclared fields silently and
    # `model_dump()` is what gets stored, so that flag never reached the task:
    # a caller followed the instruction, sent it, and got the identical refusal.
    # The escape hatch for a wide creature or a vehicle did not exist.
    #
    # This is the pairing to watch for anywhere - a field the model does not
    # declare, read downstream with `.get(name, default)`. The drop is silent,
    # the default is plausible, and nothing between them can tell that the
    # caller ever spoke. `tiles.py` shows the other correct shape: it augments
    # the dump explicitly, `{**spec.model_dump(), "tile_h": h}`, for values the
    # server computes rather than the caller supplies.
    concept_check: bool = True

    # Run the redraw rounds. The CHECKS always run - they are a by-product of
    # the cutout that every cell needs anyway, and the atlas reports them
    # either way - so this only controls whether a failing frame is
    # re-rendered. Declared here rather than read with `.get` downstream for
    # the reason `concept_check` records: Pydantic drops an undeclared field
    # silently and the caller cannot tell it was ignored.
    quality_checks: bool = True

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
    _require_auth(authorization, "generate")

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


class RedrawFrame(BaseModel):
    action: str
    direction: str
    frame: int


class RedrawSpec(BaseModel):
    """Which cells to re-render. Everything else on the sheet is left alone."""

    frames: list[RedrawFrame] = Field(default_factory=list)

    @field_validator("frames")
    @classmethod
    def _not_empty(cls, v):
        # An empty list is refused rather than treated as "all": a redraw is
        # GPU time, and "redraw nothing" and "redraw everything" are too far
        # apart to guess between.
        if not v:
            raise ValueError("name at least one frame to redraw")
        if len(v) > 64:
            raise ValueError(
                f"{len(v)} frames is more than a redraw is for. Past about a "
                f"row and a half it is cheaper to rebuild the sheet.")
        return v


def _work_dir(job_id: str) -> str:
    """The job's work directory, for the routes that read files from it.

    The id is parsed as a UUID first, and that is not decoration. Every other
    route puts `job_id` into a parameterised SQL query, where a strange value
    is simply not found; these are the first to put it into a FILESYSTEM
    PATH, and `_job_../../etc` normalises to `/etc`. A job id is a uuid4 in
    every code path that creates one, so parsing it costs nothing and closes
    the traversal.
    """
    try:
        return os.path.join(IMAGES_DIR, f"_job_{uuid.UUID(job_id)}")
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=400, detail="bad job id")


@router.get("/api/jobs/{job_id}/frames")
def list_frames(job_id: str, authorization: str | None = Header(None)):
    """The per-frame PNGs that this sheet was assembled from.

    A sheet IS a folder of frames; the grid is the packaging. Exposed so a
    consumer can fetch one cell without slicing, and so an admin UI can show a
    frame beside its quality verdict and ask for that one to be redrawn.
    """
    _require_auth(authorization)
    root = os.path.join(_work_dir(job_id), "frames")
    if not os.path.isdir(root):
        raise HTTPException(
            status_code=404,
            detail="no per-frame output for this job; it was built before "
                   "frames were written, or its work directory is gone")
    rows = {}
    for row in sorted(os.listdir(root)):
        d = os.path.join(root, row)
        if not os.path.isdir(d):
            continue
        names = sorted((f for f in os.listdir(d) if f.endswith(".png")),
                       key=lambda f: int(os.path.splitext(f)[0] or 0))
        rows[row] = [f"/api/jobs/{job_id}/frames/{row}/{f}" for f in names]
    return {"job_id": job_id, "rows": rows,
            "count": sum(len(v) for v in rows.values())}


@router.get("/api/jobs/{job_id}/frames/{row}/{name}")
def get_frame(job_id: str, row: str, name: str,
              authorization: str | None = Header(None)):
    """One finished cell as its own PNG."""
    _require_auth(authorization)
    # Both components are joined into a filesystem path, so they are checked
    # rather than trusted. `..` in either would otherwise read any file the
    # API process can see.
    for part in (row, name):
        if not part or "/" in part or "\\" in part or part.startswith("."):
            raise HTTPException(status_code=400, detail="bad frame path")
    if not name.endswith(".png"):
        raise HTTPException(status_code=400, detail="frames are .png")
    path = os.path.join(_work_dir(job_id), "frames", row, name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="no such frame")
    return FileResponse(path, media_type="image/png")


@router.post("/api/jobs/{job_id}/redraw", status_code=202)
def redraw_frames(job_id: str, spec: RedrawSpec,
                  authorization: str | None = Header(None)):
    """Re-render named frames of a finished sheet.

    Returns a NEW job id to poll with the usual GET /api/jobs/{id}, rather
    than inventing a second status shape. The parent's sheet and atlas URLs
    are unchanged and are rewritten in place when the redraw finishes, so a
    caller holding `/api/jobs/{parent}/sheet` keeps working.
    """
    _require_auth(authorization, "generate")

    # Parsed before the query, so a malformed id is a 400 rather than a
    # psycopg2 DataError surfacing as a 500.
    work = _work_dir(job_id)

    with _db() as conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT status, spec FROM jobs WHERE id = %s", (job_id,))
        parent = cur.fetchone()
    if not parent:
        raise HTTPException(status_code=404, detail="no such job")
    if parent["status"] != "done":
        raise HTTPException(
            status_code=409,
            detail=f"job is {parent['status']}, not done; there is nothing "
                   f"to redraw yet")
    if not os.path.isdir(work):
        raise HTTPException(
            status_code=410,
            detail="this job's cells are no longer on disk, so a single "
                   "frame cannot be re-rendered; rebuild the sheet")

    pspec = parent["spec"] if isinstance(parent["spec"], dict) else {}
    known = set(pspec.get("actions") or [])
    bad = [f"{f.action}/{f.direction}/{f.frame}" for f in spec.frames
           if (known and f.action not in known)
           or f.direction not in (pspec.get("directions") or [])
           or f.frame < 0]
    if bad:
        raise HTTPException(
            status_code=400,
            detail=f"not in this sheet: {', '.join(bad[:6])}. It has "
                   f"actions {sorted(known)} and directions "
                   f"{pspec.get('directions')}.")

    child = uuid.uuid4()
    cspec = {
        "parent_job": job_id,
        "frames": [f.model_dump() for f in spec.frames],
        "cell": pspec.get("cell", "48x64"),
        "colors": pspec.get("colors", 24),
        "kind": "redraw",
    }
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO jobs (id, status, spec, progress_msg) "
            "VALUES (%s, 'queued', %s, %s)",
            (str(child), json.dumps(cspec),
             f"queued, redraw of {len(spec.frames)} frame(s)"),
        )

    from tasks import redraw_frames_job

    async_result = redraw_frames_job.delay(str(child))
    with _db() as conn, conn.cursor() as cur:
        cur.execute("UPDATE jobs SET celery_task_id = %s WHERE id = %s",
                    (async_result.id, str(child)))

    logger.info("redraw %s queued for job %s: %d frame(s)", child, job_id,
                len(spec.frames))
    return {
        "job_id": str(child),
        "parent_job": job_id,
        "status": "queued",
        "frames": len(spec.frames),
        # One encode pass and one denoise pass over the named frames, plus a
        # compose. The model loads dominate for a small redraw.
        "estimated_seconds": len(spec.frames) * 33 + 200,
        "poll": f"/api/jobs/{child}",
        "sheet_url": f"/api/jobs/{job_id}/sheet",
    }


@router.delete("/api/jobs/{job_id}")
def cancel_job(job_id: str, authorization: str | None = Header(None)):
    """Ask a job to stop.

    Cooperative, not immediate: a stage in the middle of a denoise finishes its
    current cell first. Already-terminal jobs are left alone rather than
    reported as an error - cancelling something that already finished is a
    race, not a mistake.
    """
    _require_auth(authorization, "generate")
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


def _sheet_engine() -> dict:
    """Which model the conveyor will run, for the UI to name it.

    The sheet tab has no model dropdown and should not get one - the editor is
    fixed by the task, not chosen per request (model_gateway.QWEN_EDIT, ADR
    0005). The cost of that was a tab that said nothing at all about what it
    was about to spend an hour of GPU time on: the only mention lived in an
    HTML comment, where no user has ever read it.

    Served rather than written into the page for the same reason the core
    roster is: every field here is an env override (QWEN_EDIT_GGUF_FILE,
    QWEN_LIGHTNING_STEPS and friends), so text hardcoded in the tab would be
    wrong on the first machine that sets one.

    `qwen_edit` is imported HERE, not at module scope, on purpose. This module
    is otherwise stdlib-only so the API process can consult the pose library
    without the torch import chain - and qwen_edit imports torch. That is
    already paid for in practice (main.py imports tasks.py, which imports torch
    and calls torch.cuda.is_available() at module scope in this same process),
    and a failure here degrades to "unknown" instead of taking the sheet tab
    down. If this module ever has to stay genuinely torch-free, move those
    constants to a dependency-free module and import them from both sides - do
    NOT copy their defaults here, which is the drift core_models.py exists to
    prevent.
    """
    try:
        import core_models
        import qwen_edit as qe
    except Exception as e:
        logger.warning("could not read the sheet engine config: %s", e)
        return {"fixed": True, "label": "Qwen-Image-Edit (GGUF)",
                "unreadable": "could not be read in this process"}

    # Repo-level, which is all core_models.is_cached can answer: a cached repo
    # whose snapshot lacks THIS quant still reports present. So the result is
    # reported as "missing" when non-empty and claims no readiness when it is
    # empty - a wrong green light is worse than none.
    repos = [qe.REPO, qe.CONFIG_REPO, qe.GGUF_REPO,
             qe.ANGLES_LORA, qe.LIGHTNING_LORA]
    return {
        # Fixed by the task, so there is nothing for the UI to offer.
        "fixed": True,
        "label": "Qwen-Image-Edit-2511 (GGUF)",
        "transformer": qe.GGUF_FILE,
        "transformer_repo": qe.GGUF_REPO,
        "encoder_repo": qe.REPO,
        "steps": qe.LIGHTNING_STEPS,
        "loras": [
            {"repo": qe.ANGLES_LORA, "weight": qe.ANGLES_WEIGHT,
             "role": "8 azimuths - the camera, and what pins framing"},
            {"repo": qe.LIGHTNING_LORA, "weight": qe.LIGHTNING_WEIGHT,
             "role": (f"step distillation - {qe.LIGHTNING_STEPS} steps "
                      f"at scale {qe.LIGHTNING_SCALE}")},
        ],
        "missing": [r for r in repos if not core_models.is_cached(r)],
        # Step 1 IS a choice, and a different model. Named so this tab cannot
        # be read as "the model the whole app uses".
        "concept_note": ("The concept image comes from the Entity tab, which "
                         "has its own model roster."),
    }


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
        # The model the sheet lane runs. Not a choice, which is exactly why
        # it has to be stated somewhere a user can see it.
        "engine": _sheet_engine(),
    }


@router.get("/api/jobs-health")
def jobs_health():
    """Queue depth, so an operator can see a backlog without reading logs."""
    with _db() as conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT status, count(*) AS n FROM jobs GROUP BY status")
        counts = {r["status"]: r["n"] for r in cur.fetchall()}
    return JSONResponse({"counts": counts})
