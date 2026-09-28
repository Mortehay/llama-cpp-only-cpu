"""The audio surface: music and ambience for something2 maps.

Shaped after the TILE facade in `a1111.py` - cache first, then build inside a
budget - with one deliberate difference that is the whole point of
`.ai/specs/audio/contract.md` D5:

    the tile facade REVOKES its task and answers 504 when the budget runs out.
    This one does NOT. The build keeps running, the ledger row stays open, and
    the caller gets 503 + Retry-After so the next request cache-reads what
    this one paid for.

A music build is a model swap plus generation; throwing it away because one
HTTP client got bored is how something2 ends up paying for the same track
three times. `tasks.generate_audio_task` closes its own row precisely so the
row survives the request.

Never a placeholder: not-ready is always non-2xx here. The map facade answers
with a magenta cross when a map is provisional; the audio equivalent would be
silence, and a caller that caches silence as final has a silent map forever.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

import audio_styles
import auth
import generations
from tasks import celery_app

logger = logging.getLogger(__name__)
router = APIRouter()


# Just under what something2 is told to set. Measured 2026-09-12: a warm
# 120 s track is 7 s of model time, but a cold one pays a 23 s load and an
# SDXL eviction first, and the worker may be finishing someone else's job.
AUDIO_GENERATE_TIMEOUT_S = int(os.environ.get("AUDIO_GENERATE_TIMEOUT_S", "240"))

KINDS = ("music", "ambience")


class AudioRequest(BaseModel):
    kind: str = "music"
    name: str = ""
    style: str | None = None
    prompt: str | None = None
    context: str | None = None
    slots: dict | None = None
    seed: int | None = None
    duration_s: float | None = None


def _b64(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def _params(row: dict) -> dict:
    p = row.get("params")
    if isinstance(p, str):
        try:
            return json.loads(p)
        except ValueError:
            return {}
    return p or {}


def _info(row: dict, *, cached: bool, elapsed_ms: float) -> dict:
    params = _params(row)
    return {
        "kind": row["kind"],
        "name": row["name"],
        "style": params.get("style"),
        "prompt": row.get("prompt"),
        "author": params.get("author"),
        "seed": row.get("seed"),
        "duration_s": params.get("duration_s"),
        "sample_rate": params.get("sample_rate"),
        "loop_start": params.get("loop_start"),
        "loop_end": params.get("loop_end"),
        "bars": params.get("bars"),
        "seam_rms_jump_db": params.get("seam_rms_jump_db"),
        "cached": cached,
        "served_from": "cache" if cached else "generated",
        "generation_id": str(row["id"]),
        "duration_ms": round(elapsed_ms),
    }


def _payload(row: dict, *, cached: bool, started: float) -> dict:
    elapsed = (time.time() - started) * 1000
    encoded = _b64(row["file_path"])
    logger.info("audio served %s %r %s in %.0fms (%d b64 chars)",
                row["kind"], row["name"], "from cache" if cached else "fresh",
                elapsed, len(encoded))
    return {"audio": [encoded], "info": _info(row, cached=cached,
                                              elapsed_ms=elapsed)}


def _running_build(kind: str, name: str) -> dict | None:
    """A build for this name already on the worker, or None.

    Because an over-budget request does NOT cancel its build, a second request
    for the same name arrives while the first is still running. Queueing
    another would spend the card twice for one artefact and leave two rows
    claiming the same name, so the second request JOINS the first instead.
    """
    import psycopg2
    import psycopg2.extras
    try:
        with psycopg2.connect(os.environ.get("DB_URL")) as conn, conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, celery_task_id, created_at FROM generations "
                "WHERE kind = %s AND lower(name) = lower(%s) "
                "  AND status = 'running' AND deleted = false "
                "  AND celery_task_id IS NOT NULL "
                "ORDER BY created_at DESC LIMIT 1", (kind, name))
            return cur.fetchone()
    except Exception as e:
        # A lookup that cannot run means "no build in flight": generating a
        # second time is wasteful, refusing a first one is broken.
        logger.warning("audio: in-flight lookup for %r failed: %s", name, e)
        return None


def _long_job_ahead() -> str | None:
    """A non-audio job holding the solo worker, described, or None.

    Same advisory check as the tile facade: the worker runs one job at a time
    and a sheet build is hours, so a caller blocking behind one can only spend
    its whole budget and time out. Advisory, not a lock.
    """
    import psycopg2
    import psycopg2.extras
    try:
        with psycopg2.connect(os.environ.get("DB_URL")) as conn, conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT kind, started_at FROM jobs "
                "WHERE status = 'running' AND deleted = false "
                "ORDER BY started_at LIMIT 1")
            row = cur.fetchone()
    except Exception as e:
        logger.warning("audio: could not check the queue: %s", e)
        return None
    if not row:
        return None
    return f"a {row['kind']} job has been running since {row['started_at']}"


def _require_kind(kind: str) -> str:
    kind = (kind or "").strip().lower()
    if kind not in KINDS:
        raise HTTPException(status_code=422,
                            detail=f"kind must be one of {', '.join(KINDS)}")
    return kind


def _require_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise HTTPException(
            status_code=422,
            detail="name is required: audio is addressed by map name, so a "
                   "caller can ask for the same track twice and get the same "
                   "file rather than a new generation.")
    return name


def _await_build(task_id: str, gen_id: str | None, kind: str, name: str,
                 started: float, *, queued: bool) -> dict:
    """Block on a build, then answer - or hand back a 503 that is not a cancel.

    THE ONE PLACE THIS DIFFERS FROM THE TILE FACADE. `_serve_tile` revokes the
    task and returns 504 when its budget expires. Here the task keeps running,
    keeps its ledger row, and finishes into the cache; the caller is told to
    come back rather than told the work is gone. `generate_audio_task` closes
    its own row precisely so an abandoned build does not stay `running`.
    """
    try:
        result = celery_app.AsyncResult(task_id).get(
            timeout=AUDIO_GENERATE_TIMEOUT_S)
    except Exception:
        # DELIBERATELY NO REVOKE.
        elapsed = time.time() - started
        logger.info("audio: %s %r still building after %.0fs; answering 503 "
                    "and leaving it to finish (task %s)",
                    kind, name, elapsed, task_id)
        raise HTTPException(
            status_code=503,
            detail={
                "reason": "building",
                "kind": kind, "name": name, "generation_id": gen_id,
                "detail": (f"{kind} {name!r} is still building after "
                           f"{elapsed:.0f}s. It has NOT been cancelled"
                           f"{'' if queued else ' (another request started it)'}"
                           f" - ask again and it will be served from cache."),
                "retry_after_s": 60,
            },
            headers={"Retry-After": "60"})

    if result and result.get("error_kind") == "gpu_faulted":
        # Never retried here: a retry storm against a faulted card is what the
        # breaker exists to stop.
        raise HTTPException(
            status_code=503,
            detail={"reason": "gpu_faulted", "kind": kind, "name": name,
                    "detail": result.get("error"),
                    "retry_after_s": result.get("retry_after_s", 90)},
            headers={"Retry-After": str(result.get("retry_after_s", 90))})
    if not result or result.get("error"):
        raise HTTPException(
            status_code=500,
            detail=f"{kind} {name!r} failed to build: "
                   f"{(result or {}).get('error', 'unknown failure')}")

    row = generations.resolve_name(name, kind=kind)
    if not row:
        raise HTTPException(
            status_code=500,
            detail=f"{kind} {name!r} reported success but is not readable back")
    return _payload(row, cached=False, started=started)


@router.get("/api/audio/styles")
def list_styles(kind: str | None = Query(None),
                authorization: str | None = Header(default=None)):
    """The roster. `$[*].value` is the discovery pointer for something2."""
    auth.require(authorization, "read")
    return audio_styles.roster(kind)


@router.post("/api/audio")
def generate_audio(req: AudioRequest, request: Request,
                   authorization: str | None = Header(default=None)):
    principal = auth.require(authorization, "generate")
    started = time.time()
    kind = _require_kind(req.kind)
    name = _require_name(req.name)
    caller = generations.describe_caller(principal, request)

    cached = generations.resolve_name(name, kind=kind)
    if cached:
        generations.record(kind=kind, name=name, served_from="cache",
                           route="/api/audio", prompt=cached.get("prompt") or "",
                           duration_ms=(time.time() - started) * 1000,
                           params={"cache_of": str(cached["id"])}, caller=caller)
        return _payload(cached, cached=True, started=started)

    joined = _running_build(kind, name)
    if joined:
        logger.info("audio: %s %r is already building as %s; joining it",
                    kind, name, joined["celery_task_id"])
        return _await_build(joined["celery_task_id"], str(joined["id"]),
                            kind, name, started, queued=False)

    busy = _long_job_ahead()
    if busy:
        detail = (f"{kind} {name!r} is not built yet and cannot be built now: "
                  f"{busy}. The GPU worker runs one job at a time.")
        # A REFUSAL IS ACTIVITY - the same reasoning as the tile facade. Left
        # unrecorded, the Activity tab shows a quiet API while something2's
        # operator sees only a 503.
        generations.record(kind=kind, name=name, status="failed",
                           route="/api/audio", served_from="generated",
                           error=detail, caller=caller,
                           duration_ms=(time.time() - started) * 1000)
        raise HTTPException(status_code=503,
                            detail={"reason": "busy", "kind": kind,
                                    "name": name, "detail": detail,
                                    "retry_after_s": 120},
                            headers={"Retry-After": "120"})

    style = req.style
    if not style and req.context and not req.prompt:
        # Keyword fallback only; the LLM path is ticket 08. Recorded either
        # way so `author` never claims more than actually happened.
        style = audio_styles.style_from_context(req.context, kind)
    author = ("style chosen by keyword from context" if style and req.context
              and not req.style else "style given by the caller" if req.style
              else "roster default")

    gen = generations.begin(kind=kind, name=name, route="/api/audio",
                            prompt=(req.prompt or ""), seed=req.seed,
                            params={"style": style, "requested_duration_s":
                                    req.duration_s, "author": author},
                            caller=caller)

    task = celery_app.send_task(
        "tasks.generate_audio_task",
        kwargs={"kind": kind, "name": name, "style": style,
                "prompt": req.prompt, "slots": req.slots, "seed": req.seed,
                "duration_s": req.duration_s, "gen_id": gen})
    generations.attach_task(gen, task.id)
    return _await_build(task.id, gen, kind, name, started, queued=True)


@router.get("/api/audio")
def list_audio(kind: str | None = Query(None), name: str | None = Query(None),
               limit: int = Query(50, ge=1, le=200),
               authorization: str | None = Header(default=None)):
    """Audio rows from the ledger, newest first. Includes the ones still running."""
    auth.require(authorization, "read")
    import psycopg2
    import psycopg2.extras

    # ARTEFACTS and work in flight, not every request. A finished cache read
    # is a real ledger row but owns no file, so listing it here would show a
    # second, emptier copy of a track that already appears above it.
    where = ["kind = ANY(%s)", "deleted = false",
             "(file_path IS NOT NULL OR status <> 'done')"]
    args: list = [list(KINDS) if not kind else [_require_kind(kind)]]
    if name:
        where.append("lower(name) = lower(%s)")
        args.append(name.strip())
    args.append(limit)

    try:
        with psycopg2.connect(os.environ.get("DB_URL")) as conn, conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, kind, name, status, served_from, prompt, seed, "
                "       params, file_path, error, duration_ms, principal_name, "
                "       created_at, finished_at "
                f"FROM generations WHERE {' AND '.join(where)} "
                "ORDER BY created_at DESC LIMIT %s", args)
            rows = cur.fetchall()
    except Exception as e:
        logger.warning("audio list failed: %s", e)
        raise HTTPException(status_code=503, detail="ledger unavailable")

    out = []
    for r in rows:
        params = _params(r)
        base = f"/api/audio/{r['kind']}/{r['name']}" if r["name"] else None
        out.append({
            "id": str(r["id"]), "kind": r["kind"], "name": r["name"],
            "status": r["status"], "served_from": r["served_from"],
            "style": params.get("style"), "prompt": r["prompt"],
            "seed": r["seed"], "author": params.get("author"),
            "duration_s": params.get("duration_s"),
            "sample_rate": params.get("sample_rate"),
            "loop_start": params.get("loop_start"),
            "loop_end": params.get("loop_end"),
            "bars": params.get("bars"),
            "seam_rms_jump_db": params.get("seam_rms_jump_db"),
            "error": r["error"], "duration_ms": r["duration_ms"],
            "requested_by": r["principal_name"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            "finished_at": (r["finished_at"].isoformat()
                            if r["finished_at"] else None),
            # The open /audio mount, so a browser <audio> tag can play it
            # without a bearer; the /api/audio route below needs one.
            "url": generations._url(r["file_path"]),
            "download_url": base,
        })
    return {"items": out, "count": len(out)}


def _resolve_or_404(kind: str, name: str) -> dict:
    row = generations.resolve_name(name, kind=kind)
    if not row:
        # 404, not 409: a map with no track yet is the NORMAL state, and the
        # player should treat missing and unfinished the same way - play
        # nothing. See the contract.
        raise HTTPException(
            status_code=404,
            detail={"kind": kind, "name": name,
                    "detail": f"no finished {kind} named {name!r}"})
    return row


@router.get("/api/audio/{kind}/{name}")
def get_audio_file(kind: str, name: str, master: bool = Query(False),
                   authorization: str | None = Header(default=None)):
    auth.require(authorization, "read")
    kind = _require_kind(kind)
    row = _resolve_or_404(kind, _require_name(name))
    path = row["file_path"]
    if master:
        wav = os.path.splitext(path)[0] + ".wav"
        if not os.path.exists(wav):
            raise HTTPException(status_code=404,
                                detail={"kind": kind, "name": name,
                                        "detail": "no WAV master on disk"})
        return FileResponse(wav, media_type="audio/wav",
                            filename=os.path.basename(wav))
    return FileResponse(path, media_type="audio/ogg",
                        filename=os.path.basename(path))


@router.get("/api/audio/{kind}/{name}/info")
def get_audio_info(kind: str, name: str,
                   authorization: str | None = Header(default=None)):
    auth.require(authorization, "read")
    kind = _require_kind(kind)
    row = _resolve_or_404(kind, _require_name(name))
    return _info(row, cached=True, elapsed_ms=0)
