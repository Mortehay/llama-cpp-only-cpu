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
from tasks import celery_app, worker_busy_reason

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
    # Both registries - jobs table AND the Redis flag a Qwen-Image core sets.
    # Reading only the table let an audio build queue behind a 4-minute core.
    busy = worker_busy_reason()
    return busy["detail"] if busy else None


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


class ProposeRequest(BaseModel):
    context: str
    kind: str = "music"


@router.post("/api/audio/propose")
def propose_style(req: ProposeRequest,
                  authorization: str | None = Header(default=None)):
    """What a map description would get, WITHOUT generating anything.

    No GPU, no ledger row: the brain (two attempts, since a person is waiting
    and a cold load can outlast one) or the keyword rules pick a style and
    slots, and the template renders the prompt. The Audio tab fills its form
    from this; generating is still a separate click.
    """
    auth.require(authorization, "read")
    kind = _require_kind(req.kind)
    if not req.context.strip():
        raise HTTPException(status_code=422, detail="context is required")
    style, slots, author = audio_styles.plan_style(req.context, kind)
    rendered = audio_styles.render(style, **slots)
    return {"kind": kind, "style": style, "slots": rendered["slots"],
            "author": author, "prompt": rendered["prompt"],
            "adjusted": rendered["adjusted"]}


@router.get("/api/audio/styles")
def list_styles(kind: str | None = Query(None),
                authorization: str | None = Header(default=None)):
    """The roster. `$[*].value` is the discovery pointer for something2.

    `?kind=sfx` lists CUES instead (each with the engines it has a recipe
    for): a cue is not a style and has no slots.
    """
    auth.require(authorization, "read")
    if kind and kind.strip().lower() == "sfx":
        return audio_styles.cue_roster()
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

    # Precedence: an explicit prompt > an explicit style > the map description
    # (brain, else keyword rules) > the roster default. Slots the caller sent
    # beat the brain's, so a pinned mood survives a context. ONE brain
    # attempt here, not two: this request's budget is something2's 300 s and
    # the build below already gets 240 of it (see audio_styles).
    style, slots = req.style, dict(req.slots or {})
    if req.prompt:
        author = "prompt given by the caller (style template bypassed)"
    elif style:
        author = "style given by the caller"
    elif req.context:
        style, planned, author = audio_styles.plan_style(
            req.context, kind, attempts=1)
        slots = {**planned, **slots}
    else:
        author = "roster default"

    gen = generations.begin(kind=kind, name=name, route="/api/audio",
                            prompt=(req.prompt or ""), seed=req.seed,
                            params={"style": style, "slots": slots, "requested_duration_s":
                                    req.duration_s, "author": author},
                            caller=caller)

    task = celery_app.send_task(
        "tasks.generate_audio_task",
        kwargs={"kind": kind, "name": name, "style": style,
                "prompt": req.prompt, "slots": slots or None, "seed": req.seed,
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
    listed = KINDS + ("sfx",)
    if kind and kind.strip().lower() == "sfx":
        args: list = [["sfx"]]
    else:
        args = [list(listed) if not kind else [_require_kind(kind)]]
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
            # sfx only: every variant, so the tab can play each one.
            "variants": [generations._url(v.get("file_path"))
                         for v in (params.get("variants") or [])] or None,
            "cue": params.get("cue"), "entity": params.get("entity"),
            "engine": params.get("engine"),
            "engine_from": params.get("engine_from"),
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


# ---------------------------------------------------------------------------
# Sound effects: one-shot cues (0010 D7/D8, ticket 16)
# ---------------------------------------------------------------------------
#
# Addressed as `<cue>/<entity>`; the engine is part of the ledger name
# (`realistic:hit/slime`) so a realistic and a retro cue never share a cache
# slot. Same D5 facade as music: cache first, build within the budget, 503 +
# Retry-After on overshoot WITHOUT cancelling. A pack is one worker task and
# ONE model load; each cue still owns its own ledger row.

class SfxItem(BaseModel):
    cue: str
    entity: str | None = None
    engine: str | None = None


class SfxRequest(SfxItem):
    world: str | None = None
    variants: int = 3
    seed: int | None = None


class SfxPackRequest(BaseModel):
    items: list[SfxItem]
    world: str | None = None
    variants: int = 3
    seed: int | None = None


def _world_engine(world: str | None) -> str | None:
    """`sfx_engine` from the world's generation sidecar, or None.

    Level 2 of the precedence. The sidecar is the world's `.gen.json` - the
    file something2's seeder never reads - so the spec they consume is
    untouched. A world named but not found is a 404: a typo must not quietly
    drop to the cue default and mix looks.
    """
    if not world:
        return None
    import worlds
    _, _, gen_path = worlds._paths(world)
    if not os.path.exists(gen_path):
        raise HTTPException(status_code=404,
                            detail=f"no world named {world!r} to read an sfx "
                                   f"engine from")
    with open(gen_path, encoding="utf-8") as fh:
        return (json.load(fh) or {}).get("sfx_engine")


def _sfx_entry(row: dict, *, cached: bool, want: int, engine_from: str) -> dict:
    p = _params(row)
    variants = (p.get("variants") or [])[:want]
    return {
        "cue": p.get("cue"), "entity": p.get("entity"),
        "engine": p.get("engine"), "engine_from": engine_from,
        "name": row["name"], "prompt": row.get("prompt"), "seed": row.get("seed"),
        "sample_rate": p.get("sample_rate"),
        "variants": [{"url": generations._url(v["file_path"]),
                      "duration_s": v.get("duration_s"),
                      "onset_ms": v.get("onset_ms")} for v in variants],
        "audio": [_b64(v["file_path"]) for v in variants],
        "cached": cached, "served_from": "cache" if cached else "generated",
        "generation_id": str(row["id"]),
    }


def _serve_sfx(items: list[SfxItem], *, world: str | None, variants: int,
               seed: int | None, principal, request: Request) -> list[dict]:
    started = time.time()
    if not items:
        raise HTTPException(status_code=422, detail="no cues given")
    if not 1 <= variants <= 5:
        raise HTTPException(status_code=422, detail="variants must be 1-5")
    caller = generations.describe_caller(principal, request)
    world_engine = _world_engine(world)

    plan: list = []
    out: list[dict] = [{} for _ in items]
    for i, it in enumerate(items):
        try:
            engine, source = audio_styles.resolve_engine(it.cue, it.engine,
                                                         world_engine)
        except audio_styles.UnknownStyle:
            raise HTTPException(status_code=422,
                                detail=f"unknown cue {it.cue!r}; see "
                                       f"GET /api/audio/styles?kind=sfx")
        except audio_styles.NoRecipe as e:
            raise HTTPException(status_code=422, detail=str(e))
        name = audio_styles.sfx_name(engine, it.cue, it.entity)
        cached = generations.resolve_name(name, kind="sfx")
        if cached and len(_params(cached).get("variants") or []) >= variants:
            generations.record(kind="sfx", name=name, served_from="cache",
                               route="/api/audio/sfx",
                               prompt=cached.get("prompt") or "",
                               params={"cache_of": str(cached["id"])},
                               caller=caller,
                               duration_ms=(time.time() - started) * 1000)
            out[i] = _sfx_entry(cached, cached=True, want=variants,
                                engine_from=source)
            continue
        if _running_build("sfx", name):
            raise HTTPException(
                status_code=503, headers={"Retry-After": "60"},
                detail={"reason": "building", "kind": "sfx", "name": name,
                        "detail": f"{name!r} is already being built; ask "
                                  f"again and it will be served from cache.",
                        "retry_after_s": 60})
        plan.append((i, it, engine, source, name))

    # RETRO renders here, in the API process, before any worker check: it is
    # milliseconds of numpy and must never queue behind a GPU job (ticket 17).
    # With no seed, audio_retro derives one from the name - same name, same
    # sound. Only realistic cues continue to the worker below.
    gpu_plan = []
    for n, (i, it, engine, source, name) in enumerate(plan):
        if engine != "retro":
            gpu_plan.append((i, it, engine, source, name))
            continue
        gen = generations.begin(
            kind="sfx", name=name, route="/api/audio/sfx", seed=seed,
            params={"cue": it.cue, "entity": it.entity, "engine": engine,
                    "engine_from": source, "requested_variants": variants,
                    "world": world},
            caller=caller)
        t0 = time.time()
        try:
            import audio_engine
            import audio_retro
            res = audio_retro.build(it.cue, it.entity,
                                    (seed + n) if seed else None, variants,
                                    audio_engine.AUDIO_DIR)
        except Exception as e:  # noqa: BLE001 - one cue fails, not the pack
            generations.fail(gen, str(e), duration_ms=(time.time() - t0) * 1000)
            out[i] = {"cue": it.cue, "entity": it.entity, "engine": engine,
                      "engine_from": source, "name": name, "error": str(e)}
            continue
        generations.finish(gen, file_path=res["variants"][0]["file_path"],
                           seed=res["seed"], prompt=res["prompt"],
                           duration_ms=(time.time() - t0) * 1000,
                           params=audio_engine.sfx_ledger_params(
                               {"engine_from": source}, res))
        row = generations.resolve_name(name, kind="sfx")
        out[i] = (_sfx_entry(row, cached=False, want=variants,
                             engine_from=source) if row else
                  {"cue": it.cue, "name": name,
                   "error": "built but not readable back from the ledger"})
    plan = gpu_plan

    if not plan:
        return out

    busy = _long_job_ahead()
    if busy:
        raise HTTPException(status_code=503, headers={"Retry-After": "120"},
                            detail={"reason": "busy", "kind": "sfx",
                                    "detail": f"cannot build now: {busy}",
                                    "retry_after_s": 120})

    batch = []
    for n, (i, it, engine, source, name) in enumerate(plan):
        gen = generations.begin(
            kind="sfx", name=name, route="/api/audio/sfx", seed=seed,
            params={"cue": it.cue, "entity": it.entity, "engine": engine,
                    "engine_from": source, "requested_variants": variants,
                    "world": world},
            caller=caller)
        batch.append({"cue": it.cue, "entity": it.entity, "engine": engine,
                      "engine_from": source, "variants": variants,
                      # Distinct seeds per cue in a pack, still reproducible.
                      "seed": (seed + n) if seed else None, "gen_id": gen})
    task = celery_app.send_task("tasks.generate_sfx_task",
                                kwargs={"items": batch})
    for b in batch:
        generations.attach_task(b["gen_id"], task.id)

    try:
        result = celery_app.AsyncResult(task.id).get(
            timeout=AUDIO_GENERATE_TIMEOUT_S)
    except Exception:
        # DELIBERATELY NO REVOKE - the task closes its own rows (D5).
        raise HTTPException(
            status_code=503, headers={"Retry-After": "60"},
            detail={"reason": "building", "kind": "sfx",
                    "detail": f"{len(batch)} cue(s) still building after "
                              f"{time.time() - started:.0f}s. NOT cancelled - "
                              f"ask again and they will be served from cache.",
                    "retry_after_s": 60})
    if result and result.get("error_kind") == "gpu_faulted":
        raise HTTPException(
            status_code=503,
            headers={"Retry-After": str(result.get("retry_after_s", 90))},
            detail={"reason": "gpu_faulted", "kind": "sfx",
                    "detail": result.get("error"),
                    "retry_after_s": result.get("retry_after_s", 90)})
    if not result or result.get("error"):
        raise HTTPException(status_code=500,
                            detail=f"sfx build failed: "
                                   f"{(result or {}).get('error', 'unknown')}")

    for (i, it, engine, source, name), res in zip(plan, result["items"]):
        if res.get("error"):
            out[i] = {"cue": it.cue, "entity": it.entity, "engine": engine,
                      "engine_from": source, "name": name,
                      "error": res["error"]}
            continue
        row = generations.resolve_name(name, kind="sfx")
        out[i] = (_sfx_entry(row, cached=False, want=variants,
                             engine_from=source) if row else
                  {"cue": it.cue, "name": name,
                   "error": "built but not readable back from the ledger"})
    return out


@router.post("/api/audio/sfx")
def generate_sfx(req: SfxRequest, request: Request,
                 authorization: str | None = Header(default=None)):
    """One cue, 1-5 variants. `audio` holds one base64 OGG per variant."""
    principal = auth.require(authorization, "generate")
    entry = _serve_sfx([req], world=req.world, variants=req.variants,
                       seed=req.seed, principal=principal, request=request)[0]
    if entry.get("error"):
        raise HTTPException(status_code=500, detail=entry["error"])
    audio = entry.pop("audio")
    return {"audio": audio, "info": {"kind": "sfx", **entry}}


@router.post("/api/audio/sfx-pack")
def generate_sfx_pack(req: SfxPackRequest, request: Request,
                      authorization: str | None = Header(default=None)):
    """Many cues in ONE model load. A cue that fails is reported in its own
    entry; the others still return."""
    principal = auth.require(authorization, "generate")
    if len(req.items) > 40:
        raise HTTPException(status_code=422, detail="at most 40 cues per pack")
    items = _serve_sfx(req.items, world=req.world, variants=req.variants,
                       seed=req.seed, principal=principal, request=request)
    return {"items": items, "count": len(items),
            "failed": sum(1 for i in items if i.get("error"))}
