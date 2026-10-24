"""The API request ledger, and the entity registry built on top of it.

WHAT WAS BROKEN

`generate_raw_task` saved `raw_<uuid>.png` and returned the path. Nothing
recorded it. Measured 2026-09-10: 731 `raw_*.png` files on disk, zero rows in
`sprite_images` pointing at any of them. Every image something2 ever pulled
through the A1111 facade was an orphan - not in the gallery, not addressable,
not cached, so asking for the same goblin twice cost the GPU twice.

Tiles and maps were already exempt because a NAME gave them a row: the tile
facade cache-reads `jobs` by `spec->>'name'`. This module gives entities the
same thing, and gives every API call the provenance that `jobs` never carried.

TWO OBJECTS, ONE TABLE

A `generations` row is a REQUEST, not an image. Whether it also owns an image
is decided by one rule, and the rest of this module depends on it:

    file_path set, job_id NULL   this row owns the PNG (entity generation)
    job_id set, file_path NULL   a job owns the PNG (tile/map facade)

That is what keeps a facade-built tile from appearing twice in the gallery -
once as its job, once as the request that asked for it. See migration 017.

THE LEDGER MUST NEVER BREAK A GENERATION

Every write here is wrapped and logged rather than raised. An image that
generated correctly must not turn into a 500 because a bookkeeping INSERT
failed; the same trade `references.make_thumb` makes, for the same reason.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime

import psycopg2
import psycopg2.extras
from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse

import auth

logger = logging.getLogger(__name__)
router = APIRouter()

DB_URL = os.environ.get("DB_URL")
IMAGES_DIR = "/app/images"

# What produced the row. `raw` is the untagged txt2img call: something2 asks
# for barrels and for grass through the same route, and this service cannot
# tell them apart from pixels, so it declines to guess. A caller that knows
# says so with `override_settings.kind`; asking for a cutout is taken as saying
# "entity", since only an object composited over terrain needs one.
KINDS = ("raw", "entity", "tile", "map", "music", "ambience", "sfx")

# Terminal states, matching `jobs.TERMINAL` minus the ones a synchronous facade
# cannot reach: nothing here is queued long enough to be cancelled.
TERMINAL = {"done", "failed"}


def _db():
    return psycopg2.connect(DB_URL)


AUDIO_DIR = os.environ.get("AUDIO_DIR", "/app/audio")


def _url(path: str | None) -> str | None:
    """The open static URL for a file this ledger owns.

    Audio lives in its own tree (`<AUDIO_DIR>/<kind>/...`, served at
    `/audio/`) and keeps its subfolder in the URL; everything else is flat in
    IMAGES_DIR and served by basename.
    """
    if not path:
        return None
    root = AUDIO_DIR.rstrip("/") + "/"
    if path.startswith(root):
        return "/audio/" + path[len(root):]
    return "/images/" + os.path.basename(path)


# ---------------------------------------------------------------------------
# Who asked
# ---------------------------------------------------------------------------

def describe_caller(principal: dict | None, request: Request | None) -> dict:
    """Flatten the caller into the four columns the ledger stores.

    `client_addr` IS THE GATEWAY, NOT THE CLIENT. Measured 2026-09-10: every
    external request reaches this process as 172.18.0.1, the docker bridge
    gateway, and LAN traffic crosses `netsh interface portproxy` first, which
    SNATs again. Two machines on the Wi-Fi are indistinguishable by address.
    It is stored anyway - a value the UI can label honestly beats a blank
    someone will later "fix" by trusting `request.client.host` - but the
    identity that actually separates callers here is the API KEY, which is why
    `principal_name` is what the Activity tab shows first.
    """
    p = principal or {}
    name = p.get("name") or "unknown"
    # `auth.require` returns a synthetic principal in open mode. Saying so is
    # more useful than recording it as a named caller, because it means the
    # request carried no credential at all.
    if p.get("open"):
        name = "anonymous (auth not enforced)"

    addr = fwd = agent = None
    if request is not None:
        addr = request.client.host if request.client else None
        fwd = request.headers.get("x-forwarded-for")
        agent = request.headers.get("user-agent")

    return {
        "principal_id": p.get("id"),
        "principal_name": name,
        "client_addr": addr,
        "forwarded_for": fwd,
        "user_agent": (agent or "")[:400] or None,
    }


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def begin(*, kind: str = "entity", name: str | None = None,
          route: str = "/sdapi/v1/txt2img", prompt: str = "",
          negative_prompt: str = "", model: str | None = None,
          seed: int | None = None, params: dict | None = None,
          job_id: str | None = None, image_id: int | None = None,
          caller: dict | None = None) -> str | None:
    """Open a ledger row for a request that is about to run. Returns its id.

    Written BEFORE the work starts, not after it finishes, and that is the
    whole reason the Activity tab can show a pending list at all: a row that
    only appears on success cannot represent the thing you want to look at
    while you are waiting for it, and a request that dies mid-generation would
    leave no trace of ever having arrived.

    Returns None if the ledger could not be written. Callers pass that None
    straight back into `finish`/`fail`, which no-op on it, so a ledger outage
    degrades to "no record" rather than to a failed generation.

    `job_id` and `image_id` are set HERE, not at `finish`, whenever the work
    belongs to another table. While the link is NULL the `activity_v` dedup
    cannot see that this row and that job (or that `sprite_images` row) are the
    same thing, so the work appears twice in the pending list for exactly as
    long as it is pending - which is the whole window anyone is looking at it.

    An `image_id` row is also the one case that never calls `finish`: the
    worker writes the outcome to `sprite_images` and does not know this table
    exists, so `activity_v` resolves the status through the link instead. See
    migration 018.
    """
    gen_id = str(uuid.uuid4())
    c = caller or {}
    try:
        with _db() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO generations "
                "(id, kind, name, status, route, prompt, negative_prompt, "
                " model, seed, params, job_id, image_id, principal_id, "
                " principal_name, client_addr, forwarded_for, user_agent) "
                "VALUES (%s, %s, %s, 'running', %s, %s, %s, %s, %s, %s, %s, "
                "        %s, %s, %s, %s, %s, %s)",
                (gen_id, kind, (name or None), route, prompt or "",
                 negative_prompt or "", model, seed,
                 json.dumps(params or {}), job_id, image_id,
                 c.get("principal_id"), c.get("principal_name") or "unknown",
                 c.get("client_addr"), c.get("forwarded_for"),
                 c.get("user_agent")))
        return gen_id
    except Exception as e:
        logger.warning("ledger: could not open a row for %s %r: %s",
                       kind, name, e)
        return None


def record(*, kind: str, name: str | None = None, status: str = "done",
           served_from: str = "cache", route: str = "/sdapi/v1/txt2img",
           prompt: str = "", model: str | None = None,
           file_path: str | None = None, job_id: str | None = None,
           error: object = None, duration_ms: float | None = None,
           params: dict | None = None, caller: dict | None = None) -> str | None:
    """Write one already-finished row, in a single statement.

    `begin` + `finish` exists for work with a middle - a generation you want to
    watch on the Activity tab while it runs. A cache read has no middle: it is
    a file read that is over before a poll could observe it, and paying two
    round trips to represent a state that lasts microseconds would put database
    latency on the fast path the cache exists to provide.
    """
    gen_id = str(uuid.uuid4())
    c = caller or {}
    try:
        with _db() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO generations "
                "(id, kind, name, status, served_from, route, prompt, model, "
                " file_path, job_id, error, duration_ms, params, "
                " principal_id, principal_name, client_addr, forwarded_for, "
                " user_agent, finished_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                "        %s, %s, %s, %s, %s, now())",
                (gen_id, kind, (name or None), status, served_from, route,
                 prompt or "", model, file_path, job_id,
                 (str(error)[:2000] if error else None),
                 int(duration_ms) if duration_ms is not None else None,
                 json.dumps(params or {}),
                 c.get("principal_id"), c.get("principal_name") or "unknown",
                 c.get("client_addr"), c.get("forwarded_for"),
                 c.get("user_agent")))
        return gen_id
    except Exception as e:
        logger.warning("ledger: could not record %s %r: %s", kind, name, e)
        return None


def _update(gen_id: str | None, sets: str, params: tuple) -> None:
    if not gen_id:
        return
    try:
        with _db() as conn, conn.cursor() as cur:
            cur.execute(f"UPDATE generations SET {sets} WHERE id = %s",
                        params + (gen_id,))
    except Exception as e:
        logger.warning("ledger: could not update %s: %s", gen_id, e)


def finish(gen_id: str | None, *, file_path: str | None = None,
           job_id: str | None = None, seed: int | None = None,
           duration_ms: float | None = None,
           served_from: str = "generated",
           celery_task_id: str | None = None,
           params: dict | None = None, prompt: str | None = None) -> None:
    """Close a row as done.

    Pass EITHER `file_path` (this request made the image) OR `job_id` (a job
    owns it). Passing both would put the same PNG in the gallery twice; the
    caller knows which it is, so this does not guess.

    `params` MERGES into what `begin` wrote rather than replacing it, so a
    producer can add what it only learns by producing - an audio loop's real
    duration, sample rate and loop points - without erasing what the request
    asked for.

    `prompt` is for the same case: an audio request names a STYLE, and the
    text actually sent to the model is rendered from the roster inside the
    worker. Without this the Activity tab shows "(no prompt)" for every track.
    """
    # COALESCE on job_id, because `begin` may already have set it and a caller
    # that only has a file path would otherwise blank it - re-exposing the job
    # in activity_v as a second copy of itself.
    _update(gen_id,
            "status = 'done', file_path = %s, job_id = COALESCE(%s, job_id), "
            "seed = %s, "
            "params = params || %s::jsonb, "
            "prompt = COALESCE(NULLIF(%s, ''), prompt), "
            "duration_ms = %s, served_from = %s, "
            "celery_task_id = COALESCE(%s, celery_task_id), "
            "finished_at = now()",
            (file_path, job_id, seed, json.dumps(params or {}), prompt or "",
             int(duration_ms) if duration_ms is not None else None,
             served_from, celery_task_id))


def fail(gen_id: str | None, error: object, *,
         duration_ms: float | None = None) -> None:
    """Close a row as failed. The error text is the one the caller was given.

    Takes `object` rather than `str` because half the call sites have an
    exception in hand and the other half have an HTTP detail string; making
    each one remember to stringify is how one of them eventually does not.
    """
    _update(gen_id,
            "status = 'failed', error = %s, duration_ms = %s, "
            "finished_at = now()",
            (str(error)[:2000],
             int(duration_ms) if duration_ms is not None else None))


def is_closed(gen_id: str | None) -> bool:
    """True if this row already reached a terminal state.

    A producer checks this before writing: the audio facade lets a second
    request JOIN a build in flight, so by the time a task finishes its row may
    already have been closed by someone else. Unknown ids read as NOT closed,
    because refusing to record a real result is worse than a duplicate write.
    """
    if not gen_id:
        return False
    try:
        with _db() as conn, conn.cursor() as cur:
            cur.execute("SELECT status FROM generations WHERE id = %s", (gen_id,))
            row = cur.fetchone()
    except Exception as e:
        logger.warning("ledger: could not read %s: %s", gen_id, e)
        return False
    return bool(row) and row[0] in TERMINAL


def attach_task(gen_id: str | None, celery_task_id: str | None) -> None:
    """Record the Celery id, so a stuck row can be traced to a live task."""
    _update(gen_id, "celery_task_id = %s", (celery_task_id,))


# ---------------------------------------------------------------------------
# Reading by name - the entity registry
# ---------------------------------------------------------------------------

def resolve_name(name: str, kind: str = "entity") -> dict | None:
    """The newest FINISHED generation with this name, or None.

    Mirrors `tiles.resolve_name` deliberately, including the two properties
    that make it usable from the facade:

    Names are NOT unique. Re-rolling an entity writes a second row and the
    newest finished one wins, so a better result takes effect without anyone
    editing configuration on the calling machine.

    A row whose PNG has been swept off disk is NOT a hit. Returning it would
    make the facade answer 200 and then fail on the file read - the same bug
    the tile and map resolvers each guard against.
    """
    if not name or not name.strip():
        return None
    try:
        with _db() as conn, conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, name, kind, file_path, prompt, model, seed, params, "
                "       created_at, finished_at "
                "FROM generations "
                "WHERE kind = %s AND deleted = false AND status = 'done' "
                "  AND file_path IS NOT NULL AND lower(name) = lower(%s) "
                "ORDER BY COALESCE(finished_at, created_at) DESC LIMIT 1",
                (kind, name.strip()))
            row = cur.fetchone()
    except Exception as e:
        # A cache lookup that cannot run means "miss", never "error": the
        # facade can still generate, and failing the request here would turn a
        # database hiccup into a 500 on a route that had everything it needed.
        logger.warning("ledger: name lookup for %r failed: %s", name, e)
        return None

    if not row or not os.path.exists(row["file_path"]):
        return None
    return dict(row)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _iso(v) -> str | None:
    return v.isoformat() if isinstance(v, datetime) else None


def _activity_row(r: dict) -> dict:
    return {
        # WHO ASKED: api | internal | unknown. Not "which table owns the
        # artefact", which is what it meant before 0017 - and is why there is
        # no longer a `ui` value. See migration 022.
        "source": r["source"],
        # WHAT RAN IT: job (a `jobs` row owns the work, 202/poll) | sync.
        "lane": r["lane"],
        "id": r["id"],
        "kind": r["kind"],
        "name": r["name"],
        "status": r["status"],
        "served_from": r["served_from"],
        "title": r["title"],
        "model": r["model"],
        "url": _url(r["file_path"]),
        # Filled by `_attach_audio` for audio rows only; images stay None.
        "audio": None,
        # Filled by `_attach_progress` for rows still in flight; a finished
        # row keeps None, because its duration is the better answer.
        "progress": None,
        "job_id": str(r["job_id"]) if r["job_id"] else None,
        "error": r["error"],
        "duration_ms": r["duration_ms"],
        "requested_by": r["requested_by"],
        # Named `client_addr`, never `client_ip`. See describe_caller.
        "client_addr": r["client_addr"],
        # Whatever the caller put in X-Forwarded-For. Nothing in front of this
        # service sets it, so it is a hint the caller volunteered - useful when
        # a script identifies itself, worthless as proof of origin.
        "forwarded_for": r["forwarded_for"],
        "created_at": _iso(r["created_at"]),
        "finished_at": _iso(r["finished_at"]),
    }


AUDIO_KINDS = ("music", "ambience", "sfx")

# The kinds whose `model` is a BRAIN rather than an image checkpoint. One kind
# today, named rather than inlined because the two rosters are kept separate
# everywhere else in this service (the Models tab has three tables on purpose)
# and a second text kind would otherwise be filed as an image model silently.
BRAIN_KINDS = ("text",)
_AUDIO_FIELDS = ("style", "seed", "bpm", "time_signature", "duration_s",
                 "sample_rate", "loop_start", "loop_end", "seam_rms_jump_db",
                 "author")


def _attach_audio(cur, rows: list[dict]) -> None:
    """Give audio rows the fields a listener needs to judge a loop.

    `activity_v` does not carry `params`, and widening a UNION view is a
    migration; one extra query for the audio rows on this page is not.
    """
    ids = [r["id"] for r in rows
           if r["source"] == "api" and r["kind"] in AUDIO_KINDS]
    if not ids:
        return
    cur.execute("SELECT id::text AS id, seed, params FROM generations "
                "WHERE id::text = ANY(%s)", (ids,))
    found = {}
    for g in cur.fetchall():
        p = dict(g["params"] or {})
        p.setdefault("seed", g["seed"])
        info = {k: p.get(k) for k in _AUDIO_FIELDS}
        # A cache read owns no file and learned nothing; an all-null block
        # would render as a row of dashes.
        if any(v is not None for v in info.values()):
            found[g["id"]] = info
    for r in rows:
        if r["id"] in found:
            r["audio"] = found[r["id"]]



def _attach_progress(cur, rows: list[dict]) -> None:
    """Give the rows that are still moving their stage and percentage.

    Same trade as `_attach_audio`: `activity_v` carries neither `stage` nor
    `progress_pct`, widening a three-branch UNION view is a migration, and one
    extra query for the handful of unfinished rows on this page is not.

    Why it is worth a query at all: a sheet job is HOURS of GPU time, and the
    feed showed it as the bare word "running" for all of them. "Running" and
    "wedged" look identical, so the only way to tell them apart was the worker
    log - which is exactly the question this tab exists to answer. The counter
    is already in the jobs row (`cell 75/208`); nothing here computes it.

    Terminal rows are skipped on purpose. A finished job's last progress
    message is noise next to its duration, and this keeps the query bounded to
    what is actually in flight.
    """
    live = [r for r in rows if r["status"] in ACTIVE]
    if not live:
        return

    # A job-backed row is either the job itself, or a ledger row that points at
    # one (something2's sheet requests take that shape).
    job_ids = {r["job_id"] for r in live if r["job_id"]}
    job_ids |= {r["id"] for r in live if r["source"] == "job"}
    jobs = {}
    if job_ids:
        cur.execute("SELECT id::text AS id, stage, progress_pct, progress_msg "
                    "FROM jobs WHERE id::text = ANY(%s)", (list(job_ids),))
        jobs = {j["id"]: dict(j) for j in cur.fetchall()}

    # Browser tasks keep the same two columns in sprite_images, so a core that
    # is 60% through its steps can say so rather than also reading "running".
    ui_ids = [r["id"] for r in live if r["source"] == "ui"]
    tasks = {}
    if ui_ids:
        cur.execute("SELECT id::text AS id, progress_pct, progress_msg "
                    "FROM sprite_images WHERE id::text = ANY(%s)", (ui_ids,))
        tasks = {t["id"]: dict(t) for t in cur.fetchall()}

    for r in live:
        src = jobs.get(r["job_id"]) or jobs.get(r["id"]) or tasks.get(r["id"])
        if not src:
            continue
        pct, msg = src.get("progress_pct"), src.get("progress_msg")
        stage = src.get("stage")
        # All three empty is the state between "row written" and "worker picked
        # it up". Reporting 0% there would claim knowledge the row does not
        # have; leaving it null lets the UI keep saying "queued".
        if pct is None and not msg and not stage:
            continue
        r["progress"] = {"pct": pct, "stage": stage, "msg": msg}

# Anything still moving. `queued` and `running` are the two the jobs table
# uses; the ledger only ever writes `running`, because a synchronous facade
# call is already on the worker by the time the row exists.
ACTIVE = ("queued", "running")

# The two axes `activity_v` reports, named here so the filter validation and
# the view cannot disagree. `source` is WHO ASKED and `lane` is WHAT RAN IT;
# before 0017 one column tried to be both and was neither. The old `ui` value
# is gone: this browser authenticates with an ordinary API key, so "which
# browser" is not a question this service can answer - `requested_by` is.
SOURCES = ("api", "internal", "unknown")
LANES = ("job", "sync")


@router.get("/api/activity")
def list_activity(source: str | None = Query(None, description="api | internal | unknown"),
                  lane: str | None = Query(None, description="job | sync"),
                  requested_by: str | None = Query(None, description="exact API key name"),
                  status: str | None = Query(None),
                  kind: str | None = Query(None),
                  model: str | None = Query(None, description="exact model string"),
                  q: str | None = Query(None, description="substring of prompt"),
                  limit: int = Query(60, ge=1, le=500),
                  offset: int = Query(0, ge=0),
                  authorization: str | None = Header(None)):
    """Everything that has asked this machine to generate, newest first.

    Returns the history AND the active list in one response. Two round trips
    would let the tab render a pending item that the history below it says is
    already finished - a poll every few seconds makes that window frequent
    rather than theoretical.

    `active` is ordered OLDEST FIRST, which is the order the one-at-a-time GPU
    worker will get to them, and is therefore the only order in which the list
    answers "when is mine".
    """
    auth.require(authorization, "read")

    where, params = [], []
    if source:
        if source not in SOURCES:
            raise HTTPException(status_code=400,
                                detail=f"unknown source {source!r}; expected "
                                       f"one of {', '.join(SOURCES)}. `ui` was "
                                       f"removed in 0017 - the browser calls "
                                       f"with an API key like any other "
                                       f"consumer, so filter by requested_by")
        where.append("source = %s")
        params.append(source)
    if lane:
        if lane not in LANES:
            raise HTTPException(status_code=400,
                                detail=f"unknown lane {lane!r}; expected "
                                       f"one of {', '.join(LANES)}")
        where.append("lane = %s")
        params.append(lane)
    if requested_by:
        where.append("requested_by = %s")
        params.append(requested_by)
    if status:
        where.append("status = %s")
        params.append(status)
    if kind:
        where.append("kind = %s")
        params.append(kind)
    if model:
        where.append("model = %s")
        params.append(model)
    if q:
        where.append("title ILIKE %s")
        params.append(f"%{q}%")
    clause = ("WHERE " + " AND ".join(where)) if where else ""

    try:
        with _db() as conn, conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"SELECT count(*) AS n FROM activity_v {clause}", params)
            total = cur.fetchone()["n"]

            cur.execute(
                f"SELECT * FROM activity_v {clause} "
                f"ORDER BY created_at DESC NULLS LAST LIMIT %s OFFSET %s",
                params + [limit, offset])
            items = [_activity_row(dict(r)) for r in cur.fetchall()]

            # The pending list ignores the filters above on purpose: "what is
            # the GPU doing" is not a question about the page you are looking
            # at, and a filter that hid the running job would be actively
            # misleading.
            cur.execute(
                "SELECT * FROM activity_v WHERE status = ANY(%s) "
                "ORDER BY created_at ASC", (list(ACTIVE),))
            active = [_activity_row(dict(r)) for r in cur.fetchall()]
            _attach_audio(cur, items + active)
            _attach_progress(cur, items + active)

            cur.execute("SELECT source, lane, status, count(*) AS n "
                        "FROM activity_v GROUP BY source, lane, status "
                        "ORDER BY source, lane, status")
            counts = [dict(r) for r in cur.fetchall()]

            # WHO ASKED, as filter options. This replaces the old three-value
            # source dropdown: the key name is the finest identity this service
            # has (`describe_caller` explains why the address is not one), and
            # a fixed enum cannot answer "which of my consumers" - something2
            # alone holds three keys.
            cur.execute("SELECT requested_by AS name, source, count(*) AS n "
                        "FROM activity_v WHERE requested_by IS NOT NULL "
                        "GROUP BY requested_by, source ORDER BY n DESC, name")
            principals = [dict(r) for r in cur.fetchall()]

            # The model filter's options. Unfiltered, like `counts`: options
            # narrowed by the current filter could not be used to leave it.
            # Sheet rows carry no model in the view, so they never appear here.
            #
            # `model_kind` SPLITS BRAINS FROM IMAGE CHECKPOINTS. They share one
            # column and must not share one flat list: `qwen3.6-35b-a3b` is the
            # single largest value in it (2060 rows) while the column it lands
            # in is documented as "the image model that drew it - not a
            # language model". Derived from `kind` here rather than from
            # `models_status.IMAGE_KINDS`, which omits `edit`, `prop`, `sheet`
            # and `map_props` - all of which this view produces - and which the
            # Models tab's own queries depend on.
            cur.execute("SELECT model, "
                        "       CASE WHEN kind = ANY(%s) THEN 'brain' "
                        "            ELSE 'image' END AS model_kind, "
                        "       count(*) AS n "
                        "FROM activity_v WHERE model IS NOT NULL "
                        "GROUP BY model, model_kind ORDER BY n DESC, model",
                        (list(BRAIN_KINDS),))
            models = [dict(r) for r in cur.fetchall()]
    except psycopg2.Error as e:
        logger.exception("activity listing failed")
        raise HTTPException(status_code=503, detail=f"database error: {e}")

    return {"total": total, "limit": limit, "offset": offset,
            "items": items, "active": active, "counts": counts,
            "principals": principals, "models": models}


@router.get("/api/entities")
def list_entities(authorization: str | None = Header(None)):
    """Every NAMED entity this service can serve from cache, newest first.

    The counterpart of `GET /api/tiles`, and it exists for the same reason:
    something2 addresses an entity by name, and with no listing the only way to
    find out whether a name resolves is to ask for it and read the 404.

    Unnamed generations are omitted. They cannot be addressed, so listing them
    here would only invite someone to try; they are in the Gallery and on the
    Activity tab, which is where a one-off belongs.
    """
    auth.require(authorization, "read")
    with _db() as conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT DISTINCT ON (lower(name)) "
            "       name, id, prompt, model, seed, params, file_path, "
            "       finished_at "
            "FROM generations "
            "WHERE kind = 'entity' AND deleted = false AND status = 'done' "
            "  AND COALESCE(name, '') <> '' AND file_path IS NOT NULL "
            "ORDER BY lower(name), COALESCE(finished_at, created_at) DESC")
        rows = cur.fetchall()

    out = []
    for r in rows:
        params = r["params"] or {}
        out.append({
            "name": r["name"],
            "id": str(r["id"]),
            "prompt": r["prompt"],
            "model": r["model"],
            "seed": r["seed"],
            "cutout": bool(params.get("cutout")),
            "url": _url(r["file_path"]),
            # Reported so a caller can tell a stale row from a servable one
            # without fetching the PNG to find out.
            "on_disk": os.path.exists(r["file_path"]),
            "finished_at": _iso(r["finished_at"]),
        })
    out.sort(key=lambda e: e["finished_at"] or "", reverse=True)
    return {"entities": out, "count": len(out)}


@router.get("/api/entities/by-name/{name}")
def get_entity_by_name(name: str, authorization: str | None = Header(None)):
    """A finished entity PNG, by name. A cache READ - it never generates.

    The build-if-missing behaviour lives on the A1111 facade, because that is
    the surface something2 actually calls. This route is the plain way to check
    what that facade would serve without registering a provider to find out -
    exactly the split `tiles.get_tile_by_name` already makes.
    """
    auth.require(authorization, "read")
    row = resolve_name(name, "entity")
    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"no finished entity named {name!r}. GET /api/entities lists "
                   f"what exists; a txt2img call with 'entity:{name} <prompt>' "
                   f"builds one.")
    return FileResponse(row["file_path"], media_type="image/png")
