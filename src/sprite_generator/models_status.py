"""What models exist, which are on disk, which is on the card, and how they run.

The read surface behind the Models tab, plus the queue the tab uses to ask for
a download in the background.

WHY THIS IS MOSTLY A REPORTING MODULE
Nothing here measures anything new. Every brain call already persists its token
counts and timings: `tasks._text_ledger_params` writes `usage`, `timings` and
`finish_reason` into `generations.params` for `/api/text` and for in-job brain
calls alike. The gateway already knows what holds the card. `models.toml`
already declares what a machine should have, with authoritative sizes. The only
thing missing was something that read all four and put them in one place.

THREE FAMILIES, AND THEY ARE NOT INTERCHANGEABLE
`.ai/domain.md`: `llm_name` is the IMAGE model; a *brain* is the text/vision
model that plans and never draws. Tokens and tok/s exist only for brains, so
brains and image models get separate shapes rather than one table with blank
columns. Audio is a third case and gets neither: its ledger rows record no
model at all (`model` is NULL for every `music`/`ambience`/`sfx` row, measured
2026-10-13 over 2,854 of them), so a per-model audio breakdown is not possible
without a change to the audio path. It is reported as a total, honestly
labelled, instead of as an empty table.

DOWNLOADS ARE QUEUED, NEVER RUN HERE
The API does not fetch. It inserts a `model_downloads` row; the downloader
container's `scripts/provision-daemon.py` is the single consumer. Two writers
on one /models cache deadlock on huggingface_hub's per-file locks - the reason
that service used to refuse to download on startup - so there is exactly one,
and both it and `make provision` take the advisory lock in
`models_manifest.provision_lock`. See `.ai/decisions/0014`.
"""

from __future__ import annotations

import logging
import os
import sys
import uuid

import psycopg2
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

import auth
import core_models
import model_gateway

logger = logging.getLogger(__name__)
router = APIRouter()

DB_URL = os.environ.get("DB_URL", "")

# The manifest reader lives in scripts/, which compose mounts read-only at
# /app/scripts. Imported from there rather than copied in: "is this model
# present" is subtle enough to get wrong, and a second copy of it is precisely
# the drift `make test-models-manifest` exists to prevent.
SCRIPTS_DIR = os.environ.get("SCRIPTS_DIR", "/app/scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# Outside /app deliberately: /app is a bind mount of the source tree, and
# mounting a file beneath it makes Docker create the mount point there - a
# 0-byte src/sprite_generator/models.toml turned up in git status the first
# time this was /app/models.toml.
MANIFEST_PATH = os.environ.get("MODELS_MANIFEST", "/etc/sprite/models.toml")

# Ledger `kind` values, measured against the live table 2026-10-13. Listed
# explicitly rather than inferred: a new kind should show up as "not counted
# anywhere" and prompt a decision, not silently join the image numbers.
TEXT_KINDS = ("text",)
IMAGE_KINDS = ("entity", "raw", "core", "spritesheet", "tile", "map")
AUDIO_KINDS = ("music", "ambience", "sfx")


def _db():
    return psycopg2.connect(DB_URL)


def _manifest() -> dict:
    """The parsed manifest, or {} when it is not mounted.

    Degrades rather than 500s: the stats and the gateway state are useful on
    their own, and an operator whose compose predates the mount should see a
    working tab with a clear gap, not an error page.
    """
    try:
        from models_manifest import load
        return load(MANIFEST_PATH)
    except SystemExit as e:        # models_manifest.load exits on a bad path
        logger.warning("models.toml not readable at %s (%s)", MANIFEST_PATH, e)
        return {}
    except Exception as e:         # noqa: BLE001
        logger.warning("could not read %s: %s", MANIFEST_PATH, e)
        return {}


def _present(root: str, entry: dict) -> bool:
    try:
        from models_manifest import have
        return bool(have(root, entry))
    except Exception:              # noqa: BLE001
        return False


def _size_bytes(entry: dict) -> int | None:
    """What this entry will take on disk, in bytes.

    `bytes` is an exact measurement and wins; `approx_gb` is the Hub's own sum
    over the files the entry's globs select (see `make provision audit=1`), in
    GiB. An entry may legitimately have neither - `fetch = "self"` leaves the
    size to the owning project - so None is a real answer.
    """
    if entry.get("bytes"):
        return int(entry["bytes"])
    if entry.get("approx_gb"):
        return int(float(entry["approx_gb"]) * 1024 ** 3)
    return None


def _root(doc: dict) -> str:
    try:
        from models_manifest import cache_root
        return cache_root(doc)
    except Exception:              # noqa: BLE001
        return os.environ.get("HF_HUB_CACHE") or "/models"


# --- statistics ---------------------------------------------------------
#
# All of it from `generations`, which the Activity tab already reads. Grouped
# by `model`, which for a text row is the brain id and for an image row is
# `llm_name`.

_BRAIN_SQL = """
SELECT model,
       count(*)                                        AS calls,
       count(*) FILTER (WHERE status = 'failed')       AS failures,
       sum((params->'usage'->>'completion_tokens')::bigint)  AS out_tokens,
       sum((params->'usage'->>'total_tokens')::bigint)       AS total_tokens,
       avg((params->'timings'->>'decode_tok_s')::double precision) AS avg_tok_s,
       percentile_cont(0.5) WITHIN GROUP (
           ORDER BY (params->'timings'->>'generate_s')::double precision)
                                                       AS p50_generate_s,
       percentile_cont(0.95) WITHIN GROUP (
           ORDER BY (params->'timings'->>'generate_s')::double precision)
                                                       AS p95_generate_s,
       -- NULLIF: load_s is 0 on a warm call and only the FIRST call after a
       -- model switch pays it, so averaging the zeros would hide the real
       -- cold-start cost (36.6 s for the VL-8B, measured in 0013).
       avg(NULLIF((params->'timings'->>'load_s')::double precision, 0))
                                                       AS avg_cold_load_s,
       max(created_at)                                 AS last_used
  FROM generations
 WHERE kind = ANY(%s) AND NOT deleted AND model IS NOT NULL
 GROUP BY model
"""

_IMAGE_SQL = """
SELECT model,
       count(*)                                        AS calls,
       count(*) FILTER (WHERE status = 'failed')       AS failures,
       count(*) FILTER (WHERE served_from = 'cache')   AS cache_hits,
       avg(duration_ms) FILTER (WHERE status = 'done'
                                AND served_from = 'generated') AS avg_ms,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_ms)
           FILTER (WHERE status = 'done' AND served_from = 'generated')
                                                       AS p50_ms,
       percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms)
           FILTER (WHERE status = 'done' AND served_from = 'generated')
                                                       AS p95_ms,
       max(created_at)                                 AS last_used
  FROM generations
 WHERE kind = ANY(%s) AND NOT deleted AND model IS NOT NULL
 GROUP BY model
"""

# Audio has no model attribution (see the module docstring), so it is a single
# row per kind rather than per model.
_AUDIO_SQL = """
SELECT kind,
       count(*)                                        AS calls,
       count(*) FILTER (WHERE status = 'failed')       AS failures,
       avg(duration_ms) FILTER (WHERE status = 'done') AS avg_ms,
       max(created_at)                                 AS last_used
  FROM generations
 WHERE kind = ANY(%s) AND NOT deleted
 GROUP BY kind
"""

_TIMESERIES_SQL = """
SELECT date_trunc('day', created_at) AS day,
       kind,
       count(*)                                        AS calls,
       coalesce(sum((params->'usage'->>'total_tokens')::bigint), 0) AS tokens
  FROM generations
 WHERE NOT deleted AND created_at > now() - make_interval(days => %s)
 GROUP BY 1, 2
 ORDER BY 1
"""


def _rows(sql: str, args: tuple) -> list[dict]:
    with _db() as conn, conn.cursor() as cur:
        cur.execute(sql, args)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def _iso(v):
    return v.isoformat() if hasattr(v, "isoformat") else v


def _clean(d: dict) -> dict:
    """JSON-safe: Decimal -> float, datetime -> iso, and round the noise off."""
    out = {}
    for k, v in d.items():
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        elif isinstance(v, (int, str, bool)) or v is None:
            out[k] = v
        else:
            try:
                f = float(v)
                out[k] = round(f, 3)
            except (TypeError, ValueError):
                out[k] = str(v)
    return out


@router.get("/api/models/status")
def models_status(days: int = 30, authorization: str | None = Header(None)):
    """Declared / on disk / in use / how it performs, in one payload."""
    auth.require(authorization, "read")
    days = max(1, min(int(days), 365))

    doc = _manifest()
    root = _root(doc)
    entries = doc.get("model", [])
    dead = [d.get("ref") for d in doc.get("dead", [])]

    gw = {}
    try:
        gw = model_gateway.status()
    except Exception as e:         # noqa: BLE001 - redis down must not 500 this
        logger.warning("gateway status unavailable: %s", e)
        gw = {"error": str(e)}
    # `status()["active"]` is a dict ({model, pinned, ...}) today. Tolerate a
    # bare string too: this is a read-only status surface and a gateway shape
    # change should not 500 the whole tab.
    raw_active = gw.get("active") if isinstance(gw, dict) else None
    if isinstance(raw_active, dict):
        active = raw_active.get("model")
    elif isinstance(raw_active, str):
        active = raw_active
    else:
        active = None

    brains = {r["model"]: _clean(r) for r in _rows(_BRAIN_SQL, (list(TEXT_KINDS),))}
    images = {r["model"]: _clean(r) for r in _rows(_IMAGE_SQL, (list(IMAGE_KINDS),))}

    declared = []
    for e in entries:
        present = _present(root, e)
        declared.append({
            # One normalised size so the UI does not have to know that an
            # hf-file carries exact `bytes` while a repo entry carries
            # `approx_gb` (the Hub's sum over the files its globs select).
            # Summing only approx_gb was a real bug: it silently dropped both
            # brains and every GGUF, reporting 28 GB where the manifest
            # declares 56.
            "size_bytes": _size_bytes(e),
            "name": e.get("name"),
            "kind": e.get("kind"),
            "tier": e.get("tier"),
            "repo": e.get("repo"),
            "dest": e.get("dest"),
            "gated": bool(e.get("gated")),
            "fetch_self": e.get("fetch") == "self",
            "bytes": e.get("bytes"),
            "approx_gb": e.get("approx_gb"),
            "why": e.get("why"),
            "skipping": e.get("skipping"),
            "provenance": e.get("provenance"),
            "present": present,
            # A `local` entry can never be fetched; the UI must offer a copy
            # instruction rather than a download button.
            "downloadable": (e.get("kind") != "local"
                             and e.get("fetch") != "self"
                             and not present),
        })

    missing_required = [d["name"] for d in declared
                        if d["tier"] == "required" and not d["present"]]

    return {
        "manifest_loaded": bool(entries),
        "manifest_path": MANIFEST_PATH,
        "models_dir": root,
        "declared": declared,
        "dead": dead,
        "missing_required": missing_required,
        "totals": {
            "declared": len(declared),
            "present": sum(1 for d in declared if d["present"]),
            "required_gb": round(sum(d["size_bytes"] or 0 for d in declared
                                     if d["tier"] == "required") / 1024 ** 3, 2),
            "all_gb": round(sum(d["size_bytes"] or 0
                                for d in declared) / 1024 ** 3, 2),
            "missing_gb": round(sum(d["size_bytes"] or 0 for d in declared
                                    if not d["present"]) / 1024 ** 3, 2),
        },
        "gateway": gw,
        "active": active,
        # The roster is what the UI offers for generation, which is a different
        # question from what the manifest declares - a trained `local:` adapter
        # is in one and not the other.
        "roster": core_models.roster(),
        "brains": brains,
        "images": images,
        "audio": [_clean(r) for r in _rows(_AUDIO_SQL, (list(AUDIO_KINDS),))],
        "audio_note": ("audio rows record no model, so these are per-kind "
                       "totals rather than per-model statistics"),
        "timeseries": [_clean(r) for r in _rows(_TIMESERIES_SQL, (days,))],
        "days": days,
    }


# --- the download queue -------------------------------------------------

class DownloadRequest(BaseModel):
    entry: str


@router.get("/api/models/downloads")
def list_downloads(limit: int = 50, authorization: str | None = Header(None)):
    auth.require(authorization, "read")
    limit = max(1, min(int(limit), 200))
    rows = [_clean(r) for r in _rows(
        "SELECT id::text, entry, status, bytes_total, bytes_done, error, "
        "       requested_by, created_at, started_at, finished_at "
        "  FROM model_downloads ORDER BY created_at DESC LIMIT %s",
        (limit,))]
    for r in rows:
        # Clamped here rather than in the UI, because the raw bytes are
        # deliberately honest and CAN exceed the declared total: the on-disk
        # tree carries HF metadata (locks, refs, download state) that the Hub's
        # file-size sum does not. A real fetch measured 43.2 MiB against a
        # declared 41.0 MiB - 105% - which is correct data and a broken
        # progress bar. `done` is pinned to 100 so a finished row never shows
        # 97% because of blob dedup.
        total, got = r.get("bytes_total") or 0, r.get("bytes_done") or 0
        if r.get("status") == "done":
            r["progress_pct"] = 100
        elif total > 0:
            r["progress_pct"] = max(0, min(100, int(100 * got / total)))
        else:
            r["progress_pct"] = None    # size unknown: show indeterminate
    return {"downloads": rows}


@router.post("/api/models/download")
def queue_download(req: DownloadRequest,
                   authorization: str | None = Header(None)):
    """Queue one manifest entry for the background downloader.

    `generate` scope, not `read`: this spends bandwidth and tens of GB of disk.
    """
    principal = auth.require(authorization, "generate")
    name = (req.entry or "").strip()
    doc = _manifest()
    entries = {m["name"]: m for m in doc.get("model", [])}
    entry = entries.get(name)
    if entry is None:
        raise HTTPException(status_code=404,
                            detail=f"'{name}' is not a models.toml entry")
    if entry.get("kind") == "local":
        raise HTTPException(status_code=400, detail=(
            f"'{name}' was trained on a machine, not downloaded - copy "
            f"{_root(doc)}/{entry.get('dest', 'loras')} by hand, or retrain"))
    if entry.get("fetch") == "self":
        raise HTTPException(status_code=400, detail=(
            f"'{name}' is fetched by the owning project on first use; "
            f"there is nothing to queue"))
    if _present(_root(doc), entry):
        raise HTTPException(status_code=409,
                            detail=f"'{name}' is already on disk")

    total = entry.get("bytes")
    if not total and entry.get("approx_gb"):
        total = int(float(entry["approx_gb"]) * 1024 ** 3)
    gen_id = str(uuid.uuid4())
    try:
        with _db() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO model_downloads "
                "(id, entry, status, bytes_total, requested_by) "
                "VALUES (%s, %s, 'queued', %s, %s)",
                (gen_id, name, total,
                 principal.get("name") or "unknown"))
            conn.commit()
    except psycopg2.errors.UniqueViolation:
        # The partial unique index on (entry) WHERE status IN (queued,running).
        # This is the double-click, and letting it through would put two
        # writers on one cache.
        raise HTTPException(status_code=409, detail=(
            f"'{name}' is already queued or downloading"))
    return {"id": gen_id, "entry": name, "status": "queued",
            "bytes_total": total}


@router.post("/api/models/downloads/{download_id}/cancel")
def cancel_download(download_id: str, authorization: str | None = Header(None)):
    """Cancel a download that has not started.

    A RUNNING one is deliberately not interruptible: huggingface_hub would
    leave `.incomplete` blobs behind, and the presence check reads those as
    absent, so the entry would silently start from scratch next time rather
    than resume. Let it finish.
    """
    auth.require(authorization, "generate")
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE model_downloads SET status='cancelled', finished_at=now() "
            "WHERE id = %s AND status = 'queued' RETURNING id::text",
            (download_id,))
        row = cur.fetchone()
        conn.commit()
    if not row:
        raise HTTPException(status_code=409, detail=(
            "only a queued download can be cancelled; a running one is left "
            "to finish so it can resume rather than restart"))
    return {"id": download_id, "status": "cancelled"}
