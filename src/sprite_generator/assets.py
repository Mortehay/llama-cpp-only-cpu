"""One list of everything this system has produced.

WHY THIS EXISTS

The gallery read `sprite_images` and nothing else. On 2026-08-25 that table
held **2** undeleted rows while `jobs` held **13** finished spritesheets - so
thirteen completed sheets, each an hour or more of GPU time, had never been
visible anywhere in the UI. Not a rendering bug: nothing ever asked for them.

The join lives in SQL (`assets_v`, migration 013) rather than here, so the two
producers keep writing exactly where they already write and there is no
synchronisation to get wrong. This module is the HTTP shape over that view.

Paths in the database are container-absolute (`/app/images/x.png`). The browser
needs `/images/x.png`, which is what `NoStoreStaticFiles` is mounted on. That
translation happens here, once, rather than in every template that ever renders
an image - which is how `gallery.html` ended up doing it inline with a Jinja
`split('/')[-1]`.
"""

from __future__ import annotations

import logging
import os

import psycopg2
import psycopg2.extras
from fastapi import APIRouter, Header, HTTPException, Query

import auth

logger = logging.getLogger(__name__)
router = APIRouter()

DB_URL = os.environ.get("DB_URL")
IMAGES_DIR = "/app/images"

# Anything the UI is allowed to filter on. A whitelist because these values
# reach an SQL WHERE clause; the value itself is still parameterised, but a
# typo should be a 400 rather than an empty list the user has to explain.
SOURCES = {"image", "job"}


def _db():
    return psycopg2.connect(DB_URL)


def to_url(file_path: str | None) -> str | None:
    """`/app/images/sheet_x.png` -> `/images/sheet_x.png`."""
    if not file_path:
        return None
    return "/images/" + os.path.basename(file_path)


def _row(r: dict) -> dict:
    return {
        "id": r["id"],
        "source": r["source"],
        "kind": r["kind"],
        "title": r["title"],
        "url": to_url(r["file_path"]),
        "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        "job_id": str(r["job_id"]) if r["job_id"] else None,
        "model": r["model"],
        # A sheet has an atlas describing its grid; a plain image does not.
        "atlas_url": (f"/api/jobs/{r['job_id']}/atlas" if r["job_id"] else None),
    }


@router.get("/api/assets")
def list_assets(kind: str | None = Query(None, description="core, sheet, ..."),
                source: str | None = Query(None, description="image | job"),
                q: str | None = Query(None, description="substring of title"),
                limit: int = Query(60, ge=1, le=500),
                offset: int = Query(0, ge=0),
                authorization: str | None = Header(None)):
    """Everything generated, newest first, filterable.

    Returns `total` alongside the page so the UI can page without a second
    round trip and can say "13 sheets" rather than "13 shown".
    """
    auth.require(authorization, "read")

    if source is not None and source not in SOURCES:
        raise HTTPException(
            status_code=400,
            detail=f"unknown source {source!r}; expected one of "
                   f"{', '.join(sorted(SOURCES))}")

    where, params = [], []
    if kind:
        where.append("kind = %s")
        params.append(kind)
    if source:
        where.append("source = %s")
        params.append(source)
    if q:
        where.append("title ILIKE %s")
        params.append(f"%{q}%")
    clause = ("WHERE " + " AND ".join(where)) if where else ""

    try:
        with _db() as conn, conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"SELECT count(*) AS n FROM assets_v {clause}", params)
            total = cur.fetchone()["n"]

            cur.execute(
                f"SELECT * FROM assets_v {clause} "
                f"ORDER BY created_at DESC NULLS LAST LIMIT %s OFFSET %s",
                params + [limit, offset])
            items = [_row(dict(r)) for r in cur.fetchall()]
    except psycopg2.Error as e:
        logger.exception("asset listing failed")
        raise HTTPException(status_code=503, detail=f"database error: {e}")

    return {"total": total, "limit": limit, "offset": offset, "items": items}


@router.get("/api/assets/kinds")
def asset_kinds(authorization: str | None = Header(None)):
    """The filter vocabulary, counted - so the UI renders only tabs that would
    return something, instead of offering an empty filter."""
    auth.require(authorization, "read")
    with _db() as conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT source, kind, count(*) AS n FROM assets_v "
                    "GROUP BY source, kind ORDER BY source, kind")
        return {"groups": [dict(r) for r in cur.fetchall()]}


def _purge_file(path: str | None) -> str | None:
    """Unlink a file, but only inside IMAGES_DIR. Returns its name, or None.

    The containment check is not decoration. These paths come from our own
    database, but this is the one route that turns a row into an `os.remove`,
    and a bad row should cost a log line rather than a file outside the images
    directory.
    """
    if not path:
        return None

    root = os.path.realpath(IMAGES_DIR)
    resolved = os.path.realpath(path)
    if os.path.commonpath([resolved, root]) != root:
        logger.warning("refusing to purge outside %s: %s", root, path)
        return None

    try:
        os.remove(resolved)
        return os.path.basename(resolved)
    except FileNotFoundError:
        return None
    except OSError as e:
        logger.warning("purge failed for %s: %s", path, e)
        return None


@router.delete("/api/assets/{source}/{asset_id}")
def delete_asset(source: str, asset_id: str,
                 purge: bool = Query(False,
                                     description="also unlink the file on disk"),
                 authorization: str | None = Header(None)):
    """Hide an asset, and with `purge=true` delete its file too.

    A job ROW is never destroyed either way: something2 may still poll that id,
    and a 404 on a job it was told to expect is a worse outcome than a hidden
    thumbnail.

    `purge` is the difference between reclaiming the listing and reclaiming the
    disk. It is opt-in because it is the half that cannot be undone, and
    because a something2 that polls afterwards gets a job whose `sheet_url` no
    longer resolves - which is precisely why hiding leaves the file alone.
    """
    auth.require(authorization, "generate")

    if source not in SOURCES:
        raise HTTPException(status_code=400, detail=f"unknown source {source!r}")

    # Paths are read before the UPDATE and unlinked after the transaction
    # commits: a file removed against a row that then fails to commit would
    # leave a visible asset pointing at nothing.
    paths: list[str | None] = []

    with _db() as conn, conn.cursor() as cur:
        if source == "image":
            if not asset_id.isdigit():
                raise HTTPException(status_code=400,
                                    detail="image ids are integers")
            cur.execute("SELECT file_path FROM sprite_images "
                        "WHERE id = %s AND deleted = false", (int(asset_id),))
            row = cur.fetchone()
            paths = list(row) if row else []
            cur.execute("UPDATE sprite_images SET deleted = true "
                        "WHERE id = %s AND deleted = false", (int(asset_id),))
        else:
            cur.execute("SELECT sheet_path, atlas_path FROM jobs "
                        "WHERE id = %s::uuid AND deleted = false", (asset_id,))
            row = cur.fetchone()
            paths = list(row) if row else []
            cur.execute("UPDATE jobs SET deleted = true "
                        "WHERE id = %s::uuid AND deleted = false", (asset_id,))

        if cur.rowcount == 0:
            raise HTTPException(status_code=404,
                                detail="no such visible asset")

    purged = [n for n in (_purge_file(p) for p in paths) if n] if purge else []
    return {"deleted": {"source": source, "id": asset_id}, "purged": purged}
