"""Reference examples: upload art the output should look like, and measure it.

The three kinds map to the three UI tabs (`reference-core`, `reference-sprite`,
`reference-tile`) and to the three questions in `measure.py`.

WHAT AN UPLOAD ACTUALLY DOES

It measures. Storing the file is the boring half; the useful half is that a
tile upload answers "what camera does this world use?" - a question that has
been guessed at since ADR 0003 and never once measured.

A reference is never rejected. `usable` records the verdict and `why` explains
it, but the file is kept either way: knowing that an example is unsuitable, and
why, is worth more than a silent 400. Only `derive` filters on it.

STYLE PROFILES

A profile turns N references into the constraints the conveyor applies. It is
deliberately a separate, explicit step rather than something an upload mutates:
uploading one odd tile should not silently repoint the camera for every job
that follows.
"""

from __future__ import annotations

import json
import logging
import os
import uuid

import psycopg2
import psycopg2.extras
from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

import auth
import measure

logger = logging.getLogger(__name__)
router = APIRouter()

DB_URL = os.environ.get("DB_URL")
IMAGES_DIR = "/app/images"

KINDS = ("core", "sprite", "tile")

# Uploads are art, not archives. 24 MB is generous for a spritesheet and small
# enough that a mis-drag of a video file fails fast.
MAX_UPLOAD_BYTES = 24 * 1024 * 1024


def _db():
    return psycopg2.connect(DB_URL)


def _url(path: str | None) -> str | None:
    return ("/images/" + os.path.basename(path)) if path else None


def _row(r: dict) -> dict:
    d = dict(r)
    d["id"] = str(d["id"])
    d["url"] = _url(d.pop("file_path", None))
    d["created_at"] = d["created_at"].isoformat() if d.get("created_at") else None
    return d


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------

@router.post("/api/references", status_code=201)
async def upload_reference(kind: str = Form(...),
                           file: UploadFile = File(...),
                           label: str | None = Form(None),
                           authorization: str | None = Header(None)):
    """Store one example and measure it immediately.

    The measurement is returned in the response, so the UI can show what was
    learned from the file the moment it lands rather than after a refresh.
    """
    auth.require(authorization, "generate")

    if kind not in KINDS:
        raise HTTPException(status_code=400,
                            detail=f"kind must be one of {', '.join(KINDS)}")

    blob = await file.read()
    if not blob:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(blob) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"{len(blob) / 1e6:.1f} MB exceeds the "
                   f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB limit")

    ref_id = uuid.uuid4()
    path = os.path.join(IMAGES_DIR, f"ref_{kind}_{ref_id.hex[:12]}.png")

    # Normalise to RGBA PNG on the way in. Every measurement assumes RGBA, and
    # a JPEG reference would otherwise report "no transparency" as a property
    # of the art rather than of the format.
    try:
        import io
        img = Image.open(io.BytesIO(blob))
        img.convert("RGBA").save(path, "PNG")
    except Exception as e:
        raise HTTPException(status_code=400,
                            detail=f"not a readable image: {e}")

    try:
        verdict = measure.measure(kind, path)
    except Exception as e:
        logger.exception("measurement failed for %s", path)
        verdict = {"usable": None, "why": f"measurement failed: {e}",
                   "metrics": {}}

    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO reference_assets "
            "  (id, kind, file_path, label, metrics, usable, why, "
            "   trainable, trainable_why) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (str(ref_id), kind, path, label or file.filename,
             json.dumps(verdict["metrics"]), verdict["usable"], verdict["why"],
             verdict.get("trainable"), verdict.get("trainable_why")))

    return {"id": str(ref_id), "kind": kind, "url": _url(path),
            "label": label or file.filename, **verdict}


@router.get("/api/references")
def list_references(kind: str | None = None,
                    authorization: str | None = Header(None)):
    auth.require(authorization, "read")
    if kind and kind not in KINDS:
        raise HTTPException(status_code=400, detail=f"unknown kind {kind!r}")

    clause, params = "", []
    if kind:
        clause, params = "AND kind = %s", [kind]

    with _db() as conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            f"SELECT id, kind, file_path, label, metrics, usable, why, "
            f"       trainable, trainable_why, created_at FROM reference_assets "
            f"WHERE deleted = false {clause} ORDER BY created_at DESC", params)
        items = [_row(dict(r)) for r in cur.fetchall()]

    usable = sum(1 for i in items if i["usable"])
    trainable = sum(1 for i in items if i["trainable"])
    return {"items": items, "total": len(items),
            # TWO different counts, because they answer two different
            # questions. `usable` is measurement-grade - palette-locked, hard
            # alpha, isolated subject - and is what a style profile needs.
            # `trainable` is nearly everything, and is what training needs.
            "usable": usable, "trainable": trainable,
            "enough_to_train": trainable >= 20,
            "enough_to_measure": usable >= 1}


@router.delete("/api/references/{ref_id}")
def delete_reference(ref_id: str, authorization: str | None = Header(None)):
    auth.require(authorization, "generate")
    with _db() as conn, conn.cursor() as cur:
        cur.execute("UPDATE reference_assets SET deleted = true "
                    "WHERE id = %s::uuid AND deleted = false", (ref_id,))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="no such reference")
    return {"deleted": ref_id}


@router.post("/api/references/{ref_id}/remeasure")
def remeasure(ref_id: str, authorization: str | None = Header(None)):
    """Re-run measurement on a stored file.

    Exists because the measurements themselves are still being developed: when
    a rule improves, the references already uploaded should benefit without
    being re-uploaded.
    """
    auth.require(authorization, "generate")
    with _db() as conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT kind, file_path FROM reference_assets "
                    "WHERE id = %s::uuid AND deleted = false", (ref_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="no such reference")
        if not os.path.exists(row["file_path"]):
            raise HTTPException(status_code=410,
                                detail="the stored file is gone")

        v = measure.measure(row["kind"], row["file_path"])
        cur.execute("UPDATE reference_assets SET metrics = %s, usable = %s, "
                    "why = %s, trainable = %s, trainable_why = %s "
                    "WHERE id = %s::uuid",
                    (json.dumps(v["metrics"]), v["usable"], v["why"],
                     v.get("trainable"), v.get("trainable_why"), ref_id))
    return {"id": ref_id, **v}


# ---------------------------------------------------------------------------
# Style profiles
# ---------------------------------------------------------------------------

class DeriveRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    reference_ids: list[str] = Field(
        default_factory=list,
        description="empty means every usable reference")


def _mode(values):
    """Most common value, or None. Plain `max(set, key=count)` on an empty
    sequence raises, and half these fields are legitimately absent."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return max(set(vals), key=vals.count)


@router.post("/api/style-profiles/derive", status_code=201)
def derive_profile(body: DeriveRequest,
                   authorization: str | None = Header(None)):
    """Turn measured references into one set of constraints.

    Only `usable` references contribute. An unusable example is kept and shown,
    but letting a 200-colour render vote on the palette would defeat the point.
    """
    auth.require(authorization, "generate")

    where = "WHERE deleted = false AND usable = true"
    params: list = []
    if body.reference_ids:
        where += " AND id = ANY(%s::uuid[])"
        params.append(body.reference_ids)

    with _db() as conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(f"SELECT id, kind, metrics FROM reference_assets {where}",
                    params)
        refs = [dict(r) for r in cur.fetchall()]

    if not refs:
        raise HTTPException(
            status_code=400,
            detail="no usable references. Upload examples first - the tab "
                   "shows why any rejected file was rejected.")

    tiles = [r["metrics"] for r in refs if r["kind"] == "tile"]
    sprites = [r["metrics"] for r in refs if r["kind"] == "sprite"]

    # Camera, from tiles only. A character reference cannot supply this.
    elevation = _mode([t.get("elevation") for t in tiles])
    ratios = [t.get("projection_ratio") for t in tiles
              if t.get("projection_ratio")]
    projection = round(sum(ratios) / len(ratios), 3) if ratios else None

    # Palette: the union across sprite references, ordered by how many
    # references use each colour, so a shared core survives and one outlier's
    # accent does not crowd it out.
    seen: dict[str, int] = {}
    for s in sprites:
        for hexcol in s.get("palette", []):
            seen[hexcol] = seen.get(hexcol, 0) + 1
    palette = [c for c, _ in sorted(seen.items(), key=lambda kv: -kv[1])][:64]

    colors = max([s.get("colors", 0) for s in sprites], default=0) or None
    cell_w = _mode([s.get("art_w") for s in sprites])
    cell_h = _mode([s.get("art_h") for s in sprites])

    outline = None
    outlined = [s for s in sprites if s.get("has_outline")]
    if outlined and len(outlined) >= max(1, len(sprites) // 2):
        outline = {"width": 1, "color": _mode([s.get("outline_color")
                                               for s in outlined])}

    profile_id = uuid.uuid4()
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO style_profiles "
            "  (id, name, palette, cell_w, cell_h, colors, outline, "
            "   projection_ratio, elevation, derived_from) "
            # derived_from needs the explicit cast: psycopg2 sends a Python
            # list of str as text[], and Postgres will not coerce text[] to
            # uuid[] on its own. Same trap as `id = ANY(%s::uuid[])` in jobs.py.
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::uuid[]) "
            "ON CONFLICT (name) DO UPDATE SET "
            "  palette = EXCLUDED.palette, cell_w = EXCLUDED.cell_w, "
            "  cell_h = EXCLUDED.cell_h, colors = EXCLUDED.colors, "
            "  outline = EXCLUDED.outline, "
            "  projection_ratio = EXCLUDED.projection_ratio, "
            "  elevation = EXCLUDED.elevation, "
            "  derived_from = EXCLUDED.derived_from "
            "RETURNING id",
            (str(profile_id), body.name,
             json.dumps(palette) if palette else None,
             cell_w, cell_h, colors,
             json.dumps(outline) if outline else None,
             projection, elevation,
             [str(r["id"]) for r in refs]))
        profile_id = cur.fetchone()[0]

    gaps = []
    if not tiles:
        gaps.append("no tile reference, so the camera angle is still a guess "
                    "- this is the single most valuable thing to upload")
    if not sprites:
        gaps.append("no sprite reference, so palette, cell size and outline "
                    "are unset")

    return {
        "id": str(profile_id), "name": body.name,
        "from_references": len(refs),
        "palette": palette, "colors": colors,
        "cell_w": cell_w, "cell_h": cell_h, "outline": outline,
        "projection_ratio": projection, "elevation": elevation,
        "gaps": gaps,
    }


@router.get("/api/style-profiles")
def list_profiles(authorization: str | None = Header(None)):
    auth.require(authorization, "read")
    with _db() as conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT id, name, palette, cell_w, cell_h, colors, outline, "
            "       projection_ratio, elevation, lora_path, trigger_token, "
            "       created_at, updated_at FROM style_profiles "
            "ORDER BY updated_at DESC")
        out = []
        for r in cur.fetchall():
            d = dict(r)
            d["id"] = str(d["id"])
            for f in ("created_at", "updated_at"):
                d[f] = d[f].isoformat() if d[f] else None
            out.append(d)
    return {"items": out, "total": len(out)}


@router.delete("/api/style-profiles/{profile_id}")
def delete_profile(profile_id: str, authorization: str | None = Header(None)):
    """Hard delete - a profile is derived data and can be rebuilt from the
    references it names, so there is nothing to preserve by hiding it."""
    auth.require(authorization, "generate")
    with _db() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM style_profiles WHERE id = %s::uuid",
                    (profile_id,))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="no such profile")
    return {"deleted": profile_id}


@router.post("/api/references/remeasure-all")
def remeasure_all(kind: str | None = None,
                  authorization: str | None = Header(None)):
    """Re-run measurement over every stored reference.

    Exists because the rules themselves change. When `trainable` was split out
    of `usable`, 227 already-uploaded references carried verdicts from the old
    rule - and asking someone to re-upload 227 files because the judge improved
    is not a reasonable thing to ask.

    Synchronous: measurement is numpy over an already-decoded image, not GPU
    work. 227 references take a few seconds.
    """
    auth.require(authorization, "generate")
    if kind and kind not in KINDS:
        raise HTTPException(status_code=400, detail=f"unknown kind {kind!r}")

    clause, params = "", []
    if kind:
        clause, params = "AND kind = %s", [kind]

    with _db() as conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(f"SELECT id, kind, file_path FROM reference_assets "
                    f"WHERE deleted = false {clause}", params)
        rows = cur.fetchall()

    changed = missing = failed = 0
    for row in rows:
        if not os.path.exists(row["file_path"]):
            missing += 1
            continue
        try:
            v = measure.measure(row["kind"], row["file_path"])
        except Exception as e:
            logger.warning("remeasure failed for %s: %s", row["id"], e)
            failed += 1
            continue
        with _db() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE reference_assets SET metrics = %s, usable = %s, "
                "why = %s, trainable = %s, trainable_why = %s "
                "WHERE id = %s::uuid",
                (json.dumps(v["metrics"]), v["usable"], v["why"],
                 v.get("trainable"), v.get("trainable_why"), str(row["id"])))
        changed += 1

    with _db() as conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT kind, count(*) AS total, "
            "       count(*) FILTER (WHERE usable) AS usable, "
            "       count(*) FILTER (WHERE trainable) AS trainable "
            "FROM reference_assets WHERE deleted = false GROUP BY kind "
            "ORDER BY kind")
        summary = [dict(r) for r in cur.fetchall()]

    return {"remeasured": changed, "file_missing": missing,
            "failed": failed, "by_kind": summary}
