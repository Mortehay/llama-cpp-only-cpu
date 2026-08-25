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
from pydantic import BaseModel, Field, field_validator

import auth

logger = logging.getLogger(__name__)
router = APIRouter()

DB_URL = os.environ.get("DB_URL")
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

# Below this, a style LoRA memorises rather than generalises. Measurement needs
# three examples; training needs an order of magnitude more, and saying so up
# front is cheaper than discovering it after an hour of GPU time.
MIN_IMAGES = 8

# The three reference tabs. Kept here rather than imported from references.py
# so the API process does not need that module loaded to validate a request.
REFERENCE_KINDS = ("core", "sprite", "tile")


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

    # Which reference tabs feed the dataset. Defaults to characters, because
    # that is what most runs want - but tiles are a first-class choice, not an
    # exclusion.
    #
    # TRAIN CHARACTERS AND TILES SEPARATELY. One adapter over both teaches a
    # single trigger to mean "character" AND "ground texture", and the model
    # cannot tell which you meant - you get characters with grass in them. Two
    # adapters, two triggers, each one sharp:
    #
    #     kinds=["core", "sprite"]  -> a character style adapter
    #     kinds=["tile"]            -> a ground/terrain adapter
    #
    # Nothing stops you combining them; `mixed_kinds` in the response says so
    # rather than silently letting it happen.
    kinds: list[str] = Field(default_factory=lambda: ["sprite", "core"])

    @field_validator("kinds")
    @classmethod
    def _known_kinds(cls, v):
        bad = [k for k in v if k not in REFERENCE_KINDS]
        if bad:
            raise ValueError(f"unknown reference kind(s): {', '.join(bad)}; "
                             f"expected any of {', '.join(REFERENCE_KINDS)}")
        if not v:
            raise ValueError("choose at least one reference kind to train on")
        return v

    def pattern(self) -> str:
        """The glob the trainer reads, derived from the chosen tabs."""
        return ",".join(f"ref_{k}_*.png" for k in self.kinds)


def _trainable_reference_count(kinds) -> int:
    """How many TRAINABLE references these reference kinds hold.

    Counts `trainable`, not `usable`. Those answer different questions and
    conflating them was a real bug: `usable` is measurement-grade - palette
    locked, hard alpha, isolated subject - and gating training on it rejected
    100 of 106 real sprites for having too many colours, when a style LoRA is
    perfectly happy learning from a JPEG reference board. See migration 014.
    """
    kinds = list(kinds)
    if not kinds:
        return 0
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM reference_assets "
            "WHERE deleted = false AND trainable = true AND kind = ANY(%s)",
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

    usable = _trainable_reference_count(body.kinds)
    if usable < MIN_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"only {usable} trainable reference(s) in "
                   f"{', '.join(body.kinds)}; "
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
              "pattern": body.pattern(), "kinds": body.kinds,
              "trigger": body.trigger,
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

    # Say it rather than prevent it: mixing characters and ground tiles into one
    # adapter binds a single trigger to both, and the model cannot tell which
    # you meant. It is a legitimate thing to try; it is not a good default.
    mixed = "tile" in body.kinds and len(set(body.kinds) - {"tile"}) > 0

    return {"run_id": str(run_id), "status": "queued",
            "dataset_size": usable, "steps": body.steps,
            "kinds": body.kinds,
            "trigger": body.trigger or f"<{body.name}-style>",
            "mixed_kinds": mixed,
            "note": ("Training tiles together with characters - one trigger "
                     "will mean both, which usually reads as characters with "
                     "terrain texture in them. Two adapters give sharper "
                     "results." if mixed else None)}


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
def readiness(kinds: str = "sprite,core",
              authorization: str | None = Header(None)):
    """Can training start with these reference kinds, and if not, what is missing?

    Exists so the UI can disable the button with a reason rather than offering
    an action that will 400. `per_kind` lets it show which tabs are trainable -
    "tiles: 3/8" is a more useful prompt than a disabled button.
    """
    auth.require(authorization, "read")
    chosen = [k.strip() for k in kinds.split(",")
              if k.strip() in REFERENCE_KINDS] or ["sprite", "core"]
    usable = _trainable_reference_count(chosen)
    per_kind = {k: _trainable_reference_count([k]) for k in REFERENCE_KINDS}
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
            "kinds": chosen, "per_kind": per_kind,
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
