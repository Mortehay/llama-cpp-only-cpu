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

# The four reference tabs. Kept here rather than imported from references.py
# so the API process does not need that module loaded to validate a request.
REFERENCE_KINDS = ("core", "sprite", "tile", "map")

# Where scripts/train-lora.py writes adapters. Kept beside the HF cache rather
# than inside it so archive-models.sh cannot sweep the most expensive artefact
# on the machine into cold storage.
LORA_DIR = "/models/loras"

# A resumed run refines an adapter that has already seen the full set, so it
# does not need a from-scratch dataset. It does need more than one or two, or
# it just drags the adapter toward whatever arrived most recently.
MIN_NEW_IMAGES = 4


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
    #     kinds=["map"]             -> a world-map adapter
    #
    # Nothing stops you combining them; `mixed_kinds` in the response says so
    # rather than silently letting it happen.
    kinds: list[str] = Field(default_factory=lambda: ["sprite", "core"])

    # Retrain from scratch on EVERY trainable reference, rather than continuing
    # the existing adapter on the ones it has not seen.
    #
    # Default false, so adding twelve references costs twelve images of
    # training instead of two hundred. But note what incremental means here: it
    # RESUMES from the existing adapter. Training a fresh LoRA on only the new
    # images would produce one that has seen only those twelve - strictly worse
    # than what you already had, and with no error to say so. Incremental
    # without resume is not a cheaper version of training, it is a broken one.
    full_retrain: bool = False

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


def untrained_refs(kinds) -> list[tuple[str, str]]:
    """(id, file_path) for trainable references no SUCCESSFUL run has consumed.

    The join back to `training_runs` and the `status = 'done'` test are not
    incidental. Rows land in training_run_refs at SUBMIT time, before the run
    has proven anything, so a run that failed - or whose queue was purged -
    would otherwise remove its images from every future dataset permanently,
    having taught the adapter nothing.
    """
    kinds = list(kinds)
    if not kinds:
        return []
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT ra.id, ra.file_path FROM reference_assets ra "
            "WHERE ra.deleted = false AND ra.trainable = true "
            "  AND ra.kind = ANY(%s) "
            "  AND NOT EXISTS ("
            "     SELECT 1 FROM training_run_refs trr "
            "     JOIN training_runs tr ON tr.id = trr.run_id "
            "     WHERE trr.reference_id = ra.id AND tr.status = 'done') "
            "ORDER BY ra.created_at", (kinds,))
        return [(str(r[0]), r[1]) for r in cur.fetchall()]


def all_trainable_refs(kinds) -> list[tuple[str, str]]:
    """(id, file_path) for every trainable reference of these kinds."""
    kinds = list(kinds)
    if not kinds:
        return []
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, file_path FROM reference_assets "
            "WHERE deleted = false AND trainable = true AND kind = ANY(%s) "
            "ORDER BY created_at", (kinds,))
        return [(str(r[0]), r[1]) for r in cur.fetchall()]


def existing_adapter(name: str) -> str | None:
    """Path of an already-trained adapter with this name, or None."""
    path = os.path.join(LORA_DIR, f"{name}.safetensors")
    return path if os.path.isfile(path) else None


def trainable_files(kinds) -> list[str]:
    """The exact file paths training should read, straight from the DB.

    The trainer used to glob `/app/images/ref_sprite_*.png` itself, which
    disagreed with this count: 231 files against 191 trainable rows, because
    the directory still holds images belonging to DELETED references and to
    ones the judge rejected. The UI promised one dataset and the trainer used
    another, silently. Now the queue writes these paths to a manifest and the
    trainer reads only that.
    """
    kinds = list(kinds)
    if not kinds:
        return []
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT file_path FROM reference_assets "
            "WHERE deleted = false AND trainable = true AND kind = ANY(%s) "
            "ORDER BY created_at", (kinds,))
        return [r[0] for r in cur.fetchall()]


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

    # Incremental means: continue THIS adapter on the references it has not
    # seen. Only possible when the adapter already exists.
    resume_from = None if body.full_retrain else existing_adapter(body.name)

    if resume_from:
        refs = untrained_refs(body.kinds)
        floor, mode = MIN_NEW_IMAGES, "incremental"
    else:
        refs = all_trainable_refs(body.kinds)
        floor, mode = MIN_IMAGES, "full"

    usable = len(refs)
    if usable < floor:
        if mode == "incremental":
            total = _trainable_reference_count(body.kinds)
            raise HTTPException(
                status_code=400,
                detail=f"only {usable} reference(s) in "
                       f"{', '.join(body.kinds)} that '{body.name}' has not "
                       f"already trained on (need {floor}). Upload more, or "
                       f"tick 'retrain on all images' to rebuild the adapter "
                       f"from all {total}.")
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

    # Freeze the dataset as a manifest NOW, at submit time, rather than letting
    # the worker glob later. Two reasons: the glob and this count disagreed
    # (deleted and rejected references still have files on disk), and a run
    # should train on what was counted when it was queued, not on whatever the
    # directory happens to hold when it finally reaches the front of the queue.
    manifest_dir = "/models/loras"
    os.makedirs(manifest_dir, exist_ok=True)
    manifest = os.path.join(manifest_dir, f".{run_id}-files.txt")
    with open(manifest, "w") as f:
        f.write("\n".join(p for _, p in refs))

    config = {"name": body.name, "steps": body.steps, "rank": body.rank,
              "lr": body.lr, "resolution": body.resolution,
              "pattern": body.pattern(), "kinds": body.kinds,
              "files": manifest, "trigger": body.trigger,
              "mode": mode, "resume_from": resume_from,
              "min_images": floor}

    import json
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO training_runs (id, profile_id, base_model, config, "
            "                           dataset_size, steps_total) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (str(run_id), profile_id, BASE_MODEL, json.dumps(config),
             usable, body.steps))

    # Record which references this run consumed, so the NEXT run can tell what
    # is new. Written now, at submit; `untrained_refs` requires status='done'
    # before treating them as taught, so a failed run does not consume them.
    with _db() as conn, conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO training_run_refs (run_id, reference_id) VALUES %s "
            "ON CONFLICT DO NOTHING",
            [(str(run_id), rid) for rid, _ in refs])

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
            "kinds": body.kinds, "mode": mode,
            "resuming": bool(resume_from),
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
    # How many of those no successful run has consumed - what an incremental
    # run would actually train on.
    new_per_kind = {k: len(untrained_refs([k])) for k in REFERENCE_KINDS}
    new_count = len(untrained_refs(chosen))
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
            # What an INCREMENTAL run would use: references no successful run
            # has consumed yet. Lets each tab show "12 new" beside its Train
            # button instead of the full total, which would be a lie about
            # what pressing it does.
            "new_references": new_count, "new_per_kind": new_per_kind,
            "min_new_images": MIN_NEW_IMAGES,
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
