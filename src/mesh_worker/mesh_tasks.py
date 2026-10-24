#!/usr/bin/env python3
"""Celery worker for the static 8-direction turnaround lane.

WHY THIS IS A SECOND WORKER AND NOT A TASK ON `sprite-worker`
`sprite-worker` runs `--pool=solo`: one task at a time, because one card. A
turnaround measures 2m42s and uses NO GPU, so putting it there would block
every sheet, core and entity request for most of three minutes to do work that
never touches the thing being blocked. That is the same reasoning CLAUDE.md
records for the downloader not being a Celery task.

So this consumes its own queue (`mesh3d`) in its own container, and the two
lanes run genuinely in parallel: a turnaround can proceed while the GPU is
mid-sheet.

WHY IT IS NOT IN `model_gateway.GATED`
The gateway decides who gets the card. This lane never asks for it - TripoSR
runs on CPU here (~45s warm) and the renderer is numpy. Registering it would
make it wait for a resource it does not use. If a GPU reconstructor is ever
wired in (Hunyuan3D-2mv is the candidate, see decisions/0016) that changes and
it MUST be registered, or it bypasses the gateway silently.

It writes `sheet_path` and `atlas_path` on the jobs row, so the existing
GET /api/jobs/{id}, /sheet and /atlas serve the result with no new read side.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys

import psycopg2
from celery import Celery

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
DB_URL = os.environ.get("DB_URL", "")
IMAGES = os.environ.get("IMAGES_ROOT", "/app/images")

log = logging.getLogger("mesh_tasks")

celery_app = Celery("mesh_tasks", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.task_default_queue = "mesh3d"
# A turnaround is ~3 minutes and the stages are subprocesses, so a late ack
# means a worker restart re-runs it rather than losing it.
celery_app.conf.task_acks_late = True
celery_app.conf.worker_prefetch_multiplier = 1


def _patch(job_id: str, **cols):
    if not DB_URL or not cols:
        return
    sets = ", ".join(f"{k} = %s" for k in cols)
    try:
        with psycopg2.connect(DB_URL) as conn, conn.cursor() as cur:
            cur.execute(f"UPDATE jobs SET {sets} WHERE id = %s",
                        (*cols.values(), job_id))
    except Exception as e:                                # noqa: BLE001
        # Never fail the WORK because the ledger write failed - the same trade
        # every other status patch in this project makes.
        log.warning("job %s: status patch failed: %s", job_id, e)


def _stage(job_id: str, argv: list[str], label: str, pct: int) -> None:
    """Run one stage as a subprocess, streaming its output to the log."""
    _patch(job_id, progress_pct=pct, progress_msg=label)
    log.info("job %s: %s", job_id, label)
    proc = subprocess.run(argv, capture_output=True, text=True)
    if proc.stdout:
        for line in proc.stdout.splitlines():
            log.info("  %s", line)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-800:]
        raise RuntimeError(f"{label} failed (rc={proc.returncode}): {tail}")


@celery_app.task(name="mesh3d.turnaround", bind=True)
def turnaround_job(self, job_id: str, source_job: str, cell: str = "48x64"):
    """2D turnaround of `source_job` -> a static 8-direction sheet."""
    src = os.path.join(IMAGES, f"_job_{source_job}")
    work = os.path.join(src, "t3d")
    py = sys.executable
    t = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "turnaround.py")
    m = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh.py")
    sheet = os.path.join(work, "sheet.png")
    atlas = os.path.splitext(sheet)[0] + ".atlas.json"

    try:
        if not os.path.isdir(src):
            raise RuntimeError(f"no such job directory: {src}")
        _patch(job_id, status="running", progress_pct=2,
               progress_msg="preparing views")
        os.makedirs(work, exist_ok=True)

        _stage(job_id, [py, t, "views", src, os.path.join(work, "views")],
               "keying the turnaround views", 10)
        # --device cpu deliberately: this lane must not contend for the card.
        _stage(job_id, [py, m, os.path.join(work, "views", "front.png"),
                        "-o", os.path.join(work, "mesh.obj"),
                        "--device", "cpu"],
               "reconstructing the mesh", 30)
        _stage(job_id, [py, t, "colour", os.path.join(work, "mesh.obj"),
                        os.path.join(work, "views"),
                        os.path.join(work, "mesh_col.glb")],
               "projecting observed colour", 60)
        _stage(job_id, [py, t, "render", os.path.join(work, "mesh_col.glb"),
                        os.path.join(work, "renders"),
                        "--views", os.path.join(work, "views")],
               "rendering 8 azimuths", 80)
        _stage(job_id, [py, t, "sheet", os.path.join(work, "renders"),
                        sheet, "--cell", cell],
               "composing the sheet", 95)

        if not os.path.isfile(sheet):
            raise RuntimeError("the sheet stage reported success but wrote "
                               "no file")
        _patch(job_id, status="done", progress_pct=100,
               progress_msg="Complete", sheet_path=sheet,
               atlas_path=atlas if os.path.isfile(atlas) else None)
        log.info("job %s: done -> %s", job_id, sheet)
        return {"job_id": job_id, "sheet_path": sheet}
    except Exception as e:                                # noqa: BLE001
        log.exception("job %s failed", job_id)
        _patch(job_id, status="error", progress_pct=100,
               progress_msg="Failed", error=str(e)[:2000])
        raise
