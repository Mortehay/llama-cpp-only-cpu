"""Bookkeeping for queued jobs: DB patches, stage subprocesses, stranded rows.

WHY THIS IS A SEPARATE MODULE

`tasks.py` is 3,200 lines. Every feature that touches generation has to be
edited inside a file too large to hold in your head, and the newest code - the
job orchestration added for something2 - was buried in the middle of it.

This is the part of that file with the LEAST coupling to everything else: it
talks to Postgres and to `subprocess`, and it imports no torch, no diffusers, no
Celery. That makes it the safe piece to lift out first, and the useful one -
it is also the piece under active change.

WHAT DELIBERATELY STAYED BEHIND

The Celery task functions themselves. `build_sheet_job`, `train_lora_job` and
`build_tile_job` remain registered in `tasks.py`, because:

  * `celery -A tasks worker` imports `tasks`, and a task defined elsewhere is
    only registered if something imports that module first - an easy way to get
    "Received unregistered task" on a queue that used to work;
  * moving them would need `celery_app` in a third module to avoid a circular
    import, which is a bigger change than this one, and bigger changes to a
    working GPU conveyor are how working software gets broken for tidiness.

The pipeline builders (`get_sd_pipeline`, `get_flux_pipeline`) also stayed.
They share module-level mutable state (`pipes`) with a dozen call sites and
have no tests; extracting them safely needs tests first, not confidence.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

DB_URL = os.environ.get("DB_URL")

# `denoised N/M` - the per-cell line the build stages print. The stage runs for
# the better part of an hour, so this is the difference between a progress bar
# and a number that never moves.
CELL_RE = re.compile(r"denoised\s+(\d+)/(\d+)")


def now():
    return datetime.now(timezone.utc)


def _connect():
    return psycopg2.connect(DB_URL)


def job_update(job_id: str, **fields):
    """Patch a jobs row. Never raises - bookkeeping must not fail a build."""
    if not fields:
        return
    sets = ", ".join(f"{k} = %s" for k in fields)
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(f"UPDATE jobs SET {sets} WHERE id = %s",
                        (*fields.values(), job_id))
    except Exception as e:
        logger.warning("could not update job %s: %s", job_id, e)


def training_update(run_id: str, **fields):
    """Patch one training_runs row. Same shape, same never-raises rule."""
    if not fields:
        return
    cols = ", ".join(f"{k} = %s" for k in fields)
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(f"UPDATE training_runs SET {cols} WHERE id = %s::uuid",
                        list(fields.values()) + [run_id])
    except Exception as e:
        logger.warning("training_runs update failed for %s: %s", run_id, e)


def load_style_profile(name):
    """The named style profile as a dict, or None.

    Returns None for a missing name rather than raising: a profile is an
    optional refinement, and failing an hour-long sheet build because a profile
    was renamed would be a poor trade. The absence is logged instead, so a job
    that silently used defaults can still be explained afterwards.
    """
    if not name:
        return None
    try:
        with _connect() as conn, conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT name, palette, cell_w, cell_h, colors, outline, "
                "       projection_ratio, elevation, lora_path, trigger_token "
                "FROM style_profiles WHERE name = %s", (name,))
            row = cur.fetchone()
    except Exception as e:
        logger.warning("could not read style profile %r: %s", name, e)
        return None
    if not row:
        logger.warning("style profile %r does not exist - using job defaults",
                       name)
        return None
    return dict(row)


def run_stage(cmd, job_id, pct_from, pct_to, tail_lines=6, env=None):
    """Run one build stage, streaming its output to update job progress.

    Returns (returncode, tail_of_output).

    Streaming rather than `subprocess.run(capture_output=True)`, which was the
    first version: the denoise stage runs for the better part of an hour and the
    job sat at a single percentage the whole time, with nothing to distinguish
    "working" from "hung". The stage already logs `denoised N/M` per cell, so
    the information existed - it was just being swallowed until the process
    exited.

    Progress is interpolated between the stage's start and end percentages so
    the number stays monotonic across stages.

    `env` defaults to None, which makes Popen inherit this process's
    environment - the previous behaviour. It is passed explicitly when a style
    profile has a camera to impose.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            env=env)
    recent = []
    last_written = -1
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        # Keep only a short tail: a denoise stage emits hundreds of progress
        # bar lines and none of them identify a failure.
        recent.append(line)
        if len(recent) > tail_lines:
            recent.pop(0)

        m = CELL_RE.search(line)
        if not m:
            continue
        done, total = int(m.group(1)), int(m.group(2))
        if total <= 0:
            continue
        pct = pct_from + int((pct_to - pct_from) * done / total)
        # Only write when the number actually moves - one UPDATE per cell is
        # fine, one per output line is not.
        if pct != last_written:
            job_update(job_id, progress_pct=pct,
                       progress_msg=f"cell {done}/{total}")
            last_written = pct

    proc.wait()
    return proc.returncode, "\n".join(recent)


def fail_stranded_jobs():
    """Mark jobs as failed when the worker that owned them died.

    `_fail_stranded_tasks` in tasks.py covers sprite_images. Jobs were left out,
    and they strand for longer and more visibly: a sheet job runs for an hour
    across five subprocesses, so a worker that dies mid-run leaves a row
    claiming `running` with a stage and a percentage that will never move
    again. Observed directly - a report showed "running: 2" while exactly one
    job was alive.

    That is worse than a stale sprite_images row, because something2 polls jobs
    and its whole model is "ask about this id later". A job frozen at
    `actions-denoise 44%` tells a caller to keep waiting forever.

    Called from a worker-ready signal, where one-worker/solo-pool means nothing
    can be running - so anything still queued or running is dead. The 30-second
    floor covers a submit-just-before-boot race.
    """
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE jobs
                   SET status = 'failed',
                       error = COALESCE(NULLIF(error, ''),
                                        'worker died before this job '
                                        'finished; resubmit to retry'),
                       progress_msg = 'stranded',
                       finished_at = now()
                 WHERE status IN ('queued', 'running')
                   AND created_at < NOW() - INTERVAL '30 seconds'
                """
            )
            if cur.rowcount:
                logger.warning(
                    "Marked %d stranded job(s) as failed: their worker died "
                    "before they finished.", cur.rowcount)
    except Exception as e:
        # Never fatal at boot. A worker that refuses to start because a
        # bookkeeping sweep failed is a worse outcome than a stale row.
        logger.error("Stranded-job reaper failed: %s", e)


def fail_stranded_training_runs():
    """Mark training runs as failed when nothing is left to run them.

    The third table that needed this, and the one where the gap bit hardest.
    `training_runs` had no reaper at all, so a run whose Celery message was
    lost - a purged queue, a worker killed mid-encode - stayed `queued`
    forever. That is worse than a stale row: `POST /api/training` refuses with
    409 while any run is queued or running, so ONE orphan permanently blocks
    all future training, and the only symptom is a 409 that looks correct.

    Observed exactly that after purging a 168-message backlog: the run row
    survived its own task.

    Same one-worker/solo-pool reasoning as the other two reapers - when this
    fires nothing can be running - and the same 30-second floor for the
    submit-just-before-boot race.
    """
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE training_runs
                   SET status = 'failed',
                       error = COALESCE(NULLIF(error, ''),
                                        'worker restarted before this run '
                                        'finished; resubmit to retry'),
                       finished_at = now()
                 WHERE status IN ('queued', 'running')
                   AND created_at < NOW() - INTERVAL '30 seconds'
                """
            )
            if cur.rowcount:
                logger.warning(
                    "Marked %d stranded training run(s) as failed. Each one "
                    "was blocking every future run with a 409.", cur.rowcount)
    except Exception as e:
        logger.error("Stranded training-run reaper failed: %s", e)
