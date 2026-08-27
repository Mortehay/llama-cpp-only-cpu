"""Operator commands, over HTTP, without a shell.

    GET  /api/commands              -> the allowlist, plus queue state
    POST /api/commands/{name}       -> 202 {"task_id": ...}
    GET  /api/commands/{task_id}    -> status and the tail of the output

Every command here was already a Makefile target, which meant it needed a
terminal, a checkout and Docker to run - so the audit that explains why a
reference was excluded could only be produced by someone with all three.

(An earlier version of this note claimed the UI had no way to SHOW those
reasons either. It does: `ReferenceTab` renders `trainable_why` in place of
`why` whenever `trainable` is false, and has all along. The claim came from a
grep whose output was cut off by `head` at the line above the one that matters.
Recorded because the gap this module actually fills is running the command, not
displaying its result, and overstating it would misdescribe the whole file.)

WHAT THIS ENDPOINT DOES NOT DO

It does not take a command line. `commands.py` holds the argv; the client sends
a key. This matters more than usual here because `auth.is_enforced()` is false
until somebody sets a token, so in the default configuration these endpoints
are reachable without credentials. An endpoint that shelled out would be a
remote shell on an unauthenticated port.

THE WRITE IS GATED DIFFERENTLY FROM THE REST

`audit-refs-apply` sets `trainable = false` on live rows. Its failure mode is
not an error - it is a quietly smaller dataset discovered weeks later as a
training run that used less than someone expected. So it needs `confirm=true`
in the body, and the listing tells the UI how many rows are at stake BEFORE the
button is pressed. The confirmation is not a modal for its own sake; it exists
so the rowcount has somewhere to be shown.
"""

import logging
import os

import psycopg2
from celery.result import AsyncResult
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

import auth
import commands as command_table

logger = logging.getLogger(__name__)
router = APIRouter()

DB_URL = os.environ.get("DB_URL")


class RunRequest(BaseModel):
    # Only meaningful for a command whose `writes` is "database". Ignored
    # elsewhere rather than rejected, so the UI can send one shape.
    confirm: bool = False


def _celery():
    """The Celery app, imported late.

    `tasks` pulls torch and diffusers. Importing it at module scope would make
    the API process load the whole GPU stack to serve a listing - which is the
    exact cost `jobs.py` avoids by keeping `actions.py` stdlib-only.
    """
    from tasks import celery_app
    return celery_app


def _rows_at_stake() -> dict:
    """How many live references the write could touch, and how many it has.

    Shown beside the confirm button. "573 trainable references" is a number
    somebody can weigh; "are you sure?" is not.
    """
    if not DB_URL:
        return {"trainable": None, "with_reason": None, "why": "no DB_URL"}
    try:
        with psycopg2.connect(DB_URL, connect_timeout=5) as conn, \
                conn.cursor() as cur:
            cur.execute("SELECT count(*) FILTER (WHERE trainable), "
                        "       count(*) FILTER (WHERE trainable_why IS NOT NULL) "
                        "FROM reference_assets WHERE deleted = false")
            trainable, with_reason = cur.fetchone()
        return {"trainable": int(trainable), "with_reason": int(with_reason)}
    except Exception as e:                       # noqa: BLE001 - reported, not raised
        # A listing must not 500 because the database is down. The UI can
        # still show the commands; it just cannot show the stakes, and saying
        # so is better than a blank panel or a fabricated zero.
        logger.warning("could not count rows at stake: %s", e)
        return {"trainable": None, "with_reason": None, "why": str(e)}


@router.get("/api/commands")
def list_commands(authorization: str | None = Header(None)):
    auth.require(authorization, "read")
    return {
        "commands": command_table.listing(),
        "groups": command_table.GROUPS,
        "stakes": _rows_at_stake(),
        # The worker is --concurrency=1 and shared with generation, so a
        # five-minute audit delays the next sheet by five minutes. Stated here
        # rather than left for someone to discover from a slow queue.
        "shares_worker": True,
    }


@router.post("/api/commands/{name}", status_code=202)
def run_command(name: str, body: RunRequest | None = None,
                authorization: str | None = Header(None)):
    auth.require(authorization, "write")

    spec = command_table.COMMANDS.get(name)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"no such command: {name}")

    missing = command_table.missing_requirements(name)
    if missing:
        raise HTTPException(
            status_code=409,
            detail=(f"'{spec['label']}' needs {', '.join(missing)}, which is "
                    f"not installed in this image. Queueing it would produce a "
                    f"job that fails on a missing package rather than a real "
                    f"result."))

    if spec["writes"] == "database" and not (body and body.confirm):
        stakes = _rows_at_stake()
        n = stakes.get("trainable")
        raise HTTPException(
            status_code=400,
            detail=(f"'{spec['label']}' writes to live reference rows"
                    + (f" ({n} currently trainable)" if n is not None else "")
                    + ". Send confirm: true to proceed."))

    async_result = _celery().send_task("tasks.run_command_job", args=[name])
    logger.info("queued command %s as %s", name, async_result.id)
    return {"task_id": async_result.id, "name": name,
            "writes": spec["writes"], "status": "queued"}


@router.get("/api/commands/{task_id}")
def command_status(task_id: str, authorization: str | None = Header(None)):
    auth.require(authorization, "read")
    res = AsyncResult(task_id, app=_celery())

    # PROGRESS carries the running tail; SUCCESS carries the final result. They
    # are different shapes and the UI should not have to know which it has, so
    # both are flattened to `lines` here.
    info = res.info if isinstance(res.info, dict) else {}
    done = res.status == "SUCCESS"
    payload = res.result if done and isinstance(res.result, dict) else {}

    return {
        "task_id": task_id,
        "status": res.status,
        "name": payload.get("name") or info.get("name"),
        "lines": payload.get("lines") or info.get("lines") or [],
        "message": info.get("msg"),
        "exit_code": payload.get("exit_code"),
        # FAILURE means the task itself blew up. A non-zero exit code is NOT a
        # failure here - a red test suite is a successful run of a test suite,
        # and collapsing the two would hide which one happened.
        "crashed": res.status == "FAILURE",
        "error": payload.get("error") or (str(res.info) if res.status == "FAILURE" else None),
    }
