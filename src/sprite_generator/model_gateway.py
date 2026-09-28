"""Model gateway: one active model on the card, other jobs wait their turn.

Spec: .ai/specs/model-gateway/plan.md.

WHY. The worker is one solo process on a 12 GB card, and `get_sd_pipeline`
evicts everything on any cache miss. A queue that alternates models reloads on
every job - a minute or more each, for nothing. This module decides, per job,
whether it may run now, must wait, or may take the card by switching models.

THE RULES (`decide`, a pure function so they are testable without Redis):

  - A job for the ACTIVE model runs.
  - A job for another model waits while any other job for the active model is
    queued or running - the active model's queue drains first.
  - A PINNED model (switched to from the UI) also holds the card until it has
    been idle for MODEL_SWITCH_IDLE_S, so hand-clicking with gaps between
    requests does not let a deferred job sneak in and force two reloads.
  - An unpinned model is held for at most MODEL_SWITCH_IDLE_S against an older
    waiting job, so a steady stream for one model cannot starve the rest.
  - Among waiting jobs, the OLDEST one's model goes next. The others keep
    waiting, so two deferred models do not ping-pong.

Every GPU task is gated, including the fixed-model ones (sheets, edit, audio,
training), each under a label from `GATED`. No-GPU tasks (commands, probes,
device readouts) are not in `GATED` and are never held.

Safe to import from the API and the worker: Redis and Postgres only, no torch.
"""

from __future__ import annotations

import json
import logging
import os
import time

import redis as _redis

from core_models import persistent_qwen

logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
_r = _redis.from_url(REDIS_URL, decode_responses=True)

# How long a pinned model holds the card after its last job, and the cap on how
# long an unpinned model can hold it against an older waiting job. .env.
IDLE_S = int(os.environ.get("MODEL_SWITCH_IDLE_S", "300"))
# How often a deferred job re-asks. Each ask is a Celery retry of the same task
# id, so polling clients and `task.get()` see one task throughout.
RECHECK_S = int(os.environ.get("MODEL_GATEWAY_RECHECK_S", "5"))
# The popup's estimate for a model never measured on this box.
DEFAULT_SWITCH_S = 10
# A pending entry older than this belongs to a task that died without its
# postrun signal. Training runs for hours, so this is generous.
PENDING_MAX_AGE_S = 12 * 3600

ACTIVE_KEY = "gw:active"
SWITCHING_KEY = "gw:switching"
PENDING_KEY = "gw:pending"
LOAD_S_KEY = "gw:load_s"

# Labels for jobs whose model is fixed by the task, not chosen by the caller.
QWEN_EDIT = "qwen-image-edit"
FLUX_EDIT = "flux-kontext-edit"
AUDIO_MUSIC = "audio:ace-step"
AUDIO_STABLE = "audio:stable-audio"
TRAINING = "training:sdxl-lora"
FIXED_LABELS = (QWEN_EDIT, FLUX_EDIT, AUDIO_MUSIC, AUDIO_STABLE, TRAINING)

# Tasks whose caller blocks on the result (the something2 facades). A deferral
# there is returned as an error at once - holding it would only end in the
# caller's timeout with the GPU work thrown away.
SYNC_TASKS = {"tasks.generate_raw_task"}


def _arg(args, kwargs, index, name, default=None):
    if name in kwargs:
        return kwargs[name]
    return args[index] if len(args) > index else default


def _job_model(job_id) -> str:
    """`spec.llm_name` of a jobs row (tile, map, props), or the default."""
    import core_models
    try:
        import job_runner
        conn = job_runner._connect()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT spec->>'llm_name' FROM jobs WHERE id = %s",
                            (str(job_id),))
                row = cur.fetchone()
        finally:
            conn.close()
        if row and row[0]:
            return row[0]
    except Exception as e:
        logger.warning("gateway: could not read the model of job %s: %s",
                       job_id, e)
    return core_models.default_model()


# task name -> (args, kwargs) -> model label. Positions mirror the task
# signatures in tasks.py / map_tasks.py; a keyword wins when one is passed.
GATED = {
    "tasks.generate_core_task":
        lambda a, k: _arg(a, k, 1, "llm_name", "stabilityai/sdxl-turbo"),
    "tasks.generate_raw_task": lambda a, k: _arg(a, k, 2, "llm_name"),
    "tasks.generate_spritesheet_task": lambda a, k: _arg(a, k, 2, "llm_name"),
    "tasks.warm_model_task": lambda a, k: _arg(a, k, 0, "llm_name"),
    "tasks.build_tile_job": lambda a, k: _job_model(_arg(a, k, 0, "job_id")),
    "maps.build_map_job": lambda a, k: _job_model(_arg(a, k, 0, "job_id")),
    "maps.resolve_map_props":
        lambda a, k: _job_model(_arg(a, k, 0, "props_job_id")),
    "tasks.build_sheet_job": lambda a, k: QWEN_EDIT,
    "tasks.edit_image_task": lambda a, k: FLUX_EDIT,
    "tasks.train_lora_job": lambda a, k: TRAINING,
    "tasks.generate_audio_task":
        lambda a, k: (AUDIO_MUSIC if _arg(a, k, 0, "kind") == "music"
                      else AUDIO_STABLE),
    "tasks.generate_sfx_task": lambda a, k: AUDIO_STABLE,
}


def model_of(task_name: str, args, kwargs) -> str | None:
    fn = GATED.get(task_name)
    if not fn:
        return None
    try:
        return fn(tuple(args or ()), dict(kwargs or {})) or None
    except Exception as e:
        logger.warning("gateway: no model for %s: %s", task_name, e)
        return None


def preloadable(model: str) -> bool:
    """Is this a diffusers checkpoint the worker can load ahead of the job?
    GGUF runs in a subprocess per image and the fixed labels load inside their
    own tasks, so for those a switch only frees the card."""
    return bool(model) and model not in FIXED_LABELS and not model.startswith("gguf:")


def warmable(model: str) -> bool:
    """Does a switch TO this model do real loading worth timing and showing?
    preloadable() models load a diffusers pipeline; persistent_qwen() models
    start the long-lived Qwen process (tasks._qwen_server_ensure, ~85 s cold).
    Everything else only frees the card, so its switch is ~instant."""
    return preloadable(model) or persistent_qwen(model)


# --- The rules --------------------------------------------------------------

def decide(model: str, task_id: str | None, active: dict | None,
           pending: dict, now: float, idle_s: int = IDLE_S) -> tuple[str, str | None]:
    """("run" | "switch" | "defer", reason). Pure.

    `pending` maps task_id -> {"model", "queued_at"} and includes running jobs.
    `task_id` is the asking job (None for a facade asking before it queues).
    """
    if not active or not active.get("model"):
        return "switch", None
    current = active["model"]
    if model == current:
        return "run", None

    others = {t: p for t, p in pending.items() if t != task_id}
    for_current = [p for p in others.values() if p.get("model") == current]
    waiting = sorted((p for p in others.values() if p.get("model") != current),
                     key=lambda p: p.get("queued_at", now))
    mine_at = pending.get(task_id, {}).get("queued_at", now) if task_id else now
    idle_for = now - float(active.get("last_activity") or active.get("since") or now)

    if active.get("pinned"):
        if for_current:
            return "defer", (f"{len(for_current)} job(s) queued for the active "
                             f"model {current}")
        if idle_for < idle_s:
            return "defer", (f"{current} was selected in the UI and holds the "
                             f"card until {idle_s}s idle "
                             f"({int(idle_s - idle_for)}s left)")
    elif for_current:
        # The starvation cap: an unpinned model's stream yields to a job that
        # has waited IDLE_S, oldest first.
        oldest = min([mine_at] + [p.get("queued_at", now) for p in waiting])
        if not (now - oldest >= idle_s and mine_at <= oldest):
            return "defer", (f"{len(for_current)} job(s) queued for the active "
                             f"model {current}")

    if waiting and waiting[0].get("queued_at", now) < mine_at \
            and waiting[0].get("model") != model:
        return "defer", (f"an older job for {waiting[0].get('model')} goes "
                         f"first")
    return "switch", None


def retry_after_s(active: dict | None, now: float, idle_s: int = IDLE_S) -> int:
    if active and active.get("pinned"):
        idle_for = now - float(active.get("last_activity") or now)
        return max(RECHECK_S, int(idle_s - idle_for) + 1)
    return 60


# --- Redis shell ------------------------------------------------------------

def _load(key):
    try:
        raw = _r.get(key)
        return json.loads(raw) if raw else None
    except Exception as e:
        logger.warning("gateway: could not read %s: %s", key, e)
        return None


def get_active() -> dict | None:
    return _load(ACTIVE_KEY)


def get_switching() -> dict | None:
    return _load(SWITCHING_KEY)


def get_pending() -> dict:
    try:
        raw = _r.hgetall(PENDING_KEY)
    except Exception as e:
        logger.warning("gateway: could not read pending: %s", e)
        return {}
    now, out, stale = time.time(), {}, []
    for tid, val in raw.items():
        try:
            p = json.loads(val)
        except ValueError:
            stale.append(tid)
            continue
        if now - float(p.get("queued_at", now)) > PENDING_MAX_AGE_S:
            stale.append(tid)
            continue
        out[tid] = p
    if stale:
        try:
            _r.hdel(PENDING_KEY, *stale)
        except Exception:
            pass
    return out


def set_active(model: str, pinned: bool) -> dict:
    now = time.time()
    state = {"model": model, "pinned": bool(pinned), "since": now,
             "last_activity": now}
    _r.set(ACTIVE_KEY, json.dumps(state))
    return state


def touch(model: str) -> None:
    """Record activity on the active model; resets the pinned idle clock."""
    state = get_active()
    if state and state.get("model") == model:
        state["last_activity"] = time.time()
        _r.set(ACTIVE_KEY, json.dumps(state))


def register(task_id: str, task_name: str, model: str) -> None:
    """Called on publish. HSETNX: a Celery retry republishes the same id and
    must keep its original place in line."""
    try:
        _r.hsetnx(PENDING_KEY, task_id, json.dumps(
            {"model": model, "task": task_name, "queued_at": time.time()}))
    except Exception as e:
        logger.warning("gateway: could not register %s: %s", task_id, e)


def mark_deferred(task_id: str, reason: str | None) -> None:
    try:
        raw = _r.hget(PENDING_KEY, task_id)
        if raw:
            p = json.loads(raw)
            p["deferred"] = reason
            _r.hset(PENDING_KEY, task_id, json.dumps(p))
    except Exception:
        pass


def mark_running(task_id: str) -> None:
    try:
        raw = _r.hget(PENDING_KEY, task_id)
        if raw:
            p = json.loads(raw)
            p.pop("deferred", None)
            p["running"] = True
            _r.hset(PENDING_KEY, task_id, json.dumps(p))
    except Exception:
        pass


def unregister(task_id: str) -> None:
    try:
        _r.hdel(PENDING_KEY, task_id)
    except Exception:
        pass


def begin_switch(to_model: str, phase: str) -> dict:
    active = get_active()
    try:
        expected = float(_r.hget(LOAD_S_KEY, to_model) or DEFAULT_SWITCH_S)
    except Exception:
        expected = DEFAULT_SWITCH_S
    if not warmable(to_model):
        expected = min(expected, 2.0)
    state = {"from": (active or {}).get("model"), "to": to_model,
             "started": time.time(), "expected_s": round(expected, 1),
             "phase": phase}
    # TTL so a worker that dies mid-switch cannot leave the popup up forever.
    _r.setex(SWITCHING_KEY, 900, json.dumps(state))
    return state


def set_switch_phase(phase: str) -> None:
    state = get_switching()
    if state:
        state["phase"] = phase
        state["loading_started"] = time.time()
        _r.setex(SWITCHING_KEY, 900, json.dumps(state))


def end_switch(model: str, load_s: float | None) -> None:
    if load_s is not None and warmable(model):
        try:
            _r.hset(LOAD_S_KEY, model, round(load_s, 1))
        except Exception:
            pass
    try:
        _r.delete(SWITCHING_KEY)
    except Exception:
        pass


def admit_sync(model: str) -> dict | None:
    """For a blocking facade, BEFORE it queues: None to proceed, or
    {detail, retry_after_s} for an immediate 503."""
    active, now = get_active(), time.time()
    action, reason = decide(model, None, active, get_pending(), now)
    if action != "defer":
        return None
    return {"detail": (f"Model gateway: requested {model}, but {reason}. "
                       f"Active model: {(active or {}).get('model')}."),
            "retry_after_s": retry_after_s(active, now)}


def status() -> dict:
    """What both UIs poll: GET /api/model-gateway."""
    now = time.time()
    active, switching, pending = get_active(), get_switching(), get_pending()
    by_model: dict = {}
    for p in pending.values():
        m = by_model.setdefault(p.get("model"), {"queued": 0, "running": 0,
                                                  "deferred": 0})
        if p.get("running"):
            m["running"] += 1
        elif p.get("deferred"):
            m["deferred"] += 1
        else:
            m["queued"] += 1
    if active:
        idle_for = now - float(active.get("last_activity") or now)
        active = {**active, "idle_for_s": int(idle_for),
                  "pin_release_in_s": (max(0, int(IDLE_S - idle_for))
                                       if active.get("pinned") else None)}
    if switching:
        switching = {**switching,
                     "elapsed_s": round(now - float(switching["started"]), 1)}
    return {
        "active": active,
        "switching": switching,
        "pending": sorted(({"task_id": t, **p} for t, p in pending.items()),
                          key=lambda p: p.get("queued_at", 0)),
        "by_model": by_model,
        "idle_s": IDLE_S,
        "recheck_s": RECHECK_S,
    }
