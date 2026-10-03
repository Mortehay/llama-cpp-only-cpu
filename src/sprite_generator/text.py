"""`/api/text` and `/api/text/models` - the brain as a text provider.

Contract: .ai/specs/something2-text/contract.md. Decision: .ai/decisions/0013.

SYNCHRONOUS, REFUSE-WHEN-BUSY. something2's provider client makes one attempt
with a 5-minute budget and falls back on 409/503, so a busy card is answered
at once rather than queued: waiting behind a long job would only spend the
budget and then fall back anyway. 422 is reserved for requests that can never
succeed as sent - the client treats it as terminal.

`run_text` is also what the in-process callers (worlds, audio_styles) use, so
every brain call goes through the same gate and the same ledger.
"""

from __future__ import annotations

import logging
import os
import time

from fastapi import APIRouter, Header, HTTPException, Request

import auth
import brain_engine
import generations
import model_gateway

logger = logging.getLogger(__name__)

router = APIRouter()

# Below something2's 300 s client budget, above a cold load (~37 s for the 8B)
# plus a long answer. On overshoot the task is NOT revoked: it finishes and
# closes its own ledger row.
WAIT_S = float(os.environ.get("TEXT_GENERATE_TIMEOUT_S", "240"))
MAX_PROMPT_CHARS = int(os.environ.get("TEXT_MAX_PROMPT_CHARS", "24000"))

STATUS_OF_KIND = {"unsupported_schema": 422, "truncated": 422,
                  "unavailable": 503, "gpu_faulted": 503,
                  "model_deferred": 503, "failed": 502}


class TextRefused(Exception):
    """A request the brain did not answer. `status` follows the contract."""

    def __init__(self, status: int, reason: str, detail: str,
                 retry_after_s: int | None = None):
        super().__init__(detail)
        self.status, self.reason, self.detail = status, reason, detail
        self.retry_after_s = retry_after_s

    def http(self) -> HTTPException:
        headers = ({"Retry-After": str(int(self.retry_after_s))}
                   if self.retry_after_s else None)
        return HTTPException(status_code=self.status, headers=headers,
                             detail={"reason": self.reason, "detail": self.detail,
                                     "retry_after_s": self.retry_after_s})


# --- Validation: pure, so it is testable without the stack ------------------

_BOUNDS = (("minLength", "maxLength"), ("minItems", "maxItems"),
           ("minimum", "maximum"), ("minProperties", "maxProperties"))
_PY_TYPES = {"string": str, "integer": int, "number": (int, float),
             "boolean": bool, "object": dict, "array": list}


def schema_problems(schema, path: str = "$") -> list[str]:
    """Why this schema can never be satisfied, or [] when it can.

    Not a full satisfiability check - the cases a caller actually writes:
    an empty enum, an enum value of the wrong type, min > max, a required
    property that `additionalProperties: false` forbids.
    """
    out: list[str] = []
    if not isinstance(schema, dict):
        return [f"{path}: a schema must be an object"]
    enum = schema.get("enum")
    if enum is not None:
        if not isinstance(enum, list) or not enum:
            out.append(f"{path}.enum is empty - no value can match")
        else:
            t = schema.get("type")
            py = _PY_TYPES.get(t) if isinstance(t, str) else None
            if py and not any(isinstance(v, py) and not (t != "boolean" and isinstance(v, bool))
                              for v in enum):
                out.append(f"{path}.enum has no value of type {t}")
    for lo, hi in _BOUNDS:
        a, b = schema.get(lo), schema.get(hi)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a > b:
            out.append(f"{path}: {lo} {a} > {hi} {b}")
    props = schema.get("properties")
    props = props if isinstance(props, dict) else {}
    if schema.get("additionalProperties") is False:
        for r in schema.get("required") or []:
            if r not in props:
                out.append(f"{path}: '{r}' is required but additionalProperties "
                           f"is false and it is not in properties")
    for k, sub in props.items():
        out += schema_problems(sub, f"{path}.properties.{k}")
    items = schema.get("items")
    if isinstance(items, dict):
        out += schema_problems(items, f"{path}.items")
    for key in ("anyOf", "oneOf", "allOf"):
        for i, sub in enumerate(schema.get(key) or []):
            out += schema_problems(sub, f"{path}.{key}[{i}]")
    return out


def validate_schema(schema) -> None:
    """422 unless the schema is a valid, satisfiable JSON Schema."""
    import jsonschema
    if not isinstance(schema, dict):
        raise TextRefused(422, "invalid_schema", "schema must be a JSON object")
    try:
        jsonschema.validators.validator_for(schema).check_schema(schema)
    except jsonschema.SchemaError as e:
        raise TextRefused(422, "invalid_schema", f"not a valid JSON Schema: {e.message}")
    problems = schema_problems(schema)
    if problems:
        raise TextRefused(422, "impossible_schema",
                          "the schema can never be satisfied: " + "; ".join(problems))


# --- The one path every brain call takes -------------------------------------

def run_text(*, prompt: str, system: str | None = None, schema: dict | None = None,
             temperature: float = 0.7, max_tokens: int = 512,
             brain: str | None = None, caller: dict | None = None,
             route: str = "/api/text", wait_s: float = WAIT_S) -> dict:
    """Gate, enqueue, wait, check. Returns the task's result dict
    ({text, json?, usage, timings, finish_reason, model}). Raises TextRefused.
    """
    from tasks import celery_app, worker_busy_reason

    started = time.time()
    brain = brain or brain_engine.default_brain()
    if brain not in brain_engine.BRAINS:
        raise TextRefused(422, "unknown_model", (
            f"unknown model {brain!r}; one of {', '.join(brain_engine.BRAINS)}"))
    if not (prompt or "").strip():
        raise TextRefused(422, "invalid_request", "prompt is required")
    if len(prompt) + len(system or "") > MAX_PROMPT_CHARS:
        raise TextRefused(422, "invalid_request",
                          f"prompt + system over {MAX_PROMPT_CHARS} characters")
    if not (0 <= temperature <= 2):
        raise TextRefused(422, "invalid_request", "temperature must be 0-2")
    if not (1 <= max_tokens <= 4096):
        raise TextRefused(422, "invalid_request", "max_tokens must be 1-4096")
    if schema is not None:
        validate_schema(schema)

    label = brain_engine.label(brain)
    params = {"system": system, "schema": schema, "temperature": temperature,
              "max_tokens": max_tokens}

    def refuse(status, reason, detail, retry_after_s=None):
        generations.record(kind="text", status="failed", route=route,
                           served_from="generated", prompt=prompt, model=brain,
                           error=detail, params=params, caller=caller,
                           duration_ms=(time.time() - started) * 1000)
        raise TextRefused(status, reason, detail, retry_after_s)

    switching = model_gateway.get_switching()
    if switching and switching.get("model") != label:
        refuse(409, "switching", f"the card is switching to {switching.get('model')}",
               30)
    busy = worker_busy_reason()
    if busy:
        refuse(503, "busy", busy["detail"], busy.get("retry_after_s", 60))
    gate = model_gateway.admit_sync(label)
    if gate:
        refuse(503, "busy", gate["detail"], gate["retry_after_s"])

    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    gen = generations.begin(kind="text", route=route, prompt=prompt, model=brain,
                            params=params, caller=caller)
    task = celery_app.send_task(
        "tasks.generate_text_task",
        kwargs={"brain": brain, "messages": messages, "schema": schema,
                "temperature": temperature, "max_tokens": max_tokens,
                "gen_id": gen})
    generations.attach_task(gen, task.id)
    try:
        result = celery_app.AsyncResult(task.id).get(timeout=wait_s)
    except Exception:
        # Deliberately no revoke: the task finishes and closes its own row.
        raise TextRefused(503, "building", (
            f"no answer after {time.time() - started:.0f}s; not cancelled"), 60)

    if not result or result.get("error"):
        kind = (result or {}).get("error_kind", "failed")
        raise TextRefused(STATUS_OF_KIND.get(kind, 502), kind,
                          (result or {}).get("error", "the brain returned nothing"),
                          (result or {}).get("retry_after_s")
                          or (60 if STATUS_OF_KIND.get(kind) == 503 else None))
    if schema is not None:
        import jsonschema
        try:
            jsonschema.validate(result.get("json"), schema)
        except jsonschema.ValidationError as e:
            # The grammar should make this impossible; if it happens it is the
            # brain's fault, not the caller's - a 5xx, so the client falls back.
            raise TextRefused(502, "schema_mismatch",
                              f"the answer does not match the schema: {e.message}")
    return result


# --- HTTP -------------------------------------------------------------------

@router.get("/api/text/models")
def text_models(authorization: str | None = Header(default=None)):
    """Static roster; never loads a brain (something2's discovery budget is 10 s)."""
    auth.require(authorization, "read")
    return {"data": brain_engine.roster()}


@router.post("/api/text")
async def text(request: Request, authorization: str | None = Header(default=None)):
    principal = auth.require(authorization, "generate")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=422, detail="body must be JSON")
    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="body must be a JSON object")
    # Raw JSON rather than a pydantic model: a field named `schema` shadows
    # BaseModel.schema, and the contract's field name is not negotiable.
    try:
        prompt = body.get("prompt")
        system = body.get("system")
        if not isinstance(prompt, str) or (system is not None and not isinstance(system, str)):
            raise ValueError("prompt and system must be strings")
        temperature = float(body.get("temperature", 0.7))
        max_tokens = int(body.get("max_tokens", 512))
        brain = body.get("model")
        if brain is not None and not isinstance(brain, str):
            raise ValueError("model must be a string")
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=422, detail=f"invalid request: {e}")
    caller = generations.describe_caller(principal, request)
    from starlette.concurrency import run_in_threadpool
    try:
        res = await run_in_threadpool(
            run_text, prompt=prompt, system=system, schema=body.get("schema"),
            temperature=temperature, max_tokens=max_tokens, brain=brain,
            caller=caller)
    except TextRefused as e:
        raise e.http()
    out = {"text": res.get("text", ""), "model": res.get("model"),
           "usage": res.get("usage"), "timings": res.get("timings")}
    if "json" in res:
        out["json"] = res["json"]
    return out
