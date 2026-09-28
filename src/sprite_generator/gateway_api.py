"""HTTP surface of the model gateway: status for both UIs, and the switch.

The rules live in model_gateway.py; the worker hooks in tasks.py. Spec:
.ai/specs/model-gateway/plan.md.
"""

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

import auth
import core_models
import model_gateway

router = APIRouter()


class SwitchRequest(BaseModel):
    model: str
    force: bool = False


def _choices() -> list[dict]:
    """What the switcher offers: the step-1 roster plus the something2 list,
    and the fixed-model labels so a force switch can hand the card to them."""
    import a1111
    seen, out = set(), []
    for e in core_models.roster():
        seen.add(e["value"])
        out.append({"value": e["value"], "label": e["label"],
                    "available": e.get("available", True)})
    for m in a1111.KNOWN_MODELS:
        if m not in seen:
            seen.add(m)
            out.append({"value": m, "label": m,
                        "available": not core_models.unavailable_reason(m)})
    for m in model_gateway.FIXED_LABELS:
        out.append({"value": m, "label": m, "available": True, "fixed": True})
    return out


@router.get("/api/model-gateway")
def gateway_status(authorization: str | None = Header(None)):
    auth.require(authorization, "read")
    return {**model_gateway.status(), "choices": _choices()}


@router.post("/api/model-gateway/switch")
def gateway_switch(req: SwitchRequest,
                   authorization: str | None = Header(None)):
    """Make `model` the active, pinned model.

    Normal: refused with 409 while ANY job is queued, running or deferred -
    "switch only when nothing is pending". Force: always; jobs for the old
    model defer and run after the new one has been idle MODEL_SWITCH_IDLE_S.
    A job already running is never interrupted; the switch waits for it.
    """
    auth.require(authorization, "generate")
    model = req.model.strip()
    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    if model not in model_gateway.FIXED_LABELS:
        reason = core_models.unavailable_reason(model)
        if reason:
            raise HTTPException(status_code=400,
                                detail=f"'{model}' is not available: {reason}")

    active = model_gateway.get_active()
    pending = model_gateway.get_pending()
    if active and active.get("model") == model:
        model_gateway.set_active(model, pinned=True)
        return {"status": "already_active", **model_gateway.status()}
    if pending and not req.force:
        by = {}
        for p in pending.values():
            by[p.get("model")] = by.get(p.get("model"), 0) + 1
        summary = ", ".join(f"{n} for {m}" for m, n in by.items())
        raise HTTPException(status_code=409, detail=(
            f"{len(pending)} job(s) pending ({summary}). A normal switch waits "
            f"for an empty queue; use force to switch now and defer them."))

    running = any(p.get("running") for p in pending.values())
    model_gateway.set_active(model, pinned=True)
    model_gateway.begin_switch(model, "waiting" if running else "queued")
    from tasks import model_switch_task
    model_switch_task.delay(model)
    return {"status": "switching", "forced": req.force,
            "deferred": sum(1 for p in pending.values()
                            if p.get("model") != model),
            **model_gateway.status()}
