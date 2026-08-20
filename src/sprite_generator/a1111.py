"""Automatic1111-compatible façade.

something2's admin panel reaches remote image providers through a generic
template system (its `docs/ai-providers.md`) and ships a worked example for
Automatic1111. Speaking that dialect means the integration needs **zero code
changes on their side** — an admin registers a provider pointing here and picks
the stock A1111 preset.

Three constraints from their side shape this module:

1.  **It must be synchronous.** They explicitly do not support submit/poll
    queues (their SOMET-334) and default to a 5 minute timeout. Our pipeline is
    Celery-based and asynchronous, so every route here blocks on the task result
    and must return inside that window. GENERATE_TIMEOUT_S is deliberately set
    below their default so we return a clean error rather than having them time
    out on us.

2.  **Their templates quote numbers.** The documented A1111 template sends
    `"width": "{{width}}"` — a JSON string, not an int. Pydantic's lax mode
    coerces most of these, but `_as_int` makes it explicit rather than
    incidental, and tolerates the empty string an unsubstituted placeholder
    leaves behind.

3.  **They read the image from `images[0]` as base64** and cap it at 32MB.

Deliberately NOT implemented: progress, interrupt, options, and the rest of the
A1111 surface. something2 uses exactly two endpoints, and stubbing more would
invite clients to depend on behaviour we do not have.
"""

import base64
import json
import os
import time
import logging

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

from tasks import celery_app, generate_raw_task

logger = logging.getLogger(__name__)
router = APIRouter()

# Their AI_PROVIDER_GENERATE_TIMEOUT_MS defaults to 5 minutes. Stay under it so
# the failure surfaces as our error message, not their opaque timeout.
GENERATE_TIMEOUT_S = int(os.environ.get("A1111_GENERATE_TIMEOUT_S", "240"))

# Optional shared secret. something2 sends it as a configurable auth header.
# Unset means no auth, which is only acceptable on a trusted LAN.
API_TOKEN = os.environ.get("SPRITE_API_TOKEN", "").strip()

# text2img models this service will serve. `model_name` is the value clients
# send back in override_settings.sd_model_checkpoint.
#
# Every entry must satisfy the `is_sdxl` NAME heuristic in get_sd_pipeline: the
# pipeline class is chosen from whether the repo name contains "sdxl"/"turbo",
# not from model_index.json. An SDXL repo named otherwise (e.g.
# "…/pixel-art-diffusion-xl") is handed an SD1.5 pipeline class and fails to
# load. See .ai/decisions/0002.
KNOWN_MODELS = [
    "stabilityai/sdxl-turbo",
    "Onodofthenorth/SD_PixelArt_SpriteSheet_Generator",
    # Non-distilled. Turbo and friends run at guidance 0, so the negative_prompt
    # something2 sends is a silent no-op; these honour it at 20-30 steps.
    "John6666/super-pixelart-xl-m-v1-v10-sdxl",
    "PublicPrompts/All-In-One-Pixel-Model",
    "kohbanye/pixel-art-style",
]


def _require_auth(authorization: str | None):
    if not API_TOKEN:
        return
    expected = f"Bearer {API_TOKEN}"
    # Their docs note the token is stored in plaintext and that any admin who can
    # register a provider can point the backend at arbitrary hosts. A shared
    # secret here at least stops unauthenticated LAN clients driving the GPU.
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


# Per-field fallbacks for when a {{placeholder}} arrives unsubstituted as "" or
# null. Kept at module scope rather than on the model: Pydantic v2 rejects
# non-annotated class attributes on a BaseModel.
_INT_DEFAULTS = {"steps": 20, "width": 512, "height": 512, "seed": -1, "frames": 1}


def _as_int(value, default: int) -> int:
    """Coerce A1111-template values that arrive as strings ("512", "", None)."""
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


class Txt2ImgRequest(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    steps: int = 20
    cfg_scale: float = 7.0
    width: int = 512
    height: int = 512
    # A1111 uses -1 for "random". Their template forwards {{seed}} verbatim.
    seed: int = -1
    override_settings: dict = Field(default_factory=dict)
    # Not an A1111 field. something2 templates may substitute {{frames}}; we
    # accept it so a sheet request does not 422, and widen the canvas below.
    frames: int = 1

    @field_validator("steps", "width", "height", "seed", "frames", mode="before")
    @classmethod
    def _coerce_ints(cls, v, info):
        return _as_int(v, _INT_DEFAULTS.get(info.field_name, 0))

    @field_validator("cfg_scale", mode="before")
    @classmethod
    def _coerce_float(cls, v):
        if v is None or v == "":
            return 7.0
        try:
            return float(v)
        except (TypeError, ValueError):
            return 7.0


@router.get("/sdapi/v1/sd-models")
def sd_models(authorization: str | None = Header(default=None)):
    """Model discovery.

    something2's models pointer for A1111 is `$[*].model_name`, so the response
    must be a bare array of objects each carrying `model_name`.
    """
    _require_auth(authorization)
    return [
        {
            "title": name,
            "model_name": name,
            "hash": None,
            "sha256": None,
            "filename": name,
            "config": None,
        }
        for name in KNOWN_MODELS
    ]


@router.post("/sdapi/v1/txt2img")
def txt2img(req: Txt2ImgRequest, authorization: str | None = Header(default=None)):
    """Blocking text2img. Returns base64 PNG at `images[0]`, as A1111 does."""
    _require_auth(authorization)

    model = req.override_settings.get("sd_model_checkpoint") or KNOWN_MODELS[0]

    width = req.width or 512
    height = req.height or 512
    # A multi-frame request becomes one wide grid image; something2 slices it
    # itself using the columns/rows declared in its provider config. Their
    # constraint is that the sheet divides evenly, so widen by whole frames.
    frames = max(1, req.frames)
    if frames > 1:
        width = width * frames

    # Sampling params are NOT reconciled here. something2's stock A1111 template
    # sends 20 steps / cfg 7, which is wrong for a distilled checkpoint — but
    # that is true of every caller, not just this one, so the correction lives in
    # tasks.resolve_sampling_params where the sprite UI and raw API get it too.
    steps = max(1, req.steps)
    cfg = req.cfg_scale

    started = time.time()
    task = generate_raw_task.delay(
        req.prompt,
        req.negative_prompt,
        model,
        width,
        height,
        steps,
        cfg,
        req.seed,
    )

    try:
        result = task.get(timeout=GENERATE_TIMEOUT_S)
    except Exception as e:
        # Do not leave the worker grinding on output nobody will read.
        try:
            celery_app.control.revoke(task.id, terminate=True)
        except Exception:
            pass
        logger.error(f"txt2img task {task.id} did not complete: {e}")
        raise HTTPException(
            status_code=504,
            detail=f"Generation did not finish within {GENERATE_TIMEOUT_S}s: {e}",
        )

    if not result or result.get("error"):
        detail = (result or {}).get("error", "unknown generation failure")
        raise HTTPException(status_code=500, detail=detail)

    file_path = result.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=500, detail="Generation reported success but produced no file")

    with open(file_path, "rb") as fh:
        encoded = base64.b64encode(fh.read()).decode("ascii")

    elapsed_ms = (time.time() - started) * 1000
    logger.info(f"txt2img served {model} in {elapsed_ms:.0f}ms ({len(encoded)} b64 chars)")

    return {
        "images": [encoded],
        "parameters": req.model_dump(),
        # A1111 returns `info` as a JSON-encoded string; clients that parse it
        # expect a string, not an object.
        "info": json.dumps({
            "seed": result.get("seed"),
            "model": model,
            "width": width,
            "height": height,
            "steps": req.steps,
            "duration_ms": result.get("duration_ms"),
        }),
    }
