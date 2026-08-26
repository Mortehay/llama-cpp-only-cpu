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

import auth
import core_models
from tasks import celery_app, generate_raw_task

logger = logging.getLogger(__name__)
router = APIRouter()

# Their AI_PROVIDER_GENERATE_TIMEOUT_MS defaults to 5 minutes. Stay under it so
# the failure surfaces as our error message, not their opaque timeout.
GENERATE_TIMEOUT_S = int(os.environ.get("A1111_GENERATE_TIMEOUT_S", "240"))

# The legacy shared secret is no longer read here. `auth.py` owns it - it still
# honours SPRITE_API_TOKEN as a valid credential, but an UNSET one no longer
# means "no auth". Deleting the module-level constant is the point: while it
# existed, the check above could be re-broken by anyone who reinstated the
# `if not API_TOKEN: return` shortcut without realising what it disabled.

# text2img models this service will serve. `model_name` is the value clients
# send back in override_settings.sd_model_checkpoint.
#
# Family detection now reads model_index.json (_is_sdxl_checkpoint), so a repo
# no longer has to be NAMED "sdxl" to be loaded as one. That constraint used to
# rule out stable-diffusion-xl-base-1.0 entirely. See .ai/decisions/0002.
KNOWN_MODELS = [
    # Non-distilled. Turbo and friends run at guidance 0, so the negative_prompt
    # something2 sends is a silent no-op; these honour it at 20-30 steps.
    # Measured 2026-08-21, same prompt and seed across all four - see
    # .ai/decisions/0002. Best real pixel art of the set, 3.0s at 20 steps:
    "PublicPrompts/All-In-One-Pixel-Model",
    # Full SDXL. Non-distilled, native 1024px, and now loadable because family
    # detection reads the config rather than the name. Pairs with
    # thibaud/controlnet-openpose-sdxl-1.0 for pose-conditioned step 2.
    "stabilityai/stable-diffusion-xl-base-1.0",
    # "<base>+<lora>" fuses a style LoRA onto the base - see get_sd_pipeline.
    # Measured 2026-08-21: this is the first configuration to produce
    # structurally real pixel art. 86.6% of pixels drawn from 32 colours and
    # 74.7% blockiness, against 36.9%/25.2% for sdxl-turbo. Needs "pixel art"
    # in the prompt to trigger. 21s at 25 steps, 1024px.
    "stabilityai/stable-diffusion-xl-base-1.0+nerijs/pixel-art-xl",
    # Two more pixel LoRAs on the SAME base - a different style each, for 0.32GB
    # and 0.08GB. Swapping LoRAs costs a pipeline reload but no extra base
    # download and no extra VRAM: fuse_lora folds the delta into the base
    # weights rather than keeping a second set of tensors alive. Both declare
    # stable-diffusion-xl-base-1.0 as their base_model.
    # UNMEASURED - only nerijs/pixel-art-xl has been benchmarked.
    "stabilityai/stable-diffusion-xl-base-1.0+Muapi/soft-pixel-art-xl",
    "stabilityai/stable-diffusion-xl-base-1.0+ntc-ai/SDXL-LoRA-slider.pixel-art",
    # NOT LISTED: Limbicnation/pixel-art-lora declares base_model
    # black-forest-labs/FLUX.2-klein-4B. A LoRA's ranks are tied to the UNet it
    # was trained against, so it cannot fuse onto SDXL - load_lora_weights
    # raises and get_sd_pipeline degrades to the bare base with a warning.
    # NOT SERVED: "John6666/super-pixelart-xl-m-v1-v10-sdxl" loads without error
    # and returns pure RGB noise - undenoised latents, 25 steps at cfg 7, 1024px.
    # Not the fp16-VAE black-image failure documented in the README; something
    # about the checkpoint's scheduler/prediction config does not survive this
    # diffusers version. Left out rather than handing something2 a model that
    # fails silently with a 200.
]


def _require_auth(authorization: str | None, scope: str = "generate"):
    """Authorise a facade call through the shared key system.

    THIS USED TO BE ITS OWN TOKEN CHECK, AND IT BEGAN `if not API_TOKEN: return`.

    That is the exact silent-open bug `auth.py` was written to replace, and it
    survived here longer than anywhere else - which was the worst possible place
    for it. This facade is the surface something2 calls, so it is the one most
    likely to be reachable from another machine, and `SPRITE_API_TOKEN` is empty
    in `.env.example`. A fresh install therefore published unauthenticated
    txt2img to the LAN while the settings UI could truthfully report that keys
    existed and the API was secured.

    `auth.require` honours the legacy `SPRITE_API_TOKEN` too, so an admin who
    already configured one keeps working. The difference is that an UNSET token
    no longer means "let everyone in" - it means "fall back to whether any key
    exists".

    Scope defaults to `generate` because the endpoint that matters here queues
    GPU work. Discovery passes `read`, so a read-only key can answer "what
    models do you have?" without being able to spend the card - which is what
    something2's reachability check needs and nothing more.
    """
    auth.require(authorization, scope)


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

    # Return an RGBA cutout instead of an opaque square.
    #
    # EXPLICIT, never inferred from the prompt. something2 asked for it this way
    # and they were right: their prompts do say "solid transparent background",
    # but making that phrase load-bearing means a copy edit silently turns every
    # entity into an opaque block. Entity images composite over terrain; tiles
    # fill their diamond and are SUPPOSED to be opaque. That is a per-request
    # decision the caller owns.
    cutout: bool = False

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
    _require_auth(authorization, "read")

    # Stock checkpoints, then adapters trained on this machine.
    #
    # The trained ones were missing entirely, so a model trained here could
    # never be selected by the game - `/api/core-models` listed them and this
    # endpoint, the one something2 actually discovers against, did not. Any
    # adapter is useless to the client until it appears here.
    #
    # Only AVAILABLE trained adapters: an entry whose .safetensors has been
    # deleted would fail at generation time, minutes later and opaquely, which
    # is worse than not offering it.
    trained = [e["value"] for e in core_models.local_roster()
               if core_models.unavailable_reason(e["value"]) is None]

    out = []
    for name in KNOWN_MODELS + trained:
        trigger = core_models.trigger_for(name)
        out.append({
            "title": name,
            "model_name": name,
            "hash": None,
            "sha256": None,
            "filename": name,
            "config": None,
            # Extra, non-A1111 fields. Harmless to a `$[*].model_name` pointer,
            # and they let a client show which entries are local and what token
            # they respond to - though it does NOT need to send the trigger,
            # because txt2img injects it. See apply_trigger below.
            "trained": bool(trigger),
            "trigger": trigger,
        })
    return out


@router.post("/sdapi/v1/txt2img")
def txt2img(req: Txt2ImgRequest, authorization: str | None = Header(default=None)):
    """Blocking text2img. Returns base64 PNG at `images[0]`, as A1111 does."""
    _require_auth(authorization)

    model = req.override_settings.get("sd_model_checkpoint") or KNOWN_MODELS[0]

    # Inject the adapter's trigger token server-side.
    #
    # A trained LoRA is INERT without its trigger: it loads, fuses, and returns
    # plain base-model output with nothing saying why. Making each client carry
    # a table of triggers is a design that leaks - something2's prompts come
    # from its own tile and entity rows and have no business knowing what this
    # machine last trained. Idempotent, so a caller that does send the trigger
    # is not penalised with a doubled token.
    # Optional, non-A1111: how strongly to fold the LoRA in. Clients that want
    # less of an over-trained adapter can send override_settings.lora_scale.
    # Accept the flag at the top level or inside override_settings: their
    # template system substitutes into a fixed body shape, and which of the two
    # is reachable depends on the template.
    cutout = bool(req.cutout or req.override_settings.get("cutout"))

    raw_scale = req.override_settings.get("lora_scale")
    try:
        lora_scale = float(raw_scale) if raw_scale not in (None, "") else None
    except (TypeError, ValueError):
        lora_scale = None

    prompt = core_models.apply_trigger(model, req.prompt)
    if prompt != req.prompt:
        logger.info("txt2img: injected trigger for %s", model)

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
        prompt,
        req.negative_prompt,
        model,
        width,
        height,
        steps,
        cfg,
        req.seed,
        cutout,
        lora_scale,
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
        # A failed cutout is the CALLER's request being unsatisfiable, not this
        # service breaking, and the distinction matters to a bulk runner: 422
        # means "this subject will not cut out, skip or reword it", 500 means
        # "retry later". Returning an opaque image instead would be worse than
        # either - it stores clean-looking data that is wrong.
        status = 422 if (result or {}).get("error_kind") == "cutout_failed" else 500
        raise HTTPException(status_code=status, detail=detail)

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
