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


# How a caller asks for a map that already exists rather than a new image.
#
# THE NAME TRAVELS IN THE PROMPT, and that is not the same thing as keying the
# facade on the prompt. `plan.md` Q6 rejects prompt-KEYING - matching whatever
# text a caller happens to send against whatever maps happen to exist - because
# it is fuzzy and breaks when either side rewords. An explicit `map:` prefix is
# the opposite: unambiguous, impossible to hit by accident, and it fails loudly
# rather than silently returning the wrong map.
#
# It has to be the prompt because the contract promises ZERO laptop-side code.
# something2's connector substitutes into a fixed body shape, so the prompt is
# the only field guaranteed to carry an arbitrary string. `override_settings`
# is accepted too, for callers that can reach it - the same two-channel shape
# `cutout` and `lora_scale` already use, and for the same reason.
MAP_PREFIX = "map:"

# The same two-channel addressing for tiles, with one decisive difference in
# BEHAVIOUR: `map:` is a cache reader that never queues, `tile:` will build.
#
# That asymmetry is not an inconsistency, it is the measurement. A map build is
# minutes to hours and no HTTP budget survives it, so offering to build one
# synchronously would only convert a clear 404 into an opaque timeout. A tile is
# one small image: nine finished tile jobs on this hardware ran 21s to 125s,
# against something2's 300s default and our 240s ceiling. Refusing to build
# inside that headroom would force an operator to hand-make every tile on this
# machine before something2 could ask for it - which is most of the value gone.
#
# Addressed as `tile:<name> <the rest of the prompt>` - see `_split_tile_prompt`
# for why the name has to share the prompt field.
TILE_PREFIX = "tile:"

# What a MISSING tile is allowed to spend. Defaults to the same ceiling as any
# other generation, and is separable because the two answer different questions:
# GENERATE_TIMEOUT_S is "how long may one image take", this is "how long may we
# make something2 wait for ground it did not know was absent".
TILE_BUILD_BUDGET_S = int(os.environ.get("A1111_TILE_BUILD_BUDGET_S",
                                         str(GENERATE_TIMEOUT_S)))


def _map_request(req: "Txt2ImgRequest") -> str | None:
    """The map name this request is asking for, or None for a normal generate."""
    explicit = req.override_settings.get("map")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    prompt = (req.prompt or "").strip()
    if prompt.lower().startswith(MAP_PREFIX):
        return prompt[len(MAP_PREFIX):].strip() or None
    return None


def _serve_map(name: str, req: "Txt2ImgRequest", started: float) -> dict:
    """An already-built map picture, in the A1111 response shape.

    A CACHE READ. It never queues anything: a map build is minutes to hours and
    no HTTP timeout survives that, so a name that has not been built is a 404
    telling the caller to build it - not a request that hangs and then fails.

    `info` carries the map's real identity and its provisional state, because a
    consumer that caches `images[0]` needs to know whether the picture still has
    magenta placeholders on it. That is the one thing this response can say that
    a generated image never has to.
    """
    import maps

    row = maps.resolve_name(name)
    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"no finished map named {name!r}. Maps are authored on this "
                   f"service and collected here - build it first, then ask "
                   f"again. GET /api/maps lists what exists.")

    picture = row.get("sheet_path")
    if not picture or not os.path.exists(picture):
        raise HTTPException(
            status_code=409,
            detail=f"map {name!r} is finished but its picture is missing from "
                   f"disk; rebuild it")

    with open(picture, "rb") as fh:
        encoded = base64.b64encode(fh.read()).decode("ascii")

    complete, pending = True, []
    try:
        with open(row["atlas_path"], "r", encoding="utf-8") as fh:
            tilemap = json.load(fh)
        complete = bool(tilemap.get("complete", True))
        pending = tilemap.get("pending") or []
    except Exception as e:
        # The picture is the artefact and it is already read. Failing the whole
        # request because the sidecar would not parse would withhold something
        # that is fine.
        logger.warning("map %s: could not read tilemap for status: %s", name, e)

    elapsed_ms = (time.time() - started) * 1000
    logger.info("txt2img served MAP %r from cache in %.0fms (%d b64 chars, "
                "complete=%s)", name, elapsed_ms, len(encoded), complete)

    return {
        "images": [encoded],
        "parameters": req.model_dump(),
        "info": json.dumps({
            "map": name,
            "job_id": str(row["id"]),
            "cached": True,
            # NOT a generation. A caller measuring model performance off this
            # would be measuring a file read.
            "generated": False,
            # `false` means the picture still carries placeholder art. A
            # consumer that caches this as final keeps a magenta cross forever.
            "complete": complete,
            "pending": pending,
            "tilemap_url": f"/api/maps/by-name/{name}",
            "duration_ms": round(elapsed_ms),
        }),
    }


def _tile_request(req: "Txt2ImgRequest") -> str | None:
    """The tile name this request is asking for, or None for a normal generate."""
    explicit = req.override_settings.get("tile")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    return _split_tile_prompt(req.prompt)[0]


def _split_tile_prompt(raw: str | None) -> tuple[str | None, str | None]:
    """`tile:road_sand cracked red stone` -> ("road_sand", "cracked red stone").

    THE NAME IS THE FIRST TOKEN AND THE REST IS THE PROMPT, which is forced by
    something2's template system rather than chosen. Their request template
    substitutes `{{prompt}}`, `{{width}}`, `{{height}}`, `{{seed}}`, `{{frames}}`
    and `{{model}}` - and nothing that carries a tile's NAME. So the one field
    that can hold a name is the prompt, alongside the prompt.

    That makes configuring a tile an edit an operator can actually make: put
    `tile:rocks ` in front of the text already in something2's tile row and
    change nothing else. No new placeholder, no code on their side, and the
    prompt still reaches the model intact.

    A bare `tile:rocks` with no remaining text is valid - the name doubles as
    the prompt, which is what a one-word ground like "sand" wants anyway.
    """
    text = (raw or "").strip()
    if not text.lower().startswith(TILE_PREFIX):
        return None, None

    rest = text[len(TILE_PREFIX):].strip()
    if not rest:
        return None, None

    parts = rest.split(None, 1)
    name = parts[0]
    prompt = parts[1].strip() if len(parts) > 1 else ""
    return name, (prompt or name)


def _long_job_ahead() -> str | None:
    """A non-tile job already on the worker, described, or None.

    The Celery worker is --concurrency=1 and shared with sheet, map and training
    builds, so a queued tile waits behind whatever holds the card. A sheet is
    ~2 hours. Submitting a tile behind one and then blocking cannot succeed; it
    can only spend something2's whole budget and surface as their opaque
    timeout, with the real reason - "something else is using the GPU" - visible
    nowhere.

    So we look before we queue. This is advisory, not a lock: a long job can
    start in the gap between this check and the enqueue. That race costs a slow
    tile, not a wrong one, and paying for a real lock here would mean holding it
    across a two-minute GPU build.
    """
    try:
        import psycopg2
        import psycopg2.extras
        with psycopg2.connect(os.environ.get("DB_URL")) as conn, conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT kind, started_at FROM jobs "
                "WHERE status = 'running' AND kind <> 'tile' AND deleted = false "
                "ORDER BY started_at LIMIT 1")
            row = cur.fetchone()
    except Exception as e:
        # Never fail a tile because the ADVISORY check could not run. The build
        # below is the thing that matters and it has its own timeout.
        logger.warning("tile facade: could not check the queue: %s", e)
        return None

    if not row:
        return None
    return "a {} job has been running since {}".format(row["kind"],
                                                       row["started_at"])


def _tile_payload(name: str, row: dict, req: "Txt2ImgRequest", started: float,
                  *, generated: bool) -> dict:
    """One finished tile, in the A1111 response shape."""
    with open(row["sheet_path"], "rb") as fh:
        encoded = base64.b64encode(fh.read()).decode("ascii")

    spec = row.get("spec") or {}
    elapsed_ms = (time.time() - started) * 1000
    logger.info("txt2img served TILE %r in %.0fms (%d b64 chars, generated=%s)",
                name, elapsed_ms, len(encoded), generated)

    return {
        "images": [encoded],
        "parameters": req.model_dump(),
        "info": json.dumps({
            "tile": name,
            "job_id": str(row["id"]),
            "cached": not generated,
            # A cache read is a file read. A caller measuring model throughput
            # off this number would be measuring a disk.
            "generated": generated,
            # The projection this tile was actually cut at. A consumer that
            # tessellates it needs the ratio to lay it out, and a tile cut at a
            # ratio the world does not use looks fine alone and seams in situ.
            "tile_w": spec.get("tile_w"),
            "tile_h": spec.get("tile_h"),
            "ratio": spec.get("ratio"),
            "colors": spec.get("colors"),
            "style_profile": spec.get("style_profile"),
            "tile_url": "/api/tiles/by-name/{}".format(name),
            "duration_ms": round(elapsed_ms),
        }),
    }


def _serve_tile(name: str, req: "Txt2ImgRequest", started: float) -> dict:
    """A named ground tile: served from disk, or BUILT and then served.

    Unlike `_serve_map` this may queue work - see TILE_PREFIX for the
    measurement that makes blocking honest. The order matters: cache first, so
    a name that already exists never costs the GPU and never costs the caller
    the wait.
    """
    import tiles

    row = tiles.resolve_name(name)
    if row:
        return _tile_payload(name, row, req, started, generated=False)

    # A miss, so we need the text to paint. Two channels, and they differ:
    #
    #   override_settings.tile  - the name arrived out of band, so the whole
    #                             prompt field is something2's own tile-row text
    #   tile:<name> <prompt>    - the name is the first token; strip it back off
    #                             or the model paints the word "road_sand"
    _, from_prefix = _split_tile_prompt(req.prompt)
    prompt = from_prefix or (req.prompt or "").strip() or name

    busy = _long_job_ahead()
    if busy:
        raise HTTPException(
            status_code=503,
            detail="tile {!r} is not built yet and cannot be built now: {}. "
                   "The GPU worker runs one job at a time. Retry when it is "
                   "free, or build the tile ahead of time with "
                   "POST /api/tiles.".format(name, busy))

    spec = tiles.TileSpec(
        prompt=prompt,
        name=name,
        # `width` is deliberately NOT read from the request. A tile's size is a
        # property of the world's projection, not of the caller's template -
        # something2 sends width=512 from the stock A1111 preset, which would
        # silently produce a 512px tile for a 64px world.
        tile_w=int(req.override_settings.get("tile_w") or tiles.DEFAULT_TILE_W),
        colors=int(req.override_settings.get("colors") or 16),
        style_profile=(req.override_settings.get("style_profile") or None),
        seed=req.seed if req.seed and req.seed > 0 else 0,
    )

    logger.info("tile facade: %r is not built; building it now (budget %ds)",
                name, TILE_BUILD_BUDGET_S)
    envelope = tiles.queue_tile(spec)
    task_id = envelope.get("celery_task_id")

    try:
        result = celery_app.AsyncResult(task_id).get(timeout=TILE_BUILD_BUDGET_S)
    except Exception as e:
        try:
            celery_app.control.revoke(task_id, terminate=True)
        except Exception:
            pass
        raise HTTPException(
            status_code=504,
            detail="tile {!r} did not finish within {}s: {}. It is still queued "
                   "as job {} - poll /api/jobs/{} and ask again once it is "
                   "done.".format(name, TILE_BUILD_BUDGET_S, e,
                                  envelope.get("job_id"),
                                  envelope.get("job_id")))

    if not result or result.get("error"):
        raise HTTPException(
            status_code=500,
            detail="tile {!r} failed to build: {}".format(
                name, (result or {}).get("error", "unknown failure")))

    # Re-resolve rather than trusting the task's return path: `resolve_name` is
    # the one place that checks the row is done AND the file is on disk, and the
    # facade should serve exactly what a later cache hit would serve.
    row = tiles.resolve_name(name)
    if not row:
        raise HTTPException(
            status_code=500,
            detail="tile {!r} reported success but is not readable back".format(
                name))
    return _tile_payload(name, row, req, started, generated=True)


@router.post("/sdapi/v1/txt2img")
def txt2img(req: Txt2ImgRequest, authorization: str | None = Header(default=None)):
    """Blocking text2img. Returns base64 PNG at `images[0]`, as A1111 does.

    Also the MAP FACADE: a prompt of `map:<name>` returns an already-built map
    picture from disk rather than generating anything. See `_map_request`.

    And the TILE FACADE: `tile:<name> <prompt>` returns a named ground tile,
    building it first if it does not exist. See `_serve_tile` for why tiles may
    build where maps may not, and `_split_tile_prompt` for why the name rides
    in the prompt field.
    """
    _require_auth(authorization)
    started = time.time()

    # THE MAP FACADE. A map that already exists is served from disk instead of
    # being generated - see `_map_request` for why the name travels in the
    # prompt.
    wanted_map = _map_request(req)
    if wanted_map:
        return _serve_map(wanted_map, req, started)

    # THE TILE FACADE. Cache read when the name exists, a real build when it
    # does not - see `_serve_tile`. Checked after maps so neither prefix can
    # shadow the other.
    wanted_tile = _tile_request(req)
    if wanted_tile:
        return _serve_tile(wanted_tile, req, started)

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
            # Present only when a cutout was requested. The caller knows
            # whether it asked for an object or a texture; this service only
            # knows the pixels, so it reports them rather than guessing.
            **({"cutout": result["cutout"]} if result.get("cutout") else {}),
        }),
    }
