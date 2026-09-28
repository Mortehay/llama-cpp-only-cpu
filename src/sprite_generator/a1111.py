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

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

import auth
import core_models
import generations
from tasks import (celery_app, generate_raw_task, gpu_fault_block_reason,
                   worker_busy_reason)
from core_models import is_gguf

logger = logging.getLogger(__name__)
router = APIRouter()

# Their AI_PROVIDER_GENERATE_TIMEOUT_MS defaults to 5 minutes. Stay under it so
# the failure surfaces as our error message, not their opaque timeout.
#
# 285, raised from 240 on 2026-09-28 at the owner's request (decisions/0012):
# 15s under their 300 leaves room for our error to reach them before they give
# up. Raising it further only helps if something2 raises theirs first - past
# 300 we would just be waiting on a connection they already closed. It does
# NOT cover a request queued behind a Qwen core (~230s + an SDXL cold reload);
# worker_busy_reason() turns that case away up front instead.
GENERATE_TIMEOUT_S = int(os.environ.get("A1111_GENERATE_TIMEOUT_S", "285"))

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
    # [0] is what something2 gets when it names no model. Owner's choice,
    # 2026-09-28: Qwen-Image-2512 Q2_K + 8-step Lightning LoRA. Benched 0/12
    # contact sheets against SDXL + nerijs's 2/12, 24 s/image warm, ~130 s
    # cold (it evicts SDXL, so a model change pays a load). It is step-
    # distilled and runs at CFG 1, so the negative prompt - including what
    # split_negations moves there - is INERT; the bench measured the sheets
    # anyway. Square only (see txt2img). decisions/0012 2b.
    "gguf:qwen-image-2512-Q2_K+lightning8",
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
    #
    # Qwen-Image-2512 GGUF: best quality on this box, and slow - ~200-260s per
    # image, close to GENERATE_TIMEOUT_S, so a cold run can time out and a
    # rejected cutout gets no retry. Added 2026-09-28 at the owner's request,
    # knowing that. Square only; see txt2img. decisions/0012 D2.
    "gguf:Qwen-Image-2512-Q3_K_M",
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

    RETURNS THE PRINCIPAL, which it used to discard. That discard is why the
    ledger could not exist: `auth.require` has always known which key made the
    call, and this was the one place that knew it for the facade something2
    uses. See generations.describe_caller for why the key - not the source
    address - is the identity that separates callers on this network.
    """
    return auth.require(authorization, scope)


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

# THE ENTITY FACADE. Same addressing as tiles, and the same cache-then-build
# behaviour, but the storage underneath it is different and deliberately so.
#
# A tile is a JOB because building one is a pipeline - paint, quantise, cut to
# the world's rhombus - and `jobs` already carried the spec that pipeline needs.
# An entity is one `generate_raw_task` call, measured in seconds; wrapping it in
# a job row would mean a queue, a poll contract and a stage machine for work
# that finishes before the response is written. So an entity is a row in
# `generations`, the ledger every API call now writes anyway, and the NAME on
# that row is what makes it addressable.
#
# The behaviour that matters to the caller is identical to tiles: ask twice and
# the second answer is a file read. Before this, 731 raw generations had been
# written to disk and forgotten, so asking twice cost the GPU twice.
ENTITY_PREFIX = "entity:"


def _map_request(req: "Txt2ImgRequest") -> str | None:
    """The map name this request is asking for, or None for a normal generate."""
    explicit = req.override_settings.get("map")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    prompt = (req.prompt or "").strip()
    if prompt.lower().startswith(MAP_PREFIX):
        return prompt[len(MAP_PREFIX):].strip() or None
    return None


def _serve_map(name: str, req: "Txt2ImgRequest", started: float,
               caller: dict | None = None) -> dict:
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

    # `job_id` and not `file_path`: the map job already owns this picture, and
    # a ledger row that also claimed it would put the same PNG in the gallery
    # twice. See migration 017.
    generations.record(kind="map", name=name, served_from="cache",
                       prompt=(req.prompt or ""), job_id=str(row["id"]),
                       duration_ms=elapsed_ms, caller=caller,
                       params={"complete": complete})

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


def _split_prefixed(raw: str | None, prefix: str) -> tuple[str | None, str | None]:
    """`tile:road_sand cracked red stone` -> ("road_sand", "cracked red stone").

    THE NAME IS THE FIRST TOKEN AND THE REST IS THE PROMPT, which is forced by
    something2's template system rather than chosen. Their request template
    substitutes `{{prompt}}`, `{{width}}`, `{{height}}`, `{{seed}}`, `{{frames}}`
    and `{{model}}` - and nothing that carries a NAME. So the one field that can
    hold a name is the prompt, alongside the prompt.

    That makes configuring a tile an edit an operator can actually make: put
    `tile:rocks ` in front of the text already in something2's tile row and
    change nothing else. No new placeholder, no code on their side, and the
    prompt still reaches the model intact. `entity:goblin_scout ` in front of an
    entity row is the identical edit, which is the whole reason entities reuse
    this shape rather than inventing a second one.

    A bare `tile:rocks` with no remaining text is valid - the name doubles as
    the prompt, which is what a one-word ground like "sand" wants anyway.
    """
    text = (raw or "").strip()
    if not text.lower().startswith(prefix):
        return None, None

    rest = text[len(prefix):].strip()
    if not rest:
        return None, None

    parts = rest.split(None, 1)
    name = parts[0]
    prompt = parts[1].strip() if len(parts) > 1 else ""
    return name, (prompt or name)


def _split_tile_prompt(raw: str | None) -> tuple[str | None, str | None]:
    """`tile:<name> <prompt>` -> (name, prompt). See `_split_prefixed`."""
    return _split_prefixed(raw, TILE_PREFIX)


def _split_entity_prompt(raw: str | None) -> tuple[str | None, str | None]:
    """`entity:<name> <prompt>` -> (name, prompt). See `_split_prefixed`."""
    return _split_prefixed(raw, ENTITY_PREFIX)


def _entity_request(req: "Txt2ImgRequest") -> tuple[str | None, str | None]:
    """The entity name and the prompt to paint, or (None, None).

    Two channels, as for tiles. `override_settings.entity` carries the name out
    of band and leaves the whole prompt field as something2's own entity-row
    text; `entity:<name> <prompt>` puts the name in front, and the name is then
    stripped back off or the model paints the words "goblin_scout".
    """
    explicit = req.override_settings.get("entity")
    if isinstance(explicit, str) and explicit.strip():
        name = explicit.strip()
        # The prefix may ALSO be present - an operator who set both should not
        # get the name painted into the picture.
        _, stripped = _split_entity_prompt(req.prompt)
        return name, (stripped or (req.prompt or "").strip() or name)

    return _split_entity_prompt(req.prompt)


def _declared_kind(req: "Txt2ImgRequest", cutout: bool,
                   entity_name: str | None) -> str:
    """What this txt2img call produced: entity | tile | map | raw.

    THIS SERVICE CANNOT TELL A BARREL FROM A PATCH OF GRASS by looking at the
    pixels, and something2 asks for both down this one route. So it does not
    guess from the prompt text - the same reason `cutout` is an explicit flag
    and not inferred from the words "transparent background".

    Three honest signals, in this order:

      entity_name              the caller ADDRESSED it as an entity, which is
                               the strongest declaration available
      override_settings.kind   the caller said so outright
      cutout                   only an object composited over terrain needs
                               one; a ground texture is supposed to be opaque

    Everything else is `raw`, which is what the file on disk has always been
    called and is the one label that claims nothing.

    THE NAME HAS TO WIN, and this function got it wrong first time round. With
    the cutout heuristic ranked above it, `entity:test_probe a mushroom` with
    no cutout stored `kind='raw'` WITH a name attached - a row the entity
    resolver (which filters on kind) could never find. Measured: the second
    identical request regenerated instead of cache-reading, and
    `GET /api/entities` reported zero while two named rows sat in the table.
    Naming and kind are one decision, so they are made in one place.
    """
    if entity_name:
        return "entity"
    declared = req.override_settings.get("kind")
    if isinstance(declared, str) and declared.strip().lower() in generations.KINDS:
        return declared.strip().lower()
    return "entity" if cutout else "raw"


def _entity_payload(name: str, row: dict, req: "Txt2ImgRequest", started: float,
                    caller: dict | None = None) -> dict:
    """A named entity served from the ledger, in the A1111 response shape.

    The counterpart of `_tile_payload`, and the reason the entity facade is
    worth having at all: this path never touches the GPU. Before it existed,
    something2 asking for the same entity twice paid for it twice, and on a
    card with no spare VRAM the second request was also a chance to fault the
    context.
    """
    with open(row["file_path"], "rb") as fh:
        encoded = base64.b64encode(fh.read()).decode("ascii")

    elapsed_ms = (time.time() - started) * 1000
    logger.info("txt2img served ENTITY %r from cache in %.0fms (%d b64 chars)",
                name, elapsed_ms, len(encoded))

    generations.record(kind="entity", name=name, served_from="cache",
                       prompt=(row.get("prompt") or ""),
                       model=row.get("model"),
                       job_id=None, duration_ms=elapsed_ms, caller=caller,
                       params={"cache_of": str(row["id"])})

    params = row.get("params") or {}
    return {
        "images": [encoded],
        "parameters": req.model_dump(),
        "info": json.dumps({
            "entity": name,
            "generation_id": str(row["id"]),
            "cached": True,
            # A cache read is a file read. A caller measuring model throughput
            # off this number would be measuring a disk.
            "generated": False,
            "seed": row.get("seed"),
            "model": row.get("model"),
            "cutout": bool(params.get("cutout")),
            "entity_url": "/api/entities/by-name/{}".format(name),
            "duration_ms": round(elapsed_ms),
        }),
    }


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
    # tasks.worker_busy_reason reads BOTH registries - the jobs table and the
    # Redis flag a Qwen core sets. This used to read only the table, so a tile
    # queued behind a 4-minute core and timed out. Another tile building does
    # not count: tiles are short and queue fine behind each other.
    busy = worker_busy_reason(exclude_kinds=("tile",))
    return busy["detail"] if busy else None


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


def _serve_tile(name: str, req: "Txt2ImgRequest", started: float,
                caller: dict | None = None) -> dict:
    """A named ground tile: served from disk, or BUILT and then served.

    Unlike `_serve_map` this may queue work - see TILE_PREFIX for the
    measurement that makes blocking honest. The order matters: cache first, so
    a name that already exists never costs the GPU and never costs the caller
    the wait.
    """
    import tiles

    row = tiles.resolve_name(name)
    if row:
        generations.record(kind="tile", name=name, served_from="cache",
                           prompt=(req.prompt or ""), job_id=str(row["id"]),
                           duration_ms=(time.time() - started) * 1000,
                           caller=caller)
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
        detail = ("tile {!r} is not built yet and cannot be built now: {}. "
                  "The GPU worker runs one job at a time. Retry when it is "
                  "free, or build the tile ahead of time with "
                  "POST /api/tiles.".format(name, busy))
        # A REFUSAL IS ACTIVITY. Left unrecorded, the Activity tab shows a
        # quiet API and something2's operator sees only a 503 - and the two
        # facts that explain each other (a sheet holds the card, tile requests
        # are bouncing off it) never appear in the same place.
        generations.record(kind="tile", name=name, status="failed",
                           served_from="generated", prompt=(req.prompt or ""),
                           error=detail, caller=caller,
                           duration_ms=(time.time() - started) * 1000)
        raise HTTPException(status_code=503, detail=detail)

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

    # The job is already in `jobs`; this row is who ASKED for it. Opened with
    # the job id attached so the Activity tab shows one pending item rather
    # than two - see generations.begin.
    gen = generations.begin(kind="tile", name=name, prompt=prompt,
                            job_id=envelope.get("job_id"),
                            params={"tile_w": spec.tile_w,
                                    "colors": spec.colors,
                                    "style_profile": spec.style_profile},
                            caller=caller)
    generations.attach_task(gen, task_id)

    try:
        result = celery_app.AsyncResult(task_id).get(timeout=TILE_BUILD_BUDGET_S)
    except Exception as e:
        try:
            celery_app.control.revoke(task_id, terminate=True)
        except Exception:
            pass
        detail = ("tile {!r} did not finish within {}s: {}. It is still queued "
                  "as job {} - poll /api/jobs/{} and ask again once it is "
                  "done.".format(name, TILE_BUILD_BUDGET_S, e,
                                 envelope.get("job_id"),
                                 envelope.get("job_id")))
        generations.fail(gen, detail,
                         duration_ms=(time.time() - started) * 1000)
        raise HTTPException(status_code=504, detail=detail)

    if not result or result.get("error"):
        detail = "tile {!r} failed to build: {}".format(
            name, (result or {}).get("error", "unknown failure"))
        generations.fail(gen, detail,
                         duration_ms=(time.time() - started) * 1000)
        raise HTTPException(status_code=500, detail=detail)

    # Re-resolve rather than trusting the task's return path: `resolve_name` is
    # the one place that checks the row is done AND the file is on disk, and the
    # facade should serve exactly what a later cache hit would serve.
    row = tiles.resolve_name(name)
    if not row:
        detail = "tile {!r} reported success but is not readable back".format(
            name)
        generations.fail(gen, detail,
                         duration_ms=(time.time() - started) * 1000)
        raise HTTPException(status_code=500, detail=detail)

    generations.finish(gen, job_id=str(row["id"]), served_from="generated",
                       duration_ms=(time.time() - started) * 1000)
    return _tile_payload(name, row, req, started, generated=True)


@router.post("/sdapi/v1/txt2img")
def txt2img(req: Txt2ImgRequest, request: Request,
            authorization: str | None = Header(default=None)):
    """Blocking text2img. Returns base64 PNG at `images[0]`, as A1111 does.

    Also the MAP FACADE: a prompt of `map:<name>` returns an already-built map
    picture from disk rather than generating anything. See `_map_request`.

    The TILE FACADE: `tile:<name> <prompt>` returns a named ground tile,
    building it first if it does not exist. See `_serve_tile` for why tiles may
    build where maps may not, and `_split_prefixed` for why the name rides in
    the prompt field.

    And the ENTITY FACADE: `entity:<name> <prompt>` does the same for a single
    entity image, cached in `generations` rather than `jobs` - see
    ENTITY_PREFIX.

    EVERY PATH THROUGH HERE NOW WRITES A LEDGER ROW. It did not, and the cost
    was measured on 2026-09-10: 731 `raw_*.png` on disk against zero database
    rows that knew about any of them.
    """
    principal = _require_auth(authorization)
    caller = generations.describe_caller(principal, request)
    started = time.time()

    # THE MAP FACADE. A map that already exists is served from disk instead of
    # being generated - see `_map_request` for why the name travels in the
    # prompt.
    wanted_map = _map_request(req)
    if wanted_map:
        return _serve_map(wanted_map, req, started, caller)

    # THE TILE FACADE. Cache read when the name exists, a real build when it
    # does not - see `_serve_tile`. Checked after maps so neither prefix can
    # shadow the other.
    wanted_tile = _tile_request(req)
    if wanted_tile:
        return _serve_tile(wanted_tile, req, started, caller)

    # THE ENTITY FACADE. Checked last of the three, so a name that happens to
    # begin with another prefix cannot be captured here.
    #
    # An unnamed request falls through with `entity_name = None`: it still gets
    # a ledger row and still lands in the gallery, it just cannot be asked for
    # again by name. That is the honest outcome - there is nothing to address
    # it BY - and it is why naming is worth an operator's one-line edit.
    entity_name, entity_prompt = _entity_request(req)
    if entity_name:
        hit = generations.resolve_name(entity_name, "entity")
        if hit:
            return _entity_payload(entity_name, hit, req, started, caller)

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

    # `entity:goblin_scout a small green raider` must reach the model as "a
    # small green raider". Painting the handle into the picture is the exact
    # failure `_split_prefixed` exists to prevent, and it is silent - the image
    # comes back looking almost right, with lettering in it.
    asked = (entity_prompt if entity_name else req.prompt) or ""

    prompt = core_models.apply_trigger(model, asked)
    if prompt != asked:
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

    # The ledger row, opened BEFORE the task is queued.
    #
    # Two things depend on the order. A row written only on success cannot
    # appear in the pending list, which is the one moment anyone wants to look
    # at it; and a request that dies mid-generation - the timeout below, a
    # faulted CUDA context, a worker restart - would leave no trace of having
    # arrived at all. That is precisely the six-hour outage recorded in
    # `.ai/project-context.md`, where the API answered 200 and nothing
    # generated.
    gen = generations.begin(
        kind=_declared_kind(req, cutout, entity_name),
        name=entity_name,
        prompt=asked,
        negative_prompt=req.negative_prompt or "",
        model=model,
        seed=(req.seed if req.seed and req.seed > 0 else None),
        params={"width": width, "height": height, "steps": steps,
                "cfg_scale": cfg, "frames": frames, "cutout": cutout,
                "lora_scale": lora_scale},
        caller=caller)

    # Turn the storm away HERE, before a task is queued.
    #
    # The worker gate in `_gpu_breaker_admit` is the authoritative one - only
    # the process that owns the context may decide it is alive. This is the
    # cheap one, and it is placed after `generations.begin` on purpose: a
    # refusal is still a request that arrived, and the activity dashboard is
    # where anyone looking at "why is nothing generating" will be. Refusals
    # that leave no trace are how the six-hour silent outage happened.
    blocked = gpu_fault_block_reason()
    if blocked:
        detail = ("GPU context faulted %ds ago and is cooling down; retry in "
                  "%ds. (%s)" % (blocked["faulted_for_s"],
                                 blocked["retry_after_s"], blocked["error"]))
        logger.warning("txt2img refused: %s", detail)
        generations.fail(gen, detail,
                         duration_ms=(time.time() - started) * 1000)
        raise HTTPException(status_code=503, detail=detail,
                            headers={"Retry-After": str(blocked["retry_after_s"])})

    # Qwen-Image is offered here at the owner's request (2026-09-28), slow
    # budget and all: ~200-260s per image against GENERATE_TIMEOUT_S, and no
    # room for a cutout retry. decisions/0012 D2. It renders square only, so a
    # multi-frame or non-square request is refused here rather than stretched.
    if is_gguf(model) and width != height:
        detail = (f"'{model}' renders square images only; got {width}x{height}"
                  f"{' (%d frames)' % frames if frames > 1 else ''}. Use an "
                  f"SDXL model for sheets or non-square sizes.")
        generations.fail(gen, detail, duration_ms=(time.time() - started) * 1000)
        raise HTTPException(status_code=400, detail=detail)

    # Same idea for a long job already holding the one worker - a Qwen core or
    # a sheet/map/training job: queueing behind it can only time out late, so
    # say "busy" now and when to come back.
    busy = worker_busy_reason()
    if busy:
        busy["retry_after_s"] = busy["retry_after_s"] or 120
        detail = ("GPU is busy: %s; retry in %ds."
                  % (busy["detail"], busy["retry_after_s"]))
        logger.info("txt2img refused: %s", detail)
        generations.fail(gen, detail, duration_ms=(time.time() - started) * 1000)
        raise HTTPException(status_code=503, detail=detail,
                            headers={"Retry-After": str(busy["retry_after_s"])})

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
    generations.attach_task(gen, task.id)

    try:
        result = task.get(timeout=GENERATE_TIMEOUT_S)
    except Exception as e:
        # Do not leave the worker grinding on output nobody will read.
        try:
            celery_app.control.revoke(task.id, terminate=True)
        except Exception:
            pass
        logger.error(f"txt2img task {task.id} did not complete: {e}")
        detail = f"Generation did not finish within {GENERATE_TIMEOUT_S}s: {e}"
        generations.fail(gen, detail,
                         duration_ms=(time.time() - started) * 1000)
        raise HTTPException(status_code=504, detail=detail)

    if not result or result.get("error"):
        detail = (result or {}).get("error", "unknown generation failure")
        kind = (result or {}).get("error_kind")
        # A failed cutout is the CALLER's request being unsatisfiable, not this
        # service breaking, and the distinction matters to a bulk runner: 422
        # means "this subject will not cut out, skip or reword it", 500 means
        # "retry later". Returning an opaque image instead would be worse than
        # either - it stores clean-looking data that is wrong.
        status = 422 if kind == "cutout_failed" else 500
        headers = None
        if kind == "gpu_faulted":
            # 503 + Retry-After, and the header is the point of the whole
            # exercise. Under 500 a bulk runner reasonably retries at once, and
            # the ledger shows what that costs: ten requests in one minute,
            # each bouncing off a dead context in ~91ms, for hours. The card
            # recovers in a quiet window and cannot get one while it is being
            # asked. This names the length of that window in the one place a
            # generic HTTP client will actually obey.
            status = 503
            headers = {"Retry-After": str(result.get("retry_after_s") or 90)}
        generations.fail(gen, detail,
                         duration_ms=(time.time() - started) * 1000)
        raise HTTPException(status_code=status, detail=detail, headers=headers)

    file_path = result.get("file_path")
    if not file_path or not os.path.exists(file_path):
        detail = "Generation reported success but produced no file"
        generations.fail(gen, detail,
                         duration_ms=(time.time() - started) * 1000)
        raise HTTPException(status_code=500, detail=detail)

    with open(file_path, "rb") as fh:
        encoded = base64.b64encode(fh.read()).decode("ascii")

    elapsed_ms = (time.time() - started) * 1000
    logger.info(f"txt2img served {model} in {elapsed_ms:.0f}ms ({len(encoded)} b64 chars)")

    # `file_path` and no job_id: this request owns the PNG, so assets_v shows
    # it and the gallery stops under-reporting what this machine has made.
    generations.finish(gen, file_path=file_path, seed=result.get("seed"),
                       duration_ms=result.get("duration_ms") or elapsed_ms)

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
            # The handle this image can be asked for by from now on, and the
            # flag that says the next identical request will be free. Absent on
            # an unnamed call, because there would be nothing to address.
            **({"entity": entity_name,
                "cached": False,
                "generated": True,
                "entity_url": f"/api/entities/by-name/{entity_name}"}
               if entity_name else {}),
            # Present only when a cutout was requested. The caller knows
            # whether it asked for an object or a texture; this service only
            # knows the pixels, so it reports them rather than guessing.
            **({"cutout": result["cutout"]} if result.get("cutout") else {}),
        }),
    }
