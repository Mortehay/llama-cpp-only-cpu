import os
import io
import json
import time
import uuid
import torch
import diffusers.loaders.single_file_utils as sf_utils
import diffusers.loaders.single_file_model as sf_model
import sys

from diffusers import (
    StableDiffusionXLImg2ImgPipeline, 
    StableDiffusionXLPipeline, 
    FluxImg2ImgPipeline, 
    FluxPipeline, 
    StableDiffusionPipeline,
    FluxTransformer2DModel,
    GGUFQuantizationConfig,
    StableDiffusionPipeline, 
    StableDiffusionXLPipeline, 
    StableDiffusionXLImg2ImgPipeline,
    DPMSolverMultistepScheduler
)
import psycopg2
import random
import logging
import requests
from collections import namedtuple
from celery import Celery, chord, group
from celery.signals import worker_ready, task_prerun, task_postrun

from PIL import Image, ImageDraw
import base64
import io
import multiprocessing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Compute device ------------------------------------------------------
# Set by compose: docker-compose.cuda.yml forces cuda on the worker and cpu on
# the API process (which imports this module but never runs inference).
# "auto" keeps bare-metal runs working without any env at all.
_requested_device = os.environ.get("COMPUTE_DEVICE", "auto").lower()
if _requested_device == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
elif _requested_device == "cuda" and not torch.cuda.is_available():
    # Refuse to start. This used to warn and fall back to CPU, which was the
    # wrong call: the worker came up "healthy", accepted jobs, and produced the
    # same sprites two orders of magnitude slower, with one WARNING line buried
    # in a log nobody reads. A container that cannot do the one thing it exists
    # to do should not report itself ready.
    #
    # The usual cause is nvidia-container-toolkit missing or unconfigured on the
    # Docker host inside WSL, not a missing card. `make gpu-check` distinguishes
    # them: it runs nvidia-smi INSIDE a container.
    raise RuntimeError(
        "COMPUTE_DEVICE=cuda but torch.cuda.is_available() is False. This "
        "service is GPU-only. Check nvidia-container-toolkit on the Docker "
        "host with `make gpu-check`; set COMPUTE_DEVICE=cpu only for the API "
        "process, which never runs inference."
    )
else:
    DEVICE = _requested_device

# float16 on CUDA: the 3060 is Ampere consumer silicon with weak bf16 throughput,
# and fp16 halves both VRAM and memory bandwidth. On CPU fp16 is unusably slow,
# so the CPU path stays float32 (and bfloat16 for FLUX, see get_flux_pipeline).
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

if DEVICE == "cpu":
    # Thread pinning only matters on CPU; on CUDA these threads just spin.
    cpu_limit = multiprocessing.cpu_count()
    os.environ["OMP_NUM_THREADS"] = str(cpu_limit)
    os.environ["MKL_NUM_THREADS"] = str(cpu_limit)
    torch.set_num_threads(cpu_limit)
    logger.info(f"Compute device: CPU ({cpu_limit} inference threads).")
else:
    # Do NOT call torch.cuda.get_device_properties() at import.
    #
    # Property access initializes a CUDA context in whatever process imports
    # this module. Celery's prefork pool imports tasks here in the parent, then
    # forks children to run tasks — and a forked child of a CUDA-initialized
    # parent dies with "Cannot re-initialize CUDA in forked subprocess". That
    # breaks every GPU task, not just diagnostics.
    #
    # torch.cuda.is_available() above is fork-safe (it uses NVML and does not
    # create a context). Device details are logged lazily instead, from inside
    # whichever process actually loads a pipeline.
    logger.info(f"Compute device: CUDA (device details on first model load), dtype={DTYPE}.")

_cuda_details_logged = False


def log_cuda_details_once():
    """Log GPU name/VRAM from within the process that will actually use it."""
    global _cuda_details_logged
    if _cuda_details_logged or DEVICE != "cuda":
        return
    try:
        props = torch.cuda.get_device_properties(0)
        logger.info(f"CUDA device: {props.name} "
                    f"({props.total_memory / 1024**3:.1f} GiB VRAM)")
        _cuda_details_logged = True
    except Exception as e:
        logger.warning(f"Could not read CUDA device properties: {e}")

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
DB_URL = os.environ.get("DB_URL")
IMAGES_DIR = "/app/images"

celery_app = Celery("sprite_tasks", broker=REDIS_URL, backend=REDIS_URL)

# Redis client for cooperative cancellation flags
import redis as _redis
_redis_client = _redis.from_url(REDIS_URL, decode_responses=True)

def set_cancel_flag(task_id: str):
    """Mark a task for cooperative cancellation (expires in 10 min)."""
    _redis_client.setex(f"cancel:{task_id}", 600, "1")

def is_cancelled(task_id: str) -> bool:
    """Check if a task has been flagged for cancellation."""
    return _redis_client.exists(f"cancel:{task_id}") > 0

def clear_cancel_flag(task_id: str):
    """Remove the cancellation flag."""
    _redis_client.delete(f"cancel:{task_id}")

pipes = {}
PipelineOutput = namedtuple("PipelineOutput", ["images"])

# Checkpoints distilled for few-step sampling. They are trained to run at 1-4
# steps with classifier-free guidance switched off; conventional settings
# (20 steps / cfg 7) do not error, they just produce over-guided, washed-out
# images — which is worse, because it looks like a prompt problem.
import poses

# Sampling for pose-conditioned frames. Kept as constants because they are the
# two knobs worth sweeping when output quality drifts:
#   POSED_STRENGTH   how much of the core img2img may overwrite. Higher than
#                    the unposed 0.55 because ControlNet, not the init image,
#                    is holding the pose now.
#   CONTROLNET_SCALE how hard the skeleton is enforced. Near 1.0 follows the
#                    pose closely at the cost of style; too low and the pose
#                    is a suggestion the model ignores.
# Swept on a clean core, 4-frame walk. mean inter-frame difference /
# max difference from frame 0 / mean difference from the core (0-255):
#   0.50  1.3 / 1.5 / 14.2   barely moves
#   0.60  3.6 / 3.9 / 16.6   moves, character intact      <- chosen
#   0.70  9.1 / 9.7 / 20.2   silhouette starts breaking up
# 0.75 was a guess and it was wrong: it returned a different character in
# different clothes. Conditioning scale 1.0 beat 0.6 at every strength.
#
# The constant then sat at 0.75 anyway - the value this very comment records as
# measured and rejected - and produced exactly the failure it predicts. A white
# shirted zombie came back in a dark outfit holding a pole, with the OpenPose
# limb palette (magenta head, orange shoulders, yellow arms, cyan legs) printed
# onto the sprite itself. At 0.75 so little of the init image survives that the
# skeleton is the strongest signal left in the frame, and the model reproduces
# its COLOURS and not just its geometry. Use the value the sweep chose.
# 0.60 was chosen by the sweep above, and that sweep asked the wrong question.
# It scored MOTION and DRIFT, and never scored whether the frame came back
# looking like pixel art at all. At 0.60 only ~60% of the denoising steps run,
# so the output is the ORIGINAL core lightly modified - and the core is soft and
# painterly, so the sprite inherits that and is never re-rendered in the
# checkpoint's own crisp style. Idle at 0.60 came out mushy and blotchy: an
# unreadable face and a torso of green camouflage smear. The walk rows looked
# better not because they move but because they are re-drawn at high strength.
#
# 0.75 redraws. What makes it affordable is the IP-Adapter holding identity,
# which is also why this could not have been the answer before it was wired up:
# 0.75 without an identity anchor is where the character starts changing
# clothes. Actions that need to ADD an effect still override this downward or
# upward per action in action_prompts.json.
POSED_STRENGTH = 0.75
CONTROLNET_SCALE = 1.0

# IP-ADAPTER WAS TRIED HERE AND REJECTED. Do not reach for it again without
# reading this.
#
# The reasoning that leads to it is sound, and it is the same reasoning that
# put ControlNet in poses.py. img2img injects the core image exactly ONCE: it
# noises the core to `strength` and denoises from there, after which the model
# has no reference to the original at all. So `strength` is a single number
# deciding two things - how much identity survives, and how much freedom the
# prompt has - which is why no value satisfies both. IP-Adapter should fix that
# by re-injecting the reference through cross-attention at every step, giving
# identity its own channel exactly as ControlNet gave pose one.
#
# It does preserve identity. It also suppresses the effect completely, which
# makes it useless for the actions that actually need help. Measured on burning
# at strength 0.85, fraction of the sprite that is flame-coloured:
#   no adapter        15.57%   real flames
#   scale 0.15         7.98%   NO flames - detector counting orange clothing
#   scale 0.25         6.28%   NO flames - same false positive
#   scale 0.35         0.27%   no flames
#   scale 0.50         0.00%   no flames
#   scale 0.80         0.00%   no flames
# ip-adapter-plus_sd15 was worse than the base adapter at every scale.
#
# The mechanism is why there is no scale to tune to: the adapter anchors
# appearance to a reference that contains no fire, so every step pulls back
# toward not-on-fire. For any action whose whole purpose is to ADD something
# the core does not have - flames, a bow, a spell - the adapter is pulling
# against the prompt by construction. It would help the actions that add
# nothing (idle, walks, melee), and those already work at 0.60.
#
# Weights are cached at /models/models--h94--IP-Adapter (2.6GB); the image
# encoder is the bulk of it.
#
# It is now wired up, OPT-IN PER ACTION via "ip_adapter_scale" in
# action_prompts.json, and off everywhere by default. The measurements above
# are why it is off rather than why it is absent: they say it is wrong for
# actions that ADD something, not that it is wrong for everything. An action
# that only changes pose has no effect to suppress, so there is nothing for the
# adapter to fight and its identity anchoring is free. Whether that is worth
# ~1.2GB of resident image encoder is a judgement call, so it is a switch.
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "models"
# The base adapter, not -plus. Plus encodes finer detail and was worse here at
# every scale tested - it copies the reference harder, which is precisely the
# behaviour that suppresses whatever the prompt is trying to add.
IP_ADAPTER_WEIGHT = "ip-adapter_sd15.bin"


def ensure_ip_adapter(pipe):
    """Load the IP-Adapter onto `pipe` once, returning whether it is available.

    Pipelines are cached and reused across sheets, so this attaches to whatever
    came back from get_sd_pipeline rather than being part of the cache key -
    otherwise turning the adapter on for one action would evict the pipeline
    and force a full checkpoint reload.

    A failure here is not fatal. The adapter is an enhancement to identity, and
    a sheet rendered without it is still a sheet; refusing to generate because
    an optional 2.6GB download is missing would be the wrong trade.
    """
    if getattr(pipe, "_ipa_loaded", False):
        return True
    if not hasattr(pipe, "load_ip_adapter"):
        logger.warning(f"{type(pipe).__name__} has no load_ip_adapter; "
                       "ip_adapter_scale will be ignored.")
        return False
    try:
        pipe.load_ip_adapter(IP_ADAPTER_REPO, subfolder=IP_ADAPTER_SUBFOLDER,
                             weight_name=IP_ADAPTER_WEIGHT, cache_dir="/models")
        pipe._ipa_loaded = True
        logger.info(f"IP-Adapter loaded ({IP_ADAPTER_WEIGHT}).")
        return True
    except Exception as e:
        logger.warning(f"Could not load IP-Adapter ({e}); continuing without "
                       "it. Frames will render, identity anchoring is just off.")
        return False

# Per-checkpoint trigger tokens for STEP 1 (one character, not a sheet).
#
# A DreamBooth finetune only activates its trained style when its own trigger
# appears in the prompt, so running one of these without its trigger gets you
# the base model wearing a thin coat of prompt wording. Measured 2026-08-21 with
# identical subject and seed: All-In-One with "pixelsprite" produced the only
# genuine pixel art of four candidates - hard grid, bounded palette, one
# character, 3.0s - where the generic prompt gives soft continuous-tone output.
#
# A trigger belonging to a DIFFERENT checkpoint is inert at best. PixelartFSS is
# worse than inert: it expands to "Front Sprite Sheet", and on Onodofthenorth it
# returned FOUR characters in a row even at guidance 7.5 with "multiple
# characters, group, crowd, twins, clones" in the negative prompt. Training
# beats guidance. That checkpoint therefore has no usable step-1 trigger at all
# - every trigger it ships (FSS/RSS/LSS/BSS) asks for a sheet, and step 1 wants
# exactly one character - so it is deliberately absent from this map.
#
# Keys are lowercased repo ids; look-ups lowercase the incoming name.
CORE_TRIGGERS = {
    "publicprompts/all-in-one-pixel-model": "pixelsprite",
    "kohbanye/pixel-art-style": "pixelartstyle",
}

# STEP 2 view triggers. Different job from CORE_TRIGGERS above, which activates
# a checkpoint's art style; these select which way the character FACES.
#
# Onodofthenorth ships four trained directional views. That is the capability
# nothing else here has: All-In-One has never seen a character from behind, so
# `move up` renders a face no matter what the prompt says, while `PixelartBSS`
# returns a genuine faceless back.
#
# Why this is safe in step 2 when the same triggers were banned from step 1:
# CORE_TRIGGERS records that FSS in txt2img returns four characters in a row
# even at guidance 7.5 with duplicate-suppression negatives. Step 2 is different
# in kind - img2img from a single-character core AND a single ControlNet
# skeleton both pin the composition - and rendering FSS/BSS/RSS through that
# path returned single characters every time. The trigger asks for a sheet and
# the init image plus skeleton overrule it.
#
# Keys are lowercased repo ids. A checkpoint absent from this map simply gets no
# view trigger, and falls back to derive_side_core for profile actions.
VIEW_TRIGGERS = {
    "onodofthenorth/sd_pixelart_spritesheet_generator": {
        "front": "PixelartFSS",
        "back": "PixelartBSS",
        "left": "PixelartLSS",
        "right": "PixelartRSS",
    },
}


def view_trigger(llm_name: str, view: str):
    """The trigger token selecting `view` on this checkpoint, or None."""
    if not view:
        return None
    return VIEW_TRIGGERS.get((llm_name or "").strip().lower(), {}).get(view)


DISTILLED_MARKERS = ("turbo", "schnell", "lightning", "lcm")

# Shared quality exclusions. Applies to both generation steps.
_NEGATIVE_QUALITY = (
    "blurry, deformed, extra limbs, cropped, low quality, watermark, text, "
    "noise, messy pixels, artifacting, gradient, shadows on background"
)

# Step 1 (core): exactly ONE character. Duplicate-suppression belongs here only.
# Ground terms come FIRST, ahead of the duplicate-character clause.
#
# They were absent entirely, and that is where the brown patch under every
# sprite came from. Step 1 was never asked for a character standing on nothing,
# so it drew one standing on dirt with grass tufts - and because that patch is
# fused to the feet it is neither background-coloured nor a separate blob, so
# neither remove_background nor _isolate_largest_sprite can touch it. Step 2
# then inherits it through img2img: NEGATIVE_FRAME does list the ground terms,
# but at strength 0.60 there is nothing it can do about pixels that are already
# in the init image. The only place this is fixable by prompting is here.
#
# Terse on purpose. NEGATIVE_SINGLE was 59 tokens and the limit is 77, so the
# obvious wording ("ground, floor, dirt, soil, grass, ground shadow, drop
# shadow, base, platform") measured 79 and would have silently lost its tail -
# the exact failure check_prompt_length exists to catch. This lands at 72.
NEGATIVE_SINGLE = (
    "ground, floor, dirt, grass, ground shadow, base, "
    "multiple characters, two characters, group, horde, crowd, twins, clones, "
    "split screen, collage, grid, set, " + _NEGATIVE_QUALITY
)

# Step 2 (spritesheet): each FRAME is generated on its own and the strip is
# composed in PIL, so a frame wants exactly one character - same rules as step
# 1 - plus terms against the design drifting between frames. The old
# NEGATIVE_SHEET deliberately allowed grids, because step 2 used to ask the
# model for a whole row; it no longer does.
# MUST fit in 77 CLIP tokens. Everything past that is silently discarded -
# no error, no warning from diffusers, just a shorter prompt than written.
# This list reached 89 tokens and the dropped tail was the ENTIRE ground-shadow
# clause plus the end of the quality terms, so an earlier "fix" for the shadow
# under every sprite had literally never been applied. Keep it terse, keep the
# most important terms first, and check with check_prompt_length().
#
# Ordered by what actually goes wrong here: duplicates, then layout, then the
# ground shadow (connected to the feet, so it survives both the background key
# and _isolate_largest_sprite), then generic quality.
# The ground clause is longer than the rest suggests it needs to be, because
# there are two different ground problems and only one of them is fixed here.
# The patch INHERITED from the core is handled geometrically by
# strip_ground_patch - no prompt can remove pixels that are already in the init
# image. What is left after that is a smaller shadow step 2 draws fresh under
# the feet, and that one is a prompting problem. At 43 tokens there was room to
# say so properly, so it now names the shape ("dark patch under feet") rather
# than relying on "ground shadow" alone. Lands at 64 of 77.
NEGATIVE_FRAME = (
    "multiple characters, crowd, clones, sprite sheet, grid, collage, "
    "ground shadow, drop shadow, floor, grass, ground, dirt, soil, "
    "standing on ground, pedestal, dark patch under feet, contact shadow, "
    "blurry, deformed, extra limbs, cropped, low quality, watermark, text"
)

# Per-frame phase hints for a walk/action cycle. Each frame of a strip is
# generated from the same core image, so the seed and this hint are the only
# things that differ — they are what turns four copies into an animation
# rather than four identical stills.
PHASE_HINTS = (
    "contact pose, weight forward, leading leg extended",
    "passing pose, legs together, body at highest point",
    "contact pose mirrored, opposite leg extended",
    "passing pose mirrored, body settling down",
)


# What each UI action means, spelled out for the text encoder.
#
# The UI sends the checkbox value verbatim - "idle", "burning", "got damage" -
# and those went to the model almost unexpanded. Four tokens of "idle standing"
# is not a description of anything; the model fell back on whatever the core
# image and the checkpoint's own bias suggested, which is why several actions
# came back looking like the same neutral standing pose.
#
# WHAT BELONGS HERE, AND WHAT DOES NOT
#
# ControlNet owns geometry now. The skeleton states where every limb is, far
# more precisely than words can, and prompt text that argues with it makes both
# signals worse - the same reason PHASE_HINTS is suppressed on posed frames
# just below. So these do NOT describe limb positions.
#
# They describe the three things a skeleton cannot express:
#   * FACING - which way the camera sees the character. The skeleton encodes
#     this (side views collapse the shoulders onto one x), so saying it in
#     words reinforces the skeleton instead of fighting it.
#   * INTENT and energy - "aggressive", "off balance", "at rest". Mood is not
#     a joint position.
#   * EFFECTS - fire, smoke, embers. This is the big one. No skeleton can ask
#     for flames, which is exactly why the burning row came back as a zombie
#     standing calmly: nothing in the pipeline had ever mentioned fire.
#
# TOKEN BUDGET. CLIP keeps 77 tokens and silently bins the rest - see
# check_prompt_length. The trigger is only one part of the frame prompt; the
# core's own prompt and the style suffix share the same 77. Measured against a
# 31-token core prompt, the full frame prompt lands at 62-74 tokens with these,
# so there is room but not much. Keep additions under ~24 tokens each, and
# watch for the truncation warning rather than assuming it fits.
#
# The table itself lives in action_prompts.json, beside this file, so the
# wording can be tuned without touching Python or rebuilding the image. Its
# "_readme" block carries the authoring rules above in a form an LLM asked to
# edit the file will actually read.
ACTION_PROMPTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "action_prompts.json")

# (mtime, parsed actions). Re-read when the file changes, so editing prompts
# does not need a worker restart - which matters more than it sounds, because a
# restart also drops the loaded pipeline and costs a ~25s reload on the next
# sheet. Keyed on mtime rather than reloaded per call so a 16-frame sheet does
# not stat-and-parse the file 16 times.
_action_prompts_cache = (None, [])


def _clamp_strength(value, action_id):
    """Validate an optional per-action strength from JSON.

    Anything out of range is dropped rather than passed to the pipeline: a
    strength above 1.0 raises deep inside diffusers with a message that says
    nothing about this file, and a hand-typed "0.9" is far more likely to be a
    mistake than an intent - by 0.9 the character is gone and the skeleton's
    own limb colours start printing onto the sprite.
    """
    if value is None:
        return None
    try:
        s = float(value)
    except (TypeError, ValueError):
        logger.warning(f"action '{action_id}': strength {value!r} is not a "
                       "number; using the default.")
        return None
    if not 0.0 < s <= 0.85:
        logger.warning(f"action '{action_id}': strength {s} is outside the "
                       "usable 0-0.85 range; using the default.")
        return None
    return s


def _clamp_unit(value, action_id, field):
    """Validate an optional 0..1 knob from JSON, or None if absent/bad."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        logger.warning(f"action '{action_id}': {field} {value!r} is not a "
                       "number; ignoring.")
        return None
    if not 0.0 <= v <= 1.0:
        logger.warning(f"action '{action_id}': {field} {v} is outside 0..1; "
                       "ignoring.")
        return None
    return v


def load_action_prompts():
    """Parse action_prompts.json, or keep the last good copy on failure.

    A syntax error in a hand-edited prompt file must not take out sprite
    generation: the previous version stays live and the problem is logged.
    """
    global _action_prompts_cache
    try:
        mtime = os.path.getmtime(ACTION_PROMPTS_PATH)
    except OSError as e:
        if _action_prompts_cache[1]:
            return _action_prompts_cache[1]
        logger.error(f"{ACTION_PROMPTS_PATH} is unreadable ({e}); actions will "
                     "be sent to the model as their bare UI names.")
        return []

    if mtime == _action_prompts_cache[0]:
        return _action_prompts_cache[1]

    try:
        with open(ACTION_PROMPTS_PATH, encoding="utf-8") as fh:
            entries = json.load(fh).get("actions", [])
        parsed = [(tuple(e.get("match", ())), e.get("prompt", ""),
                   _clamp_strength(e.get("strength"), e.get("id")),
                   _clamp_unit(e.get("ip_adapter_scale"), e.get("id"),
                               "ip_adapter_scale"),
                   (e.get("view") or "").strip().lower() or None)
                  for e in entries if e.get("match") and e.get("prompt")]
        if not parsed:
            raise ValueError("no usable entries under 'actions'")
    except Exception as e:
        logger.error(f"Could not load {ACTION_PROMPTS_PATH}: {e}. Keeping the "
                     f"previously loaded {len(_action_prompts_cache[1])} entries.")
        return _action_prompts_cache[1]

    _action_prompts_cache = (mtime, parsed)
    logger.info(f"Loaded {len(parsed)} action prompts from action_prompts.json.")
    return parsed


def action_trigger(action: str) -> str:
    """Expand a UI action name into a full prompt fragment.

    Falls back to the raw action text, which is what an unrecognised action got
    before this table existed - a custom action typed by hand still reaches the
    model as its own words rather than being silently dropped.
    """
    return action_entry(action)["prompt"]


def action_entry(action: str) -> dict:
    """Prompt fragment and optional per-action overrides for a UI action.

    A dict rather than a tuple: this started as (prompt,), became
    (prompt, strength), and is now three fields. Every widening of a tuple
    silently breaks every caller that unpacked the old width.
    """
    a = (action or "").lower()
    for keys, text, strength, ipa, view in load_action_prompts():
        if any(k in a for k in keys):
            return {"prompt": text, "strength": strength,
                    "ip_adapter_scale": ipa, "view": view}
    return {"prompt": action, "strength": None, "ip_adapter_scale": None,
            "view": None}


def fit_into_frame(img, box, frame_w, frame_h):
    """Crop `box` out of `img` and centre it in a frame, preserving aspect."""
    crop = img.crop(box)
    if crop.width == 0 or crop.height == 0:
        return Image.new("RGBA", (frame_w, frame_h), (0, 0, 0, 0))

    scale = min(frame_w / crop.width, frame_h / crop.height)
    new_size = (max(1, int(crop.width * scale)), max(1, int(crop.height * scale)))
    # Pick the filter by DIRECTION. NEAREST is right for pixel art only when
    # scaling UP, where it keeps edges hard instead of blurring them into a
    # gradient. This path is almost always scaling DOWN, and hard by default:
    # frames render at 512 with the character ~423px tall, into a 128px frame -
    # a 3.3x reduction. Downscaling with NEAREST keeps every third pixel and
    # discards the other two, so a one-pixel outline survives or vanishes
    # depending on where it lands on the sampling grid. Thin detail breaks into
    # speckle, and that reads as the sprite being noisy rather than small.
    #
    # Measured on the zombie core, mean edge energy per opaque pixel (higher =
    # more high-frequency content, which at this scale is aliasing, not detail):
    #   into 128px:  NEAREST 52.07   BOX 43.95
    #   into 256px:  NEAREST 31.32   BOX 29.26
    # BOX area-averages the pixels being merged, which is what a downscale
    # should do. Above 1:1 nothing changes - NEAREST still wins there.
    #
    # This is also the argument for a larger frame size: at 256 the reduction
    # is only 1.65x and visibly more of the character survives, whichever
    # filter is used. The frame-size selector already offers 256 and 512.
    downscaling = scale < 1.0
    crop = crop.resize(new_size, Image.Resampling.BOX if downscaling
                       else Image.Resampling.NEAREST)

    frame = Image.new("RGBA", (frame_w, frame_h), (0, 0, 0, 0))
    # Centre horizontally, sit on the bottom edge: sprites share a ground line,
    # and centring vertically makes a walk cycle bob against its own baseline.
    frame.paste(crop, ((frame_w - crop.width) // 2, frame_h - crop.height), crop)
    return frame


def check_prompt_length(pipe, text: str, label: str):
    """Warn when a prompt will be silently truncated by CLIP.

    diffusers does not raise on an over-long prompt and does not warn either -
    transformers emits one line about "indexing errors" that is easy to miss in
    a wall of progress bars. Everything past 77 tokens is simply dropped, so a
    carefully worded exclusion at the end of a negative prompt can be absent
    from every generation while looking present in the source. That happened
    here: NEGATIVE_FRAME grew to 89 tokens and its ground-shadow clause never
    reached the model.

    Cheap enough to run per generation, but it is called once per sheet.
    """
    tok = getattr(pipe, "tokenizer", None)
    if tok is None or not text:
        return
    try:
        n = len(tok(text)["input_ids"])
    except Exception:
        return
    limit = getattr(tok, "model_max_length", 77)
    if n > limit:
        logger.warning(
            f"{label} is {n} tokens; CLIP keeps {limit}. The tail is being "
            f"DISCARDED, not applied. Shorten it."
        )


def resolve_sampling_params(llm_name: str, steps: int, cfg: float, negative_prompt: str = ""):
    """Reconcile caller-supplied sampling params with what the model expects.

    Single source of truth for every entry point — the sprite UI, the raw API,
    and the A1111 façade all funnel through here, so a caller using A1111's
    conventional defaults against a distilled checkpoint gets usable output
    instead of mush.

    Returns (steps, cfg, negative_prompt).
    """
    name = (llm_name or "").lower()

    if any(k in name for k in DISTILLED_MARKERS) and (steps > 8 or cfg > 1.0):
        logger.info(
            f"'{llm_name}' is a distilled checkpoint; clamping "
            f"steps {steps}->4, guidance {cfg}->0.0"
        )
        steps, cfg = min(steps, 4), 0.0

    # A negative prompt only does anything through classifier-free guidance, and
    # CFG is inactive at guidance_scale <= 1.0. Passing one anyway is a silent
    # no-op: the caller believes their exclusions are applied when they are not.
    # Say so once rather than letting it look like the negative prompt failed.
    if negative_prompt and cfg <= 1.0:
        logger.info(
            f"Ignoring negative prompt for '{llm_name}': guidance_scale={cfg} "
            "disables classifier-free guidance, so negatives have no effect. "
            "Use a non-distilled checkpoint with guidance > 1 if you need them."
        )
        negative_prompt = ""

    return steps, cfg, negative_prompt

# OpenPose ControlNet checkpoints, one per model family.
#
# A ControlNet is trained against a specific UNet and its blocks have that
# UNet's dimensions, so the SD1.5 checkpoint cannot condition an SDXL model or
# vice versa. Callers keep passing OPENPOSE_CONTROLNET; get_sd_pipeline
# substitutes the SDXL one when the checkpoint turns out to be SDXL, so no call
# site has to know which family it is driving.
#
# Both consume the same COCO-18 skeletons that poses.py renders - OpenPose
# keypoint format is shared - so poses.py needs no change. Its `Y` proportions
# may still want retuning if SDXL draws a different body plan.
OPENPOSE_CONTROLNET = "lllyasviel/control_v11p_sd15_openpose"
OPENPOSE_CONTROLNET_SDXL = "thibaud/controlnet-openpose-sdxl-1.0"


def _is_sdxl_checkpoint(llm_name: str) -> bool:
    """Decide the model family from the checkpoint's config, not its name.

    The previous rule was `"sdxl" in name or "turbo" in name`, which is a guess
    about how somebody chose to name a repo. It is wrong for at least two
    checkpoints this project wants: `stabilityai/stable-diffusion-xl-base-1.0`
    and `stablediffusionapi/pixel-art-diffusion-xl` are both SDXL and both fail
    it, and are then handed an SD1.5 pipeline class that cannot load them.

    `model_index.json` records the pipeline class the checkpoint was saved
    with, which is the actual answer. The name heuristic stays as a fallback
    for the case where the config cannot be read at all - offline with a cold
    cache - because guessing beats raising there.
    """
    try:
        from diffusers import DiffusionPipeline
        cfg = DiffusionPipeline.load_config(
            llm_name,
            cache_dir="/models",
            token=os.environ.get("HF_TOKEN") or None,
        )
        class_name = str(cfg.get("_class_name", ""))
        if class_name:
            return "XL" in class_name
    except Exception as e:
        logger.warning(
            f"Could not read model_index.json for '{llm_name}' "
            f"({e.__class__.__name__}); falling back to the name heuristic."
        )
    return "sdxl" in llm_name.lower() or "turbo" in llm_name.lower()


def get_sd_pipeline(llm_name: str = "stabilityai/sdxl-turbo",
                    pipeline_type: str = "text2img", controlnet: str = None):
    if llm_name == "models--stabilityai--sdxl-turbo":
        llm_name = "stabilityai/sdxl-turbo"
    global pipes

    # "<base>+<lora>" selects a base checkpoint with a LoRA fused on top.
    #
    # A LoRA is a small rank-decomposition delta over the base UNet's attention
    # weights - tens to hundreds of MB against a multi-GB base - so it is not a
    # model in its own right and cannot be listed like one. Encoding it in the
    # name means the whole existing surface keeps working unchanged: the UI
    # dropdowns, KNOWN_MODELS, the A1111 facade's sd_model_checkpoint, and the
    # `pipes` cache key all treat "base+lora" as one more model string.
    #
    # This is how a general 6.7GB SDXL base is made to draw actual pixel art:
    # the base supplies anatomy and prompt adherence, the LoRA supplies style.
    # Style finetunes at SD1.5 scale cannot match the first half, and SDXL base
    # alone measurably cannot do the second - it returns continuous-tone
    # illustration (see .ai/decisions/0002).
    lora = None
    if "+" in llm_name:
        llm_name, lora = (p.strip() for p in llm_name.split("+", 1))

    # The controlnet is part of the pipeline identity: the same checkpoint
    # with and without one are different objects, and keying them the same
    # hands back a plain img2img pipeline that rejects control_image. The LoRA
    # is part of that identity for the same reason.
    cache_key = f"{llm_name}_{pipeline_type}_{controlnet or 'none'}_{lora or 'nolora'}"
    if cache_key in pipes:
        return pipes[cache_key]
        
    log_cuda_details_once()

    # Evict other pipelines before loading, on CUDA only.
    #
    # `pipes` previously grew without bound. On a 12GB card that is not a cache,
    # it is a leak: SDXL-Turbo (~7GB) plus SD1.5 (~2GB) already drives the card
    # to thrashing — a load that takes ~20s alone stretched past 240s with a
    # second model resident — and a third would OOM. This service is meant to
    # serve many models, so keep exactly one on the GPU and reload on switch.
    # Reloading from the local cache costs seconds; thrashing costs minutes.
    #
    # On CPU the pipelines live in host RAM and swap, so leave that path alone.
    if DEVICE == "cuda" and pipes:
        evicted = sorted(pipes.keys())
        logger.info(f"Evicting {len(evicted)} pipeline(s) from VRAM before loading "
                    f"'{llm_name}': {evicted}")
        pipes.clear()
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    logger.info(f"Loading '{llm_name}' ({pipeline_type}) on {DEVICE.upper()} ({DTYPE})...")
    try:
        from diffusers import (StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline,
                               StableDiffusionPipeline, StableDiffusionImg2ImgPipeline,
                               StableDiffusionControlNetImg2ImgPipeline,
                               StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel)

        is_sdxl = _is_sdxl_checkpoint(llm_name)
        want_img2img = pipeline_type == "img2img"

        # ControlNet now works on both families. The checkpoint has to match the
        # UNet it conditions, so swap in the SDXL-trained openpose model when the
        # target turns out to be SDXL - callers pass OPENPOSE_CONTROLNET without
        # knowing which family they will get.
        cn_model = None
        if controlnet and want_img2img:
            cn_repo = controlnet
            if is_sdxl and controlnet == OPENPOSE_CONTROLNET:
                cn_repo = OPENPOSE_CONTROLNET_SDXL
                logger.info(f"'{llm_name}' is SDXL; substituting '{cn_repo}' for the "
                            "SD1.5 openpose checkpoint.")
            logger.info(f"Loading ControlNet '{cn_repo}'...")
            cn_model = ControlNetModel.from_pretrained(
                cn_repo, torch_dtype=DTYPE, cache_dir="/models",
                token=os.environ.get("HF_TOKEN") or None,
            )
        elif controlnet:
            logger.warning(f"Ignoring ControlNet '{controlnet}': only the img2img "
                           f"path conditions on it, and this is '{pipeline_type}'.")

        if is_sdxl and cn_model is not None:
            # SDXL UNet (~5GB fp16) + SDXL ControlNet (~2.5GB) + VAE and two text
            # encoders is roughly 9.5GB against ~11.7GB free. It fits with VAE
            # slicing, but it is the tightest combination this service loads -
            # see the OOM fallback after .to(DEVICE).
            pipeline_class = StableDiffusionXLControlNetImg2ImgPipeline
        elif is_sdxl:
            pipeline_class = StableDiffusionXLImg2ImgPipeline if want_img2img else StableDiffusionXLPipeline
        elif cn_model is not None:
            pipeline_class = StableDiffusionControlNetImg2ImgPipeline
        else:
            # The SD1.5 branch previously ignored pipeline_type and always built
            # a text2img pipeline, so asking for img2img silently returned the
            # wrong class and the `image=` argument was rejected at call time.
            pipeline_class = StableDiffusionImg2ImgPipeline if want_img2img else StableDiffusionPipeline
            
        common = dict(
            torch_dtype=DTYPE,
            cache_dir="/models",
            # `or None` is load-bearing: compose always defines HF_TOKEN, so an
            # unconfigured token arrives as "" rather than being absent. Passing
            # the empty string makes huggingface_hub send an empty credential,
            # which can fail auth on repos that need none at all — SDXL-Turbo is
            # ungated and must work with no token.
            token=os.environ.get("HF_TOKEN") or None,
        )
        if cn_model is not None:
            common["controlnet"] = cn_model

        if not is_sdxl:
            # Disable SD1.5's safety checker.
            #
            # It false-positives badly on pixel art and REPLACES the output with
            # a solid black image rather than raising, so the pipeline reports
            # success and every downstream step happily processes a blank frame:
            #   "Potential NSFW content was detected ... A black image will be
            #    returned instead."
            # That silently produced entirely blank spritesheets. This is a
            # local, single-user service generating game sprites from prompts the
            # operator wrote, so the checker costs correctness and buys nothing.
            # SDXL pipelines ship without one, hence the is_sdxl guard.
            common["safety_checker"] = None
            common["requires_safety_checker"] = False

        # Prefer the fp16 weight variant where the repo publishes one.
        #
        # SDXL-Turbo's default weights are fp32: ~13GB on disk, and the load
        # materialises them in host RAM before casting to fp16. This host gives
        # WSL 11GB, so that swaps and a load that should take seconds ran past a
        # 240s request timeout. The fp16 variant is the same model at half the
        # bytes. Not every repo publishes one, hence the fallback.
        try:
            pipe = pipeline_class.from_pretrained(llm_name, variant="fp16", **common)
            logger.info(f"Loaded '{llm_name}' using the fp16 weight variant.")
        except Exception as variant_err:
            logger.info(f"No fp16 variant for '{llm_name}' ({variant_err.__class__.__name__}); "
                        "falling back to default weights.")
            pipe = pipeline_class.from_pretrained(llm_name, **common)

        if lora:
            # Fuse rather than just load. `load_lora_weights` keeps the adapter
            # as a separate set of tensors consulted on every forward pass;
            # `fuse_lora` folds the delta into the base weights once, which costs
            # nothing at inference and avoids the adapter being silently dropped
            # by pipeline surgery later (ControlNet swaps, VAE config edits).
            #
            # Failure here is NOT fatal: a missing or incompatible LoRA should
            # degrade to the plain base model with a loud warning, not take down
            # a generation. An SDXL LoRA on an SD1.5 base raises here, which is
            # the common mistake - the ranks do not match.
            try:
                logger.info(f"Loading LoRA '{lora}' onto '{llm_name}'...")
                pipe.load_lora_weights(
                    lora, cache_dir="/models",
                    token=os.environ.get("HF_TOKEN") or None,
                )
                pipe.fuse_lora()
                logger.info(f"LoRA '{lora}' fused.")
            except Exception as lora_err:
                logger.warning(
                    f"Could not apply LoRA '{lora}' to '{llm_name}': "
                    f"{lora_err.__class__.__name__}: {lora_err}. "
                    "Continuing with the base checkpoint only - output will not "
                    "carry the LoRA's style."
                )

        try:
            pipe.to(DEVICE)
        except torch.cuda.OutOfMemoryError:
            # Only SDXL-plus-ControlNet gets close enough to the card's limit for
            # this to fire (~9.5GB of ~11.7GB free), and it fires as a hard error
            # rather than degrading. Offload streams weights through system RAM
            # instead - much slower, but it produces an image rather than a 500.
            # The WSL memory cap is the real ceiling for this path, not VRAM.
            logger.warning(
                f"'{llm_name}' did not fit in VRAM; retrying with "
                "enable_model_cpu_offload(). Expect a large slowdown."
            )
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            pipe.enable_model_cpu_offload()
        if DEVICE == "cuda":
            # SDXL decode of a 1024px latent spikes VRAM well past the UNet's
            # working set; slicing keeps the peak flat on a 12GB card.
            pipe.enable_vae_slicing()

            if is_sdxl:
                # The original SDXL VAE overflows in fp16 and decodes to pure
                # black — no exception, just black PNGs, which reads as a broken
                # prompt or a broken GPU. diffusers ships force_upcast=True to
                # run the decode in fp32, but a checkpoint can carry a config
                # that turns it off, so assert it rather than assume.
                vae_cfg = getattr(pipe.vae, "config", None)
                if vae_cfg is not None and not getattr(vae_cfg, "force_upcast", True):
                    logger.warning(
                        "SDXL VAE has force_upcast=False under fp16; re-enabling "
                        "to avoid black images."
                    )
                    pipe.vae.config.force_upcast = True
        pipes[cache_key] = pipe
    except Exception as e:
        logger.error(f"Error loading model '{llm_name}' ({pipeline_type}): {e}")
        return None
    return pipes[cache_key]

# Hot-patch for FLUX.2-klein GGUF support in diffusers
def apply_klein_patch():
    try:
   
        # Define the keys known to be missing in Klein/Pruned Flux models
        KLEIN_MISSING_KEYS = {
            "time_in.in_layer.bias": (256,),
            "time_in.out_layer.bias": (3072,),
            "vector_in.in_layer.weight": (256, 768),
            "vector_in.in_layer.bias": (256,),
            "guidance_in.in_layer.bias": (256,),
            "guidance_in.out_layer.bias": (3072,),
        }

        orig_func = sf_utils.convert_flux_transformer_checkpoint_to_diffusers

        def patched_convert(checkpoint, *args, **kwargs):
            # Inject zeros for missing keys so checkpoint.pop() doesn't crash
            # Use bfloat16 as it's the primary inference dtype for FLUX on CPU
            for key, shape in KLEIN_MISSING_KEYS.items():
                if key not in checkpoint:
                    checkpoint[key] = torch.zeros(shape, dtype=torch.bfloat16)
            return orig_func(checkpoint, *args, **kwargs)

        # Force the patch into the module
        sf_utils.convert_flux_transformer_checkpoint_to_diffusers = patched_convert
        print("FLUX.2-klein compatibility patch is now ACTIVE.", flush=True)
    except Exception as e:
        print(f"Failed to apply FLUX.2-klein patch: {e}", flush=True)

apply_klein_patch()

def get_flux_pipeline(pipeline_type: str = "img2img"):
    llm_name = "flux-2-klein-4b-Q8_0.gguf"
    global pipes
    
    cache_key = f"flux_{pipeline_type}"
    if cache_key in pipes:
        return pipes[cache_key]
        
    # On CUDA use fp16 (see DTYPE above). On CPU keep bfloat16: float32 would
    # need ~23GB, bfloat16 halves it to ~11.5GB — still more than this host's
    # WSL allocation, which is precisely why the GPU path exists.
    dtype = DTYPE if DEVICE == "cuda" else torch.bfloat16

    # Clear cache if memory is tight or model changes
    if len(pipes) > 0:
        logger.info("Clearing pipeline cache to free memory...")
        pipes.clear()
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # flux1-schnell.safetensors contains only the transformer weights.
    # Load the transformer from local safetensors, then build the full pipeline
    # using the cached HuggingFace snapshot for all other components (CLIP, T5, VAE, scheduler).
    safetensors_path = "/models/flux1-schnell.safetensors"
    base_repo = "black-forest-labs/FLUX.1-schnell"
    hf_cache = "/models/models--black-forest-labs--FLUX.1-schnell/snapshots/741f7c3ce8b383c54771c7003378a50191e9efe9"
    logger.info(f"Loading FLUX transformer from: {safetensors_path}")
    try:
        from diffusers import FluxImg2ImgPipeline, FluxPipeline, FluxTransformer2DModel

        pipeline_class = FluxImg2ImgPipeline if pipeline_type == "img2img" else FluxPipeline

        logger.info("Loading FluxTransformer2DModel from local safetensors...")
        transformer = FluxTransformer2DModel.from_single_file(
            safetensors_path,
            torch_dtype=dtype,
            config=hf_cache,
            subfolder="transformer",
        )

        logger.info(f"Assembling full pipeline from cached HF snapshot: {hf_cache}")
        if pipeline_type == "img2img":
            p = FluxImg2ImgPipeline.from_pretrained(
                hf_cache,
                transformer=transformer,
                torch_dtype=dtype,
                local_files_only=True,
            )
        else:
            p = FluxPipeline.from_pretrained(
                hf_cache,
                transformer=transformer,
                torch_dtype=dtype,
                local_files_only=True,
            )
        
        # Tiling + slicing keep VAE peak memory flat on both paths: it is the
        # difference between fitting and OOM on a 12GB card at sheet widths.
        logger.info(f"Stabilizing VAE for {DEVICE.upper()} ({dtype} + tiling + slicing)...")
        p.vae.to(dtype=dtype)
        p.vae.enable_tiling()
        p.vae.enable_slicing()
        
        # CPU Dtype Alignment: Force the ENTIRE pipeline to match the target dtype
        logger.info(f"Strict alignment: Moving entire pipeline to {dtype}...")
        p.transformer.to(dtype=dtype)
        p.text_encoder.to(dtype=dtype)
        if hasattr(p, "text_encoder_2") and p.text_encoder_2:
            p.text_encoder_2.to(dtype=dtype)
        
        # Diagnostic: Verify dtypes of major components
        def get_mod_dtype(mod):
            try:
                # Check first parameter for dtype
                return next(mod.parameters()).dtype
            except:
                return "unknown"
        
        logger.info(f"DIAGNOSTIC - Transformer: {get_mod_dtype(p.transformer)}, "
                    f"VAE: {get_mod_dtype(p.vae)}, "
                    f"TextEncoder1: {get_mod_dtype(p.text_encoder)}")
        
        # LoRA Loading Logic
        lora_path = "/models/flux-spritesheet-lora.safetensors"
        if os.path.exists(lora_path):
            try:
                logger.info(f"Loading LoRA weights from {lora_path}...")
                p.load_lora_weights(lora_path)
                # p.fuse_lora() # Optional: Fuse for a slight speedup if stable
                logger.info("LoRA loaded successfully.")
            except Exception as lora_e:
                logger.warning(f"LoRA loading failed, continuing without it: {lora_e}")

        if DEVICE == "cuda":
            # Do NOT .to("cuda") here. A full FLUX pipeline at fp16 is ~24GB —
            # double this card. enable_model_cpu_offload keeps only the module
            # currently executing (T5, then transformer, then VAE) resident and
            # streams the rest from system RAM. accelerate does the hooking.
            # NOTE: this stages weights through host RAM, so WSL's memory cap is
            # the real ceiling here, not VRAM. See README "WSL2 memory".
            logger.info("Enabling sequential model CPU offload for FLUX on CUDA...")
            p.enable_model_cpu_offload()
        else:
            p.to("cpu")
        pipes[cache_key] = p
        logger.info(f"FLUX pipeline loaded and ready on {DEVICE.upper()}.")
    except Exception as e:
        logger.error(f"Error loading FLUX pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    return pipes[cache_key]


def get_db():
    if not DB_URL:
        return None
    try:
        return psycopg2.connect(DB_URL)
    except Exception as e:
        logger.error(f"DB connection failed: {e}")
        return None

def update_task_record(task_id: str, file_path: str = None, duration_ms: float = 0, 
                       error_msg: str = None, progress_pct: int = None, progress_msg: str = None,
                       image_type: str = None, parent_id: int = None, components: list = None,
                       requested_actions: list = None, seed: int = None, sub_task_ids: list = None):
    conn = get_db()
    if not conn:
        return
    try:
        with conn:
            with conn.cursor() as cur:
                update_fields = []
                values = []
                if file_path is not None:
                    update_fields.append("file_path = %s")
                    values.append(file_path)
                    # A row with an image succeeded, so it must not still carry
                    # an error. Matters because the reaper below writes one on
                    # every restart, and the queue renders error before
                    # file_path — a retried task would come back looking failed.
                    # Guarded: assigning the same column twice in one UPDATE is
                    # a Postgres error, and one caller passes both.
                    if error_msg is None:
                        update_fields.append("error = NULL")
                if duration_ms > 0:
                    update_fields.append("duration_ms = %s")
                    values.append(duration_ms)
                if error_msg is not None:
                    update_fields.append("error = %s")
                    values.append(error_msg)
                if progress_pct is not None:
                    update_fields.append("progress_pct = GREATEST(COALESCE(progress_pct, 0), %s)")
                    values.append(progress_pct)
                if progress_msg is not None:
                    update_fields.append("progress_msg = %s")
                    values.append(progress_msg)
                if image_type is not None:
                    update_fields.append("image_type = %s")
                    values.append(image_type)
                if parent_id is not None:
                    update_fields.append("parent_id = %s")
                    values.append(parent_id)
                if components is not None:
                    update_fields.append("components = %s")
                    values.append(json.dumps(components))
                if requested_actions is not None:
                    update_fields.append("requested_actions = %s")
                    values.append(json.dumps(requested_actions))
                if seed is not None:
                    update_fields.append("seed = %s")
                    values.append(seed)
                if sub_task_ids is not None:
                    update_fields.append("sub_task_ids = %s")
                    values.append(json.dumps(sub_task_ids))
                
                if update_fields:
                    values.append(task_id)
                    cur.execute(
                        f"UPDATE sprite_images SET {', '.join(update_fields)} WHERE task_id = %s",
                        tuple(values)
                    )
    except Exception as e:
        logger.error(f"Could not update record {task_id}: {e}")
    finally:
        conn.close()

def get_core_image_path(parent_id: int):
    conn = get_db()
    if not conn: return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT file_path FROM sprite_images WHERE id = %s", (parent_id,))
            row = cur.fetchone()
            return row[0] if row else None
    except Exception as e:
        logger.error(f"Error fetching core image path: {e}")
        return None
    finally:
        conn.close()

def log_stats(task_id, llm_name, clean_prompt, total_steps, start_time, end_time, total_duration_ms):
    try:
        import requests
        tokens = float(len(str(clean_prompt).split()) * 10)
        requests.post(
            "http://stats_collector:8000/v1/internal/log_stats",
            json={
                "model_name": llm_name,
                "prompt_tokens": tokens,
                "completion_tokens": float(total_steps),
                "total_tokens": tokens + float(total_steps),
                "tokens_per_second": float(total_steps) / max(end_time - start_time, 0.001),
                "prompt_eval_ms": 0.0,
                "total_duration_ms": total_duration_ms,
            },
            timeout=2,
        )
    except Exception as e:
        logger.error(f"Could not log stats for {task_id}: {e}")

def _isolate_largest_sprite(arr):
    """Drop every opaque blob except the biggest one.

    SDXL-Turbo runs at guidance 0, so classifier-free guidance is OFF and the
    negative prompt - "multiple characters, group, crowd, twins, clones" and
    all - has no effect whatsoever. It duplicates the subject anyway: a
    generated core came back as one large character ringed by five small copies
    of itself, and step 2 then faithfully carried all six into every animation
    frame. Step-2 tuning was chasing a step-1 defect.

    Prompting cannot fix this on a distilled checkpoint. Geometry can. A sprite
    is one connected shape, so keep the largest connected opaque region and
    delete the rest - deterministic, and independent of the sampler.

    8-connectivity, so a limb or outline joined only diagonally is not severed
    from its own body.
    """
    import numpy as np
    from scipy import ndimage

    opaque = arr[:, :, 3] > 0
    labels, n = ndimage.label(opaque, structure=np.ones((3, 3), dtype=bool))
    if n <= 1:
        return arr

    sizes = ndimage.sum(opaque, labels, range(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    dropped = int(opaque.sum() - sizes[keep - 1])
    arr[(labels != keep) & opaque] = (0, 0, 0, 0)
    logger.info(f"Isolated main sprite: removed {n - 1} stray blob(s), {dropped} px.")
    return arr


# Deriving a side-facing core, for actions whose skeleton is in profile.
#
# WHY THIS EXISTS. img2img cannot rotate a character - it preserves composition,
# and which way a body faces IS composition. So a profile skeleton over a
# camera-facing core asks for a turn the init image forbids, and 'move right'
# rendered as a front-facing zombie shifting its weight while a walk skeleton
# was ignored. Raising strength far enough to force the turn (0.90+) destroyed
# the character, which is why this went unsolved.
#
# WHAT CHANGED. The IP-Adapter. It was rejected for burning because it
# suppresses anything the reference does not contain - but a rotation ADDS
# nothing. The whole character is present in the reference; only the viewing
# angle differs. So it holds identity at a strength high enough to actually
# turn the body, which is a combination nothing else offered.
#
# Swept strength x adapter scale, judged on the rendered images:
#   ipa 0.0  turns, but degrades into a blob at every strength
#   ipa 0.4  clean profile, character intact          <- chosen
#   ipa 0.7  clings to the front-facing core, will not turn
# Aspect ratio was tried as an automatic "is it in profile" score and is NOT
# reliable - a forward-reaching arm widens the bounds and it ranked ipa 0.4
# worst when ipa 0.4 was visibly best. These values come from looking.
SIDE_CORE_STRENGTH = 0.90
SIDE_CORE_IPA_SCALE = 0.4


VIEW_PROMPTS = {
    "back": "back view seen from directly behind, facing away from the camera, "
            "back of the head, no face, no eyes",
    "right": "strict side view profile, facing right, standing in profile, "
             "head turned right",
    "left": "strict side view profile, facing left, standing in profile, "
            "head turned left",
}


def derive_view_core(pipe, core_rgb, control, base_prompt, seed, view,
                     llm_name=None):
    """Render a profile-facing version of a core, or None if unavailable.

    Returns None rather than raising: a sheet rendered from the front core is
    the previous behaviour, which is worse but not broken, and losing a whole
    generation job over an optional enhancement is the wrong trade.

    Always produced facing RIGHT. A left-facing core is that image mirrored -
    free, and it guarantees the two directions are the same character, which
    two independent generations would not.
    """
    if not getattr(pipe, "_ipa_loaded", False):
        logger.warning("Side-facing core needs the IP-Adapter and it is not "
                       "loaded; falling back to the front core.")
        return None
    # The trained trigger LEADS the derivation prompt when the checkpoint has
    # one, with the descriptive wording kept behind it as reinforcement. This is
    # where a view trigger belongs, and where an earlier attempt got it wrong:
    # applying the trigger to the per-frame prompt instead did nothing visible,
    # because at frame settings (strength 0.75, adapter 0.5) the front init and
    # the adapter outvote it. Derivation runs at 0.90 with a weaker adapter,
    # which is the only place there is enough freedom to turn a body.
    vt = view_trigger(llm_name, view)
    described = VIEW_PROMPTS.get(view, VIEW_PROMPTS["right"])
    view_text = f"{vt}, {described}" if vt else described
    try:
        pipe.set_ip_adapter_scale(SIDE_CORE_IPA_SCALE)
        img = pipe(
            prompt=(f"{view_text}, {base_prompt}, "
                    "single character, full body"),
            negative_prompt=NEGATIVE_FRAME,
            image=core_rgb,
            ip_adapter_image=core_rgb,
            strength=SIDE_CORE_STRENGTH,
            num_inference_steps=30,
            guidance_scale=8.0,
            generator=torch.Generator("cpu").manual_seed(seed),
            control_image=control,
            controlnet_conditioning_scale=CONTROLNET_SCALE,
        ).images[0]
        keyed = remove_background(img, keep_largest=True)
        logger.info(f"Derived '{view}' core"
                    + (f" via trained trigger {vt}" if vt else " by prompt")
                    + f" (strength {SIDE_CORE_STRENGTH}, "
                    f"ip_adapter {SIDE_CORE_IPA_SCALE}); bounds {keyed.getbbox()}.")
        return keyed
    except Exception as e:
        logger.warning(f"Could not derive a '{view}' core ({e}); using the "
                       "front core, so this action will not turn.")
        return None


def strip_ground_patch(img, width_ratio: float = 1.8, max_band_frac: float = 0.25):
    """Delete a ground/shadow patch fused to the bottom of a sprite.

    NEGATIVE_SINGLE now asks step 1 not to draw one, but prompting is a request
    rather than a guarantee, and every core generated before that change still
    has one baked in. This is the geometric backstop, and it is needed because
    the two existing cleanups both structurally cannot see this:
    remove_background only clears regions reachable from the border and the
    patch is enclosed by the sprite; _isolate_largest_sprite keeps the largest
    connected blob and the patch is JOINED to the feet, so it is part of it.

    Detection is by WIDTH, not colour. Colour would have to know that this
    particular ground is brown, which does not generalise past one character -
    a stone slab or a pool of shadow is not brown. Width does generalise: feet
    are narrow and the thing they stand on spreads out. Measured on the zombie
    core, legs run ~99px and the patch peaks at 319px, a 3.2x step that no
    part of a character body produces.

    `width_ratio` is how much wider than the body a row must be to count as
    ground; 1.5 and 1.8 both cleared this core, 2.2 left a rim behind, and 2.6
    missed it entirely. `max_band_frac` caps the damage at the bottom quarter
    of the sprite, so a misfire cannot eat the legs.

    Deliberately NOT applied to generated frames - a wide attack or a flared
    robe genuinely is wider at the bottom, and clipping that would be worse
    than the patch. It runs on the core, once, and every frame inherits the
    clean version through img2img.

    KNOWN GAP - measured on core_21f88cbbe9b4 (stocky zombie, arm outstretched):

        body median (reference)  194    <- inflated by the raised arm
        patch peak               344    <- ratio 1.77, just under width_ratio
        boots                    206-209 <- ratio 1.06, just over the 1.05 walk

    Both numbers land on the wrong side of their limit at the same time, so the
    patch is missed AND lowering width_ratio does not rescue it: the hysteresis
    walk would then run up through the boots and amputate the feet. The rule
    assumes a body whose reference width is dominated by the torso and whose
    feet are clearly narrower than it - true for the slim zombie it was tuned on
    (legs 0.77x, patch 3.2x), false for a stocky character holding an arm out.

    A per-row gradient detector was tried as a replacement and rejected: the
    shadow ramps in over ~12 rows rather than stepping, so the largest single
    row jump is the ankle-to-boot flare (+20%), not the ground. A windowed
    version would need a threshold that separates +36% (boots) from +66%
    (ground) - another number fitted to two samples, so it was not adopted.

    This is cosmetic in 2D. It is NOT cosmetic for the 3D path, where the patch
    reconstructs as a disc fused to the soles: on charB it took silhouette IoU
    from 0.908 to 0.412 and turned a 19-bone branching skeleton into a 6-bone
    straight chain. Anything feeding TripoSR should be visually checked until
    this gap is closed.
    """
    import numpy as np

    arr = np.array(img.convert("RGBA"))
    opaque = arr[:, :, 3] > 0
    rows = np.where(opaque.any(axis=1))[0]
    if rows.size == 0:
        return img

    top, bot = int(rows[0]), int(rows[-1])
    height = bot - top + 1
    widths = opaque.sum(axis=1).astype(float)

    # Reference body width, measured ABOVE the bottom quarter so the patch
    # cannot inflate the very number it is being compared against.
    body = widths[top:max(bot + 1 - int(height * 0.25), top + 1)]
    body = body[body > 0]
    if body.size == 0:
        return img
    reference = float(np.median(body))
    if reference <= 0:
        return img

    band = np.arange(max(bot - int(height * max_band_frac), top), bot + 1)
    wide = band[widths[band] > reference * width_ratio]
    if wide.size == 0:
        return img

    # Cut from the TOPMOST offending row, not upward from the bottom: the patch
    # tapers to a few pixels of grass at its lowest rows, so a scan starting at
    # the bottom edge stops on the first narrow row and removes nothing.
    cut = int(wide.min())

    # Then walk further up with a LOWER threshold. Two thresholds, because one
    # cannot do both jobs: the high one has to be high enough that a wide stance
    # or a flared coat is not mistaken for ground, but the patch does not stop
    # abruptly at that width - it tapers. Cutting only where it exceeds the high
    # threshold left the taper behind, which is exactly the residual shadow that
    # then looked like step 2 drawing a fresh one.
    #
    # Measured on the zombie core: body median 128, legs ~99, patch 134-319. The
    # high threshold (230) first fires at row 464; extending up while rows stay
    # above 1.05x the body width carries the cut to the leg/ground boundary near
    # 447, where the width drops to 91 and stops it. Legs are NARROWER than the
    # body median, so they cannot sustain this walk - the taper can.
    floor = max(top, bot - int(height * max_band_frac))
    while cut - 1 >= floor and widths[cut - 1] > reference * 1.05:
        cut -= 1
    removed = int(opaque[cut:bot + 1].sum())
    arr[cut:bot + 1, :, 3] = 0
    logger.info(f"Stripped ground patch: {removed} px below row {cut} "
                f"(body width {reference:.0f}, cut threshold "
                f"{reference * width_ratio:.0f}).")
    return Image.fromarray(arr)


def remove_background(master, tolerance: int = 22, keep_largest: bool = False):
    """Make the background transparent, keeping interior detail intact.

    Only removes background-coloured regions **connected to the image border**.
    The previous implementation matched the background colour globally, which
    punched transparent holes through every matching pixel inside the sprite —
    white teeth, pale clothing and highlights all vanished on a white
    background. Flood-filling from the edge is also the right model for pixel
    art, where hard edges matter and a segmentation network's soft alpha would
    be wrong.
    """
    try:
        master = master.convert("RGBA")
        corners = [(0,0), (master.width-1, 0), (0, master.height-1), (master.width-1, master.height-1)]
        bg_r, bg_g, bg_b = 255, 255, 255

        # Take the colour MOST corners agree on. The old rule took the
        # first corner brighter than a fixed threshold, which silently
        # refused to key a dark background: a black-backed frame fell
        # through to the white default, matched nothing and stayed fully
        # opaque. A majority vote also survives a sprite that happens to
        # reach into one corner, which the first-match rule did not.
        samples = [master.getpixel(c)[:3] for c in corners]
        bg_r, bg_g, bg_b = max(set(samples), key=samples.count)

        # Vectorised with numpy. This was a per-pixel Python loop: 262k iterations
        # for a 512x512 image, running on every generated image and every action
        # strip. On a 4-thread CPU that cost seconds per sprite — immediately
        # after the GPU produced the image in under one. numpy arrives with torch.
        import numpy as np
        from scipy import ndimage

        arr = np.array(master, dtype=np.int16)          # H x W x 4 (RGBA)
        rgb = arr[:, :, :3]
        bg = np.array([bg_r, bg_g, bg_b], dtype=np.int16)

        # Every pixel close enough to the sampled background colour...
        colour_match = np.all(np.abs(rgb - bg) < tolerance, axis=-1)

        # ...but only the regions REACHABLE FROM THE BORDER are background.
        # Interior pixels that happen to share the colour (teeth, pale clothing,
        # highlights on a white background) must survive. 4-connectivity matches
        # how pixel art reads: diagonal-only gaps are not leaks.
        labels, n = ndimage.label(colour_match)
        if n == 0:
            return master

        border = np.concatenate([
            labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]
        ])
        border_labels = np.unique(border[border > 0])
        mask = np.isin(labels, border_labels)

        total_pix = mask.size
        pixels_removed = int(mask.sum())

        if pixels_removed / total_pix > 0.98:
            # Almost everything is border-connected background, which means the
            # corner sample was not background at all (e.g. a full-bleed sprite).
            # Wiping it would return an empty image, so keep the original.
            logger.warning(
                f"Safety trigger: background removal would clear "
                f"{pixels_removed / total_pix:.1%} of pixels. Keeping background."
            )
            return master

        interior_kept = int(colour_match.sum()) - pixels_removed
        if interior_kept:
            logger.info(
                f"Background removed ({pixels_removed / total_pix:.1%}); kept "
                f"{interior_kept} interior pixels matching the background colour."
            )

        arr[mask] = (0, 0, 0, 0)

        if keep_largest:
            arr = _isolate_largest_sprite(arr)

        return Image.fromarray(arr.astype(np.uint8), mode="RGBA")
    except Exception as e:
        logger.error(f"BG removal failed: {e}")
        return master


@celery_app.task(name="tasks.generate_core_task", bind=True)
def generate_core_task(self, prompt: str, llm_name: str = "stabilityai/sdxl-turbo"):
    task_id = self.request.id
    logger.info(f"Task {task_id} generated core with llm {llm_name}")
    p = get_sd_pipeline(llm_name)
    if not p:
        update_task_record(task_id, error_msg="Model failed to load on worker")
        return {"error": "Model failed to load"}

    seed = random.randint(0, 10**9)
    generator = torch.Generator("cpu").manual_seed(seed)
    negative = NEGATIVE_SINGLE

    clean_prompt = prompt.replace("PixelartFSS", "").strip().lstrip(",").strip()
    
    # Strictly aligned prefix: "PixelartFSS, idle front,"
    # Duplicate suppression lives HERE, in the positive prompt, and is worded
    # as an assertion rather than a negation.
    #
    # This checkpoint is distilled and runs at guidance 0, so classifier-free
    # guidance is off and the negative prompt does nothing whatsoever. The old
    # prompt tried to compensate by stating the exclusions positively - "lone
    # character, no duplicates, one standalone character" - and that backfired
    # twice over. A text encoder cannot apply "no"; the token "duplicates"
    # simply lands in the conditioning. And "PixelartFSS" is a trigger word for
    # the SD1.5 sheet checkpoint, not for this one, where it reads as generic
    # sprite-sheet flavouring and invites a grid.
    #
    # It produced cores that were a crowd: one large character ringed by five
    # to nine smaller copies, sometimes touching it, which then defeated
    # _isolate_largest_sprite too - a crowd that overlaps is one blob.
    #
    # Measured over three seeds, share of opaque pixels in the main figure:
    #   old prompt   99.9 / 93.2 / 99.8 %, with up to 44 stray regions
    #   this prompt  100 / 100 / 100 %, and clean across 8 varied subjects
    #
    # Raising guidance to 2.0-3.5 also fixes the duplication, by making the
    # negative prompt work at all - but it visibly flattens the pixel art,
    # posterising faces and detail, and costs 2-3x the steps. Not worth it when
    # the prompt alone does the job at 4 steps.
    background = ("" if "background" in clean_prompt.lower()
                  else ", plain white background")
    trigger = CORE_TRIGGERS.get((llm_name or "").strip().lower())
    if trigger:
        logger.info(f"Using trigger '{trigger}' for '{llm_name}'.")
    full_prompt = (f"{trigger + ', ' if trigger else ''}"
                   f"a single {clean_prompt}, one character alone, full body, "
                   f"standing, centered, pixel art sprite, 16-bit, sharp focus"
                   f"{background}")

    start_time = time.time()
    
    # Dynamic parameters based on model type. resolve_sampling_params also drops
    # the negative prompt when guidance is 0 — on a distilled checkpoint there is
    # no classifier-free guidance for it to act through, so the long exclusion
    # list below is inert and would otherwise look like it was being applied.
    is_turbo = any(k in llm_name.lower() for k in DISTILLED_MARKERS)
    num_steps = 4 if is_turbo else 35
    guidance = 0.0 if is_turbo else 9.0
    num_steps, guidance, negative = resolve_sampling_params(
        llm_name, num_steps, guidance, negative
    )

    try:
        update_task_record(task_id, progress_pct=0, progress_msg="Generating core image...", seed=seed)

        def progress_callback(pipe, i, t, callback_kwargs):
            pct = int((i / num_steps) * 100)
            logger.info(f"  > Core generation progress: {pct}%")
            if i % 1 == 0:
                update_task_record(task_id, progress_pct=pct, progress_msg=f"Generating: {int(pct)}%")
                self.update_state(state="PROGRESS", meta={"pct": pct, "msg": "Generating core image"})
            return callback_kwargs

        img = p(
            full_prompt,
            negative_prompt=negative or None,
            height=512,
            width=512,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            generator=generator,
            callback_on_step_end=progress_callback
        ).images[0]
            
    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}", exc_info=True)
        update_task_record(task_id, error_msg=f"Generation failed: {str(e)}")
        return {"error": str(e)}

    end_time = time.time()
    total_duration_ms = (end_time - start_time) * 1000
    log_stats(task_id, llm_name, clean_prompt, num_steps, start_time, end_time, total_duration_ms)

    update_task_record(task_id, progress_pct=90, progress_msg="Finalizing: Removing background...")
    # keep_largest: a core MUST be a single character. See the note in
    # _isolate_largest_sprite - on a distilled checkpoint the negative
    # prompt cannot enforce that, so the geometry does.
    img = remove_background(img, keep_largest=True)
    # After the background key, because the patch has to be a distinct alpha
    # shape before its width can be measured against the body's. The negative
    # prompt asks step 1 not to draw ground at all; this catches the times it
    # does anyway, so a core is clean the moment it is saved rather than being
    # repaired on every sheet that later uses it.
    img = strip_ground_patch(img)

    # Smart Aspect Ratio Detection:
    # If the model natively generates a 4x1 animation sequence, strip out Frame 1.
    # Otherwise, if it produces a square character image, do not crop.
    width, height = img.size
    if width > (height * 3.5):
        logger.info(f"Detected 4x1 sheet ({width}x{height}). Cropping first frame as core.")
        frame_width = width // 4
        core_crop = img.crop((0, 0, frame_width, height))
    else:
        logger.info(f"Detected square core ({width}x{height}). No cropping needed.")
        core_crop = img
    
    filename = f"core_{uuid.uuid4().hex[:12]}.png"
    filepath = os.path.join(IMAGES_DIR, filename)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    core_crop.save(filepath, format="PNG")

    update_task_record(task_id, file_path=filepath, duration_ms=total_duration_ms, 
                       error_msg=None, progress_pct=100, progress_msg="Complete", image_type="core", seed=seed)

    return {"status": "success", "url": f"/images/{filename}", "duration_ms": total_duration_ms }

@celery_app.task(name="tasks.generate_spritesheet_task", bind=True)
def generate_spritesheet_task(self, parent_id: int, actions: list, llm_name: str, frame_width: int, frame_height: int, motion_steps: int):
    task_id = self.request.id
    sheet_start = time.time()
    logger.info(f"Task {task_id} generating sheet with {llm_name}, actions: {actions}")
    
    update_task_record(task_id, progress_pct=5, progress_msg="Loading context...")
    
    # Fetch Core Image
    core_path = get_core_image_path(parent_id)
    core_img = None
    if core_path and os.path.exists(core_path):
        # Composite onto WHITE before dropping the alpha channel.
        # .convert("RGB") on an RGBA sprite maps every transparent pixel to
        # BLACK. img2img therefore started from a black-backed character,
        # produced black-backed frames, and remove_background could not key
        # them (it will not treat a dark corner as background) - so the
        # finished sheet came back 100% opaque with a black background, and
        # every frame was unusable as a sprite.
        core_img = Image.open(core_path)
        # Before anything else measures this image. The ground patch is part of
        # the sprite's alpha, so leaving it in would put core_box - and with it
        # the pose skeleton, and the crop applied to every frame - around the
        # character PLUS its dirt, scaling the character down to make room for
        # something that should not be there.
        if core_img.mode in ("RGBA", "LA", "P"):
            core_img = strip_ground_patch(core_img.convert("RGBA"))
        # Keep the sprite's own alpha bounds before compositing flattens them.
        # The pose skeleton is fitted to this box so it lands on the character
        # rather than on the middle of the canvas; a skeleton centred while the
        # sprite sits low and left conditions for a second, differently placed
        # figure, and the frame comes back with two of them.
        core_box = None
        if core_img.mode in ("RGBA", "LA", "P"):
            core_box = core_img.convert("RGBA").getbbox()
        if core_img.mode in ("RGBA", "LA", "P"):
            core_img = core_img.convert("RGBA")
            core_img = Image.alpha_composite(
                Image.new("RGBA", core_img.size, (255, 255, 255, 255)), core_img
            )
        core_img = core_img.convert("RGB")
    else:
        err = "Parent core image not found"
        logger.error(err)
        update_task_record(task_id, error_msg=err)
        return {"error": err}
        
    conn = get_db()
    parent_prompt = ""
    parent_seed = random.randint(0, 10**9)
    if conn:
        with conn.cursor() as cur:
            cur.execute("SELECT prompt, seed FROM sprite_images WHERE id = %s", (parent_id,))
            row = cur.fetchone()
            if row:
                parent_prompt, _seed = row
                if _seed: parent_seed = _seed
        conn.close()

    clean_prompt = parent_prompt.replace("PixelartFSS", "").strip().lstrip(",").strip()
    base_prompt = f"{clean_prompt}, flat solid white background, high quality pixel art, 16-bit, sharp focus" if "background" not in clean_prompt.lower() else f"{clean_prompt}, high quality pixel art, sharp focus"
    
    # Thread count is set once at import from the actual core count; hardcoding
    # 12 here oversubscribed this 4-thread host and is a no-op on CUDA.
    logger.info(f"Task started on {DEVICE.upper()} (torch threads: {torch.get_num_threads()}).")
    
    # Step 2 runs through get_sd_pipeline, not get_flux_pipeline.
    #
    # The FLUX path cannot load on a clean machine: it expects
    # /models/flux1-schnell.safetensors plus a hardcoded HF snapshot hash, while
    # models.txt downloads flux-2-klein-4b-Q8_0.gguf. Neither is wired to the
    # other, so this step had no working pipeline at all.
    #
    # Measured alternative: Onodofthenorth/SD_PixelArt_SpriteSheet_Generator
    # renders the SAME character consistently across a 4-frame row in ~7s, which
    # is exactly what this step needs. Frame-to-frame identity is the hard part
    # of sprite animation, and SDXL-Turbo cannot do it.
    # One pipeline for the whole sheet, ControlNet included if ANY requested
    # action has a pose cycle. Loading a posed and an unposed pipeline would
    # make them evict each other on a 12GB card - see get_sd_pipeline - and a
    # sheet of four actions would then reload the checkpoint four times.
    # Unposed actions run through the same pipeline with a blank control image
    # and conditioning scale 0, which is exactly plain img2img.
    use_pose = any(poses.cycle_for(a) for a in actions)
    if use_pose:
        logger.info("Pose cycles found; loading ControlNet for per-frame pose.")
    p = get_sd_pipeline(llm_name, pipeline_type="img2img",
                        controlnet=OPENPOSE_CONTROLNET if use_pose else None)

    # Same "load it only if some action wants it" rule as ControlNet above. The
    # adapter drags in a CLIP image encoder that stays resident in VRAM beside
    # the checkpoint, so a sheet of actions that all leave ip_adapter_scale unset
    # should never pay for it.
    # Also required by derive_side_core, which is the only thing that lets a
    # profile action actually turn - so a sheet containing one needs the adapter
    # whether or not that action sets ip_adapter_scale for its own frames.
    needs_side = any((action_entry(a).get("view") or "front") != "front"
                     for a in actions)
    ipa_active = False
    if needs_side or any((action_entry(a)["ip_adapter_scale"] or 0.0) > 0
                         for a in actions):
        ipa_active = ensure_ip_adapter(p)
    elif getattr(p, "_ipa_loaded", False):
        # A previous sheet on this cached pipeline turned it on. Unload rather
        # than just zeroing the scale: once the adapter is attached, the UNet
        # expects image embeds on every call, and this sheet will not be
        # passing any. Zero scale would leave that mismatch in place and the
        # first frame would fail inside diffusers rather than here. Unloading
        # also hands the image encoder's VRAM back.
        try:
            p.unload_ip_adapter()
            p._ipa_loaded = False
            logger.info("IP-Adapter unloaded; no action in this sheet uses it.")
        except Exception as e:
            logger.warning(f"Could not unload IP-Adapter: {e}")
    if not p:
        update_task_record(task_id, error_msg=f"Pipeline '{llm_name}' failed to load")
        return {"error": f"Pipeline '{llm_name}' failed to load"}

    check_prompt_length(p, NEGATIVE_FRAME, "NEGATIVE_FRAME")

    # img2img derives output size from the input image; StableDiffusionImg2ImgPipeline
    # takes no width/height. Resize the core to the full strip up front so the
    # model composes across the whole row rather than one frame.
    # Resize to ONE FRAME, preserving the core's aspect ratio.
    #
    # Do not resize to the full strip (frame_width * motion_steps). img2img
    # starts from this image, so a 512x512 core squashed into 512x128 hands the
    # model a 4:1 smear and it denoises from garbage — the output was
    # consistent across frames and consistently unrecognisable. Each frame is
    # generated separately from the undistorted core instead.
    #
    # Generate at the model's NATIVE resolution, then downscale to the sprite
    # frame size afterwards. Do not generate directly at frame_width/height.
    #
    # SD1.5 is trained at 512x512. At a 128x128 request the latent is 16x16,
    # far below what the UNet can work with, and the output is RGB noise — not
    # a poor sprite, actual static. SDXL has the same problem below ~512.
    # Downscaling a good 512px render to 128px is also simply better for pixel
    # art than asking the model for a tiny image.
    GEN_SIZE = 512
    core_frame = core_img.resize((GEN_SIZE, GEN_SIZE), Image.Resampling.LANCZOS)

    # The alpha box was measured on the ORIGINAL core, which is not necessarily
    # GEN_SIZE. Scale it, or the skeleton lands off the character.
    pose_box = None
    if use_pose and core_box:
        sx, sy = GEN_SIZE / core_img.width, GEN_SIZE / core_img.height
        pose_box = (core_box[0] * sx, core_box[1] * sy,
                    core_box[2] * sx, core_box[3] * sy)
    BLANK_CONTROL = Image.new("RGB", (GEN_SIZE, GEN_SIZE), (0, 0, 0))
    logger.info(f"Generating frames at {GEN_SIZE}x{GEN_SIZE} (model native), "
                f"downscaling to {frame_width}x{frame_height} for the sheet.")

    # A profile-facing init for the profile actions, derived once and shared.
    #
    # Derived lazily and cached for the whole sheet: it costs a full generation,
    # so a sheet with four side actions must not pay for it four times, and a
    # sheet with none must not pay for it at all.
    _side_cache = {}

    def view_core_for(view):
        """(init image, pose box) for a non-front view, deriving on first use."""
        # Left is the mirror of right, never its own generation: two independent
        # generations do not agree on the character, and a walk that changes
        # clothes when it turns around is worse than one that does not turn.
        source = "right" if view == "left" else view

        if source not in _side_cache:
            ctrl_action = "move up" if source == "back" else "move right"
            ctrl = poses.control_images(ctrl_action, 1, (GEN_SIZE, GEN_SIZE),
                                        pose_box)[0]
            _side_cache[source] = derive_view_core(
                p, core_frame, ctrl, base_prompt, parent_seed, source, llm_name)
        base_img = _side_cache[source]
        if base_img is None:
            return core_frame, pose_box

        if view not in _side_cache:
            _side_cache[view] = (base_img.transpose(Image.FLIP_LEFT_RIGHT)
                                 if view == "left" else base_img)
        img = _side_cache[view]

        # The turned body has its own bounds, so the skeleton must be refitted
        # to them. Reusing the front core's box would put the skeleton back
        # around a silhouette that no longer exists.
        box = img.getbbox() or (0, 0, GEN_SIZE, GEN_SIZE)
        flat = Image.alpha_composite(
            Image.new("RGBA", img.size, (255, 255, 255, 255)),
            img.convert("RGBA")).convert("RGB")
        return flat, box

    action_strips = []
    failed_actions = []

    # We loop each action sequentially. Flux Img2Img performs best this way.
    total = len(actions)
    for i, action in enumerate(actions):
        if is_cancelled(task_id):
            return {"error": "Cancelled"}
        
        logger.info(f"--- Action {i+1}/{total}: '{action}' ---")
        update_task_record(task_id, progress_pct=10 + int((i/total)*80), progress_msg=f"Generating {action}...")
        
        is_dynamic = any(kw in action.lower() for kw in ["move", "walk", "attack", "damage", "burning"])
        
        _entry = action_entry(action)
        trigger = _entry["prompt"]
        action_strength = _entry["strength"]
        ipa_scale = _entry["ip_adapter_scale"] or 0.0
        
        # Increase strength to force adaptation to new movements
        # img2img strength is how much of the core image to discard. At 0.95 the
        # core is almost entirely overwritten, which is why dynamic actions came
        # back as a completely different character each time — the reference was
        # being thrown away in the name of "more movement". Keep it low enough
        # that identity survives; the pose comes from the prompt, not from
        # destroying the input.
        # Measured on a 4-frame walk, shared seed, mean absolute
        # inter-frame difference (0-255) / max difference from frame 0:
        #   0.35 -> motion 1.7 / drift 2.1   barely moves
        #   0.45 -> motion 3.0 / drift 3.7
        #   0.55 -> motion 4.7 / drift 5.8   best motion still holding identity
        #   0.65 -> motion 8.0 / drift 10.1  design starts changing (a hat
        #                                    appeared halfway through the strip)
        strength = 0.55 if is_dynamic else 0.45

        # Facing comes from ONE of two sources, never both.
        #
        # A checkpoint with a trained view trigger knows what the character
        # looks like from that angle, so the trigger does the work and the front
        # core stays the init. That is strictly better than deriving: a derived
        # core is this pipeline guessing at an unseen angle, where the trigger is
        # the model recalling one it was trained on.
        #
        # Without such a trigger we fall back to derive_side_core, which is the
        # only option for profile actions on All-In-One.
        # Facing is decided ONCE, when the view core is derived - not per frame.
        #
        # Trigger and derivation compose rather than competing. The trigger says
        # WHAT the far side of the character looks like, which only a checkpoint
        # trained on that view knows; the derivation is the one moment with
        # enough denoising freedom to actually turn the body. Applying the
        # trigger per frame instead was measured and did nothing: every row came
        # back front-facing, because at frame settings the init and the adapter
        # outvote a prompt token.
        #
        # `view` is data from action_prompts.json, so this no longer infers
        # facing from skeleton shape - which could only ever distinguish profile
        # from not-profile, and so had no way to express "back".
        view = (_entry.get("view") or "front")
        action_init, action_box = core_frame, pose_box
        if view != "front":
            action_init, action_box = view_core_for(view)

        # Skeletons for this action, or None when it has no cycle.
        frame_controls = (poses.control_images(action, motion_steps,
                                               (GEN_SIZE, GEN_SIZE), action_box)
                          if use_pose else None)
        if frame_controls:
            # ControlNet now holds the pose, so strength no longer has to serve
            # two masters. Without it, strength had to stay low to keep the
            # character - which is exactly why the pose barely moved. Raise it
            # and let the skeleton, not the init image, decide the posture.
            strength = POSED_STRENGTH

        # A per-action strength in action_prompts.json wins over both defaults.
        #
        # One global strength cannot serve every action, because the actions do
        # not all ask for the same thing. Most only need a new POSE, and the
        # skeleton already supplies that - so 0.60 is right, and raising it just
        # trades identity away for nothing. But an action that has to ADD
        # something absent from the core image is a different problem: measured
        # on burning with identical prompt, seed and skeleton, the fraction of
        # the sprite that came back flame-coloured was 0.07% at 0.60 and 4.36%
        # at 0.75. Below ~0.70 there is simply not enough of the frame left
        # unfrozen to paint fire into.
        if action_strength is not None:
            strength = action_strength
        if frame_controls:
            logger.info(f"  > '{action}': pose-conditioned, strength {strength}"
                        + (" (from action_prompts.json)"
                           if action_strength is not None else ""))

        negative = NEGATIVE_FRAME
        
        # Was hardcoded to 4 steps / guidance 0 for FLUX-schnell. Now derived
        # from the selected checkpoint: distilled models get clamped back down
        # to 4/0, non-distilled ones get enough steps and guidance for the
        # negative prompt to actually apply.
        num_inf_steps, sheet_guidance, negative = resolve_sampling_params(
            llm_name, 30, 8.0, negative
        ) 
        
        # The old per-diffusion-step progress callback is gone: progress is now
        # reported per frame, which is both more useful and cheaper (it wrote to
        # the DB every other denoise step). It also shadowed `i` — the action
        # index — with the step index, which was a bug waiting to be triggered.
        #
        # Seeding moved into the per-frame loop below: each frame needs its own
        # generator, derived from parent_seed so the sheet stays reproducible.
        try:
            # ONE GENERATION PER FRAME, from the same undistorted core.
            #
            # Five earlier attempts all asked the model for the LAYOUT - "a row
            # of N animation frames" - and hoped it complied. It does not do so
            # reliably: the same prompt produced a row, a vertical stack and a
            # 2x2 grid on consecutive runs, and no negative prompt fixed it.
            # Layout is not something classifier-free guidance steers well.
            #
            # So stop asking. Render one 512x512 single-character image per
            # frame - which is exactly what step 1 already does reliably - and
            # compose the strip here, where layout is arithmetic instead of a
            # sample from a distribution.
            #
            # What keeps the frames the SAME character:
            #   * every frame starts from the same core image (img2img)
            #   * strength stays low, so the core is not overwritten
            #   * every frame uses the SAME seed, so the added noise and the
            #     denoising trajectory are identical; only the prompt differs
            # What creates motion: the per-frame phase hint, and nothing else.
            #
            # Attempt 4 did generate per frame - but at strength 0.95 with a
            # per-frame seed, breaking both of those at once, and returned four
            # different characters. Measured inter-frame difference on the same
            # walk: shared seed 4.7, per-frame seed 23.2. The seed is what holds
            # identity; the approach was never the problem.
            logger.info(f"  > Rendering {motion_steps} frames for '{action}'...")
            frame_imgs = []
            for f in range(motion_steps):
                if is_cancelled(task_id):
                    return {"error": "Cancelled"}
                # The phase hint describes a walk contact/passing cycle. When
                # a skeleton is driving the frame it already says all of that,
                # more precisely, and the words then fight it - "leading leg
                # extended" against a front-facing or attack skeleton is simply
                # wrong. Use the hint only for unposed actions.
                # No view trigger here on purpose. Facing is already baked into
                # the init image by view_core_for; repeating the trigger per
                # frame was measured to change nothing, and it would only spend
                # tokens from the 77 CLIP keeps.
                frame_prompt = (
                    f"{trigger}, {base_prompt}, single character, full body, centered"
                    if frame_controls else
                    f"{trigger}, {PHASE_HINTS[f % len(PHASE_HINTS)]}, {base_prompt}, "
                    "single character, full body, centered"
                )
                # Only NEGATIVE_FRAME was ever checked, so the positive prompt
                # could overrun 77 tokens in silence - and it is the side that
                # grew when the action triggers were expanded. Check the first
                # frame of each action: the rest differ only by phase hint.
                if f == 0:
                    check_prompt_length(p, frame_prompt, f"frame prompt for '{action}'")
                control_kwargs = {}
                if use_pose:
                    control_kwargs = {
                        "control_image": frame_controls[f] if frame_controls else BLANK_CONTROL,
                        "controlnet_conditioning_scale":
                            CONTROLNET_SCALE if frame_controls else 0.0,
                    }
                # Re-inject the core through cross-attention at every denoising
                # step, instead of only as the starting latent. Off unless this
                # action asked for it - see the IP-Adapter block near
                # POSED_STRENGTH for what it costs an action that adds an effect.
                #
                # The scale is set per frame rather than once per action: the
                # pipeline object is shared and cached, so a scale left behind
                # by a previous action would silently apply to this one.
                if ipa_active:
                    p.set_ip_adapter_scale(ipa_scale)
                    control_kwargs["ip_adapter_image"] = core_frame
                # image=core_frame, and NOT the previous frame. Chaining frame
                # N-1 forward is the obvious fix for frames that do not look
                # like each other, and it was measured and rejected.
                #
                # Swept on the bow row (the worst offender), core-ANCHORED
                # blends so drift had a floor - init = blend(core, prev):
                #   blend 0.00  frame-to-frame 10.65  loop-gap  8.36  <- kept
                #   blend 0.35  frame-to-frame  8.01  loop-gap 12.06
                #   blend 0.50  frame-to-frame  7.81  loop-gap 13.17
                #   blend 0.70  frame-to-frame  9.16  loop-gap 19.51
                # Chaining does make ADJACENT frames more alike, and that is
                # exactly the trap: it buys it by letting the strip walk away
                # from where it started, so frame 4 no longer joins back onto
                # frame 1. loop-gap is the number that matters for a cycle and
                # it gets monotonically worse. Every pass also re-encodes
                # through the VAE, so compression artifacts compound - at
                # blend 0.50 with strength 0.66 the frames came back visibly
                # speckled with yellow blocks.
                #
                # Lowering strength beat it outright on both metrics at once
                # (0.66 unchained: frame-to-frame 6.37, loop-gap 5.22) AND was
                # the only variant that kept the character a zombie rather than
                # a pale bald man. See the per-action strengths in
                # action_prompts.json.
                img = p(
                    prompt=frame_prompt,
                    negative_prompt=negative or None,
                    image=action_init,
                    strength=strength,
                    num_inference_steps=num_inf_steps,
                    guidance_scale=sheet_guidance,
                    # parent_seed, NOT parent_seed + i. The seed is what
                    # holds the character together; varying it per action made
                    # the 'idle' row a visibly different shade of the same
                    # zombie than the 'move right' row. Pose differences come
                    # from the prompt, which is what it is good at.
                    generator=torch.Generator("cpu").manual_seed(parent_seed),
                    **control_kwargs,
                ).images[0]
                # Key at full render resolution - the corner-colour match is far
                # more reliable on a clean 512px image than a resampled one.
                # keep_largest for the same reason as the core: one sprite.
                frame_imgs.append(remove_background(img, keep_largest=True))

            # ONE shared crop box for the whole strip, not one per frame.
            # Fitting each frame to its own alpha bounds rescales every frame
            # independently, so a walk cycle - where the silhouette genuinely
            # widens and narrows - comes out pulsing in size. Union the bounds
            # and apply the same crop and scale to every frame: relative motion
            # survives, apparent size does not.
            boxes = [b for b in (im.getbbox() for im in frame_imgs) if b]
            union = ((min(b[0] for b in boxes), min(b[1] for b in boxes),
                      max(b[2] for b in boxes), max(b[3] for b in boxes))
                     if boxes else (0, 0, GEN_SIZE, GEN_SIZE))

            action_strip = Image.new("RGBA", (frame_width * motion_steps, frame_height), (0, 0, 0, 0))
            for f, img in enumerate(frame_imgs):
                frame_img = fit_into_frame(img, union, frame_width, frame_height)
                action_strip.paste(frame_img, (f * frame_width, 0), frame_img)

            logger.info(f"  > Action '{action}': {motion_steps} frames composed "
                        f"at {frame_width}x{frame_height}.")
            update_task_record(
                task_id,
                progress_pct=10 + int(((i + 1) / total) * 80),
                progress_msg=f"{action}: done",
            )
            action_strips.append(action_strip)
            logger.info(f"  > Action '{action}' processed successfully.")
        except Exception as e:
            # Record and keep going. Aborting the whole sheet on one bad action
            # discards every strip already generated, which on a multi-minute GPU
            # job is the expensive thing to lose. A failed action is simply absent
            # from the sheet, and the error stays visible in the UI.
            logger.error(f"Action '{action}' generation failed: {e}", exc_info=True)
            failed_actions.append(action)
            update_task_record(
                task_id,
                error_msg=f"Failed on: {', '.join(failed_actions)}",
                progress_msg=f"Skipped '{action}' after error",
            )

    # Every action failed: there is nothing to stitch. Image.new with height 0
    # would raise deep in PIL and mask the real cause, so fail explicitly.
    if not action_strips:
        err = f"All {len(actions)} actions failed: {', '.join(failed_actions)}"
        logger.error(err)
        update_task_record(task_id, error_msg=err, progress_msg="Failed")
        return {"error": err}

    # Stitch vertically
    logger.info(f"Stitching {len(action_strips)} action strips into master sheet...")
    sheet_w = frame_width * motion_steps
    sheet_h = frame_height * len(action_strips)
    master = Image.new("RGBA", (sheet_w, sheet_h), (0,0,0,0))
    
    y = 0
    for s in action_strips:
        master.paste(s, (0, y), s)
        y += frame_height
        
    filename = f"sheet_{uuid.uuid4().hex[:12]}.png"
    filepath = os.path.join(IMAGES_DIR, filename)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    master.save(filepath, format="PNG")
    
    # duration_ms was never passed here, so the column kept its default and the
    # UI rendered "Completed in 0s" for a job that took half a minute - which
    # reads as a cache hit and hid what a sheet actually costs.
    duration_ms = (time.time() - sheet_start) * 1000
    update_task_record(task_id, file_path=filepath, duration_ms=duration_ms, progress_pct=100, progress_msg="Complete", image_type="spritesheet", requested_actions=actions, parent_id=parent_id)
    logger.info(f"Sheet generated {filepath} in {duration_ms/1000:.1f}s")
    return {"status": "success", "url": f"/images/{filename}", "duration_ms": duration_ms}


@celery_app.task(name="tasks.generate_raw_task", bind=True)
def generate_raw_task(self, prompt: str, negative_prompt: str, llm_name: str,
                      width: int, height: int, steps: int, cfg_scale: float,
                      seed: int, strip_background: bool = False):
    """Plain text2img with no prompt rewriting.

    Deliberately separate from generate_core_task, which prepends sprite-specific
    styling ("solo individual", "lone character", "centered"). That styling is
    actively wrong for callers asking for a tileable ground texture, which is
    most of what something2 requests. Callers that want the sprite treatment go
    through generate_core_task; callers that want exactly what they asked for
    come here.

    Returns the saved file path rather than image bytes: the result travels
    through the Redis result backend, and sheets can approach the 32MB cap.
    """
    task_id = self.request.id
    p = get_sd_pipeline(llm_name)
    if not p:
        return {"error": f"Model '{llm_name}' failed to load"}

    if seed is None or seed < 0:
        seed = random.randint(0, 10**9)
    generator = torch.Generator("cpu").manual_seed(seed)

    # Callers reach this task with A1111 conventions (20 steps / cfg 7), which
    # are wrong for distilled checkpoints. Reconcile here so every entry point
    # gets it, not just the ones that remembered to.
    steps, cfg_scale, negative_prompt = resolve_sampling_params(
        llm_name, steps, cfg_scale, negative_prompt
    )

    start_time = time.time()
    try:
        def progress_callback(pipe, i, t, callback_kwargs):
            if is_cancelled(task_id):
                raise RuntimeError("Cancelled")
            return callback_kwargs

        img = p(
            prompt,
            negative_prompt=negative_prompt or None,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            generator=generator,
            callback_on_step_end=progress_callback,
        ).images[0]
    except Exception as e:
        logger.error(f"Raw generation {task_id} failed: {e}", exc_info=True)
        return {"error": str(e)}

    duration_ms = (time.time() - start_time) * 1000

    if strip_background:
        img = remove_background(img)

    filename = f"raw_{uuid.uuid4().hex[:12]}.png"
    filepath = os.path.join(IMAGES_DIR, filename)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    img.save(filepath, format="PNG")

    return {
        "status": "success",
        "file_path": filepath,
        "url": f"/images/{filename}",
        "seed": seed,
        "duration_ms": duration_ms,
    }


# Where the worker leaves its last device readout for the API to find.
#
# The round-trip alone is not enough. The worker runs --pool=solo at
# concurrency=1, so while a task is executing it consumes NOTHING from the
# broker - not other tasks, not control messages. Every describe_device sent
# during a generation therefore times out, and the diagnostics panel spent the
# whole of each generation claiming "Worker unreachable" about a worker that was
# busy working. Leaving a snapshot behind gives the API something truthful to
# show for exactly the window where it cannot ask.
DEVICE_SNAPSHOT_KEY = "device:snapshot"


def _cache_device_snapshot(info: dict):
    """Publish a device readout for the API to read while the worker is busy."""
    try:
        # Stamped, because a busy worker and a dead one both fail the round-trip
        # and would otherwise be reported identically. The age does not prove
        # liveness — a long generation and a long outage look alike — but it is
        # the difference between "busy, 12s ago" and "busy, 40 minutes ago",
        # and the second one tells you to go look at the worker.
        _redis_client.set(DEVICE_SNAPSHOT_KEY,
                          json.dumps({**info, "snapshot_at": time.time()}))
    except Exception as e:
        logger.warning(f"Could not cache device snapshot: {e}")


def read_device_snapshot():
    """The worker's last published readout, or None. Safe to call from the API."""
    try:
        raw = _redis_client.get(DEVICE_SNAPSHOT_KEY)
        return json.loads(raw) if raw else None
    except Exception as e:
        logger.warning(f"Could not read the device snapshot: {e}")
        return None


def _read_device_info() -> dict:
    """Device readout, taken in whatever process calls this."""
    info = {
        "device": DEVICE,
        "dtype": str(DTYPE),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "loaded_pipelines": sorted(pipes.keys()),
    }
    if DEVICE == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update({
            "gpu_name": props.name,
            "vram_total_gb": round(props.total_memory / 1024**3, 1),
            "vram_reserved_gb": round(torch.cuda.memory_reserved(0) / 1024**3, 2),
            "vram_allocated_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 2),
        })
    return info


@celery_app.task(name="tasks.describe_device")
def describe_device():
    """Report the worker's actual compute device.

    The API process cannot answer this about itself: compose pins it to
    COMPUTE_DEVICE=cpu so it never holds VRAM, while the worker runs on cuda.
    Any device readout taken in the API would therefore always say "cpu" and be
    actively misleading. Ask the process that actually runs inference.
    """
    info = _read_device_info()
    _cache_device_snapshot(info)
    return info


@celery_app.task(name="tasks.warm_model_task", bind=True)
def warm_model_task(self, llm_name: str, pipeline_type: str = "text2img"):
    """Download and load a model into the worker without generating anything.

    Exists because model download time counts against generation timeouts.
    something2 caps a provider call at 5 minutes by default and does not support
    submit/poll, so the first request for an uncached 7GB checkpoint fails there
    no matter how the request is written. Warm the cache out-of-band instead,
    then every real request hits an already-resident pipeline.
    """
    started = time.time()
    logger.info(f"Warming '{llm_name}' ({pipeline_type})...")
    p = get_sd_pipeline(llm_name, pipeline_type)
    elapsed = time.time() - started
    if not p:
        logger.error(f"Warm failed for '{llm_name}' after {elapsed:.0f}s")
        return {"status": "error", "model": llm_name, "elapsed_s": round(elapsed, 1),
                "error": "pipeline failed to load"}
    logger.info(f"Warmed '{llm_name}' in {elapsed:.0f}s")
    return {"status": "ok", "model": llm_name, "elapsed_s": round(elapsed, 1),
            "device": DEVICE, "loaded": sorted(pipes.keys())}


# ---------------------------------------------------------------------------
# Image EDITING (instruction-driven), as opposed to generation.
# ---------------------------------------------------------------------------
#
# img2img changes an image by re-denoising it, so it cannot follow an
# instruction - it has no notion of "remove that" or "turn him to face left",
# only "stay this close to the original". That is the limitation the README's
# step 2 section runs into: pose IS composition, and img2img preserves
# composition.
#
# Qwen-Image-Edit takes the source image AND a natural-language instruction.
#
# The checkpoint is pre-quantised NF4. The unquantised pipeline is ~55GB with a
# 15.45GB fp16 text encoder, which does not fit a 12GB card; NF4 puts the
# transformer at 10.71GB and the encoder at 5.80GB, and ships it that way so
# there is nothing to quantise at load. Requires `bitsandbytes` installed -
# BitsAndBytesConfig imports fine without it and then from_pretrained raises.
#
# Capabilities are LoRAs on ONE base, the same pattern as the pixel-art LoRAs:
# ~0.5GB per capability instead of a separate multi-GB model per task.
# Chosen by TRANSFORMER size, not total size. See .ai/decisions/0002.
# Qwen-Image-Edit-2511 NF4 was tried first and does not fit: its transformer is
# 10.71GB and bitsandbytes 4-bit layers cannot be split or CPU-offloaded, so it
# must be wholly resident, leaving ~0.75GB of an 11.7GB card for the activations
# of a 20B model. FLUX Kontext's NF4 transformer is 6.24GB, which leaves ~5GB.
# Its T5 encoder is a comparable 5.89GB but DOES offload to CPU.
EDIT_BASE = "Meatfucker/Flux.1-Kontext-dev-bnb-nf4"

# Editing is OFF by default on this machine, and the endpoint refuses rather
# than trying.
#
# This is not caution: calling it actually takes the service down. Loading the
# pipeline needs all 12.52GB of weights in system RAM (see enable_model_cpu_
# offload below), the WSL VM is capped at 11GB, and the kernel OOM-kills the
# Celery worker mid-load - silently, at component 4 of 7, with no traceback.
# Generation is unavailable until the worker respawns. An endpoint that kills
# its own service on every call must not be reachable by default.
#
# Set EDIT_ENABLED=1 once the host has materially more RAM (24GB+ settles it),
# or if a smaller editing pipeline appears. Full measurements in
# .ai/decisions/0002.
EDIT_ENABLED = os.environ.get("EDIT_ENABLED", "").strip().lower() in ("1", "true", "yes")
EDIT_UNAVAILABLE_REASON = (
    "Image editing is disabled on this host. FLUX Kontext NF4 needs 12.52GB of "
    "weights resident in system RAM and the WSL VM is capped at 11GB, so loading "
    "it OOM-kills the worker. Needs more host RAM, not a different model. "
    "SDXL img2img and ControlNet cover image-to-image today."
)

# How much VRAM the editing pipeline needs free before it may start loading.
#
# This is a PREFLIGHT GATE, not a placement budget. It used to be documented as
# the latter - "max_memory keeps a margin" - but nothing ever passed max_memory
# or device_map to from_pretrained, so the margin it described did not exist.
# bitsandbytes places 4-bit weights on the card as from_pretrained walks the
# checkpoint, with nothing to stop it, and the card fills partway through.
#
# Measured on the 12GB card: at "8GiB" the pipeline loaded (11752MiB resident)
# and then died mid-inference with "CUDA driver error: device not ready" - an
# out-of-memory wearing a driver fault's clothing. Worse, that failure leaves
# the CUDA context unusable, so the NEXT request fails differently, with
# "!handles_.at(i) INTERNAL ASSERT FAILED ... CUDACachingAllocator", and the one
# after that with "Cannot copy out of meta tensor". Only a worker restart clears
# it, and in the meantime plain sprite generation is broken too.
#
# So refusing up front is worth more than any placement tuning: one declined
# edit costs nothing, one attempted edit costs every task behind it.
#
# The gate is not academic on this host. llm-server holds 8-9GB of the card
# while a chat model is awake and only releases it after --sleep-idle-seconds,
# so "free VRAM" here routinely means 3GB, not 11.7GB.
EDIT_GPU_BUDGET = os.environ.get("EDIT_GPU_BUDGET", "6GiB")

# Weights that must be resident in system RAM to load the NF4 pipeline. Checked
# against MemAvailable for the same reason as the VRAM gate: the documented
# failure mode here is the kernel OOM-killing the worker mid-load, which takes
# generation down with it and leaves no traceback to read afterwards.
EDIT_HOST_RAM_NEEDED = os.environ.get("EDIT_HOST_RAM_NEEDED", "12.5GiB")


def _parse_bytes(spec: str) -> int:
    """'6GiB' / '12.5GiB' / '6GB' -> bytes. Accepts a bare number as GiB."""
    s = str(spec).strip().lower().replace("b", "")
    mult = 1024 ** 3
    if s.endswith("gi"):
        s = s[:-2]
    elif s.endswith("g"):
        s, mult = s[:-1], 1000 ** 3
    elif s.endswith("mi"):
        s, mult = s[:-2], 1024 ** 2
    elif s.endswith("m"):
        s, mult = s[:-1], 1000 ** 2
    return int(float(s) * mult)


def _host_ram_available() -> int:
    """MemAvailable in bytes, or -1 if it cannot be read."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return -1

# Source images are downscaled to this before editing. 1024px input pushed the
# activations past the card even with weights capped; 512 is also what the
# sprite pipeline works at natively.
EDIT_MAX_SIDE = int(os.environ.get("EDIT_MAX_SIDE", "512"))
# No LoRAs. The two that were wanted -
# fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA (turn the subject) and
# prithivMLmods/Qwen-Image-Edit-2511-Object-Remover (remove an object) - are
# trained against Qwen-Image-Edit, whose transformer does not fit this card.
# A LoRA's ranks are tied to the UNet it was trained on, so they cannot move to
# FLUX Kontext.
#
# The capability is not lost: FLUX Kontext is an instruction-editing model and
# does both of those from the prompt alone ("remove the shield", "show this
# character from the side"). That is what it was built for; the Qwen LoRAs exist
# to teach a general editor one trick each.
EDIT_LORAS = {}


# Why the last load attempt returned None. The task reports this instead of
# "Editing pipeline failed to load", which named no cause and sent anyone
# debugging it to the worker log to find out what the API already knew.
_edit_load_error = None


def _refuse_edit(reason: str):
    """Record and log why the editing pipeline will not load, then return None."""
    global _edit_load_error
    _edit_load_error = reason
    logger.error(reason)
    return None


def get_edit_pipeline(lora: str = None):
    """Load the NF4 editing pipeline, optionally with a capability LoRA.

    Always uses enable_model_cpu_offload rather than .to("cuda"): the NF4
    transformer alone is ~10.7GB against ~11.7GB free, so the text encoder has
    to be evicted after it has encoded the prompt. Offload does exactly that,
    and it is why this is slow but possible at all on this card.

    Returns None on refusal or failure; the reason is in `_edit_load_error`.
    """
    global _edit_load_error
    _edit_load_error = None

    if not EDIT_ENABLED:
        return _refuse_edit(EDIT_UNAVAILABLE_REASON)

    global pipes
    cache_key = f"__edit__{EDIT_BASE}_{lora or 'nolora'}"
    if cache_key in pipes:
        return pipes[cache_key]

    log_cuda_details_once()

    # Same eviction rule as get_sd_pipeline, and it matters more here: this
    # pipeline wants nearly the whole card.
    if DEVICE == "cuda" and pipes:
        logger.info(f"Evicting {len(pipes)} pipeline(s) before loading the editor: "
                    f"{sorted(pipes.keys())}")
        pipes.clear()
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # Preflight. Refuse BEFORE touching the checkpoint.
    #
    # Every failure mode below is one that, once entered, cannot be recovered
    # inside this process: an OOM part-way through bitsandbytes placement leaves
    # the CUDA context unusable for every later task, and a host-RAM OOM has the
    # kernel kill the worker outright. Both are cheap to predict and impossible
    # to undo, which is the whole argument for checking first.
    if DEVICE == "cuda":
        need = _parse_bytes(EDIT_GPU_BUDGET)
        free, total = torch.cuda.mem_get_info()
        if free < need:
            return _refuse_edit(
                f"Not enough free VRAM to load the editing pipeline: "
                f"{free / 1024**3:.1f}GiB free of {total / 1024**3:.1f}GiB, need "
                f"{need / 1024**3:.1f}GiB (EDIT_GPU_BUDGET={EDIT_GPU_BUDGET}). "
                f"Something else holds the card - llm-server keeps a chat model "
                f"resident until it has been idle for --sleep-idle-seconds. "
                f"Loading anyway fills the card mid-placement and poisons the "
                f"CUDA context for every later task."
            )

    ram_need = _parse_bytes(EDIT_HOST_RAM_NEEDED)
    ram_free = _host_ram_available()
    if 0 <= ram_free < ram_need:
        return _refuse_edit(
            f"Not enough system RAM to load the editing pipeline: "
            f"{ram_free / 1024**3:.1f}GiB available, need "
            f"{ram_need / 1024**3:.1f}GiB. Loading anyway gets the worker "
            f"OOM-killed mid-load, silently and with no traceback. "
            f"See .ai/decisions/0002."
        )

    try:
        from diffusers import FluxKontextPipeline
        logger.info(f"Loading editing pipeline '{EDIT_BASE}' (NF4)...")

        common = dict(
            torch_dtype=DTYPE, cache_dir="/models",
            token=os.environ.get("HF_TOKEN") or None,
        )

        # There is no placement budget on the load itself, and there cannot be
        # a useful one.
        #
        # bitsandbytes places 4-bit weights on the GPU as from_pretrained walks
        # the checkpoint, before any offload hook can run. The transformer is
        # 10.71GB and the text encoder 5.80GB - 16.5GB - so if the card is
        # short, from_pretrained fills it partway through and dies with "CUDA
        # driver error: device not ready", which reads like a driver fault
        # rather than the out-of-memory it actually is.
        #
        # device_map="balanced" + max_memory does bound placement, and this
        # comment used to claim that is what happens here. It is not, and it
        # must not be: a static device map has no runtime hooks, so calling the
        # offloaded T5 afterwards dies with "Cannot copy out of meta tensor".
        # See the note on enable_model_cpu_offload below. The load stays
        # unbounded; the EDIT_GPU_BUDGET preflight above is what keeps it from
        # being attempted on a card that cannot take it.
        if DEVICE == "cuda":
            try:
                # enable_model_cpu_offload FIRST, not device_map.
                #
                # device_map="balanced" places correctly - measured: transformer
                # on cuda:0, T5 on cpu, 6.62GB allocated - but it is a STATIC
                # map with no runtime hooks, so calling the offloaded T5 dies
                # with "Cannot copy out of meta tensor; no data!".
                #
                # enable_model_cpu_offload installs hooks that move each module
                # to the GPU as it is called and evict it afterwards. The
                # transformer (6.24GB) and T5 (5.89GB) are never needed at the
                # same instant, so the peak is one of them plus VAE and CLIP,
                # rather than the 12.5GB sum that does not fit.
                pipe = FluxKontextPipeline.from_pretrained(EDIT_BASE, **common)
                pipe.enable_model_cpu_offload()
                logger.info("Editing pipeline loaded with model CPU offload.")
                pipes[cache_key] = pipe
                return pipe
            except Exception as dm_err:
                # Model-level offload could not even load. Fall back to
                # SEQUENTIAL offload, which moves weights a submodule at a time
                # instead of a whole component: far slower, but its peak is a
                # fraction of one component rather than all of one.
                logger.warning(
                    f"Model CPU offload failed ({dm_err.__class__.__name__}: "
                    f"{dm_err}); retrying with sequential CPU offload."
                )
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                pipe = FluxKontextPipeline.from_pretrained(EDIT_BASE, **common)
                pipe.enable_sequential_cpu_offload()
                pipes[cache_key] = pipe
                return pipe
        else:
            pipe = FluxKontextPipeline.from_pretrained(EDIT_BASE, **common)

        if lora:
            repo = EDIT_LORAS.get(lora, lora)
            try:
                logger.info(f"Loading edit LoRA '{repo}'...")
                pipe.load_lora_weights(repo, cache_dir="/models",
                                       token=os.environ.get("HF_TOKEN") or None)
                # NOT fused. fuse_lora writes into the base weights, and these
                # are NF4-quantised - fusing into 4-bit tensors is not supported
                # and silently degrades or raises. Keep the adapter separate.
                logger.info(f"Edit LoRA '{repo}' loaded (not fused: base is NF4).")
            except Exception as e:
                logger.warning(f"Could not apply edit LoRA '{repo}': "
                               f"{e.__class__.__name__}: {e}. Continuing without it.")

        # Only the CPU branch reaches here — the CUDA branch above returns from
        # inside its try/except, having already installed offload hooks. Nothing
        # to place on the GPU at this point.
        if DEVICE != "cuda":
            pipe.to(DEVICE)
        pipes[cache_key] = pipe
    except Exception as e:
        _edit_load_error = f"Editing pipeline failed to load: {e.__class__.__name__}: {e}"
        logger.error(f"Error loading editing pipeline: {e}", exc_info=True)
        return None
    return pipes[cache_key]


@celery_app.task(name="tasks.edit_image_task", bind=True)
def edit_image_task(self, source_path: str, instruction: str, lora: str = None,
                    steps: int = 20, cfg_scale: float = 4.0, seed: int = None):
    """Apply a natural-language edit to an existing image."""
    task_id = self.request.id
    logger.info(f"Task {task_id} editing '{source_path}' with '{instruction}' (lora={lora})")

    if not os.path.exists(source_path):
        update_task_record(task_id, error_msg=f"Source image not found: {source_path}")
        return {"error": f"Source image not found: {source_path}"}

    p = get_edit_pipeline(lora)
    if not p:
        reason = _edit_load_error or "Editing pipeline failed to load"
        update_task_record(task_id, error_msg=reason, progress_msg="Failed")
        return {"error": reason}

    if seed is None or seed < 0:
        seed = random.randint(0, 10**9)

    src = Image.open(source_path).convert("RGB")
    if max(src.size) > EDIT_MAX_SIDE:
        # Downscale before editing, not after. The activation cost scales with
        # input area, and a 1024px source is what tipped this over the card.
        # LANCZOS rather than NEAREST: the editor is not a pixel-art model and
        # reasons better about a clean downscale than an aliased one.
        before = src.size
        src.thumbnail((EDIT_MAX_SIDE, EDIT_MAX_SIDE), Image.Resampling.LANCZOS)
        logger.info(f"Downscaled source {before} -> {src.size} for editing.")
    started = time.time()
    try:
        update_task_record(task_id, progress_pct=0, progress_msg="Editing...", seed=seed)
        # FLUX Kontext takes guidance_scale, not true_cfg_scale (that is
        # Qwen-Image-Edit's parameter). It is a guidance-distilled model, so the
        # useful range is roughly 2.5-4.0 rather than SD's 7+; higher values
        # over-bake the edit and drift from the source.
        out = p(
            image=src,
            prompt=instruction,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).images[0]

        name = f"edit_{uuid.uuid4().hex[:12]}.png"
        path = os.path.join(IMAGES_DIR, name)
        out.save(path)
        ms = (time.time() - started) * 1000
        update_task_record(task_id, file_path=path, duration_ms=ms,
                           progress_pct=100, progress_msg="Complete")
        logger.info(f"Edit complete in {ms/1000:.1f}s -> {name}")
        return {"status": "success", "url": f"/images/{name}", "file_path": path,
                "seed": seed, "duration_ms": ms}
    except Exception as e:
        logger.error(f"Edit failed: {e}", exc_info=True)
        update_task_record(task_id, error_msg=str(e), progress_msg="Failed")
        return {"error": str(e)}


# --- Stranded task reaper -------------------------------------------------

@worker_ready.connect
def _fail_stranded_tasks(**_):
    """Mark rows that claim to be in progress as failed, once, at worker boot.

    This deployment runs exactly one worker at concurrency=1 on the solo pool,
    so when this signal fires nothing can be running. Any row still showing
    "Waiting in queue..." or a partial progress bar is a task whose worker died
    holding it — an OOM kill, a container restart, the file-descriptor collapse
    that motivated the nofile ulimit in docker-compose.yml.

    Those rows never resolved themselves. acks_late is off (deliberately: a task
    that OOM-kills its worker would otherwise be redelivered forever and take
    the worker down on every attempt), so the message is gone with the process
    and nothing was ever going to update the row. One sat at "Waiting in
    queue... 0%" for 45 minutes with a Delete button as its only exit, which is
    what sent someone looking for a bug in Delete.

    Failing them here is what makes them honest — and, because the queue offers
    Retry on failed cards, re-runnable.

    The 30-second floor covers the one race: a task submitted just before this
    signal fires may already be in the prefetch buffer and about to run. Rows
    that young are left alone; if one is genuinely stranded, the next restart
    catches it.
    """
    conn = get_db()
    if not conn:
        logger.warning("Stranded-task reaper skipped: no database connection.")
        return
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE sprite_images
                       SET error = %s, progress_msg = 'Failed'
                     WHERE file_path IS NULL
                       AND (error IS NULL OR error = '')
                       AND deleted = false
                       AND timestamp < NOW() - INTERVAL '30 seconds'
                    """,
                    ("Worker restarted while this task was queued or running, "
                     "so it was lost. Retry to run it again.",),
                )
                if cur.rowcount:
                    logger.warning(
                        f"Marked {cur.rowcount} stranded task(s) as failed: their "
                        f"worker died before they finished."
                    )
    except Exception as e:
        logger.error(f"Stranded-task reaper failed: {e}")
    finally:
        conn.close()


# --- Device snapshot upkeep -----------------------------------------------
#
# Refreshed around every task, not only when describe_device runs, because
# describe_device is precisely what cannot run while the worker is busy. These
# three moments bracket the blind window: what the card looked like when the
# task started, and what it looks like the instant it finishes.

@worker_ready.connect
def _snapshot_on_boot(**_):
    _cache_device_snapshot(_read_device_info())


@task_prerun.connect
def _snapshot_before_task(**_):
    _cache_device_snapshot(_read_device_info())


@task_postrun.connect
def _snapshot_after_task(**_):
    _cache_device_snapshot(_read_device_info())
