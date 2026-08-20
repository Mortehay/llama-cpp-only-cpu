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
    # Silently running on CPU here would look like a 100x perf regression with
    # no error, so say it loudly and name the usual cause.
    logger.warning("COMPUTE_DEVICE=cuda but torch.cuda.is_available() is False — "
                   "falling back to CPU. Check nvidia-container-toolkit on the host "
                   "(`make gpu-check`).")
    DEVICE = "cpu"
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
DISTILLED_MARKERS = ("turbo", "schnell", "lightning", "lcm")

# Shared quality exclusions. Applies to both generation steps.
_NEGATIVE_QUALITY = (
    "blurry, deformed, extra limbs, cropped, low quality, watermark, text, "
    "noise, messy pixels, artifacting, gradient, shadows on background"
)

# Step 1 (core): exactly ONE character. Duplicate-suppression belongs here only.
NEGATIVE_SINGLE = (
    "multiple characters, two characters, group, horde, crowd, twins, clones, "
    "split screen, collage, grid, set, " + _NEGATIVE_QUALITY
)

# Step 2 (spritesheet): each FRAME is generated on its own and the strip is
# composed in PIL, so a frame wants exactly one character - same rules as step
# 1 - plus terms against the design drifting between frames. The old
# NEGATIVE_SHEET deliberately allowed grids, because step 2 used to ask the
# model for a whole row; it no longer does.
NEGATIVE_FRAME = (
    "multiple characters, two characters, group, crowd, twins, clones, "
    "sprite sheet, multiple poses, grid, set, split screen, collage, "
    "different character, changing outfit, changing colors, "
    # The model draws a ground shadow under the character. It is connected to
    # the feet, so it is part of the sprite's own blob and survives both the
    # background key and _isolate_largest_sprite - a green ellipse rides along
    # under every frame. Guidance is active on this checkpoint (cfg 8), so
    # unlike step 1 these terms actually do something.
    "ground shadow, drop shadow, cast shadow, floor, ground, grass, platform, "
    + _NEGATIVE_QUALITY
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


def fit_into_frame(img, box, frame_w, frame_h):
    """Crop `box` out of `img` and centre it in a frame, preserving aspect."""
    crop = img.crop(box)
    if crop.width == 0 or crop.height == 0:
        return Image.new("RGBA", (frame_w, frame_h), (0, 0, 0, 0))

    scale = min(frame_w / crop.width, frame_h / crop.height)
    new_size = (max(1, int(crop.width * scale)), max(1, int(crop.height * scale)))
    # NEAREST keeps pixel-art edges hard; see the note at the call site.
    crop = crop.resize(new_size, Image.Resampling.NEAREST)

    frame = Image.new("RGBA", (frame_w, frame_h), (0, 0, 0, 0))
    # Centre horizontally, sit on the bottom edge: sprites share a ground line,
    # and centring vertically makes a walk cycle bob against its own baseline.
    frame.paste(crop, ((frame_w - crop.width) // 2, frame_h - crop.height), crop)
    return frame


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

def get_sd_pipeline(llm_name: str = "stabilityai/sdxl-turbo", pipeline_type: str = "text2img"):
    if llm_name == "models--stabilityai--sdxl-turbo":
        llm_name = "stabilityai/sdxl-turbo"
    global pipes
    
    cache_key = f"{llm_name}_{pipeline_type}"
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
                               StableDiffusionPipeline, StableDiffusionImg2ImgPipeline)

        is_sdxl = "sdxl" in llm_name.lower() or "turbo" in llm_name.lower()
        want_img2img = pipeline_type == "img2img"

        if is_sdxl:
            pipeline_class = StableDiffusionXLImg2ImgPipeline if want_img2img else StableDiffusionXLPipeline
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

        pipe.to(DEVICE)
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
    full_prompt = f"PixelartFSS, idle front, solo individual {clean_prompt}, centered, lone character, no duplicates, one standalone character, flat solid transparent background, high quality pixel art, 16-bit, sharp focus" if "background" not in clean_prompt.lower() else f"PixelartFSS, idle front, solo individual {clean_prompt}, centered, lone character, no duplicates, one standalone character, high quality pixel art, sharp focus"

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
    p = get_sd_pipeline(llm_name, pipeline_type="img2img")
    if not p:
        update_task_record(task_id, error_msg=f"Pipeline '{llm_name}' failed to load")
        return {"error": f"Pipeline '{llm_name}' failed to load"}

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
    logger.info(f"Generating frames at {GEN_SIZE}x{GEN_SIZE} (model native), "
                f"downscaling to {frame_width}x{frame_height} for the sheet.")

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
        
        action_lower = action.lower()
        trigger = action
        if "move right" in action_lower: trigger = "side view profile, walking right, character facing right, dynamic legs moving"
        elif "move left" in action_lower: trigger = "side view profile, walking left, character facing left, dynamic legs moving"
        elif "move down" in action_lower: trigger = "walking front, character facing forward, legs moving"
        elif "move up" in action_lower: trigger = "walking back, character facing away, legs moving"
        elif "idle" in action_lower: trigger = "idle standing"
        elif "attack" in action_lower: trigger = "dramatic action pose, fast strike attack, swinging arms"
        elif "got damage" in action_lower: trigger = "taking damage, hurt posture, recoiling"
        elif "burning" in action_lower: trigger = "in flames burning, expressive movement"
        
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
                frame_prompt = (
                    f"{trigger}, {PHASE_HINTS[f % len(PHASE_HINTS)]}, {base_prompt}, "
                    "single character, full body, centered"
                )
                img = p(
                    prompt=frame_prompt,
                    negative_prompt=negative or None,
                    image=core_frame,
                    strength=strength,
                    num_inference_steps=num_inf_steps,
                    guidance_scale=sheet_guidance,
                    # parent_seed, NOT parent_seed + i. The seed is what
                    # holds the character together; varying it per action made
                    # the 'idle' row a visibly different shade of the same
                    # zombie than the 'move right' row. Pose differences come
                    # from the prompt, which is what it is good at.
                    generator=torch.Generator("cpu").manual_seed(parent_seed),
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
    
    update_task_record(task_id, file_path=filepath, progress_pct=100, progress_msg="Complete", image_type="spritesheet", requested_actions=actions, parent_id=parent_id)
    logger.info(f"Sheet generated {filepath}")
    return {"status": "success", "url": f"/images/{filename}"}


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


@celery_app.task(name="tasks.describe_device")
def describe_device():
    """Report the worker's actual compute device.

    The API process cannot answer this about itself: compose pins it to
    COMPUTE_DEVICE=cpu so it never holds VRAM, while the worker runs on cuda.
    Any device readout taken in the API would therefore always say "cpu" and be
    actively misleading. Ask the process that actually runs inference.
    """
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
