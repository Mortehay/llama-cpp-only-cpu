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

# Maximize CPU utilization based on user request (set to 70% of available logical cores)
cpu_limit = max(1, int(multiprocessing.cpu_count() * 0.70))
os.environ["OMP_NUM_THREADS"] = str(cpu_limit)
os.environ["MKL_NUM_THREADS"] = str(cpu_limit)
torch.set_num_threads(cpu_limit)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"PyTorch CPU inference threads set to {cpu_limit} (Targeting 70% of host capacity).")

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

def get_sd_pipeline(llm_name: str = "stabilityai/sdxl-turbo", pipeline_type: str = "text2img"):
    if llm_name == "models--stabilityai--sdxl-turbo":
        llm_name = "stabilityai/sdxl-turbo"
    global pipes
    
    cache_key = f"{llm_name}_{pipeline_type}"
    if cache_key in pipes:
        return pipes[cache_key]
        
    device = "cpu"
    dtype = torch.float32

    logger.info(f"Loading '{llm_name}' ({pipeline_type}) on CPU (FLOAT32)...")
    try:
        from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline, StableDiffusionPipeline
        
        is_sdxl = "sdxl" in llm_name.lower() or "turbo" in llm_name.lower()
        
        if is_sdxl:
            pipeline_class = StableDiffusionXLImg2ImgPipeline if pipeline_type == "img2img" else StableDiffusionXLPipeline
        else:
            pipeline_class = StableDiffusionPipeline
            
        pipe = pipeline_class.from_pretrained(
            llm_name, 
            torch_dtype=dtype,
            cache_dir="/models",
            token=os.environ.get("HF_TOKEN")
        )
            
        pipe.to("cpu")
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
            for key, shape in KLEIN_MISSING_KEYS.items():
                if key not in checkpoint:
                    checkpoint[key] = torch.zeros(shape, dtype=torch.float32)
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
        
    device = "cpu"
    dtype = torch.bfloat16  # float32 would need ~23GB; bfloat16 halves it to ~11.5GB

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
        pipe = pipeline_class.from_pretrained(
            hf_cache,
            transformer=transformer,
            torch_dtype=dtype,
            local_files_only=True,
        )

        lora_path = "/models/flux-spritesheet-lora.safetensors"
        if os.path.exists(lora_path):
            try:
                logger.info(f"Loading LoRA weights from {lora_path}...")
                pipe.load_lora_weights(lora_path)
                pipe.fuse_lora()
                logger.info("LoRA fused successfully.")
            except Exception as lora_e:
                logger.warning(f"LoRA loading failed, continuing without it: {lora_e}")

        pipe.to("cpu")
        pipes[cache_key] = pipe
        logger.info("FLUX pipeline loaded and ready.")
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

def remove_background(master):
    try:
        master = master.convert("RGBA")
        corners = [(0,0), (master.width-1, 0), (0, master.height-1), (master.width-1, master.height-1)]
        bg_r, bg_g, bg_b = 255, 255, 255
        
        for cx, cy in corners:
            r, g, b, *_ = master.getpixel((cx, cy))
            if r + g + b > 50:
                bg_r, bg_g, bg_b = r, g, b
                break

        new_data = []
        pixels_removed = 0
        total_pix = master.width * master.height
        
        for r, g, b, a in master.getdata():
            if abs(r - bg_r) < 22 and abs(g - bg_g) < 22 and abs(b - bg_b) < 22:
                new_data.append((0, 0, 0, 0))
                pixels_removed += 1
            else:
                new_data.append((r, g, b, a))
        
        if pixels_removed / total_pix > 0.98:
            logger.warning("Safety trigger: Background removal too aggressive. Keeping background.")
        else:
            master.putdata(new_data)
            
        return master
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
    negative = "multiple characters, two characters, group, horde, crowd, twins, clones, split screen, collage, grid, set, blurry, deformed, extra limbs, cropped, low quality, watermark, text, noise, messy pixels, artifacting, gradient, shadows on background"

    clean_prompt = prompt.replace("PixelartFSS", "").strip().lstrip(",").strip()
    
    # Strictly aligned prefix: "PixelartFSS, idle front,"
    full_prompt = f"PixelartFSS, idle front, solo individual {clean_prompt}, centered, lone character, no duplicates, one standalone character, flat solid transparent background, high quality pixel art, 16-bit, sharp focus" if "background" not in clean_prompt.lower() else f"PixelartFSS, idle front, solo individual {clean_prompt}, centered, lone character, no duplicates, one standalone character, high quality pixel art, sharp focus"

    start_time = time.time()
    
    # Dynamic parameters based on model type
    is_turbo = "turbo" in llm_name.lower() or "schnell" in llm_name.lower()
    num_steps = 4 if is_turbo else 35
    guidance = 0.0 if is_turbo else 9.0

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
            negative_prompt=negative,
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
    img = remove_background(img)

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
        core_img = Image.open(core_path).convert("RGB")
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
    
    p = get_flux_pipeline(pipeline_type="img2img")
    if not p:
        update_task_record(task_id, error_msg="Pipeline load failed")
        return {"error": "Pipeline load failed"}
        
    action_strips = []
    
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
        strength = 0.95 if is_dynamic else 0.70
        
        prompt = f"spritesheet, {trigger}, {base_prompt}, 1x{motion_steps} horizontal grid, {motion_steps} animation frames in a row"
        negative = "multiple characters, two characters, split screen, collage, grid, set, blurry, deformed, extra limbs, cropped, low quality, watermark, text, noise, messy pixels, artifacting, gradient, shadows on background"
        
        # Optimization: FLUX.1-schnell is a 4-step model. 
        # Reducing steps and setting guidance to 0 for massive speedup on CPU.
        num_inf_steps = 4 
        
        def action_progress_callback(pipe, i, t, callback_kwargs):
            pct = int((i / num_inf_steps) * 100)
            logger.info(f"  > '{action}' progress: {pct}%")
            # Update DB periodically
            if i % 2 == 0:
                update_task_record(task_id, progress_msg=f"{action}: {pct}%")
            return callback_kwargs

        generator = torch.Generator("cpu").manual_seed(parent_seed + i)
        try:
            grid_img = p(
                prompt=prompt,
                image=core_img,
                strength=strength,
                num_inference_steps=num_inf_steps,
                guidance_scale=0.0,
                generator=generator,
                width=frame_width * motion_steps,
                height=frame_height,
                callback_on_step_end=action_progress_callback
            ).images[0]
            
            # Slice the generated horizontal grid
            grid_w, grid_h = grid_img.size
            qw = grid_w // motion_steps
            
            logger.info(f"  > Action '{action}' complete. Removing background from grid...")
            grid_img = remove_background(grid_img)
            
            logger.info(f"  > Slicing action strip: {grid_w}x{grid_h} into {motion_steps} frames...")
            action_strip = Image.new("RGBA", (frame_width * motion_steps, frame_height), (0,0,0,0))
            for f in range(motion_steps):
                frame = grid_img.crop((f * qw, 0, (f + 1) * qw, grid_h))
                if frame.width != frame_width or frame.height != frame_height:
                    frame = frame.resize((frame_width, frame_height), Image.Resampling.LANCZOS)
                action_strip.paste(frame, (f * frame_width, 0), frame)
            
            action_strips.append(action_strip)
            logger.info(f"  > Action '{action}' processed successfully.")
        except Exception as e:
            logger.error(f"Action '{action}' generation failed: {e}")
            update_task_record(task_id, error_msg=f"Failed on {action}")
            return {"error": str(e)}

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
