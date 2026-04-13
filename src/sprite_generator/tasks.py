import os
import io
import json
import time
import uuid
import torch
import psycopg2
import random
import logging
import requests
from collections import namedtuple
from celery import Celery, chord, group
from diffusers import (
    StableDiffusionPipeline, 
    StableDiffusionXLPipeline, 
    StableDiffusionXLImg2ImgPipeline,
    DPMSolverMultistepScheduler
)
from PIL import Image, ImageDraw
import base64
import io

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
DB_URL = os.environ.get("DB_URL")
IMAGES_DIR = "/app/images"

celery_app = Celery("sprite_tasks", broker=REDIS_URL, backend=REDIS_URL)

pipes = {}
PipelineOutput = namedtuple("PipelineOutput", ["images"])

class LLMProxyPipeline:
    def __init__(self, model_name, fallback_pipeline=None):
        self.model_name = model_name
        self.endpoint = "http://llm_engine:8080/v1/chat/completions"
        self.fallback_pipeline = fallback_pipeline
        # Mapping for llama.cpp server model selection if needed
        self.model_file_map = {
            "Aakash010/MedGemma_FineTuned": "medgemma-Q4_K_M",
            "rafacost/DreamOmni2-7.6B-GGUF": "DreamOmni2-Vlm-Model-7.6B-Q4_K_M",
            "Qwen/Qwen2-VL-7B-Instruct-GGUF": "Qwen2-VL-7B-Instruct-Q4_K_M"
        }

    def _encode_image(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def __call__(self, prompt, **kwargs):
        logger.info(f"Proxying sprite generation to LLM server ({self.model_name})")
        
        # 1. Prepare Request to VLM
        model_file = self.model_file_map.get(self.model_name, self.model_name)
        messages = [
            {"role": "system", "content": "You are a pixel art sprite expert. Analyze character images and describe their movement for animation frames. Be brief but highly descriptive of posture and specific limbs."},
            {"role": "user", "content": f"Enhance this sprite generation prompt: {prompt}"}
        ]
        
        try:
            resp = requests.post(
                self.endpoint,
                json={
                    "model": model_file,
                    "messages": messages,
                    "max_tokens": 150
                },
                timeout=30
            )
            if resp.status_code != 200:
                logger.error(f"Proxy call failed ({resp.status_code}): {resp.text}")
                return self.fallback_pipeline(prompt, **kwargs) if self.fallback_pipeline else PipelineOutput(images=[])
            
            enhanced = resp.json()['choices'][0]['message']['content'].strip()
            # Clean up VLM output (sometimes they wrap in quotes or add 'Revised prompt:')
            enhanced = enhanced.replace("Revised prompt:", "").replace('"', '').strip()
            
            # 2. Use Fallback SD Pipeline with Enhanced Prompt
            if self.fallback_pipeline:
                logger.info("Using fallback SD pipeline with VLM guidance...")
                return self.fallback_pipeline(enhanced, **kwargs)
            
            return PipelineOutput(images=[])
        except Exception as e:
            logger.error(f"Proxy call exception: {e}")
            if self.fallback_pipeline:
                return self.fallback_pipeline(prompt, **kwargs)
            raise e

    def enhance_animation(self, action_label, base_prompt):
        """Specifically useful for animation frames. Returns 4 descriptions."""
        logger.info(f"VLM: Requesting 4-frame animation breakdown for '{action_label}'")
        model_file = self.model_file_map.get(self.model_name, self.model_name)
        
        prompt_text = (
            f"Describe 4 sequential movement poses for a character performing: {action_label}. "
            f"The character looks like: {base_prompt}. "
            "Respond ONLY in exactly 4 lines, one for each pose. Each line must be a concise visual description of posture and limb placement. "
            "Do NOT mention 'frames', 'borders', 'grid', or 'layout'."
        )
        
        messages = [
            {"role": "system", "content": "You are a professional pixel art animator. Provide exactly 4 lines of visual descriptions for sequential animation poses. Focus on weight distribution and movement. Never describe the background or layout."},
            {"role": "user", "content": prompt_text}
        ]

        try:
            resp = requests.post(
                self.endpoint,
                json={
                    "model": model_file,
                    "messages": messages,
                    "max_tokens": 300
                },
                timeout=45
            )
            if resp.status_code != 200:
                logger.error(f"VLM enhance_animation failed ({resp.status_code}): {resp.text}")
                return [f"Frame {i+1} of {action_label} animation" for i in range(4)]
            
            content = resp.json()['choices'][0]['message']['content'].strip()
            lines = [line.strip() for line in content.split('\n') if line.strip()][:4]
            
            # Pad if less than 4
            while len(lines) < 4:
                lines.append(f"Frame {len(lines)+1} of {action_label}")
            
            logger.info(f"VLM Animation Guides: {lines}")
            return lines
        except Exception as e:
            logger.error(f"VLM enhance_animation failed: {e}")
            return [f"Frame {i+1} of {action_label}" for i in range(4)]

def get_pipeline(llm_name: str = "stabilityai/sdxl-turbo", pipeline_type: str = "text2img"):
    if llm_name == "models--stabilityai--sdxl-turbo":
        llm_name = "stabilityai/sdxl-turbo"
    global pipes
    
    cache_key = f"{llm_name}_{pipeline_type}"
    if cache_key in pipes:
        return pipes[cache_key]
    
    # Memory Optimization: Check for existing pipeline of the same model to share components
    other_type = "img2img" if pipeline_type == "text2img" else "text2img"
    other_key = f"{llm_name}_{other_type}"
    shared_components = None
    if other_key in pipes and not hasattr(pipes[other_key], "fallback_pipeline"):
        logger.info(f"Memory Save: Sharing components from '{other_key}' to create '{cache_key}'")
        shared_components = pipes[other_key].components

    if "gguf" in llm_name.lower() or "medgemma" in llm_name.lower() or "dreamomni" in llm_name.lower() or "qwen" in llm_name.lower():
        logger.info(f"Using LLMProxyPipeline for '{llm_name}' (Type: {pipeline_type})")
        fallback = get_pipeline("stabilityai/sdxl-turbo", pipeline_type=pipeline_type)
        pipes[cache_key] = LLMProxyPipeline(llm_name, fallback_pipeline=fallback)
        return pipes[cache_key]

    # Dynamic Hardware Detection
    settings = get_compute_settings()
    device = "cuda" if torch.cuda.is_available() and settings.get("compute_mode") == "cuda" else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    logger.info(f"Loading '{llm_name}' ({pipeline_type}) on {device.upper()} ({dtype})...")
    try:
        if "sdxl" in llm_name.lower():
            pipeline_class = StableDiffusionXLImg2ImgPipeline if pipeline_type == "img2img" else StableDiffusionXLPipeline
        else:
            pipeline_class = StableDiffusionPipeline 

        if shared_components:
            # Re-use already loaded weights
            pipe = pipeline_class(**shared_components)
        else:
            # Load from disk
            pipe = pipeline_class.from_pretrained(
                llm_name,
                torch_dtype=dtype,
                cache_dir="/models",
                token=os.environ.get("HF_TOKEN"),
            )
        
        pipe.to(device)
        
        if device == "cuda":
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except:
                pipe.enable_attention_slicing()
        else:
            pipe.enable_attention_slicing()
            
        logger.info(f"Pipeline '{llm_name}' ({pipeline_type}) ready (Cached components: {shared_components is not None})")
        pipes[cache_key] = pipe
    except Exception as e:
        logger.error(f"Error loading model '{llm_name}' ({pipeline_type}): {e}")
        return None
    return pipes[cache_key]

def get_compute_settings():
    conn = get_db()
    if not conn: return {"compute_mode": "cpu"}
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT key, value FROM app_settings")
            rows = cur.fetchall()
            return {row[0]: row[1] if isinstance(row[1], dict) else json.loads(row[1]) for row in rows}
    except Exception as e:
        logger.error(f"Error fetching settings in worker: {e}")
        return {"compute_mode": "cpu"}
    finally: conn.close()

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
                       requested_actions: list = None, seed: int = None):
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

@celery_app.task(name="tasks.generate_sprite_task", bind=True)
def generate_sprite_task(self, prompt: str, llm_name: str = "stabilityai/sdxl-turbo"):
    task_id = self.request.id
    logger.info(f"Task {task_id} generated sprite with llm {llm_name}")
    p = get_pipeline(llm_name)
    if not p:
        update_task_record(task_id, error_msg="Model failed to load on worker")
        return {"error": "Model failed to load"}

    DIRECTIONS = [
        ("PixelartFSS", "front"),
        ("PixelartBSS", "back"),
        ("PixelartLSS", "left"),
        ("PixelartRSS", "right"),
    ]
    
    seed = random.randint(0, 10**9)
    generator = torch.Generator("cpu").manual_seed(seed)
    negative = "multiple characters, two characters, group, horde, crowd, split screen, collage, grid, set, blurry, deformed, extra limbs, cropped, low quality, watermark, text, noise, messy pixels, artifacting, gradient, shadows on background"

    clean_prompt = prompt
    for t, _ in DIRECTIONS:
        clean_prompt = clean_prompt.replace(t, "").strip().lstrip(",").strip()
    
    full_prompt_base = f"solo individual {clean_prompt}, centered, lone character, no duplicates, one standalone character, flat solid white background, high quality pixel art, 16-bit, sharp focus" if "background" not in clean_prompt.lower() else f"solo individual {clean_prompt}, centered, lone character, no duplicates, one standalone character, high quality pixel art, sharp focus"

    strips = []
    start_time = time.time()
    num_steps = 35

    try:
        for i, (trigger, label) in enumerate(DIRECTIONS):
            current_prompt = f"{trigger}, {full_prompt_base}"
            logger.info(f"Generating {label} (Seed {seed}): {current_prompt}")
            update_task_record(task_id, progress_pct=int((i/4)*100), progress_msg=f"Pass {i+1}/4: {label}", seed=seed)

            def progress_callback(step, timestep, latents):
                if step % 4 == 0:
                    step_pct = (step / num_steps)
                    total_pct = int(((i / 4) + (step_pct / 4)) * 100)
                    update_task_record(task_id, progress_pct=total_pct, progress_msg=f"Pass {i+1}/4 ({label}): {int(step_pct*100)}%")
                    self.update_state(state="PROGRESS", meta={"pct": total_pct, "msg": label})

            img = p(
                current_prompt,
                negative_prompt=negative,
                height=512,
                width=512,
                num_inference_steps=num_steps,
                guidance_scale=9.0,
                generator=generator,
                callback=progress_callback,
                callback_steps=1
            ).images[0]
            strips.append(img)
            
    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}", exc_info=True)
        update_task_record(task_id, error_msg=f"Generation failed: {str(e)}")
        return {"error": str(e)}

    end_time = time.time()
    total_duration_ms = (end_time - start_time) * 1000
    log_stats(task_id, llm_name, clean_prompt, num_steps * 4, start_time, end_time, total_duration_ms)

    update_task_record(task_id, progress_pct=90, progress_msg="Finalizing: Stitching...")

    total_height = sum(img.height for img in strips)
    master = Image.new("RGB", (strips[0].width, total_height))
    y = 0
    for img in strips:
        master.paste(img, (0, y))
        y += img.height

    master = remove_background(master)
    filename = f"sprite_{uuid.uuid4().hex[:12]}.png"
    filepath = os.path.join(IMAGES_DIR, filename)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    master.save(filepath, format="PNG")

    update_task_record(task_id, file_path=filepath, duration_ms=total_duration_ms, 
                       error_msg=None, progress_pct=100, progress_msg="Complete", image_type="spritesheet", seed=seed)

    return {"status": "success", "url": f"/images/{filename}", "duration_ms": total_duration_ms }

@celery_app.task(name="tasks.generate_core_task", bind=True)
def generate_core_task(self, prompt: str, llm_name: str = "stabilityai/sdxl-turbo"):
    task_id = self.request.id
    logger.info(f"Task {task_id} generated core with llm {llm_name}")
    p = get_pipeline(llm_name)
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
    is_turbo = "turbo" in llm_name.lower()
    num_steps = 4 if is_turbo else 35
    guidance = 1.0 if is_turbo else 9.0

    try:
        update_task_record(task_id, progress_pct=0, progress_msg="Generating core image...", seed=seed)

        def progress_callback(step, timestep, latents):
            if step % 1 == 0:
                pct = int((step / num_steps) * 100)
                update_task_record(task_id, progress_pct=pct, progress_msg=f"Generating: {int(pct)}%")
                self.update_state(state="PROGRESS", meta={"pct": pct, "msg": "Generating core image"})

        img = p(
            full_prompt,
            negative_prompt=negative,
            height=512,
            width=512,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            generator=generator,
            callback=progress_callback,
            callback_steps=1
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

@celery_app.task(name="tasks.generate_sheet_task", bind=True)
def generate_sheet_task(self, parent_id: int, actions: list, llm_name: str = "stabilityai/sdxl-turbo", frame_width: int = 128, frame_height: int = 128):
    task_id = self.request.id
    logger.info(f"Orchestrating Distributed Sheet Task {task_id} with llm {llm_name}")
    
    # 1. Fetch parent info for all sub-tasks
    conn = get_db()
    if not conn:
        update_task_record(task_id, error_msg="DB connection failed")
        return {"error": "DB connection failed"}
    
    parent_prompt = ""
    parent_seed = None
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT prompt, seed FROM sprite_images WHERE id = %s", (parent_id,))
                row = cur.fetchone()
                if row:
                    parent_prompt, parent_seed = row
    except Exception as e:
        logger.error(f"Error fetching parent: {e}")
    finally:
        conn.close()

    if parent_seed is None: parent_seed = random.randint(0, 10**9)

    # 2. Fire off the Chord
    # Group of action generators -> Finalizer
    header = [
        generate_action_strip_task.s(task_id, action, i, len(actions), parent_id, parent_prompt, parent_seed, llm_name, frame_width, frame_height)
        for i, action in enumerate(actions)
    ]
    
    callback = finalize_sheet_task.s(task_id, parent_id, actions, parent_seed, llm_name)
    
    update_task_record(task_id, progress_pct=5, progress_msg="Distributed: Queuing sub-tasks...", requested_actions=actions)
    
    chord(header)(callback)
    return {"status": "orchestrated", "task_id": task_id}

@celery_app.task(name="tasks.generate_action_strip_task", bind=True)
def generate_action_strip_task(self, main_task_id: str, action: str, action_index: int, total_actions: int, parent_id: int, parent_prompt: str, parent_seed: int, llm_name: str, frame_width: int = 128, frame_height: int = 128):
    logger.info(f"Sub-task {self.request.id} starting action '{action}' ({action_index+1}/{total_actions}) for main task {main_task_id}")
    
    # Use Img2Img pipeline for Stage 2
    p = get_pipeline(llm_name, pipeline_type="img2img")
    
    # 0. Fetch Core Image for Img2Img
    core_path = get_core_image_path(parent_id)
    core_img = None
    if core_path and os.path.exists(core_path):
        core_img = Image.open(core_path).convert("RGB")
        logger.info(f"Loaded core image from {core_path} for Img2Img.")
    else:
        logger.warning(f"Core image not found at {core_path}. Falling back to Text2Img.")
        p = get_pipeline(llm_name, pipeline_type="text2img")

    clean_prompt = parent_prompt.replace("PixelartFSS", "").strip().lstrip(",").strip()
    base_prompt = f"{clean_prompt}, flat solid white background, high quality pixel art, 16-bit, sharp focus" if "background" not in clean_prompt.lower() else f"{clean_prompt}, high quality pixel art, sharp focus"
    negative = "multiple characters, two characters, split screen, collage, grid, set, blurry, deformed, extra limbs, cropped, low quality, watermark, text, noise, messy pixels, artifacting, gradient, shadows on background"
    
    # Optimized settings for SDXL-Turbo Img2Img
    num_steps = 4 
    guidance = 0.0 # Turbo usually works best with 0 or 1 guidance in Img2Img
    
    # 1. Map triggers and identify if this is a movement/dynamic action
    action_lower = action.lower()
    is_dynamic = any(kw in action_lower for kw in ["move", "walk", "attack", "damage", "burning"])
    strength = 0.75 if is_dynamic else 0.5
    
    trigger = ""
    if "move right" in action_lower: 
        trigger = "side view profile, walking right, character facing right, dynamic legs moving"
    elif "move left" in action_lower: 
        trigger = "side view profile, walking left, character facing left, dynamic legs moving"
    elif "move down" in action_lower: 
        trigger = "walking front, character facing forward, legs moving"
    elif "move up" in action_lower: 
        trigger = "walking back, character facing away, legs moving"
    elif "idle" in action_lower: 
        trigger = "idle standing"
    elif "attack" in action_lower: 
        trigger = "dramatic action pose, fast strike attack, swinging arms"
    elif "got damage" in action_lower: 
        trigger = "taking damage, hurt posture, recoiling"
    elif "burning" in action_lower: 
        trigger = "in flames burning, expressive movement"
    else: 
        trigger = action

    # 2. Get 4 Frame descriptions
    is_vlm = isinstance(p, LLMProxyPipeline)
    frame_descriptions = []
    if is_vlm:
        vlm_poses = p.enhance_animation(trigger, base_prompt)
        frame_descriptions = [f"{pose}, {base_prompt}" for pose in vlm_poses]
    else:
        frame_descriptions = [
            f"Frame 1 of {trigger} animation, {base_prompt}",
            f"Frame 2 of {trigger} animation, movement sequence, stride, {base_prompt}",
            f"Frame 3 of {trigger} animation, movement sequence, stride, {base_prompt}",
            f"Frame 4 of {trigger} animation, finish pose, {base_prompt}"
        ]

    # 2. Generate 4 frames
    settings = get_compute_settings()
    device = "cuda" if torch.cuda.is_available() and settings.get("compute_mode") == "cuda" else "cpu"
    
    action_frames = []
    for f_idx, frame_prompt in enumerate(frame_descriptions):
        generator = torch.Generator(device).manual_seed(parent_seed + f_idx)
        
        logger.info(f"Worker {self.request.id} Generating Frame {f_idx+1}/4 for '{action}'")
        
        def frame_progress_callback(step, timestep, latents):
            # Update every step for Stage 2 (since it only has 4 steps total)
            if step % 1 == 0:
                frame_pct = (step / (num_steps - 1)) if num_steps > 1 else 1.0
                global_pct = int(((action_index / total_actions) + ((f_idx + frame_pct) / 4 / total_actions)) * 100)
                safe_pct = min(global_pct, 98)
                update_task_record(main_task_id, progress_pct=safe_pct, progress_msg=f"Sheet: {action} ({f_idx+1}/4) {int(frame_pct*100)}%")
                self.update_state(state="PROGRESS", meta={"pct": safe_pct, "msg": f"{action} F{f_idx+1}"})

        # Pipeline Call (Switches based on core_img availability)
        pipe_args = {
            "prompt": frame_prompt,
            "negative_prompt": negative,
            "height": 512,
            "width": 512,
            "num_inference_steps": num_steps,
            "guidance_scale": guidance,
            "generator": generator,
            "callback": frame_progress_callback,
            "callback_steps": 1
        }
        
        if core_img:
            pipe_args["image"] = core_img
            pipe_args["strength"] = strength
            # Remove height/width for Img2Img as it takes from source or defaults
            pipe_args.pop("height")
            pipe_args.pop("width")
        
        img = p(**pipe_args).images[0]
        
        img_transparent = remove_background(img)
        
        if img_transparent.width != frame_width or img_transparent.height != frame_height:
            img_transparent = img_transparent.resize((frame_width, frame_height), Image.Resampling.LANCZOS)
            
        action_frames.append(img_transparent)

    # 3. Stitch Action Strip
    action_strip = Image.new("RGBA", (frame_width * 4, frame_height), (0,0,0,0))
    for x_idx, frame_img in enumerate(action_frames):
        action_strip.paste(frame_img, (x_idx * frame_width, 0), frame_img)

    # 4. Save intermediate
    buf = io.BytesIO()
    action_strip.save(buf, format="PNG")
    b64_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    logger.info(f"Sub-task {self.request.id} for action '{action}' COMPLETE")
    return {"action": action, "image_b64": b64_data}

@celery_app.task(name="tasks.finalize_sheet_task", bind=True)
def finalize_sheet_task(self, results, main_task_id: str, parent_id: int, actions_order: list, parent_seed: int, llm_name: str):
    logger.info(f"Finalizing distributed task {main_task_id}")
    
    # Sort results to match requested action order
    results_map = {res['action']: res['image_b64'] for res in results}
    
    strips = []
    component_files = []
    
    for action in actions_order:
        if action in results_map:
            # Reconstruct image from b64
            img_data = base64.b64decode(results_map[action])
            img = Image.open(io.BytesIO(img_data))
            strips.append(img)
            
            # Save as persistent component
            comp_filename = f"comp_{uuid.uuid4().hex[:12]}.png"
            comp_filepath = os.path.join(IMAGES_DIR, comp_filename)
            img.save(comp_filepath, format="PNG")
            component_files.append(f"/images/{comp_filename}")

    # Final vertical stitch
    sheet_w = strips[0].width if strips else 512
    sheet_h = sum(s.height for s in strips)
    master = Image.new("RGBA", (sheet_w, sheet_h), (0,0,0,0))
    y = 0
    for action_strip in strips:
        master.paste(action_strip, (0, y), action_strip)
        y += action_strip.height

    filename = f"sheet_{uuid.uuid4().hex[:12]}.png"
    filepath = os.path.join(IMAGES_DIR, filename)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    master.save(filepath, format="PNG")

    # Update the MAIN task record
    update_task_record(main_task_id, file_path=filepath, 
                       error_msg=None, progress_pct=100, progress_msg="Complete", image_type="spritesheet",
                       parent_id=parent_id, requested_actions=actions_order, components=component_files, seed=parent_seed)

    logger.info(f"Master Sheet {filename} SAVED for task {main_task_id}")
    return {"status": "success", "url": f"/images/{filename}"}
