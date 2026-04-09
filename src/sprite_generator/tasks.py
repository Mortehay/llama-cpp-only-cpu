import os
import io
import json
import time
import uuid
import torch
import psycopg2
import random
import logging
from celery import Celery
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
DB_URL = os.environ.get("DB_URL")
IMAGES_DIR = "/app/images"

celery_app = Celery("sprite_tasks", broker=REDIS_URL, backend=REDIS_URL)

pipes = {}

def get_pipeline(llm_name: str = "stabilityai/sdxl-turbo"):
    if llm_name == "models--stabilityai--sdxl-turbo":
        llm_name = "stabilityai/sdxl-turbo"
    global pipes
    if llm_name in pipes:
        return pipes[llm_name]
    logger.info(f"Loading '{llm_name}' Stable Diffusion Pipeline on CPU (Worker)...")
    try:
        pipeline_class = StableDiffusionXLPipeline if "sdxl" in llm_name.lower() else StableDiffusionPipeline
        pipe = pipeline_class.from_pretrained(
            llm_name,
            torch_dtype=torch.float32,
            cache_dir="/models",
            token=os.environ.get("HF_TOKEN"),
        )
        if "sdxl" not in llm_name.lower():
            pass # Apply SD-specific options if strictly necessary
        
        pipe.to("cpu")
        pipe.enable_attention_slicing()
        logger.info(f"Pipeline '{llm_name}' loaded successfully.")
        pipes[llm_name] = pipe
    except Exception as e:
        logger.error(f"Error loading model '{llm_name}': {e}")
        return None
    return pipes[llm_name]

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
                    update_fields.append("progress_pct = %s")
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
    negative = "blurry, deformed, extra limbs, cropped, low quality, watermark, text, noise, messy pixels, artifacting, gradient, shadows on background"

    clean_prompt = prompt
    for t, _ in DIRECTIONS:
        clean_prompt = clean_prompt.replace(t, "").strip().lstrip(",").strip()
    
    full_prompt_base = f"{clean_prompt}, flat solid white background, high quality pixel art, 16-bit, sharp focus" if "background" not in clean_prompt.lower() else f"{clean_prompt}, high quality pixel art, sharp focus"

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
    negative = "blurry, deformed, extra limbs, cropped, low quality, watermark, text, noise, messy pixels, artifacting, gradient, shadows on background"

    clean_prompt = prompt.replace("PixelartFSS", "").strip().lstrip(",").strip()
    
    # Strictly aligned prefix length: "PixelartFSS, idle front,"
    full_prompt = f"PixelartFSS, idle front, {clean_prompt}, flat solid white background, high quality pixel art, 16-bit, sharp focus" if "background" not in clean_prompt.lower() else f"PixelartFSS, idle front, {clean_prompt}, high quality pixel art, sharp focus"

    start_time = time.time()
    num_steps = 35

    try:
        update_task_record(task_id, progress_pct=0, progress_msg="Generating core image...", seed=seed)

        def progress_callback(step, timestep, latents):
            if step % 4 == 0:
                pct = int((step / num_steps) * 100)
                update_task_record(task_id, progress_pct=pct, progress_msg=f"Generating: {int(pct)}%")
                self.update_state(state="PROGRESS", meta={"pct": pct, "msg": "Generating core image"})

        img = p(
            full_prompt,
            negative_prompt=negative,
            height=512,
            width=512,
            num_inference_steps=num_steps,
            guidance_scale=9.0,
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

    # Core physics logic: The model natively generates a 4x1 animation sequence.
    # We strip out Frame 1 dynamically to present solely the physical character structure.
    width, height = img.size
    frame_width = width // 4
    core_crop = img.crop((0, 0, frame_width, height))
    
    filename = f"core_{uuid.uuid4().hex[:12]}.png"
    filepath = os.path.join(IMAGES_DIR, filename)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    core_crop.save(filepath, format="PNG")

    update_task_record(task_id, file_path=filepath, duration_ms=total_duration_ms, 
                       error_msg=None, progress_pct=100, progress_msg="Complete", image_type="core", seed=seed)

    return {"status": "success", "url": f"/images/{filename}", "duration_ms": total_duration_ms }

@celery_app.task(name="tasks.generate_sheet_task", bind=True)
def generate_sheet_task(self, parent_id: int, actions: list, llm_name: str = "stabilityai/sdxl-turbo"):
    task_id = self.request.id
    logger.info(f"Task {task_id} generated sheet with llm {llm_name}")
    p = get_pipeline(llm_name)
    if not p:
        update_task_record(task_id, error_msg="Model failed to load on worker")
        return {"error": "Model failed to load"}

    # Fetch parent info from DB
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

    if parent_seed is None:
        parent_seed = random.randint(0, 10**9)

    clean_prompt = parent_prompt.replace("PixelartFSS", "").strip().lstrip(",").strip()
    base_prompt = f"{clean_prompt}, flat solid white background, high quality pixel art, 16-bit, sharp focus" if "background" not in clean_prompt.lower() else f"{clean_prompt}, high quality pixel art, sharp focus"
    negative = "blurry, deformed, extra limbs, cropped, low quality, watermark, text, noise, messy pixels, artifacting, gradient, shadows on background"
    
    strips = []
    component_files = []
    start_time = time.time()
    num_steps = 35
    total_actions = len(actions)

    try:
        for i, action in enumerate(actions):
            generator = torch.Generator("cpu").manual_seed(parent_seed)

            trigger = ""
            action_lower = action.lower()
            # Precisely structured positional alignment map natively matched to the identical 3-slot physics template structure.
            if "move right" in action_lower: trigger = "PixelartRSS, walk right"
            elif "move left" in action_lower: trigger = "PixelartLSS, walk left."
            elif "move down" in action_lower or "move front" in action_lower: trigger = "PixelartFSS, walk front"
            elif "move top" in action_lower or "move up" in action_lower or "move back" in action_lower: trigger = "PixelartBSS, walk back."
            elif "idle" in action_lower: trigger = "PixelartFSS, idle front"
            elif "attack" in action_lower: trigger = "PixelartFSS, fast strike"
            elif "got damage" in action_lower: trigger = "PixelartFSS, take damage"
            elif "burning" in action_lower: trigger = "PixelartFSS, in flames!!"
            else: trigger = f"PixelartFSS, {action}       "

            current_prompt = f"{trigger}, {base_prompt}"
            logger.info(f"Generating {action} (Seed {parent_seed}): {current_prompt}")
            
            update_task_record(task_id, progress_pct=int((i/total_actions)*100), progress_msg=f"Pass {i+1}/{total_actions}: {action}", seed=parent_seed)

            def progress_callback(step, timestep, latents):
                if step % 4 == 0:
                    step_pct = (step / num_steps)
                    total_pct = int(((i / total_actions) + (step_pct / total_actions)) * 100)
                    update_task_record(task_id, progress_pct=total_pct, progress_msg=f"Pass {i+1}/{total_actions} ({action}): {int(step_pct*100)}%")
                    self.update_state(state="PROGRESS", meta={"pct": total_pct, "msg": action})

            img = p(
                prompt=current_prompt,
                negative_prompt=negative,
                height=512,
                width=512,
                num_inference_steps=num_steps,
                guidance_scale=9.0,
                generator=generator,
                callback=progress_callback,
                callback_steps=1
            ).images[0]
            
            img_transparent = remove_background(img)
            
            comp_filename = f"comp_{uuid.uuid4().hex[:12]}.png"
            comp_filepath = os.path.join(IMAGES_DIR, comp_filename)
            img_transparent.save(comp_filepath, format="PNG")
            
            component_files.append(f"/images/{comp_filename}")
            strips.append(img_transparent)
            
    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}", exc_info=True)
        update_task_record(task_id, error_msg=f"Generation failed: {str(e)}")
        return {"error": str(e)}

    end_time = time.time()
    total_duration_ms = (end_time - start_time) * 1000
    log_stats(task_id, llm_name, clean_prompt, num_steps * total_actions, start_time, end_time, total_duration_ms)

    update_task_record(task_id, progress_pct=90, progress_msg="Finalizing: Stitching...")

    total_height = sum(img.height for img in strips)
    master = Image.new("RGBA", (strips[0].width, total_height), (0,0,0,0))
    y = 0
    for img in strips:
        master.paste(img, (0, y), img) 
        y += img.height

    filename = f"sheet_{uuid.uuid4().hex[:12]}.png"
    filepath = os.path.join(IMAGES_DIR, filename)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    master.save(filepath, format="PNG")

    update_task_record(task_id, file_path=filepath, duration_ms=total_duration_ms, 
                       error_msg=None, progress_pct=100, progress_msg="Complete", image_type="spritesheet",
                       parent_id=parent_id, requested_actions=actions, components=component_files, seed=parent_seed)

    return {"status": "success", "url": f"/images/{filename}", "duration_ms": total_duration_ms }
