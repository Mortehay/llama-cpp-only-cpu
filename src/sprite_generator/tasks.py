import os
import io
import json
import time
import uuid
import torch
import requests
import psycopg2
import random
from celery import Celery
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from PIL import Image

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
DB_URL = os.environ.get("DB_URL")
IMAGES_DIR = "/app/images"

celery_app = Celery("sprite_tasks", broker=REDIS_URL, backend=REDIS_URL)

# Pipeline global for the worker process
pipe = None
img2img_pipe = None

def get_pipeline():
    global pipe, img2img_pipe
    if pipe is not None and img2img_pipe is not None:
        return pipe, img2img_pipe
    print("Loading Stable Diffusion Pipeline on CPU (Worker)...")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            "Onodofthenorth/SD_PixelArt_SpriteSheet_Generator",
            torch_dtype=torch.float32,
            cache_dir="/models",
            token=os.environ.get("HF_TOKEN"),
        )
        # Using a very high quality scheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
        pipe.to("cpu")
        pipe.enable_attention_slicing()
        
        print("Reusing components for Img2Img Pipeline...")
        img2img_pipe = StableDiffusionImg2ImgPipeline(**pipe.components)
        
        print("Pipelines loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        pipe = None
        img2img_pipe = None
    return pipe, img2img_pipe

def get_db():
    if not DB_URL:
        return None
    try:
        return psycopg2.connect(DB_URL)
    except Exception as e:
        print(f"DB connection failed: {e}")
        return None

def update_task_record(task_id: str, file_path: str = None, duration_ms: float = 0, 
                       error_msg: str = None, progress_pct: int = None, progress_msg: str = None,
                       image_type: str = None, parent_id: int = None, components: list = None,
                       requested_actions: list = None):
    """Update progress or final result for a task record in the DB."""
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
                
                if update_fields:
                    values.append(task_id)
                    cur.execute(
                        f"UPDATE sprite_images SET {', '.join(update_fields)} WHERE task_id = %s",
                        tuple(values)
                    )
    except Exception as e:
        print(f"Could not update record {task_id}: {e}")
    finally:
        conn.close()

def remove_background(master):
    """ADVANCED CHROMACAY: Pick multiple corners and safety check"""
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
            print("Safety trigger: Background removal too aggressive. Keeping background.")
        else:
            master.putdata(new_data)
            
        return master
    except Exception as e:
        print(f"BG removal failed: {e}")
        return master

@celery_app.task(name="tasks.generate_sprite_task", bind=True)
def generate_sprite_task(self, prompt: str):
    task_id = self.request.id
    p, _ = get_pipeline()
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
    
    if "background" not in clean_prompt.lower():
        full_prompt_base = f"{clean_prompt}, flat solid white background, high quality pixel art, 16-bit, sharp focus"
    else:
        full_prompt_base = f"{clean_prompt}, high quality pixel art, sharp focus"

    strips = []
    start_time = time.time()
    num_steps = 35

    try:
        for i, (trigger, label) in enumerate(DIRECTIONS):
            current_prompt = f"{trigger}, {full_prompt_base}"
            print(f"Generating {label} (Seed {seed}): {current_prompt}")
            
            update_task_record(task_id, progress_pct=int((i/4)*100), progress_msg=f"Pass {i+1}/4: {label}")

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
        update_task_record(task_id, error_msg=f"Generation failed: {str(e)}")
        return {"error": str(e)}

    end_time = time.time()
    total_duration_ms = (end_time - start_time) * 1000

    update_task_record(task_id, progress_pct=90, progress_msg="Finalizing: Stitching...")

    total_height = sum(img.height for img in strips)
    master = Image.new("RGB", (strips[0].width, total_height))
    y = 0
    for img in strips:
        master.paste(img, (0, y))
        y += img.height

    print("Removing background...")
    master = remove_background(master)

    filename = f"sprite_{uuid.uuid4().hex[:12]}.png"
    filepath = os.path.join(IMAGES_DIR, filename)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    master.save(filepath, format="PNG")

    update_task_record(task_id, file_path=filepath, duration_ms=total_duration_ms, 
                       error_msg=None, progress_pct=100, progress_msg="Complete", image_type="spritesheet")

    return {"status": "success", "url": f"/images/{filename}", "duration_ms": total_duration_ms }

@celery_app.task(name="tasks.generate_core_task", bind=True)
def generate_core_task(self, prompt: str):
    task_id = self.request.id
    p, _ = get_pipeline()
    if not p:
        update_task_record(task_id, error_msg="Model failed to load on worker")
        return {"error": "Model failed to load"}

    seed = random.randint(0, 10**9)
    generator = torch.Generator("cpu").manual_seed(seed)
    negative = "blurry, deformed, extra limbs, cropped, low quality, watermark, text, noise, messy pixels, artifacting, gradient, shadows on background"

    # Assume we want a front-facing sprite as the core image
    clean_prompt = prompt.replace("PixelartFSS", "").strip().lstrip(",").strip()
    
    if "background" not in clean_prompt.lower():
        full_prompt = f"PixelartFSS, {clean_prompt}, flat solid white background, high quality pixel art, 16-bit, sharp focus"
    else:
        full_prompt = f"PixelartFSS, {clean_prompt}, high quality pixel art, sharp focus"

    start_time = time.time()
    num_steps = 35

    try:
        update_task_record(task_id, progress_pct=0, progress_msg="Generating core image...")

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
        update_task_record(task_id, error_msg=f"Generation failed: {str(e)}")
        return {"error": str(e)}

    end_time = time.time()
    total_duration_ms = (end_time - start_time) * 1000

    update_task_record(task_id, progress_pct=90, progress_msg="Finalizing: Removing background...")
    
    img = remove_background(img)

    filename = f"core_{uuid.uuid4().hex[:12]}.png"
    filepath = os.path.join(IMAGES_DIR, filename)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    img.save(filepath, format="PNG")

    update_task_record(task_id, file_path=filepath, duration_ms=total_duration_ms, 
                       error_msg=None, progress_pct=100, progress_msg="Complete", image_type="core")

    return {"status": "success", "url": f"/images/{filename}", "duration_ms": total_duration_ms }

@celery_app.task(name="tasks.generate_sheet_task", bind=True)
def generate_sheet_task(self, parent_id: int, actions: list):
    task_id = self.request.id
    _, p2 = get_pipeline()
    if not p2:
        update_task_record(task_id, error_msg="Model failed to load on worker")
        return {"error": "Model failed to load"}

    # Fetch parent info from DB
    conn = get_db()
    if not conn:
        update_task_record(task_id, error_msg="DB connection failed")
        return {"error": "DB connection failed"}
        
    parent_path = None
    parent_prompt = ""
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT file_path, prompt FROM sprite_images WHERE id = %s", (parent_id,))
                row = cur.fetchone()
                if row:
                    parent_path, parent_prompt = row
    except Exception as e:
        print(f"Error fetching parent: {e}")
    finally:
        conn.close()

    if not parent_path or not os.path.exists(parent_path):
        update_task_record(task_id, error_msg="Parent core image not found")
        return {"error": "Parent core image not found"}

    try:
        # Load and prepare init image
        init_image = Image.open(parent_path).convert("RGB") # Img2Img expects RGB typically
    except Exception as e:
        update_task_record(task_id, error_msg=f"Could not read parent image: {str(e)}")
        return {"error": str(e)}

    # Remove any trigger words from parent prompt
    clean_prompt = parent_prompt
    for x in ["PixelartFSS", "PixelartBSS", "PixelartLSS", "PixelartRSS"]:
        clean_prompt = clean_prompt.replace(x, "").strip().lstrip(",").strip()
    
    base_prompt = f"{clean_prompt}, flat solid white background, high quality pixel art, 16-bit, sharp focus"
    negative = "blurry, deformed, extra limbs, cropped, low quality, watermark, text, noise, messy pixels, artifacting, gradient, shadows on background"

    seed = random.randint(0, 10**9)
    generator = torch.Generator("cpu").manual_seed(seed)
    
    strips = []
    component_files = []
    start_time = time.time()
    num_steps = 35
    total_actions = len(actions)

    try:
        for i, action in enumerate(actions):
            # Try to map common actions to triggers if possible, otherwise just prompt it
            trigger = action
            action_lower = action.lower()
            if "move right" in action_lower: trigger = "PixelartRSS, walk right"
            elif "move left" in action_lower: trigger = "PixelartLSS, walk left"
            elif "move down" in action_lower or "move front" in action_lower: trigger = "PixelartFSS, walk down"
            elif "move top" in action_lower or "move up" in action_lower or "move back" in action_lower: trigger = "PixelartBSS, walk up"
            
            current_prompt = f"{trigger}, {base_prompt}"
            print(f"Generating {action} (Seed {seed}): {current_prompt}")
            
            update_task_record(task_id, progress_pct=int((i/total_actions)*100), progress_msg=f"Pass {i+1}/{total_actions}: {action}")

            def progress_callback(step, timestep, latents):
                if step % 4 == 0:
                    step_pct = (step / num_steps)
                    total_pct = int(((i / total_actions) + (step_pct / total_actions)) * 100)
                    update_task_record(task_id, progress_pct=total_pct, progress_msg=f"Pass {i+1}/{total_actions} ({action}): {int(step_pct*100)}%")
                    self.update_state(state="PROGRESS", meta={"pct": total_pct, "msg": action})

            # For img2img, strength > 0.6 usually changes composition a lot, < 0.4 keeps it very similar but might not change action
            img = p2(
                prompt=current_prompt,
                image=init_image,
                negative_prompt=negative,
                strength=0.75, # 0.75 is often a sweet spot to allow changes but keep colors
                num_inference_steps=num_steps,
                guidance_scale=9.0,
                generator=generator,
                callback=progress_callback,
                callback_steps=1
            ).images[0]
            
            # Remove bg on the individual frame
            img_transparent = remove_background(img)
            
            # Save the individual component
            comp_filename = f"comp_{uuid.uuid4().hex[:12]}.png"
            comp_filepath = os.path.join(IMAGES_DIR, comp_filename)
            img_transparent.save(comp_filepath, format="PNG")
            
            component_files.append(f"/images/{comp_filename}")
            strips.append(img_transparent)
            
    except Exception as e:
        update_task_record(task_id, error_msg=f"Generation failed: {str(e)}")
        return {"error": str(e)}

    end_time = time.time()
    total_duration_ms = (end_time - start_time) * 1000

    update_task_record(task_id, progress_pct=90, progress_msg="Finalizing: Stitching...")

    total_height = sum(img.height for img in strips)
    master = Image.new("RGBA", (strips[0].width, total_height), (0,0,0,0))
    y = 0
    for img in strips:
        master.paste(img, (0, y), img) # use img as mask since it has alpha
        y += img.height

    filename = f"sheet_{uuid.uuid4().hex[:12]}.png"
    filepath = os.path.join(IMAGES_DIR, filename)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    master.save(filepath, format="PNG")

    update_task_record(task_id, file_path=filepath, duration_ms=total_duration_ms, 
                       error_msg=None, progress_pct=100, progress_msg="Complete", image_type="spritesheet",
                       parent_id=parent_id, requested_actions=actions, components=component_files)

    return {"status": "success", "url": f"/images/{filename}", "duration_ms": total_duration_ms }
