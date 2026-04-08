import os
import io
import time
import uuid
import torch
import requests
import psycopg2
import random
from celery import Celery
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
DB_URL = os.environ.get("DB_URL")
IMAGES_DIR = "/app/images"

celery_app = Celery("sprite_tasks", broker=REDIS_URL, backend=REDIS_URL)

# Pipeline global for the worker process
pipe = None

def get_pipeline():
    global pipe
    if pipe is not None:
        return pipe
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
        print("Pipeline loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        pipe = None
    return pipe

def get_db():
    if not DB_URL:
        return None
    try:
        return psycopg2.connect(DB_URL)
    except Exception as e:
        print(f"DB connection failed: {e}")
        return None

def update_task_record(task_id: str, file_path: str = None, duration_ms: float = 0, 
                       error_msg: str = None, progress_pct: int = None, progress_msg: str = None):
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

@celery_app.task(name="tasks.generate_sprite_task", bind=True)
def generate_sprite_task(self, prompt: str):
    task_id = self.request.id
    p = get_pipeline()
    if not p:
        update_task_record(task_id, error_msg="Model failed to load on worker")
        return {"error": "Model failed to load"}

    DIRECTIONS = [
        ("PixelartFSS", "front"),
        ("PixelartBSS", "back"),
        ("PixelartLSS", "left"),
        ("PixelartRSS", "right"),
    ]
    
    # CRITICAL FOR QUALITY & CONSISTENCY: 
    # Use the same seed for all 4 directions so it generates the SAME character.
    seed = random.randint(0, 10**9)
    generator = torch.Generator("cpu").manual_seed(seed)
    
    # QUALITY: Enhanced Negative Prompt
    negative = "blurry, deformed, extra limbs, cropped, low quality, watermark, text, noise, messy pixels, artifacting, gradient, shadows on background"

    # QUALITY: Enhanced Prompt Prefix
    clean_prompt = prompt
    for t, _ in DIRECTIONS:
        clean_prompt = clean_prompt.replace(t, "").strip().lstrip(",").strip()
    
    # Force a solid background for better post-processing
    if "background" not in clean_prompt.lower():
        full_prompt_base = f"{clean_prompt}, flat solid white background, high quality pixel art, 16-bit, sharp focus"
    else:
        full_prompt_base = f"{clean_prompt}, high quality pixel art, sharp focus"

    strips = []
    start_time = time.time()
    num_steps = 35 # Increased for higher quality detail

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
                guidance_scale=9.0, # Balanced for creativity vs consistency
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

    # Stitch vertically
    total_height = sum(img.height for img in strips)
    master = Image.new("RGB", (strips[0].width, total_height))
    y = 0
    for img in strips:
        master.paste(img, (0, y))
        y += img.height

    # ADVANCED CHROMACAY: Pick multiple corners and safety check
    print("Removing background...")
    try:
        master = master.convert("RGBA")
        # Sample multiple corners
        corners = [(0,0), (master.width-1, 0), (0, master.height-1), (master.width-1, master.height-1)]
        bg_r, bg_g, bg_b = 255, 255, 255
        
        # Look for the most common corner color
        for cx, cy in corners:
            r, g, b, *_ = master.getpixel((cx, cy))
            if r + g + b > 50: # Avoid overly dark pixels as background
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
        
        # If removal wiped everything, it matched character color. Revert to 12.
        if pixels_removed / total_pix > 0.98:
            print("Safety trigger: Background removal too aggressive. Keeping background.")
        else:
            master.putdata(new_data)
    except Exception as e:
        print(f"BG removal failed: {e}")

    # Save to disk
    filename = f"sprite_{uuid.uuid4().hex[:12]}.png"
    filepath = os.path.join(IMAGES_DIR, filename)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    master.save(filepath, format="PNG")

    update_task_record(task_id, file_path=filepath, duration_ms=total_duration_ms, 
                       error_msg=None, progress_pct=100, progress_msg="Complete")

    # Log stats
    try:
        tokens = float(len(clean_prompt.split()) * 10)
        requests.post("http://stats_collector:8000/v1/internal/log_stats", json={
            "model_name": "Onodofthenorth/SD_PixelArt_SpriteSheet_Generator",
            "prompt_tokens": tokens,
            "completion_tokens": float(num_steps * 4),
            "total_tokens": tokens + float(num_steps * 4),
            "total_duration_ms": total_duration_ms,
        }, timeout=2)
    except: pass

    return {"status": "success", "url": f"/images/{filename}", "duration_ms": total_duration_ms }
