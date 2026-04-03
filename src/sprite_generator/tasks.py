import os
import io
import time
import uuid
import torch
import requests
import psycopg2
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
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
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
    negative = "blurry, deformed, extra limbs, cropped, low quality, watermark, text, noise"

    clean_prompt = prompt
    for t, _ in DIRECTIONS:
        clean_prompt = clean_prompt.replace(t, "").strip().lstrip(",").strip()

    strips = []
    start_time = time.time()
    num_steps = 25

    try:
        for i, (trigger, label) in enumerate(DIRECTIONS):
            full_prompt = f"{trigger}, {clean_prompt}"
            print(f"Generating {label}: {full_prompt}")
            
            # Update DB with new direction
            update_task_record(task_id, progress_pct=int((i/4)*100), progress_msg=f"Pass {i+1}/4: {label}")

            # Define progress callback
            def progress_callback(step, timestep, latents):
                step_pct = (step / num_steps)
                total_pct = int(((i / 4) + (step_pct / 4)) * 100)
                # We update every 3 steps to avoid DB spam
                if step % 3 == 0:
                    update_task_record(task_id, progress_pct=total_pct, progress_msg=f"Pass {i+1}/4 ({label}): {int(step_pct*100)}%")
                    # Also update Celery state for fallback polling
                    self.update_state(state="PROGRESS", meta={"pct": total_pct, "msg": label})

            img = p(
                full_prompt,
                negative_prompt=negative,
                height=512,
                width=512,
                num_inference_steps=num_steps,
                guidance_scale=7.5,
                callback=progress_callback,
                callback_steps=1
            ).images[0]
            strips.append(img)
            
    except Exception as e:
        update_task_record(task_id, error_msg=f"Generation failed: {str(e)}", progress_pct=0)
        return {"error": str(e)}

    end_time = time.time()
    total_duration_ms = (end_time - start_time) * 1000

    # UI cleanup: stitching state
    update_task_record(task_id, progress_pct=90, progress_msg="Finalizing: Stitching...")

    # Stitch vertically
    total_height = sum(img.height for img in strips)
    master = Image.new("RGB", (strips[0].width, total_height))
    y = 0
    for img in strips:
        master.paste(img, (0, y))
        y += img.height

    # Chromakey background removal
    print("Removing background (chromakey)...")
    generation_error = None
    try:
        master = master.convert("RGBA")
        bg_r, bg_g, bg_b, *_ = master.getpixel((0, 0))
        new_data = []
        for r, g, b, a in master.getdata():
            if abs(r - bg_r) < 18 and abs(g - bg_g) < 18 and abs(b - bg_b) < 18:
                new_data.append((0, 0, 0, 0))
            else:
                new_data.append((r, g, b, a))
        master.putdata(new_data)
    except Exception as e:
        print(f"Background removal failed: {e}")
        generation_error = f"Background removal failed: {str(e)}"

    # Save to disk
    filename = f"sprite_{uuid.uuid4().hex[:12]}.png"
    filepath = os.path.join(IMAGES_DIR, filename)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    master.save(filepath, format="PNG")

    # Final DB Update
    update_task_record(task_id, file_path=filepath, duration_ms=total_duration_ms, 
                       error_msg=generation_error, progress_pct=100, progress_msg="Complete")

    # Log stats to collector
    try:
        tokens = float(len(clean_prompt.split()) * 10)
        requests.post(
            "http://stats_collector:8000/v1/internal/log_stats",
            json={
                "model_name": "Onodofthenorth/SD_PixelArt_SpriteSheet_Generator",
                "prompt_tokens": tokens,
                "completion_tokens": float(num_steps * 4),
                "total_tokens": tokens + float(num_steps * 4),
                "tokens_per_second": float(num_steps * 4) / max(end_time - start_time, 0.001),
                "prompt_eval_ms": 0.0,
                "total_duration_ms": total_duration_ms,
            },
            timeout=2,
        )
    except Exception as e:
        print(f"Could not log stats: {e}")

    return {
        "status": "success",
        "url": f"/images/{filename}",
        "filename": filename,
        "duration_ms": total_duration_ms
    }
