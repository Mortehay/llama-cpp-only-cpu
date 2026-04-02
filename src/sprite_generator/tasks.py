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

def update_task_record(task_id: str, file_path: str = None, duration_ms: float = 0, error_msg: str = None):
    """Update an existing record (identified by task_id) with results."""
    conn = get_db()
    if not conn:
        return
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE sprite_images 
                    SET file_path = %s, duration_ms = %s, error = %s 
                    WHERE task_id = %s
                    """,
                    (file_path, duration_ms, error_msg, task_id),
                )
    except Exception as e:
        print(f"Could not update task record {task_id}: {e}")
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

    try:
        for trigger, label in DIRECTIONS:
            full_prompt = f"{trigger}, {clean_prompt}"
            print(f"Generating {label}: {full_prompt}")
            img = p(
                full_prompt,
                negative_prompt=negative,
                height=512,
                width=512,
                num_inference_steps=25,
                guidance_scale=7.5,
            ).images[0]
            strips.append(img)
    except Exception as e:
        update_task_record(task_id, error_msg=f"Generation failed: {str(e)}")
        return {"error": str(e)}

    end_time = time.time()
    total_duration_ms = (end_time - start_time) * 1000

    # Stitch vertically
    print("Stitching 4 directions...")
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
    print(f"Saved image: {filepath}")

    # Update record in DB
    update_task_record(task_id, file_path=filepath, duration_ms=total_duration_ms, error_msg=generation_error)

    # Log stats to collector
    try:
        tokens = float(len(clean_prompt.split()) * 10)
        requests.post(
            "http://stats_collector:8000/v1/internal/log_stats",
            json={
                "model_name": "Onodofthenorth/SD_PixelArt_SpriteSheet_Generator",
                "prompt_tokens": tokens,
                "completion_tokens": float(25 * 4),
                "total_tokens": tokens + float(25 * 4),
                "tokens_per_second": float(25 * 4) / max(end_time - start_time, 0.001),
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
