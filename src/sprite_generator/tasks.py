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
            settings = {}
            for key, value in rows:
                if isinstance(value, dict):
                    settings[key] = value
                else:
                    try:
                        settings[key] = json.loads(value)
                    except ValueError:
                        settings[key] = value
            return settings
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
    pass

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
    pass

@celery_app.task(name="tasks.generate_action_strip_task", bind=True)
def generate_action_strip_task(self, main_task_id: str, action: str, action_index: int, total_actions: int, parent_id: int, parent_prompt: str, parent_seed: int, llm_name: str, frame_width: int = 128, frame_height: int = 128):
    pass

@celery_app.task(name="tasks.finalize_sheet_task", bind=True)
def finalize_sheet_task(self, results, main_task_id: str, parent_id: int, actions_order: list, parent_seed: int, llm_name: str):
    pass
