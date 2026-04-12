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
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DPMSolverMultistepScheduler
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
            "Aakash010/MedGemma_FineTuned": "medgemma-Q4_K_M.gguf",
            "rafacost/DreamOmni2-7.6B-GGUF": "DreamOmni2-Vlm-Model-7.6B-Q4_K_M.gguf"
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
            {"role": "user", "content": [{"type": "text", "text": f"Enhance this sprite generation prompt: {prompt}"}]}
        ]

        # If a reference image is provided in the generator (Step 2 logic usually passes it as part of a different flow, but here we can check kwargs)
        # Note: In our current get_pipeline architecture, 'prompt' is the primary string.
        # If we wanted to pass an image, we'd need to modify the call site or extract it if embedded.
        
        try:
            # Call Local VLM
            resp = requests.post(
                self.endpoint, 
                json={
                    "model": model_file,
                    "messages": messages,
                    "max_tokens": 150
                },
                timeout=30
            )
            resp.raise_for_status()
            vlm_text = resp.json()['choices'][0]['message']['content']
            logger.info(f"VLM Enhanced Prompt: {vlm_text}")
            
            # 2. Use Fallback SD Pipeline with Enhanced Prompt
            if self.fallback_pipeline:
                logger.info("Using fallback SD pipeline with VLM guidance...")
                # We combine original prompt words with VLM description
                new_prompt = f"{prompt}, {vlm_text}"
                return self.fallback_pipeline(new_prompt, **kwargs)
            
            # 3. Last Resort: Placeholder if no fallback
            img = Image.new('RGB', (512, 512), color=(30, 30, 40))
            d = ImageDraw.Draw(img)
            d.text((20, 20), f"VLM Guide: {vlm_text[:50]}...", fill=(200, 200, 200))
            return PipelineOutput(images=[img])

        except Exception as e:
            logger.error(f"Proxy call failed: {e}. Falling back to default if possible.")
            if self.fallback_pipeline:
                return self.fallback_pipeline(prompt, **kwargs)
            raise e

    def enhance_animation(self, action_label, base_prompt):
        """Specifically useful for animation frames. Returns 4 descriptions."""
        logger.info(f"VLM: Requesting 4-frame animation breakdown for '{action_label}'")
        model_file = self.model_file_map.get(self.model_name, self.model_name)
        
        prompt_text = (
            f"Describe 4 individual animation frames for a character doing: {action_label}. "
            f"The character is: {base_prompt}. "
            "Respond in exactly 4 lines, one for each frame. Each line must be a concise visual description."
        )
        
        messages = [
            {"role": "system", "content": "You are a professional pixel art animator. Provide exactly 4 lines of visual descriptions for animation frames."},
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
            resp.raise_for_status()
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

def get_pipeline(llm_name: str = "stabilityai/sdxl-turbo"):
    if llm_name == "models--stabilityai--sdxl-turbo":
        llm_name = "stabilityai/sdxl-turbo"
    global pipes
    if llm_name in pipes:
        return pipes[llm_name]
    
    if "gguf" in llm_name.lower() or "medgemma" in llm_name.lower() or "dreamomni" in llm_name.lower():
        logger.info(f"Using LLMProxyPipeline for '{llm_name}'")
        # For GGUF/VLM models, we use SDXL-Turbo as the fallback generator
        fallback = get_pipeline("stabilityai/sdxl-turbo")
        pipes[llm_name] = LLMProxyPipeline(llm_name, fallback_pipeline=fallback)
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
    negative = "multiple characters, two characters, split screen, collage, grid, set, blurry, deformed, extra limbs, cropped, low quality, watermark, text, noise, messy pixels, artifacting, gradient, shadows on background"

    clean_prompt = prompt
    for t, _ in DIRECTIONS:
        clean_prompt = clean_prompt.replace(t, "").strip().lstrip(",").strip()
    
    full_prompt_base = f"single standalone {clean_prompt}, one centered character, no duplicates, flat solid white background, high quality pixel art, 16-bit, sharp focus" if "background" not in clean_prompt.lower() else f"single standalone {clean_prompt}, one centered character, no duplicates, high quality pixel art, sharp focus"

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
    negative = "multiple characters, two characters, split screen, collage, grid, set, blurry, deformed, extra limbs, cropped, low quality, watermark, text, noise, messy pixels, artifacting, gradient, shadows on background"

    clean_prompt = prompt.replace("PixelartFSS", "").strip().lstrip(",").strip()
    
    # Strictly aligned prefix: "PixelartFSS, idle front,"
    full_prompt = f"PixelartFSS, idle front, single standalone {clean_prompt}, one centered character, no duplicates, flat solid transparent background, high quality pixel art, 16-bit, sharp focus" if "background" not in clean_prompt.lower() else f"PixelartFSS, idle front, single standalone {clean_prompt}, one centered character, no duplicates, high quality pixel art, sharp focus"

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
    p = get_pipeline(llm_name)
    
    clean_prompt = parent_prompt.replace("PixelartFSS", "").strip().lstrip(",").strip()
    base_prompt = f"{clean_prompt}, flat solid white background, high quality pixel art, 16-bit, sharp focus" if "background" not in clean_prompt.lower() else f"{clean_prompt}, high quality pixel art, sharp focus"
    negative = "multiple characters, two characters, split screen, collage, grid, set, blurry, deformed, extra limbs, cropped, low quality, watermark, text, noise, messy pixels, artifacting, gradient, shadows on background"
    
    # Dynamic parameters based on model type
    is_turbo = "turbo" in llm_name.lower()
    num_steps = 4 if is_turbo else 35
    guidance = 1.0 if is_turbo else 9.0
    is_vlm = isinstance(p, LLMProxyPipeline)

    # 1. Get 4 Frame descriptions
    frame_descriptions = []
    if is_vlm:
        frame_descriptions = p.enhance_animation(action, base_prompt)
    else:
        action_lower = action.lower()
        trigger = ""
        if "move right" in action_lower: trigger = "walk right"
        elif "move left" in action_lower: trigger = "walk left"
        elif "move down" in action_lower: trigger = "walk front"
        elif "move up" in action_lower: trigger = "walk back"
        elif "idle" in action_lower: trigger = "idle standing"
        elif "attack" in action_lower: trigger = "fast strike attack"
        elif "got damage" in action_lower: trigger = "taking damage"
        elif "burning" in action_lower: trigger = "in flames burning"
        else: trigger = action
        
        frame_descriptions = [
            f"Frame 1 of {trigger} animation, {base_prompt}",
            f"Frame 2 of {trigger} animation, movement sequence, {base_prompt}",
            f"Frame 3 of {trigger} animation, movement sequence, {base_prompt}",
            f"Frame 4 of {trigger} animation, finish pose, {base_prompt}"
        ]

    # 2. Generate 4 frames
    action_frames = []
    for f_idx, frame_prompt in enumerate(frame_descriptions):
        generator = torch.Generator("cpu").manual_seed(parent_seed + f_idx)
        
        # Note: In a distributed system, we don't update the MAIN task progress here 
        # unless we want to do complex accumulation. For now, we log to stdout.
        logger.info(f"Worker {self.request.id} Generating Frame {f_idx+1}/4 for '{action}'")
        
        # Define per-frame progress callback for distributed awareness
        def frame_progress_callback(step, timestep, latents):
            if step % 1 == 0:
                frame_pct = (step / num_steps)
                # Global Progress = (Actions Done / Total) + (This Action Progress / Total)
                global_pct = int(((action_index / total_actions) + (frame_pct / 4 / total_actions)) * 100)
                # Ensure we don't hit 100% until the finalizer finishes
                safe_pct = min(global_pct, 95)
                update_task_record(main_task_id, progress_pct=safe_pct, progress_msg=f"{action}: {int(frame_pct*100)}%")
                self.update_state(state="PROGRESS", meta={"pct": safe_pct, "msg": f"{action} F{f_idx+1}"})

        img = p(
            prompt=frame_prompt,
            negative_prompt=negative,
            height=512,
            width=512,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            generator=generator,
            callback=frame_progress_callback,
            callback_steps=1
        ).images[0]
        
        img_transparent = remove_background(img)
        
        # Resize to requested frame dimensions
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
