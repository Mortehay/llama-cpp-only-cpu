import os
import io
import time
import json
import requests
import psycopg2
import uuid
import random
from PIL import Image
from contextlib import asynccontextmanager
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from migrations import run_migrations
from celery.result import AsyncResult
from tasks import celery_app, generate_core_task, generate_spritesheet_task, remove_background, set_cancel_flag
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_URL = os.environ.get("DB_URL")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run DB migrations before accepting traffic."""
    run_migrations(DB_URL)
    yield

app = FastAPI(lifespan=lifespan)

# Ensure images directory exists
IMAGES_DIR = "/app/images"
os.makedirs(IMAGES_DIR, exist_ok=True)

# Templates and Static files setup
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount static files for serving saved images
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

def get_db():
    if not DB_URL:
        return None
    try:
        return psycopg2.connect(DB_URL)
    except Exception as e:
        print(f"DB connection failed: {e}")
        return None

def fetch_gallery_rows(limit=None):
    conn = get_db()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            query = """
                SELECT
                    id,
                    timestamp,
                    prompt,
                    file_path,
                    duration_ms,
                    COALESCE(error, '') as error,
                    task_id,
                    progress_pct,
                    progress_msg,
                    attempt_number,
                    image_type, parent_id, components, requested_actions,
                    COALESCE(llm_name, 'Unknown') as llm_name,
                    COALESCE(step_number, 0) as step_number
                FROM (
                    SELECT
                        id,
                        timestamp,
                        prompt,
                        file_path,
                        duration_ms,
                        error,
                        task_id,
                        progress_pct,
                        progress_msg,
                        ROW_NUMBER() OVER (PARTITION BY prompt ORDER BY timestamp) AS attempt_number,
                        image_type, parent_id, components, requested_actions,
                        llm_name, step_number
                    FROM sprite_images WHERE deleted = false
                ) AS sub
                ORDER BY timestamp DESC
            """
            if limit:
                query += f" LIMIT {limit}"
            cur.execute(query)
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception as e:
        print(f"Could not fetch gallery: {e}")
        return []
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"active_page": "gen"}
    )

@app.get("/gallery", response_class=HTMLResponse)
async def gallery(request: Request):
    rows = fetch_gallery_rows()
    return templates.TemplateResponse(
        request=request, 
        name="gallery.html", 
        context={"rows": rows, "active_page": "gallery"}
    )

@app.get("/api/settings")
def get_settings():
    conn = get_db()
    if not conn: return {"compute_mode": "cpu"}
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT key, value FROM app_settings")
            rows = cur.fetchall()
            return {row[0]: row[1] for row in rows}
    except Exception as e:
        print(f"Error fetching settings: {e}")
        return {"compute_mode": "cpu"}
    finally: conn.close()

@app.post("/api/settings/{key}")
def update_setting(key: str, request: Request):
    import json
    try:
        # We handle raw JSON input for flexible settings
        body = time.sleep(0.01) # ensure a bit of delay if needed
        # Actually, let's use a simpler approach for now
        pass
    except: pass
    return {"status": "error"}

@app.post("/api/settings")
async def save_settings(request: Request):
    data = await request.json()
    conn = get_db()
    if not conn: return JSONResponse({"status": "error"}, status_code=500)
    try:
        with conn:
            with conn.cursor() as cur:
                for k, v in data.items():
                    cur.execute(
                        "INSERT INTO app_settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = %s, updated_at = CURRENT_TIMESTAMP",
                        (k, json.dumps(v), json.dumps(v))
                    )
        return {"status": "success"}
    except Exception as e:
        print(f"Error saving settings: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    finally: conn.close()


@app.post("/api/generate_core")
def generate_core(prompt: str = Form(...), llm_name: str = Form("stabilityai/sdxl-turbo")):
    task = generate_core_task.delay(prompt, llm_name)
    conn = get_db()
    if conn:
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO sprite_images (prompt, task_id, progress_msg, image_type, llm_name, step_number) VALUES (%s, %s, %s, %s, %s, %s)",
                        (prompt, task.id, "Waiting in queue...", "core", llm_name, 1)
                    )
        except Exception as e: print(f"Record error: {e}")
        finally: conn.close()
    return JSONResponse({"status": "queued", "task_id": task.id})

@app.post("/api/generate_sheet")
def generate_sheet(parent_id: int = Form(...), actions: str = Form(...), 
                   llm_name: str = Form("stabilityai/sdxl-turbo"),
                   width: int = Form(128), height: int = Form(128),
                   motion_steps: int = Form(4)):
    actions_list = json.loads(actions)
    task = generate_spritesheet_task.delay(parent_id, actions_list, llm_name, width, height, motion_steps)
    conn = get_db()
    if conn:
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO sprite_images (prompt, task_id, progress_msg, image_type, parent_id, requested_actions, llm_name, step_number) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                        (str(actions_list), task.id, "Waiting in queue...", "spritesheet", parent_id, json.dumps(actions_list), llm_name, 2)
                    )
        except Exception as e: print(f"Record error: {e}")
        finally: conn.close()
    return JSONResponse({"status": "queued", "task_id": task.id})
    
@app.post("/api/crop")
async def crop_sprite(request: Request):
    try:
        data = await request.json()
        source_id = data.get('source_id')
        x = int(data.get('x', 0))
        y = int(data.get('y', 0))
        w = int(data.get('w', 0))
        h = int(data.get('h', 0))
        
        if not source_id or w == 0 or h == 0:
            raise HTTPException(status_code=400, detail="Invalid crop data")
            
        conn = get_db()
        if not conn: raise HTTPException(status_code=500, detail="DB Connection failed")
        
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT file_path, prompt, llm_name FROM sprite_images WHERE id = %s", (source_id,))
                    row = cur.fetchone()
                    if not row: raise HTTPException(status_code=404, detail="Source image not found")
                    
                    orig_path, prompt, llm_name = row
                    if not os.path.exists(orig_path):
                        raise HTTPException(status_code=404, detail="Original file missing on disk")
                        
                    # Perform Crop
                    with Image.open(orig_path) as img:
                        # PIL crop uses (left, top, right, bottom)
                        cropped = img.crop((x, y, x + w, y + h))
                        cropped = remove_background(cropped)
                        
                        filename = f"crop_{uuid.uuid4().hex[:12]}.png"
                        filepath = os.path.join(IMAGES_DIR, filename)
                        cropped.save(filepath, "PNG")
                        
                        # Save new core record with source link
                        cur.execute(
                            "INSERT INTO sprite_images (prompt, file_path, image_type, parent_id, cropped_from, progress_pct, progress_msg, llm_name, duration_ms) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
                            (f"Cropped: {prompt}", filepath, "core", source_id, source_id, 100, "Cropped & Saved", llm_name, 0)
                        )
                        new_id = cur.fetchone()[0]
                        return {"status": "success", "id": new_id, "url": f"/images/{filename}"}
        finally: conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/cores")
def get_cores():
    conn = get_db()
    if not conn: return []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, file_path, prompt 
                FROM sprite_images 
                WHERE image_type='core' AND file_path IS NOT NULL AND deleted = false
                ORDER BY timestamp DESC LIMIT 24
            """)
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception as e:
        print(f"Error fetching cores: {e}")
        return []
    finally: conn.close()

@app.delete("/api/task/{id}")
def delete_task(id: int):
    conn = get_db()
    if not conn: raise HTTPException(status_code=500, detail="DB Connection failed")
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT file_path, components, task_id, sub_task_ids FROM sprite_images WHERE id = %s", (id,))
                row = cur.fetchone()
                if row:
                    filepath = row[0]
                    comps = row[1]
                    task_id_to_revoke = row[2]
                    sub_task_ids_json = row[3] if len(row) > 3 else None
                     # 1. Set cooperative cancel flags (stops inference callback loops)
                    if task_id_to_revoke:
                        try:
                            set_cancel_flag(task_id_to_revoke)
                        except Exception as e:
                            print(f"Error setting cancel flag for {task_id_to_revoke}: {e}")

                    if sub_task_ids_json:
                        try:
                            sub_ids = sub_task_ids_json if isinstance(sub_task_ids_json, list) else json.loads(sub_task_ids_json)
                            if isinstance(sub_ids, list):
                                for sid in sub_ids:
                                    try:
                                        set_cancel_flag(sid)
                                    except Exception as e:
                                        print(f"Error setting cancel flag for sub-task {sid}: {e}")
                        except Exception as parse_e:
                            print(f"Error parsing sub_task_ids: {parse_e}")

                    # 2. Also send SIGTERM via Celery revoke (belt + suspenders)
                    if sub_task_ids_json:
                        try:
                            sub_ids = sub_task_ids_json if isinstance(sub_task_ids_json, list) else json.loads(sub_task_ids_json)
                            if isinstance(sub_ids, list):
                                for sid in sub_ids:
                                    try:
                                        celery_app.control.revoke(sid, terminate=True)
                                    except Exception as e:
                                        print(f"Error revoking sub-task {sid}: {e}")
                        except Exception:
                            pass

                    # 3. Revoke primary task
                    if task_id_to_revoke:
                        try:
                            celery_app.control.revoke(task_id_to_revoke, terminate=True)
                        except Exception as revoke_e:
                            print(f"Error revoking main task {task_id_to_revoke}: {revoke_e}")

                # 4. GUARANTEE database flag update
                cur.execute("UPDATE sprite_images SET deleted = true WHERE id = %s", (id,))
                logger.info(f"Task record {id} marked as deleted in DB.")
        return {"status": "deleted"}
    finally: conn.close()

@app.post("/api/task/{id}/retry")
def retry_task(id: int):
    conn = get_db()
    if not conn: raise HTTPException(status_code=500, detail="DB Connection failed")
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT prompt, image_type, parent_id, requested_actions, llm_name, step_number FROM sprite_images WHERE id = %s", (id,))
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Task not found")
                
                prompt, image_type, parent_id, requested_actions, llm_name, step_number = row
                
                llm_actual = llm_name if llm_name and llm_name != 'Unknown' else "stabilityai/sdxl-turbo"
                
                if image_type == "core":
                    task = generate_core_task.delay(prompt, llm_actual)
                    cur.execute(
                        "INSERT INTO sprite_images (prompt, task_id, progress_msg, image_type, llm_name, step_number) VALUES (%s, %s, %s, %s, %s, %s)",
                        (prompt, task.id, "Waiting in queue...", "core", llm_actual, step_number)
                    )
                elif image_type == "spritesheet":
                    task = generate_spritesheet_task.delay(parent_id, requested_actions, llm_actual, 128, 128, 4)
                    cur.execute(
                        "INSERT INTO sprite_images (prompt, task_id, progress_msg, image_type, parent_id, requested_actions, llm_name, step_number) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                        (prompt, task.id, "Waiting in queue...", "spritesheet", parent_id, json.dumps(requested_actions), llm_actual, step_number)
                    )
                else:
                    return {"status": "error", "message": "Invalid image type", "task_id": task.id, "image_type": image_type}
                
                    
                return {"status": "queued", "task_id": task.id, "image_type": image_type}
    finally: conn.close()

@app.get("/api/task-status/{task_id}")
def get_task_status(task_id: str):
    res = AsyncResult(task_id, app=celery_app)
    result_data = res.result if res.ready() else None
    return {"task_id": task_id, "status": res.status, "result": result_data}

@app.get("/api/tasks/recent")
def recent_tasks():
    return fetch_gallery_rows(limit=12)