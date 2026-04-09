import os
import io
import time
import json
import requests
import psycopg2
from contextlib import asynccontextmanager
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from migrations import run_migrations
from celery.result import AsyncResult
from tasks import celery_app, generate_sprite_task, generate_core_task, generate_sheet_task

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

@app.post("/api/generate")
def generate_sprite(prompt: str = Form(...), llm_name: str = Form("stabilityai/sdxl-turbo")):
    # Fallback to older generation functionality if accessed directly
    task = generate_sprite_task.delay(prompt, llm_name)
    conn = get_db()
    if conn:
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO sprite_images (prompt, task_id, progress_msg, image_type, llm_name) VALUES (%s, %s, %s, %s, %s)",
                        (prompt, task.id, "Waiting in queue...", "spritesheet", llm_name)
                    )
        except Exception as e: print(f"Record error: {e}")
        finally: conn.close()
    return JSONResponse({"status": "queued", "task_id": task.id})

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
def generate_sheet(parent_id: int = Form(...), actions: str = Form(...), llm_name: str = Form("stabilityai/sdxl-turbo")):
    actions_list = json.loads(actions)
    task = generate_sheet_task.delay(parent_id, actions_list, llm_name)
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
                cur.execute("SELECT file_path, components, task_id FROM sprite_images WHERE id = %s", (id,))
                row = cur.fetchone()
                if row:
                    filepath = row[0]
                    comps = row[1]
                    task_id_to_revoke = row[2]
                    
                    if task_id_to_revoke:
                        try:
                            celery_app.control.revoke(task_id_to_revoke, terminate=True)
                        except Exception as revoke_e:
                            print(f"Revoke warning: {revoke_e}")

                    if filepath and os.path.exists(filepath):
                        pass # Soft-delete protocol: retain file on disk
                    if comps and isinstance(comps, list):
                        for c in comps:
                            c_path = c.replace("/images/", IMAGES_DIR + "/")
                            if os.path.exists(c_path): pass
                cur.execute("UPDATE sprite_images SET deleted = true WHERE id = %s", (id,))
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
                    task = generate_sheet_task.delay(parent_id, requested_actions, llm_actual)
                    cur.execute(
                        "INSERT INTO sprite_images (prompt, task_id, progress_msg, image_type, parent_id, requested_actions, llm_name, step_number) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                        (prompt, task.id, "Waiting in queue...", "spritesheet", parent_id, json.dumps(requested_actions), llm_actual, step_number)
                    )
                else:
                    task = generate_sprite_task.delay(prompt, llm_actual)
                    cur.execute(
                        "INSERT INTO sprite_images (prompt, task_id, progress_msg, image_type, llm_name) VALUES (%s, %s, %s, %s, %s)",
                        (prompt, task.id, "Waiting in queue...", "spritesheet", llm_actual)
                    )
                    
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