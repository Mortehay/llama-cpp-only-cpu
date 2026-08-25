import os
import io
import time
import json
import requests
import psycopg2
import uuid
import random
from urllib.parse import urlparse
from PIL import Image
from contextlib import asynccontextmanager
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from migrations import run_migrations
from celery.result import AsyncResult
from tasks import (celery_app, generate_core_task, generate_spritesheet_task,
                   remove_background, set_cancel_flag, describe_device,
                   read_device_snapshot,
                   warm_model_task, edit_image_task, EDIT_LORAS,
                   EDIT_BASE, EDIT_ENABLED, EDIT_UNAVAILABLE_REASON)
from core_models import roster as core_model_roster, unavailable_reason
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

# Automatic1111-compatible routes, so something2's admin can register this
# service as a stock A1111 provider with no code changes on its side.
from a1111 import router as a1111_router
app.include_router(a1111_router)

# Async sheet jobs. The A1111 facade above stays for single-image txt2img,
# which fits in one request; a full character is ~2 hours and cannot.
from jobs import router as jobs_router
app.include_router(jobs_router)

# API keys. Replaces the single shared token that was a no-op when unset, so
# "is my API open?" has an answer the UI can show.
from auth import router as auth_router
app.include_router(auth_router)

# One list over generated images AND finished job sheets. Before this, the
# gallery read sprite_images only and 13 finished sheets were invisible.
from assets import router as assets_router
app.include_router(assets_router)

# Reference examples and the style profiles measured from them. A tile upload
# is how the camera angle stops being a guess.
from references import router as references_router
app.include_router(references_router)

# LoRA training, queued through the same worker as generation - one GPU means
# a training run and a sheet build cannot overlap.
from training import router as training_router
app.include_router(training_router)

# Isometric ground tiles. A separate spec from sheets - no actions, no frames -
# but the same queue and the same GET /api/jobs/{id} polling contract.
from tiles import router as tiles_router
app.include_router(tiles_router)

# Ensure images directory exists
IMAGES_DIR = "/app/images"
os.makedirs(IMAGES_DIR, exist_ok=True)

# Templates and Static files setup
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

class NoStoreStaticFiles(StaticFiles):
    """Serve generated images without letting the browser cache them.

    The gallery was rendering broken thumbnails with ERR_CACHE_READ_FAILURE in
    the console: the sheets had been cached, this mount answered the reload with
    304 Not Modified, and reading the (unreadable) disk-cache entry then failed -
    which leaves the browser with no body *and* no network fallback, so the <img>
    simply breaks. no-store keeps these responses out of the disk cache entirely,
    the one setting where that failure cannot happen. Always sending the body is
    cheap here: the files are local and the gallery lazy-loads its thumbnails.
    """

    def file_response(self, full_path, stat_result, scope, status_code: int = 200):
        # Deliberately not calling super(): its conditional-request check is what
        # produces the 304 this class exists to avoid.
        response = FileResponse(full_path, status_code=status_code, stat_result=stat_result)
        response.headers["cache-control"] = "no-store, max-age=0"
        return response


# Mount static files for serving saved images
app.mount("/images", NoStoreStaticFiles(directory=IMAGES_DIR), name="images")

def _db_target():
    """Host:port/name from DB_URL, without the credentials — safe to log."""
    if not DB_URL:
        return "<unset>"
    parsed = urlparse(DB_URL)
    return f"{parsed.hostname}:{parsed.port}{parsed.path}"


def get_db():
    if not DB_URL:
        logger.error("DB_URL is not set — gallery and task history are unavailable.")
        return None
    try:
        return psycopg2.connect(DB_URL)
    except Exception as e:
        # The callers degrade to an empty gallery, which is indistinguishable from
        # "no sprites yet" in the UI — so name the target the connection failed to.
        logger.error(f"DB connection to {_db_target()} failed: {e}")
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

# The React build, emitted by scripts/build-frontend.sh into static/app. Its
# assets are served by the /static mount above; only the entry document needs a
# route, because the SPA owns its own routing from there.
REACT_INDEX = os.path.join("static", "app", "index.html")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """The React UI, falling back to the legacy template if it is not built.

    The fallback is not politeness: `static/app` is a build artifact and is
    gitignored, so a fresh clone has no UI at all until someone runs the build.
    Serving a blank page there would look like a broken deployment.
    """
    if os.path.isfile(REACT_INDEX):
        return FileResponse(REACT_INDEX, media_type="text/html",
                            headers={"cache-control": "no-store"})
    logger.warning("static/app/index.html missing - serving the legacy UI. "
                   "Run scripts/build-frontend.sh to build the React app.")
    return await legacy_index(request)


@app.get("/legacy", response_class=HTMLResponse)
async def legacy_index(request: Request):
    """The pre-React UI.

    Feature parity was reached on 2026-08-25: React now covers core generation,
    spritesheets, references, tiles, training, gallery, settings, crop, edit,
    task retry/delete and model warm. This is no longer a feature fallback.

    It stays for one reason only - `static/app` is a build artefact and is
    gitignored, so a fresh clone has no React UI until someone runs
    scripts/build-frontend.sh. `/` falls back here rather than serving a blank
    page that looks like a broken deployment. Safe to delete once the build is
    part of the image.
    """
    # The core-model dropdown is rendered from the roster, not hardcoded in the
    # template, so an archived checkpoint is shown as unselectable rather than
    # offered and then failed on by the worker. Both containers mount the same
    # /models cache, so this is a stat() here, not a Celery round-trip.
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"active_page": "gen", "core_models": core_model_roster()}
    )

@app.get("/gallery", response_class=HTMLResponse)
async def gallery(request: Request):
    rows = fetch_gallery_rows()
    return templates.TemplateResponse(
        request=request, 
        name="gallery.html", 
        context={"rows": rows, "active_page": "gallery"}
    )

@app.post("/api/warm")
def warm_model(model: str = Form("stabilityai/sdxl-turbo")):
    """Queue a model download+load. Returns immediately with a task id.

    Non-blocking on purpose: warming an uncached checkpoint can take far longer
    than any sane HTTP timeout, which is the whole reason this endpoint exists.
    Poll /api/task-status/{task_id} for completion.
    """
    task = warm_model_task.delay(model)
    return JSONResponse({"status": "queued", "task_id": task.id, "model": model})


@app.get("/api/compute-info")
def compute_info():
    """Report the worker's real compute device, for the diagnostics panel.

    Deliberately round-trips through Celery rather than reading DEVICE here:
    this process is pinned to COMPUTE_DEVICE=cpu so it never competes for VRAM,
    so a local readout would always report "cpu" and mislead.

    A timeout is NOT a dead worker. The worker runs --pool=solo at concurrency
    1, so for the whole duration of a generation it consumes nothing from the
    broker and this round-trip cannot possibly return. Reporting that as 503
    "Worker unreachable" meant the panel called the worker dead every time it
    was doing its job, and filled the browser console with failed requests on
    every sprite. Fall back to the snapshot the worker leaves in Redis and say
    "busy"; only a missing snapshot means genuinely unreachable.
    """
    try:
        # expires: a poll that has already timed out client-side is of no use
        # when it finally reaches the front of the queue. Without this, a long
        # generation accumulates one stale describe_device per 15s and runs the
        # lot in a burst the moment it finishes.
        return describe_device.apply_async(expires=12).get(timeout=4)
    except Exception as e:
        cached = read_device_snapshot()
        if cached:
            age = max(0, int(time.time() - cached.get("snapshot_at", 0)))
            cached.update({
                "stale": True,
                "worker_state": "busy",
                "snapshot_age_s": age,
                "note": f"Worker is running a task and cannot answer; figures "
                        f"are {age}s old. A number in the minutes here means "
                        f"a long generation — or a worker that has stopped.",
            })
            return JSONResponse(cached)

        return JSONResponse(
            {"error": f"Worker did not respond: {e}", "device": "unknown",
             "worker_state": "unreachable"},
            status_code=503,
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


@app.get("/api/core-models")
def core_models():
    """The step-1 model roster with live availability.

    Lets the page re-check after a restore without a container restart, and
    gives any other client the same answer the dropdown was rendered from.
    """
    return JSONResponse({"models": core_model_roster()})


@app.post("/api/generate_core")
def generate_core(prompt: str = Form(...), llm_name: str = Form("stabilityai/sdxl-turbo")):
    # Refuse a model that is not on disk instead of queueing work that cannot
    # succeed. Offline, a missing checkpoint is not a transient failure: the
    # worker would log a load error, write "Model failed to load on worker" to
    # the task row, and leave a dead entry in the queue panel that invites a
    # Retry which fails identically. Say why here, where the operator is.
    reason = unavailable_reason(llm_name)
    if reason:
        logger.warning(f"Refusing generate_core for '{llm_name}': {reason}")
        raise HTTPException(status_code=409, detail=reason)

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

@app.get("/api/edit-capabilities")
def edit_capabilities():
    """What the editing model can be asked to do.

    Each capability is a LoRA on one shared NF4 base, so adding another costs
    ~0.5GB rather than another multi-GB model. `null` means the base model with
    no adapter, which still follows general instructions.
    """
    return JSONResponse({
        "model": "FLUX.1-Kontext-dev (NF4)",
        "capabilities": [None] + sorted(EDIT_LORAS.keys()),
        "note": "Editing is instruction-driven, unlike img2img: say what to "
                "change ('remove the shield', 'show this character from the "
                "side'), not how far to deviate. No capability LoRAs are "
                "needed - Kontext does these from the prompt. One task at a "
                "time: the NF4 transformer takes most of the card, so this "
                "evicts any generation pipeline.",
        "source_max_side": 512,
        "guidance_range": [2.5, 4.0],
    })


@app.post("/api/edit")
def edit_image(source: str = Form(...), instruction: str = Form(...),
               capability: str = Form(None), steps: int = Form(20),
               cfg_scale: float = Form(4.0), seed: int = Form(-1)):
    """Apply a natural-language edit to an image already in IMAGES_DIR.

    `source` is a filename, not a path: joining a caller-supplied path would let
    any file on the worker be opened. Only the images directory is reachable.
    """
    if not EDIT_ENABLED:
        # 503, not 500: the service is fine, this capability is not
        # available on this hardware. Refuse before queueing, because
        # the task would OOM-kill the worker and take generation with it.
        return JSONResponse({"error": EDIT_UNAVAILABLE_REASON,
                             "enable_with": "EDIT_ENABLED=1"}, status_code=503)

    name = os.path.basename(source)
    path = os.path.join("/app/images", name)
    if not os.path.exists(path):
        return JSONResponse({"error": f"No such image: {name}"}, status_code=404)
    if capability and capability not in EDIT_LORAS:
        return JSONResponse(
            {"error": f"Unknown capability '{capability}'",
             "available": sorted(EDIT_LORAS.keys())}, status_code=400)

    task = edit_image_task.delay(path, instruction, capability, steps, cfg_scale, seed)
    conn = get_db()
    if conn:
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO sprite_images (prompt, task_id, progress_msg, image_type, llm_name, step_number) VALUES (%s, %s, %s, %s, %s, %s)",
                        # Record the model that actually runs. This said
                        # "qwen-image-edit-2511/..." long after the editor moved
                        # to FLUX Kontext NF4, so the queue and the gallery
                        # attributed every FLUX failure to a model that is not
                        # installed — which is a bad place to start debugging.
                        (instruction, task.id, "Waiting in queue...", "edit",
                         f"{EDIT_BASE}/{capability or 'base'}", 1)
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
                    # Reads the retryable types off the front, so an "edit" row
                    # lands here. It used to answer by putting `task.id` in the
                    # payload — a name that is only bound inside the two
                    # branches above — so this line raised NameError and the
                    # caller got a 500 with no message. Unreachable while Retry
                    # was only offered on completed tasks; not any more.
                    #
                    # 400, and honest about why: an edit is defined by its
                    # SOURCE IMAGE, and the row stores only the instruction as
                    # its prompt. There is nothing here to re-run.
                    raise HTTPException(
                        status_code=400,
                        detail=f"Tasks of type '{image_type}' cannot be retried "
                               f"from history. Re-submit it against the source image.",
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