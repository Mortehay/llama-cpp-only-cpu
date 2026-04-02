import os
import io
import time
import requests
import psycopg2
from contextlib import asynccontextmanager
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from migrations import run_migrations
from celery.result import AsyncResult
from tasks import celery_app, generate_sprite_task

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
                    ROW_NUMBER() OVER (PARTITION BY prompt ORDER BY timestamp) AS attempt_number
                FROM sprite_images
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
# HTML Helpers
# ---------------------------------------------------------------------------

SHARED_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #0f0f17;
    --surface: #1a1a2e;
    --card: #212138;
    --border: #2e2e50;
    --accent: #7c6af7;
    --accent2: #a78bfa;
    --text: #e2e0f0;
    --muted: #8884a8;
    --success: #34d399;
    --warn: #fbbf24;
    --danger: #ef4444;
  }
  body {
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    padding: 0;
  }
  nav {
    display: flex; gap: 8px; align-items: center;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 12px 28px;
  }
  nav a {
    color: var(--muted); text-decoration: none;
    font-size: 14px; font-weight: 500;
    padding: 6px 14px; border-radius: 6px;
    transition: all .2s;
  }
  nav a:hover, nav a.active { background: var(--accent); color: #fff; }
  nav .brand { font-weight: 700; font-size: 16px; color: var(--accent2); margin-right: auto; }
  .tag { font-size: 10px; padding: 2px 6px; border-radius: 4px; font-weight: 700; text-transform: uppercase; }
  .tag-danger { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.2); }
  .tag-success { background: rgba(52, 211, 153, 0.15); color: #6ee7b7; border: 1px solid rgba(52, 211, 153, 0.2); }
  .tag-working { background: rgba(251, 191, 36, 0.15); color: #fcd34d; border: 1px solid rgba(251, 191, 36, 0.2); }
</style>
"""

def nav_bar(active: str) -> str:
    gen_cls = "active" if active == "gen" else ""
    gal_cls = "active" if active == "gallery" else ""
    return f"""
    <nav>
      <span class="brand">🎮 Sprite Generator</span>
      <a href="/" class="{gen_cls}">Generate</a>
      <a href="/gallery" class="{gal_cls}">Gallery</a>
    </nav>"""

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sprite Generator (CPU Queue)</title>
  {SHARED_CSS}
  <style>
    .page {{ max-width: 900px; margin: 48px auto; padding: 0 20px; display: grid; grid-template-columns: 1fr 320px; gap: 32px; }}
    @media (max-width: 850px) {{ .page {{ grid-template-columns: 1fr; }} }}
    h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 6px; }}
    .sub {{ color: var(--muted); font-size: 14px; margin-bottom: 32px; }}
    .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 24px; height: fit-content; }}
    label {{ font-size: 12px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; display: block; margin-bottom: 8px; }}
    textarea {{
      width: 100%; background: var(--surface); color: var(--text);
      border: 1px solid var(--border); border-radius: 8px;
      padding: 12px; font-size: 15px; font-family: inherit;
      resize: vertical; transition: border .2s;
    }}
    textarea:focus {{ outline: none; border-color: var(--accent); }}
    .btn {{
      display: block; width: 100%; margin-top: 20px;
      background: linear-gradient(135deg, var(--accent) 0%, #a855f7 100%);
      color: #fff; padding: 14px; border: none; border-radius: 8px;
      font-size: 15px; font-weight: 700; cursor: pointer; transition: opacity .2s;
    }}
    .btn:hover {{ opacity: .88; }}
    .btn:disabled {{ opacity: .4; cursor: not-allowed; }}
    #status {{ margin-top: 14px; font-size: 13px; color: var(--warn); text-align: center; }}
    
    .preview {{
      margin-top: 24px; min-height: 200px; border: 2px dashed var(--border);
      border-radius: 10px; display: flex; align-items: center; justify-content: center;
      background: repeating-conic-gradient(#1e1e30 0% 25%, #161626 0% 50%) 0 0 / 20px 20px;
    }}
    .preview img {{ max-width: 100%; border-radius: 6px; image-rendering: pixelated; }}
    .preview-placeholder {{ color: var(--muted); font-size: 13px; text-align: center; padding: 0 20px; }}

    /* Task List Sidebar */
    .task-list-title {{ font-size: 14px; font-weight: 700; color: var(--text); margin-bottom: 16px; display: flex; justify-content: space-between; }}
    .task-item {{ 
      background: var(--surface); border: 1px solid var(--border); border-radius: 8px; 
      padding: 12px; margin-bottom: 12px; font-size: 13px;
    }}
    .task-item .prompt-clip {{ color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 6px; font-weight: 500; }}
    .task-item .meta {{ display: flex; justify-content: space-between; align-items: center; font-size: 11px; color: var(--muted); }}
    .pulse {{ animation: pulse 2s infinite; }}
    @keyframes pulse {{ 0% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} 100% {{ opacity: 1; }} }}
  </style>
</head>
<body>
  {nav_bar("gen")}
  <div class="page">
    <div class="main-content">
      <h1>Pixel Art Sprite Generator</h1>
      <p class="sub">CPU-Optimized sequential queue. Background auto-removed.</p>
      <div class="card">
        <label for="prompt">Character description</label>
        <textarea id="prompt" rows="3" placeholder="Describe your character...">green zombie, tattered clothes, solid white background</textarea>
        <button class="btn" id="gen-btn" onclick="generate()">⚡ Queue Sprite Generation</button>
        <div id="status"></div>
        <div class="preview" id="result">
          <span class="preview-placeholder">Enter a prompt and start a task to see the magic here.</span>
        </div>
      </div>
    </div>

    <div class="sidebar">
       <div class="task-list-title"> 
         <span>Recent Tasks</span>
         <span style="font-size: 10px; font-weight: 400; opacity: 0.6;">(Updates Live)</span>
       </div>
       <div id="task-queue">
         <!-- Tasks will be injected here -->
         <p style="font-size: 12px; color: var(--muted); text-align: center; padding: 40px 0;">Loading queue...</p>
       </div>
    </div>
  </div>

  <script>
    let pollInterval = null;
    let queueInterval = null;

    // Initial load
    updateQueue();
    queueInterval = setInterval(updateQueue, 5000);

    async function updateQueue() {{
      try {{
        const res = await fetch('/api/tasks/recent');
        const tasks = await res.json();
        const queueDiv = document.getElementById('task-queue');
        
        if (tasks.length === 0) {{
          queueDiv.innerHTML = '<p style="font-size: 12px; color: var(--muted); text-align: center; padding: 40px 0;">No recent tasks.</p>';
          return;
        }}

        queueDiv.innerHTML = tasks.map(t => {{
          let statusTag = '';
          if (t.error) statusTag = '<span class="tag tag-danger">Failed</span>';
          else if (t.file_path) statusTag = '<span class="tag tag-success">Done</span>';
          else statusTag = '<span class="tag tag-working pulse">Working...</span>';

          return `
            <div class="task-item">
              <div class="prompt-clip" title="${{t.prompt.replace(/"/g, '&quot;')}}">${{t.prompt}}</div>
              <div class="meta">
                <span>${{statusTag}}</span>
                <span>${{t.timestamp.split('T')[1].split('.')[0]}}</span>
              </div>
            </div>
          `;
        }}).join('');
      }} catch (e) {{ console.error('Queue update failed', e); }}
    }}

    async function generate() {{
      const promptVal = document.getElementById('prompt').value.trim();
      if (!promptVal) return;
      const resultDiv = document.getElementById('result');
      const statusDiv = document.getElementById('status');
      const btn = document.getElementById('gen-btn');

      resultDiv.innerHTML = '<span class="preview-placeholder pulse">⏳ Task is being added to queue...</span>';
      statusDiv.innerText = 'Communicating with Redis...';
      btn.disabled = true;

      try {{
        const fd = new FormData();
        fd.append('prompt', promptVal);
        const req = await fetch('/api/generate', {{ method: 'POST', body: fd }});

        if (req.ok) {{
          const data = await req.json();
          const taskId = data.task_id;
          statusDiv.innerText = '✅ Task Queued! Tracking...';
          updateQueue();
          pollTaskStatus(taskId);
        }} else {{
          statusDiv.innerText = '❌ Error: ' + await req.text();
          btn.disabled = false;
        }}
      }} catch (e) {{
        statusDiv.innerText = '❌ Error: ' + e.message;
        btn.disabled = false;
      }}
    }}

    function pollTaskStatus(taskId) {{
      const statusDiv = document.getElementById('status');
      const resultDiv = document.getElementById('result');
      const btn = document.getElementById('gen-btn');

      if (pollInterval) clearInterval(pollInterval);
      pollInterval = setInterval(async () => {{
        try {{
          const res = await fetch('/api/task-status/' + taskId);
          const data = await res.json();

          if (data.status === 'SUCCESS') {{
            clearInterval(pollInterval);
            const result = data.result;
            if (result.error) {{
                statusDiv.innerText = '❌ Failed: ' + result.error;
                resultDiv.innerHTML = '<span class="preview-placeholder">Generation encountered an error. Check the task list for details.</span>';
            }} else {{
                resultDiv.innerHTML = `<img src="${{result.url}}" alt="Sprite Sheet" />`;
                statusDiv.innerText = `✅ Finished in ${{(result.duration_ms / 1000).toFixed(1)}}s`;
            }}
            btn.disabled = false;
            updateQueue();
          }} else if (data.status === 'FAILURE') {{
            clearInterval(pollInterval);
            statusDiv.innerText = '❌ Task system error.';
            btn.disabled = false;
            updateQueue();
          }} else if (data.status === 'STARTED') {{
            statusDiv.innerText = '🏃 Worker is crunching pixels now...';
          }} else {{
            statusDiv.innerText = '⌛ Waiting in queue (Sequential Mode)...';
          }}
        }} catch (e) {{ console.error(e); }}
      }}, 3000);
    }}
  </script>
</body>
</html>"""

@app.get("/gallery", response_class=HTMLResponse)
async def gallery():
    rows = fetch_gallery_rows()
    if not rows:
        cards_html = """<div class="empty"><span>🖼️</span><p>No sprites yet.</p></div>"""
    else:
        cards = []
        for row in rows:
            ts = row["timestamp"].strftime("%m-%d %H:%M") if row["timestamp"] else "—"
            dur = f"{row['duration_ms'] / 1000:.1f}s" if row["duration_ms"] else "—"
            url = f"/images/{os.path.basename(row['file_path'])}" if row["file_path"] else "#"
            prompt_escaped = row["prompt"].replace('"', '&quot;').replace("<", "&lt;")
            
            error_html = ''
            if row["error"]:
                error_html = f'<div class="tag tag-danger" title="{row["error"]}">Error</div>'
            elif not row["file_path"]:
                 error_html = f'<div class="tag tag-working pulse">Queued</div>'

            cards.append(f"""
            <div class="sprite-card">
              <a href="{url if row['file_path'] else '#'}" target="_blank">
                <div class="thumb-wrap" style="{'' if row['file_path'] else 'opacity: 0.5;'}">
                  {f'<img src="{url}" alt="sprite" loading="lazy" />' if row["file_path"] else '<span>Pending...</span>'}
                </div>
              </a>
              <div class="card-body">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                  <div class="attempt-badge">Attempt #{row['attempt_number']}</div>
                  {error_html}
                </div>
                <div class="prompt-text" title="{prompt_escaped}">{prompt_escaped}</div>
                <div class="card-meta">
                  <span>🕒 {ts}</span>
                  <span>⏱ {dur}</span>
                </div>
              </div>
            </div>""")
        cards_html = "\n".join(cards)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Sprite Gallery</title>
  {SHARED_CSS}
  <style>
    .page-header {{ padding: 36px 28px 0; max-width: 1200px; margin: 0 auto; }}
    h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 4px; }}
    .sub {{ color: var(--muted); font-size: 14px; margin-bottom: 28px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 20px; padding: 0 28px 48px; max-width: 1200px; margin: 0 auto; }}
    .sprite-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }}
    .thumb-wrap {{ background: repeating-conic-gradient(#1e1e30 0% 25%, #161626 0% 50%) 0 0 / 16px 16px; display: flex; align-items: center; justify-content: center; height: 180px; }}
    .thumb-wrap img {{ max-height: 100%; max-width: 100%; image-rendering: pixelated; }}
    .card-body {{ padding: 12px 14px 14px; }}
    .attempt-badge {{ display: inline-block; font-size: 10px; font-weight: 700; background: rgba(124,106,247,.1); color: var(--accent2); border-radius: 4px; padding: 2px 6px; margin-bottom: 7px; }}
    .prompt-text {{ font-size: 13px; color: var(--text); display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; margin-bottom: 10px; }}
    .card-meta {{ display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); }}
    .empty {{ text-align: center; padding: 80px 20px; color: var(--muted); }}
  </style>
</head>
<body>
  {nav_bar("gallery")}
  <div class="page-header">
    <h1>🖼️ Sprite Gallery</h1>
    <p class="sub">Completed and queued sprites.</p>
  </div>
  <div class="grid">{cards_html}</div>
</body>
</html>"""

@app.post("/api/generate")
def generate_sprite(prompt: str = Form(...)):
    # Clean prompt for DB record
    DIRECTIONS = ["PixelartFSS", "PixelartBSS", "PixelartLSS", "PixelartRSS"]
    clean_prompt = prompt
    for t in DIRECTIONS:
        clean_prompt = clean_prompt.replace(t, "").strip().lstrip(",").strip()

    # Queue the task
    task = generate_sprite_task.delay(prompt)

    # Store skeleton record in DB immediately
    conn = get_db()
    if conn:
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO sprite_images (prompt, task_id) VALUES (%s, %s)",
                        (clean_prompt, task.id)
                    )
        except Exception as e:
            print(f"Initial record save failed: {e}")
        finally:
            conn.close()

    return JSONResponse({"status": "queued", "task_id": task.id})

@app.get("/api/task-status/{task_id}")
def get_task_status(task_id: str):
    res = AsyncResult(task_id, app=celery_app)
    result_data = None
    if res.ready():
        result_data = res.result
    
    return {
        "task_id": task_id,
        "status": res.status,
        "result": result_data
    }

@app.get("/api/tasks/recent")
def recent_tasks():
    # Fetch recent 10 records from DB (includes queued and finished)
    return fetch_gallery_rows(limit=10)