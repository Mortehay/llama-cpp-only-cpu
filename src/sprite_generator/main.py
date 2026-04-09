import os
import io
import time
import json
import requests
import psycopg2
from contextlib import asynccontextmanager
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
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
  .tag-core { background: rgba(124, 106, 247, 0.15); color: #a78bfa; border: 1px solid rgba(124, 106, 247, 0.2); }

  .progress-bg { height: 4px; width: 100%; background: var(--border); border-radius: 2px; overflow: hidden; margin-top: 6px; }
  .progress-fill { height: 100%; background: var(--accent); transition: width .3s ease-out; }
  
  .btn-sm { padding: 4px 10px; border-radius: 4px; border: 1px solid var(--border); background: var(--surface); color: var(--text); font-size: 11px; font-weight: 600; cursor: pointer; transition: all .2s; }
  .btn-sm:hover { background: var(--card); border-color: var(--muted); }
  .btn-danger-sm:hover { background: var(--danger); border-color: var(--danger); }
  .btn-retry-sm:hover { background: var(--success); border-color: var(--success); color: #000; }
  
  .tabs { display: flex; gap: 4px; margin-bottom: 20px; background: var(--surface); padding: 6px; border-radius: 10px; border: 1px solid var(--border); }
  .tab { flex: 1; text-align: center; padding: 10px; font-size: 14px; font-weight: 600; color: var(--muted); cursor: pointer; border-radius: 6px; transition: all .2s; }
  .tab.active { background: var(--card); color: var(--text); }
  .tab-content { display: none; }
  .tab-content.active { display: block; }
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
  <title>Sprite Generator</title>
  {SHARED_CSS}
  <style>
    .page {{ max-width: 900px; margin: 48px auto; padding: 0 20px; display: grid; grid-template-columns: 1fr 340px; gap: 32px; }}
    @media (max-width: 850px) {{ .page {{ grid-template-columns: 1fr; }} }}
    h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 6px; }}
    .sub {{ color: var(--muted); font-size: 14px; margin-bottom: 32px; }}
    .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 24px; }}
    label {{ font-size: 12px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; display: block; margin-bottom: 8px; }}
    textarea {{
      width: 100%; background: var(--surface); color: var(--text); border: 1px solid var(--border); border-radius: 8px;
      padding: 12px; font-size: 15px; font-family: inherit; resize: vertical;
    }}
    .btn {{
      display: block; width: 100%; margin-top: 20px;
      background: linear-gradient(135deg, var(--accent) 0%, #a855f7 100%);
      color: #fff; padding: 14px; border: none; border-radius: 8px;
      font-size: 15px; font-weight: 700; cursor: pointer;
    }}
    .btn:disabled {{ opacity: .4; cursor: not-allowed; filter: grayscale(1); }}
    .status {{ margin-top: 14px; font-size: 13px; color: var(--warn); text-align: center; line-height: 1.5; }}
    
    .preview {{
      margin-top: 24px; min-height: 200px; border: 2px dashed var(--border);
      border-radius: 10px; display: flex; align-items: center; justify-content: center;
      background: repeating-conic-gradient(#1e1e30 0% 25%, #161626 0% 50%) 0 0 / 20px 20px; overflow: hidden;
    }}
    .preview img {{ max-width: 100%; image-rendering: pixelated; }}
    .preview-placeholder {{ color: var(--muted); font-size: 13px; text-align: center; }}

    /* Task List Sidebar */
    .task-list-title {{ font-size: 14px; font-weight: 700; color: var(--text); margin-bottom: 16px; display: flex; justify-content: space-between; }}
    .task-item {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px; margin-bottom: 12px; font-size: 13px; }}
    .task-item .prompt-clip {{ color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 6px; font-weight: 500; }}
    .task-item .meta {{ display: flex; justify-content: space-between; align-items: center; font-size: 11px; color: var(--muted); }}
    .task-item .progress-info {{ font-size: 11px; color: var(--accent2); margin-top: 4px; font-weight: 600; display: block; }}
    .pulse {{ animation: pulse 2s infinite; }}
    @keyframes pulse {{ 0% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} 100% {{ opacity: 1; }} }}
    
    /* Core Image Picker */
    .core-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; max-height: 300px; overflow-y: auto; margin-bottom: 20px; }}
    .core-item {{ 
        border: 2px solid var(--surface); border-radius: 8px; padding: 4px; cursor: pointer;
        background: repeating-conic-gradient(#1e1e30 0% 25%, #161626 0% 50%) 0 0 / 10px 10px;
    }}
    .core-item img {{ width: 100%; display: block; border-radius: 4px; image-rendering: pixelated; }}
    .core-item.selected {{ border-color: var(--accent); background: var(--accent2); }}
    .core-item:hover {{ border-color: var(--accent2); }}
    
    /* Actions Grid */
    .actions-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin-bottom: 16px; }}
    .action-cb {{ display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--text); background: var(--surface); padding: 8px 12px; border-radius: 6px; border: 1px solid var(--border); cursor: pointer; }}
    .action-cb:hover {{ border-color: var(--muted); }}
  </style>
</head>
<body>
  {nav_bar("gen")}
  <div class="page">
    <div class="main-content">
      <h1>Pixel Art Generator</h1>
      <p class="sub">2-Step generation: create a core character, then build custom spritesheets.</p>
      
      <div class="tabs">
        <div class="tab active" onclick="switchTab('core')">Step 1: Core Generator</div>
        <div class="tab" onclick="switchTab('sheet'); loadCores();">Step 2: Spritesheet Builder</div>
      </div>
      
      <div class="card tab-content active" id="tab-core">
        <label>Select Model</label>
        <select id="core-llm" style="width: 100%; background: var(--surface); color: var(--text); border: 1px solid var(--border); border-radius: 8px; padding: 12px; font-size: 14px; margin-bottom: 16px;">
            <option value="stabilityai/sdxl-turbo" selected>SDXL Turbo (Default)</option>
            <option value="Onodofthenorth/SD_PixelArt_SpriteSheet_Generator">Pixel Art SD 1.5</option>
        </select>
        
        <label for="core-prompt">Core Character Description</label>
        <textarea id="core-prompt" rows="3">green zombie, tattered clothes, solid transparent background</textarea>
        <button class="btn" id="gen-core-btn" onclick="generateCore()">⚡ Generate Core Sprite</button>
        <div class="status" id="core-status"></div>
        <div class="preview" id="core-result">
          <span class="preview-placeholder">Live preview will appear here.</span>
        </div>
      </div>
      
      <div class="card tab-content" id="tab-sheet">
        <label>Select Model</label>
        <select id="sheet-llm" style="width: 100%; background: var(--surface); color: var(--text); border: 1px solid var(--border); border-radius: 8px; padding: 12px; font-size: 14px; margin-bottom: 16px;">
            <option value="stabilityai/sdxl-turbo" selected>SDXL Turbo (Sprite Grid)</option>
            <option value="Onodofthenorth/SD_PixelArt_SpriteSheet_Generator">Pixel Art SD 1.5 (Sprite Grid Default)</option>
        </select>
        
        <label>Select Core Image</label>
        <div class="core-grid" id="core-picker"><span style="color:var(--muted); font-size: 13px;">Loading cores...</span></div>
        
        <label>Select Actions</label>
        <div class="actions-grid">
            <label class="action-cb"><input type="checkbox" name="action" value="move right" checked /> Move Right</label>
            <label class="action-cb"><input type="checkbox" name="action" value="move left" checked /> Move Left</label>
            <label class="action-cb"><input type="checkbox" name="action" value="move down" checked /> Move Down</label>
            <label class="action-cb"><input type="checkbox" name="action" value="move up" checked /> Move Up</label>
            <label class="action-cb"><input type="checkbox" name="action" value="attack" /> Attack</label>
            <label class="action-cb"><input type="checkbox" name="action" value="got damage" /> Get Damage</label>
            <label class="action-cb"><input type="checkbox" name="action" value="idle" /> Idle</label>
            <label class="action-cb"><input type="checkbox" name="action" value="burning" /> Burning</label>
        </div>
        
        <button class="btn" id="gen-sheet-btn" onclick="generateSheet()">⚡ Generate Spritesheet</button>
        <div class="status" id="sheet-status"></div>
        <div class="preview" id="sheet-result">
          <span class="preview-placeholder">Live preview will appear here.</span>
        </div>
      </div>
    </div>

    <div class="sidebar">
       <div class="task-list-title"> 
         <span>Live Tasks</span>
       </div>
       <div id="task-queue">
         <p style="font-size: 12px; color: var(--muted); text-align: center; padding: 40px 0;">Loading tasks...</p>
       </div>
    </div>
  </div>

  <script>
    let pollInterval = null;
    let selectedCoreId = null;

    updateQueue();
    setInterval(updateQueue, 3000);

    function switchTab(tabId) {{
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
        event.currentTarget.classList.add('active');
        document.getElementById('tab-' + tabId).classList.add('active');
        if (pollInterval) clearInterval(pollInterval);
    }}

    async function loadCores() {{
        try {{
            const res = await fetch('/api/cores');
            const cores = await res.json();
            const picker = document.getElementById('core-picker');
            if (cores.length === 0) {{
                picker.innerHTML = '<span style="color:var(--muted); font-size: 13px;">No core images found. Generate one first!</span>';
                return;
            }}
            picker.innerHTML = cores.map(c => `
                <div class="core-item" id="core-sel-${{c.id}}" onclick="selectCore(${{c.id}})">
                    <img src="${{c.file_path.split('/app').pop()}}" title="${{c.prompt}}"/>
                </div>
            `).join('');
            
            // auto-select first
            if(!selectedCoreId && cores.length > 0) selectCore(cores[0].id);
        }} catch(e) {{ console.error("Error loading cores:", e); }}
    }}

    function selectCore(id) {{
        selectedCoreId = id;
        document.querySelectorAll('.core-item').forEach(e => e.classList.remove('selected'));
        const el = document.getElementById(`core-sel-${{id}}`);
        if(el) el.classList.add('selected');
    }}

    async function updateQueue() {{
      try {{
        const res = await fetch('/api/tasks/recent');
        const tasks = await res.json();
        const queueDiv = document.getElementById('task-queue');
        
        if (tasks.length === 0) {{
          queueDiv.innerHTML = '<p style="font-size: 12px; color: var(--muted); text-align: center; padding: 40px 0;">No history yet.</p>';
          return;
        }}

        queueDiv.innerHTML = tasks.map(t => {{
          let statusTag = '';
          let progressLine = '';
          
          if (t.error) {{
             statusTag = '<span class="tag tag-danger">Failed</span>';
             progressLine = `<div style="color: var(--danger); font-size: 11px; margin-top: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${{t.error}}">${{t.error}}</div>`;
          }} else if (t.file_path) {{
             statusTag = '<span class="tag tag-success">Done</span>';
          }} else {{
             statusTag = `<span class="tag tag-working pulse">${{t.progress_pct}}%</span>`;
             progressLine = `
              <span class="progress-info">${{t.progress_msg || 'Preparing...'}}</span>
              <div class="progress-bg"><div class="progress-fill" style="width: ${{t.progress_pct}}%"></div></div>
             `;
          }}
          
          let typeTag = t.image_type === 'core' ? '<span class="tag tag-core" style="margin-left: 4px;">Core</span>' : '';

          return `
            <div class="task-item" id="live-task-${{t.id}}">
              <div class="prompt-clip">${{typeTag}} ${{t.prompt}}</div>
              <div class="meta">
                <span>${{statusTag}}</span>
                <span>${{t.timestamp.split('T')[1].split('.')[0]}}</span>
              </div>
              ${{progressLine}}
              <div style="display: flex; gap: 4px; margin-top: 8px; justify-content: flex-end;">
                  <button class="btn-sm btn-retry-sm" onclick="retryLiveTask(${{t.id}})">Retry</button>
                  <button class="btn-sm btn-danger-sm" onclick="deleteTask(${{t.id}})">Delete</button>
              </div>
            </div>
          `;
        }}).join('');
      }} catch (e) {{ console.error(e); }}
    }}

    async function retryLiveTask(id) {{
        if(!confirm('Duplicate and retry this task?')) return;
        try {{
            const res = await fetch('/api/task/' + id + '/retry', {{ method: 'POST' }});
            if (res.ok) {{
                const data = await res.json();
                const mode = data.image_type === 'spritesheet' ? 'sheet' : data.image_type;
                pollTaskStatus(data.task_id, mode);
                updateQueue();
            }} else {{
                alert('Retry failed: ' + await res.text());
            }}
        }} catch(e) {{ alert('Retry failed: ' + e.message); }}
    }}

    async function deleteTask(id) {{
        if(!confirm('Permanently cancel and remove this task?')) return;
        try {{
            const res = await fetch('/api/task/' + id, {{ method: 'DELETE' }});
            if (res.ok) updateQueue();
        }} catch(e) {{ alert('Delete failed: ' + e.message); }}
    }}

    async function generateCore() {{
      const promptVal = document.getElementById('core-prompt').value.trim();
      if (!promptVal) return;
      const resultDiv = document.getElementById('core-result');
      const statusDiv = document.getElementById('core-status');
      const btn = document.getElementById('gen-core-btn');

      resultDiv.innerHTML = '<span class="preview-placeholder pulse">⏳ Sending task to worker...</span>';
      statusDiv.innerText = 'Initializing...';
      btn.disabled = true;

      try {{
        const llm_name = document.getElementById('core-llm').value;
        const fd = new FormData();
        fd.append('prompt', promptVal);
        fd.append('llm_name', llm_name);
        const req = await fetch('/api/generate_core', {{ method: 'POST', body: fd }});

        if (req.ok) {{
          const data = await req.json();
          pollTaskStatus(data.task_id, 'core');
          updateQueue();
        }} else {{
          statusDiv.innerText = '❌ Error: ' + await req.text();
          btn.disabled = false;
        }}
      }} catch (e) {{
        statusDiv.innerText = '❌ Error: ' + e.message;
        btn.disabled = false;
      }}
    }}

    async function generateSheet() {{
      if (!selectedCoreId) {{ alert("Please select a core image first!"); return; }}
      const checkboxes = document.querySelectorAll('input[name="action"]:checked');
      if (checkboxes.length === 0) {{ alert("Select at least one action!"); return; }}
      
      const actions = Array.from(checkboxes).map(c => c.value);
      
      const resultDiv = document.getElementById('sheet-result');
      const statusDiv = document.getElementById('sheet-status');
      const btn = document.getElementById('gen-sheet-btn');

      resultDiv.innerHTML = '<span class="preview-placeholder pulse">⏳ Sending task to worker...</span>';
      statusDiv.innerText = 'Initializing...';
      btn.disabled = true;

      try {{
        const llm_name = document.getElementById('sheet-llm').value;
        const fd = new FormData();
        fd.append('parent_id', selectedCoreId);
        fd.append('actions', JSON.stringify(actions));
        fd.append('llm_name', llm_name);
        const req = await fetch('/api/generate_sheet', {{ method: 'POST', body: fd }});

        if (req.ok) {{
          const data = await req.json();
          pollTaskStatus(data.task_id, 'sheet');
          updateQueue();
        }} else {{
          statusDiv.innerText = '❌ Error: ' + await req.text();
          btn.disabled = false;
        }}
      }} catch (e) {{
        statusDiv.innerText = '❌ Error: ' + e.message;
        btn.disabled = false;
      }}
    }}

    function pollTaskStatus(taskId, mode) {{
      const statusDiv = document.getElementById(`${{mode}}-status`);
      const resultDiv = document.getElementById(`${{mode}}-result`);
      const btn = document.getElementById(`gen-${{mode}}-btn`);

      if (pollInterval) clearInterval(pollInterval);
      pollInterval = setInterval(async () => {{
        try {{
          const resRecent = await fetch('/api/tasks/recent');
          const recentTasks = await resRecent.json();
          const me = recentTasks.find(t => t.task_id === taskId);

          if (me) {{
            if (me.file_path) {{
                clearInterval(pollInterval);
                resultDiv.innerHTML = `<img src="/images/${{me.file_path.split('/').pop()}}" alt="Sprite" />`;
                statusDiv.innerText = `✅ Success! Completed in ${{me.duration_ms / 1000}}s`;
                btn.disabled = false;
                updateQueue();
                return;
            }}
            if (me.error) {{
                clearInterval(pollInterval);
                statusDiv.innerText = '❌ Error: ' + me.error;
                btn.disabled = false;
                updateQueue();
                return;
            }}
            statusDiv.innerHTML = `
              <div style="font-weight: 700; color: var(--accent2);">${{me.progress_msg || 'Queued'}}</div>
              <div class="progress-bg" style="width: 240px; margin: 8px auto;"><div class="progress-fill" style="width: ${{me.progress_pct}}%"></div></div>
            `;
          }}
        }} catch (e) {{ console.error(e); }}
      }}, 1500);
    }}
  </script>
</body>
</html>"""

@app.get("/gallery", response_class=HTMLResponse)
async def gallery():
    rows = fetch_gallery_rows()
    if not rows:
        cards_html = """<div class="empty"><span>🖼️</span><p>No history.</p></div>"""
    else:
        cards = []
        for row in rows:
            ts = row["timestamp"].strftime("%m-%d %H:%M") if row["timestamp"] else "—"
            dur = f"{row['duration_ms'] / 1000:.1f}s" if row["duration_ms"] else "—"
            prompt_escaped = row["prompt"].replace('"', '&quot;').replace("<", "&lt;")
            
            status_html = ''
            img_content = '<span>Pending...</span>'
            img_url = "#"
            components_html = ""
            
            if row["error"]:
                status_html = f'<div class="tag tag-danger" title="{row["error"]}">Error</div>'
            elif row["file_path"]:
                img_url = f"/images/{os.path.basename(row['file_path'])}"
                img_content = f'<img src="{img_url}" alt="sprite" />'
                status_html = f'<div class="tag tag-success">OK</div>'
                
                # Show un-glued components if any
                comps = row.get("components")
                if comps and isinstance(comps, list) and len(comps) > 0:
                    comps_tags = "".join([f'<img src="{c}" class="comp-img"/>' for c in comps])
                    components_html = f'<div class="components-wrap">{comps_tags}</div>'
            else:
                 status_html = f'<div class="tag tag-working pulse">{row["progress_pct"]}%</div>'
                 
            type_tag = f'<div class="tag tag-core" style="margin-right: 4px;">Core</div>' if row['image_type'] == 'core' else ''
            
            step_tag = ""
            if row.get('step_number'):
                step_tag = f'<div class="tag tag-working" style="margin-right: 4px;">Step {row["step_number"]}</div>'
            
            llm_tag = ""
            if row.get('llm_name') and row['llm_name'] != 'Unknown':
                llm_compact = "SDXL-Turbo" if "sdxl-turbo" in row['llm_name'].lower() else "PixelArt 1.5"
                llm_tag = f'<div class="tag" style="background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.15); margin-right: 4px;">{llm_compact}</div>'

            # Header with buttons
            control_html = f"""
              <div style="display: flex; gap: 4px; margin-top: 8px;">
                <button class="btn-sm btn-danger-sm" onclick="deleteTask({row['id']})">Delete</button>
              </div>
            """

            cards.append(f"""
            <div class="sprite-card" id="card-{row['id']}">
              <a href="{img_url}" target="_blank" style="display: block; width: 100%; border: none;">
                <div class="thumb-wrap">
                  {img_content}
                </div>
              </a>
              {components_html}
              <div class="card-body">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 8px;">
                  <div style="display: flex; flex-wrap: wrap; gap: 4px; align-items: center;">
                      {type_tag}
                      {step_tag}
                      {llm_tag}
                      <span class="attempt-badge" style="margin-left: 2px;">Attempt #{row['attempt_number']}</span>
                  </div>
                  {status_html}
                </div>
                <div class="prompt-text" title="{prompt_escaped}">{prompt_escaped}</div>
                <div class="card-meta"><span>🕒 {ts}</span> <span>⏱ {dur}</span></div>
                {control_html}
              </div>
            </div>""")
        cards_html = "\n".join(cards)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Gallery</title>
  {SHARED_CSS}
  <style>
    .page-header {{ padding: 36px 28px 0; max-width: 1200px; margin: 0 auto; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 24px; padding: 24px 28px; max-width: 1200px; margin: 0 auto; }}
    .sprite-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }}
    .thumb-wrap {{ background: repeating-conic-gradient(#1e1e30 0% 25%, #161626 0% 50%) 0 0 / 16px 16px; height: 180px; display: flex; align-items: center; justify-content: center; cursor: zoom-in; }}
    .thumb-wrap img {{ max-height: 100%; max-width: 100%; image-rendering: pixelated; }}
    .components-wrap {{ display: flex; flex-wrap: wrap; gap: 4px; padding: 8px; background: var(--surface); border-bottom: 1px solid var(--border); }}
    .comp-img {{ height: 40px; background: repeating-conic-gradient(#1e1e30 0% 25%, #161626 0% 50%) 0 0 / 5px 5px; border-radius: 4px; border: 1px solid var(--border); image-rendering: pixelated; }}
    .card-body {{ padding: 12px 14px 14px; position: relative; }}
    .attempt-badge {{ font-size: 10px; font-weight: 700; background: rgba(124,106,247,.1); color: var(--accent2); padding: 2px 6px; border-radius: 4px; display: inline-block; }}
    .prompt-text {{ font-size: 13px; color: var(--text); overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; margin-bottom: 8px; font-weight: 500; }}
    .card-meta {{ display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); }}
    .empty {{ text-align: center; padding: 100px; color: var(--muted); }}
  </style>
</head>
<body>
  {nav_bar("gallery")}
  <div class="page-header">
    <h1>🖼️ Sprite Gallery</h1>
  </div>
  <div class="grid">{cards_html}</div>

  <script>
    async function deleteTask(id) {{
        if(!confirm('Permanently remove this task and image?')) return;
        try {{
            const res = await fetch('/api/task/' + id, {{ method: 'DELETE' }});
            if (res.ok) document.getElementById('card-' + id).remove();
        }} catch(e) {{ alert('Delete failed: ' + e.message); }}
    }}

  </script>
</body>
</html>"""

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