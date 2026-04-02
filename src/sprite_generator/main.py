import os
import io
import time
import uuid
import torch
import requests
import psycopg2
from contextlib import asynccontextmanager
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
from migrations import run_migrations

DB_URL = os.environ.get("DB_URL")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run DB migrations before accepting traffic."""
    run_migrations(DB_URL)
    yield


app = FastAPI(lifespan=lifespan)

os.environ["U2NET_HOME"] = "/models/rembg"

# Ensure images directory exists
IMAGES_DIR = "/app/images"
os.makedirs(IMAGES_DIR, exist_ok=True)

# Mount static files for serving saved images
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

pipe = None
is_loading = False


def get_db():
    """Get a fresh psycopg2 connection."""
    if not DB_URL:
        return None
    try:
        return psycopg2.connect(DB_URL)
    except Exception as e:
        print(f"DB connection failed: {e}")
        return None


def save_image_record(prompt: str, file_path: str, duration_ms: float):
    """Insert a sprite_images row into the database."""
    conn = get_db()
    if not conn:
        return
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO sprite_images (prompt, file_path, duration_ms) VALUES (%s, %s, %s)",
                    (prompt, file_path, duration_ms),
                )
    except Exception as e:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO sprite_images (prompt, duration_ms, error) VALUES (%s, %s, %s)",
                    (prompt, duration_ms, str(e)),
                )
        print(f"Could not save image record: {e}")
    finally:
        conn.close()


def fetch_gallery_rows():
    """Return all sprite_images rows plus attempt_number (rank within same prompt)."""
    conn = get_db()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    timestamp,
                    prompt,
                    file_path,
                    duration_ms,
                    ROW_NUMBER() OVER (PARTITION BY prompt ORDER BY timestamp) AS attempt_number
                FROM sprite_images
                ORDER BY timestamp DESC
                """
            )
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception as e:
        print(f"Could not fetch gallery: {e}")
        return []
    finally:
        conn.close()


def get_pipeline():
    global pipe, is_loading
    if pipe is not None:
        return pipe
    is_loading = True
    print("Loading Stable Diffusion Pipeline on CPU...")
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
    finally:
        is_loading = False
    return pipe


# ---------------------------------------------------------------------------
# HTML helpers
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
  <title>Sprite Generator (CPU)</title>
  {SHARED_CSS}
  <style>
    .page {{ max-width: 740px; margin: 48px auto; padding: 0 20px; }}
    h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 6px; }}
    .sub {{ color: var(--muted); font-size: 14px; margin-bottom: 32px; }}
    .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 28px; }}
    label {{ font-size: 13px; font-weight: 600; color: var(--muted); text-transform: uppercase;
             letter-spacing: .04em; display: block; margin-bottom: 8px; }}
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
      color: #fff; padding: 15px; border: none; border-radius: 8px;
      font-size: 16px; font-weight: 700; cursor: pointer; transition: opacity .2s;
      letter-spacing: .02em;
    }}
    .btn:hover {{ opacity: .88; }}
    .btn:disabled {{ opacity: .4; cursor: not-allowed; }}
    #status {{
      margin-top: 14px; min-height: 22px; font-size: 14px;
      color: var(--warn); font-weight: 500; text-align: center;
    }}
    .preview {{
      margin-top: 24px; min-height: 220px; border: 2px dashed var(--border);
      border-radius: 10px; display: flex; align-items: center; justify-content: center;
      background: repeating-conic-gradient(#1e1e30 0% 25%, #161626 0% 50%) 0 0 / 20px 20px;
    }}
    .preview img {{ max-width: 100%; border-radius: 6px; image-rendering: pixelated; }}
    .preview-placeholder {{ color: var(--muted); font-size: 14px; }}
    .result-meta {{ margin-top: 12px; font-size: 13px; color: var(--muted); text-align: center; }}
    .result-meta a {{ color: var(--accent2); text-decoration: none; }}
    .result-meta a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  {nav_bar("gen")}
  <div class="page">
    <h1>Pixel Art Sprite Generator</h1>
    <p class="sub">Generates a 4-direction sprite sheet (front, back, left, right) — CPU-only, background auto-removed.</p>
    <div class="card">
      <label for="prompt">Character description</label>
      <textarea id="prompt" rows="3" placeholder="Describe your character...">green zombie, tattered clothes, solid white background</textarea>
      <button class="btn" id="gen-btn" onclick="generate()">⚡ Generate Sprite Sheet</button>
      <div id="status"></div>
      <div class="preview" id="result">
        <span class="preview-placeholder">Image preview will appear here</span>
      </div>
      <div class="result-meta" id="result-meta"></div>
    </div>
  </div>
  <script>
    async function generate() {{
      const promptVal = document.getElementById('prompt').value.trim();
      if (!promptVal) return;
      const resultDiv = document.getElementById('result');
      const statusDiv = document.getElementById('status');
      const metaDiv   = document.getElementById('result-meta');
      const btn = document.getElementById('gen-btn');

      resultDiv.innerHTML = '<span class="preview-placeholder">⏳ Generating… this may take several minutes on CPU.</span>';
      statusDiv.innerText = 'Generating 4 direction passes on CPU...';
      metaDiv.innerHTML   = '';
      btn.disabled = true;

      try {{
        const fd = new FormData();
        fd.append('prompt', promptVal);
        const req = await fetch('/api/generate', {{ method: 'POST', body: fd }});

        if (req.ok) {{
          const data = await req.json();
          resultDiv.innerHTML = `<img src="${{data.url}}" alt="Sprite Sheet" />`;
          statusDiv.innerText = `✅ Done in ${{(data.duration_ms / 1000).toFixed(1)}}s`;
          metaDiv.innerHTML   = `Saved as <a href="${{data.url}}" target="_blank">${{data.filename}}</a>
                                  &nbsp;·&nbsp; Attempt #${{data.attempt_number}} for this prompt
                                  &nbsp;·&nbsp; <a href="/gallery">View Gallery →</a>`;
        }} else {{
          statusDiv.innerText = '❌ Error: ' + await req.text();
        }}
      }} catch (e) {{
        statusDiv.innerText = '❌ Error: ' + e.message;
      }} finally {{
        btn.disabled = false;
      }}
    }}
  </script>
</body>
</html>"""


@app.get("/gallery", response_class=HTMLResponse)
async def gallery():
    rows = fetch_gallery_rows()

    if not rows:
        cards_html = """
        <div class="empty">
          <span>🖼️</span>
          <p>No sprites generated yet. <a href="/">Generate your first one!</a></p>
        </div>"""
    else:
        cards = []
        for row in rows:
            ts = row["timestamp"].strftime("%Y-%m-%d %H:%M") if row["timestamp"] else "—"
            dur = f"{row['duration_ms'] / 1000:.1f}s" if row["duration_ms"] else "—"
            url = f"/images/{os.path.basename(row['file_path'])}"
            prompt_escaped = row["prompt"].replace('"', '&quot;').replace("<", "&lt;")
            cards.append(f"""
            <div class="sprite-card">
              <a href="{url}" target="_blank">
                <div class="thumb-wrap">
                  <img src="{url}" alt="sprite" loading="lazy" />
                </div>
              </a>
              <div class="card-body">
                <div class="attempt-badge">Attempt #{row['attempt_number']}</div>
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
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sprite Gallery</title>
  {SHARED_CSS}
  <style>
    .page-header {{ padding: 36px 28px 0; max-width: 1200px; margin: 0 auto; }}
    h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 4px; }}
    .sub {{ color: var(--muted); font-size: 14px; margin-bottom: 28px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
      gap: 20px; padding: 0 28px 48px;
      max-width: 1200px; margin: 0 auto;
    }}
    .sprite-card {{
      background: var(--card); border: 1px solid var(--border);
      border-radius: 12px; overflow: hidden;
      transition: transform .2s, box-shadow .2s;
    }}
    .sprite-card:hover {{ transform: translateY(-3px); box-shadow: 0 8px 24px rgba(124,106,247,.18); }}
    .thumb-wrap {{
      background: repeating-conic-gradient(#1e1e30 0% 25%, #161626 0% 50%) 0 0 / 16px 16px;
      display: flex; align-items: center; justify-content: center; height: 200px; overflow: hidden;
    }}
    .thumb-wrap img {{
      max-height: 100%; max-width: 100%;
      image-rendering: pixelated; object-fit: contain;
    }}
    .card-body {{ padding: 12px 14px 14px; }}
    .attempt-badge {{
      display: inline-block; font-size: 11px; font-weight: 700;
      background: rgba(124,106,247,.2); color: var(--accent2);
      border: 1px solid rgba(124,106,247,.35);
      border-radius: 20px; padding: 2px 9px; margin-bottom: 7px;
    }}
    .prompt-text {{
      font-size: 13px; color: var(--text);
      display: -webkit-box; -webkit-line-clamp: 2;
      -webkit-box-orient: vertical; overflow: hidden;
      margin-bottom: 10px; line-height: 1.45;
    }}
    .card-meta {{
      display: flex; justify-content: space-between;
      font-size: 12px; color: var(--muted); gap: 8px;
    }}
    .empty {{
      text-align: center; padding: 80px 20px;
      color: var(--muted); font-size: 15px;
    }}
    .empty span {{ font-size: 48px; display: block; margin-bottom: 16px; }}
    .empty a {{ color: var(--accent2); }}
  </style>
</head>
<body>
  {nav_bar("gallery")}
  <div class="page-header">
    <h1>🖼️ Sprite Gallery</h1>
    <p class="sub">{len(rows)} sprite{"s" if len(rows) != 1 else ""} generated — sorted newest first. Attempt # resets per unique prompt.</p>
  </div>
  <div class="grid">
    {cards_html}
  </div>
</body>
</html>"""


@app.post("/api/generate")
def generate_sprite(prompt: str = Form(...)):
    global is_loading
    if is_loading:
        return Response(
            content="Model is currently loading. Please wait and try again.",
            status_code=503,
        )

    p = get_pipeline()
    if not p:
        return Response(content="Model failed to load.", status_code=500)

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

    # Save to disk
    filename = f"sprite_{uuid.uuid4().hex[:12]}.png"
    filepath = os.path.join(IMAGES_DIR, filename)
    buf = io.BytesIO()
    master.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    with open(filepath, "wb") as f:
        f.write(png_bytes)
    print(f"Saved image: {filepath}")

    # Save record to DB
    save_image_record(clean_prompt, filepath, total_duration_ms)

    # Get attempt number for this prompt
    attempt_number = 1
    conn = get_db()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM sprite_images WHERE prompt = %s",
                    (clean_prompt,),
                )
                attempt_number = cur.fetchone()[0]
        except Exception:
            pass
        finally:
            conn.close()

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

    from fastapi.responses import JSONResponse
    return JSONResponse({
        "url": f"/images/{filename}",
        "filename": filename,
        "duration_ms": total_duration_ms,
        "attempt_number": attempt_number,
    })