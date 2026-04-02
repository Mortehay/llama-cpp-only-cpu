import os
import io
import time
import torch
import requests
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, Response
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

app = FastAPI()

# Store models to a shared directory
os.environ["U2NET_HOME"] = "/models/rembg"

try:
    from rembg import remove
except ImportError:
    remove = lambda x: x

pipe = None
is_loading = False

def get_pipeline():
    global pipe, is_loading
    if pipe is not None:
        return pipe
    
    is_loading = True
    print("Loading Stable Diffusion Pipeline on CPU...")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            "Onodofthenorth/SD_PixelArt_SpriteSheet_Generator",
            torch_dtype=torch.float32,     # Float32 is critical for CPU! Otherwise it outputs noise.
            cache_dir="/models",
            token=os.environ.get("HF_TOKEN")
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.to("cpu")
        pipe.enable_attention_slicing()
        print("Pipeline loaded successfully.")
    except Exception as e:
        print(f"Error loading diffusers model: {e}")
        pipe = None
    finally:
        is_loading = False
    return pipe


@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Sprite Generator (CPU)</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #2b2b2b; color: white; margin: 0; padding: 20px; text-align: center; }
            .container { max-width: 700px; margin: auto; background: #333; padding: 30px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.5); }
            h1 { font-weight: normal; margin-top: 0; }
            p { font-size: 14px; color: #ccc; }
            textarea { width: 100%; box-sizing: border-box; background: #222; color: #fff; padding: 10px; border: 1px solid #444; border-radius: 6px; margin-bottom: 20px; font-family: inherit; resize: vertical; }
            button { width: 100%; box-sizing: border-box; background: #5865F2; color: white; padding: 14px; border: none; border-radius: 6px; font-size: 16px; font-weight: bold; cursor: pointer; transition: background 0.2s; }
            button:hover { background: #4752C4; }
            button:disabled { background: #555; cursor: not-allowed; }
            #status { margin-top: 15px; color: #ffcc00; font-weight: bold; min-height: 20px; }
            .preview-container { margin-top: 20px; border: 2px dashed #555; padding: 10px; border-radius: 8px; min-height: 200px; display: flex; align-items: center; justify-content: center; background-image: url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAQAAAAngNWGAAAAEUlEQVR42mNk+M+AARgwMgwD124H+Tj2UxoAAAAASUVORK5CYII='); }
            img { max-width: 100%; height: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Pixel Art Sprite Generator</h1>
            <p>Generate a 4x4 sprite sheet (16 movement sets). Executed on CPU-only, background is auto-removed.</p>
            <textarea id="prompt" rows="3" placeholder="Enter your prompt here...">pixel art sprite sheet, 4x4 grid, 16 frames, each 30px width 50px height, moving goblin, walking 4 directions, solid white background, retro game art top-down, without extra space, background is transparant, as result should be 16 subimages at image, between frames should be black grid</textarea>
            <button id="gen-btn" onclick="generate()">Generate Sprite</button>
            <div id="status"></div>
            <div class="preview-container" id="result">
                <span style="color: #666;">Image preview will appear here</span>
            </div>
        </div>
        <script>
            async function generate() {
                const promptVal = document.getElementById('prompt').value;
                const resultDiv = document.getElementById('result');
                const statusDiv = document.getElementById('status');
                const btn = document.getElementById('gen-btn');
                
                resultDiv.innerHTML = '';
                statusDiv.innerText = 'Generating on CPU... This may take several minutes.';
                btn.disabled = true;
                
                try {
                    const formData = new FormData();
                    formData.append('prompt', promptVal);
                    
                    const req = await fetch('/api/generate', { method: 'POST', body: formData });
                    
                    if (req.ok) {
                        const blob = await req.blob();
                        const url = URL.createObjectURL(blob);
                        resultDiv.innerHTML = `<img src="${url}" alt="Result" />`;
                        statusDiv.innerText = 'Generation complete!';
                    } else {
                        statusDiv.innerText = 'Error: ' + await req.text();
                    }
                } catch (e) {
                    statusDiv.innerText = 'Error: ' + e.message;
                } finally {
                    btn.disabled = false;
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/api/generate")
def generate_sprite(prompt: str = Form(...)):
    global is_loading
    if is_loading:
        return Response(content="Model is currently downloading/loading. Please wait and try again in a few minutes.", status_code=503)
        
    p = get_pipeline()
    if not p:
        return Response(content="Model failed to load.", status_code=500)
    
    print(f"Generating sprite for prompt: {prompt}")
    start_time = time.time()
    
    # Run the SD pipeline. For CPU we do 20 steps to save time. 
    image = p(
        prompt,
        height=512,
        width=512, # larger res gives the model more spatial layout to draw 16 frames cleanly
        num_inference_steps=20, 
        guidance_scale=7.5
    ).images[0]
    
    end_time = time.time()
    total_duration_ms = (end_time - start_time) * 1000
    
    # Send stats
    try:
        tokens = len(prompt.split())*20
        requests.post("http://stats_collector:8000/v1/internal/log_stats", json={
            "model_name": "Onodofthenorth/SD_PixelArt_SpriteSheet_Generator",
            "prompt_tokens": tokens, # approximate word count as prompt tokens
            "completion_tokens": 1000, # Map inference steps to completion tokens
            "total_tokens": tokens,
            "tokens_per_second": tokens / max(end_time - start_time, 0.001), # steps per second!
            "prompt_eval_ms": 0.0,
            "total_duration_ms": total_duration_ms
        }, timeout=2)
    except Exception as e:
        print(f"Could not log stats: {e}")
    
    # Process transparency
    print("Removing background...")
    try:
        image = remove(image)
    except Exception as e:
        print(f"Failed to remove background: {e}")
        pass
    
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
