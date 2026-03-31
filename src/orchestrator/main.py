from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import os, subprocess, asyncio

app = FastAPI()
MODELS_DIR = "/models"

def list_models():
    files = []
    for f in os.listdir(MODELS_DIR):
        if f.endswith(".gguf"):
            path = os.path.join(MODELS_DIR, f)
            size_mb = round(os.path.getsize(path) / (1024 * 1024), 1)
            files.append({"name": f[:-5], "file": f, "size_mb": size_mb})
    return sorted(files, key=lambda x: x["name"])

@app.get("/api/models")
def get_models():
    return list_models()

@app.delete("/api/models/{filename}")
def delete_model(filename: str):
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path) and filename.endswith(".gguf"):
        os.remove(path)
        return {"status": "deleted", "file": filename}
    return {"status": "not_found"}

@app.get("/api/download/stream")
async def download_stream(repo: str, file: str):
    async def generator():
        process = await asyncio.create_subprocess_exec(
            "hf", "download", repo, file, "--local-dir", MODELS_DIR,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        async for line in process.stdout:
            yield f"data: {line.decode().rstrip()}\n\n"
        await process.wait()
        yield f"data: __DONE__\n\n"
    return StreamingResponse(generator(), media_type="text/event-stream")

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("/app/index.html") as f:
        return f.read()
