import os
import json
import time
import requests
import psycopg2
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()

LLM_URL = os.getenv("LLM_URL", "http://llm-server:8080")

# Connect to Postgres with retry
for _ in range(15):
    try:
        conn = psycopg2.connect(os.getenv("DB_URL"))
        conn.autocommit = True
        break
    except psycopg2.OperationalError:
        print("Database not ready yet, waiting 2 seconds...")
        time.sleep(2)
else:
    raise Exception("Could not connect to database after 30 seconds.")

cursor = conn.cursor()


def log_stats(model_name, prompt_tokens, completion_tokens, total_tokens, tps, prompt_ms, total_ms):
    try:
        cursor.execute(
            """INSERT INTO llm_stats
               (model_name, prompt_tokens, completion_tokens, total_tokens,
                tokens_per_second, prompt_eval_ms, total_duration_ms)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (model_name, prompt_tokens, completion_tokens, total_tokens, tps, prompt_ms, total_ms),
        )
    except Exception as e:
        print(f"[stats] failed to log: {e}")


@app.get("/v1/models")
async def list_models():
    """Proxy the models list from llama.cpp so Open WebUI can discover them."""
    resp = requests.get(f"{LLM_URL}/v1/models")
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.post("/v1/chat/completions")
async def proxy_and_log(request: Request):
    body = await request.json()
    model_name = body.get("model", "default-model")
    is_stream = body.get("stream", False)

    if is_stream:
        # ── Streaming path ────────────────────────────────────────────────
        upstream = requests.post(
            f"{LLM_URL}/v1/chat/completions",
            json=body,
            stream=True,
        )

        # Collect completion tokens as we stream so we can log at the end
        completion_tokens = 0
        prompt_tokens = 0

        def generate():
            nonlocal completion_tokens, prompt_tokens
            for line in upstream.iter_lines():
                if not line:
                    yield "\n"
                    continue
                decoded = line.decode("utf-8")
                yield decoded + "\n"

                # Parse usage from the final [DONE] chunk if present
                if decoded.startswith("data:"):
                    data_str = decoded[5:].strip()
                    if data_str == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data_str)
                        usage = chunk.get("usage") or {}
                        if usage:
                            prompt_tokens = usage.get("prompt_tokens", 0)
                            completion_tokens = usage.get("completion_tokens", 0)
                            total = usage.get("total_tokens", 0)
                            log_stats(model_name, prompt_tokens, completion_tokens, total, 0, 0, 0)
                    except Exception:
                        pass

        return StreamingResponse(generate(), media_type="text/event-stream")

    else:
        # ── Non-streaming path (curl, etc.) ───────────────────────────────
        response_raw = requests.post(f"{LLM_URL}/v1/chat/completions", json=body)
        response = response_raw.json()

        usage = response.get("usage", {})
        timings = response.get("timings", {})
        log_stats(
            model_name,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            usage.get("total_tokens", 0),
            timings.get("predicted_per_second", 0),
            timings.get("prompt_ms", 0),
            timings.get("predicted_ms", 0),
        )
        return response