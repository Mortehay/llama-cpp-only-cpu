import os
import json
import time
import requests
import psycopg2
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()

LLM_URL = os.getenv("LLM_URL", "http://llm-server:8080")

# Models that don't support a leading system role in their chat template
MODELS_WITHOUT_SYSTEM = ("mistral", "mixtral")


def preprocess_messages(model_name: str, messages: list) -> list:
    """
    Mistral requires strictly alternating user/assistant messages with no system role.
    This function:
      1. Collects ALL system messages and merges them into the first user turn.
      2. Collapses any consecutive same-role messages into one (merges content).
    """
    if not any(m in model_name.lower() for m in MODELS_WITHOUT_SYSTEM):
        return messages

    # Step 1: Extract system content, collect the rest
    system_parts = []
    non_system = []
    for msg in messages:
        if msg.get("role") == "system":
            system_parts.append(msg.get("content", "").strip())
        else:
            non_system.append(dict(msg))

    # Step 2: Prepend system content to the first user message
    if system_parts:
        system_text = "\n\n".join(filter(None, system_parts))
        for i, msg in enumerate(non_system):
            if msg.get("role") == "user":
                non_system[i]["content"] = f"{system_text}\n\n{msg['content']}"
                break

    # Step 3: Collapse consecutive same-role messages (strict alternation)
    result = []
    for msg in non_system:
        if result and result[-1]["role"] == msg["role"]:
            result[-1]["content"] += "\n\n" + msg["content"]
        else:
            result.append(msg)

    return result


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
    body["messages"] = preprocess_messages(model_name, body.get("messages", []))

    if is_stream:
        # ── Streaming path ────────────────────────────────────────────────
        # Pipe raw bytes directly — iter_lines() drops the empty lines that
        # SSE uses as event delimiters, breaking the client parser.
        upstream = requests.post(
            f"{LLM_URL}/v1/chat/completions",
            json=body,
            stream=True,
        )

        def generate():
            for chunk in upstream.iter_content(chunk_size=None):
                yield chunk

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

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