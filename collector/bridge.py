import os
import requests
import psycopg2
from fastapi import FastAPI, Request

app = FastAPI()

import time

# Connect using environment variables with a retry mechanism
for _ in range(15):
    try:
        conn = psycopg2.connect(os.getenv("DB_URL"))
        break
    except psycopg2.OperationalError:
        print("Database not ready yet, waiting 2 seconds...")
        time.sleep(2)
else:
    raise Exception("Could not connect to database after 30 seconds.")

cursor = conn.cursor()

@app.post("/v1/chat/completions")
async def proxy_and_log(request: Request):
    body = await request.json()
    
    # Forward to the llama-server router
    response_raw = requests.post(f"{os.getenv('LLM_URL')}/v1/chat/completions", json=body)
    response = response_raw.json()

    usage = response.get("usage", {})
    timings = response.get("timings", {})
    
    # Extract stats from the llama.cpp 2026 response format
    query = """
    INSERT INTO llm_stats (model_name, prompt_tokens, completion_tokens, total_tokens, tokens_per_second, prompt_eval_ms, total_duration_ms)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(query, (
        body.get("model", "default-model"), # Log the model you requested
        usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
        usage.get("total_tokens", 0),
        timings.get("predicted_per_second", 0),
        timings.get("prompt_ms", 0),
        timings.get("predicted_ms", 0)
    ))
    conn.commit()

    return response