import os
import requests
import psycopg2
from fastapi import FastAPI, Request

app = FastAPI()

# Connect to Postgres
conn = psycopg2.connect(os.getenv("DB_URL"))
cursor = conn.cursor()

@app.post("/v1/chat/completions")
async def proxy_and_log(request: Request):
    # 1. Get user request and forward to LLM
    body = await request.json()
    response = requests.post(f"{os.getenv('LLM_URL')}/v1/chat/completions", json=body).json()

    # 2. Extract Statistics
    usage = response.get("usage", {})
    # llama.cpp 2026 version often includes 'timings' in the response
    timings = response.get("timings", {})
    
    tps = timings.get("predicted_per_second", 0)
    p_eval = timings.get("prompt_ms", 0)
    total_ms = timings.get("predicted_ms", 0)

    # 3. Save to Postgres
    query = """
    INSERT INTO llm_stats (model_name, prompt_tokens, completion_tokens, total_tokens, tokens_per_second, prompt_eval_ms, total_duration_ms)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(query, (
        response.get("model", "unknown"),
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        usage.get("total_tokens"),
        tps, p_eval, total_ms
    ))
    conn.commit()

    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)