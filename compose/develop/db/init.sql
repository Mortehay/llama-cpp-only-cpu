CREATE TABLE IF NOT EXISTS llm_stats (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_name TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    tokens_per_second FLOAT,
    prompt_eval_ms FLOAT,
    total_duration_ms FLOAT
);