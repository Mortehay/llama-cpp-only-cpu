# llama-cpp-only-cpu
llama-cpp-only-cpu

## Usage
### Send a request: Instead of hitting the LLM directly, send your request to the Collector on port 8000

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Llama-3.2-1B-Instruct-Q4_K_M",
    "messages": [{"role": "user", "content": "How does quantization work?"}]
  }'
```

*(Note: Change `"model"` to any filename listed in your `models.txt` like `"gemma-2-2b-it-Q4_K_M"` or `"Mistral-7B-Instruct-v0.3-Q4_K_M"` to route requests to different models!)*

### Check your stats: Log into your Postgres container to see the saved data:
```bash
docker exec -it stats_db psql -U user -d llm_monitoring -c "SELECT * FROM llm_stats;"
```