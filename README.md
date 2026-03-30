# llama-cpp-only-cpu
llama-cpp-only-cpu

## Usage
### Send a request: Instead of hitting the LLM directly, send your request to the Collector on port 8000

```curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "How does quantization work?"}]}'```
### Check your stats: Log into your Postgres container to see the saved data:
  ```docker exec -it stats_db psql -U user -d llm_monitoring -c "SELECT * FROM llm_stats;"```