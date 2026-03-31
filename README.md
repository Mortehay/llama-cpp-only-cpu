# llama-cpp-only-cpu

A fully **local, CPU-only LLM cluster** built on top of [`llama.cpp`](https://github.com/ggml-org/llama.cpp). Run multiple open-weight language models side-by-side on any x86 Linux machine — no GPU required. Includes automatic model downloading, a stats-collecting API proxy, a chat UI, a model management dashboard, and a Grafana metrics visualizer, all wired together with Docker Compose.

---

## Architecture

```
Browser / curl
     │
     ▼
┌─────────────────┐     ┌──────────────────────┐
│  stats_collector │────▶│     llm_engine        │
│  (port 8000)    │     │  llama.cpp router     │
│  Logs usage to  │     │  (port 8080)          │
│  Postgres DB    │     │  4 models on-demand   │
└────────┬────────┘     └──────────────────────┘
         │
         ▼
┌─────────────────┐
│   stats_db      │
│  PostgreSQL 15  │
│  (port 5432)    │
└─────────────────┘

┌─────────────────┐     ┌──────────────────────┐
│  open_webui     │     │  model_orchestrator  │
│  Chat UI        │     │  Download/manage     │
│  (port 3000)    │     │  models visually     │
└─────────────────┘     │  (port 7860)         │
                        └──────────────────────┘
┌─────────────────┐
│  grafana        │
│  Metrics charts │
│  (port 3001)    │
└─────────────────┘
```

---

## Services at a Glance

| Container | URL | Description |
|---|---|---|
| `stats_collector` | http://localhost:8000 | OpenAI-compatible API proxy. All requests should go here — it logs usage stats to Postgres. |
| `llm_engine` | http://localhost:8080 | Raw `llama.cpp` multi-model router. Loads a model into memory on first request, unloads when idle. |
| `open_webui` | http://localhost:3000 | ChatGPT-style browser chat interface pre-wired to the collector. |
| `model_orchestrator` | http://localhost:7860 | Visual dashboard to view, download, and delete GGUF models. |
| `grafana` | http://localhost:3001 | Grafana metrics dashboard — connect to Postgres to visualize token rates, model usage, and more. |
| `stats_db` | `localhost:5432` | PostgreSQL database storing all LLM request statistics. |
| `model_downloader` | *(internal)* | Downloads models from HuggingFace on startup; stays running to serve `make download` commands. |

---

## Quick Start

```bash
# Clone and enter the project
git clone <repo-url>
cd llama-cpp-only-cpu

# Copy and fill in your HuggingFace token (needed for gated models)
cp compose/develop/.env.example compose/develop/.env

# Build images, download models, and start everything
make rebuild
```

After the first run, all 4 models are cached on disk — subsequent `make rebuild` calls are instant.

---

## Sending Requests

### Via the Chat UI (recommended)

Open **http://localhost:3000**, select a model from the dropdown, and start chatting.

### Via API (curl)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Llama-3.2-1B-Instruct-Q4_K_M",
    "messages": [{"role": "user", "content": "Explain quantization in simple terms."}]
  }'
```

> The API is OpenAI-compatible — any client that supports `OPENAI_API_BASE_URL` can point to `http://localhost:8000/v1`.

---

## Available Models

Defined in `compose/develop/downloader/models.txt`:

| Model | Size | Speed |
|---|---|---|
| `Llama-3.2-1B-Instruct-Q4_K_M` | ~760 MB | ⚡ Fastest |
| `Llama-3.2-3B-Instruct-Q4_K_M` | ~2 GB | 🔥 Fast |
| `gemma-2-2b-it-Q4_K_M` | ~1.5 GB | 🔥 Fast |
| `Mistral-7B-Instruct-v0.3-Q4_K_M` | ~4.1 GB | 🧠 Smartest |

Models are loaded **on-demand** — only the first request to a model triggers it to load into RAM.

---

## Downloading New Models

### Via the Model Orchestrator UI

Open **http://localhost:7860**, enter a HuggingFace repo and filename, and click **Download**. Progress streams live in the browser.

### Via Make command

```bash
make download repo=bartowski/Qwen2.5-1.5B-Instruct-GGUF file=Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
```

The model is immediately available and saved to `models.txt` for future rebuilds.

---

## Viewing Stats

### Postgres (raw)

```bash
docker exec -it stats_db psql -U postgres -d llm_monitoring -c "SELECT * FROM llm_stats;"
```

### Grafana (visual)

1. Open **http://localhost:3001** (login: `admin` / `admin`)
2. Go to **Connections → Add data source → PostgreSQL**
3. Fill in:
   - **Host:** `db:5432`
   - **Database:** `llm_monitoring`
   - **User:** `postgres` | **Password:** `postgres`
4. Build dashboards with `llm_stats` columns: `model_name`, `tokens_per_second`, `total_tokens`, `total_duration_ms`

---

## Project Structure

```
llama-cpp-only-cpu/
├── src/
│   ├── collector/          ← FastAPI proxy that logs stats to Postgres
│   │   └── bridge.py
│   └── orchestrator/       ← Model management web UI (FastAPI + HTML)
│       ├── main.py
│       └── index.html
├── compose/develop/
│   ├── collector/          ← Dockerfile for collector
│   ├── orchestrator/       ← Dockerfile for orchestrator
│   ├── downloader/         ← Dockerfile + models.txt + download script
│   ├── db/                 ← init.sql (creates llm_stats table)
│   ├── docker-compose.yml
│   └── .env                ← HF_TOKEN, DB credentials
├── models/                 ← Downloaded .gguf files (gitignored)
└── Makefile
```

---

## Make Commands

| Command | Description |
|---|---|
| `make rebuild` | Build images, run downloader, start full stack |
| `make build` | Build Docker images only |
| `make dev` | Start the stack without rebuilding |
| `make download repo=<hf-repo> file=<filename>` | Download a new model into the running stack |
| `make rebuild-app` | Rebuild and restart only the collector |