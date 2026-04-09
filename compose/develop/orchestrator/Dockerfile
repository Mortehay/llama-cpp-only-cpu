FROM python:3.12-slim

RUN pip install --no-cache-dir fastapi uvicorn huggingface_hub requests jinja2

WORKDIR /app
# Copy from src/orchestrator at project root
COPY src/orchestrator/ .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--reload"]
