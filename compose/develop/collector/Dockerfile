FROM python:3.12-slim
RUN pip install fastapi uvicorn requests psycopg2-binary jinja2
WORKDIR /app

# Copy from src/collector at project root
COPY src/collector/ .

# Use --reload so changes to the volume-mapped file take effect immediately
CMD ["uvicorn", "bridge:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]