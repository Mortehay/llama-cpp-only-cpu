FROM python:3.11-slim
RUN pip install fastapi uvicorn requests psycopg2-binary
WORKDIR /app
COPY bridge.py .
CMD ["python", "bridge.py"]