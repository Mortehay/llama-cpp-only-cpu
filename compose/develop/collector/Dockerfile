FROM python:3.12-slim
RUN pip install fastapi uvicorn requests psycopg2-binary
WORKDIR /app

# We still copy the file for the initial image build
COPY collector/ .

# Use --reload so changes to the volume-mapped file take effect immediately
CMD ["uvicorn", "bridge.py:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]