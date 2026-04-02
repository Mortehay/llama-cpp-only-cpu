FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed for rembg and OpenCV (which rembg uses)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY compose/develop/sprite_generator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/sprite_generator/ .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
