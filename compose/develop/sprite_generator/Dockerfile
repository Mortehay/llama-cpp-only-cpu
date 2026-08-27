FROM python:3.10-slim

WORKDIR /app

# libgl1 and libglib2.0-0 are for rembg and the OpenCV it pulls in.
#
# postgresql-client is for pg_dump, which scripts/test-apply-verdicts.py
# uses to copy production's SCHEMA (never its rows) into a throwaway
# database. Without it that test is the one command in the UI allowlist
# that cannot run where the UI runs it. The client is 17 against a
# postgres:15 server, which is the supported direction, and the test
# already strips the psql meta-commands a newer dump emits.
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    libgl1 \
    libglib2.0-0 \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY compose/develop/sprite_generator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/sprite_generator/ .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
