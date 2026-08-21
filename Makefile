# The CUDA overlay is NOT optional. This stack targets an NVIDIA card and every
# service is configured for one.
#
# It used to be applied only when `which nvidia-smi` succeeded, and silently
# dropped otherwise - which meant a failed check did not stop anything, it just
# ran the whole stack on CPU. The check fails for reasons that have nothing to
# do with the card being absent: a WSL restart before the driver shim is
# mounted, a different PATH under sudo. The symptom was a stack that came up
# fine and was ~50x slower, with nothing in any log saying why.
COMPOSE_FILE := compose/develop/docker-compose.yml -f compose/develop/docker-compose.cuda.yml
HAS_GPU := $(shell which nvidia-smi > /dev/null 2>&1 && echo yes || echo no)

ENV_FILE := compose/develop/.env
SERVICE_NAME := collector
# Services `make up` blocks on. Deliberately excludes open-webui/grafana, whose
# own healthchecks take minutes to settle and shouldn't fail a start.
CORE_SERVICES := db redis sprite-generator sprite-worker
DB_PASSWORD ?= password
DB_URL=postgresql://postgres:$(DB_PASSWORD)@127.0.0.1:5432/postgres

.PHONY: dev build stop clean logs shell up down recreate rebuild rebuild-clean rebuild-app download sync-models models gpu-check env warm smoke test-flow require-gpu

# Create compose/develop/.env from the example if it is missing. Every target
# below passes --env-file, and compose aborts outright when the file is absent.
env:
	@if [ ! -f $(ENV_FILE) ]; then \
		cp compose/develop/.env.example $(ENV_FILE); \
		echo "Created $(ENV_FILE) from .env.example — set HF_TOKEN before downloading gated models."; \
	fi

# Verify the GPU is actually reachable *from inside a container*. `nvidia-smi`
# on the host passing proves nothing: the Makefile picks the CUDA override off
# the host binary, but inference runs in a container and needs
# nvidia-container-toolkit installed on the Docker host.
gpu-check:
	@echo "Host GPU detected by Makefile: $(HAS_GPU)"
	@echo "Compose files in use: $(COMPOSE_FILE)"
	@echo "--- Running nvidia-smi inside a container ---"
	@docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi \
		|| (echo ""; \
		    echo "GPU is NOT visible to Docker. Install nvidia-container-toolkit on the"; \
		    echo "Docker host (inside WSL2 Ubuntu), then: sudo nvidia-ctk runtime configure"; \
		    echo "--runtime=docker && sudo service docker restart"; exit 1)

# Load models into VRAM out-of-band. A cold checkpoint takes minutes to fetch
# and load, which blows past any HTTP client's timeout -- including something2's
# 5 minute cap. Run this after `up`, before pointing anything at the API.
warm:
	./scripts/warm-models.sh

# Verify the API contract: GPU in use, model discovery, txt2img round-trip.
# Pass HOST=<lan-ip> to test what other machines see rather than localhost.
smoke:
	./scripts/smoke-test.sh $(HOST)

# Verify the full core -> spritesheet workflow the browser UI drives.
test-flow:
	./scripts/two-step-test.sh $(HOST)

# Refuse to start without a GPU, rather than quietly running on CPU.
require-gpu:
	@if [ "$(HAS_GPU)" != "yes" ]; then \
	  echo "nvidia-smi not found on PATH. This stack is GPU-only."; \
	  echo ""; \
	  echo "Run make from INSIDE the WSL distro, where the NVIDIA driver shim is"; \
	  echo "mounted - not from Windows PowerShell or Git Bash."; \
	  echo "If you are already inside WSL, the distro may have restarted without"; \
	  echo "the shim; confirm container access with: make gpu-check"; \
	  exit 1; \
	fi

# Start containers in the background
up: env require-gpu
	@echo "Running model downloader interactively to show progress..."
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) run --rm downloader /usr/local/bin/download_models.sh
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) up -d --remove-orphans
	@echo "Waiting for db, redis and the sprite services to report healthy..."
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) up -d --wait $(CORE_SERVICES)
	@echo "Checking/applying database migrations..."
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) exec -T sprite-generator python migrations.py

# Force a rebuild of the images and start
build: env require-gpu
	docker compose -f $(COMPOSE_FILE) build
	@echo "Running model downloader interactively to show progress..."
	docker compose -f $(COMPOSE_FILE) run --rm downloader /usr/local/bin/download_models.sh
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) build
	@echo "Running model downloader interactively to show progress..."
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) run --rm downloader /usr/local/bin/download_models.sh
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) up -d --remove-orphans
	@echo "Waiting for db, redis and the sprite services to report healthy..."
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) up -d --wait $(CORE_SERVICES)
	@echo "Checking/applying database migrations..."
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) exec -T sprite-generator python migrations.py

# Rebuild the app containers from scratch without touching images.
# Use this when sprite_generator/sprite_worker are "Running" but cannot resolve
# db/redis — `up -d` treats them as satisfied, only a recreate re-wires DNS.
recreate:
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) up -d --wait --force-recreate sprite-generator sprite-worker
	@echo "Checking/applying database migrations..."
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) exec -T sprite-generator python migrations.py

# Stop the containers
stop:
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) stop

# Down the containers (removes network/containers)
down:
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) down

# View logs for the next-app
logs:
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) logs -f ${SERVICE_NAME}

# Clean up Docker (removes unused volumes/images)
clean:
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) down -v --rmi local

# Jump inside the container shell for debugging
shell:
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) exec ${SERVICE_NAME} sh

# Force a total rebuild from scratch (no cache)
rebuild-clean: require-gpu
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) build --no-cache
	@echo "Running model downloader interactively to show progress..."
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) run --rm downloader /usr/local/bin/download_models.sh
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) up -d --remove-orphans
	@echo "Waiting for db, redis and the sprite services to report healthy..."
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) up -d --wait $(CORE_SERVICES)
	@echo "Checking/applying database migrations..."
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) exec -T sprite-generator python migrations.py

# Rebuild
rebuild: require-gpu
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) build
	@echo "Running model downloader interactively to show progress..."
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) run --rm downloader /usr/local/bin/download_models.sh
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) up -d --remove-orphans
	@echo "Waiting for db, redis and the sprite services to report healthy..."
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) up -d --wait $(CORE_SERVICES)
	@echo "Checking/applying database migrations..."
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) exec -T sprite-generator python migrations.py

# Download a new model dynamically using the background running downloader container
download:
	@if [ -z "$(repo)" ] || [ -z "$(file)" ]; then \
		echo "Usage: make download repo=<hf-repo> file=<filename>"; \
		exit 1; \
	fi
	@echo "Downloading $(file) from $(repo)..."
	docker exec -it model_downloader hf download "$(repo)" "$(file)" --local-dir /models
	@echo "$(repo) $(file)" >> compose/develop/downloader/models.txt
	@echo "Model appended to models.txt for future rebuilds."

# Show what models are declared, what is on disk, and what is actually served.
# Read-only, and the three answers routinely disagree - see the script header.
# Pass HOST=<lan-ip> to query another machine's stack.
models:
	./scripts/list-models.sh $(HOST)

# Check models.txt against the local directory and download any missing weights
sync-models:
	@echo "Checking models.txt for any missing models..."
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) run --rm downloader /usr/local/bin/download_models.sh

# Shortcut to just rebuild the specific service without cache
rebuild-app:
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) build --no-cache $(SERVICE_NAME)
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) up -d $(SERVICE_NAME)
