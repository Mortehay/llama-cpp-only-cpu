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

.PHONY: dev build stop clean logs shell up down recreate rebuild rebuild-clean rebuild-app download sync-models models gpu-check env warm smoke test-flow require-gpu fetch-qwen turnaround pixelate check-sprite smoke-sheet sheet8

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

# --- 2D conveyor (ADR 0005) ---------------------------------------------
#
# These run inside sprite-worker rather than on the host: PIL, numpy and
# diffusers are in the image and nowhere else on this machine.

# Fetch the Qwen-Image-Edit-2511 stack, assembled from two publishers so it fits
# a 12GB card. ~16.5GB. See scripts/fetch-qwen-edit.py for why it is not a
# single snapshot_download.
#
# --models-dir is passed EXPLICITLY and must stay that way. --env-file puts the
# host's MODELS_DIR into the container, where it names a path that is not the
# bind mount, and the download then lands in a layer `--rm` discards.
fetch-qwen:
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) run --rm \
		--entrypoint python sprite-worker \
		/app/scripts/fetch-qwen-edit.py --models-dir /models

# One concept image -> 8 directions. ~33s per direction plus a ~90s model load.
# core=<path> is any images/core_*.png; dirs=s,e limits it for a quick check.
#
# Verified working 2026-08-23 on the 3060: Q2_K resident on the GPU, no offload.
# See .ai/decisions/0005 for why offloading is the wrong call on this machine.
turnaround:
	@if [ -z "$(core)" ]; then echo "Usage: make turnaround core=images/core_XXXX.png [dirs=s,e] [size=512]"; exit 1; fi
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) run --rm \
		--entrypoint python sprite-worker \
		/app/qwen_edit.py --selftest --image /app/$(core) \
		--size $(or $(size),512) \
		$(if $(dirs),--directions $(dirs),) \
		--out /app/images/_turnaround.png

# The whole conveyor on one character: turnaround -> key -> pixelate -> sheet.
# Produces a transparent, palette-locked 8-direction sheet and checks it.
sheet8:
	@if [ -z "$(core)" ]; then echo "Usage: make sheet8 core=images/core_XXXX.png"; exit 1; fi
	$(MAKE) turnaround core=$(core)
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) run --rm \
		--entrypoint python sprite-worker /app/pixelate.py \
		/app/images/_turnaround.png /app/images/_sheet8.png \
		--grid 8x1 --cell 48x64 --colors 24 --key --key-tolerance 10 \
		--preview-scale 6
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) run --rm \
		--entrypoint python sprite-worker /app/scripts/check-sprite.py \
		/app/images/_sheet8.png --grid 8x1

# Pixelate anything into a palette-locked, hard-alpha sheet. No GPU, no model.
#   make pixelate src=images/foo.png grid=4x2 cell=48x48
pixelate:
	@if [ -z "$(src)" ]; then echo "Usage: make pixelate src=<png> [grid=4x2] [cell=48x48]"; exit 1; fi
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) run --rm \
		--entrypoint python sprite-worker \
		/app/pixelate.py /app/$(src) /app/$(basename $(src))_px.png \
		--grid $(or $(grid),1x1) --cell $(or $(cell),48x48) --key --preview-scale 6

# Assert a finished sheet really is transparent, palette-locked pixel art.
# A viewer composites RGBA over white, so an unkeyed sheet LOOKS correct.
check-sprite:
	@if [ -z "$(src)" ]; then echo "Usage: make check-sprite src=<png> [grid=4x2]"; exit 1; fi
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) run --rm \
		--entrypoint python sprite-worker \
		/app/scripts/check-sprite.py /app/$(src) --grid $(or $(grid),1x1)

# Composition + pixelation under test without any model involved.
smoke-sheet:
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) run --rm \
		--entrypoint python sprite-worker \
		/app/scripts/smoke-sheet.py images/sheet_3025b822691a.png images/_smoke_sheet.png

# Shortcut to just rebuild the specific service without cache
rebuild-app:
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) build --no-cache $(SERVICE_NAME)
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) up -d $(SERVICE_NAME)
