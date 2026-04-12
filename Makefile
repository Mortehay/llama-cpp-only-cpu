COMPOSE_FILE := compose/develop/docker-compose.yml
ENV_FILE := compose/develop/.env
SERVICE_NAME := collector
DB_PASSWORD ?= password
DB_URL=postgresql://postgres:$(DB_PASSWORD)@127.0.0.1:5432/postgres

.PHONY: dev build stop clean logs shell

# Start containers in the background
up:
	@echo "Running model downloader interactively to show progress..."
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) run --rm downloader /usr/local/bin/download_models.sh
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) up -d
	@echo "Checking/applying database migrations..."
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) exec -T sprite-generator python migrations.py

# Force a rebuild of the images and start
build:
	docker compose -f $(COMPOSE_FILE) build
	@echo "Running model downloader interactively to show progress..."
	docker compose -f $(COMPOSE_FILE) run --rm downloader /usr/local/bin/download_models.sh
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) build
	@echo "Running model downloader interactively to show progress..."
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) run --rm downloader /usr/local/bin/download_models.sh
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) up -d
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
rebuild-clean:
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) build --no-cache
	@echo "Running model downloader interactively to show progress..."
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) run --rm downloader /usr/local/bin/download_models.sh
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) up -d
	@echo "Checking/applying database migrations..."
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) exec -T sprite-generator python migrations.py

# Rebuild
rebuild:
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) build
	@echo "Running model downloader interactively to show progress..."
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) run --rm downloader /usr/local/bin/download_models.sh
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) up -d
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

# Check models.txt against the local directory and download any missing weights
sync-models:
	@echo "Checking models.txt for any missing models..."
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) run --rm downloader /usr/local/bin/download_models.sh

# Shortcut to just rebuild the specific service without cache
rebuild-app:
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) build --no-cache $(SERVICE_NAME)
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) up -d $(SERVICE_NAME)
