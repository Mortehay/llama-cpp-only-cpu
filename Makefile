# Variables to keep things clean
COMPOSE_FILE := compose/develop/docker-compose.yml
SERVICE_NAME := collector
DB_PASSWORD ?= password
DB_URL=postgresql://postgres:$(DB_PASSWORD)@127.0.0.1:5432/postgres

.PHONY: dev build stop clean logs shell

# Start containers in the background
dev:
	docker compose -f $(COMPOSE_FILE) up -d

# Force a rebuild of the images and start
build:
	docker compose -f $(COMPOSE_FILE) up -d --build

# Stop the containers
stop:
	docker compose -f $(COMPOSE_FILE) stop

# Down the containers (removes network/containers)
down:
	docker compose -f $(COMPOSE_FILE) down

# View logs for the next-app
logs:
	docker compose -f $(COMPOSE_FILE) logs -f ${SERVICE_NAME}

# Clean up Docker (removes unused volumes/images)
clean:
	docker compose -f $(COMPOSE_FILE) down -v --rmi local

# Jump inside the container shell for debugging
shell:
	docker compose -f $(COMPOSE_FILE) exec ${SERVICE_NAME} sh

# Force a total rebuild from scratch (no cache)
rebuild-clean:
	docker compose -f $(COMPOSE_FILE) build --no-cache
	docker compose -f $(COMPOSE_FILE) up -d

# Rebuild
rebuild:
	docker compose -f $(COMPOSE_FILE) build
	docker compose -f $(COMPOSE_FILE) up -d

# Shortcut to just rebuild the specific service without cache
rebuild-app:
	docker compose -f $(COMPOSE_FILE) build --no-cache $(SERVICE_NAME)
	docker compose -f $(COMPOSE_FILE) up -d $(SERVICE_NAME)
