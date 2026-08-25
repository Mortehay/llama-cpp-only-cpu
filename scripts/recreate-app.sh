#!/usr/bin/env bash
# Recreate the two app containers so a compose ENVIRONMENT change takes effect.
#
# `docker restart` does NOT do this. It restarts the process inside the existing
# container, which keeps the environment it was created with - so an added
# variable (PYTHONPATH, a new token) appears to have no effect and the change
# looks broken rather than unapplied.
#
# Scoped to sprite-generator and sprite-worker on purpose: a bare `up -d` would
# also touch the database and the model downloader, which have no reason to
# restart because the app's environment changed.
#
# Usage:
#   bash scripts/recreate-app.sh          # validate, then recreate
#   bash scripts/recreate-app.sh --check  # validate only, change nothing

set -euo pipefail

# Bearer auth from $SPRITE_API_KEY. Unset is fine while the API has no keys.
source "$(dirname "${BASH_SOURCE[0]}")/lib-auth.sh"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

COMPOSE=(-f compose/develop/docker-compose.yml
         -f compose/develop/docker-compose.cuda.yml
         --env-file compose/develop/.env)
SERVICES=(sprite-generator sprite-worker)

echo "==> validating compose config"
if ! docker compose "${COMPOSE[@]}" config >/dev/null; then
    echo "compose config is invalid - not touching anything" >&2
    exit 1
fi

# Show what the app services will actually receive, so a typo in the env block
# is visible BEFORE the containers are replaced.
echo "==> PYTHONPATH as compose resolves it:"
docker compose "${COMPOSE[@]}" config \
    | grep -E "^\s+(sprite-generator|sprite-worker):|PYTHONPATH" || true

if [ "${1:-}" = "--check" ]; then
    echo "==> --check given, stopping here"
    exit 0
fi

echo "==> recreating ${SERVICES[*]}"
docker compose "${COMPOSE[@]}" up -d --no-deps "${SERVICES[@]}"

echo "==> waiting for the API"
for _ in $(seq 1 60); do
    code=$(sprite_curl -s -o /dev/null -w "%{http_code}" http://localhost:8001/api/auth/mode || true)
    if [ "$code" = "200" ]; then
        echo "    API is up"
        break
    fi
    sleep 2
done

echo "==> environment actually seen by the worker:"
docker exec sprite_worker printenv PYTHONPATH || echo "    PYTHONPATH is UNSET"
