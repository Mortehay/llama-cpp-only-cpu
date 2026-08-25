#!/usr/bin/env bash
# Build the React UI without installing Node on the host.
#
# There is no node/npm on this machine or in the WSL distro, and adding one
# would be a second toolchain to keep in step with the container. Instead the
# build runs in the official node image against the repo's frontend/ directory,
# the same way every other tool here runs in a container.
#
# Output goes to src/sprite_generator/static/app/, which FastAPI already serves
# as part of its /static mount - so a build is the entire deployment step.
#
# Usage:
#   bash scripts/build-frontend.sh          # install if needed, then build
#   bash scripts/build-frontend.sh install  # force a fresh npm install
#   bash scripts/build-frontend.sh dev      # run the vite dev server on :5173

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND="$REPO/frontend"
IMAGE="node:20-alpine"
MODE="${1:-build}"

if [ ! -d "$FRONTEND" ]; then
    echo "no frontend/ directory at $FRONTEND" >&2
    exit 1
fi

# The REPO is mounted, not just frontend/ - vite writes its bundle to
# `../src/sprite_generator/static/app`, and with only frontend/ mounted that
# `..` is the container root. The build then "succeeds", reports the files it
# wrote, and they disappear with --rm.
#
# npm's cache lives in a named volume rather than the bind mount: it is
# thousands of small files, and on a /mnt/c bind mount that is punishingly slow.
run_node() {
    docker run --rm \
        -v "$REPO":/work \
        -v sprite_npm_cache:/root/.npm \
        -w /work/frontend \
        "$@"
}

case "$MODE" in
install)
    echo "==> npm install (forced)"
    run_node "$IMAGE" npm install
    ;;
dev)
    echo "==> vite dev server on http://localhost:5173"
    echo "    API calls proxy to http://localhost:8001 (see vite.config.ts)"
    run_node -p 5173:5173 "$IMAGE" sh -c "npm install && npm run dev -- --host"
    exit 0
    ;;
build) ;;
*)
    echo "unknown mode: $MODE (expected build, install or dev)" >&2
    exit 2
    ;;
esac

if [ ! -d "$FRONTEND/node_modules" ]; then
    echo "==> node_modules missing, installing first"
    run_node "$IMAGE" npm install
fi

echo "==> building"
run_node "$IMAGE" npm run build

OUT="$REPO/src/sprite_generator/static/app"
if [ -f "$OUT/index.html" ]; then
    echo "==> built into $OUT"
    ls -la "$OUT" | head -10
else
    echo "build finished but $OUT/index.html is missing" >&2
    exit 1
fi
