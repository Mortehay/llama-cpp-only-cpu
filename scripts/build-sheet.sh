#!/usr/bin/env bash
# Drive build-sheet.py's five stages, each in its own container.
#
# WHY FIVE PROCESSES
# Loading a large quantised model and releasing it fragments the CUDA allocator
# under WSL past the point of reuse. Measured 2026-08-23: a second model load in
# the same process failed on a 30 MiB allocation with 2.23 GiB free. Splitting
# turnaround from actions was not enough - encode and denoise each load their
# own model, so they need separate processes too. Embeddings are persisted
# between them as .pt files.
#
# Each stage is resumable: re-run from wherever it failed without redoing the
# GPU work before it.
#
# Usage:
#   ./scripts/build-sheet.sh <concept.png> <out.png> [directions] [actions] [frames]
set -euo pipefail

CONCEPT="${1:?usage: build-sheet.sh <concept.png> <out.png> [dirs] [actions] [frames]}"
OUT="${2:?missing output path}"
DIRS="${3:-s,e,n,w}"
ACTS="${4:-walk}"
FRAMES="${5:-4}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE="docker compose -f compose/develop/docker-compose.yml -f compose/develop/docker-compose.cuda.yml --env-file compose/develop/.env"
RUN="$COMPOSE run --rm --no-deps --entrypoint python sprite-worker"

cd "$HERE"

# llama.cpp holds the same 12GB card and stats_collector polls it, which can
# trigger a model load mid-run. Stop them rather than race them.
echo "==> stopping GPU-contending services"
docker stop llm_engine stats_collector model_orchestrator 2>/dev/null || true

run_stage() {
    echo "==> $*"
    # shellcheck disable=SC2086
    $RUN /app/scripts/build-sheet.py "$@"
}

run_stage turnaround-encode "/app/$CONCEPT" --directions "$DIRS"
run_stage turnaround-denoise
run_stage actions-encode --actions "$ACTS" --frames "$FRAMES"
run_stage actions-denoise
run_stage compose "/app/$OUT"

echo "==> verifying"
$RUN /app/scripts/check-sprite.py "/app/$OUT" --grid "$FRAMES"x"$(python3 - <<EOF
print(len("$DIRS".split(",")) * len("$ACTS".split(",")))
EOF
)"
