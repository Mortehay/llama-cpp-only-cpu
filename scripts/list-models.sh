#!/usr/bin/env bash
# What models does this stack have, and which of them is actually loadable?
#
#   ./scripts/list-models.sh [host]
#
# Three different answers, which routinely disagree, and the disagreement is
# the reason this script exists:
#
#   DECLARED  models.txt — what `make up` will download. Editing this file does
#             nothing until the downloader runs again (`make sync-models`).
#   ON DISK   the actual GGUFs under MODELS_DIR. A model deleted through the
#             orchestrator UI (DELETE /api/models/...) is gone from here while
#             still listed in models.txt, and re-downloads on the next `make up`.
#   SERVED    what llama.cpp advertises right now. It scans --models-dir, so a
#             file that arrived after the container started may not be listed
#             until a restart — and `--models-max 1` means at most one of them
#             is resident in VRAM regardless of how many appear here.
#
# Read-only: nothing here downloads, deletes or loads anything.
set -uo pipefail

HOST="${1:-localhost}"
COLLECTOR="http://${HOST}:8002"   # llm-server publishes no host port; the
                                  # collector proxies GET /v1/models to it.
SPRITE="http://${HOST}:8001"
COMPOSE="compose/develop/docker-compose.yml -f compose/develop/docker-compose.cuda.yml"
ENV_FILE="compose/develop/.env"
MODELS_TXT="compose/develop/downloader/models.txt"

step() { echo; echo "=== $* ==="; }

step "DECLARED — $MODELS_TXT"
# Strip CR: the file is edited on Windows and checked out with CRLF (see the
# same defence in download_models.sh).
tr -d '\r' < "$MODELS_TXT" | awk '
  /^[[:space:]]*#/ { next }
  NF >= 2 { printf "  active      %-52s %s\n", $2, $1 }
'
tr -d '\r' < "$MODELS_TXT" | awk '
  /^#[[:space:]]+[0-9.]+ GB/ { printf "  commented   %-52s %s\n", $5, $4 }
'
echo "  (uncomment a line, then \`make sync-models\` to fetch it)"

step "ON DISK — MODELS_DIR, via the downloader container"
# Ask docker rather than reading MODELS_DIR from .env directly: the value is
# routinely an absolute WSL ext4 path or a path relative to compose/develop,
# and the container mount resolves both without this script guessing. It also
# means this reports what the SERVICES see, which is the only thing that
# matters. `run --no-deps` so it works with the stack down.
#
# No `sh` before -c: the image's ENTRYPOINT is already /bin/sh, so passing one
# runs `/bin/sh sh -c ...` and fails with "can't open 'sh'".
if ! disk=$(docker compose -f $COMPOSE --env-file "$ENV_FILE" run --rm --no-deps \
              -T downloader -c 'ls -l /models/*.gguf 2>/dev/null' 2>/dev/null); then
  echo "  cannot reach docker — run this from inside WSL, where the daemon lives"
elif [ -z "$(printf '%s' "$disk" | tr -d '[:space:]')" ]; then
  echo "  no .gguf files in MODELS_DIR (\`make sync-models\` downloads what models.txt declares)"
else
  printf '%s\n' "$disk" | awk 'NF >= 5 { printf "  %8.2f GB  %s\n", $5/1073741824, $NF }'
fi

step "SERVED — llama.cpp, via the collector at ${COLLECTOR}/v1/models"
served=$(curl -s --max-time 10 "$COLLECTOR/v1/models" || true)
if echo "$served" | grep -q '"id"'; then
  echo "$served" | grep -oE '"id":[[:space:]]*"[^"]*"' | cut -d'"' -f4 | sed 's/^/  /'
  echo "  (--models-max 1: only one is resident; --sleep-idle-seconds 120"
  echo "   releases the card so sprite generation can use it)"
else
  echo "  llm-server not answering (is the stack up? \`make up\`)"
fi

step "IMAGE CHECKPOINTS — sprite-generator at ${SPRITE}/sdapi/v1/sd-models"
# Not chat models and not in models.txt: diffusers pulls these itself on first
# use. Listed here because "what models does this thing have" otherwise gets a
# misleading answer.
sd=$(curl -s --max-time 10 "$SPRITE/sdapi/v1/sd-models" || true)
if echo "$sd" | grep -q '"model_name"'; then
  echo "$sd" | grep -oE '"model_name":[[:space:]]*"[^"]*"' | cut -d'"' -f4 | sed 's/^/  /'
else
  echo "  sprite-generator not answering (is the stack up? \`make up\`)"
fi
echo
