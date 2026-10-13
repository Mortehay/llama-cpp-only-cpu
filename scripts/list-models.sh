#!/usr/bin/env bash
# What models does this stack have, and which of them is actually loadable?
#
#   ./scripts/list-models.sh [host]
#
# Three different answers, which routinely disagree, and the disagreement is
# the reason this script exists:
#
#   DECLARED  models.toml — what `make provision` will fetch. Editing it does
#             nothing until the provisioner runs again. (models.txt is the
#             LEGACY llm-server lane and is not shown here.)
#   CODE-REF  what the source actually asks for. Drift between this and
#             DECLARED is the bug class that let models.txt rot unnoticed, so
#             it gets its own check: `make test-models-manifest`.
#   ON DISK   the actual GGUFs under MODELS_DIR. A model deleted through the
#             orchestrator UI (DELETE /api/models/...) is gone from here while
#             still declared, and re-appears on the next `make provision`.
#   SERVED    what llama.cpp advertises right now. It scans --models-dir, so a
#             file that arrived after the container started may not be listed
#             until a restart — and `--models-max 1` means at most one of them
#             is resident in VRAM regardless of how many appear here.
#
# Read-only: nothing here downloads, deletes or loads anything.
set -uo pipefail

# Bearer auth from $SPRITE_API_KEY. Unset is fine while the API has no keys.
source "$(dirname "${BASH_SOURCE[0]}")/lib-auth.sh"

HOST="${1:-localhost}"
COLLECTOR="http://${HOST}:8002"   # llm-server publishes no host port; the
                                  # collector proxies GET /v1/models to it.
SPRITE="http://${HOST}:8001"
COMPOSE="compose/develop/docker-compose.yml -f compose/develop/docker-compose.cuda.yml"
ENV_FILE="compose/develop/.env"
MODELS_TOML="compose/develop/downloader/models.toml"

step() { echo; echo "=== $* ==="; }

step "DECLARED — $MODELS_TOML"
# Parsed with Python, not awk: the manifest has multi-line strings and awk
# would have to half-implement TOML to skip them. python3 is already required
# by `make test-models-manifest`.
python3 - "$MODELS_TOML" <<'PY' || echo "  (could not read the manifest)"
import sys, os
sys.path.insert(0, "scripts")
from models_manifest import load
doc = load(sys.argv[1])
for tier in ("required", "optional"):
    rows = [m for m in doc.get("model", []) if m.get("tier") == tier]
    for m in rows:
        what = m.get("repo") or m.get("dest", "")
        flag = " GATED" if m.get("gated") else ""
        flag += " MISSING-AT-SOURCE" if m.get("missing_on_source_host") else ""
        print(f"  {tier:<9} {m['kind']:<16} {m['name']:<26} {what}{flag}")
print(f"  {len(doc.get('dead', []))} entries are explicitly DEAD (see [[dead]])")
PY
echo "  (\`make provision\` fetches the required tier; tier=all for everything)"

step "CODE-REF vs DECLARED — drift"
# The same check `make test-models-manifest` runs, reported inline. Non-fatal
# here: this script is read-only and informational.
MODELS_DIR_ENV="$(grep -E '^MODELS_DIR=' "$ENV_FILE" 2>/dev/null | cut -d= -f2-)"
python3 scripts/check-models-manifest.py \
    ${MODELS_DIR_ENV:+--models-dir "$MODELS_DIR_ENV"} 2>&1 \
  | sed -n '/^[0-9]* advisory/,$p' | sed 's/^/  /' \
  || echo "  (drift check failed to run)"

step "ON DISK — MODELS_DIR, via the downloader container"
# Ask docker rather than reading MODELS_DIR from .env directly: the value is
# routinely an absolute WSL ext4 path or a path relative to compose/develop,
# and the container mount resolves both without this script guessing. It also
# means this reports what the SERVICES see, which is the only thing that
# matters. `run --no-deps` so it works with the stack down.
#
# No `sh` before -c: the image's ENTRYPOINT is already /bin/sh, so passing one
# runs `/bin/sh sh -c ...` and fails with "can't open 'sh'".
# `find`, not `ls /models/*.gguf`: every live GGUF is TWO levels down
# (brain-gguf/..., image-gguf/...), and the one-level glob reported only the
# dead Qwen2.5-3B at the root while claiming to list what is on disk.
if ! disk=$(docker compose -f $COMPOSE --env-file "$ENV_FILE" run --rm --no-deps \
              -T downloader -c 'find /models -maxdepth 3 -name "*.gguf" -type f \
                                  -exec ls -l {} + 2>/dev/null | sort -k9' 2>/dev/null); then
  echo "  cannot reach docker — run this from inside WSL, where the daemon lives"
elif [ -z "$(printf '%s' "$disk" | tr -d '[:space:]')" ]; then
  echo "  no .gguf files in MODELS_DIR (\`make provision\` fetches what models.toml declares)"
else
  printf '%s\n' "$disk" | awk 'NF >= 5 { printf "  %8.2f GB  %s\n", $5/1073741824, $NF }'
fi

step "SERVED — llama.cpp, via the collector at ${COLLECTOR}/v1/models"
served=$(sprite_curl -s --max-time 10 "$COLLECTOR/v1/models" || true)
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
sd=$(sprite_curl -s --max-time 10 "$SPRITE/sdapi/v1/sd-models" || true)
if echo "$sd" | grep -q '"model_name"'; then
  echo "$sd" | grep -oE '"model_name":[[:space:]]*"[^"]*"' | cut -d'"' -f4 | sed 's/^/  /'
else
  echo "  sprite-generator not answering (is the stack up? \`make up\`)"
fi
echo
