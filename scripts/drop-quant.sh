#!/usr/bin/env bash
# Delete one GGUF quant from the HF cache, blob and snapshot symlink together.
#
# WHY THIS IS A FILE AND NOT A ONE-LINER
# The inline version of this went wrong badly: nested quoting through
# PowerShell -> wsl.exe -> docker -> bash collapsed, `find` lost its path
# argument and its -name filter, and ran `-print -delete` against the
# container's WORKDIR - which is a bind mount of src/sprite_generator. The
# source tree survived by luck, not by design.
#
# So: explicit path, explicit guard, refuse anything that is not under the
# models cache, and never take the pattern from an unquoted glob.
#
# Usage (inside WSL, from the repo root):
#   ./scripts/drop-quant.sh qwen-image-edit-2511-Q3_K_M.gguf
set -euo pipefail

QUANT="${1:?usage: drop-quant.sh <filename.gguf>}"

# MODELS_DIR may be passed in, because the cache is written root-owned by the
# containers and deleting from it needs to run as root inside one - where the
# cache is mounted at /models, not at the host path .env records.
if [ -z "${MODELS_DIR:-}" ]; then
    ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/compose/develop/.env"
    MODELS_DIR="$(grep -E '^MODELS_DIR=' "$ENV_FILE" | cut -d= -f2-)"
fi
[ -d "$MODELS_DIR" ] || { echo "MODELS_DIR not usable: '$MODELS_DIR'" >&2; exit 1; }

case "$QUANT" in
    *.gguf) ;;
    *) echo "refusing: '$QUANT' is not a .gguf filename" >&2; exit 2 ;;
esac
case "$QUANT" in
    */*) echo "refusing: '$QUANT' must be a bare filename, not a path" >&2; exit 2 ;;
esac

# Resolve the symlink to its blob BEFORE unlinking anything.
link="$(find "$MODELS_DIR" -type l -name "$QUANT" -print -quit)"
if [ -z "$link" ]; then
    echo "not found in cache: $QUANT"
    exit 0
fi

blob="$(readlink -f "$link")"
case "$blob" in
    "$MODELS_DIR"/*) ;;
    *) echo "refusing: $link resolves outside MODELS_DIR ($blob)" >&2; exit 3 ;;
esac

size="$(du -h "$blob" | cut -f1)"
echo "symlink : $link"
echo "blob    : $blob ($size)"
rm -f "$link" "$blob"
echo "deleted $QUANT ($size reclaimed inside the VHDX)"
