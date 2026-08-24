#!/usr/bin/env bash
# Delete abandoned .incomplete blobs from the HF cache.
#
# huggingface_hub does NOT resume a killed download - the next attempt starts a
# new temp file under a different suffix and the old one is simply orphaned.
# Measured 2026-08-23: a 1.0 GB partial abandoned, the file re-fetched from
# zero, and a 3.68 GB orphan left in blobs/ beside the finished weight.
#
# Written as a file, and run as root inside a container, for the same reason as
# drop-quant.sh: an inline `find ... -delete` through
# PowerShell -> wsl -> docker -> bash loses its quoting and has already once
# aimed itself at the source tree instead of the cache.
#
# Usage (as root, cache mounted at /models):
#   docker run --rm -e MODELS_DIR=/models -v <cache>:/models \
#       -v <repo>/scripts:/s:ro --entrypoint bash <image> /s/clean-partials.sh
set -euo pipefail

MODELS_DIR="${MODELS_DIR:?set MODELS_DIR to the cache root}"
[ -d "$MODELS_DIR" ] || { echo "not a directory: $MODELS_DIR" >&2; exit 1; }

# Guard: never run against / or a path that is obviously not a model cache.
case "$MODELS_DIR" in
    /|/usr|/etc|/app|/home) echo "refusing: $MODELS_DIR" >&2; exit 2 ;;
esac

found=0
total=0
while IFS= read -r -d '' f; do
    size=$(stat -c %s "$f")
    total=$((total + size))
    found=$((found + 1))
    # awk, not bc: bc is not in the python:3.10-slim base this runs in, and its
    # absence silently printed every size as 0.0.
    printf '  %8.1f MB  %s\n' "$(awk -v b="$size" 'BEGIN{print b/1048576}')" \
        "$(basename "$f")"
    rm -f "$f"
done < <(find "$MODELS_DIR" -name '*.incomplete' -print0)

if [ "$found" -eq 0 ]; then
    echo "no partial downloads found"
else
    printf 'deleted %d partial(s), %.1f GB reclaimed inside the VHDX\n' \
        "$found" "$(awk -v b="$total" 'BEGIN{print b/1073741824}')"
fi
echo
du -sh "$MODELS_DIR"
