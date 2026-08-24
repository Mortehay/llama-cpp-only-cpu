#!/usr/bin/env bash
# Archive EVERY model in MODELS_DIR to the D: cold-storage archive.
#
# Wraps scripts/archive-models.sh, which handles one model at a time. Same
# copy-verify-remove discipline: nothing is deleted from the live directory
# until the destination file count matches.
#
# WHAT THIS FREES, AND WHAT IT DOES NOT
# Space comes back INSIDE the ext4 VHDX, so future downloads reuse it instead of
# growing the file. C: itself recovers only after
# scripts/compact-wsl-disk.ps1 (Administrator). See project-context.md.
#
# Restore one with:  ./scripts/archive-models.sh restore <name>
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_DIR="${ARCHIVE_DIR:-/mnt/d/wsl-model-archive}"

ENV_FILE="$(cd "$HERE/.." && pwd)/compose/develop/.env"
MODELS_DIR="$(grep -E '^MODELS_DIR=' "$ENV_FILE" | cut -d= -f2-)"
if [ -z "$MODELS_DIR" ] || [ ! -d "$MODELS_DIR" ]; then
    echo "MODELS_DIR not usable from $ENV_FILE (got: '$MODELS_DIR')" >&2
    exit 1
fi

mkdir -p "$ARCHIVE_DIR"

echo "MODELS_DIR : $MODELS_DIR"
echo "ARCHIVE_DIR: $ARCHIVE_DIR"
echo "free on D: : $(df -h "$ARCHIVE_DIR" | tail -1 | awk '{print $4}')"
echo

ok=0
failed=0
skipped=0

# Only the models--* HF cache trees and any bare .gguf files. `hub`, `xet`,
# `.locks` and CACHEDIR.TAG are cache plumbing that diffusers rebuilds on its
# own; moving them to DrvFs corrupts nothing but restores nothing either.
for path in "$MODELS_DIR"/models--* "$MODELS_DIR"/*.gguf; do
    [ -e "$path" ] || continue
    name="$(basename "$path")"

    if [ -e "$ARCHIVE_DIR/$name" ]; then
        echo "SKIP    $name (already in archive)"
        skipped=$((skipped + 1))
        continue
    fi

    echo "=== $name ==="
    if "$HERE/archive-models.sh" archive "$name"; then
        ok=$((ok + 1))
    else
        # Keep going. One unreadable model must not strand the other eleven.
        echo "FAILED  $name - left in place" >&2
        failed=$((failed + 1))
    fi
    echo
done

echo "======================================================================"
echo "archived: $ok   failed: $failed   skipped: $skipped"
echo
echo "LIVE remaining:"
du -sh "$MODELS_DIR"/* 2>/dev/null | sort -rh | sed 's/^/  /'
echo
echo "ARCHIVE now:"
du -sh "$ARCHIVE_DIR"/* 2>/dev/null | sort -rh | sed 's/^/  /'
echo
echo "NOTE: this freed space inside the ext4 VHDX, not on C:."
echo "      Run scripts/compact-wsl-disk.ps1 (Administrator) to return it to C:."

[ "$failed" -eq 0 ]
