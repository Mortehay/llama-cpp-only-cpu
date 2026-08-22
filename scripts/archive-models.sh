#!/usr/bin/env bash
# Park model weights on D: instead of deleting them.
#
# WHY ARCHIVE RATHER THAN DELETE
# Restoring is a local copy from D: instead of a multi-GB download, which for
# Qwen3-8B is 8.2GB over the internet versus roughly a minute across drives.
#
# WHY D: IS COLD STORAGE ONLY, NEVER A LIVE MODELS_DIR
# The Windows side is reached over 9p/DrvFs. scripts/link-models.ps1 measured it
# at 44 MB/s against 3.9 GB/s on ext4 for the same read, and pointing MODELS_DIR
# at /mnt/d was measured here on 2026-08-22 at 62s to load a pipeline versus 10s
# on ext4. Archive to D:, restore to ext4 before use. Never run from D:.
#
# WHAT THIS DOES NOT DO: give space back to C:. Model files live inside the ext4
# VHDX, which grows and never shrinks. Moving them out frees space INSIDE that
# file so future downloads reuse it rather than growing it - which is worth
# doing - but C: only recovers via scripts/compact-wsl-disk.ps1 (Administrator).
#
# Usage:
#   ./archive-models.sh list
#   ./archive-models.sh archive models--thibaud--controlnet-openpose-sdxl-1.0
#   ./archive-models.sh restore models--thibaud--controlnet-openpose-sdxl-1.0
set -euo pipefail

ARCHIVE_DIR="${ARCHIVE_DIR:-/mnt/d/wsl-model-archive}"

# MODELS_DIR is the single source of truth; read it rather than hardcoding, so
# this keeps working if the path moves.
ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/compose/develop/.env"
MODELS_DIR="$(grep -E '^MODELS_DIR=' "$ENV_FILE" | cut -d= -f2-)"
if [ -z "$MODELS_DIR" ] || [ ! -d "$MODELS_DIR" ]; then
    echo "MODELS_DIR not usable from $ENV_FILE (got: '$MODELS_DIR')" >&2
    exit 1
fi

cmd="${1:-list}"
name="${2:-}"

human() { du -sh "$1" 2>/dev/null | cut -f1; }

case "$cmd" in
list)
    echo "LIVE   ($MODELS_DIR)"
    du -sh "$MODELS_DIR"/* 2>/dev/null | sort -h | sed 's/^/  /'
    echo
    echo "ARCHIVED ($ARCHIVE_DIR)"
    if [ -d "$ARCHIVE_DIR" ]; then
        du -sh "$ARCHIVE_DIR"/* 2>/dev/null | sort -h | sed 's/^/  /' || echo "  (empty)"
    else
        echo "  (no archive yet)"
    fi
    ;;

archive)
    [ -n "$name" ] || { echo "usage: $0 archive <model-dir-or-file>" >&2; exit 2; }
    src="$MODELS_DIR/$name"
    [ -e "$src" ] || { echo "not found: $src" >&2; exit 1; }
    mkdir -p "$ARCHIVE_DIR"
    echo "archiving $name ($(human "$src")) -> $ARCHIVE_DIR"
    # Copy-verify-remove rather than mv: mv across filesystems is a copy plus a
    # delete with no verification step, and a truncated copy would be discovered
    # only when the model failed to load. Run as root - some HF cache metadata
    # is written root-owned 600 by the containers and is unreadable otherwise,
    # which silently skipped 4 files during an earlier move.
    rsync -aH --info=stats2 "$src" "$ARCHIVE_DIR/" | tail -3
    s=$(find "$src" -type f | wc -l)
    d=$(find "$ARCHIVE_DIR/$name" -type f | wc -l)
    if [ "$s" != "$d" ]; then
        echo "VERIFY FAILED: $s files at source, $d at destination. Source kept." >&2
        exit 1
    fi
    rm -rf "$src"
    echo "archived and verified ($s files). Freed inside the VHDX, not on C:."
    ;;

restore)
    [ -n "$name" ] || { echo "usage: $0 restore <model-dir-or-file>" >&2; exit 2; }
    src="$ARCHIVE_DIR/$name"
    [ -e "$src" ] || { echo "not in archive: $src" >&2; exit 1; }
    echo "restoring $name ($(human "$src")) -> $MODELS_DIR"
    rsync -aH --info=stats2 "$src" "$MODELS_DIR/" | tail -3
    s=$(find "$src" -type f | wc -l)
    d=$(find "$MODELS_DIR/$name" -type f | wc -l)
    if [ "$s" != "$d" ]; then
        echo "VERIFY FAILED: $s in archive, $d restored. Archive kept." >&2
        exit 1
    fi
    rm -rf "$src"
    echo "restored and verified ($s files)."
    ;;

*)
    echo "usage: $0 {list|archive <name>|restore <name>}" >&2
    exit 2
    ;;
esac
