#!/usr/bin/env bash
# Small read-only report on job and image state.
#
# A file rather than an inline psql call: quoting SQL through
# PowerShell -> wsl -> docker -> psql loses commas and quotes, and a mangled
# query either errors or - worse - runs something other than what was intended.
#
# Usage:  docker exec -e PGPASSWORD=... stats_db bash /s/db-report.sh
#     or: ./scripts/db-report.sh   (from inside WSL, via docker exec)
set -euo pipefail

DB="${DB_NAME:-llm_monitoring}"
USER_="${DB_USER:-postgres}"

run() { psql -U "$USER_" -d "$DB" -A -t -c "$1"; }

echo "=== sprite_images ==="
run "SELECT 'live: ' || count(*) FROM sprite_images WHERE NOT deleted"
run "SELECT 'soft-deleted: ' || count(*) FROM sprite_images WHERE deleted"
run "SELECT 'cores live: ' || count(*) FROM sprite_images WHERE image_type='core' AND NOT deleted"

echo
echo "=== jobs (most recent 5) ==="
run "SELECT status || '  ' || coalesce(stage,'-') || '  ' || progress_pct || '%  ' || coalesce(progress_msg,'') FROM jobs ORDER BY created_at DESC LIMIT 5"

echo
echo "=== job counts ==="
run "SELECT status || ': ' || count(*) FROM jobs GROUP BY status ORDER BY status"

echo
echo "=== most recent failure ==="
run "SELECT left(coalesce(error,'(none)'), 900) FROM jobs WHERE status='failed' AND stage IS NOT NULL ORDER BY created_at DESC LIMIT 1"
