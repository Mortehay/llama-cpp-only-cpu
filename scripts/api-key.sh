#!/usr/bin/env bash
# Create or rotate the CLIENT key the scripts use, and keep it in .env.
#
#   make api-key                         # create if none, else rotate
#   make api-key NAME=something2-audio SCOPES=read,generate
#   make api-key FRESH=1                 # a brand-new key even if one exists
#
# Writes SPRITE_API_KEY in compose/develop/.env (gitignored). That variable is
# read by scripts only - verify-audio-api.py, build-audio-corpus.py,
# lib-auth.sh - and is NOT passed to any container. It is a different thing
# from SPRITE_API_TOKEN, the server-side legacy master token in auth.py; this
# script never touches that one.
#
# Rotation keeps the key's name and scopes and revokes the old secret in the
# same transaction (auth.rotate_key). The token never appears in argv: the old
# one goes to mint-key.py on stdin, the new one comes back on stdout and goes
# straight into the file. Only its prefix is printed.
set -euo pipefail

ENV_FILE="${ENV_FILE:-compose/develop/.env}"
NAME="${NAME:-scripts}"
SCOPES="${SCOPES:-read,generate}"
CONTAINER="${CONTAINER:-sprite_generator}"

[ -f "$ENV_FILE" ] || { echo "no $ENV_FILE - run 'make env' first" >&2; exit 1; }
current="$(grep -E '^SPRITE_API_KEY=' "$ENV_FILE" | tail -1 | cut -d= -f2- || true)"

new=""
if [ -n "$current" ] && [ -z "${FRESH:-}" ]; then
    if new="$(printf '%s' "$current" | docker exec -i "$CONTAINER" \
               python /app/scripts/mint-key.py --rotate-token --quiet 2>/dev/null)"; then
        echo "rotated the key in $ENV_FILE (old secret revoked; name and scopes kept)"
    else
        echo "the key in $ENV_FILE is not active (revoked or unknown) - creating a new one"
        new=""
    fi
fi
if [ -z "$new" ]; then
    new="$(docker exec "$CONTAINER" python /app/scripts/mint-key.py \
             --name "$NAME" --scopes "$SCOPES" --quiet)"
    echo "created key '$NAME' with scopes $SCOPES"
fi
case "$new" in sk_*) ;; *) echo "mint-key.py did not return a token" >&2; exit 1 ;; esac

tmp="$(mktemp)"
if grep -qE '^SPRITE_API_KEY=' "$ENV_FILE"; then
    # Replace in place, keeping every other line (and its order) untouched.
    awk -v t="$new" '/^SPRITE_API_KEY=/{print "SPRITE_API_KEY=" t; next} {print}' \
        "$ENV_FILE" > "$tmp"
else
    { cat "$ENV_FILE"; [ -n "$(tail -c1 "$ENV_FILE")" ] && echo
      echo "# CLIENT key for scripts; not passed to containers. See scripts/api-key.sh."
      echo "SPRITE_API_KEY=$new"; } > "$tmp"
fi
cat "$tmp" > "$ENV_FILE" && rm -f "$tmp"
echo "SPRITE_API_KEY=${new:0:12}... written to $ENV_FILE"
