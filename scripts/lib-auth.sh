# Shared bearer auth for the curl-based helper scripts. Source it, then call
# `sprite_curl` wherever you would have called `curl`.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/lib-auth.sh"
#   resp=$(sprite_curl -s "$BASE/api/cores")
#
# WHY
#
# These scripts predate API keys and called the API unauthenticated, which
# worked because the API was open. It is not open any more: `auth.require` now
# covers every route that spends GPU time or enumerates images, so the first
# minted key turns all of them into 401s at once.
#
# The token is read from the environment rather than a flag so that a script
# called from another script does not have to thread it through:
#
#   export SPRITE_API_KEY=sk_...
#   bash scripts/smoke-test.sh
#
# UNSET IS DELIBERATELY FINE. While the API has no keys it is in open mode and
# answers anyway, so a fresh clone runs these with no setup. That mirrors
# `auth.require` itself - the scripts are open exactly as long as the API is,
# and stop being open at the same moment it does.
#
# `SPRITE_API_TOKEN` is accepted as a fallback for the legacy shared secret,
# which `auth.py` still honours as a valid credential.

sprite_curl() {
    local key="${SPRITE_API_KEY:-${SPRITE_API_TOKEN:-}}"
    if [ -n "$key" ]; then
        curl -H "Authorization: Bearer ${key}" "$@"
    else
        curl "$@"
    fi
}
