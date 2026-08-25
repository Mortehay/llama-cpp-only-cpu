#!/usr/bin/env bash
# Emulate a something2 provider call and validate the response against ITS rules.
#
# This does not talk to something2. It replays the request its admin panel would
# send (per its docs/ai-providers.md) and checks the reply the way its backend
# would parse it, so a contract mismatch surfaces here instead of inside their
# admin UI with an opaque error.
#
#   ./scripts/verify-something2-contract.sh [host] [columns] [rows]
#
# Use the LAN IP something2 will actually use, not localhost — reachability is
# half of what this verifies.
set -uo pipefail

# Bearer auth from $SPRITE_API_KEY. Unset is fine while the API has no keys.
source "$(dirname "${BASH_SOURCE[0]}")/lib-auth.sh"

HOST="${1:-localhost}"
COLS="${2:-4}"
ROWS="${3:-1}"
BASE="http://${HOST}:8001"
FAILED=0

pass() { echo "  PASS  $*"; }
fail() { echo "  FAIL  $*"; FAILED=1; }
step() { echo; echo "=== $* ==="; }

step "1. Models discovery  (models_path=/sdapi/v1/sd-models, pointer=\$[*].model_name)"
models_json=$(sprite_curl -s --max-time 20 "$BASE/sdapi/v1/sd-models")
if [ -z "$models_json" ]; then
    fail "no response — is the stack up and reachable at $HOST?"
    echo "  (if HOST is a LAN IP, check scripts/lan-expose.ps1 has been run elevated)"
    exit 1
fi
# their pointer selects names, not objects: a bare array whose entries have model_name
if echo "$models_json" | head -c1 | grep -q '\['; then
    pass "response is a JSON array (their pointer expects \$[*])"
else
    fail "response is not a top-level array — \$[*].model_name will not resolve"
fi
MODEL=$(echo "$models_json" | grep -oE '"model_name":[[:space:]]*"[^"]*"' | head -1 | cut -d'"' -f4)
if [ -n "$MODEL" ]; then pass "model_name resolves -> $MODEL"; else fail "no model_name field"; exit 1; fi

step "2. Generate  (their A1111 template, quoted numeric placeholders)"
FRAMES=$((COLS * ROWS))
# NOTE: width/height/seed are sent as STRINGS. That is how something2
# substitutes {{width}} etc., and a provider that 422s on it is broken for them.
req=$(cat <<JSON
{
  "prompt": "green zombie, tattered clothes, pixel art sprite",
  "negative_prompt": "",
  "steps": "4",
  "cfg_scale": "0",
  "width": "512",
  "height": "512",
  "seed": "-1",
  "frames": "${FRAMES}",
  "override_settings": { "sd_model_checkpoint": "${MODEL}" }
}
JSON
)
t0=$(date +%s)
resp=$(sprite_curl -s --max-time 600 -X POST "$BASE/sdapi/v1/txt2img" \
        -H 'Content-Type: application/json' -d "$req")
elapsed=$(( $(date +%s) - t0 ))

if echo "$resp" | grep -q '"images"'; then
    pass "responded in ${elapsed}s with an images array"
else
    fail "no images array: $(echo "$resp" | head -c 300)"; exit 1
fi

# their AI_PROVIDER_GENERATE_TIMEOUT_MS defaults to 5 minutes
if [ "$elapsed" -lt 300 ]; then
    pass "inside their 5 minute provider timeout (${elapsed}s)"
else
    fail "took ${elapsed}s — exceeds their 300s default; they will time out"
fi

step "3. Image pointer  (images[0]) and their content rules"
echo "$resp" > /tmp/s2resp.json
python3 - "$COLS" "$ROWS" <<'PY'
import base64, json, sys, io
cols, rows = int(sys.argv[1]), int(sys.argv[2])
d = json.load(open('/tmp/s2resp.json'))
raw = base64.b64decode(d["images"][0])          # their image_pointer
print(f"  PASS  images[0] decoded ({len(raw)} bytes)")

if raw[:8] == b'\x89PNG\r\n\x1a\n':
    print("  PASS  is a PNG")
else:
    print("  FAIL  not a PNG"); sys.exit(1)

if len(raw) <= 32 * 1024 * 1024:
    print(f"  PASS  under their 32MB cap ({len(raw)/1024/1024:.2f} MB)")
else:
    print("  FAIL  exceeds their 32MB cap"); sys.exit(1)

try:
    from PIL import Image
    im = Image.open(io.BytesIO(raw)); w, h = im.size
except ImportError:
    # PIL is not installed on the WSL host, only inside the containers. Read the
    # PNG header directly rather than skipping: grid divisibility is the single
    # most common reason something2 rejects a generated sheet, so this check has
    # to run. IHDR width/height are big-endian uint32 at bytes 16..24.
    import struct
    w, h = struct.unpack(">II", raw[16:24])
    print("  info  PIL unavailable; read dimensions from the PNG header")

print(f"  info  image is {w}x{h}, declared grid {cols}x{rows}")
ok = True
if w % cols:
    print(f"  FAIL  width {w} does not divide evenly into {cols} columns"); ok = False
if h % rows:
    print(f"  FAIL  height {h} does not divide evenly into {rows} rows"); ok = False
if ok:
    print(f"  PASS  divides evenly -> frames of {w//cols}x{h//rows}")
sys.exit(0 if ok else 1)
PY
[ $? -ne 0 ] && FAILED=1

echo
if [ "$FAILED" = "0" ]; then
    cat <<SUMMARY
CONTRACT OK. Register in something2's admin with:

  Base URL        http://${HOST}:8001/sdapi/v1/txt2img
  Models path     /sdapi/v1/sd-models
  Models pointer  \$[*].model_name
  Image pointer   images[0]
  Sprite sheet    flat, columns=${COLS}, rows=${ROWS}

Full request template: .ai/specs/something2-provider/contract.md
SUMMARY
else
    echo "CONTRACT MISMATCH — fix before touching something2's admin."
    exit 1
fi
