#!/usr/bin/env bash
# End-to-end verification of the sprite stack. Run from inside WSL:
#   ./scripts/smoke-test.sh [host]
# Default host is localhost; pass a LAN IP to test what other machines see.
#
# Exits non-zero on the first hard failure so it is usable in CI or a Makefile.
set -uo pipefail

HOST="${1:-localhost}"
BASE="http://${HOST}:8001"
OUT_DIR="${TMPDIR:-/tmp}/sprite-smoke"
mkdir -p "$OUT_DIR"

pass() { echo "  PASS  $*"; }
fail() { echo "  FAIL  $*"; FAILED=1; }
step() { echo; echo "=== $* ==="; }
FAILED=0

step "1. API reachable at ${BASE}"
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$BASE/" || echo 000)
if [ "$code" = "200" ]; then pass "GET / -> 200"; else
  fail "GET / -> $code (is the stack up? \`make up\`)"
  echo "Cannot continue without the API."; exit 1
fi

step "2. Worker compute device"
info=$(curl -s --max-time 20 "$BASE/api/compute-info")
device=$(echo "$info" | grep -oE '"device"[[:space:]]*:[[:space:]]*"[^"]*"' | cut -d'"' -f4)
case "$device" in
  cuda)
    pass "worker on CUDA"
    echo "$info" | tr ',' '\n' | grep -E 'gpu_name|vram_total_gb|torch_version|torch_cuda_build' | sed 's/^/        /'
    ;;
  cpu)
    # Not a warning. On a GPU host this means the pivot is not working, and
    # everything downstream will still "succeed" -- just 50x slower.
    fail "worker fell back to CPU — run \`make gpu-check\`, verify COMPUTE_DEVICE=cuda"
    ;;
  *)
    fail "could not read device from /api/compute-info: $info"
    ;;
esac

step "3. A1111 model discovery"
models=$(curl -s --max-time 15 "$BASE/sdapi/v1/sd-models")
if echo "$models" | grep -q '"model_name"'; then
  pass "sd-models returns model_name entries"
  echo "$models" | grep -oE '"model_name":[[:space:]]*"[^"]*"' | cut -d'"' -f4 | sed 's/^/        /'
else
  fail "sd-models missing model_name (something2's \$[*].model_name pointer would fail): $models"
fi

step "4. txt2img round-trip (blocking; first run also downloads the model)"
started=$(date +%s)
resp=$(curl -s --max-time 900 -X POST "$BASE/sdapi/v1/txt2img" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"green zombie, pixel art sprite","width":"512","height":"512","seed":"12345",
       "override_settings":{"sd_model_checkpoint":"stabilityai/sdxl-turbo"}}')
elapsed=$(( $(date +%s) - started ))

if echo "$resp" | grep -q '"images"'; then
  pass "txt2img returned an image in ${elapsed}s"
  echo "$resp" | grep -oE '"info":"[^"]*"' | sed 's/^/        /' | head -1
else
  fail "txt2img failed after ${elapsed}s: $(echo "$resp" | head -c 400)"
fi

step "5. Decode and sanity-check the PNG"
# Guards the SDXL fp16 VAE trap: it decodes to pure black with no error, which
# looks like a bad prompt rather than a numeric overflow. A single-colour image
# is therefore a FAILURE, not a curiosity.
echo "$resp" > "$OUT_DIR/resp.json"
python3 - "$OUT_DIR/resp.json" "$OUT_DIR/out.png" <<'PY'
import base64, json, sys
try:
    data = json.load(open(sys.argv[1]))
    raw = base64.b64decode(data["images"][0])
    open(sys.argv[2], "wb").write(raw)
    print(f"  PASS  decoded {len(raw)} bytes -> {sys.argv[2]}")
    if len(raw) < 1000:
        print("  FAIL  PNG is suspiciously small"); sys.exit(1)
except Exception as e:
    print(f"  FAIL  could not decode image: {e}"); sys.exit(1)
PY
[ $? -ne 0 ] && FAILED=1

echo
if [ "$FAILED" = "0" ]; then
  echo "ALL CHECKS PASSED. Image at $OUT_DIR/out.png"
  echo "Open it and confirm it is not a flat colour — a black or grey square means"
  echo "the VAE overflowed, which no status code will tell you."
else
  echo "SOME CHECKS FAILED (see FAIL lines above)."
  exit 1
fi
