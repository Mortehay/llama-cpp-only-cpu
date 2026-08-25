#!/usr/bin/env bash
# Exercise the full core -> spritesheet workflow through the real API.
#
# This is the path the browser UI drives, and it covers everything the A1111
# façade does not: prompt shaping, background removal, img2img, frame slicing
# and vertical stitching.
set -uo pipefail

# Bearer auth from $SPRITE_API_KEY. Unset is fine while the API has no keys.
source "$(dirname "${BASH_SOURCE[0]}")/lib-auth.sh"

B="http://${1:-localhost}:8001"
CORE_MODEL="stabilityai/sdxl-turbo"
SHEET_MODEL="Onodofthenorth/SD_PixelArt_SpriteSheet_Generator"
FAILED=0

# The daemon lives inside WSL, so `docker` is absent (or points at a dead
# Docker Desktop pipe) when this script is run from Git Bash on the Windows
# side. Reach it through wsl.exe instead of failing, because the step that
# needs it is the pixel inspection — the only part of this test that can tell
# a real sprite sheet from a blank one.
if docker info >/dev/null 2>&1; then
    DOCKER="docker"
elif command -v wsl.exe >/dev/null 2>&1 && wsl.exe -- docker info >/dev/null 2>&1; then
    DOCKER="wsl.exe -- docker"
else
    echo "No usable docker client; cannot inspect output. Run this inside WSL."
    exit 1
fi

step() { echo; echo "=== $* ==="; }
pass() { echo "  PASS  $*"; }
fail() { echo "  FAIL  $*"; FAILED=1; }

# Poll a celery task id until it leaves PENDING/PROGRESS. $2 = timeout seconds.
wait_task() {
    local tid="$1" limit="${2:-900}" waited=0 status=""
    while [ "$waited" -lt "$limit" ]; do
        # head -1: the response carries two "status" keys — Celery's, and the
        # one inside the task's own result dict. Matching both yields
        # "SUCCESS\nok", which matches no case branch and polls until timeout.
        status=$(sprite_curl -s --max-time 20 "$B/api/task-status/$tid" \
                 | grep -oE '"status"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | cut -d'"' -f4)
        case "$status" in
            SUCCESS|FAILURE|REVOKED) echo "$status"; return 0 ;;
        esac
        sleep 10; waited=$((waited+10))
    done
    echo "TIMEOUT"; return 1
}

step "1. Generate core sprite (step 1)"
resp=$(sprite_curl -s --max-time 60 -X POST "$B/api/generate_core" \
        -F "prompt=green zombie, tattered clothes" -F "llm_name=$CORE_MODEL")
core_task=$(echo "$resp" | grep -oE '"task_id"[[:space:]]*:[[:space:]]*"[^"]*"' | cut -d'"' -f4)
if [ -z "$core_task" ]; then
    fail "no task_id returned: $resp"; exit 1
fi
echo "  queued $core_task; waiting (cold model load can take minutes)"
st=$(wait_task "$core_task" 1200)
[ "$st" = "SUCCESS" ] && pass "core generated" || fail "core task ended $st"

step "2. Locate the generated core"
cores=$(sprite_curl -s --max-time 20 "$B/api/cores")
core_id=$(echo "$cores" | grep -oE '"id"[[:space:]]*:[[:space:]]*[0-9]+' | head -1 | grep -oE '[0-9]+')
if [ -n "$core_id" ]; then pass "core id=$core_id"; else fail "no cores returned: ${cores:0:200}"; exit 1; fi

step "3. Generate spritesheet (step 2) - the rewired img2img path"
resp=$(sprite_curl -s --max-time 60 -X POST "$B/api/generate_sheet" \
        -F "parent_id=$core_id" \
        -F 'actions=["move right","idle"]' \
        -F "llm_name=$SHEET_MODEL" \
        -F "width=128" -F "height=128" -F "motion_steps=4")
sheet_task=$(echo "$resp" | grep -oE '"task_id"[[:space:]]*:[[:space:]]*"[^"]*"' | cut -d'"' -f4)
if [ -z "$sheet_task" ]; then fail "no task_id: $resp"; exit 1; fi
echo "  queued $sheet_task; 2 actions x 4 frames"
st=$(wait_task "$sheet_task" 1800)
[ "$st" = "SUCCESS" ] && pass "spritesheet generated" || fail "sheet task ended $st"

step "4. Inspect the produced images"
# Status alone is not evidence. SD1.5's safety checker returns a solid BLACK
# image instead of raising, so a blank sheet completes "successfully" and every
# downstream step processes it happily. An earlier version of this script
# checked only task status and reported PASS on a 1-colour, 1142-byte sheet.
# Verify pixels.
$DOCKER exec -i sprite_worker python - <<'PY'
import glob, os, sys
import numpy as np
from PIL import Image

bad = 0
for pattern, label in (("core_*.png", "core"), ("sheet_*.png", "sheet")):
    files = sorted(glob.glob("/app/images/" + pattern), key=os.path.getmtime)
    if not files:
        print(f"  FAIL  no {label} image found"); bad = 1; continue
    path = files[-1]
    im = Image.open(path)
    a = np.array(im.convert("RGBA"))
    colours = len(np.unique(a[:, :, :3].reshape(-1, 3), axis=0))
    transparent = 100.0 * (a[:, :, 3] == 0).mean()
    print(f"  {label}: {os.path.basename(path)} {im.size} "
          f"| {colours} colours | {transparent:.1f}% transparent")
    if colours < 16:
        print(f"  FAIL  {label} has {colours} unique colours - it is effectively blank.")
        print("        Usual cause: the SD1.5 safety checker replaced it with a")
        print("        black image. Check `docker logs sprite_worker` for 'NSFW'.")
        bad = 1
    elif transparent < 5.0:
        # A fully opaque sheet is not a sprite sheet. This caught a real
        # regression that the colour check waved through: the core was handed
        # to img2img with a transparent background, .convert("RGB") turned that
        # black, and the background key then refused to touch a dark corner. The
        # result was a colourful, detailed, completely unusable 100%-opaque
        # sheet with a black backdrop.
        print(f"  FAIL  {label} is only {transparent:.1f}% transparent - the")
        print("        background was not removed. A sprite needs an alpha hole")
        print("        around it. Check `docker logs sprite_worker` for 'Safety")
        print("        trigger' or 'BG removal failed'.")
        bad = 1
    else:
        print(f"  PASS  {label} contains real image data on a transparent background")
sys.exit(bad)
PY
[ $? -ne 0 ] && FAILED=1

echo
if [ "$FAILED" = "0" ]; then
    echo "TWO-STEP FLOW PASSED. Sheet is 512x256 (4 frames x 2 actions) with real content."
else
    echo "TWO-STEP FLOW FAILED - see FAIL lines and \`docker logs sprite_worker\`."
    exit 1
fi
