#!/usr/bin/env bash
# Pre-download checkpoints so their fetch time is not charged to a request.
#
# Model download time counts against generation timeouts. something2 caps a
# provider call at 5 minutes and cannot poll, so the first request for an
# uncached multi-GB checkpoint fails there regardless of how it is written.
# Run this once after `make up`, and again after adding a model.
#
# WHAT THIS DOES AND DOES NOT GUARANTEE:
#
#   Downloaded to disk  -> yes, for every model listed. This is the point.
#   Resident in VRAM    -> only the LAST one.
#
# The worker keeps exactly one pipeline on the GPU (see get_sd_pipeline): a
# 12GB card cannot hold two ~7GB SDXL pipelines, so loading a second evicts the
# first. Warming several models therefore leaves only the final one resident;
# the others still benefit, because reloading from the local cache takes
# seconds instead of a multi-minute download.
#
# List the model you will actually use LAST if you care which stays hot.
#
#   ./scripts/warm-models.sh [model ...]
set -uo pipefail

# Overall per-model deadline, and how long a task may sit unclaimed before we
# call it lost. Both in seconds; override from the environment.
WARM_TIMEOUT="${WARM_TIMEOUT:-3600}"
PENDING_GRACE="${PENDING_GRACE:-120}"

HOST="${SPRITE_HOST:-localhost}"
BASE="http://${HOST}:8001"
MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  # Default to whatever the service advertises.
  mapfile -t MODELS < <(curl -s --max-time 15 "$BASE/sdapi/v1/sd-models" \
    | grep -oE '"model_name":[[:space:]]*"[^"]*"' | cut -d'"' -f4)
fi

if [ ${#MODELS[@]} -eq 0 ]; then
  echo "No models to warm (is the stack up? \`make up\`)"; exit 1
fi

FAILED=0
for model in "${MODELS[@]}"; do
  echo "=== Warming ${model} ==="
  task=$(curl -s --max-time 30 -X POST "$BASE/api/warm" \
           --data-urlencode "model=${model}" \
         | grep -oE '"task_id":[[:space:]]*"[^"]*"' | cut -d'"' -f4)
  if [ -z "$task" ]; then echo "  could not queue"; FAILED=1; continue; fi
  echo "  task ${task} queued; polling (a cold ~7GB checkpoint takes a while)"

  # A deadline is REQUIRED, not optional.
  #
  # Celery reports PENDING for an unknown task id exactly as it does for one
  # that is merely queued — there is no way to tell "waiting" from "this task
  # was never received". An earlier version of this loop was `while true` with
  # no deadline; a warm task that never reached the worker left it spinning for
  # over four hours. Bound it, and treat a task that never even starts as lost.
  waited=0
  started_running=0
  while [ "$waited" -lt "$WARM_TIMEOUT" ]; do
    # head -1 is load-bearing. The response is
    #   {"task_id":..., "status":"SUCCESS", "result":{"status":"ok", ...}}
    # so there are TWO "status" keys: Celery's, and the one inside the task's
    # own return value. Without head -1 this captured "SUCCESS\nok", matched no
    # case branch, fell through to the wait arm and polled until the deadline —
    # reporting a timeout for warms that had actually succeeded in ~70s.
    # Celery's status comes first because the endpoint builds the dict in that
    # order.
    status=$(curl -s --max-time 20 "$BASE/api/task-status/${task}" \
             | grep -oE '"status":[[:space:]]*"[^"]*"' | head -1 | cut -d'"' -f4)
    case "$status" in
      SUCCESS) echo "  done (${waited}s)"; break ;;
      FAILURE) echo "  FAILED — see \`docker logs sprite_worker\`"; FAILED=1; break ;;
      "")      echo "  lost contact with the API"; FAILED=1; break ;;
      PENDING)
        # Still unclaimed. If nothing picks it up within PENDING_GRACE, the
        # worker is down or the task never made it onto the queue.
        if [ "$waited" -ge "$PENDING_GRACE" ] && [ "$started_running" -eq 0 ]; then
          echo ""
          echo "  still PENDING after ${waited}s — no worker claimed it."
          echo "  Check: docker ps (is sprite_worker up?), docker logs sprite_worker,"
          echo "         docker exec redis_broker redis-cli LLEN celery"
          FAILED=1; break
        fi
        printf '.'; sleep 10; waited=$((waited+10)) ;;
      *)
        started_running=1
        printf '.'; sleep 10; waited=$((waited+10)) ;;
    esac
  done
  if [ "$waited" -ge "$WARM_TIMEOUT" ]; then
    echo ""; echo "  gave up after ${WARM_TIMEOUT}s (raise WARM_TIMEOUT to wait longer)"
    FAILED=1
  fi
done

echo
[ "$FAILED" = "0" ] && echo "All models cached on disk. Only the last one is resident in VRAM (12GB fits one)." || { echo "Some models failed to warm."; exit 1; }
