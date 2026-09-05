#!/usr/bin/env python3
"""Ask the WORKER whether its CUDA context still works, and exit non-zero if not.

WHY THIS IS NOT A CURL AT AN HTTP ROUTE

On 2026-09-04 this box generated nothing for hours while every HTTP health
signal said green. `GET /` and `/docs` answered 200 throughout, and something2's
provider Test button reported ok because the models endpoint returned 200 with a
full model list. Meanwhile the worker's CUDA context was faulted and every
single generation 500'd.

Nothing that talks to the API can detect this. The API process is pinned to
COMPUTE_DEVICE=cpu precisely so it never holds VRAM, so it has no CUDA context
to be broken and will cheerfully report health it cannot observe.

Worse, the fault is process-local to the worker's MAIN process. Any probe that
starts a fresh interpreter - `docker exec ... python -c "import torch; ..."`,
a sidecar container, a subprocess - gets a NEW CUDA context, finds it perfectly
healthy, and tells you the GPU is fine while the process that actually runs
inference is dead. That is the trap, and it is easy to write by accident.

So this dispatches `tasks.gpu_probe` through Celery and lets the worker answer
in its own process. `--pool=solo` means that is the same process that runs
inference, which is the whole point.

WHAT A TIMEOUT MEANS

Ambiguous, deliberately reported as its own state. The worker runs solo, so a
probe queues behind any in-flight generation and a sheet job can take minutes.
A timeout means "busy or dead" and nothing finer. Re-run when idle before
concluding anything; do not wire this into an automatic restart.

Usage:
    docker exec sprite_generator python /app/scripts/check-gpu-health.py
    docker exec sprite_generator python /app/scripts/check-gpu-health.py --timeout 300

Exit codes:
    0  context healthy
    1  context faulted - the worker needs restarting
    2  probe timed out (busy or dead - inconclusive)
    3  could not reach the broker / task not registered
"""

import argparse
import os
import sys

for candidate in ("/app",
                  os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "src", "sprite_generator")):
    if os.path.isfile(os.path.join(candidate, "tasks.py")):
        sys.path.insert(0, candidate)
        break
else:
    sys.exit("tasks.py not found - run this inside a container that mounts /app")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timeout", type=float, default=60.0,
                   help="seconds to wait for the worker (default 60; raise it "
                        "if a long job may be in flight)")
    a = p.parse_args()

    # Imported here, not at module scope: importing tasks pulls in torch, and a
    # failure to reach the broker should report as exit 3 rather than a
    # traceback out of an import at the top of the file.
    try:
        from tasks import gpu_probe
    except Exception as e:
        print(f"could not import tasks: {e.__class__.__name__}: {e}")
        return 3

    try:
        result = gpu_probe.apply_async().get(timeout=a.timeout)
    except Exception as e:
        name = e.__class__.__name__
        if "Timeout" in name:
            print(f"TIMEOUT after {a.timeout:g}s - the worker did not answer.")
            print("  Inconclusive: solo pool, so a running job blocks the probe.")
            print("  Re-run when idle before concluding the context is dead.")
            return 2
        print(f"could not reach the worker: {name}: {e}")
        return 3

    if result.get("ok"):
        print(f"OK - worker CUDA context is alive on {result.get('device')}")
        head = result.get("headroom_mb")
        if head is not None:
            print(f"  free {result['free_mb']} MB | torch reserved "
                  f"{result['reserved_mb']} MB, of which "
                  f"{result['reserved_mb'] - result['allocated_mb']} MB is "
                  f"reusable | effective headroom ~{head} MB "
                  f"of {result['total_mb']} MB")
            # `free` on its own is normally 0 here and means nothing: the
            # caching allocator holds the card and does not give it back.
            # Headroom is free plus what torch can reuse without asking the
            # driver, and that is what the next ~7GB pipeline load draws on.
            if result.get("headroom_tight"):
                print("  TIGHT: under ~7GB, which is one SDXL pipeline. The "
                      "next load may have to ask WDDM for memory it does not "
                      "have; that returns ENOMEM and faults the context. This "
                      "is the state that preceded all three faults on "
                      "2026-09-04. Not an error - the context is alive - but "
                      "do not start a batch here.")
        print("  NOTE: all figures are GUEST-side. Under WSL2 none of them see "
              "host allocations - this read 1342MB used while Windows showed "
              "11601MB of the same 12GB card. That 11.6GB is NOT a leak: a "
              "healthy warm worker holds about the same. On Windows run:")
        print("    (Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage')"
              ".CounterSamples")
        return 0

    print(f"FAULTED - {result.get('error') or result.get('why') or result}")
    print("  A faulted context does not recover on its own, and it holds its "
          "VRAM, so the next load fails too. Restart the worker:")
    print("    docker compose -f compose/develop/docker-compose.yml "
          "-f compose/develop/docker-compose.cuda.yml \\")
    print("      --env-file compose/develop/.env restart sprite-worker")
    return 1


if __name__ == "__main__":
    sys.exit(main())
