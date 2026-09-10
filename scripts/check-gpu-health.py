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
            resident = result.get("pipeline_resident")
            need = result.get("headroom_needed_mb", 7000)
            if result.get("headroom_tight"):
                if resident:
                    print(f"  TIGHT: under ~{need}MB with a pipeline already "
                          "resident, which is the INFERENCE peak (~1.3GB for a "
                          "tiled 1024 decode) plus margin. The next request may "
                          "have to ask WDDM for memory it does not have; that "
                          "returns ENOMEM and faults the context.")
                else:
                    print(f"  TIGHT: under ~{need}MB and NO pipeline is "
                          "resident, so the next request must load one (~7GB "
                          "for SDXL). That load may ask WDDM for memory it does "
                          "not have; ENOMEM there faults the context. This is "
                          "the state that preceded all three faults on "
                          "2026-09-04. Not an error - the context is alive - "
                          "but do not start a batch here.")
            elif resident:
                # Say the quiet part out loud, because the old flat rule called
                # this state TIGHT and told people not to run. It is the normal
                # warm steady state now that cached VRAM is released after each
                # generation, and a 12-image batch runs clean in it.
                print(f"  Pipeline resident; headroom covers the ~{need}MB an "
                      "inference needs. Normal warm state.")
        print("  NOTE: all figures are GUEST-side. Under WSL2 none of them see "
              "host allocations, so guest and host readings diverge - this "
              "once read 1342MB used while Windows showed 11601MB of the same "
              "12GB card. A high host figure is not automatically a leak. But "
              "11.6GB is no longer the warm baseline: since cached VRAM is "
              "released after each generation, a healthy warm worker measures "
              "~7GB host-side (6,966MB on 2026-09-10). On Windows run:")
        print("    (Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage')"
              ".CounterSamples")
        return 0

    print(f"FAULTED - {result.get('error') or result.get('why') or result}")
    # This used to say a faulted context "does not recover on its own". Two
    # measurements since say that is too strong: one cleared in under two
    # minutes on 2026-09-04 once a retry storm stopped, and the ledger shows
    # the same on 2026-09-10 - a burst of ten failures at ~91ms each, then
    # clean 44s generations hours later with no restart in between. The worker
    # now enforces that quiet window itself (GPU_FAULT_COOLDOWN_S), so the
    # first move is to stop sending traffic, not to restart.
    print("  STOP SENDING TRAFFIC FIRST and re-run this probe. A fault often "
          "clears once the card goes quiet; a retry storm is what prevents "
          "it. Restart only if it still reports FAULTED with the card idle:")
    print("    docker compose -f compose/develop/docker-compose.yml "
          "-f compose/develop/docker-compose.cuda.yml \\")
    print("      --env-file compose/develop/.env restart sprite-worker")
    return 1


if __name__ == "__main__":
    sys.exit(main())
