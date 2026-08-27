#!/usr/bin/env python3
"""Verify the map facade over real HTTP, from another machine.

    python scripts/verify-map-facade.py --host http://gpu-box:8001 \
        --token $SPRITE_API_KEY --map overworld

Run this from the LAPTOP, not on the service. Everything else in the map suite
runs in-process inside the container and therefore proves nothing about the
network path, the bearer, or what something2 will actually receive - which is
the whole subject of this slice.

It checks the two things something2 depends on and one thing it must never see:

    401 without a bearer            the service is not open to the network
    images[0] is a real PNG         the connector gets what it expects
    a cache miss 404s in seconds    a two-hour build is never started inside
                                    a blocking request

The last is the one worth running deliberately. A facade that falls through to
the model looks identical on a hit; the difference only shows on a miss, and
by then something2's connector has been waiting for its whole timeout.

Exits non-zero on the first failure, so it is usable in a pipeline.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

OK = "  ok   "
BAD = "  FAIL "


def call(method, url, body=None, token=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}"), time.time() - started
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            payload = json.loads(raw or b"{}")
        except Exception:
            payload = {"detail": raw[:200].decode("utf-8", "replace")}
        return e.code, payload, time.time() - started
    except Exception as e:
        return 0, {"detail": f"{type(e).__name__}: {e}"}, time.time() - started


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=os.environ.get("SPRITE_HOST",
                                                    "http://localhost:8001"))
    p.add_argument("--token", default=os.environ.get("SPRITE_API_KEY")
                   or os.environ.get("SPRITE_API_TOKEN"))
    p.add_argument("--map", required=True, help="name of a map already built")
    p.add_argument("--miss-budget", type=float, default=15.0,
                   help="seconds a cache MISS may take before it counts as "
                        "having started a build")
    a = p.parse_args()

    if not a.token:
        print("no --token and no $SPRITE_API_KEY. The 401 check will run; the "
              "rest cannot.", file=sys.stderr)

    failed = 0
    txt2img = f"{a.host}/sdapi/v1/txt2img"

    # 1. Closed to the network without a bearer.
    status, body, _ = call("POST", txt2img, {"prompt": f"map:{a.map}"})
    if status == 401:
        print(f"{OK} 401 without a bearer")
    else:
        failed += 1
        print(f"{BAD} expected 401 without a bearer, got {status} {body}")

    if not a.token:
        print("\nstopping: the remaining checks need a token")
        return 1 if failed else 0

    # 2. The picture something2's connector will read.
    status, body, secs = call("POST", txt2img,
                              {"prompt": f"map:{a.map}"}, token=a.token)
    if status != 200:
        failed += 1
        print(f"{BAD} map fetch returned {status}: {body.get('detail', body)}")
    else:
        images = body.get("images") or []
        info = {}
        try:
            info = json.loads(body.get("info") or "{}")
        except Exception:
            pass
        raw = base64.b64decode(images[0]) if images else b""
        if raw[:8] != b"\x89PNG\r\n\x1a\n":
            failed += 1
            print(f"{BAD} images[0] is not a PNG ({len(raw)} bytes)")
        else:
            print(f"{OK} images[0] is a PNG, {len(raw)} bytes, in {secs:.2f}s")

        if info.get("generated") is not False:
            failed += 1
            print(f"{BAD} info.generated is {info.get('generated')!r} - this "
                  f"should be a cache read, not a generation")
        else:
            print(f"{OK} served from cache (job {info.get('job_id')})")

        if info.get("complete") is False:
            print(f"       NOTE provisional: still missing "
                  f"{', '.join(info.get('pending') or [])}. images[0] carries "
                  f"placeholder art - do not cache it as final.")

    # 3. The tilemap, by the same name.
    status, body, _ = call("GET", f"{a.host}/api/maps/by-name/{a.map}",
                           token=a.token)
    if status == 200 and body.get("layers", {}).get("terrain"):
        rows = len(body["layers"]["terrain"])
        print(f"{OK} tilemap by name: {rows}x{rows}, "
              f"{len(body.get('terrains') or [])} terrains")
    else:
        failed += 1
        print(f"{BAD} tilemap by name returned {status}: "
              f"{str(body)[:160]}")

    # 4. A miss must 404 fast, never queue a build.
    miss = f"__no_such_map_{int(time.time())}"
    status, body, secs = call("POST", txt2img, {"prompt": f"map:{miss}"},
                              token=a.token, timeout=300)
    if status != 404:
        failed += 1
        print(f"{BAD} a cache miss returned {status}, expected 404 - the "
              f"facade may be generating on demand")
    elif secs > a.miss_budget:
        failed += 1
        print(f"{BAD} a cache miss 404'd but took {secs:.1f}s (budget "
              f"{a.miss_budget}s) - something is doing work before refusing")
    else:
        print(f"{OK} a cache miss 404s in {secs:.2f}s without queueing")

    print()
    print("FAILED" if failed else "all checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
