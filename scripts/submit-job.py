#!/usr/bin/env python3
"""Submit a sheet job and print the id.

The counterpart to await-job.py. Between them they are the whole something2
integration in two files, which is the point: if these are awkward to use, the
API is wrong.

Usage:
    python submit-job.py meshtest_core.png \
        --actions walk,attack,idle --directions s,se,e,ne,n,nw,w,sw --frames 4
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

ALL_DIRECTIONS = "s,se,e,ne,n,nw,w,sw"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("concept", help="filename under images/")
    p.add_argument("--host", default="http://127.0.0.1:8001")
    p.add_argument("--token", default=None)
    p.add_argument("--actions", default="walk")
    p.add_argument("--directions", default=ALL_DIRECTIONS)
    p.add_argument("--frames", type=int, default=4)
    p.add_argument("--cell", default="48x64")
    p.add_argument("--colors", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    body = {
        "concept_image": a.concept,
        "actions": [s.strip() for s in a.actions.split(",") if s.strip()],
        "directions": [s.strip() for s in a.directions.split(",") if s.strip()],
        "frames": a.frames,
        "cell": a.cell,
        "colors": a.colors,
        "seed": a.seed,
    }

    req = urllib.request.Request(f"{a.host}/api/jobs",
                                 data=json.dumps(body).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if a.token:
        req.add_header("Authorization", f"Bearer {a.token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            code, out = r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()[:400]}", file=sys.stderr)
        return 1

    print(f"status  {code}")
    for k in ("job_id", "cells", "estimated_seconds", "poll"):
        print(f"{k:18} {out.get(k)}")
    mins = (out.get("estimated_seconds") or 0) / 60
    print(f"\nestimate ~{mins:.0f} min. Await it with:")
    print(f"  python /app/scripts/await-job.py {out['job_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
