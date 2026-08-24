#!/usr/bin/env python3
"""Exercise the async job API the way something2 will.

Submit without waiting, get an id back, then poll - both for one job and for a
set of jobs at once. This is the contract something2 described, so it is worth
having a script that walks it rather than a paragraph claiming it works.

Uses urllib rather than requests/curl: neither is guaranteed inside the API
container, and this has to be runnable there.

Usage:
    python verify-jobs-api.py [--host http://127.0.0.1:8001] [--submit]

Without --submit it only reads, so it is safe to run against a busy stack.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request


def call(method, url, body=None, token=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="http://127.0.0.1:8001")
    p.add_argument("--token", default=None)
    p.add_argument("--submit", action="store_true",
                   help="actually enqueue a job (costs GPU time)")
    p.add_argument("--concept", default="meshtest_core.png")
    p.add_argument("--wait", type=int, default=0, metavar="SECONDS",
                   help="poll until the API answers before testing. The "
                        "service imports torch before uvicorn serves, so a "
                        "restart takes ~25s to become reachable")
    a = p.parse_args()

    if a.wait:
        import time
        deadline = time.monotonic() + a.wait
        while time.monotonic() < deadline:
            try:
                urllib.request.urlopen(f"{a.host}/api/jobs-health", timeout=3)
                break
            except Exception:
                time.sleep(3)
        else:
            print(f"API did not answer within {a.wait}s", file=sys.stderr)
            return 1

    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
        ok = ok and cond

    print("== queue health ==")
    code, health = call("GET", f"{a.host}/api/jobs-health", token=a.token)
    check("GET /api/jobs-health -> 200", code == 200, f"got {code}")
    print(f"        counts: {health.get('counts') if health else None}")

    print("\n== list (empty filter) ==")
    code, listing = call("GET", f"{a.host}/api/jobs?limit=5", token=a.token)
    check("GET /api/jobs -> 200", code == 200, f"got {code}")
    check("response carries server_time",
          bool(listing and listing.get("server_time")))
    print(f"        {listing.get('count') if listing else '?'} job(s) listed")

    print("\n== unknown id ==")
    code, _ = call("GET", f"{a.host}/api/jobs/{'0'*8}-0000-0000-0000-{'0'*12}",
                   token=a.token)
    check("unknown job -> 404", code == 404, f"got {code}")

    print("\n== bad id in batch ==")
    code, _ = call("GET", f"{a.host}/api/jobs?ids=not-a-uuid", token=a.token)
    check("malformed id -> 400", code == 400, f"got {code}")

    if not a.submit:
        print("\n(skipping submit; pass --submit to enqueue a real job)")
        return 0 if ok else 1

    print("\n== submit ==")
    code, job = call("POST", f"{a.host}/api/jobs", token=a.token, body={
        "concept_image": a.concept,
        "actions": ["walk"],
        "directions": ["s", "e"],
        "frames": 4,
    })
    check("POST /api/jobs -> 202 (accepted, not done)", code == 202, f"got {code}")
    if code != 202:
        print(f"        body: {job}")
        return 1

    job_id = job["job_id"]
    print(f"        job_id           {job_id}")
    print(f"        cells            {job.get('cells')}")
    print(f"        estimated_seconds {job.get('estimated_seconds')}")
    check("returned a poll url", bool(job.get("poll")))

    print("\n== poll one ==")
    code, one = call("GET", f"{a.host}/api/jobs/{job_id}", token=a.token)
    check("GET /api/jobs/{id} -> 200", code == 200, f"got {code}")
    check("status is queued or running",
          one.get("status") in ("queued", "running"), f"got {one.get('status')}")
    check("no sheet_url before it is done", "sheet_url" not in one)

    print("\n== poll a set ==")
    code, many = call("GET", f"{a.host}/api/jobs?ids={job_id}", token=a.token)
    check("GET /api/jobs?ids= -> 200", code == 200, f"got {code}")
    check("the submitted job comes back",
          any(j["job_id"] == job_id for j in (many.get("jobs") or [])))

    print("\n== sheet before ready ==")
    code, _ = call("GET", f"{a.host}/api/jobs/{job_id}/sheet", token=a.token)
    check("sheet while unfinished -> 409, not 404", code == 409, f"got {code}")

    print(f"\nJob {job_id} is running. Poll it with:")
    print(f"  curl {a.host}/api/jobs/{job_id}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
