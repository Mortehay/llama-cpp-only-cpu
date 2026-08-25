#!/usr/bin/env python3
"""Poll a job until it reaches a terminal state, then report.

This is what a something2-side integration does, so it doubles as a worked
example of the polling contract: ask, look at `status`, stop when it is
done/failed/cancelled, and read the sheet from `sheet_url` rather than guessing
a path.

Usage:
    python await-job.py <job_id> [--timeout 7200] [--interval 20]
    python await-job.py --latest
"""

import os
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

TERMINAL = {"done", "failed", "cancelled"}


def get(url, token=None):
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception:
        return 0, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("job_id", nargs="?")
    p.add_argument("--host", default="http://127.0.0.1:8001")
    p.add_argument("--token",
                   default=os.environ.get("SPRITE_API_KEY")
                   or os.environ.get("SPRITE_API_TOKEN"),
                   help="bearer token; defaults to $SPRITE_API_KEY")
    p.add_argument("--timeout", type=int, default=7200)
    p.add_argument("--interval", type=int, default=20)
    p.add_argument("--latest", action="store_true",
                   help="await the most recently updated job")
    a = p.parse_args()

    job_id = a.job_id
    if a.latest or not job_id:
        code, listing = get(f"{a.host}/api/jobs?limit=1", a.token)
        if code != 200 or not listing or not listing.get("jobs"):
            print("no jobs found", file=sys.stderr)
            return 1
        job_id = listing["jobs"][0]["job_id"]
        print(f"awaiting most recent job {job_id}")

    deadline = time.monotonic() + a.timeout
    last = None
    while time.monotonic() < deadline:
        code, job = get(f"{a.host}/api/jobs/{job_id}", a.token)
        if code == 404:
            print(f"job {job_id} not found", file=sys.stderr)
            return 1
        if job:
            now = (job["status"], job.get("stage"), job.get("progress_pct"))
            if now != last:
                print(f"  {job['status']:10} {job.get('stage') or '-':20} "
                      f"{job.get('progress_pct')}%  {job.get('progress_msg') or ''}")
                last = now
            if job["status"] in TERMINAL:
                print()
                if job["status"] == "done":
                    print(f"DONE  sheet: {a.host}{job.get('sheet_url')}")
                    print(f"      atlas: {a.host}{job.get('atlas_url')}")
                    return 0
                print(f"{job['status'].upper()}: {job.get('error')}")
                return 1
        time.sleep(a.interval)

    print(f"still running after {a.timeout}s", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
