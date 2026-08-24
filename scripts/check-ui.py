#!/usr/bin/env python3
"""Check the browser UI is wired to the job API, not the old Celery endpoint.

The UI is plain HTML plus one JS file with no build step and no tests, so the
failure mode is a page that loads fine and posts to a route that no longer does
what it used to. This asserts the wiring instead.

Usage:
    python check-ui.py [--host http://127.0.0.1:8001]
"""

import argparse
import re
import sys
import urllib.request

# (label, must appear, must NOT appear) for each asset.
CHECKS = {
    "/": [
        ("directions are their own axis", r'name="direction"', None),
        ("new action vocabulary", r'value="walk"', None),
        ("cell size, not render size", r'id="sheet-cell"', None),
        ("runtime estimate shown", r'id="sheet-estimate"', None),
        ("old SD model dropdown gone", None, r'id="sheet-llm"'),
        ("old free-text actions gone", None, r'value="move right"'),
        ("old render-size control gone", None, r'id="sheet-frame-size"'),
    ],
    "/static/js/generator.js": [
        ("posts to the job API", r"fetch\('/api/jobs'", None),
        ("polls the job API", r"/api/jobs/\$\{jobId\}", None),
        ("uses sheet_url as the ready signal", r"job\.sheet_url", None),
        ("offers the atlas", r"job\.atlas_url", None),
        ("reattaches after reload", r"resumeSheetJob", None),
        # Match a CALL, not a mention. The first version of this check matched
        # the comment that explains why the endpoint is no longer used, and
        # reported a correct file as broken.
        ("old sheet endpoint not called", None,
         r"fetch\(\s*['\"]/api/generate_sheet"),
    ],
}


def fetch(url):
    with urllib.request.urlopen(url, timeout=15) as r:
        return r.read().decode("utf-8", "replace")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="http://127.0.0.1:8001")
    a = p.parse_args()

    ok = True
    for path, checks in CHECKS.items():
        print(f"== {path} ==")
        try:
            body = fetch(a.host + path)
        except Exception as e:
            print(f"  FAIL  could not fetch: {e}")
            ok = False
            continue
        for label, want, unwanted in checks:
            if want is not None:
                hit = re.search(want, body) is not None
                print(f"  {'PASS' if hit else 'FAIL'}  {label}")
                ok = ok and hit
            else:
                gone = re.search(unwanted, body) is None
                print(f"  {'PASS' if gone else 'FAIL'}  {label}")
                ok = ok and gone
        print()

    print("RESULT: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
