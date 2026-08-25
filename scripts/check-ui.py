#!/usr/bin/env python3
"""Check the LEGACY browser UI is wired to the job API, not the old Celery endpoint.

The React UI now serves /, and it renders client-side - these string assertions
cannot see it. The legacy template lives at /legacy until React reaches parity,
and this keeps testing that. A rendering check for React needs a real browser:
    msedge --headless=new --dump-dom http://localhost:8001/

The UI is plain HTML plus one JS file with no build step and no tests, so the
failure mode is a page that loads fine and posts to a route that no longer does
what it used to. This asserts the wiring instead.

Usage:
    python check-ui.py [--host http://127.0.0.1:8001]
"""

import os
import argparse
import re
import sys
import urllib.request

# (label, must appear, must NOT appear) for each asset.
CHECKS = {
    "/legacy": [
        ("directions are their own axis", r'name="direction"', None),
        ("cell size, not render size", r'id="sheet-cell"', None),
        ("runtime estimate shown", r'id="sheet-estimate"', None),
        ("old SD model dropdown gone", None, r'id="sheet-llm"'),
        ("old free-text actions gone", None, r'value="move right"'),
        ("old render-size control gone", None, r'id="sheet-frame-size"'),
        # The core dropdown is rendered from core_models.roster(), which stats
        # the /models cache. The regression this guards is the options being
        # pasted back into the template: they would then keep offering
        # checkpoints that have been archived, and the only symptom would be
        # "Model failed to load on worker" arriving seconds later.
        ("core models carry availability", r'data-available="(true|false)"', None),
        ("core model warning slot present", r'id="core-model-warning"', None),
        # Actions are rendered from /api/action-catalog, which reads
        # action_prompts.json. The regression this guards is checkboxes being
        # pasted back into the template: the library would grow an action and
        # the UI would never offer it.
        ("actions rendered from the catalog", r'id="actions-grid"', None),
        ("actions not hardcoded", None, r'name="action"[^>]*value="walk"'),
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
        ("re-stats the model cache", r"fetch\('/api/core-models'\)", None),
        ("gates the core button on availability", r"updateCoreModelState", None),
        ("reads the action catalog", r"fetch\('/api/action-catalog'\)", None),
        ("sheet jobs shown in the queue", r"fetch\('/api/jobs\?limit=", None),
        # A direction is optional - an empty selection builds the front row.
        # This guards the button being re-gated on it.
        ("directions not required", None,
         r"dirs\.length === 0"),
    ],
    # The action vocabulary used to be asserted as `value="walk"` in the page.
    # It is served now, not rendered, so assert it at the source - including
    # that the library still reaches the API, which is the wiring that broke
    # when actions.py held its own hardcoded copy.
    "/api/action-catalog": [
        ("walk still defined", r'"id":\s*"walk"', None),
        ("idle still defined", r'"id":\s*"idle"', None),
        ("damage action defined", r'"id":\s*"damage"', None),
        ("use action defined", r'"id":\s*"use"', None),
        ("cast action defined", r'"id":\s*"cast"', None),
        ("sway action defined", r'"id":\s*"sway"', None),
        ("frame ceiling published", r'"max_frames"', None),
    ],
}


def fetch(url, token=None):
    req = urllib.request.Request(url)
    # Several of the paths checked here are authenticated now. Unset is fine
    # while the API has no keys - it is in open mode and answers anyway.
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.read().decode("utf-8", "replace")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="http://127.0.0.1:8001")
    p.add_argument("--token",
                   default=os.environ.get("SPRITE_API_KEY")
                   or os.environ.get("SPRITE_API_TOKEN"),
                   help="bearer token; defaults to $SPRITE_API_KEY")
    a = p.parse_args()

    ok = True
    for path, checks in CHECKS.items():
        print(f"== {path} ==")
        try:
            body = fetch(a.host + path, a.token)
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
