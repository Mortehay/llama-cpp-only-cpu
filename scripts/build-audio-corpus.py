#!/usr/bin/env python3
"""Build the fixed audio acceptance corpus (ticket 11, decisions/0010 D1).

    SPRITE_API_KEY=<key> python scripts/build-audio-corpus.py
    SPRITE_API_KEY=<key> python scripts/build-audio-corpus.py --host 192.168.0.217

Ten music tracks (5 styles x 2 seeds) and five ambience loops (one per
style), generated through the REAL `POST /api/audio` with fixed names, so the
set is the same set every time: a later change to mastering, the roster or
the model is a before/after against it. Names are cache keys, so re-running
costs nothing for what already exists - delete a row's files to rebuild it.

Writes `audio/corpus/corpus.json` (per-row data) and copies every OGG + WAV
beside it. The listening fields - `verdict` (inaudible | audible | click),
`vocals` (true/false), `note` - are left EMPTY for a person to fill in; the
summary in `.ai/specs/audio/corpus.md` is written from them afterwards. This
script never invents a verdict.

A 503 `building` is not a failure: the build carries on, and the script
waits and asks again, as something2 should.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request

KEY = os.environ.get("SPRITE_API_KEY") or ""
MUSIC = ("medieval_fantasy", "tavern", "dungeon", "battle", "village")
AMBIENCE = ("forest", "cave", "village_day", "night", "rain")
SEEDS = (11, 42)


def post(base: str, body: dict, timeout: int = 600):
    req = urllib.request.Request(base + "/api/audio", method="POST",
                                 data=json.dumps(body).encode())
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {KEY}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), {}
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read()), dict(e.headers)
        except ValueError:
            return e.code, {}, dict(e.headers)


def build_one(base: str, body: dict) -> dict:
    """POST until 200, honouring Retry-After on 503 building/busy."""
    t0 = time.time()
    for _ in range(20):
        status, resp, headers = post(base, body)
        if status == 200:
            return {"info": resp["info"], "seconds": round(time.time() - t0, 1)}
        detail = resp.get("detail") if isinstance(resp, dict) else None
        reason = detail.get("reason") if isinstance(detail, dict) else None
        if status == 503 and reason in ("building", "busy"):
            wait = int(headers.get("retry-after") or headers.get("Retry-After") or 60)
            print(f"    {reason}; waiting {wait}s")
            time.sleep(wait)
            continue
        raise RuntimeError(f"{status}: {json.dumps(resp)[:300]}")
    raise RuntimeError("still not built after 20 attempts")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--audio-dir", default=None,
                    help="the host's AUDIO_DIR (default: ./audio in this repo)")
    args = ap.parse_args()
    if not KEY:
        print("SPRITE_API_KEY is required: the corpus goes through the real, "
              "authenticated API, as something2 will.")
        return 2

    base = f"http://{args.host}:8001"
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    audio_dir = args.audio_dir or os.path.join(repo, "audio")
    out_dir = os.path.join(audio_dir, "corpus")
    os.makedirs(out_dir, exist_ok=True)
    manifest = os.path.join(out_dir, "corpus.json")
    old = {}
    if os.path.exists(manifest):
        # Keep verdicts a person already wrote; never overwrite them.
        with open(manifest, encoding="utf-8") as fh:
            old = {r["name"]: r for r in json.load(fh)}

    plan = ([("music", s, f"corpus-{s}-{seed}", seed) for s in MUSIC for seed in SEEDS]
            + [("ambience", s, f"corpus-{s}", SEEDS[0]) for s in AMBIENCE])
    rows = []
    for kind, style, name, seed in plan:
        print(f"  {kind:8} {name}")
        try:
            got = build_one(base, {"kind": kind, "name": name, "style": style,
                                   "seed": seed})
        except Exception as e:  # noqa: BLE001 - record and carry on
            print(f"    FAILED: {e}")
            rows.append({"kind": kind, "style": style, "name": name,
                         "error": str(e)})
            continue
        info = got["info"]
        # The files are on this host's disk under AUDIO_DIR; the static URL
        # mirrors the layout (`/audio/<kind>/<file>`).
        # A re-rolled name leaves older files beside the newest; the ledger
        # serves the newest, so the corpus copies the newest too.
        kind_dir = os.path.join(audio_dir, kind)
        matches = [os.path.join(kind_dir, f) for f in os.listdir(kind_dir)
                   if f.startswith(name + "_") and f.endswith(".ogg")]
        src = max(matches, key=os.path.getmtime) if matches else None
        copied = []
        if src:
            for ext in (".ogg", ".wav"):
                p = src[:-4] + ext
                if os.path.exists(p):
                    shutil.copy2(p, os.path.join(out_dir, os.path.basename(p)))
                    copied.append(os.path.basename(p))
        prev = old.get(name, {})
        rows.append({
            "kind": kind, "style": style, "name": name,
            "seed": info.get("seed"),
            "duration_s": info.get("duration_s"),
            "sample_rate": info.get("sample_rate"),
            "loop_start": info.get("loop_start"), "loop_end": info.get("loop_end"),
            "bars": info.get("bars"), "seam_rms_jump_db": info.get("seam_rms_jump_db"),
            "served_from": info.get("served_from"),
            "seconds_this_run": got["seconds"], "files": copied,
            "prompt": info.get("prompt"),
            # Filled in by a PERSON after listening; preserved across re-runs.
            "verdict": prev.get("verdict", ""), "vocals": prev.get("vocals", None),
            "note": prev.get("note", ""),
        })
        print(f"    {info.get('served_from')} in {got['seconds']}s, "
              f"seam {info.get('seam_rms_jump_db')} dB")

    with open(manifest, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    failed = sum(1 for r in rows if r.get("error"))
    print(f"\n{len(rows) - failed}/{len(rows)} built -> {manifest}")
    print("Next: listen (Audio tab, Seam button) and fill verdict / vocals / note.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
