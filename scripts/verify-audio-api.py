#!/usr/bin/env python3
"""Verify the audio contract end to end, the way something2 will call it.

    SPRITE_API_KEY=sk_... python scripts/verify-audio-api.py --submit
    SPRITE_API_KEY=sk_... python scripts/verify-audio-api.py --submit --kind ambience
    SPRITE_API_KEY=sk_... python scripts/verify-audio-api.py --overshoot
    SPRITE_API_KEY=sk_... python scripts/verify-audio-api.py --lan 192.168.0.217

Read-only by default: it checks discovery and the 404 shape without spending
GPU. `--submit` is the one that generates.

The assertions that matter are the ones a caching consumer depends on:
`audio[0]` decodes, the OGG's own LOOPSTART/LOOPLENGTH agree with `info`, and
a second request for the same name is a cache read that costs no GPU.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

KEY = os.environ.get("SPRITE_API_KEY") or os.environ.get("SPRITE_API_TOKEN") or ""
FAILED = 0


def out(ok: bool, msg: str) -> None:
    global FAILED
    if not ok:
        FAILED += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {msg}")


def call(base: str, path: str, body: dict | None = None, timeout: int = 600):
    req = urllib.request.Request(base + path, method="POST" if body else "GET")
    if KEY:
        req.add_header("Authorization", f"Bearer {KEY}")
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, data, timeout=timeout) as r:
            raw = r.read()
            # Read the type off the header OBJECT, which is case-insensitive;
            # dict(r.headers) is not, and uvicorn sends `content-type` lower.
            ctype = r.headers.get("Content-Type", "")
            # Lower-cased keys: uvicorn sends `retry-after`, and dict() over a
            # header object loses the case-insensitive lookup.
            headers = {k.lower(): v for k, v in r.headers.items()}
            # The file routes answer with audio, not JSON. Report its size and
            # type rather than trying to decode a Vorbis stream as UTF-8.
            if "json" not in ctype:
                return r.status, {"bytes": len(raw), "content_type": ctype}, headers
            return r.status, json.loads(raw.decode()), headers
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(raw), dict(e.headers)
        except ValueError:
            return e.code, {"detail": raw[:400]}, dict(e.headers)


def probe_ogg(b64: str) -> dict:
    """Decode `audio[0]` and read back what the FILE says, not what we were told.

    soundfile+mutagen when present (they are, inside the container), else
    ffprobe, else the container check alone - a caller on another LAN machine
    should still be able to run the rest.
    """
    blob = base64.b64decode(b64)
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as fh:
        fh.write(blob)
        path = fh.name
    info = {"bytes": len(blob), "path": path, "is_ogg": blob[:4] == b"OggS"}

    try:
        import soundfile as sf
        from mutagen.oggvorbis import OggVorbis
        meta = sf.info(path)
        info["duration_s"] = round(meta.duration, 2)
        info["sample_rate"] = meta.samplerate
        info["channels"] = meta.channels
        tags = OggVorbis(path)
        info["loop_start"] = int(tags.get("LOOPSTART", ["-1"])[0])
        info["loop_length"] = int(tags.get("LOOPLENGTH", ["-1"])[0])
        return info
    except ImportError:
        pass

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:format_tags=LOOPSTART,LOOPLENGTH", "-of", "json", path],
        capture_output=True, text=True)
    if probe.returncode == 0:
        d = json.loads(probe.stdout).get("format", {})
        info["duration_s"] = round(float(d.get("duration", 0)), 2)
        tags = {k.upper(): v for k, v in (d.get("tags") or {}).items()}
        info["loop_start"] = int(tags.get("LOOPSTART", -1))
        info["loop_length"] = int(tags.get("LOOPLENGTH", -1))
    else:
        info["probe_error"] = (probe.stderr or "ffprobe missing").strip()[:200]
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--lan", help="LAN IP; same checks through the portproxy")
    ap.add_argument("--kind", default="music", choices=["music", "ambience"])
    ap.add_argument("--submit", action="store_true", help="generate (spends GPU)")
    ap.add_argument("--overshoot", action="store_true",
                    help="expect 503 building; needs AUDIO_GENERATE_TIMEOUT_S=1")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    base = f"http://{args.lan or args.host}:8001"
    name = args.name or f"verify-{args.kind}-{int(time.time())}"
    print(f"=== {base}  kind={args.kind}  name={name} ===")
    if not KEY:
        print("  note: SPRITE_API_KEY is unset; every call will 401 while "
              "enforcement is on")

    status, styles, _ = call(base, "/api/audio/styles")
    ok = status == 200 and isinstance(styles, list) and styles
    out(ok, f"GET /api/audio/styles -> {status}, "
            f"{len(styles) if isinstance(styles, list) else '?'} entries")
    if ok:
        for k, expect in (("music", "medieval_fantasy"), ("ambience", "forest")):
            d = [s for s in styles if s.get("default") and s.get("kind") == k]
            out(len(d) == 1 and d[0]["value"] == expect,
                f"exactly one {k} default, and it is {expect}: "
                f"{[x['value'] for x in d]}")
        out(all("value" in s for s in styles),
            "every entry has `value` (the discovery pointer $[*].value)")

    status, body, _ = call(base, f"/api/audio/{args.kind}/definitely-not-here")
    detail = body.get("detail") if isinstance(body, dict) else {}
    out(status == 404, f"GET an unknown name -> {status} (must be 404, not 409)")
    out(isinstance(detail, dict) and detail.get("name") == "definitely-not-here",
        f"the 404 body names the artefact: {json.dumps(detail)[:110]}")

    if args.overshoot:
        t0 = time.time()
        status, body, headers = call(base, "/api/audio",
                                     {"kind": args.kind, "name": name})
        d = body.get("detail") if isinstance(body, dict) else {}
        out(status == 503, f"POST over budget -> {status} in {time.time()-t0:.1f}s")
        out(isinstance(d, dict) and d.get("reason") == "building",
            f"reason is `building`: {json.dumps(d)[:140]}")
        out("retry-after" in headers, f"Retry-After present: "
                                      f"{headers.get('retry-after')}")
        print("  ... now poll GET /api/audio?name= until it is done; the build "
              "was NOT cancelled")
        for _ in range(60):
            time.sleep(10)
            _, listing, _ = call(base, f"/api/audio?name={name}")
            items = listing.get("items") or []
            if items and items[0].get("status") in ("done", "failed"):
                out(items[0]["status"] == "done",
                    f"the orphaned row closed itself: status={items[0]['status']}")
                break
        else:
            out(False, "the row never closed - nothing finishes an abandoned build")
        return 1 if FAILED else 0

    if not args.submit:
        print("\n(read-only; pass --submit to generate)")
        return 1 if FAILED else 0

    t0 = time.time()
    status, body, _ = call(base, "/api/audio", {"kind": args.kind, "name": name})
    build_s = time.time() - t0
    if status != 200:
        out(False, f"POST /api/audio -> {status}: {json.dumps(body)[:300]}")
        return 1
    info = body.get("info", {})
    out(True, f"POST /api/audio -> 200 in {build_s:.1f}s "
              f"(style={info.get('style')}, seed={info.get('seed')})")
    out(bool(body.get("audio")), "the payload carries `audio[0]`")

    probe = probe_ogg(body["audio"][0])
    out(probe["is_ogg"], f"audio[0] decodes to an Ogg container "
                         f"({probe['bytes'] // 1024} KiB)")
    if "duration_s" in probe:
        lo = 60 if args.kind == "music" else 20
        out(probe["duration_s"] >= lo,
            f"duration {probe['duration_s']}s is at least the {lo}s floor")
        out(probe["loop_start"] == info.get("loop_start")
            and probe["loop_length"] == (info.get("loop_end", 0)
                                         - info.get("loop_start", 0)),
            f"the file's own loop tags match info: file "
            f"{probe['loop_start']}+{probe['loop_length']} vs info "
            f"{info.get('loop_start')}..{info.get('loop_end')}")
    else:
        out(False, f"could not read the file back: {probe.get('probe_error')}")
    out(info.get("cached") is False and info.get("served_from") == "generated",
        f"first call reports generated: cached={info.get('cached')}")

    t0 = time.time()
    status, body2, _ = call(base, "/api/audio", {"kind": args.kind, "name": name})
    hit_s = time.time() - t0
    info2 = body2.get("info", {})
    out(status == 200 and info2.get("cached") is True,
        f"second call is a cache read in {hit_s:.2f}s "
        f"(cached={info2.get('cached')})")
    out(hit_s < max(5.0, build_s / 4),
        f"the cache read is much faster than the build ({hit_s:.2f}s vs "
        f"{build_s:.1f}s)")

    status, _, _ = call(base, f"/api/audio/{args.kind}/{name}")
    out(status == 200, f"GET the file back by name -> {status}")
    status, _, _ = call(base, f"/api/audio/{args.kind}/{name}?master=1")
    out(status == 200, f"GET the WAV master -> {status}")
    status, i3, _ = call(base, f"/api/audio/{args.kind}/{name}/info")
    out(status == 200 and i3.get("loop_end") == info.get("loop_end"),
        f"GET .../info agrees with the generation response -> {status}")

    print(f"\n{'FAILED' if FAILED else 'OK'}: {FAILED} failure(s)")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
