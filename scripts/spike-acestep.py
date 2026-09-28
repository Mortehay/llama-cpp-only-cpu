#!/usr/bin/env python3
"""Spike for .ai/tickets/01: measure ACE-Step 1.5 turbo on this card.

Runs with ACE-Step's OWN venv python (see Dockerfile.audio-spike), never the
worker's. Generates the fixed 10-prompt set (5 styles x 2 seeds, instrumental,
120 s) and prints one JSON line per track plus a summary, so the numbers can
be pasted into .ai/decisions/0010 as measured.

    ACESTEP_CHECKPOINTS_DIR=/checkpoints SPIKE_OUT=/out \
        /opt/acestep/.venv/bin/python spike-acestep.py [--duration 120] [--only N]

`--only N` runs the first N prompts (a quick check before the full set).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# The 10-prompt set. bpm/timesignature are set on purpose: the product's loop
# cut is bar arithmetic on these values, so the spike must prove the model
# honours them.
STYLES = [
    ("medieval_fantasy", "medieval fantasy RPG overworld theme, lute, wooden flute, "
                         "soft frame drum, harp, warm and adventurous, orchestral folk, "
                         "instrumental, no vocals", 96, "4", "D major"),
    ("tavern", "lively medieval tavern music, fiddle, hurdy-gurdy, bodhran, "
               "tin whistle, cheerful dance, instrumental, no vocals", 120, "4", "G major"),
    ("dungeon", "dark dungeon ambience music, low drone, distant bells, sparse "
                "cello, dripping tension, slow, instrumental, no vocals", 70, "4", "D minor"),
    ("battle", "epic medieval battle music, war drums, brass, choir-less strings, "
               "driving rhythm, heroic, instrumental, no vocals", 140, "4", "E minor"),
    ("village", "peaceful medieval village daytime, acoustic guitar, recorder, "
                "gentle strings, pastoral, calm, instrumental, no vocals", 88, "3", "C major"),
]
SEEDS = [11, 42]


def log(msg: str) -> None:
    print(f"[spike] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--only", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()

    out_dir = os.environ.get("SPIKE_OUT", "/out")
    os.makedirs(out_dir, exist_ok=True)
    project_root = os.environ.get("ACESTEP_PROJECT_ROOT", "/opt/acestep")
    sys.path.insert(0, project_root)

    import torch
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    log(f"torch {torch.__version__} cuda {torch.version.cuda} "
        f"available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        log("no CUDA device visible - refusing to measure on CPU")
        return 2
    free, total = torch.cuda.mem_get_info()
    log(f"VRAM before load: free {free/2**20:.0f} MB of {total/2**20:.0f} MB")

    t0 = time.time()
    handler = AceStepHandler()
    status, ok = handler.initialize_service(
        project_root=project_root,
        config_path="acestep-v15-turbo",
        device="auto",
        offload_to_cpu=False,
    )
    load_s = time.time() - t0
    log(f"initialize_service ok={ok} in {load_s:.1f}s: {status}")
    if not ok:
        return 1
    torch.cuda.synchronize()
    after_load = torch.cuda.memory_allocated()
    log(f"allocated after load: {after_load/2**20:.0f} MB, "
        f"reserved {torch.cuda.memory_reserved()/2**20:.0f} MB, dtype {handler.dtype}")

    prompts = [(s, seed) for s in STYLES for seed in SEEDS]
    if args.only:
        prompts = prompts[:args.only]

    rows = []
    for i, ((style, caption, bpm, sig, key), seed) in enumerate(prompts, 1):
        torch.cuda.reset_peak_memory_stats()
        params = GenerationParams(
            task_type="text2music",
            thinking=False,
            caption=caption,
            lyrics="[Instrumental]",
            instrumental=True,
            bpm=bpm,
            keyscale=key,
            timesignature=sig,
            duration=args.duration,
            inference_steps=args.steps,
            guidance_scale=1.0,
            seed=seed,
        )
        config = GenerationConfig(batch_size=1, audio_format="wav")
        t0 = time.time()
        result = generate_music(handler, None, params=params, config=config,
                                save_dir=out_dir)
        took = time.time() - t0
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        peak_res = torch.cuda.max_memory_reserved()

        row = {
            "i": i, "style": style, "seed": seed, "bpm": bpm, "sig": sig,
            "key": key, "ok": bool(result.success), "seconds": round(took, 1),
            "cold": i == 1,
            "peak_alloc_mb": round(peak / 2**20),
            "peak_reserved_mb": round(peak_res / 2**20),
            "status": (result.status_message or "")[:200],
            "paths": [a.get("path") for a in (result.audios or [])],
        }
        for p in row["paths"]:
            if p and os.path.exists(p):
                try:
                    import soundfile as sf
                    info = sf.info(p)
                    row["sr"] = info.samplerate
                    row["channels"] = info.channels
                    row["duration_s"] = round(info.duration, 2)
                    row["bytes"] = os.path.getsize(p)
                except Exception as e:  # noqa: BLE001
                    row["sf_error"] = str(e)[:120]
                new = os.path.join(out_dir, f"spike_{style}_{seed}.wav")
                try:
                    os.replace(p, new)
                    row["file"] = new
                except OSError:
                    row["file"] = p
        print("ROW " + json.dumps(row), flush=True)
        rows.append(row)

    done = [r for r in rows if r["ok"]]
    warm = [r["seconds"] for r in done if not r["cold"]]
    summary = {
        "tracks": len(rows), "ok": len(done),
        "load_s": round(load_s, 1),
        "alloc_after_load_mb": round(after_load / 2**20),
        "cold_s": next((r["seconds"] for r in done if r["cold"]), None),
        "warm_s_min": min(warm) if warm else None,
        "warm_s_max": max(warm) if warm else None,
        "peak_alloc_mb_max": max((r["peak_alloc_mb"] for r in rows), default=None),
        "peak_reserved_mb_max": max((r["peak_reserved_mb"] for r in rows), default=None),
        "sr": next((r.get("sr") for r in done if r.get("sr")), None),
        "channels": next((r.get("channels") for r in done if r.get("channels")), None),
    }
    print("SUMMARY " + json.dumps(summary), flush=True)
    with open(os.path.join(out_dir, "spike-acestep.json"), "w") as fh:
        json.dump({"rows": rows, "summary": summary}, fh, indent=2)
    return 0 if done and len(done) == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
