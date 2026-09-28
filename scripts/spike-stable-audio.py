#!/usr/bin/env python3
"""Spike for .ai/tickets/02: measure Stable Audio Open 1.0 on this card.

Runs with the WORKER's python (diffusers 0.40 ships StableAudioPipeline), in
a `docker compose run` of sprite-worker with the worker service stopped.
Three ambience prompts, 30 s, fixed seed, negative prompt; prints one JSON
line per clip plus a summary for .ai/decisions/0010.

The model is gated. One-time, out of band, with the terms accepted on
huggingface.co/stabilityai/stable-audio-open-1.0:

    HF_HUB_OFFLINE=0 HF_TOKEN=hf_... python spike-stable-audio.py --download-only

After that the run works with HF_HUB_OFFLINE=1 like everything else here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

MODEL = "stabilityai/stable-audio-open-1.0"
NEGATIVE = "vocals, singing, music, melody, low quality"
PROMPTS = [
    ("forest", "forest ambience, wind in leaves, distant birds, calm daytime field recording"),
    ("cave", "dripping water in a stone cave, distant echo, faint airflow, damp, field recording"),
    ("village_day", "medieval village square by day, crowd murmur, footsteps on stone, "
                    "distant cart, field recording"),
]


def log(msg: str) -> None:
    print(f"[spike] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--download-only", action="store_true")
    args = ap.parse_args()

    out_dir = os.environ.get("SPIKE_OUT", "/app/audio/spike")
    os.makedirs(out_dir, exist_ok=True)

    if args.download_only:
        from huggingface_hub import snapshot_download
        path = snapshot_download(MODEL, token=os.environ.get("HF_TOKEN"))
        log(f"snapshot at {path}")
        return 0

    import numpy as np
    import torch
    from diffusers import StableAudioPipeline
    from scipy.io import wavfile  # the worker image has scipy but not soundfile

    log(f"torch {torch.__version__} cuda available={torch.cuda.is_available()} "
        f"offline={os.environ.get('HF_HUB_OFFLINE')}")
    if not torch.cuda.is_available():
        log("no CUDA device visible - refusing to measure on CPU")
        return 2
    free, total = torch.cuda.mem_get_info()
    log(f"VRAM before load: free {free/2**20:.0f} MB of {total/2**20:.0f} MB")

    t0 = time.time()
    pipe = StableAudioPipeline.from_pretrained(MODEL, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    load_s = time.time() - t0
    torch.cuda.synchronize()
    log(f"loaded in {load_s:.1f}s, allocated {torch.cuda.memory_allocated()/2**20:.0f} MB")

    rows = []
    for i, (name, prompt) in enumerate(PROMPTS, 1):
        torch.cuda.reset_peak_memory_stats()
        gen = torch.Generator("cuda").manual_seed(args.seed)
        t0 = time.time()
        audio = pipe(
            prompt,
            negative_prompt=NEGATIVE,
            num_inference_steps=args.steps,
            audio_end_in_s=args.duration,
            num_waveforms_per_prompt=1,
            generator=gen,
        ).audios
        torch.cuda.synchronize()
        took = time.time() - t0
        wav = audio[0].T.float().cpu().numpy()  # (samples, channels)
        sr = pipe.vae.sampling_rate
        path = os.path.join(out_dir, f"spike_ambience_{name}_{args.seed}.wav")
        wavfile.write(path, sr, (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16))
        row = {
            "i": i, "name": name, "seed": args.seed, "seconds": round(took, 1),
            "cold": i == 1, "sr": sr, "channels": wav.shape[1],
            "duration_s": round(wav.shape[0] / sr, 2),
            "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20),
            "file": path,
        }
        print("ROW " + json.dumps(row), flush=True)
        rows.append(row)

    warm = [r["seconds"] for r in rows if not r["cold"]]
    summary = {
        "clips": len(rows), "load_s": round(load_s, 1),
        "cold_s": rows[0]["seconds"] if rows else None,
        "warm_s_min": min(warm) if warm else None,
        "warm_s_max": max(warm) if warm else None,
        "peak_alloc_mb_max": max(r["peak_alloc_mb"] for r in rows),
        "peak_reserved_mb_max": max(r["peak_reserved_mb"] for r in rows),
        "sr": rows[0]["sr"] if rows else None,
    }
    print("SUMMARY " + json.dumps(summary), flush=True)
    with open(os.path.join(out_dir, "spike-stable-audio.json"), "w") as fh:
        json.dump({"rows": rows, "summary": summary}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
