#!/usr/bin/env python3
"""Generate one take with ACE-Step. Runs in ITS OWN venv, not the worker's.

    /opt/acestep/.venv/bin/python acestep-generate.py '<job json>'

ACE-Step pins torch 2.10.0+cu128 and transformers <4.58; the worker runs
torch 2.14.0+cu130 and transformers 5.17. One site-packages cannot hold both,
so the two talk over argv and stdout instead of an import. Nothing here may
import from /app.

Reads a job object, prints exactly one `RESULT <json>` line, exits non-zero on
failure. The parent (`audio_engine`) does the loop mastering; this only makes
the raw take.

The model stays loaded for the life of the process, so a caller that wants
several takes should pass `jobs` rather than paying the 23 s load again.
"""

from __future__ import annotations

import json
import os
import sys
import time

CHECKPOINT_CONFIG = os.environ.get("ACESTEP_DIT", "acestep-v15-turbo")


def emit(obj: dict) -> None:
    print("RESULT " + json.dumps(obj), flush=True)


def main() -> int:
    if len(sys.argv) < 2:
        emit({"error": "no job given"})
        return 2
    payload = json.loads(sys.argv[1])
    jobs = payload.get("jobs") or [payload]
    project_root = os.environ.get("ACESTEP_PROJECT_ROOT", "/opt/acestep")
    sys.path.insert(0, project_root)

    import torch
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    if not torch.cuda.is_available():
        emit({"error": "no CUDA device visible to the ACE-Step venv"})
        return 3

    t0 = time.time()
    handler = AceStepHandler()
    status, ok = handler.initialize_service(
        project_root=project_root,
        config_path=CHECKPOINT_CONFIG,
        device="auto",
        offload_to_cpu=False,
    )
    if not ok:
        emit({"error": f"initialize_service failed: {status}"})
        return 4
    load_s = time.time() - t0

    failed = False
    for job in jobs:
        params = GenerationParams(
            task_type="text2music",
            # The 1.7B LM is on disk but never loaded: it wants 3.5 GB the DiT
            # is already using, and the spike measured 10/10 without it.
            thinking=False,
            caption=job["prompt"],
            lyrics=job.get("lyrics") or "[Instrumental]",
            instrumental=True,
            bpm=job.get("bpm"),
            keyscale=job.get("keyscale") or "",
            timesignature=str(job.get("time_signature") or ""),
            duration=float(job["duration_s"]),
            inference_steps=int(job.get("steps", 8)),
            guidance_scale=float(job.get("guidance_scale", 1.0)),
            seed=int(job["seed"]),
        )
        t0 = time.time()
        try:
            result = generate_music(
                handler, None, params=params,
                config=GenerationConfig(batch_size=1, audio_format="wav"),
                save_dir=job["save_dir"],
            )
        except Exception as e:  # noqa: BLE001
            failed = True
            emit({"id": job.get("id"), "error": f"{type(e).__name__}: {e}"[:500]})
            continue
        took = time.time() - t0

        paths = [a.get("path") for a in (result.audios or []) if a.get("path")]
        if not result.success or not paths:
            failed = True
            emit({"id": job.get("id"),
                  "error": (result.status_message or "no audio returned")[:500]})
            continue
        emit({
            "id": job.get("id"),
            "path": paths[0],
            "seed": int(job["seed"]),
            "seconds": round(took, 2),
            "load_s": round(load_s, 2),
            "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20),
        })

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
