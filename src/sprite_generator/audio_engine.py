"""Make one audio artefact: evict, generate, master, write. Worker-side.

WHY A SUBPROCESS AND NOT AN IMPORT

ACE-Step pins torch 2.10.0+cu128 and transformers <4.58, this worker runs
torch 2.14.0+cu130 and transformers 5.17, and diffusers depends on the
latter. They live in separate environments and talk over argv/stdout - see
`.ai/decisions/0010`. It is still ONE process at a time on the card: the
worker is `--pool=solo` and blocks on the child, so nothing else can be
holding VRAM while it runs. That is what keeps D6 ("no second GPU process")
true despite there being a second process.

Ambience takes the other path: Stable Audio Open runs in diffusers here, in
this process, because it fits this stack.

The eviction rule is `get_sd_pipeline`'s, for the same reason: 7.5 GB
(ACE-Step) or 6.0 GB (Stable Audio) plus a warm SDXL does not fit in 12 GB.
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import time
import uuid

import audio_master as am
import audio_styles as st

logger = logging.getLogger(__name__)

ACESTEP_PYTHON = os.environ.get("ACESTEP_PYTHON", "/opt/acestep/.venv/bin/python")
ACESTEP_SCRIPT = os.environ.get("ACESTEP_SCRIPT", "/app/scripts/acestep-generate.py")
ACESTEP_CHECKPOINTS = os.environ.get("ACESTEP_CHECKPOINTS_DIR", "/models/acestep")
AMBIENCE_MODEL = os.environ.get("AMBIENCE_MODEL", "stabilityai/stable-audio-open-1.0")
AUDIO_DIR = os.environ.get("AUDIO_DIR", "/app/audio")

# The ACE-Step child's wall-clock ceiling. The worker is --pool=solo and blocks
# on the child, so a hung child without this wedges EVERY job, images included.
# Measured 2026-09-12: load 23.3 s + a 120 s take 12.9 s cold, 7.4 s warm.
# 300 s is ~8x the cold figure - room for a cold disk or the sft model, far
# short of "forever". `subprocess.run` kills the child when it expires.
ACESTEP_TIMEOUT_S = float(os.environ.get("ACESTEP_TIMEOUT_S", "300"))

# Measured 2026-09-12: ACE-Step turbo peaks at 7,488 MB reserved, Stable Audio
# at 5,988 MB. Refuse before loading rather than OOM half way, the same
# preflight `get_flux_pipeline` makes.
GPU_BUDGET_MB = {"music": 8200, "ambience": 6600}

# A take must exceed the loop by the crossfade plus a bar of slack, or the
# whole-bar cut lands under the requested minimum.
TAKE_SLACK_S = 6.0

DURATION_BOUNDS = {"music": (60.0, 240.0), "ambience": (20.0, 45.0)}
DEFAULT_DURATION = {"music": 120.0, "ambience": 30.0}


class AudioGenerationError(RuntimeError):
    pass


def clamp_duration(kind: str, requested: float | None) -> float:
    lo, hi = DURATION_BOUNDS[kind]
    if not requested:
        return DEFAULT_DURATION[kind]
    return max(lo, min(hi, float(requested)))


def resolve_seed(seed: int | None) -> int:
    return int(seed) if seed and int(seed) > 0 else random.randint(1, 2 ** 31 - 1)


def _evict_pipelines() -> None:
    """Drop every diffusion pipeline before an audio model takes the card."""
    import gc

    import torch

    import tasks
    if tasks.DEVICE == "cuda" and tasks.pipes:
        logger.info("audio: evicting %d pipeline(s) before loading: %s",
                    len(tasks.pipes), sorted(tasks.pipes.keys()))
        tasks.pipes.clear()
        gc.collect()
        torch.cuda.empty_cache()


def _preflight(kind: str) -> None:
    import torch
    if not torch.cuda.is_available():
        raise AudioGenerationError("no CUDA device on the worker")
    free, total = torch.cuda.mem_get_info()
    need = GPU_BUDGET_MB[kind] * 2 ** 20
    if free < need:
        raise AudioGenerationError(
            f"not enough free VRAM for {kind}: {free / 2**30:.1f} GiB free of "
            f"{total / 2**30:.1f} GiB, need {need / 2**30:.1f} GiB. Something "
            f"else holds the card - llm-server keeps a chat model resident "
            f"until it idles out.")


# ---------------------------------------------------------------------------
# Music: ACE-Step, in its own venv
# ---------------------------------------------------------------------------

def _generate_music(rendered: dict, duration_s: float, seed: int,
                    work_dir: str) -> dict:
    job = {
        "id": "take",
        "prompt": rendered["prompt"],
        "lyrics": rendered["lyrics"],
        "bpm": rendered["bpm"],
        "time_signature": rendered["time_signature"],
        # The cut takes whole bars out of this, so ask for more than the loop.
        "duration_s": round(duration_s + TAKE_SLACK_S, 2),
        "seed": seed,
        "save_dir": work_dir,
    }
    env = dict(os.environ,
               ACESTEP_CHECKPOINTS_DIR=ACESTEP_CHECKPOINTS,
               ACESTEP_PROJECT_ROOT=os.environ.get("ACESTEP_PROJECT_ROOT",
                                                   "/opt/acestep"),
               TOKENIZERS_PARALLELISM="false")
    logger.info("audio: spawning ACE-Step for %.1fs at %s bpm, seed %d",
                job["duration_s"], job["bpm"], seed)
    try:
        proc = subprocess.run([ACESTEP_PYTHON, ACESTEP_SCRIPT, json.dumps(job)],
                              capture_output=True, text=True, env=env,
                              timeout=ACESTEP_TIMEOUT_S)
    except FileNotFoundError:
        raise AudioGenerationError(
            f"ACE-Step is not installed in this image ({ACESTEP_PYTHON} is "
            f"missing) - rebuild the worker from Dockerfile.cuda")
    except subprocess.TimeoutExpired:
        raise AudioGenerationError(
            f"ACE-Step did not finish within {ACESTEP_TIMEOUT_S:.0f}s "
            f"(ACESTEP_TIMEOUT_S) and was killed")

    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            result = json.loads(line[len("RESULT "):])
    if result is None:
        tail = (proc.stderr or proc.stdout or "")[-600:]
        raise AudioGenerationError(
            f"ACE-Step produced no result (exit {proc.returncode}): {tail}")
    if result.get("error"):
        raise AudioGenerationError(result["error"])
    return result


# ---------------------------------------------------------------------------
# Ambience: Stable Audio Open, in this process
# ---------------------------------------------------------------------------

def _generate_ambience(rendered: dict, duration_s: float, seed: int,
                       work_dir: str) -> dict:
    import numpy as np
    import torch
    from diffusers import StableAudioPipeline

    t0 = time.time()
    pipe = StableAudioPipeline.from_pretrained(AMBIENCE_MODEL,
                                               torch_dtype=torch.float16).to("cuda")
    load_s = time.time() - t0
    try:
        t0 = time.time()
        gen = torch.Generator("cuda").manual_seed(seed)
        audio = pipe(rendered["prompt"],
                     negative_prompt=rendered["negative"],
                     num_inference_steps=100,
                     audio_end_in_s=float(duration_s + TAKE_SLACK_S),
                     num_waveforms_per_prompt=1,
                     generator=gen).audios
        took = time.time() - t0
        wav = audio[0].T.float().cpu().numpy()
        sr = pipe.vae.sampling_rate
        peak = round(torch.cuda.max_memory_allocated() / 2 ** 20)
    finally:
        del pipe
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    path = os.path.join(work_dir, f"take_{uuid.uuid4().hex[:8]}.wav")
    am.write_wav(path, np.asarray(wav), sr, subtype="FLOAT")
    return {"path": path, "seed": seed, "seconds": round(took, 2),
            "load_s": round(load_s, 2), "peak_alloc_mb": peak}


# ---------------------------------------------------------------------------

def generate(kind: str, name: str, *, style: str | None = None,
             prompt: str | None = None, slots: dict | None = None,
             seed: int | None = None, duration_s: float | None = None,
             audio_dir: str | None = None) -> dict:
    """One artefact, mastered and written. Raises `AudioGenerationError`."""
    import soundfile as sf

    if kind not in DURATION_BOUNDS:
        raise AudioGenerationError(f"unknown audio kind {kind!r}")

    audio_dir = audio_dir or AUDIO_DIR
    duration_s = clamp_duration(kind, duration_s)
    seed = resolve_seed(seed)
    rendered = st.render(style or st.default_for(kind), **(slots or {}))
    if prompt:
        # An explicit prompt overrides the template but keeps the roster's
        # negative list and metre - dropping those is how vocals get in.
        rendered = dict(rendered, prompt=prompt, adjusted=rendered["adjusted"]
                        + ["prompt overridden by the caller"])

    work_dir = os.path.join(audio_dir, "_takes")
    os.makedirs(work_dir, exist_ok=True)

    _evict_pipelines()
    _preflight(kind)

    started = time.time()
    if kind == "music":
        take = _generate_music(rendered, duration_s, seed, work_dir)
    else:
        take = _generate_ambience(rendered, duration_s, seed, work_dir)

    samples, sr = sf.read(take["path"], dtype="float32", always_2d=True)
    loop = am.make_loop(samples, sr,
                        bpm=rendered["bpm"] if kind == "music" else None,
                        time_signature=rendered["time_signature"],
                        min_s=duration_s)

    uid = uuid.uuid4().hex[:8]
    wav_path, ogg_path = am.audio_paths(audio_dir, kind, name, uid)
    am.write_wav(wav_path, loop["samples"], sr)
    am.write_ogg(ogg_path, loop["samples"], sr,
                 loop_start=loop["loop_start"], loop_end=loop["loop_end"])
    try:
        os.remove(take["path"])
    except OSError:
        pass

    from tasks import release_vram_cache
    release_vram_cache(f"audio {kind}")

    return {
        "file_path": ogg_path,
        "master_path": wav_path,
        "kind": kind,
        "name": name,
        "style": rendered["style"],
        "prompt": rendered["prompt"],
        "negative": rendered["negative"],
        "seed": seed,
        "bpm": rendered["bpm"],
        "time_signature": rendered["time_signature"],
        "slots": rendered["slots"],
        "adjusted": rendered["adjusted"],
        "duration_s": loop["duration_s"],
        "sample_rate": sr,
        "loop_start": loop["loop_start"],
        "loop_end": loop["loop_end"],
        "bars": loop["bars"],
        "seam_rms_jump_db": loop["seam_rms_jump_db"],
        "generation_seconds": round(time.time() - started, 2),
        "model_seconds": take.get("seconds"),
        "load_seconds": take.get("load_s"),
        "peak_alloc_mb": take.get("peak_alloc_mb"),
    }
