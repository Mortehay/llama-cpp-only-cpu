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
import math
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
GPU_BUDGET_MB = {"music": 8200, "ambience": 6600, "sfx": 6600}

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

    import brain_engine
    import tasks
    brain_engine.stop("an audio model is loading")
    if tasks.DEVICE == "cuda" and tasks.pipes:
        logger.info("audio: evicting %d pipeline(s) before loading: %s",
                    len(tasks.pipes), sorted(tasks.pipes.keys()))
        tasks.pipes.clear()
        gc.collect()
        torch.cuda.empty_cache()


# Stable Audio stays RESIDENT between audio jobs, under the same key the model
# gateway uses for it (model_gateway.AUDIO_STABLE). A batch of cues from
# something2's art console arrives one request at a time; reloading per
# request cost 3-46 s each (measured 2026-09-28), against ~8 s of actual work.
#
# No idle timer of its own, on purpose: the gateway already decides which
# model owns the card, and every other model's load path clears `tasks.pipes`
# (get_sd_pipeline, the Qwen and edit paths, music below) - so the first job
# for anything else evicts it, exactly as a warm SDXL is evicted today.
# AUDIO_KEEP_WARM=0 restores load-and-drop per job.
SAO_KEY = "audio:stable-audio"
KEEP_WARM = os.environ.get("AUDIO_KEEP_WARM", "1") != "0"


def _sao_pipeline(kind: str):
    """(pipeline, load_seconds). Reuses the resident one; loads otherwise.

    The window size is restored to the model's full length on every hand-out:
    sfx shrinks it (SFX_LATENT_FRAMES) and ambience must never inherit that.
    """
    import torch
    from diffusers import StableAudioPipeline

    import tasks
    pipe = tasks.pipes.get(SAO_KEY)
    if pipe is not None:
        pipe.transformer.register_to_config(sample_size=pipe._full_sample_size)
        logger.info("audio: Stable Audio already resident - no load")
        return pipe, 0.0
    _evict_pipelines()
    _preflight(kind)
    t0 = time.time()
    pipe = StableAudioPipeline.from_pretrained(AMBIENCE_MODEL,
                                               torch_dtype=torch.float16).to("cuda")
    pipe._full_sample_size = int(pipe.transformer.config.sample_size)
    if KEEP_WARM:
        tasks.pipes[SAO_KEY] = pipe
    return pipe, round(time.time() - t0, 2)


def _done_with_sao(pipe) -> None:
    """Restore the full window; free the card only if not keeping it warm."""
    pipe.transformer.register_to_config(sample_size=pipe._full_sample_size)
    if not KEEP_WARM:
        import gc

        import torch
        del pipe
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

    pipe, load_s = _sao_pipeline("ambience")
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
        _done_with_sao(pipe)

    path = os.path.join(work_dir, f"take_{uuid.uuid4().hex[:8]}.wav")
    am.write_wav(path, np.asarray(wav), sr, subtype="FLOAT")
    return {"path": path, "seed": seed, "seconds": round(took, 2),
            "load_s": round(load_s, 2), "peak_alloc_mb": peak}


# ---------------------------------------------------------------------------
# Sound effects: Stable Audio Open, ONE load for a whole batch
# ---------------------------------------------------------------------------
#
# A cue is under 2 s of audio but the model load is ~21 s, so a pack of cues
# is built in one load (owner, 2026-09-28: a pack endpoint as well as single
# cues). Variants come from ONE call with `num_waveforms_per_prompt`, so 1-5
# variants cost about one generation, not five.

def sfx_ledger_params(item: dict, res: dict) -> dict:
    """What a finished cue's ledger row records. ONE definition for both
    engines - the worker (realistic) and the API (retro) - so a row reads the
    same whichever built it. Torch-free: the API process calls it."""
    first = res["variants"][0]
    return {"cue": res["cue"], "entity": res["entity"],
            "engine": res["engine"], "engine_from": item.get("engine_from"),
            "sample_rate": res["sample_rate"],
            "duration_s": first["duration_s"], "variants": res["variants"],
            "model_seconds": res["model_seconds"],
            "load_seconds": res["load_seconds"],
            "peak_alloc_mb": res.get("peak_alloc_mb")}


SFX_TAKE_SLACK_S = 0.7      # generated past the cue so the release is not cut
SFX_STEPS = int(os.environ.get("SFX_STEPS", "100"))
MAX_VARIANTS = 5

# How much latent the transformer denoises, in frames (hop 2048 at 44.1 kHz,
# so 128 frames = 5.9 s). The pipeline's default is the model's FULL window,
# 1024 frames = 47.6 s, whatever `audio_end_in_s` asks for - so a 0.6 s hit
# cost the same 29.3 s as a 30 s ambience. Measured 2026-09-28, one cue, 100
# steps: 1024 -> 29.3 s, 128 -> 8.0 s, 64 -> 6.9 s but with a 2.36 raw peak
# and a hit that rang out to 1.1 s. 128 covers every cue (max 1.5 s + slack).
SFX_LATENT_FRAMES = int(os.environ.get("SFX_LATENT_FRAMES", "128"))


def generate_sfx(items: list[dict], *, audio_dir: str | None = None) -> list[dict]:
    """Build every item in one model load. Raises only on a load failure.

    Each item: {cue, entity, engine, variants, seed}. A per-item failure (a
    silent take, a bad cue) is returned as {"error": ...} for THAT item, so
    one bad cue does not throw away the others' GPU time. A CUDA fault still
    raises: the caller trips the breaker, as for every other GPU task.
    """
    import numpy as np
    import torch

    audio_dir = audio_dir or AUDIO_DIR
    pipe, load_s = _sao_pipeline("sfx")
    sr = pipe.vae.sampling_rate
    # Shrink the denoised window to what a cue needs (see SFX_LATENT_FRAMES).
    # The pipeline may stay RESIDENT for the next job, so `_done_with_sao`
    # restores the full window in the finally below - ambience must never
    # inherit a 6 s window.
    full = pipe._full_sample_size
    frames = min(full, max(SFX_LATENT_FRAMES, 32))
    pipe.transformer.register_to_config(sample_size=frames)
    out: list[dict] = []
    try:
        for item in items:
            try:
                rendered = st.render_cue(item["cue"], item.get("entity"),
                                         item["engine"])
                n = max(1, min(MAX_VARIANTS, int(item.get("variants") or 1)))
                seed = resolve_seed(item.get("seed"))
                t1 = time.time()
                end_s = float(rendered["duration_s"] + SFX_TAKE_SLACK_S)
                keep = int(math.ceil(end_s * sr / pipe.vae.hop_length)) + 8
                # Variants are SEQUENTIAL, seeds seed..seed+n-1, each decoded
                # from latents cut to the cue first. Both measured 2026-09-28:
                # the pipeline's own batched decode (3 variants) faulted the
                # card deterministically - dxgkio_make_resident -12 inside
                # autoencoder_oobleck.decode - and batched denoising was SLOWER
                # than separate calls (142 s for 3 vs 29 s for 1). The decoder
                # is convolutional, so decoding a prefix is valid; the edge at
                # the cut falls in the slack the mastering trims.
                audios = []
                for v in range(n):
                    gen = torch.Generator("cuda").manual_seed(seed + v)
                    latents = pipe(rendered["prompt"],
                                   negative_prompt=rendered["negative"],
                                   num_inference_steps=SFX_STEPS,
                                   audio_end_in_s=end_s,
                                   output_type="latent",
                                   generator=gen).audios
                    with torch.no_grad():
                        wave = pipe.vae.decode(latents[:1, :, :keep]).sample
                    audios.append(wave[0, :, :int(end_s * sr)])
                    del wave, latents
                    torch.cuda.empty_cache()
                took = round(time.time() - t1, 2)
                uid = uuid.uuid4().hex[:8]
                variants = []
                for v, a in enumerate(audios, start=1):
                    shot = am.master_one_shot(
                        np.asarray(a.T.float().cpu().numpy()), sr)
                    wav, ogg = am.sfx_paths(audio_dir, rendered["engine"],
                                            rendered["cue"], rendered["entity"],
                                            uid, v)
                    am.write_wav(wav, shot["samples"], sr)
                    am.write_ogg(ogg, shot["samples"], sr, loop=False)
                    variants.append({"file_path": ogg, "master_path": wav,
                                     "seed": seed + v - 1,
                                     "duration_s": shot["duration_s"],
                                     "onset_ms": shot["onset_ms"],
                                     "trimmed_lead_ms": shot["trimmed_lead_ms"]})
                out.append({**{k: rendered[k] for k in
                               ("cue", "entity", "engine", "prompt", "negative")},
                            "seed": seed, "sample_rate": sr,
                            "variants": variants, "model_seconds": took,
                            "load_seconds": load_s})
            except (st.NoRecipe, st.UnknownStyle, ValueError) as e:
                out.append({"cue": item.get("cue"), "entity": item.get("entity"),
                            "engine": item.get("engine"), "error": str(e)})
        peak = round(torch.cuda.max_memory_allocated() / 2 ** 20)
        for o in out:
            o.setdefault("peak_alloc_mb", peak)
    finally:
        _done_with_sao(pipe)
        from tasks import release_vram_cache
        release_vram_cache("audio sfx")
    return out


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

    if kind == "music":
        # ACE-Step runs in its own process and needs ~7.5 GB: EVERYTHING in
        # this one goes, a resident Stable Audio included. Ambience does its
        # own evict-if-loading in _sao_pipeline, so a warm one is reused.
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
