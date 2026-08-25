#!/usr/bin/env python3
"""Train an SDXL style LoRA on reference art, on one 12 GB card.

WHY SDXL AND NOT SOMETHING BIGGER

Qwen-Image-Edit-2511 is ~20B parameters and is not trainable on this hardware at
any setting. It keeps its job as the untrained pose editor - it is good at it,
and the turnaround/action stages depend on it. What training buys is the
CONCEPT stage: the look of the character before any pose work happens.

SDXL is the right target because the base checkpoint is already in /models, the
tooling is mature, and 12 GB is a documented working point at 768 px. Flux is
trainable in 12 GB too but slow enough that iteration suffers; it is the next
experiment, not the first.

HOW IT FITS IN 12 GB

The same trick `qwen_edit.py` uses, for the same reason: never hold two large
models at once.

    Stage 1 (cache)  VAE + both text encoders load, every image is encoded to
                     latents and every caption to embeddings, results go to
                     disk, and all three models are unloaded.
    Stage 2 (train)  Only the UNet is resident. The LoRA adapters are the only
                     trainable parameters, so optimiser state is megabytes
                     rather than gigabytes.

Without the split, VAE + 2 text encoders + UNet + activations do not fit and the
run dies partway through with a fragmented allocator - the failure this project
already hit repeatedly in ADR 0005.

WHAT IT LEARNS, AND WHAT IT CANNOT

A style LoRA learns line quality, shading idiom and how forms read at small
size. It does NOT fix geometry: if the camera angle is wrong, this will
faithfully reproduce the wrong angle. Measure a ground tile and set the camera
first - see `measure.py`.

Usage:
    python train-lora.py --name something2 --data /app/images --out /models/loras
    python train-lora.py --name x --steps 1200 --rank 32 --resolution 768
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("train-lora")

BASE = "stabilityai/stable-diffusion-xl-base-1.0"
CACHE_DIR = os.environ.get("HF_HUB_CACHE", "/models")

# SDXL's NATIVE 1024, measured rather than assumed.
#
# The usual advice is that 12 GB forces 768 and that 1024 needs 24 GB. That
# advice assumes the VAE and both text encoders are resident during training.
# They are not here - stage 1 encodes everything and unloads them - so only the
# UNet and its activations occupy the card.
#
# Measured on this 3060, rank 32, batch 1, gradient checkpointing on:
#     768 px  -> 5.38 GiB peak (45% of the card)
#    1024 px  -> 5.61 GiB peak (47%)
#
# Native resolution costs 0.23 GiB and avoids training the model at a scale it
# was not trained for, so it is the default. There is room above this for a
# larger batch or a higher rank; if either ever OOMs, the ladder down is
# batch -> rank 32/16 -> 1024/768.
DEFAULT_RESOLUTION = 1024


def load_component(cls, subfolder: str, dtype, prefer_fp16: bool = True):
    """Load one SDXL component, preferring the fp16 variant.

    The /models cache holds ONLY `model.fp16.safetensors` for the text encoders
    and the UNet - the full-precision files were never pulled, which is the
    right call at ~2x the size for weights that are used in bf16 anyway. Plain
    `from_pretrained` asks for `model.safetensors`, does not find it, and raises
    an OSError that reads like the model is missing entirely.

    Tries the variant first and falls back, so this works against either cache
    layout rather than encoding one machine's download history.
    """
    attempts = ([{"variant": "fp16"}, {}] if prefer_fp16 else [{}, {"variant": "fp16"}])
    last = None
    for kwargs in attempts:
        try:
            return cls.from_pretrained(BASE, subfolder=subfolder,
                                       torch_dtype=dtype, cache_dir=CACHE_DIR,
                                       **kwargs)
        except Exception as e:  # noqa: PERF203 - two attempts, not a hot loop
            last = e
    raise RuntimeError(
        f"could not load {subfolder} from {BASE}: {last}") from last


def log_progress(step: int, total: int, loss: float):
    """One line per report, in a shape the job runner can parse.

    Deliberately the same idea as the `denoised N/M` line build-sheet.py emits:
    the queue reads stdout, so progress has to be greppable, not just pretty.
    """
    print(f"trained {step}/{total} loss {loss:.4f}", flush=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def find_images(data_dir: str, patterns: tuple[str, ...]) -> list[str]:
    """Files matching the globs, excluding UI thumbnails.

    Thumbnails are named `thumb_<original>` precisely so these globs cannot
    match them, but the filter is here as well: a dataset silently padded with
    320px copies of its own images produces a worse adapter and no error, and
    that is exactly the kind of thing worth defending twice.
    """
    import glob
    out: list[str] = []
    for p in patterns:
        out.extend(sorted(glob.glob(os.path.join(data_dir, p))))
    return [p for p in out
            if os.path.isfile(p) and not os.path.basename(p).startswith("thumb_")]


def caption_for(path: str, trigger: str, label: str | None = None) -> str:
    """Templated caption with a trigger token.

    Style LoRAs do not need descriptive captions - they need the trigger bound
    to the look. A generic body plus the filename keeps the binding tight
    without inventing content that is not in the image.
    """
    stem = label or os.path.splitext(os.path.basename(path))[0]
    stem = stem.replace("_", " ").replace("-", " ")
    # Strip our own storage prefixes so "ref sprite a1b2c3" does not become
    # part of what the trigger means.
    for junk in ("ref sprite ", "ref core ", "ref tile ", "core ", "sheet "):
        if stem.startswith(junk):
            stem = stem[len(junk):]
    return f"{trigger} pixel art sprite, {stem}".strip().rstrip(",")


# ---------------------------------------------------------------------------
# Stage 1: cache latents and embeddings
# ---------------------------------------------------------------------------

def cache_inputs(paths, captions, resolution, cache_path, dtype):
    """Encode once, on the GPU, then free every encoder."""
    import torch
    from diffusers import AutoencoderKL
    from transformers import (CLIPTextModel, CLIPTextModelWithProjection,
                              CLIPTokenizer)
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Stage 1/2: encoding %d images at %dpx", len(paths), resolution)

    vae = AutoencoderKL.from_pretrained(
        BASE, subfolder="vae", torch_dtype=torch.float32, cache_dir=CACHE_DIR)
    vae.to(device).eval()
    vae.requires_grad_(False)

    latents = []
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGBA")
        # Composite onto neutral grey rather than black: a transparent sprite on
        # black teaches the model a black background, which is exactly the
        # artefact this pipeline spends a whole stage removing.
        bg = Image.new("RGBA", img.size, (128, 128, 128, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")

        # CENTRE-CROP to square, then scale. A bare
        # `resize((resolution, resolution))` squashes: a 1024x2048 reference
        # board arrives at the model with every figure at half width, and the
        # LoRA faithfully learns to draw squashed figures. Real references are
        # rarely square - of 106 uploaded here, 97 were between 0.5:1 and 2:1
        # and none were exactly 1:1.
        w, h = img.size
        if w != h:
            side = min(w, h)
            left, top = (w - side) // 2, (h - side) // 2
            img = img.crop((left, top, left + side, top + side))
        img = img.resize((resolution, resolution), Image.LANCZOS)

        import numpy as np
        arr = np.asarray(img).astype("float32") / 127.5 - 1.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            # VAE stays fp32: SDXL's VAE produces NaNs in fp16, a known issue
            # and an unpleasant one to debug through a training loss curve.
            dist = vae.encode(t).latent_dist
            lat = dist.sample() * vae.config.scaling_factor
        latents.append(lat.squeeze(0).to(torch.float32).cpu())
        if (i + 1) % 5 == 0 or i + 1 == len(paths):
            logger.info("  encoded %d/%d images", i + 1, len(paths))

    del vae
    torch.cuda.empty_cache()

    logger.info("  loading text encoders")
    tok1 = CLIPTokenizer.from_pretrained(BASE, subfolder="tokenizer",
                                         cache_dir=CACHE_DIR)
    tok2 = CLIPTokenizer.from_pretrained(BASE, subfolder="tokenizer_2",
                                         cache_dir=CACHE_DIR)
    te1 = load_component(CLIPTextModel, "text_encoder", dtype)
    te2 = load_component(CLIPTextModelWithProjection, "text_encoder_2", dtype)
    te1.to(device).eval(); te2.to(device).eval()
    te1.requires_grad_(False); te2.requires_grad_(False)

    embeds, pooled = [], []
    for cap in captions:
        with torch.no_grad():
            i1 = tok1(cap, padding="max_length", max_length=tok1.model_max_length,
                      truncation=True, return_tensors="pt").input_ids.to(device)
            i2 = tok2(cap, padding="max_length", max_length=tok2.model_max_length,
                      truncation=True, return_tensors="pt").input_ids.to(device)
            o1 = te1(i1, output_hidden_states=True)
            o2 = te2(i2, output_hidden_states=True)
            # SDXL concatenates the penultimate hidden states of both encoders,
            # and takes the pooled output from the second only.
            e = torch.cat([o1.hidden_states[-2], o2.hidden_states[-2]], dim=-1)
            embeds.append(e.squeeze(0).cpu())
            pooled.append(o2.text_embeds.squeeze(0).cpu())

    del te1, te2
    torch.cuda.empty_cache()

    import torch as _t
    _t.save({"latents": latents, "embeds": embeds, "pooled": pooled,
             "captions": captions, "resolution": resolution}, cache_path)
    logger.info("  cached to %s", cache_path)
    return cache_path


# ---------------------------------------------------------------------------
# Stage 2: train
# ---------------------------------------------------------------------------

def train(cache_path, out_dir, name, steps, rank, lr, batch_size, dtype, seed,
          resume_from=None):
    import torch
    import torch.nn.functional as F
    from diffusers import DDPMScheduler, UNet2DConditionModel
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig, get_peft_model_state_dict

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    random.seed(seed)

    blob = torch.load(cache_path, weights_only=False)
    latents, embeds, pooled = blob["latents"], blob["embeds"], blob["pooled"]
    resolution = blob["resolution"]
    n = len(latents)
    logger.info("Stage 2/2: training %d steps on %d images", steps, n)

    unet = load_component(UNet2DConditionModel, "unet", dtype)
    unet.to(device)
    unet.requires_grad_(False)

    # Attention projections only. Training the convolutions as well roughly
    # doubles the adapter for little gain on a style LoRA, and the memory is
    # not there to spare.
    unet.add_adapter(LoraConfig(
        r=rank, lora_alpha=rank, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"]))

    # RESUME: continue an existing adapter rather than starting from noise.
    #
    # This is what makes "train on the new images only" a real feature instead
    # of a trap. Without it, a run over twelve new references produces an
    # adapter that has seen twelve images - strictly worse than the one it
    # replaces, with nothing to say so. With it, the adapter keeps everything
    # it learned and refines on what is new.
    #
    # Rank must match what was saved; a mismatch raises here rather than
    # loading a subset of the tensors and training a half-initialised adapter.
    if resume_from:
        from diffusers import StableDiffusionXLPipeline as _SDXL
        from diffusers.utils import convert_unet_state_dict_to_peft
        from peft import set_peft_model_state_dict

        logger.info("Resuming from %s", resume_from)
        state = _SDXL.lora_state_dict(resume_from)
        if isinstance(state, tuple):
            state = state[0]
        unet_sd = {k.removeprefix("unet."): v for k, v in state.items()
                   if k.startswith("unet.")} or state
        peft_sd = convert_unet_state_dict_to_peft(unet_sd)
        missing, unexpected = set_peft_model_state_dict(
            unet, peft_sd, adapter_name="default")
        if unexpected:
            raise RuntimeError(
                f"{resume_from} does not fit a rank-{rank} adapter "
                f"({len(unexpected)} unexpected tensors). Train with the same "
                f"rank it was created with, or tick a full retrain.")
        logger.info("  resumed %d tensors", len(peft_sd))

    # Checkpointing trades ~30% speed for a large activation saving. On 12 GB
    # that is not a tuning knob, it is the difference between running and not.
    unet.enable_gradient_checkpointing()

    params = [p for p in unet.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in params)
    logger.info("  %d trainable parameters (%.1f M)", trainable, trainable / 1e6)

    # LoRA params are cast to fp32: bf16 optimiser states on a rank-32 adapter
    # lose enough precision to visibly stall the loss.
    for p in params:
        p.data = p.data.to(torch.float32)

    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=lr, weight_decay=1e-2)
        logger.info("  optimiser: AdamW8bit")
    except Exception:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
        logger.info("  optimiser: AdamW (bitsandbytes unavailable)")

    sched = DDPMScheduler.from_pretrained(BASE, subfolder="scheduler",
                                          cache_dir=CACHE_DIR)

    # SDXL's extra conditioning: original size, crop offset, target size. The
    # images are pre-resized square, so all three agree.
    add_time = torch.tensor([[resolution, resolution, 0, 0, resolution, resolution]],
                            device=device, dtype=dtype)

    unet.train()
    t0 = time.time()
    running = 0.0
    for step in range(1, steps + 1):
        idx = [random.randrange(n) for _ in range(batch_size)]
        lat = torch.stack([latents[i] for i in idx]).to(device, dtype=dtype)
        emb = torch.stack([embeds[i] for i in idx]).to(device, dtype=dtype)
        pol = torch.stack([pooled[i] for i in idx]).to(device, dtype=dtype)

        noise = torch.randn_like(lat)
        t = torch.randint(0, sched.config.num_train_timesteps,
                          (lat.shape[0],), device=device).long()
        noisy = sched.add_noise(lat, noise, t)

        pred = unet(
            noisy, t,
            encoder_hidden_states=emb,
            added_cond_kwargs={"text_embeds": pol,
                               "time_ids": add_time.repeat(lat.shape[0], 1)},
        ).sample

        # SDXL base predicts epsilon. Reading the target from the scheduler
        # rather than assuming it keeps this correct if the base ever changes.
        target = (noise if sched.config.prediction_type == "epsilon"
                  else sched.get_velocity(lat, noise, t))
        loss = F.mse_loss(pred.float(), target.float(), reduction="mean")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        running += loss.item()
        if step % 10 == 0 or step == steps:
            log_progress(step, steps, running / min(step, 10))
            running = 0.0

    os.makedirs(out_dir, exist_ok=True)
    lora_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    from diffusers import StableDiffusionXLPipeline
    StableDiffusionXLPipeline.save_lora_weights(
        save_directory=out_dir, unet_lora_layers=lora_state,
        weight_name=f"{name}.safetensors")

    path = os.path.join(out_dir, f"{name}.safetensors")
    mins = (time.time() - t0) / 60
    logger.info("Saved %s (%.1f MB) after %.1f min",
                path, os.path.getsize(path) / 1e6, mins)

    # Peak VRAM is the number that decides whether a setting is usable on this
    # card, and it is invisible unless recorded: an OOM says what failed, never
    # how close the successful run was. Reported so the headroom is known
    # before someone raises the resolution.
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        logger.info("Peak VRAM %.2f/%.2f GiB (%.0f%% of the card)",
                    peak, total, 100 * peak / total)
    return path


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="LoRA name; also the filename")
    p.add_argument("--data", default="/app/images",
                   help="directory to read training images from")
    p.add_argument("--pattern", default="ref_sprite_*.png,ref_core_*.png",
                   help="comma-separated globs within --data")
    p.add_argument("--files", default=None,
                   help="path to a manifest of image paths, one per line. "
                        "Takes precedence over --data/--pattern.")
    p.add_argument("--out", default="/models/loras")
    p.add_argument("--trigger", default=None,
                   help="trigger token; defaults to <name-style>")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default=None,
                   help="continue this existing adapter instead of starting "
                        "from scratch. Required for incremental training - "
                        "without it a run on only the new images produces an "
                        "adapter that has seen only those.")
    p.add_argument("--min-images", type=int, default=8,
                   help="refuse to start below this; a LoRA trained on three "
                        "images memorises them")
    a = p.parse_args()

    import torch
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if not torch.cuda.is_available():
        logger.warning("No CUDA device - this will be unusably slow.")

    # A manifest wins over globbing, and the queue always sends one.
    #
    # Globbing the images directory and counting rows in the database gave
    # different answers - 231 files against 191 trainable rows - because the
    # filesystem still holds images belonging to DELETED references and to ones
    # the judge rejected. So the UI promised one dataset and the trainer used
    # another, with no error either side. The manifest makes the counted set
    # and the trained set the same set by construction.
    if a.files:
        with open(a.files) as f:
            paths = [ln.strip() for ln in f if ln.strip()]
        paths = [p for p in paths if os.path.isfile(p)]
        logger.info("dataset from manifest %s: %d image(s)", a.files, len(paths))
    else:
        paths = find_images(a.data, tuple(s.strip() for s in a.pattern.split(",")))
        logger.info("dataset from glob %r: %d image(s)", a.pattern, len(paths))

    if len(paths) < a.min_images:
        sys.exit(f"only {len(paths)} image(s) matched {a.pattern} in {a.data}; "
                 f"need at least {a.min_images}. Upload more references - "
                 f"measurement works from three examples, but training does "
                 f"not.")

    trigger = a.trigger or f"<{a.name}-style>"
    captions = [caption_for(p, trigger) for p in paths]
    logger.info("%d images, trigger %r", len(paths), trigger)
    logger.info("example caption: %s", captions[0])

    os.makedirs(a.out, exist_ok=True)
    cache_path = os.path.join(a.out, f".{a.name}-cache.pt")

    try:
        cache_inputs(paths, captions, a.resolution, cache_path, dtype)
        out = train(cache_path, a.out, a.name, a.steps, a.rank, a.lr,
                    a.batch_size, dtype, a.seed, resume_from=a.resume)
    finally:
        # The cache is tens of MB per image and worthless once training ends.
        if os.path.exists(cache_path):
            os.remove(cache_path)

    meta = {"name": a.name, "base": BASE, "trigger": trigger,
            "steps": a.steps, "rank": a.rank, "lr": a.lr,
            "resolution": a.resolution, "images": len(paths),
            "weights": out}
    with open(os.path.join(a.out, f"{a.name}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
