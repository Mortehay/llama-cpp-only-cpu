"""Qwen-Image-Edit-2511 on a 12 GB card, via GGUF.

This module exists to overturn one line in `tasks.py`:

    the two that were wanted - fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA
    ... - are trained against Qwen-Image-Edit, whose transformer does not fit
    this card.

That is true of the **bitsandbytes NF4** path `tasks.get_edit_pipeline` takes.
It is not true of GGUF. `unsloth/Qwen-Image-Edit-2511-GGUF` publishes the 20B
transformer at 7.47 GB (Q2_K), 9.92 GB (Q3_K_M) and 11.9 GB (Q4_0), and
diffusers loads those through `from_single_file` + `GGUFQuantizationConfig`.

Why this model and not a smaller one: it is the only open editor with a
purpose-built **8-azimuth camera LoRA**. Everything ADR 0003 and 0004 built to
fake directional consistency - derived view cores, 3/4-view skeletons, the
TripoSR to UniRig conveyor - collapses into one prompt token here. See ADR 0005.

STATUS: written against the published model cards and the diffusers GGUF docs;
NOT yet executed on this machine (no disk free at the time of writing). Run
`python qwen_edit.py --selftest` before trusting any of it.

--------------------------------------------------------------------------
The memory plan, which is the whole reason this is a separate module
--------------------------------------------------------------------------

Three components, and they do NOT fit simultaneously:

    transformer (GGUF Q3_K_M)         9.92 GB
    text encoder Qwen2.5-VL (NF4)      5.80 GB
    VAE                                ~0.25 GB
                                      ---------
                                       ~16 GB   against ~11.7 GB free

They do not need to. The text encoder runs ONCE per prompt and is then dead
weight for the entire denoising loop. So the batch structure is:

    pass 1   load TE, encode EVERY prompt for the sheet, cache, unload TE
    pass 2   load the transformer ONCE, denoise all ~150 cells, never reload
    pass 3   VAE decode + pixelate on the way out

Pass 2 is the one that matters. Reloading a 10 GB transformer per cell turns a
40-minute sheet into an overnight one, and that is the failure mode to watch
for - it does not error, it is just slow.

There is no GGUF for the text encoder: `GGUFQuantizationConfig` covers diffusion
transformers, not `transformers` text encoders. NF4 via bitsandbytes is the
supported route for that half, which is why both quantisation stacks are
required at once.

The API supports the split. Verified against diffusers 0.39.0 with
`scripts/inspect-qwen-api.py` (run it again after any upgrade):

    encode_prompt(prompt, image, device, num_images_per_prompt,
                  prompt_embeds, prompt_embeds_mask, max_sequence_length)
    __call__(..., prompt_embeds, prompt_embeds_mask,
             negative_prompt_embeds, negative_prompt_embeds_mask, ...)
    model_cpu_offload_seq = text_encoder->transformer->vae

**But note `encode_prompt` takes `image`.** Qwen-Image-Edit's text encoder is a
vision-language model, so an embedding is a function of the (prompt, image)
PAIR, not of the prompt alone. That bounds the batching: all 8 directions of one
turnaround share a single concept image and can be encoded together, but a later
stage whose input is the PREVIOUS stage's output cannot be encoded ahead of
time. The batch structure is therefore per-stage, not global - which is not what
ADR 0005 originally assumed.

MEASURED CONSTRAINT (2026-08-23): this WSL has 10 GiB of RAM, not the 11 GiB the
config requests, with ~7 GiB available. `enable_model_cpu_offload` keeps every
idle component in system RAM, so ~15 GiB of residency against 7 GiB free means
swapping. The simple path below still runs; it is just slow. Prove it works
first, then split into encode/denoise passes - in that order, because an
optimisation layered onto something unproven hides which half is broken.
"""

from __future__ import annotations

import logging
import os
import sys

import torch

logger = logging.getLogger(__name__)

# The pipeline is assembled from TWO publishers, because neither alone fits.
#
#   REPO         supplies the NF4 text encoder (5.09 GB), VAE, tokenizer and
#                scheduler. Its own transformer subfolder is 11.59 GB and is
#                deliberately NOT downloaded - see scripts/fetch-qwen-edit.py.
#   GGUF_REPO    supplies the transformer, smaller than the NF4 one.
#   CONFIG_REPO  supplies the transformer's architecture config only (a few KB).
#                Taken from the official repo rather than the 4bit one, whose
#                config carries an embedded bitsandbytes quantization_config
#                that contradicts the GGUF weights being loaded against it.
REPO = os.environ.get("QWEN_EDIT_REPO", "ovedrive/Qwen-Image-Edit-2511-4bit")
CONFIG_REPO = os.environ.get("QWEN_EDIT_CONFIG_REPO", "Qwen/Qwen-Image-Edit-2511")

# Which GGUF to pull. Q3_K_M is the default because it is the largest quant that
# leaves headroom for activations on an 11.7 GB budget:
#
#   Q2_K     7.47 GB   comfortable, visibly degraded
#   Q3_K_M   9.92 GB   default - fits with ~1.8 GB spare
#   Q4_0    11.9 GB    does NOT fit resident; needs offload, which costs the
#                      whole speed argument for keeping it resident
#
# Overridable because "fits" is a claim about this card and this image size, and
# the first thing to try when quality is short is a bigger quant at 384px.
#
# Mind the case: the repo is `Qwen-Image-Edit-2511-GGUF`, its files are
# `qwen-image-edit-2511-*.gguf`. A wrong-cased name here does not 404 loudly -
# it makes huggingface_hub match nothing and report success.
GGUF_REPO = os.environ.get("QWEN_EDIT_GGUF_REPO", "unsloth/Qwen-Image-Edit-2511-GGUF")
# Q2_K, and that is a measured choice, not a conservative one.
#
# This module keeps the transformer RESIDENT on the card rather than offloading
# it (see denoise_only for why), so the quant has to leave room for activations
# and the VAE inside 11.7 GiB:
#
#   Q2_K     7.47 GB   works. 2.7 GiB VRAM still free, 7.9 s/step at 512px
#   Q3_K_M   9.92 GB   does NOT fit resident - OOMs by ~24 MiB mid-denoise
#
# Q3_K_M only runs with QWEN_OFFLOAD=1, which then dies on host RAM instead.
# Quality at 48px is dominated by the pixelation stage, not by the quant.
GGUF_FILE = os.environ.get("QWEN_EDIT_GGUF_FILE", "qwen-image-edit-2511-Q2_K.gguf")

# bfloat16, not float16, despite the 3060's weak bf16 throughput.
# Qwen-Image is a flow-matching model trained in bf16; fp16 has a much smaller
# exponent range and flow-matching sampling is where that shows up, as NaN
# latents that surface only as a black image. Losing some speed beats losing the
# frame. This is the opposite of the call `tasks.py` makes for SD1.5/SDXL, and
# deliberately so.
DTYPE = torch.bfloat16

# Where the text encoder runs during the encode pass. It is the only thing on
# the card at that point, so it goes on directly rather than through an offload
# hook - offloading a component that has the GPU to itself is pure overhead.
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"

# fal's LoRA. Trained on 3000+ Gaussian-Splatting renders; 96 poses.
ANGLES_LORA = os.environ.get("QWEN_ANGLES_LORA",
                             "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA")
ANGLES_WEIGHT = "qwen-image-edit-2511-multiple-angles-lora.safetensors"

# Step distillation: 40 steps -> 4. The model card's own recommendation is to
# run the LoRA at strength 0.7 rather than 1.0.
#
# weight_name is REQUIRED for this one. The repo holds 4-step and 8-step LoRAs
# in both bf16 and fp32, plus a 20 GB fp8 full checkpoint - load_lora_weights
# cannot guess, and naming the file is also what keeps the fetch script from
# pulling a 107 GB repo.
LIGHTNING_LORA = os.environ.get("QWEN_LIGHTNING_LORA",
                                "lightx2v/Qwen-Image-Edit-2511-Lightning")
LIGHTNING_WEIGHT = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
LIGHTNING_SCALE = float(os.environ.get("QWEN_LIGHTNING_SCALE", "0.7"))
LIGHTNING_STEPS = int(os.environ.get("QWEN_LIGHTNING_STEPS", "4"))


# --- the angle vocabulary -----------------------------------------------
#
# These strings are not descriptive prose. They are the exact tokens the LoRA
# was trained on, and paraphrasing them ("from the left", "side-on") silently
# drops back to the base model's own interpretation - which is the failure this
# whole module exists to avoid. Copy them verbatim.

AZIMUTHS = {
    "s":  "front view",             #   0 deg - toward the camera
    "se": "front-right quarter",    #  45
    "e":  "right side",             #  90
    "ne": "back-right quarter",     # 135
    "n":  "back view",              # 180 - away from the camera
    "nw": "back-left quarter",      # 225
    "w":  "left side",              # 270
    "sw": "front-left quarter",     # 315
}

ELEVATIONS = {
    "low":      "low-angle shot",   # -30
    "eye":      "eye-level shot",   #   0
    "elevated": "elevated shot",    #  30
    "high":     "high-angle shot",  #  60
}

DISTANCES = {
    "close":  "close-up",   # x0.6
    "medium": "medium shot",  # x1.0
    "wide":   "wide shot",  # x1.8
}

# An isometric game wants a raised camera, not eye level. "elevated" (30 deg) is
# the closest trained elevation to the classic 30 deg isometric projection;
# "high" (60) looks top-down and reads as a different genre.
ISO_ELEVATION = os.environ.get("QWEN_ISO_ELEVATION", "elevated")

# Per-azimuth prompt reinforcement. Measured, not decorative.
#
# The LoRA card says to use its trigger tokens alone. For the front/back axis
# that is true - `back view` returns a genuine back with no face. It is FALSE
# for the two 90-degree profiles: measured 2026-08-23, `right side` and
# `left side` both came back near-frontal with the face fully visible.
#
# Hypotheses tested and rejected before landing here:
#   - the Lightning LoRA diluting the angles LoRA (tried without it, 20 steps)
#   - the elevated camera flattening the rotation (tried at eye level)
#   - Q2_K quantisation being too coarse to steer (disproved by this fix -
#     the model produces clean profiles at Q2_K once asked properly)
#
# Only the sides get a hint. Do NOT apply this globally: telling the FRONT view
# that the character faces sideways is a direct contradiction, and the model
# resolves contradictions by picking one at random per seed.
#
# Kept SHORT deliberately. Prompt length is not free: stage 1's bitsandbytes
# dequant fallback allocates against sequence length, and the first working
# version of this hint ("strict profile view, the character faces exactly
# sideways, only one eye visible, nose in silhouette") reliably OOM'd on a
# 1.02 GiB buffer once more than one long-hinted direction was in a batch.
AZIMUTH_HINTS = {
    "e": ", exact side profile, one eye visible",
    "w": ", exact side profile, one eye visible",
}

# Global override, still available for experiments. Applied to every azimuth.
EXTRA_PROMPT = os.environ.get("QWEN_EXTRA_PROMPT", "")


def angle_prompt(azimuth: str,
                 elevation: str = ISO_ELEVATION,
                 distance: str = "medium") -> str:
    """Build the LoRA's trigger string: `<sks> <azimuth> <elevation> <distance>`.

    Raises on an unknown key rather than falling through to a paraphrase. A
    silently wrong camera angle is the exact class of bug ADR 0003 spent a week
    on ("move up right" contained "move up"), and it is cheap to make loud.
    """
    for key, table, what in ((azimuth, AZIMUTHS, "azimuth"),
                             (elevation, ELEVATIONS, "elevation"),
                             (distance, DISTANCES, "distance")):
        if key not in table:
            raise KeyError(
                f"unknown {what} {key!r}; the LoRA only recognises "
                f"{sorted(table)} and a paraphrase silently does nothing"
            )
    base = f"<sks> {AZIMUTHS[azimuth]} {ELEVATIONS[elevation]} {DISTANCES[distance]}"
    return base + AZIMUTH_HINTS.get(azimuth, "") + EXTRA_PROMPT


# --- loading ------------------------------------------------------------

_pipe = None


def _parse_gib(spec: str) -> int:
    """'10.5GiB' / '10.5' / '11GB' -> bytes. A bare number is read as GiB."""
    s = str(spec).strip().lower().rstrip("b")
    if s.endswith("gi"):
        return int(float(s[:-2]) * 1024 ** 3)
    if s.endswith("g"):
        return int(float(s[:-1]) * 1000 ** 3)
    return int(float(s) * 1024 ** 3)


def _host_ram_available() -> int:
    """MemAvailable in bytes, or -1 if it cannot be read."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return -1


def _gguf_url() -> str:
    return f"https://huggingface.co/{GGUF_REPO}/blob/main/{GGUF_FILE}"


def load_pipeline(with_angles: bool = True,
                  with_lightning: bool = True,
                  sequential: bool = False,
                  cache_dir: str = "/models"):
    """Build the editing pipeline. Cached; call repeatedly without cost.

    Deliberately does NOT call `.to('cuda')`. `enable_model_cpu_offload` moves
    each component on as it is needed and off afterwards, which is what lets the
    text encoder and the transformer share a card that cannot hold both.
    """
    global _pipe
    if _pipe is not None:
        return _pipe

    from diffusers import (GGUFQuantizationConfig, QwenImageEditPlusPipeline,
                           QwenImageTransformer2DModel)

    token = os.environ.get("HF_TOKEN") or None

    # Evict whatever else holds the card BEFORE touching the checkpoint.
    #
    # The worker keeps SD pipelines resident in `tasks.pipes`, and a warm
    # All-In-One sits on ~2.7 GB. A 9.92 GB transformer plus activations does
    # not fit alongside that, and the failure is not a clean OOM: GGUF weights
    # are placed as from_single_file walks the file, so a short card fills
    # part-way through and dies with a CUDA error that reads like a driver
    # fault. `tasks.get_edit_pipeline` learned this the same way.
    #
    # Checked via sys.modules rather than `import tasks`.
    #
    # If tasks is already imported we are inside the Celery worker and its
    # `pipes` dict is the thing holding VRAM. If it is not, we are running
    # standalone (--selftest) and there is nothing to evict - and IMPORTING it
    # would be actively harmful: tasks.py raises at import when
    # COMPUTE_DEVICE=cuda is set without a usable card, builds a Celery app,
    # and pulls in poses/psycopg2. None of that belongs in a self-test whose
    # entire purpose is to isolate whether this pipeline loads.
    tasks = sys.modules.get("tasks")
    if tasks is not None and getattr(tasks, "pipes", None):
        logger.info("Evicting %d resident pipeline(s) before loading the "
                    "editor: %s", len(tasks.pipes), sorted(tasks.pipes))
        tasks.pipes.clear()

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        logger.info("VRAM after eviction: %.1f / %.1f GiB free",
                    free / 1024 ** 3, total / 1024 ** 3)
        need = _parse_gib(os.environ.get("QWEN_GPU_BUDGET", "10.5GiB"))
        if free < need:
            raise RuntimeError(
                f"Only {free / 1024 ** 3:.1f} GiB VRAM free, need about "
                f"{need / 1024 ** 3:.1f} GiB for {GGUF_FILE}. Something else "
                f"holds the card - llm-server keeps a chat model resident until "
                f"it has been idle for --sleep-idle-seconds. Loading anyway "
                f"fills the card mid-placement and poisons the CUDA context for "
                f"every later task. Use a smaller quant via QWEN_EDIT_GGUF_FILE "
                f"(Q2_K is 7.47 GB) or lower QWEN_GPU_BUDGET if you know better."
            )

    # Host RAM, which on this machine is tighter than VRAM and fails far less
    # legibly.
    #
    # enable_model_cpu_offload keeps every component that is not currently on
    # the GPU in system RAM: the 9.92 GB transformer plus the ~5 GB NF4 text
    # encoder is ~15 GB of residency, against a WSL cap of 11 GB
    # (`~/.wslconfig`, memory=11GB). There is 16 GB of swap behind that, so the
    # usual outcome is not a clean error - it is minutes of thrashing, or the
    # kernel OOM-killing the worker with no Python traceback at all, which
    # reads like the container crashing for no reason.
    #
    # Warn rather than refuse: swap makes it slow, not impossible, and on a
    # machine with a larger cap this is a non-issue. Raising memory= in
    # ~/.wslconfig needs `wsl --shutdown`, which stops every container.
    ram_free = _host_ram_available()
    ram_want = _parse_gib(os.environ.get("QWEN_HOST_RAM_WANTED", "14GiB"))
    if 0 <= ram_free < ram_want:
        logger.warning(
            "Only %.1f GiB host RAM available, and model offload wants about "
            "%.1f GiB. This will swap. If the worker dies with no traceback "
            "that is the OOM killer, not a bug in this module - raise "
            "memory= in ~/.wslconfig (needs `wsl --shutdown`) or use a smaller "
            "quant via QWEN_EDIT_GGUF_FILE.",
            ram_free / 1024 ** 3, ram_want / 1024 ** 3)

    logger.info("Loading Qwen-Image-Edit transformer from GGUF: %s", GGUF_FILE)
    # `config` and `subfolder` are REQUIRED for a diffusers-format GGUF. Without
    # them from_single_file cannot tell what architecture the tensors belong to
    # and raises something that reads like a corrupt file.
    transformer = QwenImageTransformer2DModel.from_single_file(
        _gguf_url(),
        quantization_config=GGUFQuantizationConfig(compute_dtype=DTYPE),
        config=CONFIG_REPO,
        subfolder="transformer",
        torch_dtype=DTYPE,
        cache_dir=cache_dir,
        token=token,
    )

    logger.info("Loading the rest of the pipeline from %s", REPO)
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        REPO,
        transformer=transformer,
        torch_dtype=DTYPE,
        cache_dir=cache_dir,
        token=token,
    )

    adapters, weights = [], []
    if with_angles:
        logger.info("Fusing the multiple-angles LoRA")
        pipe.load_lora_weights(ANGLES_LORA, weight_name=ANGLES_WEIGHT,
                               adapter_name="angles")
        adapters.append("angles")
        weights.append(1.0)
    if with_lightning:
        logger.info("Fusing the Lightning %d-step LoRA at %.2f",
                    LIGHTNING_STEPS, LIGHTNING_SCALE)
        pipe.load_lora_weights(LIGHTNING_LORA, weight_name=LIGHTNING_WEIGHT,
                               adapter_name="lightning")
        adapters.append("lightning")
        weights.append(LIGHTNING_SCALE)
    if adapters:
        # set_adapters, not fuse_lora. Fusing folds the delta into the base
        # weights, which is fine for one LoRA and wrong for two at different
        # strengths - and it cannot be undone without reloading 10 GB.
        pipe.set_adapters(adapters, adapter_weights=weights)

    # sequential offload moves weights a MODULE at a time instead of a
    # component at a time, so peak host residency is a fraction of the model
    # rather than all of it. It is markedly slower per step and it is the only
    # thing that fits both LoRAs on a 12GB-RAM box - measured 2026-08-23, where
    # model_cpu_offload got OOM-killed part-way through fusing the second LoRA,
    # with exit code 1 and no traceback.
    if sequential:
        logger.info("Using sequential CPU offload (low RAM, slower)")
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.enable_model_cpu_offload()
    _pipe = pipe
    return pipe


def unload_pipeline():
    """Drop the pipeline and give the card back."""
    global _pipe
    if _pipe is None:
        return
    _pipe = None
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- generating ---------------------------------------------------------

def edit(image,
         prompt: str,
         seed: int = 0,
         steps: int | None = None,
         true_cfg_scale: float = 1.0,
         size: int = 512,
         pipe=None):
    """Run one edit. `image` is a PIL image; returns a PIL image.

    `true_cfg_scale` stays at 1.0 with the Lightning LoRA loaded - the same trap
    ADR 0002 records for SDXL-Turbo. A step-distilled model at cfg > 1 does not
    error, it produces over-guided, washed-out output that looks like a prompt
    problem.
    """
    pipe = pipe or load_pipeline()
    steps = steps if steps is not None else LIGHTNING_STEPS

    return pipe(
        image=image.convert("RGB").resize((size, size)),
        prompt=prompt,
        num_inference_steps=steps,
        true_cfg_scale=true_cfg_scale,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).images[0]


def _free():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Resolution the image is fed to the TEXT ENCODER at, independent of render size.
#
# The encoder is a VL model, so the image becomes vision tokens and dominates the
# sequence length. bitsandbytes warns
#     inner dimension (3420) is not aligned for fast kernel with blocksize=64,
#     falling back to slower implementation
# and that fallback dequantises 4-bit weights to bf16 to do a plain matmul -
# throwing away the NF4 saving and allocating over a gigabyte at a time.
# Measured 2026-08-23: at encode size 512 the stage sat at 8.8 GiB with ~2 GiB
# free, so adding even four words of prompt tipped it into OOM.
#
# Encoding smaller shortens the sequence and buys that headroom back. It costs
# little: these tokens only CONDITION the text embedding. Stage 2 still receives
# the image at full render resolution, and that is what the edit is applied to.
ENCODE_SIZE = int(os.environ.get("QWEN_ENCODE_SIZE", "384"))


def encode_only(image, prompts: list[str], size: int = ENCODE_SIZE,
                cache_dir: str = "/models") -> dict:
    """Stage 1: load ONLY the text encoder, embed every prompt, unload it.

    Returns {prompt: (prompt_embeds, prompt_embeds_mask)} with tensors parked on
    the CPU.

    `transformer=None` and `vae=None` keep from_pretrained from instantiating
    the 9.92 GB transformer, which is the entire point: this process must never
    hold the text encoder and the transformer at the same time. On a 12 GB VM
    holding both is not slow, it is fatal - measured 2026-08-23, the whole WSL
    VM went down during LoRA fusion, taking dockerd and every container with it,
    with no traceback anywhere.

    All the prompts share one image on purpose - see the module docstring on why
    embeddings are per (prompt, image) pair.
    """
    from diffusers import QwenImageEditPlusPipeline

    logger.info("Stage 1/2: loading text encoder only, to embed %d prompt(s)",
                len(prompts))
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        REPO, transformer=None, vae=None, dtype=DTYPE,
        cache_dir=cache_dir, token=os.environ.get("HF_TOKEN") or None,
    )
    pipe.to(DEVICE_STR)

    # Resize BEFORE encoding, to the same size stage 2 will render at.
    #
    # The text encoder is a VL model: the image becomes vision tokens, and their
    # count scales with resolution. Feeding a full-size core here inflates the
    # sequence, and bitsandbytes then reports
    #     inner dimension (N) is not aligned for fast kernel with blocksize=64,
    #     falling back to slower implementation
    # That fallback DEQUANTISES each 4-bit weight matrix to bf16 to do an
    # ordinary matmul, which throws away the entire memory saving of NF4 and
    # allocates a full-precision copy per layer. Measured 2026-08-23: the first
    # prompt encoded, the second died with "CUDA driver error: device not
    # ready" - which is bitsandbytes' way of saying out of VRAM, not a driver
    # fault. See the same note in tasks.get_edit_pipeline.
    src = image.convert("RGB").resize((size, size))
    out = {}
    for p in prompts:
        # no_grad is NOT optional here, and its absence does not look like a
        # memory bug - it looks like the text encoder being mysteriously huge.
        #
        # `pipe.__call__` carries @torch.no_grad(); `encode_prompt` called
        # directly does not. Without it every dequantised bf16 weight the
        # bitsandbytes fallback materialises is kept alive for a backward pass
        # that never comes. Measured 2026-08-23: a 5 GB NF4 encoder sat at
        # 8.77 GiB and OOM'd on a further 120 MiB.
        with torch.no_grad():
            embeds, mask = pipe.encode_prompt(prompt=[p], image=[src],
                                              device=torch.device(DEVICE_STR))
        # mask is legitimately None. encode_prompt ends with
        #     if prompt_embeds_mask.all(): prompt_embeds_mask = None
        # so a single unpadded prompt - which is every prompt here - returns no
        # mask at all. Calling .detach() on it raises an AttributeError that
        # reads as if the EMBEDDINGS failed, because both are on one line.
        out[p] = (embeds.detach().cpu(),
                  mask.detach().cpu() if mask is not None else None)
        # Embeddings go to the CPU immediately and the cache is dropped between
        # prompts. The dequantisation fallback leaves large transient buffers
        # behind, and without this the SECOND prompt is the one that dies.
        del embeds, mask
        _free()
        logger.info("  embedded: %s", p)

    del pipe
    _free()
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        logger.info("Text encoder unloaded. VRAM %.1f/%.1f GiB free",
                    free / 1024 ** 3, total / 1024 ** 3)
    return out


def denoise_only(image, embeds: dict, seed: int = 0, steps: int | None = None,
                 size: int = 512, with_angles: bool = True,
                 with_lightning: bool = True, sequential: bool = False,
                 angles_scale: float = 1.0,
                 cache_dir: str = "/models") -> dict:
    """Stage 2: load the transformer (no text encoder) and render every prompt.

    The transformer stays resident across all of them - reloading 9.92 GB per
    cell is the difference between a sheet taking under an hour and taking all
    night.
    """
    from diffusers import (GGUFQuantizationConfig, QwenImageEditPlusPipeline,
                           QwenImageTransformer2DModel)

    token = os.environ.get("HF_TOKEN") or None
    steps = steps if steps is not None else LIGHTNING_STEPS

    logger.info("Stage 2/2: loading transformer from GGUF: %s", GGUF_FILE)
    transformer = QwenImageTransformer2DModel.from_single_file(
        _gguf_url(),
        quantization_config=GGUFQuantizationConfig(compute_dtype=DTYPE),
        config=CONFIG_REPO, subfolder="transformer", dtype=DTYPE,
        cache_dir=cache_dir, token=token,
    )
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        REPO, transformer=transformer, text_encoder=None, dtype=DTYPE,
        cache_dir=cache_dir, token=token,
    )

    adapters, weights = [], []
    if with_angles:
        pipe.load_lora_weights(ANGLES_LORA, weight_name=ANGLES_WEIGHT,
                               adapter_name="angles")
        # angles_scale exists because this LoRA pins POSE as well as framing.
        # At 1.0 it is what you want for a turnaround - the body should not move
        # between directions. For ACTION frames it is the opposition: measured
        # 2026-08-23, camera prompt + pose instruction at scale 1.0 held the
        # framing perfectly and flattened the walk cycle to four near-identical
        # standing poses, while dropping the camera prompt freed the pose and
        # destroyed the framing. Lowering the weight is the dial between them.
        adapters.append("angles"); weights.append(angles_scale)
    if with_lightning:
        pipe.load_lora_weights(LIGHTNING_LORA, weight_name=LIGHTNING_WEIGHT,
                               adapter_name="lightning")
        adapters.append("lightning"); weights.append(LIGHTNING_SCALE)
    if adapters:
        pipe.set_adapters(adapters, adapter_weights=weights)

    if sequential:
        # Refused, not attempted. accelerate's sequential hook moves each
        # parameter to the "meta" device, and a GGUFParameter loses its
        # quant_type on the way, so diffusers dies with a bare `KeyError: None`
        # from GGML_QUANT_SIZES[quant_type] - which names neither offload nor
        # GGUF and sends you looking in the wrong place entirely.
        raise RuntimeError(
            "enable_sequential_cpu_offload is incompatible with a GGUF "
            "transformer (KeyError: None in diffusers/quantizers/gguf). Use a "
            "smaller quant via QWEN_EDIT_GGUF_FILE instead - Q2_K is 7.47 GB "
            "against Q3_K_M's 9.92 GB."
        )
    # Resident on the GPU, NOT offloaded - which is backwards from the usual
    # advice, and right on this machine.
    #
    # enable_model_cpu_offload parks idle components in system RAM. Here that is
    # the scarcer resource: 11.7 GiB of free VRAM against ~9 GiB of available
    # host RAM. Offload therefore trades the plentiful resource for the scarce
    # one, and the bill comes at the END of the forward pass, when accelerate
    # moves the 7.1 GB transformer back to RAM - measured 2026-08-23, denoising
    # finished in 64s and the container then died with "unexpected EOF", the
    # OOM killer, before a single line of the decode ran.
    #
    # Q2_K is 7.1 GB, so it sits on the card with ~4.6 GiB left for activations
    # and the VAE at 384px, and host RAM is never asked for anything.
    # Q3_K_M (9.92 GB) does NOT fit this way; that is what QWEN_EDIT_GGUF_FILE
    # is for.
    if os.environ.get("QWEN_OFFLOAD", "0").lower() in ("1", "true", "yes"):
        logger.info("Using model CPU offload (needs ~10 GiB free host RAM)")
        pipe.enable_model_cpu_offload()
    else:
        logger.info("Placing the pipeline on %s (no CPU offload)", DEVICE_STR)
        pipe.to(DEVICE_STR)
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            logger.info("VRAM after placement: %.1f/%.1f GiB free",
                        free / 1024 ** 3, total / 1024 ** 3)

    # The VAE decodes in tiles, so its peak is set by tile size rather than by
    # image size. Cheap insurance on a card this full.
    try:
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
    except AttributeError:
        pass

    src = image.convert("RGB").resize((size, size))

    # Pass 2a: denoise everything to LATENTS, holding no VAE work at all.
    #
    # output_type="latent" is the important part. Left to itself the pipeline
    # decodes each image immediately, which means the 7.1 GB transformer has to
    # be evicted from VRAM to system RAM so the VAE can take its place - and on
    # a box whose RAM is already full of that same transformer, that eviction is
    # a swap storm. Measured 2026-08-23: denoising finished in 50s and the
    # process then hung in VAE decode until WSL stopped responding to the
    # Windows side entirely. Decode once, at the end, after the transformer is
    # gone.
    latents = {}
    for prompt, (pe, mask) in embeds.items():
        latents[prompt] = pipe(
            image=src,
            # height/width are passed EXPLICITLY. Without them Qwen-Image-Edit
            # picks its own target from the input via an internal ~1-megapixel
            # heuristic, so a 384px input was being rendered at 1024px: 4x the
            # latent tokens, 4x the step time, and a latent that no longer
            # matches the size this code thought it asked for. That mismatch
            # surfaced as a shape error in _unpack_latents, which reads like a
            # bug in the unpacking rather than a silent upscale upstream.
            height=size,
            width=size,
            prompt_embeds=pe.to(DEVICE_STR),
            prompt_embeds_mask=(mask.to(DEVICE_STR)
                                if mask is not None else None),
            num_inference_steps=steps,
            true_cfg_scale=1.0,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            output_type="latent",
        ).images
        logger.info("  denoised: %s", prompt)

    # Pass 2b: drop the transformer, THEN decode.
    #
    # Grab the pipeline's OWN scale factor and image processor before deleting
    # it - deriving them by hand from the VAE config is how the shape error
    # above happened.
    vae = pipe.vae
    vae_scale_factor = pipe.vae_scale_factor
    processor = pipe.image_processor
    height = width = size
    pipe.transformer = None
    del pipe, transformer
    _free()
    logger.info("Transformer released; decoding %d latent(s)", len(latents))

    vae.to(DEVICE_STR)
    out = {}
    for prompt, lat in latents.items():
        out[prompt] = _decode_latent(vae, lat, height, width,
                                     vae_scale_factor, processor)
        logger.info("  decoded: %s", prompt)

    vae.to("cpu")
    _free()
    return out


def _decode_latent(vae, latents, height: int, width: int,
                   vae_scale_factor: int, processor):
    """Unpack Qwen-Image latents and VAE-decode one image.

    `vae_scale_factor` and `processor` are the PIPELINE's own, passed in rather
    than re-derived here. Deriving them from the VAE config produced a factor
    twice too large and a `RuntimeError: shape ... is invalid for input of size
    262144` out of _unpack_latents - an error that points at the unpacking and
    says nothing about the wrong constant feeding it.
    """
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
        QwenImageEditPlusPipeline as _P)

    lat = latents.to(DEVICE_STR, dtype=vae.dtype)

    # Qwen-Image packs latents 2x2 into the channel dim; the pipeline's own
    # helper is the only thing that undoes it correctly.
    if lat.ndim == 3:
        lat = _P._unpack_latents(lat, height, width, vae_scale_factor)

    # The VAE is 3D (it also does video), so the statistics are 5D and the
    # decoded result carries a leading frame axis to drop.
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(
        lat.device, lat.dtype)
    std = 1.0 / torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(
        lat.device, lat.dtype)
    lat = lat / std + mean

    with torch.no_grad():
        image = vae.decode(lat, return_dict=False)[0][:, :, 0]
    return processor.postprocess(image, output_type="pil")[0]


def encode_cells(cells: list, size: int = ENCODE_SIZE,
                 cache_dir: str = "/models") -> dict:
    """Embed a whole sheet's worth of (image, prompt) pairs in ONE encoder load.

    `cells` is a list of dicts with at least `key`, `image` and `prompt`.
    Returns {key: (prompt_embeds, prompt_embeds_mask)} parked on the CPU.

    This exists because a sheet's cells do not share one source image - each
    direction is its own - and `encode_prompt` takes the image as well as the
    prompt. Encoding per direction would mean reloading the text encoder eight
    times, then the 7.1 GB transformer eight times after it. Load each exactly
    once instead: ~90 s of model loading for the whole character rather than
    ~12 minutes.
    """
    from diffusers import QwenImageEditPlusPipeline

    logger.info("Encode pass: %d cell(s), one text-encoder load", len(cells))
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        REPO, transformer=None, vae=None, dtype=DTYPE,
        cache_dir=cache_dir, token=os.environ.get("HF_TOKEN") or None,
    )
    pipe.to(DEVICE_STR)

    out = {}
    # Cache by (id of image, prompt): a sheet repeats the same prompt across
    # directions and the same image across a direction's frames, but the PAIR
    # is what an embedding belongs to.
    for i, c in enumerate(cells):
        src = c["image"].convert("RGB").resize((size, size))
        with torch.no_grad():
            embeds, mask = pipe.encode_prompt(
                prompt=[c["prompt"]], image=[src],
                device=torch.device(DEVICE_STR))
        out[c["key"]] = (embeds.detach().cpu(),
                         mask.detach().cpu() if mask is not None else None)
        del embeds, mask
        _free()
        if (i + 1) % 10 == 0 or i + 1 == len(cells):
            logger.info("  encoded %d/%d", i + 1, len(cells))

    del pipe
    _free()
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        logger.info("Text encoder unloaded. VRAM %.1f/%.1f GiB free",
                    free / 1024 ** 3, total / 1024 ** 3)
    return out


def denoise_cells(cells: list, embeds: dict, seed: int = 0,
                  steps: int | None = None, size: int = 512,
                  with_angles: bool = True, with_lightning: bool = True,
                  angles_scale: float = 1.0, cache_dir: str = "/models") -> dict:
    """Render every cell with ONE transformer load. Returns {key: PIL image}."""
    from diffusers import (GGUFQuantizationConfig, QwenImageEditPlusPipeline,
                           QwenImageTransformer2DModel)

    token = os.environ.get("HF_TOKEN") or None
    steps = steps if steps is not None else LIGHTNING_STEPS

    logger.info("Denoise pass: %d cell(s), loading %s", len(cells), GGUF_FILE)
    transformer = QwenImageTransformer2DModel.from_single_file(
        _gguf_url(),
        quantization_config=GGUFQuantizationConfig(compute_dtype=DTYPE),
        config=CONFIG_REPO, subfolder="transformer", dtype=DTYPE,
        cache_dir=cache_dir, token=token,
    )
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        REPO, transformer=transformer, text_encoder=None, dtype=DTYPE,
        cache_dir=cache_dir, token=token,
    )

    adapters, weights = [], []
    if with_angles:
        pipe.load_lora_weights(ANGLES_LORA, weight_name=ANGLES_WEIGHT,
                               adapter_name="angles")
        adapters.append("angles"); weights.append(angles_scale)
    if with_lightning:
        pipe.load_lora_weights(LIGHTNING_LORA, weight_name=LIGHTNING_WEIGHT,
                               adapter_name="lightning")
        adapters.append("lightning"); weights.append(LIGHTNING_SCALE)
    if adapters:
        pipe.set_adapters(adapters, adapter_weights=weights)

    pipe.to(DEVICE_STR)
    try:
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
    except AttributeError:
        pass
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        logger.info("VRAM after placement: %.1f/%.1f GiB free",
                    free / 1024 ** 3, total / 1024 ** 3)

    # Latents first, transformer released, then decode - same reason as
    # denoise_only: the VAE cannot share the card with a resident transformer.
    latents = {}
    for i, c in enumerate(cells):
        pe, mask = embeds[c["key"]]
        latents[c["key"]] = pipe(
            image=c["image"].convert("RGB").resize((size, size)),
            height=size, width=size,
            prompt_embeds=pe.to(DEVICE_STR),
            prompt_embeds_mask=(mask.to(DEVICE_STR)
                                if mask is not None else None),
            num_inference_steps=steps,
            true_cfg_scale=1.0,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            output_type="latent",
        ).images
        logger.info("  denoised %d/%d  %s", i + 1, len(cells), c["key"])

    vae = pipe.vae
    vae_scale_factor = pipe.vae_scale_factor
    processor = pipe.image_processor
    pipe.transformer = None
    del pipe, transformer
    _free()
    logger.info("Transformer released; decoding %d latent(s)", len(latents))

    vae.to(DEVICE_STR)
    out = {}
    for k in list(latents):
        out[k] = _decode_latent(vae, latents.pop(k), size, size,
                                vae_scale_factor, processor)
    vae.to("cpu")

    # Drop the VAE and every latent explicitly.
    #
    # Leaving them to go out of scope is not enough when this function is called
    # TWICE in one process - once for the turnaround, once for the action pass.
    # The second call then starts against an allocator still holding the first
    # call's arena, and fails on a 72 MiB request with 2.12 GiB nominally free:
    # fragmentation, not exhaustion. Measured 2026-08-23.
    del vae, processor, latents
    _free()
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        logger.info("Denoise pass done. VRAM %.1f/%.1f GiB free",
                    free / 1024 ** 3, total / 1024 ** 3)
    return out


def action_frames(image, frame_prompts: list[str], seed: int = 0,
                  size: int = 512, steps=None, with_angles: bool = False,
                  with_lightning: bool = True, angles_scale: float = 0.5):
    """One direction image -> one frame per prompt, as a plain instruction edit.

    Tried BEFORE reaching for a pose-transfer LoRA, on the grounds that
    Qwen-Image-Edit-2511 is already an instruction-following editor and this
    costs nothing: no extra download, no extra adapters, no pose reference
    images to source.

    The alternative, `lilylilith/AnyPose`, needs TWO input images (character +
    pose reference), two more 295 MB adapters stacked on the two already loaded,
    and a long instruction - and prompt length is VRAM-bound in stage 1 here.
    Its own card also warns it "struggles with 2D art styles", which is exactly
    what a pixel-art sprite is. Escalate to it only if this fails.

    `with_angles` defaults False: the `<sks>` trigger is not used for action
    prompts, and an untriggered adapter is dead weight on the card.

    Every frame is an edit of the SAME source image, never of the previous
    frame. ADR 0005's predecessor measured chaining and rejected it: it makes
    adjacent frames more alike by letting the strip walk away from where it
    started, so the cycle no longer closes.
    """
    embeds = encode_only(image, frame_prompts, size=ENCODE_SIZE)
    return denoise_only(image, embeds, seed=seed, steps=steps, size=size,
                        with_angles=with_angles, with_lightning=with_lightning,
                        angles_scale=angles_scale)


def turnaround_split(image, elevation: str = ISO_ELEVATION, seed: int = 0,
                     directions=None, size: int = 512, steps=None,
                     with_lightning: bool = True, sequential: bool = False):
    """One concept image -> one image per direction, in two memory-disjoint passes.

    This is the shape the whole conveyor has to take on this hardware.
    """
    dirs = list(directions or AZIMUTHS)
    prompts = {d: angle_prompt(d, elevation) for d in dirs}

    # ENCODE_SIZE, not `size`. Encoding is memory-bound on sequence length;
    # rendering is not. They are deliberately decoupled.
    embeds = encode_only(image, list(prompts.values()), size=ENCODE_SIZE)
    rendered = denoise_only(image, embeds, seed=seed, steps=steps, size=size,
                            with_lightning=with_lightning,
                            sequential=sequential)
    return {d: rendered[p] for d, p in prompts.items()}


def turnaround(image, elevation: str = ISO_ELEVATION, seed: int = 0,
               directions=None, size: int = 512, pipe=None) -> dict:
    """One concept image -> one image per compass direction.

    This is the function ADR 0004 built a mesh-and-rig conveyor to approximate.
    The same seed is used for every direction on purpose: the LoRA is supposed
    to be re-rendering one subject, so varying the seed only adds drift.
    """
    pipe = pipe or load_pipeline()
    out = {}
    for d in (directions or list(AZIMUTHS)):
        out[d] = edit(image, angle_prompt(d, elevation), seed=seed,
                      size=size, pipe=pipe)
        logger.info("turnaround: %s done", d)
    return out


# --- self-test ----------------------------------------------------------

def _selftest(argv):
    """Smallest thing that proves the claim: load, and render 8 directions."""
    import argparse

    from PIL import Image

    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--image", help="concept image to turn; required")
    p.add_argument("--out", default="/app/images/_turnaround.png")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--elevation", default=ISO_ELEVATION, choices=sorted(ELEVATIONS))
    # Both of these exist to get UNDER the host-RAM ceiling, which is what
    # actually blocks this machine - not VRAM.
    p.add_argument("--no-lightning", action="store_true",
                   help="skip the 4-step LoRA. Costs speed (needs ~20 steps "
                        "instead of 4) but drops one full adapter injection, "
                        "which is where the OOM kill landed")
    p.add_argument("--sequential", action="store_true",
                   help="sequential CPU offload: much lower peak RAM, slower")
    p.add_argument("--steps", type=int, default=None,
                   help="override step count (default 4 with Lightning)")
    p.add_argument("--single-pass", action="store_true",
                   help="hold the text encoder and transformer in one pipeline. "
                        "Needs ~14 GiB of host RAM; on a 12 GiB VM this kills "
                        "the whole WSL VM. Default is the two-pass split")
    p.add_argument("--directions", default=None,
                   help="comma-separated subset, e.g. s,e,n,w - fewer images "
                        "is a cheaper first proof than all eight")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if not a.image:
        p.error("--image is required; point it at any core_*.png")

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        logger.info("VRAM %.1f/%.1f GiB free", free / 1024 ** 3, total / 1024 ** 3)
    else:
        logger.warning("no CUDA - this will not fit or finish on CPU")

    src = Image.open(a.image)

    # Without the distillation LoRA the model needs its full step budget. 4
    # steps on the undistilled model is not "a bit rough", it is noise.
    steps = a.steps if a.steps else (20 if a.no_lightning else LIGHTNING_STEPS)
    order = ([d.strip() for d in a.directions.split(",")]
             if a.directions else ["s", "se", "e", "ne", "n", "nw", "w", "sw"])
    logger.info("Rendering %d direction(s) at %d steps: %s",
                len(order), steps, " ".join(order))

    if a.single_pass:
        # The naive path: one pipeline holding everything. Kept because it is
        # the right shape on a machine with enough RAM, and because comparing
        # against it is how the split gets validated.
        pipe = load_pipeline(with_lightning=not a.no_lightning,
                             sequential=a.sequential)
        views = {d: edit(src, angle_prompt(d, a.elevation), seed=a.seed,
                         steps=steps, size=a.size, pipe=pipe)
                 for d in order}
    else:
        views = turnaround_split(src, elevation=a.elevation, seed=a.seed,
                                 directions=order, size=a.size, steps=steps,
                                 with_lightning=not a.no_lightning,
                                 sequential=a.sequential)
    sheet = Image.new("RGB", (a.size * len(order), a.size), (255, 255, 255))
    for i, d in enumerate(order):
        sheet.paste(views[d], (i * a.size, 0))
    sheet.save(a.out)
    print(f"wrote {a.out} ({sheet.width}x{sheet.height}, order: {' '.join(order)})")


if __name__ == "__main__":
    _selftest(sys.argv[1:])
