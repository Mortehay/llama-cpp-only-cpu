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

WHAT CHANGED AFTER THE FIRST TWO ADAPTERS FAILED, 2026-08-27

Both adapters produced a lattice of framed cells instead of a subject. The
dataset explained most of it - see `scripts/audit-character-refs.py`, which
found 125 of 149 sprite references were contact sheets - but auditing the data
turned up four things wrong on THIS side of the line as well. All four are
invisible in the loss curve, which is why they survived a run that "worked":

  * CAPTIONS WERE THE FILENAME HASH. `caption_for` appended the file stem, and
    every reference is stored as `ref_sprite_<12 hex>.png`, so each caption
    ended '..., 004228080a22'. Not a weak caption - a harmful one: a unique
    random token per image is a handle for memorising that image, and it
    dilutes the trigger meant to carry the style.
  * EVERY IMAGE WAS CENTRE-CROPPED TO A SQUARE. Only past 3:1 was anything
    refused, so an ordinary 512x1024 full-body reference trained as a torso.
    The default is now to pad.
  * PIXEL ART WAS RESAMPLED WITH LANCZOS. A 64px sprite enlarged to 1024 with a
    smooth filter is a blur, and the blur is what gets learned - the same
    mechanism `curate-training-set.py` documented on the tile side. NEAREST is
    now chosen automatically for hard-edged art.
  * SDXL'S SIZE CONDITIONING WAS A CONSTANT. Every sample claimed to be a
    native-resolution uncropped image, whatever it actually was.

Also added, none of them exotic and all of them standard for this kind of run:
noise offset (SDXL cannot otherwise produce the flat dark and light fields
pixel art is made of), min-SNR loss weighting, a warmup-then-cosine LR schedule
where there was no schedule at all, and shuffled sampling without replacement
in place of `random.randrange` per step.

Usage:
    python train-lora.py --name something2 --data /app/images --out /models/loras
    python train-lora.py --name x --steps 1200 --rank 32 --resolution 768
    python train-lora.py --name x --caption "isometric pixel art character"
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
# THOSE TWO FIGURES PREDATE THE CURRENT TRAINER AND ARE LOW. A real run of THIS
# code, sampled every 10s across 1000 steps at 1024px, rank 32, 114 images,
# AdamW8bit, 46.4M trainable parameters, peaked at 6741 MiB - 6.58 GiB, about
# 1 GiB above the number above. Anyone treating 5.61 as headroom would have
# been wrong in the dangerous direction. Kept side by side rather than
# overwritten, because the gap IS the point: the old figures were quoted
# confidently for two days by someone who had not run the code they describe.
#
# The staging holds, and was watched: VRAM fell to 732 MiB between stages as
# the VAE and text encoders unloaded, then climbed to 6.7 GB for the UNet.
#
# Native resolution costs 0.23 GiB and avoids training the model at a scale it
# was not trained for, so it is the default. There is room above this for a
# larger batch or a higher rank; if either ever OOMs, the ladder down is
# batch -> rank 32/16 -> 1024/768.
DEFAULT_RESOLUTION = 1024

# At or below this many distinct colours, treat the image as hard-edged art and
# enlarge it with NEAREST. 512 is comfortably above any palette-locked sprite
# (the reference set's pixel art sits at 16-32 colours) and far below a render
# or a JPEG board, which measure in the tens of thousands.
PIXEL_ART_MAX_COLORS = 512


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


# What every caption says when the manifest carries no label of its own.
# Overridable with --caption so a run over painted concept art does not claim
# to be pixel art.
DEFAULT_CAPTION = "pixel art sprite"

# Words that are storage naming rather than description. Dropped token by token
# rather than as fixed prefixes, because prefix matching kept losing:
# `ref_map_*` was missing from the original list, and the cells `split-sheets.py`
# writes are named `cell_sprite_sprite_<hash>_000`, which no prefix in a list
# like that will ever strip cleanly.
_STORAGE_WORDS = {"ref", "cell", "core", "sprite", "tile", "map", "sheet",
                  "thumb", "img", "image"}

# Extensions stripped before tokenising. Needed because a label is often a
# whole filename: `0bc2ae64....jpg` is ONE token, and the hex test below fails
# on it for the sake of a dot - so without this the entire hash survives.
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")


def _describing_tokens(text: str) -> list[str]:
    """The words in `text` that describe the picture, with storage naming gone.

    Used for BOTH the label and the filename, which is the point - see
    `caption_for`. Splitting on underscores and hyphens as well as spaces is
    what makes it hold up against `cell_sprite_sprite_<hash>_000`, a name no
    prefix list can handle.
    """
    stem = text
    for ext in _IMAGE_EXTENSIONS:
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break

    kept = []
    for tok in stem.replace("_", " ").replace("-", " ").split():
        # Test the word without its punctuation, keep the word with it, so
        # "ranger," survives intact and "004228080a22," is still recognised as
        # a hash rather than saved by a trailing comma.
        bare = tok.strip(".,;:()[]{}<>'\"")
        low = bare.lower()
        if not low or low in _STORAGE_WORDS or low.isdigit():
            continue
        # A hex blob is storage naming, not a description.
        #
        # The DIGIT is what makes this safe. "pure hex" alone would also match
        # words a person might genuinely type - face, beef, dead, cafe - and
        # eat a real one. Requiring a digit keeps those, and still catches
        # every storage stem, which is random hex and effectively never
        # all-letters.
        if (len(low) >= 3 and all(c in "0123456789abcdef" for c in low)
                and any(c.isdigit() for c in low)):
            continue
        kept.append(tok)

    # A remainder of nothing but two-letter scraps is not a description.
    #
    # `cell 045 of ref_tile_<hash>` is a label this pipeline generates, and
    # every part of it is storage naming except the word "of" - so without this
    # 1,863 tile captions would have read "<trigger> pixel art, of". Measured
    # over the live table, not supposed.
    #
    # A length rule rather than a stopword list, because a list is a guess
    # about a language and this data already contains Ukrainian. Any real
    # description has at least one word of three characters.
    if not any(len(t.strip(".,;:()[]{}<>'\"")) >= 3 for t in kept):
        return []
    return kept


def caption_for(path: str, trigger: str, label: str | None = None,
                body: str = DEFAULT_CAPTION) -> str:
    """Caption with the trigger token, and NOTHING the model cannot use.

    This used to append the filename stem, which sounded harmless and was not.
    Every reference is stored as `ref_sprite_<12 hex>.png`, so after the prefix
    strip the "description" was the hash:

        '<something2-style> pixel art sprite, 004228080a22'

    That is not a weak caption, it is an actively harmful one. Each image got a
    unique random token, so the text encoder had a per-image handle to hang that
    image's specifics on - which is exactly the memorisation a style LoRA is
    trying to avoid, and it dilutes the trigger that is supposed to carry the
    style. `ref_map_*` was not even in the strip list, so those captions read
    'ref map 112233445566'.

    THE LABEL GOES THROUGH THE SAME FILTER, and the first version of this did
    not. It returned a label verbatim on the reasoning that "a person typed
    it". Measured against this database, that premise is false for 2,544 of
    2,555 live references: the label column holds UPLOAD FILENAMES.

        0bc2ae64d34fe36096ed376d5352a7f2.jpg
        9a9ad13c7208251573a6b45c2a9ea5b1 - копія.jpg
        cell 045 of ref_tile_2e2273e3ec26

    So a real training run captioned all 114 of its images

        <mapstyle-style> world map, 07692b93d80fdfec159f369b4e47b210.jpg

    which is the exact harm the paragraph above describes, arriving through the
    door this function left open. The filename was filtered and the label was
    not, and almost every image has a label. A sibling session found it by
    reading the captions of a run it had already started.

    This is not a new heuristic about whether a label "looks like" a filename -
    that judgement is exactly the kind that misfires. It is the SAME rule
    applied to both inputs. A description a person really did type has no
    hex-and-digit tokens in it, so it passes through untouched; that is what
    makes running it over human text safe rather than lossy.

    Swept over all 2,555 live references afterwards: zero captions still carry
    a hash, and 2,534 reduce to the body alone - which is the honest answer,
    because those images have no description anywhere.

    WHAT STILL GETS THROUGH, on 9 rows: `- копія` and `завантаження`, the
    Ukrainian Windows "copy" suffix and a browser's "download". They are
    bookkeeping rather than description, and they are NOT filtered, because
    removing them means a word list in a language this code cannot claim to
    know - and unlike a hash they are SHARED across images, so they give the
    text encoder no per-image handle. The harm this function exists to prevent
    is absent; the untidiness is recorded rather than guessed at.
    """
    kept = _describing_tokens(label) if label and label.strip() else []
    if not kept:
        # Either there was no label or nothing in it described the picture.
        # Fall back to the filename, which gets the identical treatment.
        kept = _describing_tokens(os.path.basename(path))

    if not kept:
        return f"{trigger} {body}".strip()
    return f"{trigger} {body}, {' '.join(kept)}".strip().rstrip(",")


# ---------------------------------------------------------------------------
# Image preparation
# ---------------------------------------------------------------------------

def has_pixel_grid(img, max_scale: int = 16) -> int:
    """Screen pixels per art pixel, or 1 when there is no grid to find.

    Decides the resampling filter. Same method as `measure.pixel_scale`, kept
    local because /app/scripts cannot import the /app package.
    """
    import numpy as np
    a = np.asarray(img.convert("RGBA"))
    both = np.concatenate([a[..., :3].astype(np.int16),
                           a[..., 3:4].astype(np.int16)], axis=2)
    xs = np.nonzero(np.any(both[:, 1:, :] != both[:, :-1, :], axis=(0, 2)))[0] + 1
    ys = np.nonzero(np.any(both[1:, :, :] != both[:-1, :, :], axis=(1, 2)))[0] + 1
    coords = np.concatenate([xs, ys])
    if coords.size == 0:
        return 1
    for s in range(max_scale, 1, -1):
        if float(np.mean(coords % s == 0)) >= 0.98:
            return s
    return 1


def grid_round_trip_error(img, max_scale: int = 12) -> float:
    """How far the image is from being flat NxN blocks, at its best N.

    Real pixel art is flat blocks by construction, so collapsing each block to
    its mean and comparing costs nothing. Imitation pixel art - art that merely
    LOOKS blocky at thumbnail size - carries noise inside every block and
    cannot survive it.

    ONE DIRECTION ONLY. High means "there is noise inside the blocks", which is
    a real finding. Low does NOT mean pixel art: a smooth gradient scores 0.42,
    because neighbouring pixels of a soft ramp are also similar. Use it to
    catch the imitation, never to certify the genuine article - `has_pixel_grid`
    and the palette test are what do that. Stating this because the first
    version of this docstring called it "the honest test for is this pixel
    art", which is precisely the overstatement that produced the signal it
    replaced.

    Measured here, in 0-255 units of mean absolute error:

        hand-made 64x96 references    0.00   (p10 and p90 also 0.00)
        1024x1024 sprite references   7.73
        the 103 recovered cells      10.79

    Nothing decides on this yet; it is logged per run so a dataset's nature is
    recorded at the time it was trained on rather than reconstructed later from
    a disappointing adapter.
    """
    import numpy as np

    rgb = np.asarray(img.convert("RGB")).astype(np.float32)
    h, w = rgb.shape[:2]
    best = float("inf")
    for n in range(2, max_scale + 1):
        if h // n < 8 or w // n < 8:
            break
        hh, ww = (h // n) * n, (w // n) * n
        blocks = rgb[:hh, :ww].reshape(hh // n, n, ww // n, n, 3)
        err = float(np.abs(
            blocks - blocks.mean(axis=(1, 3), keepdims=True)).mean())
        best = min(best, err)
    return 0.0 if best == float("inf") else best


# ABOVE this there is real noise inside the blocks. Read it in that direction
# only: below it means "not noisy", NOT "pixel art" - a smooth gradient scores
# 0.42 here. The gap on the populations that matter is enormous (0.00 against
# 7.73), so the exact value matters far less than their being disjoint.
MAX_GRID_ROUND_TRIP_ERROR = 1.0


def prepare_image(img, resolution: int, fit: str = "pad",
                  resample: str = "auto", background=(128, 128, 128)):
    """Square, resolution-sized RGB, plus the SDXL conditioning that describes it.

    Returns `(image, original_size, crop_offset, filter_name)`. The size and
    crop are not bookkeeping - SDXL is conditioned on them, and the previous
    code passed constants that were not true of the image beside them. The
    filter name is returned so a run can SAY how it resampled: getting that
    wrong is invisible in the loss and obvious in the output.

    THREE THINGS THIS FIXES

    1. PAD, NOT CROP. The old path centre-cropped to a square and only refused
       past 3:1, so a 512x1024 character - a perfectly ordinary full-body
       reference - had its head and feet cut off and trained as a torso. Of the
       483 references measured here the median aspect is 1.45 for sprites, so
       this was not an edge case, it was most of them. Padding keeps the whole
       subject; the padding itself is a flat border, which is the one thing
       these images already have plenty of.

    2. NEAREST FOR PIXEL ART. `Image.LANCZOS` is right for a photograph and
       destructive for a sprite: a 64px sprite blown up to 1024 with LANCZOS is
       a blur, and blur is then what the adapter learns. That is the same
       mechanism `curate-training-set.py` documented on the tile side, where a
       63px median cell upscaled 16x produced "mush", and the adapter learned
       the only sharp thing in the set. NEAREST at the same scale keeps every
       edge hard. 'auto' picks NEAREST when the source has a real pixel grid or
       a small palette AND is being enlarged, LANCZOS otherwise.

    3. HONEST SIZE CONDITIONING. SDXL takes original size and crop offset so it
       can learn that a cropped, low-resolution training image is not what
       'good' looks like. Passing (resolution, resolution, 0, 0) for every
       image tells it every one was a native-resolution uncropped shot. That
       makes the conditioning useless at best; at inference, asking for
       1024x1024 uncropped then means nothing in particular.
    """
    from PIL import Image as _Image

    img = img.convert("RGBA")
    original = img.size

    # Composite onto flat grey rather than black: a transparent sprite on black
    # teaches a black background, the exact artefact this pipeline spends a
    # whole stage removing.
    bg = _Image.new("RGBA", img.size, tuple(background) + (255,))
    flat = _Image.alpha_composite(bg, img).convert("RGB")

    w, h = flat.size
    crop = (0, 0)
    if fit == "crop" and w != h:
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        flat = flat.crop((left, top, left + side, top + side))
        crop = (left, top)
        w = h = side
    elif fit == "pad" and w != h:
        side = max(w, h)
        canvas = _Image.new("RGB", (side, side), tuple(background))
        off = ((side - w) // 2, (side - h) // 2)
        canvas.paste(flat, off)
        flat = canvas
        # Negative offsets: the source sits INSIDE a larger frame, which is the
        # honest description of padding, and the opposite of a crop.
        crop = (-off[0], -off[1])
        w = h = side

    if resample == "nearest":
        filt, name = _Image.NEAREST, "nearest"
    elif resample == "lanczos":
        filt, name = _Image.LANCZOS, "lanczos"
    else:
        # Three signals, any one of which is enough, in decreasing strength:
        #
        #   1. a clean pixel grid  - the art was exported at an integer upscale
        #   2. a small palette     - drawn at 1:1 and palette-locked
        #
        # There WAS a third signal, `edge_softness`, and removing it is the
        # point of this comment. It was added because signal 2 sent all 103
        # recovered Mesgard cells to LANCZOS and I believed those cells were
        # heavily-shaded pixel art that simply had too many colours to be
        # recognised. That belief was never measured, and it is false.
        #
        # The test that settles it: real pixel art survives a round-trip
        # through its own grid. Downsample by the block size and back, and the
        # error is zero, because every block is one flat colour.
        #
        #   hand-made 64x96 references   block error 0.00 (p10 and p90 both 0)
        #   the 103 recovered cells      block error 10.79, at every factor
        #   1024x1024 sprite references  block error  7.73
        #
        # The cells have no grid to preserve. They are an imitation of pixel
        # art with per-pixel noise - a 32x32 patch of one holds 985 distinct
        # colours out of 1024 pixels. NEAREST does not protect a grid there; it
        # magnifies the noise, and these get upscaled 4-6x to reach 1024.
        #
        # Worse, the signal was inverted. Across all 483 references it fired
        # alone on 65 files, and every one of those has block error >= 1.24
        # (median 9.42). It never once fired on art with a real grid - signals
        # 1 and 2 had already claimed those - so its entire effect was to send
        # grid-less art to NEAREST. Do not re-add it without the round-trip
        # measurement above.
        #
        # `getcolors` returns None above its cap, which reads as "too many".
        pixelish = (has_pixel_grid(img) > 1
                    or flat.getcolors(PIXEL_ART_MAX_COLORS) is not None)
        if pixelish and resolution > w:
            filt, name = _Image.NEAREST, "nearest"
        else:
            filt, name = _Image.LANCZOS, "lanczos"

    return flat.resize((resolution, resolution), filt), original, crop, name


# ---------------------------------------------------------------------------
# Stage 1: cache latents and embeddings
# ---------------------------------------------------------------------------

def cache_inputs(paths, captions, resolution, cache_path, dtype,
                 fit="pad", resample="auto"):
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

    latents, time_ids = [], []
    filters, grid_errors = [], []
    for i, p in enumerate(paths):
        img = Image.open(p)
        prepared, original, crop, filt = prepare_image(
            img, resolution, fit=fit, resample=resample)
        filters.append(filt)
        # Measured on the SOURCE, before resampling. Afterwards every image has
        # been through a filter and the answer would describe that instead.
        grid_errors.append(grid_round_trip_error(img))

        # SDXL's micro-conditioning, per image and TRUE of this image:
        # original height/width, crop top/left, target height/width. The old
        # code sent (resolution, resolution, 0, 0, resolution, resolution) for
        # every sample, which claimed each one was already native size and
        # uncropped.
        time_ids.append([original[1], original[0], crop[1], crop[0],
                         resolution, resolution])

        import numpy as np
        arr = np.asarray(prepared).astype("float32") / 127.5 - 1.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            # VAE stays fp32: SDXL's VAE produces NaNs in fp16, a known issue
            # and an unpleasant one to debug through a training loss curve.
            dist = vae.encode(t).latent_dist
            lat = dist.sample() * vae.config.scaling_factor
        latents.append(lat.squeeze(0).to(torch.float32).cpu())
        if (i + 1) % 5 == 0 or i + 1 == len(paths):
            logger.info("  encoded %d/%d images", i + 1, len(paths))

    # Say which filter each image got. Resampling pixel art with a smooth
    # filter is invisible in the loss curve and ruinous in the output, so the
    # split between the two belongs in the log where it can be checked.
    n_near = filters.count("nearest")
    logger.info("  resampled %d nearest / %d lanczos (fit=%s)",
                n_near, len(filters) - n_near, fit)

    # Whether this dataset is pixel art AT ALL, recorded while it is being
    # trained on. Nothing here refuses to run - that is the user's call and the
    # measurement may be wrong about some new kind of art - but a run whose
    # images cannot survive a round-trip through their own grid is producing a
    # style LoRA for imitation pixel art, and the log should say so at the time
    # rather than leaving it to be guessed from a disappointing adapter.
    #
    # Read the measure in ONE direction only. A high error proves the blocks
    # are not flat - there is noise inside them. A LOW error does not prove
    # pixel art: a smooth gradient scores 0.42 here, because neighbouring
    # pixels of a soft ramp are also similar. Signals 1 and 2 are what identify
    # pixel art; this identifies its imitation.
    if grid_errors:
        noisy = sum(1 for e in grid_errors if e > MAX_GRID_ROUND_TRIP_ERROR)
        med = sorted(grid_errors)[len(grid_errors) // 2]
        logger.info("  block noise: %d/%d images have noise inside their "
                    "apparent blocks (median error %.2f; flat-block art "
                    "scores 0.00)", noisy, len(grid_errors), med)
        if noisy == len(grid_errors):
            logger.warning("  EVERY image in this dataset carries per-pixel "
                           "noise. If it was meant to be pixel art, it is an "
                           "imitation of it, and training learns the noise "
                           "too - see .ai/decisions/0009.")

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
             "captions": captions, "resolution": resolution,
             "time_ids": time_ids}, cache_path)
    logger.info("  cached to %s", cache_path)
    return cache_path


# ---------------------------------------------------------------------------
# Stage 2: train
# ---------------------------------------------------------------------------

def train(cache_path, out_dir, name, steps, rank, lr, batch_size, dtype, seed,
          resume_from=None, noise_offset=0.0, min_snr=0.0, warmup=0):
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
    # Per-image micro-conditioning written by stage 1. Falls back to the old
    # constant so a cache from a previous version still loads rather than
    # dying on a KeyError halfway through a queued run.
    all_time_ids = blob.get("time_ids") or [
        [resolution, resolution, 0, 0, resolution, resolution]] * n
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

    # Warmup then cosine decay. There was no schedule at all: a constant 1e-4
    # from step 1. That costs most on the two runs this project actually does -
    # the first steps of a fresh adapter, where a full-size step into randomly
    # initialised LoRA weights is noise, and every INCREMENTAL run, where a
    # constant large LR on a handful of new images drags a trained adapter
    # toward whatever arrived most recently.
    warmup = min(max(warmup, 0), max(steps - 1, 0))

    def lr_at(step: int) -> float:
        if warmup and step <= warmup:
            return lr * step / warmup
        progress = (step - warmup) / max(steps - warmup, 1)
        return lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    sched = DDPMScheduler.from_pretrained(BASE, subfolder="scheduler",
                                          cache_dir=CACHE_DIR)

    # Per-image micro-conditioning, on the device once rather than per step.
    time_ids = torch.tensor(all_time_ids, device=device, dtype=dtype)

    # Min-SNR needs the schedule's signal-to-noise ratio at every timestep.
    alphas = sched.alphas_cumprod.to(device)
    snr_all = alphas / (1.0 - alphas)

    # Sample WITHOUT replacement, reshuffling each pass. The old loop drew
    # `random.randrange(n)` per step, so with 191 images and 1000 steps a few
    # images were never seen at all and others were seen a dozen times - a
    # silent, seed-dependent reweighting of the dataset. An epoch costs nothing
    # and removes the variance.
    order: list[int] = []

    unet.train()
    t0 = time.time()
    running = 0.0
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        idx = []
        while len(idx) < batch_size:
            if not order:
                order = list(range(n))
                random.shuffle(order)
            idx.append(order.pop())
        lat = torch.stack([latents[i] for i in idx]).to(device, dtype=dtype)
        emb = torch.stack([embeds[i] for i in idx]).to(device, dtype=dtype)
        pol = torch.stack([pooled[i] for i in idx]).to(device, dtype=dtype)
        tid = time_ids[idx]

        noise = torch.randn_like(lat)
        if noise_offset:
            # SDXL cannot produce a very dark or very light FLAT field, because
            # its schedule never quite reaches zero SNR - the model always sees
            # a little of the image's mean and never has to predict it. That is
            # a footnote for photographs and a real problem for pixel art,
            # which is mostly flat fields with pure black outlines. Offsetting
            # the noise per channel forces the model to learn the mean.
            noise = noise + noise_offset * torch.randn(
                (lat.shape[0], lat.shape[1], 1, 1),
                device=lat.device, dtype=lat.dtype)
        t = torch.randint(0, sched.config.num_train_timesteps,
                          (lat.shape[0],), device=device).long()
        noisy = sched.add_noise(lat, noise, t)

        pred = unet(
            noisy, t,
            encoder_hidden_states=emb,
            added_cond_kwargs={"text_embeds": pol, "time_ids": tid},
        ).sample

        # SDXL base predicts epsilon. Reading the target from the scheduler
        # rather than assuming it keeps this correct if the base ever changes.
        target = (noise if sched.config.prediction_type == "epsilon"
                  else sched.get_velocity(lat, noise, t))

        if min_snr:
            # A flat MSE over uniformly sampled timesteps is not a balanced
            # objective. Under epsilon-prediction the barely-noised steps are
            # the hard ones - the noise to predict is a small part of what the
            # model is looking at - so they carry the largest loss and dominate
            # the gradient, pulling against the noisier steps. Min-SNR-gamma
            # clamps their weight to gamma/SNR, which converges faster and more
            # evenly. gamma=5 is the published default.
            snr = snr_all[t]
            w = torch.clamp(snr, max=min_snr)
            # The divisor differs by objective: epsilon-prediction divides by
            # SNR, v-prediction by SNR+1. Reading it from the scheduler for the
            # same reason `target` above does - so this stays correct if the
            # base checkpoint ever changes.
            w = w / (snr if sched.config.prediction_type == "epsilon"
                     else snr + 1)
            per = F.mse_loss(pred.float(), target.float(),
                             reduction="none").mean(dim=(1, 2, 3))
            loss = (per * w.float()).mean()
        else:
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
    p.add_argument("--caption", default=DEFAULT_CAPTION,
                   help="the body of every caption, after the trigger. Set it "
                        "to what the dataset IS - a run over painted concept "
                        "art captioned 'pixel art sprite' is teaching the "
                        "trigger a word the images do not show")
    p.add_argument("--fit", choices=["pad", "crop"], default="pad",
                   help="pad (default) keeps the whole subject and letterboxes "
                        "it; crop is the old behaviour and cuts the head and "
                        "feet off anything taller than it is wide")
    p.add_argument("--resample", choices=["auto", "nearest", "lanczos"],
                   default="auto",
                   help="auto uses NEAREST for hard-edged art being enlarged "
                        "and LANCZOS otherwise. LANCZOS on a 64px sprite blown "
                        "up to 1024 is a blur, and blur is what gets learned")
    p.add_argument("--noise-offset", type=float, default=0.05,
                   help="forces the model to learn flat dark and flat light "
                        "fields, which SDXL otherwise cannot produce. 0 to "
                        "disable")
    p.add_argument("--min-snr", type=float, default=5.0,
                   help="min-SNR-gamma loss weighting, so the low-noise steps "
                        "that carry style detail are not drowned by the "
                        "high-noise ones. 0 to disable")
    p.add_argument("--warmup", type=int, default=None,
                   help="linear LR warmup steps before the cosine decay; "
                        "defaults to 5%% of --steps")
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
    labels: dict[str, str] = {}
    if a.files:
        # `path` or `path<TAB>label`. The tab form is optional so an existing
        # manifest still reads correctly; when a label IS present it becomes
        # the caption, which is the only way this script can be told what an
        # image actually shows.
        paths = []
        with open(a.files) as f:
            for ln in f:
                ln = ln.rstrip("\n")
                if not ln.strip():
                    continue
                path, _, label = ln.partition("\t")
                path = path.strip()
                paths.append(path)
                if label.strip():
                    labels[path] = label.strip()
        paths = [p for p in paths if os.path.isfile(p)]
        logger.info("dataset from manifest %s: %d image(s), %d labelled",
                    a.files, len(paths), len(labels))
    else:
        paths = find_images(a.data, tuple(s.strip() for s in a.pattern.split(",")))
        logger.info("dataset from glob %r: %d image(s)", a.pattern, len(paths))

    if len(paths) < a.min_images:
        sys.exit(f"only {len(paths)} image(s) matched {a.pattern} in {a.data}; "
                 f"need at least {a.min_images}. Upload more references - "
                 f"measurement works from three examples, but training does "
                 f"not.")

    trigger = a.trigger or f"<{a.name}-style>"
    captions = [caption_for(p, trigger, labels.get(p), a.caption) for p in paths]
    logger.info("%d images, trigger %r", len(paths), trigger)
    logger.info("example caption: %s", captions[0])

    os.makedirs(a.out, exist_ok=True)
    cache_path = os.path.join(a.out, f".{a.name}-cache.pt")
    warmup = a.warmup if a.warmup is not None else max(1, a.steps // 20)

    try:
        cache_inputs(paths, captions, a.resolution, cache_path, dtype,
                     fit=a.fit, resample=a.resample)
        out = train(cache_path, a.out, a.name, a.steps, a.rank, a.lr,
                    a.batch_size, dtype, a.seed, resume_from=a.resume,
                    noise_offset=a.noise_offset, min_snr=a.min_snr,
                    warmup=warmup)
    finally:
        # The cache is tens of MB per image and worthless once training ends.
        if os.path.exists(cache_path):
            os.remove(cache_path)

    # Everything that changes what the adapter learned, recorded beside it.
    # An adapter whose preparation settings are unknown cannot be compared with
    # another one, and comparing them is the whole point of a second run.
    meta = {"name": a.name, "base": BASE, "trigger": trigger,
            "steps": a.steps, "rank": a.rank, "lr": a.lr,
            "resolution": a.resolution, "images": len(paths),
            "caption": a.caption, "fit": a.fit, "resample": a.resample,
            "noise_offset": a.noise_offset, "min_snr": a.min_snr,
            "warmup": warmup, "seed": a.seed, "batch_size": a.batch_size,
            "resumed_from": a.resume, "labelled_images": len(labels),
            "weights": out}
    with open(os.path.join(a.out, f"{a.name}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
