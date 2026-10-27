"""Is the whole figure in frame, or did the model render a close-up?

WHY THIS IS NOT A MEASUREMENT OF THE ALPHA MASK
------------------------------------------------
The angles LoRA frames every subject edge to edge, so a torso close-up and a
full body produce the SAME bounding box: 478-512 of 512 tall, filling the
canvas. Six geometric detectors were measured against hand labels and all
six failed for that one reason - edge contact, `figure_ratio` (area), bottom
width, shoulder width, post-redraw geometry, and a lowered area threshold.

`figure_ratio` is the instructive failure: it is NON-MONOTONIC in the defect.
A crop gains torso pixels and loses the legs, and a full body with spread
arms carries more opaque area than a torso does, so cropped cells measured
0.980-1.726 against complete cells 0.900-1.176. No threshold separates them;
the best cut catches 4 of 10. See sheet-quality contract 1.1.

So the question is semantic - are the feet in frame - and it needs a model.

WHY THE BOTTOM BAND
-------------------
CLIP resizes its input to 224. Boots at the foot of a 512 canvas survive that
as a handful of pixels, and scoring the whole cell measured 67.6% against a
65% base rate - useless. Scoring the bottom 30% at 2x measured clean
separation on the first subject. The view was the defect, not the model.

WHAT IT IS WORTH, MEASURED
--------------------------
Swept over 68 hand-labelled cells across TWO characters: 12 of 13 cropped
caught, 0 of 55 complete wrongly flagged, at a fixed cut near 0.49.

It is NOT clean, and the limit is structural rather than a tuning problem:
one cropped cell scores 0.722 while a complete one scores 0.487, so no
threshold separates every pair. The operating point sits on that boundary -
at 0.49 the cost is one wasted redraw per boundary case, at 0.45 it is one
missed crop. 0.49 is the default because a wasted redraw is cheaper than a
bad cell shipping.

Per-sheet calibration against the turnaround panels was measured and does
NOT help: the references score 0.9931 and 0.9932 on the two subjects, so
there is no offset to correct, and subtracting or dividing by them changed
nothing. Do not rebuild that idea.

COST
----
1.18 s/cell on CPU, measured over 336 cells - the same order as BiRefNet's
1.8 s/cell. The model is parked in host RAM and never placed on the GPU:
this runs inside the denoise subprocess, which already holds the transformer
on a card with no room to spare.
"""
from __future__ import annotations

import logging
import os

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

REPO = os.environ.get("FRAMING_CLIP_REPO", "openai/clip-vit-large-patch14")
MODELS_DIR = os.environ.get("MODELS_DIR", "/models")

# Off by default. Turning it on changes every build's cost and its redraw
# behaviour, and the detector is good rather than clean - that is the
# owner's call, not a default to slip in. `FRAMING_CHECK=1` enables it.
ENABLED = os.environ.get("FRAMING_CHECK", "0").strip().lower() in (
    "1", "true", "yes")

# Below this, the feet are judged to be out of frame. See the module
# docstring for what the number costs in each direction.
FRAMING_MIN = float(os.environ.get("FRAMING_MIN", "0.49"))

# The bottom share of the canvas to read, and how much to magnify it.
BAND = float(os.environ.get("FRAMING_BAND", "0.30"))
ZOOM = int(os.environ.get("FRAMING_ZOOM", "2"))

PROMPTS = ("leather boots and feet standing on the ground",
           "a robe hem and fabric, no feet visible")

_model = None
_proc = None
_unavailable = False


def _load():
    """CLIP, parked on CPU. None when it cannot be had.

    A missing model must not fail a build: this check is advisory and a box
    without the weights still produces sheets. `make provision` fetches it.
    """
    global _model, _proc, _unavailable
    if _unavailable:
        return None, None
    if _model is not None:
        return _model, _proc
    try:
        from transformers import CLIPModel, CLIPProcessor
        _model = CLIPModel.from_pretrained(REPO, cache_dir=MODELS_DIR).eval()
        _proc = CLIPProcessor.from_pretrained(REPO, cache_dir=MODELS_DIR)
        logger.info("framing: CLIP loaded from %s (parked on CPU)", REPO)
    except Exception as e:
        _unavailable = True
        logger.warning("framing: CLIP unavailable (%s); the framing check is "
                       "skipped and cells are not judged on it", e)
        return None, None
    return _model, _proc


def _band(img: Image.Image) -> Image.Image:
    """The bottom slice, magnified, flattened onto white.

    Flattened because CLIP has no alpha channel; white matches the backdrop
    the renders already use, so a cut-out cell and a raw one look alike to it.
    """
    rgba = img.convert("RGBA")
    crop = rgba.crop((0, int(rgba.height * (1.0 - BAND)),
                      rgba.width, rgba.height))
    if ZOOM > 1:
        crop = crop.resize((crop.width * ZOOM, crop.height * ZOOM),
                           Image.Resampling.LANCZOS)
    flat = Image.new("RGB", crop.size, (255, 255, 255))
    flat.paste(crop, mask=crop.split()[3])
    return flat


def feet_in_frame(img: Image.Image) -> float | None:
    """P(the feet are in frame), or None when the model is unavailable."""
    if not ENABLED:
        return None
    model, proc = _load()
    if model is None:
        return None
    try:
        import torch
        inp = proc(text=list(PROMPTS), images=_band(img),
                   return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model(**inp)
        return float(out.logits_per_image.softmax(dim=1)[0][0])
    except Exception as e:
        # Advisory: a failure here must not lose a cell that cost GPU time.
        logger.warning("framing: could not score a cell (%s)", e)
        return None


def is_close_up(score: float | None) -> bool:
    """True only when the model actually answered and answered low."""
    return score is not None and score == score and score < FRAMING_MIN
