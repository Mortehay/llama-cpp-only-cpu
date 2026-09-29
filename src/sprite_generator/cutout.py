"""AI background removal for object cutouts: BiRefNet (MIT), hard-edged.

Spec: .ai/specs/transparent-cutouts/ ("Segmenter" outcome).

WHY NOT THE FLOOD FILL. `tasks.remove_background` keys the colour most
corners agree on and clears what is connected to the border. It cannot clear:
a painted transparency checkerboard (two tones), a wall or scene filling the
frame, a floor with a shadow, or a pocket of backdrop enclosed by the subject
(the white inside a drawn bow). Measured 2026-09-29 on those exact failures,
BiRefNet cut the subject out of all but one (a garden-grid scene), and changed
nothing visible on five already-clean cutouts. ToonOut, its anime fine-tune,
was measured too and was WORSE here: it kept walls, mountains and rooms.

PIXEL ART NEEDS HARD EDGES. The network's mask is soft; it is snapped to
0/255 at 128, so no semi-transparent fringe reaches the game.

VRAM. The model is ~0.9 GB in fp16 and runs for ~0.3 s per image, but it
must not sit on the card: the slow Qwen core refuses to start below
QWEN_MIN_FREE_MB free. So it is parked in host RAM and moved to the GPU for
each call, the same park/place trick the persistent Qwen process uses.

CUTOUT_ENGINE=floodfill turns it off (the rollback switch). Without CUDA
(the API process, which serves the manual crop route) it is never used.
"""

from __future__ import annotations

import glob
import logging
import os
import threading

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

ENGINE = os.environ.get("CUTOUT_ENGINE", "birefnet").strip().lower()
MODELS_DIR = os.environ.get("HF_HUB_CACHE") or "/models"
_WEIGHTS_GLOB = os.path.join(MODELS_DIR, "models--ZhengPeng7--BiRefNet", "snapshots", "*")
INPUT_SIZE = 1024  # what BiRefNet was trained at
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

_model = None
_lock = threading.Lock()
_unavailable: str | None = None


def weights_dir() -> str | None:
    hits = sorted(glob.glob(_WEIGHTS_GLOB))
    return hits[-1] if hits else None


def enabled(device: str) -> bool:
    """Use the segmenter here? Only on a CUDA worker, with weights present."""
    return (ENGINE == "birefnet" and device == "cuda" and _unavailable is None
            and weights_dir() is not None)


def _load():
    global _model, _unavailable
    if _model is not None:
        return _model
    import torch
    from transformers import AutoModelForImageSegmentation
    path = weights_dir()
    try:
        m = AutoModelForImageSegmentation.from_pretrained(
            path, trust_remote_code=True, local_files_only=True)
    except Exception as e:
        # Missing einops/kornia/timm, or broken weights: stop trying, fall back.
        _unavailable = f"BiRefNet failed to load from {path}: {e}"
        logger.error("cutout: %s - using the flood fill", _unavailable)
        raise
    _model = m.half().eval()  # parked in host RAM; placed per call
    logger.info("cutout: BiRefNet loaded from %s (parked on CPU)", path)
    return _model


def mask(rgb: Image.Image) -> np.ndarray:
    """Soft foreground mask (uint8 0-255) at the image's own size."""
    import torch
    with _lock:
        m = _load()
        x = rgb.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        t = torch.from_numpy(np.asarray(x, dtype=np.float32) / 255.0).permute(2, 0, 1)
        for c in range(3):
            t[c] = (t[c] - MEAN[c]) / STD[c]
        try:
            m.to("cuda")
            with torch.no_grad():
                pred = m(t.unsqueeze(0).to("cuda").half())[-1].sigmoid()
            soft = (pred[0, 0].float().cpu().numpy() * 255).astype(np.uint8)
        finally:
            m.to("cpu")
            torch.cuda.empty_cache()
    return np.asarray(Image.fromarray(soft).resize(rgb.size, Image.BILINEAR))


def cut_out(img: Image.Image) -> Image.Image:
    """RGBA with a hard 0/255 alpha from the segmenter.

    An RGBA input is flattened onto white first, so an image that already had
    some background keyed away is judged as one picture.
    """
    im = img.convert("RGBA")
    flat = Image.new("RGBA", im.size, (255, 255, 255, 255))
    flat.alpha_composite(im)
    rgb = flat.convert("RGB")
    alpha = np.where(mask(rgb) >= 128, 255, 0).astype(np.uint8)
    return Image.fromarray(np.dstack([np.asarray(rgb), alpha]), "RGBA")
