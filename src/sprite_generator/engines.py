"""Which renderer draws a sheet, whether it can run, and what it actually did.

TWO LANES, ONE CONVEYOR
-----------------------
`diffusers` is the lane that has always existed: a GGUF-quantised
Qwen-Image-Edit transformer loaded through diffusers, with the 4-step
Lightning LoRA. `comfyui-hq` renders the same two stages - the turnaround and
the action frames - through a ComfyUI server at higher precision and without
the distillation.

EVERYTHING ELSE IS SHARED CODE. Cutout, the per-frame checks, the redraw
rounds, pixelate and the atlas do not know which engine drew the pixels, and
must not learn: the moment a check behaves differently per lane, a comparison
between the lanes stops meaning anything.

IT NEVER FALLS BACK
-------------------
If ComfyUI is down, an HQ job FAILS and says so. A lane that silently degrades
to diffusers is indistinguishable from the HQ lane not working - the caller
waits hours, gets a sheet that looks like today's, and has no way to tell
whether the quality lane ran at all. That is the one outcome the `engine`
field exists to prevent, so there is no fallback path in this module to find
later and "improve".

AVAILABILITY IS A RUNTIME QUESTION, NOT A VALIDATION ONE
--------------------------------------------------------
`JobSpec` rejects an unknown engine NAME. This module answers whether a known
engine can run on THIS box right now - is the server reachable, are the
weights on disk. The two give different errors on purpose: a typo and an
unprovisioned box are different problems and a caller can only act on one of
them.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

COMFY_URL = os.environ.get("COMFYUI_URL", "http://comfyui:8188")
COMFY_TIMEOUT_S = float(os.environ.get("COMFYUI_PROBE_TIMEOUT_S", "4"))

# Where the HQ weights live. A separate tree from /models on purpose: it sits
# on D:, which has the room, while /models is on a C: that has run out twice.
HQ_MODELS = os.environ.get("COMFYUI_MODELS_DIR", "/models-hq")

# The transformer this lane draws with, and the settings it draws at. These
# are the values echoed to the caller, so they must be read from here and from
# the workflow actually submitted - never retyped into a response.
HQ_UNET = os.environ.get("COMFYUI_HQ_UNET",
                         "qwen_image_edit_2511_fp8mixed.safetensors")
HQ_PRECISION = os.environ.get("COMFYUI_HQ_PRECISION", "fp8mixed")
HQ_TEXT_ENCODER = os.environ.get(
    "COMFYUI_HQ_TEXT_ENCODER", "qwen_2.5_vl_7b_fp8_scaled.safetensors")
HQ_VAE = os.environ.get("COMFYUI_HQ_VAE", "qwen_image_vae.safetensors")
HQ_ANGLES_LORA = os.environ.get(
    "COMFYUI_HQ_ANGLES_LORA",
    "qwen-image-edit-2511-multiple-angles-lora.safetensors")

# No Lightning on this lane. That is the point of it: the 4-step distillation
# forces true_cfg_scale=1.0, and at CFG 1.0 there is no classifier-free
# guidance, so the conveyor has NO NEGATIVE PROMPT and cannot be given one -
# every "no extra limbs" instruction lands in the positive conditioning
# instead. Real steps and real CFG are what buy that back.
HQ_STEPS = int(os.environ.get("COMFYUI_HQ_STEPS", "20"))
HQ_CFG = float(os.environ.get("COMFYUI_HQ_CFG", "4.0"))
HQ_ANGLES_SCALE = float(os.environ.get("COMFYUI_HQ_ANGLES_SCALE", "1.0"))
HQ_NEGATIVE = os.environ.get(
    "COMFYUI_HQ_NEGATIVE",
    "extra limbs, extra legs, extra arms, deformed hands, melted face, "
    "blurry, duplicate character, contact sheet, multiple views, text, "
    "watermark, frame, border")


def comfy_reachable() -> tuple[bool, str]:
    """Is the ComfyUI server answering? Returns (ok, detail)."""
    try:
        req = urllib.request.Request(f"{COMFY_URL}/system_stats")
        with urllib.request.urlopen(req, timeout=COMFY_TIMEOUT_S) as r:
            json.load(r)
        return True, ""
    except urllib.error.URLError as e:
        return False, f"ComfyUI at {COMFY_URL} is not answering ({e.reason})"
    except Exception as e:  # noqa: BLE001
        return False, f"ComfyUI at {COMFY_URL} is not answering ({e})"


# Smallest plausible size for each weight, in bytes. EXISTENCE IS NOT ENOUGH:
# a half-downloaded file exists, passes `os.path.exists`, and then fails at
# load with a corrupt-safetensors error - hours after the job was accepted,
# and looking nothing like "the download did not finish". Observed exactly
# that during this lane's own provisioning, with the transformer at 11.8 of
# 19.12 GiB reporting as present.
#
# Floors rather than exact sizes: a re-quantised or re-uploaded file may
# differ by metadata, and the job of this check is to catch a truncation, not
# to pin a build. The real sizes are 19.12 GiB, 8.74 GiB, 0.24 GiB, 0.27 GiB.
# A FLOOR PER KIND CANNOT WORK FOR THE TRANSFORMER, which is the one file this
# lane is meant to swap. The first version hard-coded 17 GiB from the fp8
# file's 19.12 GiB and then reported the 15.70 GiB Q6_K as truncated - a check
# calibrated to one model, in the slot whose whole purpose is holding
# different ones.
#
# So known transformers carry their own expected size, and anything
# unregistered falls back to a weak sanity floor. Registering a size is what
# buys real truncation detection; not registering one is not an error, it just
# means a half-downloaded unknown file is only caught if it is tiny.
EXPECTED_BYTES = {
    "qwen_image_edit_2511_fp8mixed.safetensors": 20_533_762_817,
    "qwen-image-edit-2511-Q6_K.gguf": 16_852_417_120,
    "qwen-image-edit-2511-Q3_K_S.gguf": 9_218_914_912,
    "qwen_2.5_vl_7b_fp8_scaled.safetensors": 9_384_670_680,
    "qwen_image_vae.safetensors": 253_806_246,
    "qwen-image-edit-2511-multiple-angles-lora.safetensors": 295_140_688,
}

# Used only when the file is not in EXPECTED_BYTES above.
MIN_BYTES = {
    "unet": 1 * 1024 ** 3,
    "text_encoders": 1 * 1024 ** 3,
    "vae": 50 * 1024 ** 2,
    "loras": 50 * 1024 ** 2,
}

# How far under its expected size a file may sit and still count as complete.
# Not zero: a re-upload or a different build of the same quant can differ by
# metadata, and this check exists to catch a truncation, not to pin a build.
SIZE_TOLERANCE = 0.98


def missing_weights() -> list[str]:
    """HQ weight files that are absent or obviously incomplete.

    Returns a description per problem, not just a name, so the 422 says
    whether a file is missing or truncated - two different operator actions.
    """
    want = [
        ("unet", HQ_UNET), ("text_encoders", HQ_TEXT_ENCODER),
        ("vae", HQ_VAE), ("loras", HQ_ANGLES_LORA),
    ]
    out = []
    for kind, name in want:
        path = os.path.join(HQ_MODELS, kind, name)
        if not os.path.exists(path):
            out.append(name)
            continue
        try:
            size = os.path.getsize(path)
        except OSError as e:  # noqa: PERF203
            out.append(f"{name} (unreadable: {e})")
            continue
        expected = EXPECTED_BYTES.get(os.path.basename(name))
        floor = (expected * SIZE_TOLERANCE if expected
                 else MIN_BYTES.get(kind, 0))
        if size < floor:
            detail = (f"expected ~{expected / 1024 ** 3:.2f} GiB"
                      if expected else
                      f"expected at least {floor / 1024 ** 3:.1f} GiB")
            out.append(f"{name} (incomplete: {size / 1024 ** 3:.2f} GiB, "
                       f"{detail} - a download still in progress looks "
                       f"exactly like this)")
    return out


def unavailable_reason(engine: str) -> str | None:
    """Why this engine cannot run here, or None when it can.

    Checks the WEIGHTS first and the server second, because a missing file is
    a definite, actionable answer while an unreachable server might be a
    container still starting - and naming the wrong one sends an operator to
    the wrong place.
    """
    if engine == "diffusers":
        return None
    if engine != "comfyui-hq":
        return f"unknown engine {engine!r}"

    missing = missing_weights()
    if missing:
        return (f"{len(missing)} HQ weight file(s) not on disk under "
                f"{HQ_MODELS}: {', '.join(missing)}")
    ok, detail = comfy_reachable()
    if not ok:
        return detail
    return None


def provenance(engine: str, *, steps: int | None = None,
               cfg: float | None = None, loras: list | None = None,
               model_file: str | None = None,
               precision: str | None = None) -> dict:
    """What actually drew this sheet, for the atlas and the job record.

    The arguments exist so a caller passes the values IT used rather than the
    configured defaults. A job retried at different settings must echo the
    pixels that exist, not the settings a later reader happens to have.

    `loras` is a list of {name, scale}, never bare names: the scale is
    load-bearing here - `angles_scale` pins pose as well as camera - so a name
    alone discards the number most likely to explain a bad sheet.
    """
    if engine == "diffusers":
        # Imported lazily: this module is imported by the API process, which
        # must not pull torch in at import time.
        import qwen_edit
        return {
            "name": "diffusers",
            "model_file": os.path.basename(qwen_edit.GGUF_FILE),
            "precision": "gguf",
            "steps": steps if steps is not None else qwen_edit.LIGHTNING_STEPS,
            "cfg": cfg if cfg is not None else 1.0,
            "loras": loras if loras is not None else [
                {"name": qwen_edit.ANGLES_WEIGHT, "scale": 1.0},
                {"name": qwen_edit.LIGHTNING_WEIGHT,
                 "scale": qwen_edit.LIGHTNING_SCALE},
            ],
        }
    return {
        "name": "comfyui-hq",
        "model_file": model_file or HQ_UNET,
        "precision": precision or HQ_PRECISION,
        "steps": steps if steps is not None else HQ_STEPS,
        "cfg": cfg if cfg is not None else HQ_CFG,
        "loras": loras if loras is not None else [
            {"name": HQ_ANGLES_LORA, "scale": HQ_ANGLES_SCALE},
        ],
    }
