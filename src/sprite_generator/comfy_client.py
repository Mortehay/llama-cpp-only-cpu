"""Render one cell through ComfyUI, for engine="comfyui-hq".

WHAT THIS LANE IS FOR
---------------------
The diffusers lane runs a Q2_K transformer with the 4-step Lightning LoRA, and
that distillation forces `true_cfg_scale=1.0`. At CFG 1.0 there is no
classifier-free guidance, so the sheet conveyor HAS NO NEGATIVE PROMPT and
cannot be given one - every "no extra limbs" instruction lands in the positive
conditioning instead, which is the trap `action_prompts.json` records for
"no frame, no border". This lane drops Lightning, which buys back real step
counts, real CFG, and a negative prompt - plus a far less aggressive quant.

THE GRAPH IS VALIDATED, NOT TRUSTED
------------------------------------
Node names and input names are a moving target across ComfyUI versions, and
getting one wrong fails at render time with a message about an unknown node.
The first pin chosen for this lane, v0.3.45, had no Qwen nodes AT ALL. So
`validate_graph` checks every node class and every input name against the
server's own `/object_info` before anything is submitted, and says exactly
which name is wrong. A wrong guess costs a clear error, not a failed sheet.

IT NEVER FALLS BACK to the diffusers lane. See engines.py.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
import uuid

import engines

logger = logging.getLogger(__name__)

URL = engines.COMFY_URL
POLL_S = float(os.environ.get("COMFYUI_POLL_S", "2"))
# A single HQ cell is minutes, not seconds: no 4-step distillation and a much
# larger transformer being streamed off host RAM. This bounds one CELL, not a
# sheet.
CELL_TIMEOUT_S = float(os.environ.get("COMFYUI_CELL_TIMEOUT_S", "1800"))

# The sampler. Deliberately conservative defaults: euler/simple is what the
# Qwen-Image reference workflows use, and this lane's purpose is quality per
# cell rather than novelty in the sampler.
SAMPLER = os.environ.get("COMFYUI_HQ_SAMPLER", "euler")
SCHEDULER = os.environ.get("COMFYUI_HQ_SCHEDULER", "simple")
# `fp8_e4m3fn` would quantise the weights a SECOND time on load. The file is
# already fp8mixed, so it is loaded as-is and the compute dtype is left to
# ComfyUI, which picks bf16 on this card - the same dtype the diffusers lane
# uses, and the one Qwen-Image was trained in.
WEIGHT_DTYPE = os.environ.get("COMFYUI_HQ_WEIGHT_DTYPE", "default")
LATENT_NODE = os.environ.get("COMFYUI_HQ_LATENT_NODE", "EmptySD3LatentImage")


class ComfyError(RuntimeError):
    """A render could not be produced. Always names what went wrong."""


def _get(path: str, timeout: float = 10):
    with urllib.request.urlopen(f"{URL}{path}", timeout=timeout) as r:
        return json.load(r)


def _post(path: str, body: dict, timeout: float = 30):
    req = urllib.request.Request(
        f"{URL}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def upload_image(path: str) -> str:
    """Put a source image where ComfyUI can load it. Returns its served name.

    Uploaded through the API rather than written into a shared directory: the
    name ComfyUI returns is the one `LoadImage` wants, and letting the server
    own that mapping removes a whole class of "the file is there but the node
    cannot see it" problems.
    """
    name = f"hq_{uuid.uuid4().hex}_{os.path.basename(path)}"
    with open(path, "rb") as f:
        payload = f.read()
    boundary = f"----comfy{uuid.uuid4().hex}"
    parts = [
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="image"; filename="{name}"\r\n'
        f"Content-Type: image/png\r\n\r\n".encode(),
        payload, b"\r\n",
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n',
        f"--{boundary}--\r\n".encode(),
    ]
    req = urllib.request.Request(
        f"{URL}/upload/image", data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.load(r)
    except urllib.error.URLError as e:
        raise ComfyError(f"could not upload {path} to ComfyUI: {e}") from e
    served = out.get("name") or name
    if out.get("subfolder"):
        served = f"{out['subfolder']}/{served}"
    return served


def build_graph(image_name: str, prompt: str, *, negative: str, seed: int,
                steps: int, cfg: float, width: int, height: int,
                unet: str, text_encoder: str, vae: str,
                lora: str | None, lora_scale: float) -> dict:
    """The API-format graph for one edit render.

    Positive AND negative conditioning both go through
    TextEncodeQwenImageEditPlus with the source image attached: that node
    contributes reference latents from the image, and giving the negative
    branch a different reference would be comparing the sample against
    something the positive branch never saw.
    """
    g = {
        "unet": {"class_type": "UNETLoader",
                 "inputs": {"unet_name": unet, "weight_dtype": WEIGHT_DTYPE}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": text_encoder, "type": "qwen_image"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "src": {"class_type": "LoadImage", "inputs": {"image": image_name}},
    }
    model_src = ["unet", 0]
    if lora:
        g["lora"] = {"class_type": "LoraLoaderModelOnly",
                     "inputs": {"model": model_src, "lora_name": lora,
                                "strength_model": lora_scale}}
        model_src = ["lora", 0]

    g["pos"] = {"class_type": "TextEncodeQwenImageEditPlus",
                "inputs": {"clip": ["clip", 0], "prompt": prompt,
                           "vae": ["vae", 0], "image1": ["src", 0]}}
    g["neg"] = {"class_type": "TextEncodeQwenImageEditPlus",
                "inputs": {"clip": ["clip", 0], "prompt": negative,
                           "vae": ["vae", 0], "image1": ["src", 0]}}
    g["latent"] = {"class_type": LATENT_NODE,
                   "inputs": {"width": width, "height": height,
                              "batch_size": 1}}
    g["sampler"] = {"class_type": "KSampler",
                    "inputs": {"model": model_src,
                               "positive": ["pos", 0], "negative": ["neg", 0],
                               "latent_image": ["latent", 0],
                               "seed": int(seed), "steps": int(steps),
                               "cfg": float(cfg), "sampler_name": SAMPLER,
                               "scheduler": SCHEDULER, "denoise": 1.0}}
    g["decode"] = {"class_type": "VAEDecode",
                   "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}}
    g["save"] = {"class_type": "SaveImage",
                 "inputs": {"images": ["decode", 0],
                            "filename_prefix": "hq"}}
    return g


def validate_graph(graph: dict) -> None:
    """Check every class and input name against the server's own schema.

    Raises ComfyError naming the first problem. This exists because a wrong
    node name is otherwise discovered at render time, and because the node set
    genuinely moves between versions - the first ComfyUI pin chosen for this
    lane had no Qwen nodes at all.
    """
    try:
        info = _get("/object_info", timeout=30)
    except Exception as e:  # noqa: BLE001
        raise ComfyError(f"could not read /object_info from {URL}: {e}") from e

    for key, node in graph.items():
        cls = node["class_type"]
        if cls not in info:
            raise ComfyError(
                f"node '{key}' uses class {cls!r}, which this ComfyUI does not "
                f"have. Qwen edit nodes need v0.42+; check COMFY_REF.")
        spec = info[cls].get("input", {})
        known = set(spec.get("required", {})) | set(spec.get("optional", {}))
        for name in node["inputs"]:
            if name not in known:
                raise ComfyError(
                    f"node '{key}' ({cls}) has no input {name!r}; it accepts "
                    f"{sorted(known)}")


def render(image_path: str, prompt: str, *, negative: str, seed: int,
           steps: int, cfg: float, size: int = 512,
           unet: str | None = None, text_encoder: str | None = None,
           vae: str | None = None, lora: str | None = None,
           lora_scale: float | None = None, validate: bool = True) -> str:
    """Render one cell. Returns the path of the produced PNG.

    Raises ComfyError on anything that stops a render. Never returns a
    diffusers result: this lane fails loudly rather than quietly becoming the
    fast lane, because a caller cannot tell those apart from the image.
    """
    name = upload_image(image_path)
    graph = build_graph(
        name, prompt, negative=negative, seed=seed, steps=steps, cfg=cfg,
        width=size, height=size,
        unet=unet or engines.HQ_UNET,
        text_encoder=text_encoder or engines.HQ_TEXT_ENCODER,
        vae=vae or engines.HQ_VAE,
        lora=lora if lora is not None else engines.HQ_ANGLES_LORA,
        lora_scale=(engines.HQ_ANGLES_SCALE if lora_scale is None
                    else lora_scale))
    if validate:
        validate_graph(graph)

    client_id = uuid.uuid4().hex
    try:
        res = _post("/prompt", {"prompt": graph, "client_id": client_id})
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:600]
        raise ComfyError(f"ComfyUI rejected the graph (HTTP {e.code}): "
                         f"{detail}") from e
    pid = res.get("prompt_id")
    if not pid:
        raise ComfyError(f"ComfyUI accepted no prompt id: {res}")

    started = time.time()
    while True:
        waited = time.time() - started
        if waited > CELL_TIMEOUT_S:
            raise ComfyError(
                f"ComfyUI did not finish a cell within {CELL_TIMEOUT_S:.0f}s "
                f"(prompt {pid}). This bounds ONE CELL, not a sheet.")
        try:
            hist = _get(f"/history/{pid}", timeout=20)
        except Exception as e:  # noqa: BLE001 - a poll may blip, a render may not
            logger.debug("comfy: history poll failed (%s)", e)
            time.sleep(POLL_S)
            continue
        entry = hist.get(pid)
        if entry:
            status = (entry.get("status") or {})
            if status.get("status_str") == "error" or status.get("completed") is False:
                raise ComfyError(
                    f"ComfyUI reported an error for prompt {pid}: "
                    f"{json.dumps(status)[:600]}")
            for out in (entry.get("outputs") or {}).values():
                for img in out.get("images", []):
                    p = os.path.join(engines.HQ_MODELS, "output",
                                     img.get("subfolder", ""),
                                     img["filename"])
                    if os.path.exists(p):
                        logger.info("comfy: cell rendered in %.1fs -> %s",
                                    waited, os.path.basename(p))
                        return p
            raise ComfyError(
                f"ComfyUI finished prompt {pid} but produced no readable "
                f"image under {engines.HQ_MODELS}/output")
        time.sleep(POLL_S)
