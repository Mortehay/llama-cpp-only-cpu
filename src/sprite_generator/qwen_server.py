"""A persistent Qwen-Image process: load once, park in host RAM, serve many.

Why: qwen_t2i.py runs each image as two fresh subprocesses, so ~75 s of a cold
~100-110 s request is re-reading and re-placing weights. The 2026-09-28 spike
(decisions/0012 "2c") kept both halves in host RAM in ONE process and moved
them onto the card per request: 36-46 s warm, correct images, ~17 GiB RSS.

The two halves still never share the card - the encoder (~4.8 GiB) and the
Q2_K transformer + LoRA + VAE (~9 GiB) do not fit together - they take turns:

    encoder -> card, encode, encoder -> host, empty_cache
    transformer + VAE -> card, denoise, decode, REPLY
    transformer + VAE -> host          (after replying: off the request path)

Parking after the reply is the point: between requests this process holds
host RAM, not VRAM, so an SDXL or audio job can use the card without stopping
it. The model gateway still stops it on a switch (tasks._gateway_switch), and
it exits on its own after QWEN_SERVER_IDLE_S with no request, so the RAM is
not held forever by a model nobody is using.

Protocol - JSON lines on stdin, text lines on stdout:
    -> {"prompt", "negative", "seed", "png", "steps", "cfg", "size"}
    <- PROGRESS <stage> <i> <n>      (same form tasks._QWEN_PROGRESS parses)
    <- RESULT {"ok": true, "png": ..., "timings": {...}}
       RESULT {"ok": false, "error": "..."}
    -> QUIT                          exit now
On start it prints `READY <seconds>` once both halves are loaded.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import select
import sys
import time

import torch

REPO = os.environ.get("QWEN_T2I_REPO", "ovedrive/Qwen-Image-Edit-2511-4bit")
CONFIG_REPO = os.environ.get("QWEN_T2I_CONFIG_REPO", "Qwen/Qwen-Image-2512")
CACHE = os.environ.get("HF_HUB_CACHE") or "/models"
DTYPE = torch.bfloat16
IDLE_S = float(os.environ.get("QWEN_SERVER_IDLE_S", "1800"))

# Free VRAM needed before the transformer goes on. The spike measured the
# transformer + LoRA + VAE taking ~9.8 GiB with the encoder already parked,
# and denoise leaving ~0.4 GiB; 10000 MiB refuses cleanly instead of OOMing.
MIN_FREE_MB = int(os.environ.get("QWEN_SERVER_MIN_FREE_MB", "10000"))


def out(line: str) -> None:
    print(line, flush=True)


def _free_mb() -> float:
    return torch.cuda.mem_get_info()[0] / 2 ** 20


class Server:
    def __init__(self, gguf: str, lora_repo: str, lora_weight: str):
        from diffusers import (FlowMatchEulerDiscreteScheduler,
                               GGUFQuantizationConfig, QwenImagePipeline,
                               QwenImageTransformer2DModel)
        # Encoder: loaded onto the card (bitsandbytes NF4 must be quantised
        # there), then parked. Measured 23.6 s load, 4.6 s park.
        self.te = QwenImagePipeline.from_pretrained(
            REPO, transformer=None, vae=None, dtype=DTYPE,
            cache_dir=CACHE).to("cuda")
        self.te.text_encoder.to("cpu")
        torch.cuda.empty_cache()

        transformer = QwenImageTransformer2DModel.from_single_file(
            gguf, quantization_config=GGUFQuantizationConfig(compute_dtype=DTYPE),
            config=CONFIG_REPO, subfolder="transformer", dtype=DTYPE,
            cache_dir=CACHE)
        if lora_weight:
            # The distillation's own scheduler (shift=3) - see qwen_t2i.py.
            scheduler = FlowMatchEulerDiscreteScheduler.from_config({
                "base_image_seq_len": 256, "base_shift": math.log(3),
                "invert_sigmas": False, "max_image_seq_len": 8192,
                "max_shift": math.log(3), "num_train_timesteps": 1000,
                "shift": 1.0, "shift_terminal": None,
                "stochastic_sampling": False, "time_shift_type": "exponential",
                "use_beta_sigmas": False, "use_dynamic_shifting": True,
                "use_exponential_sigmas": False, "use_karras_sigmas": False})
        else:
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                CONFIG_REPO, subfolder="scheduler", cache_dir=CACHE)
        self.den = QwenImagePipeline.from_pretrained(
            REPO, transformer=transformer, scheduler=scheduler,
            text_encoder=None, tokenizer=None, dtype=DTYPE, cache_dir=CACHE)
        if lora_weight:
            self.den.load_lora_weights(lora_repo, weight_name=lora_weight,
                                       cache_dir=CACHE)
        self.den.vae.enable_tiling()

    def _park_denoiser(self) -> None:
        self.den.transformer.to("cpu")
        self.den.vae.to("cpu")
        torch.cuda.empty_cache()

    def handle(self, req: dict) -> dict:
        t, s = {}, time.time()
        steps = int(req.get("steps", 8))
        cfg = float(req.get("cfg", 1.0))
        size = int(req.get("size", 512))

        out("PROGRESS encode 0 2")
        self.te.text_encoder.to("cuda")
        with torch.no_grad():
            pe, pm = self.te.encode_prompt(prompt=[req["prompt"]],
                                           device=torch.device("cuda"))
            ne = nm = None
            # At CFG <= 1 the negative is never used - skip encoding it.
            if cfg > 1.0:
                ne, nm = self.te.encode_prompt(
                    prompt=[req.get("negative") or " "],
                    device=torch.device("cuda"))
        self.te.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        out("PROGRESS encode 2 2")
        t["encode_s"] = round(time.time() - s, 2)

        free = _free_mb()
        if free < MIN_FREE_MB:
            return {"ok": False, "error": (
                f"not enough free VRAM to place the Qwen transformer: "
                f"{free / 1024:.1f} GiB free, needs {MIN_FREE_MB / 1024:.1f}. "
                f"Another process holds the card.")}

        s2 = time.time()
        out("PROGRESS load 0 1")
        self.den.transformer.to("cuda")
        self.den.vae.to("cuda")
        out("PROGRESS load 1 1")
        t["place_s"] = round(time.time() - s2, 2)

        def on_step(p, i, ts, kw):
            out(f"PROGRESS denoise {i + 1} {steps}")
            return kw

        s3 = time.time()
        try:
            image = self.den(
                prompt_embeds=pe, prompt_embeds_mask=pm,
                negative_prompt_embeds=ne, negative_prompt_embeds_mask=nm,
                true_cfg_scale=cfg, height=size, width=size,
                num_inference_steps=steps,
                generator=torch.Generator("cpu").manual_seed(int(req["seed"])),
                callback_on_step_end=on_step).images[0]
        except Exception:
            # Leave the card empty for whoever is next, then report.
            self._park_denoiser()
            raise
        t["denoise_s"] = round(time.time() - s3, 2)
        image.save(req["png"])
        t["total_s"] = round(time.time() - s, 2)
        return {"ok": True, "png": req["png"], "timings": t}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--lora-repo", default="")
    ap.add_argument("--lora-weight", default="")
    a = ap.parse_args(argv)

    # Die with the worker. The read loop already exits on stdin EOF (a dead
    # parent closes the pipe), but not during the ~85 s load, and an orphan
    # would sit on ~17 GiB of RAM that nothing else knows to free.
    try:
        import ctypes, signal
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
        # Did the parent die before prctl took effect? Compare against the pid
        # the worker passed - NOT `getppid() == 1`: in the container the Celery
        # worker IS pid 1, so that test killed every live child at birth
        # (found 2026-09-28: "qwen server exited" with no output, x3).
        parent = os.environ.get("QWEN_SERVER_PARENT")
        if parent and os.getppid() != int(parent):
            out("EXIT parent already gone")
            return 1
    except Exception as e:
        out(f"WARN no parent-death signal: {e}")

    t0 = time.time()
    srv = Server(a.gguf, a.lora_repo, a.lora_weight)
    out(f"READY {time.time() - t0:.1f}")

    last = time.time()
    while True:
        ready, _, _ = select.select([sys.stdin], [], [], 5.0)
        if not ready:
            if time.time() - last > IDLE_S:
                out(f"IDLE_EXIT {IDLE_S:.0f}")
                return 0
            continue
        line = sys.stdin.readline()
        if not line or line.strip() == "QUIT":
            return 0
        last = time.time()
        try:
            res = srv.handle(json.loads(line))
        except Exception as e:
            res = {"ok": False, "error": f"{e.__class__.__name__}: {e}"}
        out("RESULT " + json.dumps(res))
        # Off the request path: the caller already has its answer.
        try:
            srv._park_denoiser()
        except Exception as e:
            out(f"PARK_FAILED {e}")
        last = time.time()


if __name__ == "__main__":
    sys.exit(main())
