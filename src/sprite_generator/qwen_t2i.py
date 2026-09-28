"""Qwen-Image-2512 text-to-image from a GGUF transformer, on the 12 GB card.

Why this is a CLI run as subprocesses, not a function the worker calls:
the two halves do not fit in one process. The NF4 text encoder (~4.8 GiB) and
the Q3 transformer (~9.3 GiB) together exceed the card, and holding both in one
process is the failure qwen_edit.encode_only records - on 2026-08-23 it took the
whole WSL VM down with no traceback. A process exit is also the only release of
VRAM that is guaranteed complete. So `tasks.generate_core_task` runs:

    python qwen_t2i.py encode  --prompt P --negative N --out E.pt
    python qwen_t2i.py denoise --embeds E.pt --gguf G --png OUT.png --seed S

and reads `PROGRESS <stage> <i> <n>` lines from stdout for the queue panel.

Measured 2026-09-28 (decisions/0012): encode 33.5s / 4.8 GiB peak; transformer
placed in ~20s at 9.25 GiB; 512px x 20 steps at true CFG 4 is 7.33 s/step,
9.55 GiB peak, 1.3 GiB spare. The VAE is already on the card during denoising,
so decoding in the same process is expected to fit at 512px with tiling - that
was NOT measured by the bench (which batched decodes), so a failure here is the
first thing to check if decode OOMs.

Weights, and why they come from three places:
    transformer   the GGUF file (Q3_K_M, 9.69 GB) under /models/gguf
    encoder, VAE  ovedrive/Qwen-Image-Edit-2511-4bit - hash-identical to 2512's
    config        Qwen/Qwen-Image-2512. NOT the Edit repo: its transformer
                  config carries `zero_cond_t: true`, which 2512 does not.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

REPO = os.environ.get("QWEN_T2I_REPO", "ovedrive/Qwen-Image-Edit-2511-4bit")
CONFIG_REPO = os.environ.get("QWEN_T2I_CONFIG_REPO", "Qwen/Qwen-Image-2512")
CACHE = os.environ.get("HF_HUB_CACHE") or "/models"

# bfloat16, as in qwen_edit.py: flow-matching in fp16 surfaces as NaN latents,
# i.e. a black PNG with no error.
DTYPE = torch.bfloat16

# The benched configuration. Overridable, but these are the measured values -
# 1024px was NOT tested and will very likely OOM with 1.3 GiB spare.
SIZE = int(os.environ.get("QWEN_T2I_SIZE", "512"))
STEPS = int(os.environ.get("QWEN_T2I_STEPS", "20"))
CFG = float(os.environ.get("QWEN_T2I_CFG", "4.0"))


def _say(stage: str, i: int, n: int) -> None:
    print(f"PROGRESS {stage} {i} {n}", flush=True)


def _vram(tag: str) -> None:
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"[{tag}] VRAM free {free / 2**30:.2f}/{total / 2**30:.2f} GiB",
              flush=True)


def encode(prompt: str, negative: str, out: str) -> None:
    from diffusers import QwenImagePipeline

    _say("encode", 0, 2)
    pipe = QwenImagePipeline.from_pretrained(REPO, transformer=None, vae=None,
                                             dtype=DTYPE, cache_dir=CACHE)
    pipe.to("cuda")
    _vram("encoder loaded")
    res = {}
    # no_grad is not optional: encode_prompt called directly is not wrapped in
    # it, and without it the encoder "mysteriously" doubles (qwen_edit.py).
    with torch.no_grad():
        for i, (key, text) in enumerate((("pos", prompt), ("neg", negative))):
            e, m = pipe.encode_prompt(prompt=[text or " "],
                                      device=torch.device("cuda"))
            res[key] = (e.cpu(), m.cpu() if m is not None else None)
            _say("encode", i + 1, 2)
    torch.save(res, out)


def denoise(embeds: str, gguf: str, png: str, seed: int,
            steps: int = STEPS, cfg: float = CFG, size: int = SIZE) -> None:
    from diffusers import (FlowMatchEulerDiscreteScheduler,
                           GGUFQuantizationConfig, QwenImagePipeline,
                           QwenImageTransformer2DModel)

    emb = torch.load(embeds)
    _say("load", 0, 1)
    transformer = QwenImageTransformer2DModel.from_single_file(
        gguf, quantization_config=GGUFQuantizationConfig(compute_dtype=DTYPE),
        config=CONFIG_REPO, subfolder="transformer", dtype=DTYPE,
        cache_dir=CACHE)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        CONFIG_REPO, subfolder="scheduler", cache_dir=CACHE)
    pipe = QwenImagePipeline.from_pretrained(
        REPO, transformer=transformer, scheduler=scheduler,
        text_encoder=None, tokenizer=None, dtype=DTYPE, cache_dir=CACHE)
    # Resident, not offloaded: offload moves the 9 GB transformer back through
    # host RAM at the end of the pass, which is what OOM-killed qwen_edit.
    pipe.to("cuda")
    pipe.vae.enable_tiling()
    _vram("transformer placed")
    _say("load", 1, 1)

    def on_step(p, i, t, kw):
        _say("denoise", i + 1, steps)
        return kw

    (pe, pm), (ne, nm) = emb["pos"], emb["neg"]
    image = pipe(
        prompt_embeds=pe.cuda(),
        prompt_embeds_mask=pm.cuda() if pm is not None else None,
        negative_prompt_embeds=ne.cuda(),
        negative_prompt_embeds_mask=nm.cuda() if nm is not None else None,
        true_cfg_scale=cfg, height=size, width=size, num_inference_steps=steps,
        generator=torch.Generator("cpu").manual_seed(seed),
        callback_on_step_end=on_step,
    ).images[0]
    _vram("decoded")
    image.save(png)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    e = sub.add_parser("encode")
    e.add_argument("--prompt", required=True)
    e.add_argument("--negative", default="")
    e.add_argument("--out", required=True)
    d = sub.add_parser("denoise")
    d.add_argument("--embeds", required=True)
    d.add_argument("--gguf", required=True)
    d.add_argument("--png", required=True)
    d.add_argument("--seed", type=int, required=True)
    d.add_argument("--steps", type=int, default=STEPS)
    d.add_argument("--cfg", type=float, default=CFG)
    d.add_argument("--size", type=int, default=SIZE)
    a = ap.parse_args(argv)

    t0 = time.time()
    if a.mode == "encode":
        encode(a.prompt, a.negative, a.out)
    else:
        denoise(a.embeds, a.gguf, a.png, a.seed, a.steps, a.cfg, a.size)
    print(f"DONE {a.mode} {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
