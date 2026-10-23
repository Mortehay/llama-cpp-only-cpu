#!/usr/bin/env python3
"""Does block-level offload work with a GGUF transformer on this card?

    probe-gguf-offload.py --mode none
    probe-gguf-offload.py --mode group --blocks 2
    probe-gguf-offload.py --mode model

WHY THIS EXISTS

The editor runs Q2_K, the most aggressive quant there is. Anything larger
needs VRAM this card does not have spare: the denoise pass already peaks at
~9.4 GB of 12 GB. The way past that is to keep part of the transformer in
host RAM and stream blocks onto the GPU - there is 41 GB in WSL to hold it.

But `qwen_edit.denoise_only` REFUSES `enable_sequential_cpu_offload`, with a
specific reason: the offload hook moves parameters to the meta device, and a
`GGUFParameter` loses its `quant_type` on the way. Group offload moves
parameters too, so it may hit the identical wall.

**That is the only question this answers**: does the mechanism survive GGUF,
and at what cost in VRAM and seconds. It does NOT answer whether a bigger
quant produces better sprites - run the quant A/B for that.

Each mode is a separate process, because loading and releasing a large
quantised model fragments the allocator (see build-sheet.py).

    none   baseline: the transformer resident, as production runs it
    group  block-level offload, `--blocks` transformer blocks per group
    model  whole-module offload (the coarser, older mechanism)

A failure here is a useful result. Print it and stop, rather than reaching
for a smaller quant and calling the question closed.

RESULT, 2026-10-23, Q2_K, 512px, 4 steps, one cell:

    mode              peak VRAM   free after   denoise   verdict
    none (today)        9408 MB      2662 MB     30.1 s  the baseline
    model                9382 MB         -       42.8 s  saves nothing
    group blocks=2       2028 MB     10939 MB    74.6 s  slow, no gain
    group blocks=4       2028 MB     10939 MB    47.1 s  the setting
    group blocks=8       2028 MB         -       50.4 s  no better

**GROUP OFFLOAD SURVIVES GGUF.** That is the whole point of this file, and
it was not obvious: `denoise_only` refuses sequential offload because the
hook moves parameters to the meta device and a GGUFParameter loses its
quant_type, and group offload moves parameters too. It does not hit that
wall - it returned valid latents of the right shape at every block size.

The transformer's resident footprint falls 9408 -> 2028 MB, leaving ~10.9 GB
free, for 1.57x the denoise time. That is far more headroom than Q4_K_M
needs and puts Q5_K_M and Q6_K in reach; the limits become host RAM (41 GB
in WSL) and PCIe bandwidth rather than VRAM.

`blocks` is a SPEED dial, not a memory dial - peak VRAM is identical at 2, 4
and 8, because the resident set is dominated by activations, the VAE and the
LoRAs rather than by held transformer blocks. It saturates at 4; blocks=2 is
50% slower for nothing. `model` mode is not competitive: it saves 26 MB.

ENCODE BEFORE THE TRANSFORMER EXISTS, which this script now does. The first
version encoded after placing the pipeline, so `--mode none` asked for the
~6 GB text encoder on top of the ~7 GB transformer and failed reproducibly
with "CUDA driver error: device not ready" while every offload mode passed.
That reads as "resident is broken" and is really "this probe asked for
13 GB". Production avoids it by running encode and denoise as separate
processes.
"""

import argparse
import json
import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
for _c in (os.path.join(_here, "..", "src", "sprite_generator"),
           os.path.dirname(_here), "/app"):
    if os.path.isfile(os.path.join(_c, "qwen_edit.py")):
        sys.path.insert(0, _c)
        break

from PIL import Image  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("none", "group", "model"), default="none")
    p.add_argument("--blocks", type=int, default=2,
                   help="transformer blocks per offload group")
    p.add_argument("--image", default="/app/images/core_abac255e360e.png")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--steps", type=int, default=4)
    a = p.parse_args()

    import torch
    from diffusers import (GGUFQuantizationConfig, QwenImageEditPlusPipeline,
                           QwenImageTransformer2DModel)

    import qwen_edit as q

    out = {"mode": a.mode, "gguf": q.GGUF_FILE, "blocks": a.blocks}

    # ENCODE FIRST, and release, before the transformer exists. Production
    # runs these as separate processes for exactly this reason: the text
    # encoder is ~6 GB and the transformer ~7 GB, and 12 GB does not hold
    # both. The first version of this probe encoded AFTER placing the
    # pipeline, which made `--mode none` fail reproducibly with "CUDA driver
    # error: device not ready" while every offload mode passed - a result
    # that looks like "resident does not work" and is really "this probe
    # asked for 13 GB". The offload modes only passed because the transformer
    # was not fully resident when the encoder loaded.
    img = Image.open(a.image).convert("RGB").resize((a.size, a.size))
    embeds = q.encode_cells([{"key": "probe", "image": img,
                              "prompt": q.angle_prompt("s")}])
    pe, mask = embeds["probe"]
    del embeds
    q._free()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        transformer = QwenImageTransformer2DModel.from_single_file(
            q._gguf_url(),
            quantization_config=GGUFQuantizationConfig(compute_dtype=q.DTYPE),
            config=q.CONFIG_REPO, subfolder="transformer", dtype=q.DTYPE,
            cache_dir="/models", token=os.environ.get("HF_TOKEN") or None,
        )
        out["load_s"] = round(time.time() - t0, 1)

        # The offload is applied to the TRANSFORMER, before it is placed. That
        # is the order the diffusers docs give, and the order that matters:
        # calling .to("cuda") first defeats the point.
        if a.mode == "group":
            transformer.enable_group_offload(
                onload_device=torch.device("cuda"),
                offload_device=torch.device("cpu"),
                offload_type="block_level",
                num_blocks_per_group=a.blocks,
            )

        pipe = QwenImageEditPlusPipeline.from_pretrained(
            q.REPO, transformer=transformer, text_encoder=None, dtype=q.DTYPE,
            cache_dir="/models", token=os.environ.get("HF_TOKEN") or None)
        pipe.load_lora_weights(q.ANGLES_LORA, weight_name=q.ANGLES_WEIGHT,
                               adapter_name="angles")
        pipe.load_lora_weights(q.LIGHTNING_LORA, weight_name=q.LIGHTNING_WEIGHT,
                               adapter_name="lightning")
        pipe.set_adapters(["angles", "lightning"], adapter_weights=[1.0,
                                                                   q.LIGHTNING_SCALE])
        if a.mode == "model":
            pipe.enable_model_cpu_offload()
        elif a.mode == "none":
            pipe.to("cuda")
        else:
            # group offload placed the transformer; the rest still needs a home
            pipe.vae.to("cuda")
        try:
            pipe.vae.enable_tiling()
        except AttributeError:
            pass

        free_before = torch.cuda.mem_get_info()[0]

        t1 = time.time()
        lat = pipe(image=img, height=a.size, width=a.size,
                   prompt_embeds=pe.to("cuda"),
                   prompt_embeds_mask=(mask.to("cuda") if mask is not None
                                       else None),
                   num_inference_steps=a.steps, true_cfg_scale=1.0,
                   generator=torch.Generator(device="cpu").manual_seed(0),
                   output_type="latent").images
        out["denoise_s"] = round(time.time() - t1, 1)
        out["ok"] = True
        out["latent_shape"] = list(lat.shape)
        out["free_before_mb"] = round(free_before / 2 ** 20)
    except Exception as e:                                  # noqa: BLE001
        out["ok"] = False
        out["error"] = f"{type(e).__name__}: {e}"[:400]

    out["peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2 ** 20)
    out["peak_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
    out["total_s"] = round(time.time() - t0, 1)
    print(json.dumps(out))
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
