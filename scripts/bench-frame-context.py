#!/usr/bin/env python3
"""What does conditioning a cell on EXTRA reference images cost?

    bench-frame-context.py --work <job work dir> --stage encode --images 2

Qwen-Image-Edit-2511 accepts several reference images, so a sheet cell can be
conditioned on its direction view AND on the neighbouring frames of its row.
The design brief asks for that, and asks what it costs. This answers it on
this card rather than from the model card.

ONE CONFIGURATION PER PROCESS, on purpose. In production the encode and the
denoise are separate subprocesses because loading and releasing a large
quantised model fragments the CUDA allocator past usefulness (build-sheet.py
module docstring). Measuring several configurations in one process would
measure that fragmentation instead of the thing being asked about, and the
second measurement would be the one that OOM'd.

Prints one JSON line so a wrapper can collect the runs.

THE NUMBER THAT MATTERS is peak reserved VRAM, not wall-clock. The failure
this guards against is not slowness: it is the text encoder's sequence
growing until the bitsandbytes fallback dequantises a weight matrix to bf16
and the card runs out - which surfaces as "CUDA driver error: device not
ready" and reads like a driver fault, not like an out-of-memory.
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


def build_cells(work, cfg, n_images, n_cells):
    """`n_cells` cells, each conditioned on `n_images` images.

    The extra images are real neighbours from the work directory, not copies
    of the view: a duplicate would compress differently and understate the
    cost.
    """
    import qwen_edit

    act = cfg["actions"][0]
    dirs = cfg.get("render_directions") or cfg["directions"]
    cells = []
    for i in range(n_cells):
        d = dirs[i % len(dirs)]
        f_i = i % max(1, cfg.get("frames", 1))
        view = Image.open(os.path.join(work, f"dir_{d}.png"))
        imgs = [view]
        for nb in (f_i - 1, f_i + 1, f_i + 2):
            if len(imgs) >= n_images:
                break
            for stem in ("cut", "cell"):
                p = os.path.join(work, f"{stem}_{act}_{d}_{max(nb, 0)}.png")
                if os.path.isfile(p):
                    imgs.append(Image.open(p).convert("RGB"))
                    break
        while len(imgs) < n_images:          # short row: pad with the view
            imgs.append(view)
        cells.append({
            "key": f"bench|{d}|{i}",
            "image": imgs[0] if len(imgs) == 1 else imgs,
            "prompt": f"{qwen_edit.angle_prompt(d)}, the character is standing",
        })
    return cells


def main():
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True)
    p.add_argument("--stage", choices=("encode", "denoise"), required=True)
    p.add_argument("--images", type=int, default=1)
    p.add_argument("--cells", type=int, default=4)
    a = p.parse_args()

    import qwen_edit

    with open(os.path.join(a.work, "config.json")) as fh:
        cfg = json.load(fh)

    cells = build_cells(a.work, cfg, a.images, a.cells)
    torch.cuda.reset_peak_memory_stats()
    free_before = torch.cuda.mem_get_info()[0]

    t0 = time.time()
    err = None
    try:
        if a.stage == "encode":
            out = qwen_edit.encode_cells(cells)
            torch.save(out, os.path.join(a.work, "embeds_bench.pt"))
        else:
            embeds = torch.load(os.path.join(a.work, "embeds_bench.pt"))
            qwen_edit.denoise_cells(cells, embeds, seed=cfg["seed"],
                                    size=cfg["size"], with_angles=True,
                                    angles_scale=0.4)
    except Exception as e:                      # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    elapsed = time.time() - t0

    free_after = torch.cuda.mem_get_info()[0]
    print(json.dumps({
        "stage": a.stage,
        "images": a.images,
        "cells": a.cells,
        "ok": err is None,
        "error": err,
        "seconds_total": round(elapsed, 1),
        "seconds_per_cell": round(elapsed / max(a.cells, 1), 2),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2 ** 20),
        # Driver-level free memory. Guest-side: this does not see host VRAM,
        # so it is comparable between runs here and NOT comparable with a
        # Windows-side counter.
        "free_before_mb": round(free_before / 2 ** 20),
        "free_after_mb": round(free_after / 2 ** 20),
        "min_free_mb": round(min(free_before, free_after) / 2 ** 20),
    }))
    return 0 if err is None else 1


if __name__ == "__main__":
    sys.exit(main())
