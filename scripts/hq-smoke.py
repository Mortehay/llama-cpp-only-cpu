#!/usr/bin/env python3
"""One HQ render, and the three numbers that decide whether this lane is usable.

Run BEFORE any sheet. A sheet is 30-210 cells; if s/cell is wrong this answers
it in one cell instead of in hours.

WHAT IT REPORTS, AND WHY EACH MATTERS

  s/cell        The diffusers lane is ~30.1 s/cell for Q2_K resident and
                68.8 s/cell for Q6_K with group offload. A 210-cell sheet at
                30 s is 3h07; at 120 s it is 7 h; at 300 s it is 17 h and the
                lane is not an overnight job any more. This is the number that
                decides practicality, and it is why one cell is rendered
                before anything larger is attempted.
  peak VRAM     The card is 12 GB and the whole point of this lane is a model
                that does not comfortably fit. ComfyUI's own offloading is
                what makes it run; if peak sits at the ceiling the sheet will
                fault partway through rather than at cell 1.
  host RAM      Offloading moves the transformer to host RAM. WSL has 41 GiB
                and the brain's 35B hybrid also wants a share of it.

AND THE ONE QUALITATIVE QUESTION: does the Multiple-Angles LoRA still pin the
camera when it is no longer competing with a 4-step Lightning LoRA? The whole
turnaround depends on that LoRA holding. Rendering `e` and `s` from one
concept answers it - if they come back as the same view, the LoRA is not
steering and the lane cannot build a turnaround whatever its quality.

    docker compose run --rm --entrypoint python sprite-worker \
        /app/scripts/hq-smoke.py <concept.png> [--directions s,e]
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, "/app")


def vram_now() -> tuple[float, float]:
    """(used, total) GiB on the card, from nvidia-smi. (0,0) when unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout.strip()
        used, total = (float(x) for x in out.splitlines()[0].split(","))
        return used / 1024, total / 1024
    except Exception:  # noqa: BLE001
        return 0.0, 0.0


def ram_avail_gib() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 ** 2
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("concept")
    ap.add_argument("--directions", default="s,e")
    ap.add_argument("--out", default="/app/images/_hq_smoke")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    import comfy_client
    import engines
    import qwen_edit

    why = engines.unavailable_reason("comfyui-hq")
    if why:
        print(f"comfyui-hq is not available: {why}", file=sys.stderr)
        return 2
    if not os.path.exists(a.concept):
        print(f"no such concept: {a.concept}", file=sys.stderr)
        return 2
    os.makedirs(a.out, exist_ok=True)

    print(f"model     {engines.HQ_UNET}")
    print(f"precision {engines.HQ_PRECISION}")
    print(f"steps     {engines.HQ_STEPS}    cfg {engines.HQ_CFG}")
    print(f"lora      {engines.HQ_ANGLES_LORA} @ {engines.HQ_ANGLES_SCALE}")
    print(f"negative  {engines.HQ_NEGATIVE[:70]}...")
    print(f"sampler   {comfy_client.SAMPLER}/{comfy_client.SCHEDULER}\n")

    before_used, total = vram_now()
    print(f"VRAM before: {before_used:.2f} / {total:.1f} GiB   "
          f"host RAM available: {ram_avail_gib():.1f} GiB\n")

    results = []
    peak_used = before_used
    for d in [x.strip() for x in a.directions.split(",") if x.strip()]:
        prompt = qwen_edit.angle_prompt(d)
        t0 = time.time()
        png = comfy_client.render(
            a.concept, prompt, negative=engines.HQ_NEGATIVE, seed=1234,
            steps=engines.HQ_STEPS, cfg=engines.HQ_CFG, size=512)
        dt = time.time() - t0
        used, _ = vram_now()
        peak_used = max(peak_used, used)
        dst = os.path.join(a.out, f"dir_{d}.png")
        with open(png, "rb") as src, open(dst, "wb") as out:
            out.write(src.read())
        results.append({"direction": d, "seconds": round(dt, 1),
                        "vram_after_gib": round(used, 2), "path": dst})
        print(f"  {d:<3} {dt:7.1f}s   VRAM after {used:5.2f} GiB   -> {dst}")

    if not results:
        print("nothing rendered", file=sys.stderr)
        return 2

    per = [r["seconds"] for r in results]
    mean = sum(per) / len(per)
    print(f"\ns/cell: {min(per):.1f}-{max(per):.1f} (mean {mean:.1f})")
    print(f"peak VRAM seen: {peak_used:.2f} / {total:.1f} GiB")
    print(f"host RAM available after: {ram_avail_gib():.1f} GiB")

    for cells, label in ((30, "one 5-direction action row set"),
                         (210, "a 7-action sheet")):
        hrs = mean * cells / 3600
        print(f"  projected {label:<32} {cells:>4} cells -> {hrs:5.2f} h")
    print("\nAgainst the diffusers lane: ~30.1 s/cell Q2_K resident, "
          "68.8 s/cell Q6_K with group offload (3h07 and 6h51 for 210 cells).")

    if len(results) > 1:
        print("\nNOW LOOK AT THE IMAGES. The numbers cannot tell you whether "
              "the angles LoRA held the camera - if the directions came back "
              "as the same view, this lane cannot build a turnaround whatever "
              "its per-cell quality.")

    if a.json:
        with open(a.json, "w") as f:
            json.dump({"model": engines.HQ_UNET,
                       "precision": engines.HQ_PRECISION,
                       "steps": engines.HQ_STEPS, "cfg": engines.HQ_CFG,
                       "sampler": comfy_client.SAMPLER,
                       "scheduler": comfy_client.SCHEDULER,
                       "mean_s_per_cell": round(mean, 1),
                       "peak_vram_gib": round(peak_used, 2),
                       "cells": results}, f, indent=1)
        print("wrote", a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
