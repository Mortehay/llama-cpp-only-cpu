#!/usr/bin/env python3
"""Generate the same terrain prompts on two models and compare them AT DISPLAY SCALE.

WHY DISPLAY SCALE

something2 projects a world tile to a 128x64 diamond (ISO_TILE_W/ISO_TILE_H).
A 512x512 generation is downscaled roughly 4x on the long axis before a player
sees it, so fine detail at 1:1 is detail the renderer throws away. Judging these
at full size measures something nobody looks at, and would reject a soft adapter
that is perfectly good once projected.

So every pair is shown twice: at 1:1, and downscaled to 128x64 and blown back up
with NEAREST so the pixels that actually ship are visible.

The gate is unchanged and comes first: no lattice, no framed cells, no shelving.
A soft texture passes. A crisp grid does not.

Usage:
    SPRITE_API_KEY=sk_... python compare-terrain.py
    SPRITE_API_KEY=sk_... python compare-terrain.py --seed 0 --steps 20
"""

import argparse
import base64
import io
import json
import os
import sys
import urllib.request

from PIL import Image, ImageDraw, ImageFont

BASE = "stabilityai/stable-diffusion-xl-base-1.0"
MODELS = [
    ("trained", f"{BASE}+local:something2-terrain"),
    ("nerijs", f"{BASE}+nerijs/pixel-art-xl"),
]

# The prompts the game actually sends, trimmed to the subject.
PROMPTS = {
    "grass": ("lush green meadow grass, seen from directly above, extreme "
              "close-up, fills the entire frame, even soft daylight, pixel art, "
              "spring green, wildflower yellow, warm brown palette"),
    "sand":  ("fine desert sand dunes, seen from directly above, extreme "
              "close-up, fills the entire frame, even soft daylight, pixel art, "
              "warm ochre, pale gold palette"),
    "ice":   ("cracked blue glacier ice, seen from directly above, extreme "
              "close-up, fills the entire frame, even soft daylight, pixel art, "
              "pale cyan, white, deep blue palette"),
}

TILE_W, TILE_H = 128, 64      # what the renderer projects to


def generate(host, key, model, prompt, seed, steps, size):
    body = json.dumps({
        "prompt": prompt, "steps": steps, "width": size, "height": size,
        "seed": seed, "override_settings": {"sd_model_checkpoint": model},
    }).encode()
    req = urllib.request.Request(f"{host}/sdapi/v1/txt2img", data=body,
                                 headers={"Content-Type": "application/json"})
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=900) as r:
        payload = json.load(r)
    return Image.open(io.BytesIO(base64.b64decode(payload["images"][0]))).convert("RGB")


def at_display_scale(img: Image.Image) -> Image.Image:
    """Downscale to the shipped tile size, then blow up so a human can see it."""
    small = img.resize((TILE_W, TILE_H), Image.LANCZOS)
    return small.resize((TILE_W * 3, TILE_H * 3), Image.NEAREST)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="http://localhost:8001")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--out", default="/app/images/_terrain_compare.png")
    a = p.parse_args()

    key = os.environ.get("SPRITE_API_KEY", "")
    results = {}
    for subject, prompt in PROMPTS.items():
        for tag, model in MODELS:
            print(f"  {subject:6} {tag:8} ...", flush=True)
            try:
                results[(subject, tag)] = generate(
                    a.host, key, model, prompt, a.seed, a.steps, a.size)
            except Exception as e:
                print(f"    FAILED: {e}")
                results[(subject, tag)] = None

    full, disp = a.size // 2, TILE_W * 3
    col_w = max(full, disp) + 16
    row_h = full // 2 + TILE_H * 3 + 46
    W = col_w * len(MODELS) * len(PROMPTS) // len(PROMPTS) * len(PROMPTS)
    W = col_w * (len(MODELS) * len(PROMPTS))
    sheet = Image.new("RGB", (W, row_h + 30), (24, 24, 38))
    d = ImageDraw.Draw(sheet)
    try:
        f = ImageFont.load_default(size=15)
    except TypeError:
        f = ImageFont.load_default()
    d.text((8, 6), f"seed {a.seed} - top: {a.size}px as generated - "
                   f"bottom: downscaled to {TILE_W}x{TILE_H} (what ships), "
                   f"nearest-upscaled 3x", font=f, fill=(226, 224, 240))

    i = 0
    for subject in PROMPTS:
        for tag, _ in MODELS:
            img = results[(subject, tag)]
            x = i * col_w + 8
            if img is not None:
                top = img.resize((full, full), Image.LANCZOS)
                sheet.paste(top, (x, 30))
                sheet.paste(at_display_scale(img), (x, 30 + full + 8))
            d.text((x, 30 + full + 8 + TILE_H * 3 + 4), f"{subject} / {tag}",
                   font=f, fill=(252, 211, 77) if tag == "trained" else (136, 132, 168))
            i += 1

    sheet.save(a.out)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
