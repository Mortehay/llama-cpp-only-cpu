"""TEMPORARY - which checkpoint/resolution actually follows the skeleton?

The metric is LEG SWING, not frame-to-frame difference. Frame-to-frame is
maximised by an animation that freezes, so it cannot tell "consistent" from
"not moving" - that is how the current config scored well while the walk stood
still. A walk cycle must widen at contact and narrow at passing, so the spread
of leg width across the four frames is a number that a frozen row cannot fake.

Reference: All-In-One at 512 measured a swing of 0.5px out of ~32. Flat.
"""
import numpy as np
import torch
from PIL import Image

import poses
import tasks

PARENT = 48
CONFIGS = [
    ("All-In-One SD1.5  @512", "PublicPrompts/All-In-One-Pixel-Model", 512),
    ("SDXL+pixel-art    @1024",
     "stabilityai/stable-diffusion-xl-base-1.0+nerijs/pixel-art-xl", 1024),
]


def leg_swing(frames):
    vals = []
    for im in frames:
        keyed = tasks.remove_background(im, keep_largest=True)
        op = np.array(keyed)[:, :, 3] > 0
        ys = np.where(op.any(axis=1))[0]
        if not ys.size:
            vals.append(0.0)
            continue
        top, bot = ys[0], ys[-1]
        band = op[bot - int((bot - top) * 0.30):bot + 1]
        w = [int(r.sum()) for r in band if r.any()]
        vals.append(float(np.mean(w)) if w else 0.0)
    # Normalise: a 1024 render has twice the pixels of a 512 one.
    scale = 100.0 / max(np.mean(vals), 1e-6)
    return vals, (max(vals) - min(vals)) * scale


for label, model, gen in CONFIGS:
    try:
        core = tasks.strip_ground_patch(
            Image.open(tasks.get_core_image_path(PARENT)).convert("RGBA"))
        box = core.getbbox()
        rgb = Image.alpha_composite(
            Image.new("RGBA", core.size, (255, 255, 255, 255)), core
        ).convert("RGB").resize((gen, gen), Image.Resampling.LANCZOS)
        sx, sy = gen / core.size[0], gen / core.size[1]
        ctrl = poses.control_images(
            "move right", 4, (gen, gen),
            (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy))

        p = tasks.get_sd_pipeline(model, pipeline_type="img2img",
                                  controlnet=tasks.OPENPOSE_CONTROLNET)
        trigger, _ = tasks.action_entry("move right"), None
        trigger = tasks.action_entry("move right")["prompt"]
        base = ("green zombie, tattered clothes, high quality pixel art, "
                "sharp focus")
        steps, cfg, neg = tasks.resolve_sampling_params(
            model, 30, 8.0, tasks.NEGATIVE_FRAME)

        frames = []
        for f in range(4):
            frames.append(p(
                prompt=f"{trigger}, {base}, single character, full body, centered",
                negative_prompt=neg or None, image=rgb, strength=0.60,
                num_inference_steps=steps, guidance_scale=cfg,
                generator=torch.Generator("cpu").manual_seed(1234),
                control_image=ctrl[f],
                controlnet_conditioning_scale=tasks.CONTROLNET_SCALE).images[0])

        vals, swing = leg_swing(frames)
        print(f"{label}: leg width {[round(v,1) for v in vals]}  "
              f"NORMALISED SWING {swing:.1f} (per 100 units of leg width)",
              flush=True)

        sheet = Image.new("RGB", (gen * 4, gen), (255, 255, 255))
        for i, im in enumerate(frames):
            sheet.paste(im, (i * gen, 0))
        tag = label.split()[0].lower()
        sheet.resize((200 * 4, 200)).save(f"/app/images/_dbg_ck_{tag}.png")
    except Exception as e:
        print(f"{label}: FAILED {type(e).__name__}: {e}", flush=True)
