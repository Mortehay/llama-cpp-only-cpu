#!/usr/bin/env python3
"""Was the green island the captions, or the model?

    docker exec sprite_worker python /app/scripts/map-discriminator.py

WHY THIS EXISTS

The first map adapter (`mapstyle`) painted a monochromatic green island from a
prompt that asked for a coast and a mountain range. Two explanations predicted
the SAME picture:

  1. the trigger was diluted - every caption carried a unique hash token from
     the upload filename, so `<mapstyle-style>` was one token among 114 others;
  2. the model never learned coasts at all.

`caption_for` was fixed, and `mapstyle2` was retrained on the same 114
references, the same 1000 steps, the same everything - only the captions
differ. That makes this a controlled discriminator, not a demo: generate from
both adapters at the SAME generation seed and compare.

  coast appears -> it was the caption
  coast absent  -> the caption was still a bug, and the green island is
                   something else, now isolated

WHAT THE FIRST RUN ACTUALLY FOUND, 2026-08-27

Neither. BOTH adapters painted an open coastline, water, cliffs and forest -
the green island did not reproduce at all, so the captions cannot have been
its cause and this comparison answers a dead question.

It took looking at the images to find that out. The automatic verdict said
"no coast either way" because `blue_share` measured a four-colour palette fit
instead of the pixels, and scored two visibly wet maps at 0.0%. The images
were on disk the whole time; the summary was wrong and confident.

So a third arm was added - the same first adapter with the TRIGGER WITHHELD -
because the green island predates `_triggered` in the map path, which makes an
unaddressed adapter the live candidate the first two arms cannot separate.

THE RESULT, ALL THREE ARMS

    arm                     water   flatness
    mapstyle                20.0%     13.1
    mapstyle2               15.6%     15.1
    mapstyle, no trigger    20.6%     14.3

Every arm paints a coast. Bays, cliffs, islands, forest. The trigger is not the
explanation either - withholding it costs nothing in water and only 1.2 in
flatness, and what it visibly changes is STYLE: the untriggered map is smoother
and more illustrated, with a rendered snow-capped peak, where the triggered
ones are chunkier and flatter.

CONCLUSION: I DO NOT KNOW WHAT CAUSED THE GREEN ISLAND, AND BOTH CANDIDATES
ARE OUT. It is not the captions and it is not the trigger, at this prompt and
this seed. `FIRST` does not reproduce either - the recorded 14.6 flatness and
its all-green palette came back as 13.1 with a teal in it, which the drift
check flags. That the original MEASUREMENT does not reproduce either is the
strongest hint available: the green island was probably never a property of
the adapter, but of conditions in that one run which were not written down.

Which is the whole lesson twice over - the run was not recorded, so the thing
cannot be re-examined, and only a description of it survived. This script is
kept because it can be pointed at any future adapter, and because it now
writes its evidence to disk. It is not kept as an open investigation: without
a reproducible green island there is nothing left here to discriminate, and
more GPU spent on it would be spent on a memory.

WHAT IS AND IS NOT CONTROLLED

The generation seed is controlled - both images come from seed 7. The TRAINING
is not bit-identical: the trainer takes no seed, so the two runs differ by
their own initialisation as well as by the captions. Two adapters trained on
identical captions would not be identical either. So a small metric difference
proves nothing; the question this can answer is whether the PICTURE changes
kind, not whether a number moved.
"""

from __future__ import annotations

import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/app")

import core_models  # noqa: E402
import pixelate  # noqa: E402

BASE = "stabilityai/stable-diffusion-xl-base-1.0"
PROMPT = "an island continent with a central mountain range, forests and a coast"
SEED = 7
FIT_N = 4
OUT = "/app/images/_discriminator"

# What the first adapter measured, recorded before the retrain so the
# comparison is against a number that was written down, not remembered.
FIRST = {"flatness": 14.6, "separation": 13.8, "coverage": [26, 35, 20, 19],
         "palette": ["#8db443", "#518447", "#2e6c5b", "#415a40"]}


def tile_vae(pipe) -> list:
    """Make the VAE decode in tiles, and prove it actually did.

    Two traps, both hit on the way here.

    ONE: `pipe.enable_vae_tiling()` is the name in every tutorial and does not
    exist in diffusers 0.40. The API is `pipe.vae.enable_tiling()`.

    TWO - and this is the one worth remembering: calling it and checking
    `vae.use_tiling` PASSED, and the decode still ran whole and still OOMed.
    The flag was set; the behaviour never changed. `_decode` gates on

        z.shape[-1] > self.tile_latent_min_size

    and `tile_latent_min_size` is `sample_size / 8` = 128 for the SDXL VAE,
    while a 1024x1024 image has a 128x128 latent. `128 > 128` is false, so the
    switch was on and the branch was never taken - the flag was exactly one
    pixel short of meaning anything.

    So the thresholds are lowered until the branch can fire, and then
    `tiled_decode` is WRAPPED to record that it ran. Checking the flag is
    checking the description; checking the call is checking the thing.
    """
    vae = pipe.vae
    vae.enable_tiling()
    vae.tile_sample_min_size = 512
    vae.tile_latent_min_size = 64

    calls = []
    original = vae.tiled_decode

    def watched(*a, **kw):
        calls.append(1)
        return original(*a, **kw)

    vae.tiled_decode = watched
    return calls


def paint(model: str, seed: int, trigger: bool = True) -> Image.Image:
    """Exactly the call `_paint` makes, trigger included."""
    from tasks import get_sd_pipeline
    import torch

    body = (f"{PROMPT}, top-down world map, flat regions of colour, "
            f"distinct biomes, no text, no labels, no border, no shading, "
            f"no perspective")
    brief = core_models.apply_trigger(model, body) if trigger else body
    print(f"    prompt: {brief[:110]}...")
    pipe = get_sd_pipeline(model)

    # The VAE decode is what does not fit on this box, and it is not
    # fragmentation - three identical OOMs, all asking for 512 MiB, all with
    # only 55 MiB reserved-but-unallocated, and `expandable_segments:True`
    # changed nothing. The card is simply full: 7.71 GB of fused SDXL plus the
    # 3.6 GB llama.cpp holds permanently is 11.3 of 12.
    #
    # llama.cpp does NOT give that back when it sleeps - it has been idle for
    # hours and still holds it. So a 12 GB card running both services has
    # roughly 8.4 GB for diffusion, not 12, and `_release_vram`'s measured
    # figures were taken with llama.cpp asleep, which is not the state a map
    # build runs in.
    #
    # Tiling decodes the latent in pieces. It does not touch the latent, so
    # composition and seed behaviour are unchanged; only the decode is cheaper.
    tiled = tile_vae(pipe)

    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    out = pipe(prompt=brief, num_inference_steps=25, guidance_scale=7.5,
               generator=gen).images[0].convert("RGB")
    print(f"    decode: tiled ({len(tiled)} call(s))" if tiled else
          "    decode: NOT tiled - it fit anyway, but the switch did nothing")
    return out


def evict(model: str) -> None:
    """Drop one pipeline from the cache and hand the blocks back.

    `get_sd_pipeline` caches by model string, and "base+lora" is a distinct
    string per adapter - so comparing two adapters in one process means two
    fused SDXL pipelines resident at once, about 6.6 GB each on a 12 GB card.
    The first attempt at this script died in the VAE decode for exactly that
    reason, with the queue's own pipeline making a third.

    Evicting between adapters costs a reload and buys the comparison being
    possible at all. See `map_tasks._release_vram` for why `empty_cache` is
    needed on top of dropping the reference: the allocator holds the blocks.
    """
    import gc

    import torch
    import tasks

    tasks.pipes.pop(model, None)
    for key in [k for k in tasks.pipes if model in str(k)]:
        tasks.pipes.pop(key, None)
    gc.collect()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    print(f"    freed   - {free / 2 ** 30:.2f} GiB of {total / 2 ** 30:.2f} "
          f"now free")


def flatness(img: Image.Image, n: int = FIT_N) -> dict:
    """How close is this to being n flat colours?

    Mean Lab distance from every pixel to its nearest palette entry. A map
    painted as flat regions fits a small palette tightly; a shaded, textured
    render does not. This is the number that says whether the painting is
    usable as a source for quantisation at all.
    """
    pal = pixelate.extract_palette(img, n)
    arr = np.asarray(img).reshape(-1, 3)
    lab = pixelate.srgb_to_lab(arr.astype(np.uint8))
    plab = pixelate.srgb_to_lab(pal)

    d = lab[:, None, :] - plab[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", d, d))
    idx = dist.argmin(axis=1)
    err = float(dist.min(axis=1).mean())

    counts = np.bincount(idx, minlength=len(pal))
    cover = [int(round(100 * c / len(idx))) for c in counts]

    pd = plab[:, None, :] - plab[None, :, :]
    pdist = np.sqrt(np.einsum("ijk,ijk->ij", pd, pd))
    np.fill_diagonal(pdist, np.inf)

    return {"flatness": round(err, 1),
            "separation": round(float(pdist.min()), 1),
            "coverage": cover,
            "palette": [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in pal]}


def water_share(img: Image.Image) -> float:
    """Percentage of PIXELS that read as water.

    The first version of this measured the 4-colour palette instead, summing
    the coverage of any entry whose hue looked blue. It reported 0.0% for two
    images that are roughly a fifth open water, and printed a confident verdict
    on the strength of it.

    Two compounding reasons, both worth keeping:

      - four median-cut colours over a map this varied never allocate an entry
        to the water at all - it is averaged into the greens;
      - the entry it did produce, #488269, is a teal whose hue falls just below
        a 0.47 blue threshold, so even the merged colour failed the test.

    A derived summary was standing in for the thing, and the thing was on disk
    the whole time. So this counts pixels. The band is wide because map water
    here is a desaturated cyan, not a primary blue, and green must stay out:
    the two meet near hue 0.45, which is exactly where the old cutoff sat.
    """
    import colorsys

    arr = np.asarray(img).reshape(-1, 3) / 255.0
    n = 0
    for r, g, b in arr[::17]:  # every 17th pixel; the answer is a percentage
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        if 0.45 <= h <= 0.75 and s >= 0.20 and v >= 0.25:
            n += 1
    return round(100.0 * n / len(arr[::17]), 1)


def main() -> int:
    import os

    import torch

    os.makedirs(OUT, exist_ok=True)

    # A fused SDXL pipeline needs roughly 7 GB. Refusing up front beats dying
    # in the VAE decode twenty minutes in, which is how the first attempt
    # ended - the queue was mid-job and the card had 3.18 GiB left.
    free, total = torch.cuda.mem_get_info()
    print(f"  card: {free / 2 ** 30:.2f} GiB free of {total / 2 ** 30:.2f}")
    if free < 8 * 2 ** 30:
        print("  Not enough free VRAM for a fused SDXL pipeline. Something "
              "else is on the card - let the queue drain and retry.")
        return 1
    print()

    # Three arms, not two. The third is the one that matters now: the SAME
    # first adapter with the trigger WITHHELD. The green island this was built
    # to explain was painted before `_triggered` existed in the map path, so
    # "no trigger" is a live candidate that predicts the same picture as both
    # of the original two - and it is one generation to rule in or out.
    ARMS = [("mapstyle", True), ("mapstyle2", True), ("mapstyle", False)]

    results = {}
    for name, trigger in ARMS:
        model = f"{BASE}+local:{name}"
        key = name if trigger else f"{name}-untriggered"
        if not core_models.local_lora_file(f"local:{name}"):
            print(f"  MISSING {name} - no adapter file, cannot compare")
            return 1
        print(f"  {key}: trigger "
              f"{core_models.trigger_for(model)!r}" if trigger else
              f"  {key}: NO trigger - the adapter is loaded but unaddressed")

        img = paint(model, SEED, trigger=trigger)
        path = os.path.join(OUT, f"{key}-seed{SEED}.png")
        img.save(path, format="PNG")

        m = flatness(img)
        m["blue"] = water_share(img)
        m["path"] = path
        results[key] = m
        print(f"    flatness {m['flatness']}  separation {m['separation']}  "
              f"coverage {'/'.join(str(c) for c in m['coverage'])}%")
        print(f"    palette  {' '.join(m['palette'])}")
        print(f"    blue     {m['blue']}% of the image")
        print(f"    written  {path}")
        evict(model)
        print()

    # The recorded first-run figures are checked against this run's, because if
    # they disagree the two adapters are not the only thing that changed.
    a, b = results["mapstyle"], results["mapstyle2"]
    u = results["mapstyle-untriggered"]
    drift = abs(a["flatness"] - FIRST["flatness"])
    print(f"  first adapter, then vs now: flatness {FIRST['flatness']} -> "
          f"{a['flatness']}", end="")
    print("  (same picture)" if drift < 0.5 else
          f"  DRIFT {drift:.1f} - something other than the adapter changed")
    print()

    print("  VERDICT")
    print(f"    water    mapstyle {a['blue']}%   mapstyle2 {b['blue']}%   "
          f"mapstyle-untriggered {u['blue']}%")
    print(f"    flatness mapstyle {a['flatness']}  mapstyle2 {b['flatness']}  "
          f"mapstyle-untriggered {u['flatness']}  (lower is flatter)")
    print()

    wet = 8.0
    if a["blue"] >= wet and b["blue"] >= wet:
        print("    BOTH adapters paint a coast. The premise this script was "
              "written to explain - a monochromatic green island - does not "
              "reproduce, so the captions cannot be what caused it and the "
              "retrain answers a question that is no longer live.")
        if u["blue"] < wet:
            print("    Withholding the trigger DOES lose the coast. That is "
                  "the difference that matters: the green island was an "
                  "unaddressed adapter, not a diluted one.")
        else:
            print(f"    Withholding the trigger changes little "
                  f"({u['blue']}% water), so the trigger is not the "
                  f"explanation either. The green island came from something "
                  f"this script does not vary - look at what else differed.")
    elif b["blue"] >= wet > a["blue"]:
        print("    The coast appeared once the captions were fixed. The "
              "diluted trigger was the cause.")
    else:
        print("    No coast from either. The caption was still a bug, but it "
              "is NOT what makes the island monochromatic.")

    print("\n  Look at all three - the numbers above are a screen, and an "
          "earlier version of this screen scored two coastlines at 0.0%:")
    for k, m in results.items():
        print(f"    {k:<22} {m['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
