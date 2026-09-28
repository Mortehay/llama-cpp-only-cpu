"""Bench: Qwen-Image-2512 (GGUF Q3_K_M) vs SDXL + nerijs/pixel-art-xl.

The measurement behind .ai/decisions/0012. 4 subjects x 3 seeds,
something2-shaped prompts through `split_negations`, both models judged by the
SAME code - the production cutout helpers. Every image is a `generations`
ledger row, so a run shows on the Activity tab: pending while it runs, with
its PNG when done. Qwen rows all flip to done together at the end, because the
transformer is unloaded before decoding (0012 "Open" #1).

Run inside the worker, as SEPARATE processes, in this order - never the text
encoder and the transformer in one process (see qwen_edit.encode_only):

    docker exec sprite_worker bash -c 'cd /app && python scripts/bench-qwen-image.py sdxl'
    docker exec sprite_worker bash -c 'cd /app && python scripts/bench-qwen-image.py encode'
    docker exec sprite_worker bash -c 'cd /app && python scripts/bench-qwen-image.py qwen'
    docker exec sprite_worker bash -c 'cd /app && python scripts/bench-qwen-image.py report'

`results.jsonl` accumulates under /tmp/qwen_bench; delete it before a fresh
run. The worker is not paused - a real job arriving mid-run competes for the
card. Check traffic first.
"""
import json, os, sys, time
sys.path.insert(0, "/app")
import numpy as np
import torch
from PIL import Image

OUT = "/tmp/qwen_bench"
IMAGES = "/app/images"
SUBJECTS = ["knight in obsidian armor", "obsidian boots",
            "green slime monster", "red health potion"]
SEEDS = [101, 202, 303]
TEMPLATE = ("pixel art, {s}, single object, centered, full view, plain white "
            "background, no frame, no border, no card, no text")
BASE_NEG = "blurry, photo, watermark"
CALLER = {"principal_name": "bench: qwen-image-2512 vs sdxl (claude)"}

_GGUF_NAME = os.path.basename(os.environ.get("BENCH_QWEN_GGUF", "Qwen-Image-2512-Q3_K_M.gguf"))
QWEN = dict(label=_GGUF_NAME, size=512, steps=20, cfg=4.0)

# Optional Lightning run, same subjects/seeds/judge so the rows compare:
#   BENCH_QWEN_LORA=Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors
#   BENCH_QWEN_STEPS=8  BENCH_QWEN_CFG=1.0
# A step-distilled LoRA runs at true CFG 1 - the negative prompt is then
# INERT, the SDXL-Turbo trap - which is the thing this run exists to measure.
LIGHTNING_REPO = "lightx2v/Qwen-Image-2512-Lightning"
LORA = os.environ.get("BENCH_QWEN_LORA", "")
if LORA:
    QWEN = dict(label=f"{_GGUF_NAME} + {LORA.split('-V1')[0]}",
                size=512, steps=int(os.environ.get("BENCH_QWEN_STEPS", "8")),
                cfg=float(os.environ.get("BENCH_QWEN_CFG", "1.0")))
QWEN_TAG = os.environ.get("BENCH_QWEN_TAG",
                          f"qwen2512-l{QWEN['steps']}" if LORA else "qwen2512")
SDXL = dict(label="stabilityai/stable-diffusion-xl-base-1.0+nerijs/pixel-art-xl",
            size=1024, steps=25, cfg=7.0)

REPO = "ovedrive/Qwen-Image-Edit-2511-4bit"
CFG_REPO = "Qwen/Qwen-Image-2512"
GGUF = os.environ.get("BENCH_QWEN_GGUF",
                      "/models/image-gguf/transformers/Qwen-Image-2512-Q3_K_M.gguf")
DT = torch.bfloat16


def prompts():
    from tasks import split_negations
    return {s: split_negations(TEMPLATE.format(s=s), BASE_NEG) for s in SUBJECTS}


def cases():
    return [(s, seed) for s in SUBJECTS for seed in SEEDS]


def judge(img: Image.Image) -> dict:
    """Production cutout path, plus a stray-blob count it does not report."""
    from scipy import ndimage
    from tasks import remove_background, _isolate_largest_sprite
    from pixelate import strip_ground_patch
    from measure import palette_of, pixel_scale

    raw = np.asarray(img.convert("RGBA"))
    cut = remove_background(img, keep_largest=False)
    a = np.asarray(cut.convert("RGBA")).copy()
    op = a[..., 3] >= 128
    lab, n = ndimage.label(a[..., 3] > 0, structure=np.ones((3, 3), bool))
    sizes = sorted(ndimage.sum(a[..., 3] > 0, lab, range(1, n + 1)), reverse=True) if n else []
    big = [s for s in sizes if sizes and s >= 0.15 * sizes[0]]
    fill = float(op.mean())
    iso = Image.fromarray(_isolate_largest_sprite(a), mode="RGBA")
    kept = float((np.asarray(iso)[..., 3] >= 128).mean()) / fill if fill else 1.0
    iso = strip_ground_patch(iso, require_legs=True)
    clear = float((np.asarray(iso.convert("RGBA"))[..., 3] < 128).mean())

    if clear < 0.02:
        verdict = "no_cutout"
    elif kept < 0.20:
        verdict = "sheet"           # production would reject (cutout_failed)
    elif clear > 0.995:
        verdict = "empty"
    elif len(big) > 1:
        verdict = "multi"           # production ACCEPTS, but keeps one of several
    else:
        verdict = "single"

    # Style, on the subject only (opaque cutout pixels), at native resolution.
    iso_a = np.asarray(iso.convert("RGBA"))
    px = iso_a[iso_a[..., 3] >= 128][:, :3]
    top32 = 0.0
    if len(px):
        _, counts = np.unique(px.reshape(-1, 3), axis=0, return_counts=True)
        top32 = float(np.sort(counts)[::-1][:32].sum() / counts.sum())
    ps = pixel_scale(raw)
    return {"verdict": verdict, "kept_pct": round(kept * 100, 1),
            "blobs_ge15pct": len(big), "transparent_pct": round(clear * 100, 1),
            "top32_cover_pct": round(top32 * 100, 1),
            "exact_colors": palette_of(iso_a)["colors"],
            "pixel_scale": ps.get("scale"),
            "pixel_scale_fit": ps.get("explained", ps.get("fit", ps.get("confidence"))),
            }, iso


def ledger_begin(model_cfg, subj, seed, pos, neg):
    import generations
    return generations.begin(
        kind="entity", name=None, route="bench:qwen-image-2512",
        prompt=pos, negative_prompt=neg, model=model_cfg["label"], seed=seed,
        params={"bench": "qwen2512-vs-sdxl", "subject": subj,
                "size": model_cfg["size"], "steps": model_cfg["steps"],
                "cfg": model_cfg["cfg"]},
        caller=CALLER)


def ledger_done(gid, img, tag, subj, seed, t_ms, extra=None):
    import generations
    m, cut = judge(img)
    stem = f"bench_{tag}_{subj.replace(' ', '-')}_{seed}"
    raw_path = f"{IMAGES}/{stem}_raw.png"
    img.save(raw_path)
    cut.save(f"{IMAGES}/{stem}_cut.png")
    params = {**m, **(extra or {}), "cutout_url": f"/images/{stem}_cut.png"}
    generations.finish(gid, file_path=raw_path, seed=seed, duration_ms=t_ms,
                       params=params)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/results.jsonl", "a") as f:
        f.write(json.dumps({"model": tag, "subject": subj, "seed": seed,
                            "ms": int(t_ms), **params}) + "\n")
    print(f"  {tag} {subj!r} seed {seed}: {m['verdict']} kept {m['kept_pct']}% "
          f"blobs {m['blobs_ge15pct']} top32 {m['top32_cover_pct']}% "
          f"({t_ms/1000:.0f}s)", flush=True)


# --- Qwen ------------------------------------------------------------------

def encode():
    from diffusers import QwenImagePipeline
    pipe = QwenImagePipeline.from_pretrained(REPO, transformer=None, vae=None,
                                             dtype=DT, cache_dir="/models")
    pipe.to("cuda")
    out = {}
    with torch.no_grad():
        for s, (pos, neg) in prompts().items():
            e, m = pipe.encode_prompt(prompt=[pos], device=torch.device("cuda"))
            ne, nm = pipe.encode_prompt(prompt=[neg], device=torch.device("cuda"))
            out[s] = [(e.cpu(), m.cpu() if m is not None else None),
                      (ne.cpu(), nm.cpu() if nm is not None else None)]
            print("encoded", s, flush=True)
    os.makedirs(OUT, exist_ok=True)
    torch.save(out, f"{OUT}/embeds.pt")


def qwen():
    from diffusers import (FlowMatchEulerDiscreteScheduler, GGUFQuantizationConfig,
                           QwenImagePipeline, QwenImageTransformer2DModel)
    from qwen_edit import _decode_latent, _free
    emb = torch.load(f"{OUT}/embeds.pt")
    P = prompts()
    gids = {c: ledger_begin(QWEN, c[0], c[1], *P[c[0]]) for c in cases()}

    t0 = time.time()
    tr = QwenImageTransformer2DModel.from_single_file(
        GGUF, quantization_config=GGUFQuantizationConfig(compute_dtype=DT),
        config=CFG_REPO, subfolder="transformer", dtype=DT, cache_dir="/models")
    if LORA:
        # The distillation's own scheduler (shift=3), from ModelTC's
        # generate_with_diffusers.py - the stock one is not what it learned.
        import math
        sched = FlowMatchEulerDiscreteScheduler.from_config({
            "base_image_seq_len": 256, "base_shift": math.log(3),
            "invert_sigmas": False, "max_image_seq_len": 8192,
            "max_shift": math.log(3), "num_train_timesteps": 1000,
            "shift": 1.0, "shift_terminal": None, "stochastic_sampling": False,
            "time_shift_type": "exponential", "use_beta_sigmas": False,
            "use_dynamic_shifting": True, "use_exponential_sigmas": False,
            "use_karras_sigmas": False})
    else:
        sched = FlowMatchEulerDiscreteScheduler.from_pretrained(
            CFG_REPO, subfolder="scheduler", cache_dir="/models")
    pipe = QwenImagePipeline.from_pretrained(
        REPO, transformer=tr, scheduler=sched, text_encoder=None, tokenizer=None,
        dtype=DT, cache_dir="/models")
    if LORA:
        pipe.load_lora_weights(LIGHTNING_REPO, weight_name=LORA,
                               cache_dir="/models")
    pipe.to("cuda")
    free, _ = torch.cuda.mem_get_info()
    print(f"VRAM free after placement: {free / 2**30:.2f} GiB", flush=True)
    print(f"transformer placed in {time.time()-t0:.0f}s", flush=True)

    lats, times = {}, {}
    for c in cases():
        (pe, pm), (ne, nm) = emb[c[0]]
        t1 = time.time()
        try:
            lats[c] = pipe(prompt_embeds=pe.cuda(),
                           prompt_embeds_mask=pm.cuda() if pm is not None else None,
                           negative_prompt_embeds=ne.cuda(),
                           negative_prompt_embeds_mask=nm.cuda() if nm is not None else None,
                           true_cfg_scale=QWEN["cfg"], height=QWEN["size"],
                           width=QWEN["size"], num_inference_steps=QWEN["steps"],
                           generator=torch.Generator("cpu").manual_seed(c[1]),
                           output_type="latent").images.cpu()
            times[c] = (time.time() - t1) * 1000
            print(f"denoised {c} {times[c]/1000:.0f}s", flush=True)
        except Exception as e:
            import generations
            generations.fail(gids[c], e)
            print("FAILED", c, e, flush=True)

    vae, vsf, proc = pipe.vae, pipe.vae_scale_factor, pipe.image_processor
    pipe.transformer = None; del pipe, tr; _free()
    vae.to("cuda"); vae.enable_tiling()
    for c, lat in lats.items():
        img = _decode_latent(vae, lat, QWEN["size"], QWEN["size"], vsf, proc)
        ledger_done(gids[c], img, QWEN_TAG, c[0], c[1], times[c])
    vae.to("cpu"); _free()


# --- SDXL baseline ---------------------------------------------------------

def sdxl():
    import tasks, generations
    P = prompts()
    gids = {c: ledger_begin(SDXL, c[0], c[1], *P[c[0]]) for c in cases()}
    for c in cases():
        pos, neg = P[c[0]]
        t1 = time.time()
        r = tasks._generate_raw_once(None, pos, neg, SDXL["label"], SDXL["size"],
                                     SDXL["size"], SDXL["steps"], SDXL["cfg"],
                                     c[1], strip_background=False)
        if not r or r.get("error"):
            generations.fail(gids[c], (r or {}).get("error", "no result"))
            print("FAILED", c, r, flush=True)
            continue
        img = Image.open(r["file_path"]).convert("RGB")
        os.remove(r["file_path"])
        ledger_done(gids[c], img, "sdxl-nerijs", c[0], c[1], (time.time() - t1) * 1000)
    tasks.release_vram_cache("bench")


def report():
    rows = [json.loads(l) for l in open(f"{OUT}/results.jsonl")]
    for tag in sorted({x["model"] for x in rows}):
        r = [x for x in rows if x["model"] == tag]
        if not r:
            continue
        v = {k: sum(1 for x in r if x["verdict"] == k)
             for k in ("single", "multi", "sheet", "no_cutout", "empty")}
        print(f"{tag}: n={len(r)} verdicts={v} "
              f"mean top32={np.mean([x['top32_cover_pct'] for x in r]):.1f}% "
              f"mean kept={np.mean([x['kept_pct'] for x in r]):.1f}% "
              f"mean s/img={np.mean([x['ms'] for x in r])/1000:.0f}")
        for x in r:
            print(f"   {x['subject']:<26} {x['seed']}  {x['verdict']:<9} kept {x['kept_pct']:>5}  "
                  f"blobs {x['blobs_ge15pct']}  top32 {x['top32_cover_pct']:>5}  "
                  f"colors {x['exact_colors']:>6}  scale {x['pixel_scale']} fit {x['pixel_scale_fit']}")


if __name__ == "__main__":
    {"encode": encode, "qwen": qwen, "sdxl": sdxl, "report": report}[sys.argv[1]]()
