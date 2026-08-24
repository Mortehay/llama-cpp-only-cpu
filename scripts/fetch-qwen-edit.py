#!/usr/bin/env python3
"""Fetch exactly the Qwen-Image-Edit-2511 pieces that fit a 12 GB card.

The obvious command - snapshot_download("Qwen/Qwen-Image-Edit-2511") - pulls
about 60 GB, most of it a bf16 20B transformer this card cannot run and a bf16
16 GB text encoder that would have to be quantised on the way in (which needs
more host RAM than WSL is given here). Both halves already exist pre-quantised,
from two different publishers, so this assembles them:

    transformer   unsloth/Qwen-Image-Edit-2511-GGUF     Q3_K_M    9.92 GB
    everything    ovedrive/Qwen-Image-Edit-2511-4bit    NF4       ~5.4 GB
    else                                                          -------
                                                                  ~15.3 GB

Why the transformer comes from the GGUF repo and not the 4bit one: the 4bit
repo's own transformer is 11.59 GB, which does not stay resident on an ~11.7 GB
card. Q3_K_M at 9.92 GB does, and staying resident across all ~150 cells of a
sheet is the difference between a 40-minute job and an overnight one. See
`src/sprite_generator/qwen_edit.py`.

The `transformer/` subfolder of the 4bit repo is therefore explicitly ignored.
Forgetting that ignore silently adds 11.59 GB of weights nothing ever loads.

Usage:
    python fetch-qwen-edit.py [--models-dir /models] [--dry-run]
"""

import argparse
import os
import sys

# What to pull, in the order that fails cheapest first. The small repos come
# before the 10 GB one so a bad token or a typo surfaces in seconds.
PLAN = [
    {
        "repo": "ovedrive/Qwen-Image-Edit-2511-4bit",
        "why": "NF4 text encoder, VAE, tokenizer, scheduler, model_index",
        # Everything except the transformer - see module docstring.
        "ignore": ["transformer/*"],
        "approx_gb": 5.4,
    },
    {
        "repo": "Qwen/Qwen-Image-Edit-2511",
        "why": "transformer architecture config ONLY - a few KB, not weights",
        # from_single_file needs a config to know what architecture the GGUF
        # tensors belong to, and qwen_edit points it at the official repo
        # (the 4bit repo's config carries a bitsandbytes quantization_config
        # that contradicts GGUF weights). Without this cached, the pipeline
        # works online and dies under HF_HUB_OFFLINE=1 with a misleading
        # "does not appear to have a file named config.json" - measured
        # 2026-08-24, after offline mode was switched on.
        "allow": ["transformer/config.json", "model_index.json"],
        "approx_gb": 0.001,
    },
    {
        "repo": "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
        "why": "8 azimuths x 4 elevations - the reason for this whole route",
        # The repo also ships a 35 MB demo GIF. Harmless, but there is no
        # reason to keep it in a model cache.
        "allow": ["qwen-image-edit-2511-multiple-angles-lora.safetensors"],
        "approx_gb": 0.30,
    },
    {
        "repo": "lightx2v/Qwen-Image-Edit-2511-Lightning",
        "why": "4-step distillation; 40 steps is not affordable per cell here",
        # MANDATORY, not tidiness. This repo is 107.7 GB: it carries 4- and
        # 8-step LoRAs in bf16 AND fp32, a 20.5 GB fp8 full checkpoint, and 60
        # split block files. An unfiltered snapshot_download would try to pull
        # all of it and fill the disk long before it finished.
        "allow": ["Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"],
        "approx_gb": 0.85,
    },
    {
        "repo": "unsloth/Qwen-Image-Edit-2511-GGUF",
        "why": "the transformer itself",
        "allow": ["Qwen-Image-Edit-2511-Q3_K_M.gguf"],
        "approx_gb": 9.92,
    },
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", default=os.environ.get("MODELS_DIR", "/models"))
    p.add_argument("--dry-run", action="store_true")
    # NOTE the lowercase repo prefix. The repo is named
    # `unsloth/Qwen-Image-Edit-2511-GGUF` but its files are
    # `qwen-image-edit-2511-*.gguf`. Getting this wrong does not 404 - it makes
    # allow_patterns match nothing, and snapshot_download cheerfully reports
    # "Fetching 0 files" and succeeds. See the zero-file guard below.
    # Q2_K, matching qwen_edit.GGUF_FILE. These two defaults MUST agree: they
    # disagreed once (fetch said Q3_K_M, the pipeline used Q2_K) and a routine
    # re-run silently began re-downloading 9.9 GB of a quant that had been
    # deliberately deleted for not fitting the card.
    p.add_argument("--gguf", default="qwen-image-edit-2511-Q2_K.gguf",
                   help="override the quant, e.g. qwen-image-edit-2511-Q3_K_M.gguf")
    a = p.parse_args()

    for step in PLAN:
        if step["repo"].endswith("-GGUF"):
            step["allow"] = [a.gguf]

    # Refuse to create the cache directory.
    #
    # This bit once, and expensively. Running the script in a container with
    # `--env-file compose/develop/.env` puts the HOST value of MODELS_DIR
    # (/home/markunn/sprite-data/models) into the container's environment, where
    # that path is not the mount - the mount is /models. huggingface_hub happily
    # created it inside the container's own filesystem and started downloading
    # 16 GB into a layer that `docker run --rm` discards on exit. Nothing
    # errored; the bytes simply were not there afterwards.
    #
    # A missing cache dir is therefore always a mistake, never something to fix
    # by creating it.
    if not os.path.isdir(a.models_dir):
        print(f"cache dir does not exist: {a.models_dir}\n\n"
              f"Refusing to create it. Inside a container this almost always "
              f"means MODELS_DIR leaked in from the host env-file and points "
              f"somewhere that is not the bind mount - pass --models-dir "
              f"/models explicitly.", file=sys.stderr)
        return 2

    total = sum(s["approx_gb"] for s in PLAN)
    print(f"cache dir : {a.models_dir}")
    print(f"to fetch  : ~{total:.1f} GB across {len(PLAN)} repos\n")
    for s in PLAN:
        filt = ""
        if s.get("allow"):
            filt = f"  [only {', '.join(s['allow'])}]"
        elif s.get("ignore"):
            filt = f"  [skipping {', '.join(s['ignore'])}]"
        print(f"  {s['approx_gb']:5.2f} GB  {s['repo']}{filt}")
        print(f"             {s['why']}")
    print()

    if a.dry_run:
        return 0

    # Imported late so --dry-run works without the dependency present.
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or None
    failures = []

    for s in PLAN:
        print(f"=== {s['repo']} ===", flush=True)
        try:
            path = snapshot_download(
                repo_id=s["repo"],
                cache_dir=a.models_dir,
                token=token,
                allow_patterns=s.get("allow"),
                ignore_patterns=s.get("ignore"),
                max_workers=4,
            )

            # A mistyped allow_pattern is NOT an error to snapshot_download. It
            # matches nothing, prints "Fetching 0 files", returns a valid path
            # to an empty snapshot, and exits 0. That happened here: the repo is
            # `Qwen-Image-Edit-2511-GGUF` but its files are
            # `qwen-image-edit-2511-*.gguf`, and 10 GB silently did not arrive.
            # Nothing downstream notices until the pipeline fails to load.
            got = [f for f in os.listdir(path)
                   if not f.startswith(".")] if os.path.isdir(path) else []
            if not got:
                raise RuntimeError(
                    f"snapshot is empty - allow_patterns "
                    f"{s.get('allow')} matched no file in the repo. Check the "
                    f"exact filenames at "
                    f"https://huggingface.co/{s['repo']}/tree/main"
                )

            print(f"    -> {path}", flush=True)
            print(f"       {len(got)} entr{'y' if len(got) == 1 else 'ies'}: "
                  f"{', '.join(sorted(got)[:4])}\n", flush=True)
        except Exception as e:
            # Keep going. The LoRAs are independently useful and a transient
            # failure on the 10 GB file should not discard the 5 GB that landed.
            failures.append(s["repo"])
            print(f"    FAILED: {type(e).__name__}: {e}\n", file=sys.stderr,
                  flush=True)

    if failures:
        print(f"\n{len(failures)} repo(s) FAILED: {', '.join(failures)}",
              file=sys.stderr)
        return 1

    print("done. Verify with:")
    print("  make turnaround core=images/core_XXXX.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
