"""The step-1 core-model roster, and whether each one is actually on disk.

WHY THIS MODULE EXISTS
The roster used to be five hand-written <option> tags in templates/index.html.
That was fine while every checkpoint was cached, and became a trap the moment
scripts/archive-all-models.sh moved the weights to cold storage on D:. The
dropdown kept offering all five, the worker kept answering "Model failed to
load on worker", and nothing anywhere said the model was simply not there.
HF_HUB_OFFLINE=1 means a missing checkpoint can never be recovered by trying
harder, so offering it is a dead end the UI should not present as a choice.

Both containers mount the same cache at /models, so the web process can answer
"is this on disk?" itself — no Celery round-trip, no worker wake-up. The worker
imports the same helpers so its error message and the dropdown cannot disagree.

Deliberately dependency-free (os + typing only): tasks.py imports it, and
anything that pulls torch into the web process is a mistake.
"""

import os

# huggingface_hub reads HF_HUB_CACHE; compose sets it to /models for both
# containers. Honour it rather than hardcoding, so a relocated cache does not
# make this module quietly report everything missing.
HF_CACHE = os.environ.get("HF_HUB_CACHE") or "/models"

# GGUF diffusion transformers: files, not Hub repos, like trained LoRAs.
# `gguf:<stem>` names /models/image-gguf/transformers/<stem>.gguf. They are NOT
# loaded by get_sd_pipeline - tasks.generate_core_task routes them to
# qwen_t2i.py.
#
# TWO levels deep on purpose. llm_engine runs llama.cpp as a router with
# `--models-dir /models` and offers as a TEXT model every .gguf directly in
# /models and every folder that directly contains one. At /models/gguf/ the
# Qwen-Image transformer was listed as a chat model named "gguf", and
# worlds._llm_model() takes the router's first entry (found 2026-09-28). Nested
# one level further, the router does not see it - the same reason the HF-cache
# GGUF under models--unsloth--... never appeared. Do not flatten this path.
GGUF_PREFIX = "gguf:"
GGUF_DIR = os.path.join(HF_CACHE, "image-gguf", "transformers")

# Everything a Qwen-Image GGUF needs besides itself. The Edit repo's NF4 text
# encoder and VAE are hash-identical to Qwen-Image-2512's; the 2512 repo
# supplies only the transformer/scheduler config. See decisions/0012.
GGUF_COMPANION_REPOS = ("ovedrive/Qwen-Image-Edit-2511-4bit",
                        "Qwen/Qwen-Image-2512")


def is_gguf(value: str) -> bool:
    return (value or "").strip().startswith(GGUF_PREFIX)


# Step-distilled LoRAs a GGUF value may name as `gguf:<stem>+<key>`. The
# settings travel with the key because they are not optional: a Lightning LoRA
# run at 20 steps / CFG 4 on the stock scheduler is not what it was trained for.
GGUF_LORAS = {
    "lightning8": {
        "repo": "lightx2v/Qwen-Image-2512-Lightning",
        "weight": "Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors",
        "steps": 8, "cfg": 1.0,
    },
}


def _gguf_parts(value: str) -> tuple[str, str | None]:
    """`gguf:<stem>[+<lora key>]` -> (stem, key or None)."""
    body = (value or "").strip()[len(GGUF_PREFIX):]
    stem, _, key = body.partition("+")
    return stem, (key or None)


def gguf_file(value: str) -> str | None:
    """Absolute path for a `gguf:<stem>` value, or None if it is not one."""
    if not is_gguf(value):
        return None
    return os.path.join(GGUF_DIR, _gguf_parts(value)[0] + ".gguf")


def persistent_qwen(value: str) -> bool:
    """Is this value served by the persistent Qwen process (qwen_server.py)?

    Only the step-distilled variants: a request there is ~25 s of denoising,
    so the ~75 s of per-request re-loading the persistent process removes is
    most of the wall clock. The 20-step Q3_K_M is left on the per-request
    subprocess path - its 4.8 + 9.7 GiB would not park in RAM beside anything
    else. QWEN_PERSISTENT=0 turns it off everywhere. decisions/0012 "2c".
    """
    return (os.environ.get("QWEN_PERSISTENT", "1").lower()
            not in ("0", "false", "no")
            and gguf_lora(value) is not None)


def gguf_lora(value: str) -> dict | None:
    """The LoRA settings a `gguf:<stem>+<key>` value names, or None."""
    if not is_gguf(value):
        return None
    key = _gguf_parts(value)[1]
    return GGUF_LORAS.get(key) if key else None


# UI_DEFAULT_WHY: `ui_default` is what the browser preselects; `default` is what
# default_model() returns, and default_model() is also main.py's Form default
# for every script and API caller that omits llm_name. They differ on purpose:
# the UI wants the best image, a batch caller must not silently get a model
# that takes ~4 minutes. Do not merge them. See decisions/0012 D1.

# The roster. `value` is the string the worker receives: "<base>" or
# "<base>+<lora>", parsed by get_sd_pipeline, or "gguf:<stem>" (above).
# Comments that used to sit beside the <option> tags travelled here with them.
CORE_MODELS = [
    {
        # Benched 2026-09-28 (decisions/0012 2b): 7.33 GB Q2_K transformer +
        # lightx2v 8-step Lightning LoRA, true CFG 1. 0/12 contact sheets,
        # 24 s/image flat, 3.12 GiB VRAM free; knights 3/3 full-body. Rougher
        # textures than the 20-step entry below, and less variety between
        # seeds. The owner chose it as the UI preselection and something2's
        # default (a1111.KNOWN_MODELS[0]) - speed over the last bit of polish.
        "value": "gguf:qwen-image-2512-Q2_K+lightning8",
        "label": "Qwen-Image-2512 fast (Lightning) - ~30 s, slightly rougher",
        "ui_default": True,
    },
    {
        # Benched 2026-09-28 against the default below, 4 subjects x 3 seeds:
        # 0/12 contact sheets vs 2/12, cutout kept 98.2% vs 77.4%, and judged
        # "much better" by the owner. ~3.5-4 min per core (512px, 20 steps,
        # true CFG 4), so the UI preselects it but default_model() does not,
        # and a1111 refuses it - something2's ceiling is 300s. decisions/0012.
        "value": "gguf:Qwen-Image-2512-Q3_K_M",
        "label": "Qwen-Image-2512 (GGUF) - best quality, ~4 min per image",
    },
    {
        # Default, measured 2026-08-21. SDXL base with a pixel-art LoRA fused on
        # top is the only configuration that produced structurally real pixel
        # art: 86.6% of pixels from 32 colours, 74.7% blockiness, against
        # 36.9%/25.2% for sdxl-turbo. The base supplies anatomy and prompt
        # adherence; the 0.32GB LoRA supplies the style. Put "pixel art" in the
        # prompt - that is the LoRA's trigger. ~21s at 25 steps, 1024px.
        "value": "stabilityai/stable-diffusion-xl-base-1.0+nerijs/pixel-art-xl",
        "label": 'SDXL + pixel-art LoRA - best measured, prompt with "pixel art"',
        "default": True,
    },
    {
        "value": "stabilityai/stable-diffusion-xl-base-1.0+Muapi/soft-pixel-art-xl",
        "label": "SDXL + soft-pixel-art LoRA - untested, softer style",
    },
    {
        "value": "stabilityai/stable-diffusion-xl-base-1.0+ntc-ai/SDXL-LoRA-slider.pixel-art",
        "label": "SDXL + pixel-art slider LoRA - untested, 0.08GB",
    },
    {
        "value": "stabilityai/stable-diffusion-xl-base-1.0",
        "label": "SDXL base - no pixel style, use for non-sprite subjects",
    },
    {
        # Benched 2026-08-21, same subject and seed across four models. The only
        # one that produced real pixel art unaided: hard grid, bounded palette,
        # one character, 3.0s at 20 steps / cfg 7.5. SD1.5, so ControlNet-
        # compatible and usable for step 2 as well. Needs its trigger word
        # 'pixelsprite' in the prompt.
        #
        # sdxl-turbo and kohbanye/pixel-art-style were dropped: turbo is
        # distilled (guidance 0, negative prompts inert) and lost the bench,
        # kohbanye was never used. John6666/super-pixelart-xl loads cleanly and
        # returns pure RGB noise. See .ai/decisions/0002.
        "value": "PublicPrompts/All-In-One-Pixel-Model",
        "label": "All-In-One Pixel (SD1.5) - best benched, trigger 'pixelsprite'",
    },
]


def repos_for(value: str) -> list[str]:
    """The HF repo ids a roster entry needs. "<base>+<lora>" needs both.

    `local:<name>` parts are excluded: they are files this machine trained, not
    Hub repos, and treating one as a repo id would report a perfectly good
    adapter as "not in the local cache".
    """
    if is_gguf(value):
        lora = gguf_lora(value)
        return list(GGUF_COMPANION_REPOS) + ([lora["repo"]] if lora else [])
    return [part.strip() for part in (value or "").split("+")
            if part.strip() and not part.strip().startswith("local:")]


def _cache_name(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def is_cached(repo_id: str) -> bool:
    """True when the repo has at least one non-empty snapshot on disk.

    Checking `snapshots/` rather than the repo directory is what distinguishes
    a usable cache from the empty shell huggingface_hub leaves behind after a
    failed fetch — and from the `.locks/` entry that survives archiving.
    """
    snapshots = os.path.join(HF_CACHE, _cache_name(repo_id), "snapshots")
    if not os.path.isdir(snapshots):
        return False
    try:
        for rev in os.scandir(snapshots):
            if rev.is_dir() and any(os.scandir(rev.path)):
                return True
    except OSError:
        return False
    return False


def was_cached(repo_id: str) -> bool:
    """True when this repo was cached here at some point.

    huggingface_hub's `.locks/models--*` directories are not removed when the
    weights are archived or deleted, which makes them a reliable "this used to
    be here" marker — the difference between "archived, restore it" and "never
    downloaded", which are different jobs for the operator.
    """
    return os.path.isdir(os.path.join(HF_CACHE, ".locks", _cache_name(repo_id)))


def local_weight_name(repo_id: str) -> str | None:
    """The single weights file cached for `repo_id`, or None.

    diffusers' `load_lora_weights(repo_id)` normally asks the Hub which file to
    take. Under HF_HUB_OFFLINE=1 it cannot, and raises

        ValueError: When using the offline mode, you must specify a `weight_name`.

    which get_sd_pipeline catches and downgrades to a warning — so the LoRA
    silently did not apply and the "SDXL + pixel-art LoRA" option quietly
    produced plain SDXL output. Naming the file is the whole fix, and the cache
    already knows it: every LoRA repo here snapshots exactly one .safetensors.

    Returns None when the answer is not unambiguous, leaving diffusers to do
    whatever it would have done rather than guessing at a multi-file repo.
    """
    snapshots = os.path.join(HF_CACHE, _cache_name(repo_id), "snapshots")
    if not os.path.isdir(snapshots):
        return None
    found = set()
    try:
        for rev in os.scandir(snapshots):
            if not rev.is_dir():
                continue
            for f in os.scandir(rev.path):
                if f.name.endswith((".safetensors", ".bin")):
                    found.add(f.name)
    except OSError:
        return None
    return found.pop() if len(found) == 1 else None


def missing_repos(value: str) -> list[str]:
    """Repo ids this roster entry needs that are not on disk."""
    return [r for r in repos_for(value) if not is_cached(r)]


def unavailable_reason(value: str) -> str | None:
    """Human-readable reason this model cannot be loaded, or None if it can.

    Worded for whoever has to fix it: it names the restore command when the
    weights were archived, and the fetch route when they were never here.
    """
    # A GGUF transformer is a file too. Its companions (encoder, VAE, config)
    # ARE Hub repos, and fall through to the cache check below.
    if is_gguf(value) and _gguf_parts(value)[1] and not gguf_lora(value):
        return (f"Unknown LoRA key '{_gguf_parts(value)[1]}' in '{value}'; "
                f"known: {', '.join(GGUF_LORAS)}.")
    path = gguf_file(value)
    if path and not os.path.isfile(path):
        return (f"GGUF transformer {os.path.basename(path)} is missing from "
                f"{GGUF_DIR}. Copy it there from where it was downloaded - "
                f"see .ai/decisions/0012.")

    # A trained adapter is a file, not a repo: check it exists before the Hub
    # cache questions below, which know nothing about it.
    for part in (value or "").split("+"):
        path = local_lora_file(part.strip())
        if path and not os.path.isfile(path):
            return (f"Trained adapter {os.path.basename(path)} is missing from "
                    f"{LORA_DIR}. Train it again, or restore it - adapters are "
                    f"not re-downloadable.")

    missing = missing_repos(value)
    if not missing:
        return None
    archived = [r for r in missing if was_cached(r)]
    names = ", ".join(missing)
    if archived:
        cmds = " ".join(f"./scripts/archive-models.sh restore {_cache_name(r)}"
                        for r in archived)
        return (f"Not in the local cache: {names}. It was cached here before, so "
                f"it is most likely in cold storage on D: — restore with: {cmds}")
    return (f"Not in the local cache: {names}. HF_HUB_OFFLINE=1, so the worker "
            f"cannot fetch it; download it with HF_HUB_OFFLINE=0 first.")


def roster() -> list[dict]:
    """The roster with availability resolved. Shape the UI and API both use.

    Trained adapters come FIRST: if someone has spent hours training their own
    style, it is the thing they mean to use, and burying it under five stock
    options is the wrong default.
    """
    out = []
    for entry in local_roster() + CORE_MODELS:
        reason = unavailable_reason(entry["value"])
        out.append({
            "value": entry["value"],
            "label": entry["label"],
            "default": bool(entry.get("default")),
            # What the UI preselects. Distinct from `default` on purpose -
            # see UI_DEFAULT_WHY below.
            "ui_default": bool(entry.get("ui_default")),
            "available": reason is None,
            "reason": reason,
            "missing": missing_repos(entry["value"]),
            "trained": bool(entry.get("trained")),
            "trigger": entry.get("trigger"),
        })
    return out


# ---------------------------------------------------------------------------
# Locally trained LoRAs
# ---------------------------------------------------------------------------
#
# scripts/train-lora.py writes adapters here, beside the HF cache rather than
# inside it: they are not Hub repos and must survive archive-models.sh, which
# moves `models--*` directories to cold storage on D:. A trained LoRA is the
# most expensive artefact on this machine to reproduce - hours of GPU time - so
# it does not live somewhere a cleanup script sweeps.
LORA_DIR = os.path.join(HF_CACHE, "loras")

# Marks a roster value as pointing at LORA_DIR instead of the Hub.
LOCAL_PREFIX = "local:"

# Locally trained adapters are SDXL, so they attach to the SDXL base.
LOCAL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"


def local_lora_file(spec: str) -> str | None:
    """Absolute path for a `local:<name>` spec, or None if it is not one."""
    if not spec.startswith(LOCAL_PREFIX):
        return None
    return os.path.join(LORA_DIR, spec[len(LOCAL_PREFIX):] + ".safetensors")


def trained_loras() -> list[dict]:
    """Every adapter in LORA_DIR, newest first, with its training metadata.

    The sidecar .json is written by train-lora.py and carries the trigger
    token, which is useless information to lose: a style LoRA whose trigger
    nobody remembers is an inert 186 MB file.
    """
    if not os.path.isdir(LORA_DIR):
        return []
    out = []
    try:
        entries = [e for e in os.scandir(LORA_DIR)
                   if e.is_file() and e.name.endswith(".safetensors")]
    except OSError:
        return []
    for e in sorted(entries, key=lambda x: x.stat().st_mtime, reverse=True):
        name = e.name[: -len(".safetensors")]
        meta = {}
        meta_path = os.path.join(LORA_DIR, name + ".json")
        if os.path.isfile(meta_path):
            try:
                import json
                with open(meta_path) as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
        out.append({
            "name": name,
            "path": e.path,
            "size_mb": round(e.stat().st_size / 1e6, 1),
            "trigger": meta.get("trigger", f"<{name}-style>"),
            "images": meta.get("images"),
            "steps": meta.get("steps"),
        })
    return out


def local_roster() -> list[dict]:
    """Trained adapters as roster entries, in the same shape as CORE_MODELS."""
    out = []
    for lora in trained_loras():
        detail = []
        if lora["images"]:
            detail.append(f"{lora['images']} images")
        if lora["steps"]:
            detail.append(f"{lora['steps']} steps")
        suffix = f" - {', '.join(detail)}" if detail else ""
        out.append({
            "value": f"{LOCAL_BASE}+{LOCAL_PREFIX}{lora['name']}",
            "label": (f"Trained: {lora['name']} - prompt with "
                      f"{lora['trigger']}{suffix}"),
            "default": False,
            "trained": True,
            "trigger": lora["trigger"],
        })
    return out


def default_model() -> str:
    """The roster entry marked default, for callers that need *a* model.

    Reads CORE_MODELS rather than roster(): a locally trained adapter sorts
    first in the UI but must not silently become the default for, say, a tile
    job that never asked for a character style.
    """
    for entry in CORE_MODELS:
        if entry.get("default"):
            return entry["value"]
    return CORE_MODELS[0]["value"]


def trigger_for(value: str) -> str | None:
    """The trigger token a roster value needs in its prompt, or None.

    A trained adapter is inert without its trigger: the LoRA loads, fuses, and
    produces plain base-model output, with nothing anywhere saying why. The
    token is a property of the ADAPTER, so every client having to know it is a
    design that leaks - something2 would need a lookup table that goes stale the
    moment an adapter is retrained under a new trigger.
    """
    for part in (value or "").split("+"):
        part = part.strip()
        if not part.startswith(LOCAL_PREFIX):
            continue
        name = part[len(LOCAL_PREFIX):]
        for lora in trained_loras():
            if lora["name"] == name:
                return lora["trigger"]
        return f"<{name}-style>"
    return None


def apply_trigger(value: str, prompt: str) -> str:
    """Prepend the adapter's trigger to `prompt` unless it is already there.

    Idempotent on purpose. A caller that DOES know the trigger and sends it
    should not get it twice - a doubled token skews the conditioning toward it
    rather than strengthening it.
    """
    trigger = trigger_for(value)
    if not trigger:
        return prompt
    if trigger.lower() in (prompt or "").lower():
        return prompt
    return f"{trigger}, {prompt}" if prompt else trigger
