#!/usr/bin/env python3
"""Check the installed diffusers/transformers actually expose what qwen_edit.py needs.

Cheap, offline, and worth running BEFORE a 10 GB download finishes: if
QwenImageEditPlusPipeline is not in this diffusers build, the whole 2D route
needs a version bump and it is better to learn that now.
"""

import importlib
import sys

WANT_DIFFUSERS = [
    "QwenImageEditPlusPipeline",   # the 2511 pipeline (multi-image reference)
    "QwenImageEditPipeline",       # the original, acceptable fallback
    "QwenImageTransformer2DModel",  # what the GGUF is loaded into
    "GGUFQuantizationConfig",      # the whole reason this fits a 12 GB card
]

WANT_TRANSFORMERS = [
    "Qwen2_5_VLForConditionalGeneration",  # the text encoder class
    "BitsAndBytesConfig",                  # NF4 for that text encoder
]


def check(module_name, names):
    try:
        mod = importlib.import_module(module_name)
    except ImportError as e:
        print(f"{module_name}: NOT INSTALLED ({e})")
        return False
    print(f"{module_name} {getattr(mod, '__version__', '?')}")
    ok = True
    for n in names:
        have = hasattr(mod, n)
        print(f"  {'OK  ' if have else 'MISS'} {n}")
        ok = ok and have
    return ok


def main():
    d = check("diffusers", WANT_DIFFUSERS)
    print()
    t = check("transformers", WANT_TRANSFORMERS)
    print()

    try:
        import gguf
        print(f"gguf {getattr(gguf, '__version__', '?')}  OK")
    except ImportError:
        print("gguf: NOT INSTALLED - GGUFQuantizationConfig cannot read a file")
        d = False

    try:
        import bitsandbytes
        print(f"bitsandbytes {getattr(bitsandbytes, '__version__', '?')}  OK")
    except ImportError:
        print("bitsandbytes: NOT INSTALLED - the NF4 text encoder will not load")
        t = False

    # QwenImageEditPipeline alone is enough to test the angles LoRA; Plus only
    # adds multi-image reference. Report the distinction rather than failing.
    if not d:
        import diffusers
        if hasattr(diffusers, "QwenImageEditPipeline"):
            print("\nNOTE: QwenImageEditPlusPipeline is absent but "
                  "QwenImageEditPipeline is present. Set that class in "
                  "qwen_edit.load_pipeline; single-image editing (which is all "
                  "the turnaround needs) still works.")
        else:
            print("\nFAIL: this diffusers build has no Qwen-Image-Edit pipeline "
                  "at all. Upgrade: pip install -U 'diffusers @ "
                  "git+https://github.com/huggingface/diffusers'")

    print(f"\nRESULT: {'PASS' if (d and t) else 'FAIL'}")
    return 0 if (d and t) else 1


if __name__ == "__main__":
    sys.exit(main())
