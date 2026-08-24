#!/usr/bin/env python3
"""Print the real signatures qwen_edit.py has to call. No weights loaded.

Written because the alternative is guessing. This project's history is full of
code written against a plausible-sounding API that did not match the installed
one, and every instance cost a full download-and-run cycle to discover.

Specifically needed: whether the pipeline exposes `encode_prompt` separately
from `__call__`, and whether `__call__` accepts pre-computed `prompt_embeds`.
That is what decides if the three-pass batch design in ADR 0005 is possible -
encode every prompt, free the 5 GB text encoder, then denoise ~150 cells with
only the transformer resident. On a box with 10 GB of RAM that is not an
optimisation, it is the difference between running and swapping.
"""

import inspect
import sys


def show(obj, name):
    fn = getattr(obj, name, None)
    if fn is None:
        print(f"  {name}: ABSENT")
        return None
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as e:
        print(f"  {name}: present, signature unavailable ({e})")
        return fn
    params = list(sig.parameters)
    print(f"  {name}({', '.join(params)})")
    return fn


def main():
    from diffusers import QwenImageEditPlusPipeline, QwenImageEditPipeline
    import diffusers
    print(f"diffusers {diffusers.__version__}\n")

    for cls in (QwenImageEditPlusPipeline, QwenImageEditPipeline):
        print(f"{cls.__name__}:")
        show(cls, "encode_prompt")
        call = show(cls, "__call__")

        if call is not None:
            try:
                params = set(inspect.signature(call).parameters)
            except (TypeError, ValueError):
                params = set()
            for wanted in ("prompt_embeds", "prompt_embeds_mask",
                           "negative_prompt_embeds", "true_cfg_scale",
                           "num_inference_steps", "image", "generator"):
                mark = "OK  " if wanted in params else "MISS"
                print(f"    {mark} __call__ accepts {wanted}")
        print()

    # The expected components, so a missing text_encoder in the 4bit repo shows
    # up here rather than as a confusing load error.
    print("expected pipeline components:")
    for cls in (QwenImageEditPlusPipeline,):
        for attr in ("_optional_components", "model_cpu_offload_seq"):
            print(f"  {cls.__name__}.{attr} = {getattr(cls, attr, '(absent)')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
