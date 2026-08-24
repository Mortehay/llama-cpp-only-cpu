#!/usr/bin/env python3
"""Print the source of QwenImageEditPlusPipeline.encode_prompt.

The signature alone was not enough: encode_prompt returned None for the
embeddings, so what matters is which branch it takes and what it actually
returns. Reading it beats another guess-and-run cycle at ~4 minutes a go.
"""

import inspect
import sys

from diffusers import QwenImageEditPlusPipeline


def main():
    for name in ("encode_prompt", "_get_qwen_prompt_embeds"):
        fn = getattr(QwenImageEditPlusPipeline, name, None)
        if fn is None:
            print(f"### {name}: ABSENT\n")
            continue
        print(f"### {name}\n")
        try:
            print(inspect.getsource(fn))
        except (OSError, TypeError) as e:
            print(f"(source unavailable: {e})")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
