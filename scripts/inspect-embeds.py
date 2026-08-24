#!/usr/bin/env python3
"""Report what an embeds_*.pt actually contains.

Written because the denoise stage warned that `prompt_embeds_mask` was absent,
which silently changes results: without the mask every padding token is treated
as real prompt content. The warning names the symptom, not which of encode,
save, load or the call site dropped it.

Usage:
    python inspect-embeds.py /app/images/_work/embeds_turn.pt
"""

import sys

import torch


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    blob = torch.load(argv[0], map_location="cpu")
    print(f"{argv[0]}: {len(blob)} entr{'y' if len(blob) == 1 else 'ies'}\n")
    missing = 0
    for key, value in blob.items():
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            print(f"  {key}: UNEXPECTED shape {type(value)}")
            missing += 1
            continue
        embeds, mask = value
        e = tuple(embeds.shape) if embeds is not None else None
        m = tuple(mask.shape) if mask is not None else None
        print(f"  {key:24} embeds={e}  mask={m}")
        if mask is None:
            missing += 1
    print()
    if missing:
        print(f"{missing} entr{'y' if missing == 1 else 'ies'} MISSING a mask")
        return 1
    print("every entry carries a mask")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
