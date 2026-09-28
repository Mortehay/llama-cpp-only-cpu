#!/usr/bin/env python3
"""Can an audio name write outside AUDIO_DIR?

WHY THIS EXISTS

`POST /api/audio` takes `name` from the caller, and `audio_master.audio_paths`
used to join it straight into the output path:

    os.path.join(kind_dir, f"{name}_{uid}")

"../../images/x" climbed out of the audio tree, and "/tmp/x" discarded it
entirely - os.path.join drops every part before an absolute one - so any
`generate`-scoped key could write a WAV and an OGG anywhere the worker can
write. Found by the 2026-09-28 branch review. The name is still the ledger's
lookup key; only the FILE stem is slugged now, and a containment check sits
behind the slug.

Pure: no server, no GPU, no database. Runs in any container with numpy.
"""

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
for cand in ("/app", os.path.join(HERE, "..", "src", "sprite_generator")):
    if os.path.isfile(os.path.join(cand, "audio_master.py")):
        sys.path.insert(0, cand)
        break

import audio_master as am  # noqa: E402

HOSTILE = ["../../images/x", "/tmp/evil", "..", "../..", "a/../../b",
           "dungeon/level1", "..\\..\\win", "\x00nul", "", "   ", ".hidden",
           "x" * 500]
ORDINARY = {"forest": "forest", "Cave 2": "Cave-2", "boss_theme.v2": "boss_theme.v2"}


def main():
    fails = []
    with tempfile.TemporaryDirectory() as root:
        for name in HOSTILE:
            try:
                wav, ogg = am.audio_paths(root, "music", name, "abc123")
                for p in (wav, ogg):
                    os.path.realpath(p)  # a null byte raises here
            except (ValueError, OSError) as e:
                fails.append(f"{name!r} raised {e}")
                continue
            for p in (wav, ogg):
                real = os.path.realpath(p)
                inside = os.path.commonpath([os.path.realpath(root), real]) == os.path.realpath(root)
                parent_ok = os.path.dirname(real) == os.path.realpath(os.path.join(root, "music"))
                if not (inside and parent_ok):
                    fails.append(f"{name!r} -> {p}")
            print(f"  {name[:30]!r:34} -> {os.path.basename(wav)}")
        for name, want in ORDINARY.items():
            wav, _ = am.audio_paths(root, "music", name, "u1")
            if os.path.basename(wav) != f"{want}_u1.wav":
                fails.append(f"ordinary {name!r} -> {os.path.basename(wav)}, want {want}_u1.wav")

    print("\nFAILURES:", "none" if not fails else "")
    for f in fails:
        print("  -", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
