#!/usr/bin/env python3
"""Run the loop mastering over real takes and report what the seam measures.

    docker exec sprite_generator python /app/scripts/measure-audio-seams.py \
        /app/audio/spike --bpm-from-name

The smoke test's seam threshold is a bound on a synthetic fixture. This is the
same metric on actual model output, which is what the number should be set
from. Writes the mastered OGG beside each source so the seam can be heard.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, "/app")

import soundfile as sf  # noqa: E402

import audio_master as am  # noqa: E402
import audio_styles as st  # noqa: E402

# The spike wrote `spike_<style>_<seed>.wav`; bpm and signature come from the
# roster entry that named the style, which is where the take got them.
def _params_from_name(path: str):
    stem = os.path.basename(path)[len("spike_"):].rsplit("_", 1)[0]
    try:
        rendered = st.render(stem)
    except st.UnknownStyle:
        return None, None, stem
    return rendered["bpm"], rendered["time_signature"], stem


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("--min-s", type=float, default=60.0)
    ap.add_argument("--fade-ms", type=int, default=am.DEFAULT_FADE_MS)
    ap.add_argument("--out", default=None, help="default: <directory>/mastered")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(args.directory, "mastered")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for path in sorted(glob.glob(os.path.join(args.directory, "spike_*.wav"))):
        bpm, sig, stem = _params_from_name(path)
        samples, sr = sf.read(path, dtype="float32", always_2d=True)
        try:
            loop = am.make_loop(samples, sr, bpm, sig,
                                min_s=args.min_s if bpm else 20.0,
                                fade_ms=args.fade_ms)
        except am.AudioTooShort as e:
            print(f"  SKIP  {stem}: {e}")
            continue

        ogg = os.path.join(out_dir, os.path.basename(path).replace(".wav", ".ogg"))
        am.write_ogg(ogg, loop["samples"], sr, loop_start=loop["loop_start"],
                     loop_end=loop["loop_end"])
        decoded, rate = sf.read(ogg, dtype="float32", always_2d=True)

        row = {
            "take": stem, "bpm": bpm, "sig": sig, "sr": sr,
            "source_s": round(len(samples) / sr, 2),
            "loop_s": loop["duration_s"], "bars": loop["bars"],
            "seam_before_db": loop["seam_rms_jump_db_before"],
            "seam_after_db": loop["seam_rms_jump_db"],
            "seam_after_ogg_db": round(am.seam_rms_jump(decoded, rate), 2),
            "ogg": ogg,
        }
        rows.append(row)
        print(f"  {stem:28s} {row['source_s']:6.2f}s -> {row['loop_s']:6.2f}s "
              f"({row['bars']:3d} bars)  seam {row['seam_before_db']:6.2f} -> "
              f"{row['seam_after_db']:5.2f} dB (ogg {row['seam_after_ogg_db']:5.2f})")

    if not rows:
        print("no takes measured")
        return 1

    after = [r["seam_after_db"] for r in rows]
    before = [r["seam_before_db"] for r in rows]
    print(f"\n{len(rows)} takes | raw cut {min(before):.2f}-{max(before):.2f} dB "
          f"| mastered {min(after):.2f}-{max(after):.2f} dB "
          f"(mean {sum(after) / len(after):.2f})")
    with open(os.path.join(out_dir, "seams.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
