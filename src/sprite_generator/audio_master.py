"""Turn a generated take into a file that loops: bar-aligned cut, crossfade.

WHAT MAKES THIS TRACTABLE

ACE-Step is given `bpm` and `time_signature` and honours them (measured
2026-09-12: ten 120 s tracks, all exactly 120.0 s). So the loop cut is
ARITHMETIC on numbers we chose, not beat tracking on a signal we did not.
That removes the least reliable step from `.ai/decisions/0010`'s D1.

HOW THE LOOP IS MADE, because the obvious version is wrong

To loop `L` samples seamlessly we need `L + fade` samples of material. The
`fade` samples that follow the loop are the natural continuation of its end,
so they are blended DOWN over the loop's first `fade` samples while the head
is blended UP. Playback then runs ... loop[L-1] -> loop[0], and loop[0] is
still that continuation, so nothing jumps.

The tempting version - fade the tail out to silence and the head in from
silence - produces a hole at the seam that is audible as a dip, not a click.
Do not "simplify" to it.

No torch import: the API process calls the analysis half.
"""

from __future__ import annotations

import math
import os
import re

import numpy as np

# Equal-power default. Long enough to hide a seam in sustained material,
# short enough not to eat a bar at 140 bpm (one bar = 1.7 s).
DEFAULT_FADE_MS = 250

# The seam metric compares two windows AT the join. It must be much shorter
# than the fade: measure over the whole fade and you measure the fade itself.
SEAM_WINDOW_MS = 20


class AudioTooShort(ValueError):
    """The take cannot yield a whole-bar loop of the requested length."""


def bar_seconds(bpm: float, time_signature: str | int) -> float:
    """Seconds in one bar. `time_signature` is a beat count, as ACE-Step takes it."""
    beats = int(time_signature)
    if beats <= 0 or bpm <= 0:
        raise ValueError(f"bad bar: {beats} beats at {bpm} bpm")
    return beats * 60.0 / float(bpm)


def loop_length_samples(total_samples: int, sr: int, bpm: float,
                        time_signature: str | int, min_s: float,
                        fade_samples: int) -> tuple[int, int]:
    """Largest whole-bar loop that fits, leaving `fade_samples` to blend with.

    Returns `(loop_samples, bars)`. Raises `AudioTooShort` naming what was
    available, because "0 bars" downstream is unreadable.
    """
    bar = bar_seconds(bpm, time_signature)
    usable = total_samples - fade_samples
    bars = int(usable / (bar * sr))
    loop = int(round(bars * bar * sr))
    if bars < 1 or loop < int(min_s * sr):
        raise AudioTooShort(
            f"need {min_s:.1f}s of whole bars plus a {fade_samples / sr:.2f}s "
            f"crossfade, but the take is {total_samples / sr:.1f}s at {bpm} bpm "
            f"{time_signature}/4 ({bar:.2f}s per bar, {bars} usable bars)")
    return loop, bars


def crossfade_loop(samples: np.ndarray, loop_samples: int,
                   fade_samples: int) -> np.ndarray:
    """The first `loop_samples`, with the following `fade_samples` blended in."""
    if samples.ndim == 1:
        samples = samples[:, None]
    if loop_samples + fade_samples > len(samples):
        raise AudioTooShort(
            f"crossfade needs {loop_samples + fade_samples} samples, "
            f"have {len(samples)}")

    out = samples[:loop_samples].astype(np.float32, copy=True)
    tail = samples[loop_samples:loop_samples + fade_samples].astype(np.float32)

    t = np.linspace(0.0, 1.0, fade_samples, endpoint=False, dtype=np.float32)
    fall = np.cos(t * math.pi / 2.0)[:, None]
    rise = np.sin(t * math.pi / 2.0)[:, None]
    out[:fade_samples] = tail * fall + out[:fade_samples] * rise
    return out


def seam_rms_jump(loop: np.ndarray, sr: int,
                  window_ms: int = SEAM_WINDOW_MS) -> float:
    """dB between the loop's last window and its first. 0 means continuous.

    Absolute value: a jump up and a drop down are both audible.
    """
    if loop.ndim == 1:
        loop = loop[:, None]
    n = max(1, int(sr * window_ms / 1000))
    eps = 1e-9
    head = float(np.sqrt(np.mean(np.square(loop[:n], dtype=np.float64))))
    tail = float(np.sqrt(np.mean(np.square(loop[-n:], dtype=np.float64))))
    return abs(20.0 * math.log10((tail + eps) / (head + eps)))


def make_loop(samples: np.ndarray, sr: int, bpm: float | None,
              time_signature: str | int | None, min_s: float,
              fade_ms: int = DEFAULT_FADE_MS) -> dict:
    """Cut and crossfade one take. `bpm=None` means an unmetered texture.

    Ambience has no bars, so it keeps its full length minus the crossfade -
    the contract's `ambience` kind, where a stationary texture loops at any
    cut.
    """
    if samples.ndim == 1:
        samples = samples[:, None]
    fade = max(1, int(sr * fade_ms / 1000))

    if bpm:
        loop_samples, bars = loop_length_samples(len(samples), sr, bpm,
                                                 time_signature, min_s, fade)
    else:
        # No bars to round to, so honour the requested length exactly rather
        # than passing on the slack the take was padded with; anything spare
        # becomes crossfade material.
        loop_samples, bars = min(int(min_s * sr), len(samples) - fade), 0
        if loop_samples < 1 or len(samples) - fade < int(min_s * sr):
            raise AudioTooShort(
                f"texture is {len(samples) / sr:.1f}s, needs {min_s:.1f}s "
                f"plus a {fade / sr:.2f}s crossfade")

    before = seam_rms_jump(samples[:loop_samples], sr)
    loop = crossfade_loop(samples, loop_samples, fade)
    return {
        "samples": loop,
        "sample_rate": sr,
        "loop_start": 0,
        "loop_end": int(loop_samples),
        "bars": bars,
        "fade_samples": fade,
        "duration_s": round(loop_samples / sr, 3),
        "seam_rms_jump_db": round(seam_rms_jump(loop, sr), 2),
        "seam_rms_jump_db_before": round(before, 2),
    }


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
#
# `loop_start` is 0 and `loop_end` is the whole file in v1: the cut already
# made the file BE the loop. The tags are written anyway because the contract
# promises them and an engine that reads them needs no other configuration.

def write_wav(path: str, samples: np.ndarray, sr: int,
              subtype: str = "PCM_16") -> str:
    import soundfile as sf
    sf.write(path, np.clip(samples, -1.0, 1.0), sr, subtype=subtype)
    return path


# Frames per `SoundFile.write` call for OGG. See write_ogg: one large call
# segfaults libsndfile's Vorbis encoder. 64k frames is ~1.4 s at 48 kHz.
OGG_WRITE_BLOCK = 65536


def write_ogg(path: str, samples: np.ndarray, sr: int, *,
              loop_start: int = 0, loop_end: int | None = None,
              quality: float = 0.6, loop: bool = True) -> str:
    """OGG Vorbis with LOOPSTART / LOOPLENGTH in SAMPLES, as engines expect.

    `loop=False` writes no loop tags: a one-shot cue must not carry them, or
    an engine that honours LOOPSTART would play a sword swing forever.
    """
    import soundfile as sf
    from mutagen.oggvorbis import OggVorbis

    if loop_end is None:
        loop_end = len(samples)
    with sf.SoundFile(path, "w", samplerate=sr,
                      channels=1 if samples.ndim == 1 else samples.shape[1],
                      format="OGG", subtype="VORBIS") as fh:
        try:
            fh.set_quality(quality)
        except AttributeError:      # older soundfile: default quality
            pass
        # IN BLOCKS, never one call. libsndfile 1.2.2's Vorbis encoder (the
        # one bundled with soundfile 0.14.0) overflows its stack on a large
        # single write and SEGFAULTS the calling process - measured
        # 2026-09-28: 30 s of 48 kHz stereo wrote fine, 60 s and up died with
        # SIGSEGV in libsndfile_x86_64.so. In the worker that killed the whole
        # Celery process mid-task (exit 139), so no Python handler ever ran.
        # Block size does not change the encoded output.
        clipped = np.clip(samples, -1.0, 1.0)
        for i in range(0, len(clipped), OGG_WRITE_BLOCK):
            fh.write(clipped[i:i + OGG_WRITE_BLOCK])

    if not loop:
        return path
    tags = OggVorbis(path)
    tags["LOOPSTART"] = str(int(loop_start))
    tags["LOOPLENGTH"] = str(int(loop_end - loop_start))
    tags.save()
    return path


def read_loop_tags(path: str) -> tuple[int, int]:
    """(loop_start, loop_end) in samples, from the OGG's own tags."""
    from mutagen.oggvorbis import OggVorbis
    tags = OggVorbis(path)
    start = int(tags.get("LOOPSTART", ["0"])[0])
    length = int(tags.get("LOOPLENGTH", ["0"])[0])
    return start, start + length


# ---------------------------------------------------------------------------
# One-shot mastering (sfx)
# ---------------------------------------------------------------------------
#
# A cue is judged on its ONSET: a sound that starts 80 ms late feels like
# input lag. So the generated take is trimmed to its first audible sample
# (keeping a few ms so the attack is not clipped), cut after its last audible
# one plus a short release, faded out so the cut cannot click, and peak-
# normalised so every cue in a pack sits at the same level.

def master_one_shot(samples: np.ndarray, sr: int, *,
                    threshold_db: float = -40.0, preroll_ms: float = 5.0,
                    release_ms: float = 60.0, fade_ms: float = 25.0,
                    peak_db: float = -1.0) -> dict:
    """Trim, fade and normalise one cue. Raises ValueError on a silent take."""
    x = samples if samples.ndim == 2 else samples[:, None]
    env = np.max(np.abs(x), axis=1)
    peak = float(env.max()) if len(env) else 0.0
    if peak <= 1e-6:
        raise ValueError("the take is silent")
    above = np.flatnonzero(env >= peak * 10 ** (threshold_db / 20))
    start = max(int(above[0]) - int(sr * preroll_ms / 1000), 0)
    end = min(int(above[-1]) + int(sr * release_ms / 1000), len(x))
    cut = x[start:end].astype(np.float32).copy()

    fade = min(int(sr * fade_ms / 1000), len(cut))
    if fade > 1:
        cut[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)[:, None]
    cut *= (10 ** (peak_db / 20)) / float(np.max(np.abs(cut)))

    out = cut if samples.ndim == 2 else cut[:, 0]
    # onset_ms: where the sound starts IN THE OUTPUT (ticket 16 bar: <=10 ms).
    # trimmed_lead_ms: how much silence the model had put in front of it.
    return {"samples": out,
            "onset_ms": round((int(above[0]) - start) / sr * 1000, 1),
            "trimmed_lead_ms": round(start / sr * 1000, 1),
            "duration_s": round(len(cut) / sr, 3)}


def sfx_paths(audio_dir: str, engine: str, cue: str, entity: str | None,
              uid: str, variant: int) -> tuple[str, str]:
    """`(wav, ogg)` under `<AUDIO_DIR>/sfx/<engine>/<cue>/` for one variant.

    `engine` and `cue` are roster-validated before they get here; `entity` is
    caller text and goes through `safe_stem` - the same rule as `audio_paths`,
    so one function decides what a caller's words may become on disk, and its
    length cap stops a long entity failing the write after the GPU work.
    """
    d = os.path.join(audio_dir, "sfx", safe_stem(engine), safe_stem(cue))
    os.makedirs(d, exist_ok=True)
    ent = safe_stem((entity or "generic").strip().lower(), limit=60)
    stem = os.path.join(d, f"{ent}_{uid}_v{variant}")
    root = os.path.realpath(d)
    if os.path.commonpath([root, os.path.realpath(stem)]) != root:
        raise ValueError(f"sfx entity {entity!r} resolves outside {d}")
    return stem + ".wav", stem + ".ogg"


def audio_paths(audio_dir: str, kind: str, name: str, uid: str) -> tuple[str, str]:
    """`(wav, ogg)` under `<AUDIO_DIR>/<kind>/`, created if missing.

    Audio has its own tree, not IMAGES_DIR: it used to sit flat among the PNGs
    only because `generations._url` mapped every file to `/images/<basename>`.
    """
    kind_dir = os.path.join(audio_dir, kind)
    os.makedirs(kind_dir, exist_ok=True)
    stem = os.path.join(kind_dir, f"{safe_stem(name)}_{uid}")
    # Second barrier, in case safe_stem is ever loosened: the file must land
    # inside kind_dir, whatever the caller's name was.
    root = os.path.realpath(kind_dir)
    if os.path.commonpath([root, os.path.realpath(stem)]) != root:
        raise ValueError(f"audio name {name!r} resolves outside {kind_dir}")
    return stem + ".wav", stem + ".ogg"


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_stem(name: str, limit: int = 80) -> str:
    """A filename-safe form of a caller's audio name.

    `name` comes straight from `POST /api/audio`, and a raw join let
    "../../images/x" climb out of AUDIO_DIR and "/tmp/x" discard it entirely
    (os.path.join drops everything before an absolute part) - a write anywhere
    the worker can write, for any `generate`-scoped key. "dungeon/level1" also
    pointed at a subdirectory that did not exist, failing only after the GPU
    build. The name stays the lookup key in the ledger; only the FILE is
    slugged, and the uid suffix keeps two names that slug alike distinct.
    """
    stem = _UNSAFE.sub("-", name or "").strip(".-")[:limit]
    return stem or "audio"
