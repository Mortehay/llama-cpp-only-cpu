"""The retro sfx engine: procedural 8-bit cues, sfxr-style. No GPU, no model.

Ticket 17 / decisions/0010 D7. Written here rather than vendored: it is small,
and a copied sfxr port would bring a licence question for ~150 lines.

WHY IT RUNS IN THE API PROCESS

A cue renders in milliseconds of numpy. Sent to the solo worker it would
queue behind whatever holds the card - a Qwen core is ~4 minutes - so a
"pickup" chime would take minutes to arrive. The facade renders retro cues
inline and only realistic ones go to the worker.

DETERMINISM IS THE CONTRACT

Each cue is a PRESET of parameter ranges; a seeded RNG picks values inside
them. Same cue + seed -> byte-identical samples (asserted by the smoke). With
no seed given, the seed is derived from the cue/entity name, so
`retro:hit/slime` is the same sound every time and differs from
`retro:hit/knight` - the entity has no other effect on an 8-bit blip.

Deliberately dependency-light: numpy and scipy, both already in both images.
"""

from __future__ import annotations

import hashlib
import time
import uuid

import numpy as np

import audio_master as am

SR = 44100

# Parameter RANGES per cue: (lo, hi) is drawn uniformly, a list is a choice,
# a scalar is fixed. Times in seconds; slide in octaves per second; filters
# are one-pole cutoffs as a fraction of Nyquist (1 = off).
PRESETS: dict[str, dict] = {
    # A coin/item chime: square blip that jumps up a fifth-to-octave.
    "pickup": {"wave": "square", "duty": (0.4, 0.5), "freq": (880, 1320),
               "slide": 0.0, "arp_at": (0.04, 0.07), "arp_mult": (1.5, 2.0),
               "attack": 0.0, "sustain": (0.05, 0.09), "decay": (0.12, 0.22),
               "punch": (0.3, 0.5), "lowpass": 1.0, "highpass": 0.0,
               "crush_bits": 6},
    # A blow: a noise burst dropping in pitch, darkened.
    "hit": {"wave": "noise", "freq": (900, 1800), "slide": (-3.5, -2.0),
            "attack": 0.0, "sustain": (0.02, 0.05), "decay": (0.10, 0.18),
            "punch": (0.4, 0.7), "lowpass": (0.35, 0.6), "highpass": 0.0,
            "crush_bits": 6},
    # A blade: bright noise swept down, thin (high-passed), quick.
    "slash": {"wave": "noise", "freq": (2600, 4200), "slide": (-4.0, -2.5),
              "attack": (0.005, 0.015), "sustain": (0.02, 0.04),
              "decay": (0.10, 0.16), "punch": 0.2, "lowpass": 1.0,
              "highpass": (0.08, 0.18), "crush_bits": 7},
    # Magic: a tone rising with vibrato, longer tail.
    "spell": {"wave": ["square", "sine"], "duty": (0.3, 0.5), "freq": (300, 480),
              "slide": (0.8, 1.6), "vib_depth": (0.03, 0.07),
              "vib_rate": (12.0, 20.0), "attack": (0.01, 0.03),
              "sustain": (0.22, 0.35), "decay": (0.25, 0.4), "punch": 0.1,
              "lowpass": (0.6, 0.9), "highpass": 0.0, "crush_bits": 6},
    # A menu click: a tiny, narrow square blip.
    "ui_click": {"wave": "square", "duty": (0.2, 0.3), "freq": (1500, 2400),
                 "slide": 0.0, "attack": 0.0, "sustain": (0.008, 0.015),
                 "decay": (0.02, 0.04), "punch": 0.3, "lowpass": 1.0,
                 "highpass": 0.0, "crush_bits": 5},
    # A whiff: thinner and quicker than a slash, no punch - nothing landed.
    "miss": {"wave": "noise", "freq": (3000, 4800), "slide": (-3.0, -1.5),
             "attack": (0.01, 0.02), "sustain": (0.01, 0.03),
             "decay": (0.07, 0.12), "punch": 0.0, "lowpass": 1.0,
             "highpass": (0.15, 0.25), "crush_bits": 7},
    # A chest: a low saw creak sliding up, then the classic rising reward
    # arpeggio on top.
    "chest_open": {"wave": "saw", "freq": (180, 260), "slide": (0.6, 1.1),
                   "arp_at": (0.12, 0.18), "arp_mult": (2.0, 3.0),
                   "attack": (0.01, 0.03), "sustain": (0.15, 0.25),
                   "decay": (0.2, 0.3), "punch": 0.2, "lowpass": (0.4, 0.6),
                   "highpass": 0.0, "crush_bits": 6},
    # A death: the long descending square every 8-bit game uses.
    "death": {"wave": "square", "duty": (0.4, 0.5), "freq": (500, 700),
              "slide": (-2.5, -1.6), "vib_depth": (0.02, 0.05),
              "vib_rate": (6.0, 10.0), "attack": 0.0, "sustain": (0.25, 0.4),
              "decay": (0.35, 0.5), "punch": 0.2, "lowpass": (0.5, 0.8),
              "highpass": 0.0, "crush_bits": 5},
    # A waypoint: a bright sine chime that jumps up an octave and rings.
    "waypoint": {"wave": "sine", "freq": (600, 800), "slide": (0.2, 0.5),
                 "arp_at": (0.08, 0.12), "arp_mult": 2.0,
                 "vib_depth": (0.01, 0.02), "vib_rate": (5.0, 8.0),
                 "attack": (0.005, 0.01), "sustain": (0.2, 0.3),
                 "decay": (0.4, 0.6), "punch": 0.3, "lowpass": 1.0,
                 "highpass": 0.0, "crush_bits": 7},
    # No "footstep": an 8-bit footstep is a weak blip. The cue has no retro
    # recipe, so asking for one is refused by resolve_engine (ticket 17).
}


def seed_for(cue: str, entity: str | None) -> int:
    """A stable seed from the name: same name, same sound, every time."""
    key = f"{cue}/{(entity or '').strip().lower()}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big") or 1


def draw(cue: str, seed: int) -> dict:
    """Concrete parameters for one variant, drawn from the cue's preset."""
    rng = np.random.default_rng(seed)
    out = {}
    for k, v in PRESETS[cue].items():
        if isinstance(v, tuple):
            out[k] = float(rng.uniform(v[0], v[1]))
        elif isinstance(v, list):
            out[k] = v[int(rng.integers(len(v)))]
        else:
            out[k] = v
    return out


def _one_pole_lowpass(x: np.ndarray, c: float) -> np.ndarray:
    if c >= 0.999:
        return x
    from scipy.signal import lfilter
    a = float(np.exp(-np.pi * max(c, 1e-3)))
    return lfilter([1 - a], [1, -a], x)


def _one_pole_highpass(x: np.ndarray, c: float) -> np.ndarray:
    if c <= 0.001:
        return x
    return x - _one_pole_lowpass(x, c)


def synth(p: dict, seed: int, sr: int = SR) -> np.ndarray:
    """Render drawn parameters to mono float32 in [-1, 1]."""
    rng = np.random.default_rng(seed ^ 0x5F3759DF)
    length = p["attack"] + p["sustain"] + p["decay"]
    n = max(int(length * sr), 1)
    t = np.arange(n) / sr

    freq = p["freq"] * np.exp2(p.get("slide", 0.0) * t)
    if p.get("arp_mult") and p.get("arp_at"):
        freq = np.where(t >= p["arp_at"], freq * p["arp_mult"], freq)
    if p.get("vib_depth"):
        freq = freq * (1 + p["vib_depth"] * np.sin(2 * np.pi * p["vib_rate"] * t))
    freq = np.clip(freq, 20.0, sr / 2 - 1)
    cycles = np.cumsum(freq / sr)
    frac = cycles % 1.0

    wave = p["wave"]
    if wave == "square":
        x = np.where(frac < p.get("duty", 0.5), 1.0, -1.0)
    elif wave == "saw":
        x = 2.0 * frac - 1.0
    elif wave == "sine":
        x = np.sin(2 * np.pi * frac)
    else:
        # sfxr noise: a new random value once per period, held - which is
        # what makes it pitched and gritty rather than hiss.
        table = rng.uniform(-1.0, 1.0, int(cycles[-1]) + 2)
        x = table[cycles.astype(np.int64)]

    env = np.ones(n)
    a = int(p["attack"] * sr)
    s = int(p["sustain"] * sr)
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a)
    punch = np.ones(n)
    if p.get("punch"):
        punch[a:a + s] += p["punch"] * np.linspace(1.0, 0.0, max(s, 1))[:len(punch[a:a + s])]
    d = n - a - s
    if d > 0:
        env[a + s:] = np.linspace(1.0, 0.0, d) ** 2
    x = x * env * punch

    x = _one_pole_lowpass(x, p.get("lowpass", 1.0))
    x = _one_pole_highpass(x, p.get("highpass", 0.0))
    bits = int(p.get("crush_bits", 8))
    levels = 2 ** (bits - 1)
    peak = float(np.max(np.abs(x))) or 1.0
    x = np.round(x / peak * levels) / levels
    return x.astype(np.float32)


def build(cue: str, entity: str | None, seed: int | None, variants: int,
          audio_dir: str) -> dict:
    """Render, master and write 1-5 variants. Same result shape as
    `audio_engine.generate_sfx` items, so one ledger path serves both."""
    if cue not in PRESETS:
        raise ValueError(f"cue {cue!r} has no retro recipe")
    t0 = time.time()
    base = int(seed) if seed else seed_for(cue, entity)
    uid = uuid.uuid4().hex[:8]
    out = []
    for v in range(max(1, min(5, int(variants or 1)))):
        s = base + v
        params = draw(cue, s)
        shot = am.master_one_shot(synth(params, s), SR)
        wav, ogg = am.sfx_paths(audio_dir, "retro", cue, entity, uid, v + 1)
        am.write_wav(wav, shot["samples"], SR)
        am.write_ogg(ogg, shot["samples"], SR, loop=False)
        out.append({"file_path": ogg, "master_path": wav, "seed": s,
                    "duration_s": shot["duration_s"],
                    "onset_ms": shot["onset_ms"],
                    "trimmed_lead_ms": shot["trimmed_lead_ms"],
                    "params": {k: (round(val, 4) if isinstance(val, float) else val)
                               for k, val in params.items()}})
    return {"cue": cue, "entity": (entity or "").strip() or None,
            "engine": "retro",
            "prompt": f"retro 8-bit {cue}"
                      + (f" ({entity.strip()})" if entity and entity.strip() else ""),
            "negative": "", "seed": base, "sample_rate": SR,
            "variants": out, "model_seconds": round(time.time() - t0, 3),
            "load_seconds": 0.0}
