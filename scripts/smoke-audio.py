#!/usr/bin/env python3
"""Style roster and loop mastering under test. No model, no GPU, no network.

    docker exec sprite_generator python /app/scripts/smoke-audio.py

The case that carries the weight is the seam one. A crossfade that fades the
tail to silence instead of blending it into the head looks identical in code
review and leaves a dip at the loop point, so the synthetic signal is built to
catch exactly that: loud head, quiet tail, and a seam metric measured over a
window much SHORTER than the fade. Before the crossfade the jump is ~12 dB;
after it, the loop must open with the tail's own level.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile

sys.path.insert(0, "/app")

import numpy as np  # noqa: E402

import audio_master as am  # noqa: E402
import audio_styles as st  # noqa: E402

CASES = []


def case(name):
    def wrap(fn):
        CASES.append((name, fn))
        return fn
    return wrap


def _sine(sr, seconds, hz=220.0, amp=1.0, phase=0.0):
    t = np.arange(int(sr * seconds), dtype=np.float64) / sr
    wave = (amp * np.sin(2 * math.pi * hz * t + phase)).astype(np.float32)
    return np.stack([wave, wave], axis=1)


# ---------------------------------------------------------------------------
# Roster
# ---------------------------------------------------------------------------

@case("every style renders, and each kind refuses the other's failure mode")
def _renders():
    # Music leaks VOCALS, ambience leaks MUSIC. Each entry must push its own
    # failure away, so this asserts per kind rather than one rule for both.
    for entry in st.roster():
        out = st.render(entry["value"])
        assert "{" not in out["prompt"], f"{entry['value']}: unfilled slot"
        assert out["adjusted"] == [], out["adjusted"]
        # A negation in a POSITIVE prompt asks for the noun it names; the
        # exclusion belongs in `negative`. This used to assert the opposite.
        assert " no " not in f" {out['prompt']} ", \
            f"{entry['value']}: negation in the positive prompt"
        if entry["kind"] == "music":
            assert "instrumental" in out["prompt"], entry["value"]
            # The house style (owner, 2026-09-28): smooth and medieval. Turbo
            # has no CFG, so it has to be in the positive template.
            assert "smooth legato" in out["prompt"], entry["value"]
            assert "medieval" in out["prompt"], entry["value"]
            assert "vocals" in out["negative"], entry["value"]
            assert out["lyrics"] == "[Instrumental]", entry["value"]
        else:
            assert "soft and even" in out["prompt"], entry["value"]
            assert "music" in out["negative"], entry["value"]
            assert out["lyrics"] is None, entry["value"]
            assert out["bpm"] is None, entry["value"]
    music = st.roster("music")
    amb = st.roster("ambience")
    return f"{len(music)} music, {len(amb)} ambience"


@case("each kind has exactly one default")
def _default():
    for kind, expect in (("music", "medieval_fantasy"), ("ambience", "forest")):
        defaults = [s for s in st.roster(kind) if s["default"]]
        assert len(defaults) == 1, (kind, defaults)
        assert defaults[0]["value"] == expect, (kind, defaults[0]["value"])
        assert st.default_for(kind) == expect
    assert st.DEFAULT == "medieval_fantasy"
    return "music=medieval_fantasy, ambience=forest"


@case("slots clamp and report rather than raise")
def _slots():
    # Bounds read from the roster, not literals: the bands are tuned by ear
    # (narrowed 2026-09-28 for the smooth house style) and this tests clamping.
    tempo = {s["value"]: s["slots"]["tempo_bpm"] for s in st.roster("music")}
    out = st.render("battle", tempo_bpm=300, mood="jazzy", nonsense=1)
    assert out["bpm"] == tempo["battle"]["max"], out["bpm"]
    assert out["slots"]["mood"] == "heroic", out["slots"]
    assert len(out["adjusted"]) == 3, out["adjusted"]
    bad = st.render("dungeon", tempo_bpm="fast")
    assert bad["bpm"] == tempo["dungeon"]["default"] \
        and "not a number" in bad["adjusted"][0]
    try:
        st.render("sea_shanty")
    except st.UnknownStyle:
        pass
    else:
        raise AssertionError("an unknown style must not render")
    return "; ".join(out["adjusted"])[:90]


@case("keyword fallback maps six contexts without an LLM")
def _rules():
    want = {
        "abandoned dwarven mine, danger": "dungeon",
        "the crypt beneath the chapel": "dungeon",
        "siege of the north fortress": "battle",
        "the drunken boar inn": "tavern",
        "quiet farming hamlet": "village",
        "a green valley giving way to cold highlands": "medieval_fantasy",
    }
    for ctx, expect in want.items():
        got = st.style_from_context(ctx)
        assert got == expect, f"{ctx!r} -> {got}, wanted {expect}"
    return f"{len(want)} contexts"


@case("ambience keywords pick textures, and the author names the keyword")
def _rules_ambience():
    want = {
        "abandoned dwarven mine, danger": "cave",   # music says dungeon
        "the drunken boar inn": "village_day",
        "a storm over the moors": "rain",
        "graveyard at midnight": "night",
        "a green valley giving way to cold highlands": "forest",
        "an empty plain": "forest",                 # the default
    }
    for ctx, expect in want.items():
        style, slots, author = st._rules_style_plan(ctx, "ambience")
        assert style == expect, f"{ctx!r} -> {style}, wanted {expect}"
        assert slots == {} and "keyword rules" in author, author
    _, _, author = st._rules_style_plan("an empty plain", "ambience")
    assert "no keyword matched" in author, author
    return f"{len(want)} contexts"


@case("a brain's answer is validated: garbage, wrong kind and inventions are named")
def _llm_parse():
    ok = st._parse_llm_answer(
        'Sure! ```json\n{"style": "dungeon", "slots": {"mood": "lonely", '
        '"tempo_bpm": 999, "kazoo": "yes"}}\n```', "music")
    assert ok[0] == "dungeon", ok
    assert ok[1] == {"mood": "lonely", "tempo_bpm": 999}, ok  # render clamps
    assert any("kazoo" in n for n in ok[2]), ok
    assert st.render(ok[0], **ok[1])["bpm"] == 80, "tempo not clamped"
    for text, why in (("no json here", "no JSON"),
                      ('{"style": "sea_shanty"}', "invented"),
                      ('{"style": "cave"}', "wrong-kind"),
                      ('{"style": ', "no JSON"),
                      ("[1, 2]", "no JSON")):
        style, slots, notes = st._parse_llm_answer(text, "music")
        assert style is None and slots == {}, (text, style)
        assert any(why in n for n in notes), (text, notes)
    return "1 accepted, 5 refused with a reason"


# ---------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------

@case("bar arithmetic matches hand values")
def _bars():
    want = {(120, "4"): 2.0, (90, "3"): 2.0, (140, "6"): 60 * 6 / 140.0,
            (96, "4"): 2.5}
    for (bpm, sig), expect in want.items():
        got = am.bar_seconds(bpm, sig)
        assert abs(got - expect) < 1e-9, f"{bpm}/{sig}: {got} vs {expect}"
    return ", ".join(f"{b}bpm {s} -> {v:.3f}s" for (b, s), v in want.items())


@case("125s at 120bpm 4/4 cuts to 62 bars = 124.0s")
def _cut():
    sr = 48000
    fade = int(sr * am.DEFAULT_FADE_MS / 1000)
    loop, bars = am.loop_length_samples(int(125 * sr), sr, 120, "4", 60.0, fade)
    assert bars == 62, bars
    assert abs(loop / sr - 124.0) < 1e-6, loop / sr
    return f"{bars} bars, {loop / sr:.3f}s, {fade / sr:.3f}s fade spare"


@case("a take too short to hold the minimum is refused, with numbers")
def _too_short():
    sr = 48000
    try:
        am.loop_length_samples(int(30 * sr), sr, 96, "4", 60.0, 12000)
    except am.AudioTooShort as e:
        assert "30.0s" in str(e) and "60.0s" in str(e), str(e)
        return str(e)[:95]
    raise AssertionError("30s must not satisfy a 60s minimum")


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------

@case("crossfade removes the seam a raw cut leaves")
def _seam():
    # A 12 dB step between head and tail - far harsher than music, and the
    # two halves are the same tone, so they add coherently. The residual after
    # the fade is the HEAD entering the measurement window (20 ms of a 250 ms
    # fade admits sin(0.04pi) of it), not a discontinuity; hence a ratio
    # assertion plus a loose absolute bound rather than "near zero".
    sr = 48000
    loud = _sine(sr, 3.0, amp=1.0)
    quiet = _sine(sr, 5.0, amp=0.25, phase=math.pi / 3)
    take = np.concatenate([loud, quiet])

    loop_samples = int(4.0 * sr)
    fade = int(sr * am.DEFAULT_FADE_MS / 1000)

    before = am.seam_rms_jump(take[:loop_samples], sr)
    after = am.seam_rms_jump(am.crossfade_loop(take, loop_samples, fade), sr)
    assert before > 6.0, f"the fixture is not discontinuous: {before:.2f} dB"
    assert after < 2.0, f"crossfade left {after:.2f} dB at the seam"
    assert after < before / 4, f"{before:.1f} dB -> {after:.2f} dB is not a fix"
    return f"{before:.1f} dB -> {after:.2f} dB"


@case("the crossfade blends the tail in, it does not fade to silence")
def _not_silence():
    # Both halves identical, so the equal-power pair sums to up to sqrt(2) of
    # one of them: mid-fade RMS lands ABOVE the 0.707 of a unit sine. The
    # failure this guards is the opposite - a fade to silence dips toward 0.
    sr = 48000
    take = np.concatenate([_sine(sr, 1.0, amp=1.0), _sine(sr, 1.0, amp=1.0)])
    fade = int(sr * 0.25)
    loop = am.crossfade_loop(take, int(1.0 * sr), fade)
    mid = loop[fade // 2: fade // 2 + 512]
    rms = float(np.sqrt(np.mean(np.square(mid.astype(np.float64)))))
    assert rms > 0.5, f"the seam dips to {rms:.3f} - tail faded to silence"
    return f"mid-fade rms {rms:.3f}, above the 0.707 of one unit sine"


@case("make_loop reports bars, duration and both seam figures")
def _make_loop():
    sr = 48000
    take = np.concatenate([_sine(sr, 2.0, amp=1.0), _sine(sr, 4.0, amp=0.3)])
    out = am.make_loop(take, sr, bpm=120, time_signature="4", min_s=2.0)
    assert out["bars"] == 2 and abs(out["duration_s"] - 4.0) < 1e-3, out
    assert out["seam_rms_jump_db"] < out["seam_rms_jump_db_before"], out
    assert out["loop_start"] == 0 and out["loop_end"] == len(out["samples"])
    return (f"{out['bars']} bars, {out['duration_s']}s, "
            f"{out['seam_rms_jump_db_before']} -> {out['seam_rms_jump_db']} dB")


@case("an unmetered texture loops at exactly the length asked for")
def _ambience():
    # The take is padded so there is material to crossfade with; the loop must
    # still come out at the requested length, not the padded one.
    sr = 44100
    out = am.make_loop(_sine(sr, 36.0, hz=110, amp=0.5), sr, bpm=None,
                       time_signature=None, min_s=30.0)
    assert out["bars"] == 0, out["bars"]
    assert abs(out["duration_s"] - 30.0) < 0.01, out["duration_s"]
    short = _sine(sr, 20.0, hz=110, amp=0.5)
    try:
        am.make_loop(short, sr, bpm=None, time_signature=None, min_s=30.0)
    except am.AudioTooShort:
        pass
    else:
        raise AssertionError("a 20s take must not satisfy a 30s request")
    return f"{out['duration_s']}s from a 36s take, seam {out['seam_rms_jump_db']} dB"


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

@case("wav and ogg round-trip at the same length, tags match the cut")
def _files():
    sr = 48000
    out = am.make_loop(np.concatenate([_sine(sr, 2.0), _sine(sr, 3.0, amp=0.4)]),
                       sr, bpm=120, time_signature="4", min_s=2.0)
    import soundfile as sf
    with tempfile.TemporaryDirectory() as tmp:
        wav, ogg = am.audio_paths(tmp, "music", "smoke", "0001")
        am.write_wav(wav, out["samples"], sr)
        am.write_ogg(ogg, out["samples"], sr, loop_start=out["loop_start"],
                     loop_end=out["loop_end"])

        w = sf.info(wav)
        o = sf.info(ogg)
        assert w.samplerate == o.samplerate == sr, (w.samplerate, o.samplerate)
        assert w.channels == o.channels == 2, (w.channels, o.channels)
        assert abs(w.frames - out["loop_end"]) <= 1, (w.frames, out["loop_end"])
        assert abs(o.frames - w.frames) <= sr // 100, (o.frames, w.frames)

        start, end = am.read_loop_tags(ogg)
        assert (start, end) == (out["loop_start"], out["loop_end"]), (start, end)
        sizes = (os.path.getsize(wav), os.path.getsize(ogg))
        assert sizes[1] < sizes[0], sizes
    return (f"wav {sizes[0] // 1024} KiB, ogg {sizes[1] // 1024} KiB, "
            f"tags {start}..{end}")


@case("the ogg seam survives encoding")
def _ogg_seam():
    sr = 48000
    out = am.make_loop(np.concatenate([_sine(sr, 2.0, amp=1.0),
                                       _sine(sr, 3.0, amp=0.25,
                                             phase=math.pi / 3)]),
                       sr, bpm=120, time_signature="4", min_s=2.0)
    import soundfile as sf
    with tempfile.TemporaryDirectory() as tmp:
        _, ogg = am.audio_paths(tmp, "music", "seam", "0002")
        am.write_ogg(ogg, out["samples"], sr)
        decoded, rate = sf.read(ogg, dtype="float32", always_2d=True)
    jump = am.seam_rms_jump(decoded, rate)
    assert jump < 1.5, f"{jump:.2f} dB at the seam after vorbis"
    return f"{jump:.2f} dB, {len(decoded) / rate:.3f}s decoded"


# ---------------------------------------------------------------------------
# Sound effects (ticket 16)
# ---------------------------------------------------------------------------

@case("sfx engine precedence: request > world > cue > default, refusals named")
def _sfx_engine():
    assert st.resolve_engine("hit") == ("realistic", "cue")
    assert st.resolve_engine("hit", "realistic") == ("realistic", "request")
    assert st.resolve_engine("hit", None, "realistic") == ("realistic", "world")
    assert st.resolve_engine("hit", "retro") == ("retro", "request")
    assert st.resolve_engine("hit", None, "retro") == ("retro", "world")
    assert st.resolve_engine("hit", "realistic", "retro") == ("realistic", "request")
    # A level that names an engine the cue cannot render is REFUSED, never
    # passed down - a silent fall-through would mix looks within one game.
    # footstep has no retro recipe on purpose (ticket 17).
    for req, world, where in (("retro", None, "request"),
                              (None, "retro", "world"),
                              ("retro", "realistic", "request")):
        try:
            st.resolve_engine("footstep", req, world)
        except st.NoRecipe as e:
            assert where in str(e) and "offers realistic" in str(e), e
        else:
            raise AssertionError(f"retro footstep from {where} was not refused")
    try:
        st.resolve_engine("hit", "chiptune")
    except st.NoRecipe as e:
        assert "unknown engine" in str(e), e
    else:
        raise AssertionError("an unknown engine was accepted")
    try:
        st.resolve_engine("yodel")
    except st.UnknownStyle:
        pass
    else:
        raise AssertionError("an unknown cue was accepted")
    # Engine is part of the cache key; the entity is normalised.
    assert st.sfx_name("realistic", "hit", " Slime ") == "realistic:hit/slime"
    assert st.sfx_name("retro", "hit", None) == "retro:hit"
    r = st.render_cue("hit", "a slime", "realistic")
    assert "a slime" in r["prompt"] and " no " not in f" {r['prompt']} "
    assert "music" in r["negative"] and r["duration_s"] == 0.6
    assert "a creature" in st.render_cue("hit", None, "realistic")["prompt"]
    assert {c["value"] for c in st.cue_roster()} >= {"slash", "hit", "pickup"}
    return f"{len(st.CUES)} cues, 3 levels + 3 refusals"


@case("retro engine: deterministic, fast, every recipe audible, name-seeded")
def _retro():
    import time as _t
    import audio_retro as ar
    # The roster and the synth agree on which cues exist in 8-bit.
    assert set(st.RETRO_CUES) == set(ar.PRESETS), (st.RETRO_CUES, list(ar.PRESETS))
    assert "footstep" not in ar.PRESETS
    # Same cue + seed -> byte-identical samples.
    a = ar.synth(ar.draw("hit", 7), 7)
    b = ar.synth(ar.draw("hit", 7), 7)
    assert a.tobytes() == b.tobytes(), "retro is not deterministic"
    assert a.tobytes() != ar.synth(ar.draw("hit", 8), 8).tobytes()
    # The name gives the seed: stable, and different per entity.
    assert ar.seed_for("hit", "Slime ") == ar.seed_for("hit", "slime")
    assert ar.seed_for("hit", "slime") != ar.seed_for("hit", "knight")
    worst, lengths = 0.0, []
    with tempfile.TemporaryDirectory() as tmp:
        # Warm up first. A fresh process pays scipy's import on its first
        # filtered cue (measured ~0.9 s here, cold) - a once-per-process cost,
        # not a per-cue one, and it would make this timing flaky.
        t0 = _t.time()
        ar.build("hit", None, 1, 1, tmp)
        cold = _t.time() - t0
        for cue in ar.PRESETS:
            t0 = _t.time()
            res = ar.build(cue, "slime", None, 3, tmp)
            worst = max(worst, (_t.time() - t0) / 3)
            for v in res["variants"]:
                assert v["onset_ms"] <= 10, (cue, v["onset_ms"])
                assert os.path.exists(v["file_path"]), v["file_path"]
                lengths.append(v["duration_s"])
            assert res["engine"] == "retro" and res["seed"] == ar.seed_for(cue, "slime")
            assert [v["seed"] for v in res["variants"]] == [res["seed"] + k
                                                            for k in range(3)]
        # Rebuilding by name reproduces the same audio (the cache contract).
        again = ar.build("pickup", "slime", None, 1, tmp)["variants"][0]
        first = ar.build("pickup", "slime", None, 1, tmp)["variants"][0]
        import soundfile as sf
        assert (sf.read(again["master_path"])[0] == sf.read(first["master_path"])[0]).all()
    # Ticket 17's bar is "under 1 s"; a variant is far below it.
    assert worst < 0.5, f"{worst:.3f}s per variant"
    return (f"{len(ar.PRESETS)} cues x 3 variants, worst {worst * 1000:.0f} ms/variant "
            f"warm (first call {cold:.2f}s cold), "
            f"{min(lengths):.2f}-{max(lengths):.2f} s long")


@case("one-shot mastering: onset <=10 ms, faded tail, -1 dBFS, no loop tags")
def _one_shot():
    sr = 44100
    lead = np.zeros((int(0.3 * sr), 2), np.float32)          # 300 ms silence
    hit = 0.3 * _sine(sr, 0.4)                                # stereo already
    hit = (hit * np.linspace(1, 0.2, len(hit))[:, None]).astype(np.float32)
    tail = np.zeros((int(0.5 * sr), 2), np.float32)
    out = am.master_one_shot(np.concatenate([lead, hit, tail]), sr)
    x = out["samples"]
    assert out["onset_ms"] <= 10, out["onset_ms"]
    assert 290 <= out["trimmed_lead_ms"] <= 300, out["trimmed_lead_ms"]
    assert out["duration_s"] < 0.6, out["duration_s"]         # tail cut
    assert abs(20 * np.log10(np.max(np.abs(x))) + 1.0) < 0.05
    assert np.max(np.abs(x[-5:])) < 1e-3, "the end was not faded"
    try:
        am.master_one_shot(np.zeros((sr, 2), np.float32), sr)
    except ValueError:
        pass
    else:
        raise AssertionError("a silent take was accepted")
    from mutagen.oggvorbis import OggVorbis
    with tempfile.TemporaryDirectory() as tmp:
        wav, ogg = am.sfx_paths(tmp, "realistic", "hit", "Big Slime!", "ab12", 2)
        assert ogg.endswith(os.path.join("sfx", "realistic", "hit",
                                         "big-slime_ab12_v2.ogg")), ogg
        # The entity is caller text: it must never leave the cue directory,
        # and a long one must not fail the write after the GPU work.
        cue_dir = os.path.realpath(os.path.join(tmp, "sfx", "realistic", "hit"))
        for evil in ("../../../../tmp/x", "/etc/passwd", "a/b\\c", "x" * 500):
            _, p = am.sfx_paths(tmp, "realistic", "hit", evil, "ab12", 1)
            assert os.path.dirname(os.path.realpath(p)) == cue_dir, (evil, p)
            assert len(os.path.basename(p)) < 100, (evil, p)
        am.write_ogg(ogg, x, sr, loop=False)
        assert "LOOPSTART" not in OggVorbis(ogg), "a cue carries loop tags"
    return (f"lead {out['trimmed_lead_ms']} ms trimmed, onset "
            f"{out['onset_ms']} ms, {out['duration_s']} s")


@case("a full-length music loop writes to OGG without killing the process")
def _long_ogg():
    # 2026-09-28: a single 60 s+ Vorbis write segfaulted libsndfile and took
    # the Celery worker with it (exit 139). Run in a CHILD so a regression is
    # a FAIL line here rather than this smoke dying silently mid-run.
    import subprocess
    code = (
        "import sys, numpy as np; sys.path.insert(0, '/app'); "
        "import audio_master as am; "
        "x = (0.1 * np.random.default_rng(0).standard_normal((126 * 48000, 2)))"
        ".astype('float32'); "
        "am.write_ogg('/tmp/smoke-long.ogg', x, 48000, loop_end=len(x)); "
        "print(am.read_loop_tags('/tmp/smoke-long.ogg'))"
    )
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, (
        f"exit {proc.returncode} (139 = segfault): {proc.stderr[-300:]}")
    return f"126 s at 48 kHz stereo, tags {proc.stdout.strip()}"


def main() -> int:
    failed = 0
    for name, fn in CASES:
        try:
            print(f"  ok    {name}  ({fn()})")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}\n        {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}\n        {type(e).__name__}: {e}")
    print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
