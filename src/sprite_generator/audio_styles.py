"""The audio style roster: one list the tab, the facade and the LLM all read.

WHY A ROSTER AND NOT A PROMPT STRING

A caller that sends free text to a music model gets a different track for the
same map on every re-roll, so no two attempts can be compared and "medieval
fantasy by default" has nowhere to live but a system prompt. A roster entry is
a template plus the slots that vary, so `medieval_fantasy` means one specific
prompt today and the same one tomorrow. The LLM SELECTS an entry and fills its
slots; it never writes the prompt. See `.ai/specs/audio/contract.md`.

Same shape as `core_models.CORE_MODELS` - `value`/`label`/`default` - because
the UI already knows how to render that.

Deliberately dependency-free: the API process imports it, and the worker's
subprocess wrapper does too.

HOUSE STYLE: SMOOTH AND MEDIEVAL (owner's call, 2026-09-28, after listening)

Every entry, including battle and dungeon, is period acoustic instruments
played smoothly - no modern kit, no brass section, no taiko, no piano, no
accordion or guitar. Moods and tempo bands were narrowed to match, so no slot
value can steer an entry back out of the house style.

WHERE "SMOOTH" HAS TO LIVE

The music model is ACE-Step turbo, which has NO classifier-free guidance:
`scripts/acestep-generate.py` sends the caption only, at guidance 1.0, and the
`negative` below never reaches it. So smoothness is carried by the POSITIVE
template, the tempo band and the instrument choice. The music negatives are
kept for the sft model (which has CFG) and as documentation of intent.

Positive prompts never contain a negation ("no music", "no drums"). A text
encoder has no negation operator, so each noun lands in the conditioning and
asks for the thing it names - the same trap `split_negations()` exists for on
the image side (CLAUDE.md). Exclusions go in `negative` only.
"""

from __future__ import annotations

import os
from typing import Any

# ACE-Step takes the time signature as a beat count ("2", "3", "4", "6"), so
# it is stored per style rather than offered as a slot: a waltz is a different
# style, not a knob on a march.
MUSIC_KIND = "music"
AMBIENCE_KIND = "ambience"

# Every music template ends with these. A CLIP-style negation ("no vocals") in
# a POSITIVE prompt is unreliable, so instrumental is requested three ways:
# here, in the negative list, and by `lyrics="[Instrumental]"` plus
# `instrumental=True` on the ACE-Step call itself.
_NO_VOCALS = "vocals, singing, choir, spoken word, rap"

# What would break the house style. Effective only where CFG exists (sft).
_NOT_MEDIEVAL = ("modern drum kit, electronic, synthesizer, electric guitar, "
                 "distortion, rock, EDM, brass section, piano, harsh, "
                 "aggressive, loud")

# Appended to every music template: the positive half of the house style.
_SMOOTH = ("early music, acoustic period instruments, smooth legato, "
           "gentle dynamics, warm and soft, flowing, instrumental")

# The mirror of the above for ambience: the failure there is the model
# drifting into a tune under a texture that should just be wind and birds.
# Stable Audio Open DOES honour `negative_prompt`, so this is where the
# exclusions work. Modern sounds are listed too: a car in a medieval forest
# breaks the setting worse than a melody would.
_NO_MUSIC = ("music, melody, rhythm, instruments, singing, drums, harmony, "
             "sudden loud noises, engine, traffic, car, airplane, modern")

# Appended to every ambience template: steady, even, nothing that makes a
# listener flinch on the hundredth loop.
_SMOOTH_TEXTURE = "soft and even, gentle, continuous, field recording"

DEFAULT = "medieval_fantasy"

STYLES: list[dict[str, Any]] = [
    {
        "value": "medieval_fantasy",
        "kind": MUSIC_KIND,
        "label": "Medieval fantasy - overworld theme, the default",
        "default": True,
        "template": ("medieval fantasy RPG overworld theme, {featured}, "
                     "harp, soft hand drum, {mood}, " + _SMOOTH),
        "negative": _NO_VOCALS + ", " + _NOT_MEDIEVAL,
        "time_signature": "4",
        "slots": {
            "tempo_bpm": {"type": "int", "min": 70, "max": 104, "default": 84},
            "mood": {"type": "enum", "default": "warm and adventurous",
                     "values": ["warm and adventurous", "wistful", "serene",
                                "solemn", "mysterious"]},
            "featured": {"type": "enum", "default": "lute and wooden flute",
                         "values": ["lute and wooden flute", "solo harp",
                                    "recorder and fiddle", "hurdy-gurdy",
                                    "low whistle"]},
        },
    },
    {
        "value": "tavern",
        "kind": MUSIC_KIND,
        "label": "Tavern - cosy lilting inn tune",
        "template": ("cosy medieval tavern tune, {featured}, lilting dance, "
                     "light hand percussion, {mood}, " + _SMOOTH),
        "negative": _NO_VOCALS + ", " + _NOT_MEDIEVAL + ", accordion",
        "time_signature": "3",
        "slots": {
            "tempo_bpm": {"type": "int", "min": 88, "max": 116, "default": 100},
            "mood": {"type": "enum", "default": "cheerful",
                     "values": ["cheerful", "cosy and warm", "gently swaying"]},
            "featured": {"type": "enum", "default": "fiddle and hurdy-gurdy",
                         "values": ["fiddle and hurdy-gurdy",
                                    "lute and recorder",
                                    "citole and fiddle"]},
        },
    },
    {
        "value": "dungeon",
        "kind": MUSIC_KIND,
        "label": "Dungeon - slow, dim, sparse",
        "template": ("dim medieval dungeon, slow sustained drone, {featured}, "
                     "{mood}, sparse and spacious, " + _SMOOTH),
        "negative": _NO_VOCALS + ", " + _NOT_MEDIEVAL + ", upbeat, dance rhythm",
        "time_signature": "4",
        "slots": {
            "tempo_bpm": {"type": "int", "min": 50, "max": 80, "default": 64},
            "mood": {"type": "enum", "default": "mysterious",
                     "values": ["mysterious", "lonely", "somber"]},
            "featured": {"type": "enum",
                         "default": "distant bells and viola da gamba",
                         "values": ["distant bells and viola da gamba",
                                    "low bowed strings",
                                    "slow harp over drone"]},
        },
    },
    {
        "value": "battle",
        "kind": MUSIC_KIND,
        "label": "Battle - steady, noble march",
        "template": ("noble medieval march, steady frame drums, {featured}, "
                     "{mood}, rolling rhythm, " + _SMOOTH),
        "negative": _NO_VOCALS + ", " + _NOT_MEDIEVAL + ", taiko, frantic",
        "time_signature": "4",
        "slots": {
            "tempo_bpm": {"type": "int", "min": 96, "max": 126, "default": 110},
            "mood": {"type": "enum", "default": "heroic",
                     "values": ["heroic", "determined", "noble"]},
            "featured": {"type": "enum",
                         "default": "natural horn and low fiddles",
                         "values": ["natural horn and low fiddles",
                                    "hurdy-gurdy and harp",
                                    "fiddle ostinato"]},
        },
    },
    {
        "value": "village",
        "kind": MUSIC_KIND,
        "label": "Village - pastoral, calm",
        "template": ("peaceful medieval village, {featured}, {mood}, "
                     "pastoral, " + _SMOOTH),
        "negative": _NO_VOCALS + ", " + _NOT_MEDIEVAL + ", tension, dissonance",
        "time_signature": "3",
        "slots": {
            "tempo_bpm": {"type": "int", "min": 70, "max": 100, "default": 84},
            "mood": {"type": "enum", "default": "calm and sunny",
                     "values": ["calm and sunny", "sleepy", "hopeful"]},
            "featured": {"type": "enum", "default": "lute and recorder",
                         "values": ["lute and recorder",
                                    "hammered dulcimer and fiddle",
                                    "solo wooden flute"]},
        },
    },
    # --- ambience ------------------------------------------------------
    #
    # Stable Audio Open's own card says it is better at "sound effects and
    # field recordings than music", which is this job exactly - so every
    # ambience negative pushes MUSIC away, the mirror of what the music
    # entries do with vocals. No tempo slot: a texture has no metre, and
    # `make_loop` cuts these on the crossfade alone.
    {
        "value": "forest",
        "kind": AMBIENCE_KIND,
        "label": "Forest - wind, leaves, birds",
        "default": True,
        "template": "medieval forest ambience, {featured}, {mood}, "
                    + _SMOOTH_TEXTURE,
        "negative": _NO_MUSIC,
        "time_signature": None,
        "slots": {
            "mood": {"type": "enum", "default": "calm daytime",
                     "values": ["calm daytime", "dawn chorus", "light breeze",
                                "after rain"]},
            "featured": {"type": "enum", "default": "wind in leaves and distant birds",
                         "values": ["wind in leaves and distant birds",
                                    "creaking branches", "a stream nearby",
                                    "insects and rustling"]},
        },
    },
    {
        "value": "cave",
        "kind": AMBIENCE_KIND,
        "label": "Cave - drips, echo, airflow",
        "template": "stone cave ambience, {featured}, {mood}, soft echo, "
                    + _SMOOTH_TEXTURE,
        "negative": _NO_MUSIC + ", rockfall, roar",
        "time_signature": None,
        "slots": {
            "mood": {"type": "enum", "default": "damp and still",
                     "values": ["damp and still", "vast and hollow",
                                "quiet and close"]},
            "featured": {"type": "enum", "default": "dripping water and faint airflow",
                         "values": ["dripping water and faint airflow",
                                    "distant low rumble", "shifting gravel"]},
        },
    },
    {
        "value": "village_day",
        "kind": AMBIENCE_KIND,
        "label": "Village by day - crowd, carts, work",
        "template": "medieval village square ambience, {featured}, {mood}, "
                    + _SMOOTH_TEXTURE,
        "negative": _NO_MUSIC + ", intelligible speech, shouting",
        "time_signature": None,
        "slots": {
            "mood": {"type": "enum", "default": "busy but unhurried",
                     "values": ["busy but unhurried", "sparse and early",
                                "market day"]},
            "featured": {"type": "enum",
                         "default": "distant crowd murmur and footsteps on stone",
                         "values": ["distant crowd murmur and footsteps on stone",
                                    "wooden cart wheels and livestock",
                                    "distant smithy hammering"]},
        },
    },
    {
        "value": "night",
        "kind": AMBIENCE_KIND,
        "label": "Night - crickets, owls, low wind",
        "template": "medieval countryside at night, {featured}, {mood}, "
                    + _SMOOTH_TEXTURE,
        "negative": _NO_MUSIC,
        "time_signature": None,
        "slots": {
            "mood": {"type": "enum", "default": "quiet and open",
                     "values": ["quiet and open", "hushed", "cold and clear"]},
            "featured": {"type": "enum", "default": "crickets and a distant owl",
                         "values": ["crickets and a distant owl",
                                    "low wind and rustling grass",
                                    "frogs by water"]},
        },
    },
    {
        "value": "rain",
        "kind": AMBIENCE_KIND,
        "label": "Rain - steady, on leaves or stone",
        "template": "steady rain ambience, {featured}, {mood}, "
                    + _SMOOTH_TEXTURE,
        "negative": _NO_MUSIC + ", thunder claps",
        "time_signature": None,
        "slots": {
            "mood": {"type": "enum", "default": "steady and even",
                     "values": ["steady and even", "soft rainfall",
                                "light drizzle"]},
            "featured": {"type": "enum", "default": "rain on leaves",
                         "values": ["rain on leaves", "rain on stone and puddles",
                                    "rain on a thatched roof"]},
        },
    },
]

# ---------------------------------------------------------------------------
# Sound effects: CUES, not styles (0010 D7/D8, tickets 16/17)
# ---------------------------------------------------------------------------
#
# A cue is one game event - a slash, a hit, a pickup - played ONCE, 0.1-3 s,
# never looped. Deliberately NOT called an "action": that word already means a
# sprite-sheet row (domain.md). Kept apart from STYLES because a cue has no
# slots and no metre, and `render` / `roster` must not start meeting entries
# they cannot render.
#
# Addressed as `<cue>/<entity>` (owner, 2026-09-28): `hit/slime`,
# `slash/knight`. The entity is optional and folds into the prompt; a bare
# `hit` is a generic one.
#
# Each cue carries one RECIPE PER ENGINE. `realistic` is Stable Audio Open 1.0
# (the ambience model, already proven here); `retro` is procedural 8-bit
# synthesis and arrives with ticket 17 - until then no cue has a retro recipe,
# so asking for one is refused by `resolve_engine`, never faked.

SFX_KIND = "sfx"
ENGINES = ("realistic", "retro")
DEFAULT_ENGINE = "realistic"

# Every realistic cue excludes the ambience failure (a tune under the sound)
# and the setting-breakers, plus room tone - a one-shot wants a dry, close
# sound the game can place, not a recording of a hall.
_SFX_NEGATIVE = ("music, melody, singing, speech, voice, background noise, "
                 "room ambience, reverb tail, modern, electronic, low quality")
_SFX_SUFFIX = "single isolated sound effect, close and dry, clean, game audio"

CUES: list[dict[str, Any]] = [
    {"value": "slash", "label": "Slash - a blade cutting air",
     "default_engine": "realistic", "duration_s": 0.8,
     "entity_default": "a steel sword",
     "recipes": {"realistic": {"template":
         "{entity} swung in a fast slash through the air, sharp whoosh, "
         + _SFX_SUFFIX}}},
    {"value": "hit", "label": "Hit - a blow landing",
     "default_engine": "realistic", "duration_s": 0.6,
     "entity_default": "a creature",
     "recipes": {"realistic": {"template":
         "a heavy blow landing on {entity}, short punchy impact, "
         + _SFX_SUFFIX}}},
    {"value": "pickup", "label": "Pickup - collecting an item",
     "default_engine": "realistic", "duration_s": 0.5,
     "entity_default": "a gold coin",
     "recipes": {"realistic": {"template":
         "picking up {entity}, bright short chime of metal, "
         + _SFX_SUFFIX}}},
    {"value": "spell", "label": "Spell - casting magic",
     "default_engine": "realistic", "duration_s": 1.5,
     "entity_default": "a fire spell",
     "recipes": {"realistic": {"template":
         "casting {entity}, magical shimmering swell and release, fantasy, "
         + _SFX_SUFFIX}}},
    {"value": "footstep", "label": "Footstep - one step",
     "default_engine": "realistic", "duration_s": 0.4,
     "entity_default": "a leather boot on stone",
     "recipes": {"realistic": {"template":
         "one single footstep, {entity}, " + _SFX_SUFFIX}}},
    {"value": "ui_click", "label": "UI click - a menu button",
     "default_engine": "realistic", "duration_s": 0.2,
     "entity_default": "a wooden button",
     "recipes": {"realistic": {"template":
         "a soft short click of {entity}, crisp and subtle, "
         + _SFX_SUFFIX}}},
]


class NoRecipe(ValueError):
    """The chosen engine has no recipe for this cue - refused, not faked."""


def _cue(cue: str) -> dict[str, Any]:
    for c in CUES:
        if c["value"] == cue:
            return c
    raise UnknownStyle(cue)


def cue_roster() -> list[dict[str, Any]]:
    """The cues as `GET /api/audio/styles?kind=sfx` returns them."""
    return [{"value": c["value"], "label": c["label"], "kind": SFX_KIND,
             "default": c["value"] == CUES[0]["value"],
             "default_engine": c["default_engine"],
             "engines": sorted(c["recipes"]), "duration_s": c["duration_s"],
             "entity_default": c["entity_default"]} for c in CUES]


def resolve_engine(cue: str, requested: str | None = None,
                   world_engine: str | None = None) -> tuple[str, str]:
    """(engine, engine_from) by precedence: request > world > cue > default.

    Most specific wins (0010 D8). A level that names an engine the cue has no
    recipe for RAISES `NoRecipe` rather than falling through: a silent
    fall-through is how one game ends up with a mix of looks, which is what
    the world level exists to prevent.
    """
    entry = _cue(cue)
    for value, source in ((requested, "request"), (world_engine, "world"),
                          (entry["default_engine"], "cue"),
                          (DEFAULT_ENGINE, "default")):
        if not value:
            continue
        if value not in ENGINES:
            raise NoRecipe(f"unknown engine {value!r} (from {source}); "
                           f"expected one of {', '.join(ENGINES)}")
        if value not in entry["recipes"]:
            later = " - the retro engine is ticket 17" if value == "retro" else ""
            raise NoRecipe(f"cue {cue!r} has no {value!r} recipe (engine from "
                           f"{source}){later}")
        return value, source
    raise NoRecipe(f"no engine resolved for {cue!r}")  # unreachable: default


def sfx_name(engine: str, cue: str, entity: str | None) -> str:
    """The ledger name: engine is part of the cache key (0010 D8)."""
    ent = (entity or "").strip().lower()
    return f"{engine}:{cue}/{ent}" if ent else f"{engine}:{cue}"


def render_cue(cue: str, entity: str | None, engine: str) -> dict[str, Any]:
    entry = _cue(cue)
    recipe = entry["recipes"].get(engine)
    if not recipe:
        raise NoRecipe(f"cue {cue!r} has no {engine!r} recipe")
    subject = (entity or "").strip() or entry["entity_default"]
    return {"cue": cue, "entity": (entity or "").strip() or None,
            "engine": engine, "duration_s": entry["duration_s"],
            "prompt": recipe["template"].format(entity=subject),
            "negative": recipe.get("negative", _SFX_NEGATIVE)}


# Keyword -> style, for `_rules_style_plan` in ticket 08 and for anyone who
# wants a style from a map name without waking the LLM. First match wins, so
# order matters: "mine" before "village" because "mining village" is a dungeon
# with houses attached.
#
# Music and ambience share the table; `style_from_context` skips entries of
# the other kind, so "mine" is `dungeon` for music and `cave` for ambience.
RULES: list[tuple[tuple[str, ...], str]] = [
    (("dungeon", "cave", "crypt", "mine", "tomb", "catacomb", "lair"), "dungeon"),
    (("battle", "war", "siege", "arena", "boss", "fortress"), "battle"),
    (("tavern", "inn", "alehouse", "pub", "feast"), "tavern"),
    (("village", "town", "hamlet", "farm", "market", "square"), "village"),
    (("cave", "mine", "cavern", "dungeon", "crypt", "tomb", "catacomb",
      "underground", "grotto"), "cave"),
    (("rain", "storm", "drizzle", "monsoon", "wet"), "rain"),
    (("night", "moon", "dusk", "midnight", "graveyard"), "night"),
    (("village", "town", "hamlet", "market", "square", "city", "inn",
      "tavern"), "village_day"),
    (("forest", "wood", "grove", "glade", "valley", "meadow", "field"), "forest"),
]


class UnknownStyle(KeyError):
    """Asked for a roster entry that does not exist."""


def _entry(style: str) -> dict[str, Any]:
    for s in STYLES:
        if s["value"] == style:
            return s
    raise UnknownStyle(style)


def roster(kind: str | None = None) -> list[dict[str, Any]]:
    """The roster as the UI and `GET /api/audio/styles` want it."""
    return [
        {
            "value": s["value"],
            "label": s["label"],
            "kind": s["kind"],
            "default": bool(s.get("default")),
            "time_signature": s["time_signature"],
            "slots": s["slots"],
        }
        for s in STYLES
        if kind is None or s["kind"] == kind
    ]


def default_for(kind: str = MUSIC_KIND) -> str:
    for s in STYLES:
        if s["kind"] == kind and s.get("default"):
            return s["value"]
    for s in STYLES:
        if s["kind"] == kind:
            return s["value"]
    raise UnknownStyle(kind)


def style_from_context(context: str, kind: str = MUSIC_KIND) -> str:
    """Keyword fallback for when the LLM is asleep, wrong, or not wanted."""
    text = (context or "").lower()
    for words, style in RULES:
        if any(w in text for w in words):
            entry = _entry(style)
            if entry["kind"] == kind:
                return style
    return default_for(kind)


# ---------------------------------------------------------------------------
# Choosing a style from a map description (ticket 08)
# ---------------------------------------------------------------------------
#
# The brain SELECTS a roster entry and fills its slots; it never writes the
# model prompt - `render` does, from the template. Everything it returns is
# validated against the roster, anything invented is dropped AND NAMED in the
# author note, and any failure falls back to the keyword rules. The
# `worlds._llm_biome_plan` pattern, with a different vocabulary.

def _rules_style_plan(context: str, kind: str = MUSIC_KIND
                      ) -> tuple[str, dict[str, Any], str]:
    text = (context or "").lower()
    for words, style in RULES:
        if _entry(style)["kind"] != kind:
            continue
        hit = next((w for w in words if w in text), None)
        if hit:
            return style, {}, f"style chosen by keyword rules ({hit!r})"
    return (default_for(kind), {},
            "style chosen by keyword rules (no keyword matched; the default)")


def _parse_llm_answer(text: str, kind: str
                      ) -> tuple[str | None, dict[str, Any], list[str]]:
    """(style, slots, notes) from a brain's reply, or (None, {}, notes).

    Pure, so the smoke can feed it garbage. A 3B model wraps JSON in prose,
    code fences, or answers with a style from the wrong kind or one that does
    not exist; all of that is survivable. Slots are passed through for
    `render` to clamp - it names every correction itself.
    """
    import json
    import re

    notes: list[str] = []
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None, {}, ["answer held no JSON object"]
    try:
        obj = json.loads(m.group(0))
    except ValueError:
        return None, {}, ["answer's JSON could not be read"]
    if not isinstance(obj, dict):
        return None, {}, ["answer was not a JSON object"]

    style = str(obj.get("style") or "").strip()
    valid = {s["value"] for s in STYLES if s["kind"] == kind}
    if style not in valid:
        other = {s["value"] for s in STYLES}
        notes.append(f"dropped {'wrong-kind' if style in other else 'invented'} "
                     f"style {style!r}" if style else "no style named")
        return None, {}, notes

    raw = obj.get("slots") if isinstance(obj.get("slots"), dict) else {}
    known = _entry(style)["slots"]
    slots = {k: v for k, v in raw.items() if k in known}
    dropped = sorted(set(raw) - set(known))
    if dropped:
        notes.append(f"dropped invented slot(s): {', '.join(dropped)}")
    return style, slots, notes


def _llm_model(base: str) -> str | None:
    """The TEXT model to route to - explicitly, not the router's first entry.

    llama.cpp runs as a router over /models and lists every GGUF it finds,
    including image-model GGUFs that happen to live there (the Qwen-Image
    transformer, 2026-09-28). Taking entry [0] would one day ask the router to
    load a 9.7 GB image model as a chat model. Prefer an instruct model.
    """
    import requests

    override = os.environ.get("AUDIO_LLM_MODEL") or os.environ.get("WORLD_LLM_MODEL")
    if override:
        return override
    data = requests.get(f"{base}/v1/models", timeout=10).json().get("data") or []
    ids = [d.get("id", "") for d in data]
    for i in ids:
        if "instruct" in i.lower():
            return i
    return None


def _llm_style_plan(context: str, kind: str = MUSIC_KIND, attempts: int = 2
                    ) -> tuple[str | None, dict[str, Any], str]:
    """Ask the brain for a style. Returns (style|None, slots, note); never raises.

    `attempts`: a cold router load measured >45 s on 2026-09-28 (worlds.py
    recorded ~13 s earlier), so the FIRST call after the 120 s sleep can time
    out while the load continues, and a second call then answers in ~1 s. Two
    attempts suit an interactive propose; a generation passes 1, because two
    45 s attempts plus the 240 s build budget exceed something2's 300 s cap.
    """
    import requests

    base = os.environ.get("LLM_URL", "http://llm-server:8080")
    timeout = float(os.environ.get("AUDIO_LLM_TIMEOUT", "45"))
    options = []
    for s in STYLES:
        if s["kind"] != kind:
            continue
        slots = "; ".join(
            f"{n}: {spec['min']}-{spec['max']}" if spec["type"] == "int"
            else f"{n}: one of {spec['values']}"
            for n, spec in s["slots"].items())
        options.append(f'- "{s["value"]}" ({s["label"]}). Slots - {slots}')
    prompt = (
        f"Pick the {kind} for a map in a medieval fantasy pixel-art RPG.\n"
        f"Map description: {context}\n\n"
        f"Choose exactly ONE style from this list and fill its slots with "
        f"allowed values only:\n" + "\n".join(options) + "\n\n"
        'Reply with ONLY a JSON object, e.g. '
        '{"style": "<one of the names above>", "slots": {"mood": "..."}}')
    try:
        model = _llm_model(base)
        if not model:
            return None, {}, "no text model loaded in llama.cpp"
        # Two attempts: the router loads on demand and the first call after
        # its 120 s sleep answers with a non-completion body (worlds.py).
        text = None
        timed_out = False
        for attempt in range(1, attempts + 1):
            try:
                r = requests.post(
                    f"{base}/v1/chat/completions",
                    json={"model": model, "temperature": 0, "max_tokens": 200,
                          "messages": [{"role": "user", "content": prompt}]},
                    timeout=timeout)
            except requests.Timeout:
                timed_out = True
                continue
            if r.status_code == 200:
                try:
                    text = r.json()["choices"][0]["message"]["content"]
                    break
                except (ValueError, KeyError, IndexError):
                    pass
        if text is None:
            return None, {}, (f"LLM did not answer within {timeout:.0f}s x "
                              f"{attempts} (cold load?)" if timed_out
                              else "LLM did not answer")
        style, slots, notes = _parse_llm_answer(text, kind)
        if not style:
            return None, {}, f"{model}: " + "; ".join(notes)
        note = f"style chosen by {model}"
        return style, slots, note + (f"; {'; '.join(notes)}" if notes else "")
    except Exception as e:  # noqa: BLE001 - the brain is optional
        return None, {}, f"LLM call failed ({type(e).__name__})"


def plan_style(context: str, kind: str = MUSIC_KIND, *, use_llm: bool = True,
               attempts: int = 2) -> tuple[str, dict[str, Any], str]:
    """(style, slots, author) for a map description: the brain, else the rules.

    `author` always says which ran and, on a fallback, why - so a response can
    never claim the LLM chose something it did not.
    """
    if use_llm:
        style, slots, note = _llm_style_plan(context, kind, attempts)
        if style:
            return style, slots, note
        rstyle, rslots, rnote = _rules_style_plan(context, kind)
        return rstyle, rslots, f"{rnote}; LLM not used: {note}"
    return _rules_style_plan(context, kind)


def render(style: str | None = None, **slots: Any) -> dict[str, Any]:
    """One roster entry plus slot values, as the generation call needs it.

    Out-of-range and unknown slot values are corrected rather than raised on -
    an LLM will offer both - and every correction is named in `adjusted` so
    the response can say what happened instead of quietly differing.
    """
    entry = _entry(style or DEFAULT)
    values: dict[str, Any] = {}
    adjusted: list[str] = []

    for name, spec in entry["slots"].items():
        given = slots.get(name)
        if given is None or given == "":
            values[name] = spec["default"]
            continue
        if spec["type"] == "int":
            try:
                n = int(given)
            except (TypeError, ValueError):
                adjusted.append(f"{name}={given!r} is not a number, used "
                                f"{spec['default']}")
                values[name] = spec["default"]
                continue
            clamped = max(spec["min"], min(spec["max"], n))
            if clamped != n:
                adjusted.append(f"{name}={n} clamped to {clamped} "
                                f"({spec['min']}-{spec['max']})")
            values[name] = clamped
        else:
            if given not in spec["values"]:
                adjusted.append(f"{name}={given!r} is not in the roster, used "
                                f"{spec['default']!r}")
                values[name] = spec["default"]
            else:
                values[name] = given

    for name in slots:
        if name not in entry["slots"] and slots[name] not in (None, ""):
            adjusted.append(f"{name!r} is not a slot of {entry['value']!r}")

    return {
        "style": entry["value"],
        "kind": entry["kind"],
        "prompt": entry["template"].format(**values),
        "negative": entry["negative"],
        # Only music has a lyrics field to suppress; Stable Audio has none.
        "lyrics": "[Instrumental]" if entry["kind"] == MUSIC_KIND else None,
        "bpm": values.get("tempo_bpm"),
        "time_signature": entry["time_signature"],
        "slots": values,
        "adjusted": adjusted,
    }
