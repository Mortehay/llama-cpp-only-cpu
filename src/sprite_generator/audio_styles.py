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

# Keyword -> style, for `_rules_style_plan` in ticket 08 and for anyone who
# wants a style from a map name without waking the LLM. First match wins, so
# order matters: "mine" before "village" because "mining village" is a dungeon
# with houses attached.
RULES: list[tuple[tuple[str, ...], str]] = [
    (("dungeon", "cave", "crypt", "mine", "tomb", "catacomb", "lair"), "dungeon"),
    (("battle", "war", "siege", "arena", "boss", "fortress"), "battle"),
    (("tavern", "inn", "alehouse", "pub", "feast"), "tavern"),
    (("village", "town", "hamlet", "farm", "market", "square"), "village"),
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
