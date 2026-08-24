"""Per-action, per-view-family frame poses, loaded from action_prompts.json.

THE TABLE IS DATA, NOT CODE. It used to be an ACTIONS dict literal in this
file, while action_prompts.json - which documents itself as the place to edit
how an action looks - sat entirely unread by the conveyor. That was two action
tables: the ControlNet img2img path read the JSON, the 2D conveyor that the UI
actually drives read this module, and adding an action meant editing Python and
rebuilding the worker. Now both read the one file. See its `_readme` for the
authoring rules; the two that were learned the hard way are repeated here
because they are the ones that get violated:

**Write poses, not verbs.** "walking" is a process; the model has to guess which
instant of it to draw, and it guesses the same one every time. Four frames of a
walk come back as four frames of standing. Naming limb positions makes each
frame a different, checkable target.

**Write what is VISIBLE FROM THAT CAMERA.**

    A frame pose must change the SILHOUETTE as seen from THAT camera.
    Motion along the view axis does not exist in the output.

    profile cameras  -> sagittal motion: strides, thrusts, lunges
    front/back       -> vertical and lateral motion: knee lifts, overhead
                        swings, arms going wide

Measured 2026-08-23: with one prompt set for all directions, the two profile
rows were production-quality walk cycles and the front and back rows barely
moved. Ask a front camera for a forward punch and the model does not fail
loudly - it draws the nearest thing it can see, which measured out as a literal
T-pose in six of eight directions.

Loading follows the same discipline as tasks.load_action_prompts(): re-read on
mtime change so edits apply to the next sheet without restarting the worker,
and keep the last good copy when the file will not parse, because a JSON typo
should not take the service down mid-run.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

ACTION_PROMPTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "action_prompts.json")

# Fallback if the file is unreadable on the very first load, before any good
# copy exists. Deliberately minimal: enough to keep the service answering, not
# a second copy of the table that could drift from the JSON.
_FALLBACK_FAMILIES = {
    "s": "front", "se": "front", "sw": "front",
    "n": "back", "ne": "back", "nw": "back",
    "e": "side", "w": "side",
}

# (mtime, actions, families, labels). One tuple so a reload swaps all three
# together - a half-updated view is exactly the drift this module exists to end.
_cache = (None, None, None, None)

_EMPTY = ({}, dict(_FALLBACK_FAMILIES), {})


def _parse(doc: dict):
    """Pull the conveyor's tables out of the parsed document."""
    families = dict(doc.get("view_families") or _FALLBACK_FAMILIES)

    actions = {}
    labels = {}
    for entry in doc.get("actions", []):
        frames = entry.get("frames")
        if not frames:
            # An entry with no `frames` block is ControlNet-only. Not an error:
            # the two pipelines do not have to cover the same action set.
            continue
        key = entry.get("sheet_id") or entry.get("id")
        if "side" not in frames:
            # `side` is the documented fallback for every family an action does
            # not list, so an entry without it would KeyError the first time a
            # profile direction was requested. Refuse it here, at load, where
            # the message can name the entry.
            logger.error("action %r has a frames block with no 'side' family; "
                         "skipping it. See action_prompts.json _readme.", key)
            continue
        actions[key] = {f: list(v) for f, v in frames.items() if v}
        labels[key] = entry.get("sheet_label") or entry.get("label") or key

    return actions, families, labels


def _load():
    """(actions, families, labels), re-read when the file changes on disk."""
    global _cache
    try:
        mtime = os.path.getmtime(ACTION_PROMPTS_PATH)
    except OSError as e:
        if _cache[1] is not None:
            return _cache[1:]
        logger.error("%s is unreadable (%s); no actions are available.",
                     ACTION_PROMPTS_PATH, e)
        return _EMPTY

    if mtime == _cache[0]:
        return _cache[1:]

    try:
        with open(ACTION_PROMPTS_PATH, encoding="utf-8") as fh:
            parsed = _parse(json.load(fh))
    except Exception as e:
        if _cache[1] is not None:
            logger.error("Could not reload %s: %s. Keeping the previously "
                         "loaded %d action(s).", ACTION_PROMPTS_PATH, e,
                         len(_cache[1]))
            return _cache[1:]
        logger.error("Could not load %s: %s. No actions are available.",
                     ACTION_PROMPTS_PATH, e)
        return _EMPTY

    _cache = (mtime,) + parsed
    logger.info("Loaded %d sheet action(s) from action_prompts.json: %s",
                len(parsed[0]), ", ".join(sorted(parsed[0])))
    return parsed


class _LiveMapping(dict):
    """A dict view that re-reads the JSON on every access.

    `ACTIONS` and `FAMILIES` were module-level dict literals and callers use
    them as such - `act not in action_lib.ACTIONS`, `sorted(ACTIONS)`,
    `FAMILIES.get(d)`. Keeping the names as mappings means the file can become
    the source without touching any of those call sites, and without losing the
    mtime reload the JSON already promised.
    """

    def __init__(self, index):
        super().__init__()
        self._index = index

    def _live(self):
        return _load()[self._index]

    def __getitem__(self, k):
        return self._live()[k]

    def __contains__(self, k):
        return k in self._live()

    def __iter__(self):
        return iter(self._live())

    def __len__(self):
        return len(self._live())

    def get(self, k, default=None):
        return self._live().get(k, default)

    def keys(self):
        return self._live().keys()

    def values(self):
        return self._live().values()

    def items(self):
        return self._live().items()

    def __repr__(self):
        return repr(self._live())


# action -> family -> ordered frame poses.
ACTIONS = _LiveMapping(0)

# Which prompt set a direction uses. Left and right profiles share one - they
# are the same view family, mirrored (see domain.md).
FAMILIES = _LiveMapping(1)


def family(direction: str) -> str:
    """View family for a compass direction. Unknown directions read as side."""
    return FAMILIES.get(direction, "side")


def label(action: str) -> str:
    """Display name for an action, for the UI to render its checkbox."""
    return _load()[2].get(action, action)


def frames(action: str, direction: str, count: int | None = None) -> list:
    """Frame poses for one action seen from one direction.

    Falls back to the "side" set when an action has no entry for that family,
    which is correct for anything genuinely view-independent (idle, sway) and
    is a visible-but-harmless approximation for anything else.
    """
    actions = _load()[0]
    if action not in actions:
        raise KeyError(f"unknown action {action!r}; have {sorted(actions)}")
    by_family = actions[action]
    poses = by_family.get(family(direction)) or by_family["side"]
    return poses[:count] if count else list(poses)


def available_frames(action: str, direction: str) -> int:
    """How many distinct poses exist for this action from this view family."""
    return len(frames(action, direction))


def max_frames(actions=None, directions=None) -> int:
    """The largest frame count a sheet over these axes can actually be built at.

    `frames()` truncates to the poses that exist, so asking for more than this
    does not produce more frames - it produces a request the pipeline cannot
    satisfy. Read from the library rather than hardcoded, so extending a pose
    list in the JSON raises the limit with no code change.
    """
    known = _load()[0]
    actions = list(actions) if actions else sorted(known)
    directions = list(directions) if directions else sorted(FAMILIES)
    if not actions or not directions:
        return 0
    return min(available_frames(a, d) for a in actions for d in directions)


def catalog() -> dict:
    """Everything the UI needs to render its action controls."""
    known, _, labels = _load()
    return {
        "actions": [
            {"id": a, "label": labels.get(a, a), "max_frames": max_frames([a])}
            for a in sorted(known)
        ],
        "max_frames": max_frames(),
    }
