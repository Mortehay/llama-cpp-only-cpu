"""Per-action, per-view-family frame poses.

Two rules earned the hard way, both recorded in ADR 0005.

**Write poses, not verbs.** "walking" is a process; the model has to guess which
instant of it to draw, and it guesses the same one every time. Four frames of a
walk come back as four frames of standing. Naming limb positions makes each
frame a different, checkable target.

**Write what is VISIBLE FROM THAT CAMERA.** This is the one that bit on the
first full sheet. "left leg forward, right arm forward, mid stride" is a
sagittal description - it reads clearly in profile and is almost invisible
head-on, where a forward leg is foreshortened to nothing. Measured 2026-08-23:
with one prompt set for all directions, the two profile rows were
production-quality walk cycles and the front and back rows barely moved.

So a walk seen from the FRONT is not a stride, it is a knee lift: the foot
leaves the ground and the silhouette changes vertically rather than
horizontally. That is what a hand-drawn RPG-Maker front-walk row shows too.

The general rule, after making the same mistake twice (once with a stride, once
with a punch):

    A frame pose must change the SILHOUETTE as seen from THAT camera.
    Motion along the view axis does not exist in the output.

    profile cameras  -> sagittal motion: strides, thrusts, lunges
    front/back       -> vertical and lateral motion: knee lifts, overhead
                        swings, arms going wide

Ask a front camera for a forward punch and the model does not fail loudly - it
draws the nearest thing it can see, which measured out as a literal T-pose in
six of eight directions.
"""

# Which prompt set a direction uses. Left and right profiles share one - they
# are the same view family, mirrored (see domain.md).
FAMILIES = {
    "s": "front", "se": "front", "sw": "front",
    "n": "back", "ne": "back", "nw": "back",
    "e": "side", "w": "side",
}


def family(direction: str) -> str:
    """View family for a compass direction. Unknown directions read as side."""
    return FAMILIES.get(direction, "side")


# action -> family -> ordered frame poses.
# "side" is the fallback when a family is not listed.
ACTIONS = {
    "walk": {
        "side": [
            "left leg forward, right arm forward, mid stride",
            "legs together, passing pose, arms at sides",
            "right leg forward, left arm forward, mid stride",
            "legs together, passing pose, arms at sides",
        ],
        # Vertical, not horizontal. A raised knee and a lifted foot are what a
        # head-on camera can actually see of a walk.
        "front": [
            "left knee raised, left foot off the ground, right foot planted",
            "both feet flat on the ground, standing upright",
            "right knee raised, right foot off the ground, left foot planted",
            "both feet flat on the ground, standing upright",
        ],
        "back": [
            "left knee raised, left foot off the ground, right foot planted",
            "both feet flat on the ground, standing upright",
            "right knee raised, right foot off the ground, left foot planted",
            "both feet flat on the ground, standing upright",
        ],
    },
    "attack": {
        "side": [
            "arms drawn back, winding up to strike",
            "arms swinging forward, mid strike",
            "arms fully extended forward, strike landed",
            "arms lowering, recovering from the strike",
        ],
        # An OVERHEAD swing, not a forward thrust.
        #
        # The first version of this asked for "both fists thrust toward the
        # viewer". That is unrenderable head-on - a forward punch is pure
        # foreshortening - and the model resolved it by doing the nearest thing
        # it CAN draw: arms straight out sideways. Measured 2026-08-24: frame 3
        # came back as a literal T-pose in all six non-profile directions.
        #
        # An overhead swing moves the arms VERTICALLY, which a front or back
        # camera sees in full. Same reasoning as the walk being a knee lift
        # rather than a stride.
        "front": [
            "both arms raised high overhead, leaning back",
            "arms coming down, elbows bent, weight shifting forward",
            "arms swung down low in front, body hunched forward",
            "arms returning to the sides",
        ],
        "back": [
            "both arms raised high overhead, leaning back",
            "arms coming down, elbows bent, weight shifting forward",
            "arms swung down low in front, body hunched forward",
            "arms returning to the sides",
        ],
    },
    # Idle is a breathing cycle and reads the same from every angle, so it
    # deliberately has no per-family variants.
    "idle": {
        "side": [
            "standing still, arms at sides",
            "standing still, shoulders slightly raised",
            "standing still, arms at sides",
            "standing still, shoulders slightly lowered",
        ],
    },
}


def frames(action: str, direction: str, count: int | None = None) -> list:
    """Frame poses for one action seen from one direction.

    Falls back to the "side" set when an action has no entry for that family,
    which is correct for anything genuinely view-independent (idle) and is a
    visible-but-harmless approximation for anything else.
    """
    if action not in ACTIONS:
        raise KeyError(f"unknown action {action!r}; have {sorted(ACTIONS)}")
    by_family = ACTIONS[action]
    poses = by_family.get(family(direction)) or by_family["side"]
    return poses[:count] if count else list(poses)
