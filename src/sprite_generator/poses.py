"""OpenPose control images for sprite animation frames.

WHY THIS EXISTS

img2img cannot animate a character. It preserves composition, and pose *is*
composition, so the one `strength` knob controls both identity and pose
together: low strength keeps the character but freezes the pose, high strength
frees the pose but returns a different character. Measured on a 4-frame walk,
that trade never had a usable middle - see the step 2 section of the README.

ControlNet breaks the tie by giving pose its own conditioning channel. Identity
still comes from the core image through img2img; the pose comes from a skeleton
drawn here. The two stop fighting over one parameter.

WHY THE SKELETONS ARE HAND-AUTHORED

The usual route is `controlnet_aux`'s OpenPose *detector*, run over a reference
photo. That is the wrong tool twice over: there is no reference photo of a
sprite mid-walk, and a detector trained on photographs does poorly on a 64px
stylised character with four-pixel limbs. Authoring the keypoints directly is
deterministic, needs no extra model, costs nothing at runtime, and animation
poses are a solved problem anyway - contact, passing, contact, passing.

FORMAT

COCO-18 keypoints, which is what `control_v11p_sd15_openpose` was trained on:

    0 nose        1 neck        2 r-shoulder  3 r-elbow     4 r-wrist
    5 l-shoulder  6 l-elbow     7 l-wrist     8 r-hip       9 r-knee
   10 r-ankle    11 l-hip      12 l-knee     13 l-ankle    14 r-eye
   15 l-eye      16 r-ear      17 l-ear

Coordinates are normalised to the box the character occupies, 0..1, y down. A
keypoint may be None, meaning "not visible" - the far ear in a side view, for
instance - and is then skipped along with any limb that touches it.
"""

from PIL import Image, ImageDraw

# OpenPose's own limb order and palette. ControlNet keys off these colours, so
# they are not decorative: recolouring the skeleton degrades conditioning.
LIMB_SEQ = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
]

COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85),
]


def _torso(facing_side: bool):
    """Keypoints shared by every phase of a cycle: head, neck, shoulders, hips.

    In a side view the shoulders and hips stack on one x, and the far ear is
    hidden behind the head. Facing front they spread apart and both ears show.
    """
    if facing_side:
        return {
            0: (0.56, 0.10), 1: (0.50, 0.19),
            2: (0.50, 0.21), 5: (0.50, 0.21),
            8: (0.50, 0.50), 11: (0.50, 0.50),
            14: (0.58, 0.09), 15: (0.55, 0.09),
            16: (0.52, 0.10), 17: None,
        }
    return {
        0: (0.50, 0.10), 1: (0.50, 0.19),
        2: (0.40, 0.22), 5: (0.60, 0.22),
        8: (0.44, 0.50), 11: (0.56, 0.50),
        14: (0.47, 0.09), 15: (0.53, 0.09),
        16: (0.43, 0.10), 17: (0.57, 0.10),
    }


def _phase(base, limbs, lift=0.0):
    """One frame: torso plus limb keypoints, optionally raised off the ground.

    `lift` models the vertical bob of a walk cycle. The body is highest at the
    passing pose and lowest at contact; without it a walk reads as a character
    sliding along on rails.
    """
    kp = dict(base)
    kp.update(limbs)
    if lift:
        kp = {i: (p[0], p[1] - lift) if p else None for i, p in kp.items()}
    return kp


_SIDE = _torso(facing_side=True)
_FRONT = _torso(facing_side=False)

# Walk cycle, side view, facing right. Four phases: contact, passing, contact
# mirrored, passing mirrored. Arms swing opposite the legs, which is what makes
# a walk read as a walk rather than a shuffle.
WALK_SIDE = [
    _phase(_SIDE, {
        3: (0.44, 0.35), 4: (0.40, 0.47), 6: (0.56, 0.34), 7: (0.62, 0.45),
        9: (0.58, 0.70), 10: (0.66, 0.93), 12: (0.44, 0.72), 13: (0.36, 0.93),
    }),
    _phase(_SIDE, {
        3: (0.48, 0.35), 4: (0.48, 0.47), 6: (0.52, 0.35), 7: (0.53, 0.46),
        9: (0.52, 0.70), 10: (0.52, 0.93), 12: (0.50, 0.69), 13: (0.46, 0.88),
    }, lift=0.03),
    _phase(_SIDE, {
        3: (0.56, 0.34), 4: (0.62, 0.45), 6: (0.44, 0.35), 7: (0.40, 0.47),
        9: (0.44, 0.72), 10: (0.36, 0.93), 12: (0.58, 0.70), 13: (0.66, 0.93),
    }),
    _phase(_SIDE, {
        3: (0.52, 0.35), 4: (0.53, 0.46), 6: (0.48, 0.35), 7: (0.48, 0.47),
        9: (0.50, 0.69), 10: (0.46, 0.88), 12: (0.52, 0.70), 13: (0.52, 0.93),
    }, lift=0.03),
]

# Walk cycle seen head-on. The stride reads as one leg crossing in front of the
# other, so the movement is mostly in x with a small lift.
WALK_FRONT = [
    _phase(_FRONT, {
        3: (0.36, 0.36), 4: (0.34, 0.48), 6: (0.64, 0.36), 7: (0.66, 0.48),
        9: (0.42, 0.71), 10: (0.40, 0.94), 12: (0.57, 0.72), 13: (0.58, 0.93),
    }),
    _phase(_FRONT, {
        3: (0.37, 0.36), 4: (0.36, 0.48), 6: (0.63, 0.36), 7: (0.64, 0.48),
        9: (0.45, 0.70), 10: (0.46, 0.90), 12: (0.55, 0.71), 13: (0.55, 0.94),
    }, lift=0.03),
    _phase(_FRONT, {
        3: (0.35, 0.36), 4: (0.33, 0.48), 6: (0.65, 0.36), 7: (0.67, 0.48),
        9: (0.43, 0.72), 10: (0.44, 0.93), 12: (0.56, 0.71), 13: (0.54, 0.94),
    }),
    _phase(_FRONT, {
        3: (0.37, 0.36), 4: (0.36, 0.48), 6: (0.63, 0.36), 7: (0.64, 0.48),
        9: (0.44, 0.71), 10: (0.44, 0.94), 12: (0.56, 0.70), 13: (0.57, 0.90),
    }, lift=0.03),
]

# Idle: a breathing loop. Deliberately tiny - an idle that moves as much as a
# walk looks like a nervous tic.
IDLE = [
    _phase(_FRONT, {
        3: (0.37, 0.36), 4: (0.36, 0.48), 6: (0.63, 0.36), 7: (0.64, 0.48),
        9: (0.44, 0.72), 10: (0.44, 0.95), 12: (0.56, 0.72), 13: (0.56, 0.95),
    }),
    _phase(_FRONT, {
        3: (0.37, 0.35), 4: (0.36, 0.47), 6: (0.63, 0.35), 7: (0.64, 0.47),
        9: (0.44, 0.71), 10: (0.44, 0.95), 12: (0.56, 0.71), 13: (0.56, 0.95),
    }, lift=0.012),
    _phase(_FRONT, {
        3: (0.38, 0.36), 4: (0.37, 0.48), 6: (0.62, 0.36), 7: (0.63, 0.48),
        9: (0.44, 0.72), 10: (0.44, 0.95), 12: (0.56, 0.72), 13: (0.56, 0.95),
    }),
    _phase(_FRONT, {
        3: (0.37, 0.35), 4: (0.36, 0.47), 6: (0.63, 0.35), 7: (0.64, 0.47),
        9: (0.44, 0.71), 10: (0.44, 0.95), 12: (0.56, 0.71), 13: (0.56, 0.95),
    }, lift=0.012),
]

# Attack: wind up behind the head, then swing down and through.
ATTACK = [
    _phase(_SIDE, {
        3: (0.42, 0.26), 4: (0.38, 0.16), 6: (0.52, 0.36), 7: (0.58, 0.42),
        9: (0.46, 0.71), 10: (0.40, 0.94), 12: (0.56, 0.71), 13: (0.60, 0.94),
    }),
    _phase(_SIDE, {
        3: (0.50, 0.24), 4: (0.58, 0.20), 6: (0.54, 0.36), 7: (0.60, 0.42),
        9: (0.46, 0.71), 10: (0.40, 0.94), 12: (0.57, 0.70), 13: (0.62, 0.94),
    }),
    _phase(_SIDE, {
        3: (0.60, 0.30), 4: (0.72, 0.34), 6: (0.56, 0.37), 7: (0.62, 0.44),
        9: (0.48, 0.71), 10: (0.42, 0.94), 12: (0.58, 0.70), 13: (0.64, 0.93),
    }),
    _phase(_SIDE, {
        3: (0.62, 0.42), 4: (0.70, 0.52), 6: (0.54, 0.38), 7: (0.58, 0.46),
        9: (0.48, 0.72), 10: (0.42, 0.94), 12: (0.57, 0.71), 13: (0.62, 0.94),
    }),
]

# Recoil: head and torso thrown back, arms up, weight onto the back foot.
HURT = [
    _phase(_SIDE, {
        3: (0.44, 0.32), 4: (0.40, 0.24), 6: (0.56, 0.32), 7: (0.60, 0.24),
        9: (0.48, 0.71), 10: (0.44, 0.94), 12: (0.56, 0.72), 13: (0.60, 0.94),
    }),
    _phase(_SIDE, {
        3: (0.40, 0.30), 4: (0.34, 0.22), 6: (0.54, 0.30), 7: (0.58, 0.20),
        9: (0.46, 0.72), 10: (0.38, 0.94), 12: (0.54, 0.73), 13: (0.58, 0.95),
    }),
    _phase(_SIDE, {
        3: (0.38, 0.32), 4: (0.32, 0.26), 6: (0.52, 0.32), 7: (0.56, 0.24),
        9: (0.44, 0.73), 10: (0.36, 0.95), 12: (0.53, 0.73), 13: (0.57, 0.95),
    }),
    _phase(_SIDE, {
        3: (0.42, 0.33), 4: (0.37, 0.26), 6: (0.55, 0.33), 7: (0.59, 0.26),
        9: (0.47, 0.72), 10: (0.42, 0.94), 12: (0.55, 0.72), 13: (0.59, 0.94),
    }),
]


def _mirror(cycle):
    """Flip a cycle horizontally, swapping left and right keypoints.

    A character walking left is a character walking right, mirrored - there is
    no reason to author the coordinates twice, and a hand-authored mirror would
    drift out of sync with its original on the first edit.
    """
    swap = {2: 5, 5: 2, 3: 6, 6: 3, 4: 7, 7: 4,
            8: 11, 11: 8, 9: 12, 12: 9, 10: 13, 13: 10,
            14: 15, 15: 14, 16: 17, 17: 16}
    out = []
    for kp in cycle:
        m = {}
        for i, p in kp.items():
            m[swap.get(i, i)] = (1.0 - p[0], p[1]) if p else None
        out.append(m)
    return out


# Matched against the action text in order, so put the specific entries first:
# "move right" has to win before a bare "move" would.
POSE_LIBRARY = [
    (("move right", "walk right", "run right"), WALK_SIDE),
    (("move left", "walk left", "run left"), _mirror(WALK_SIDE)),
    (("move up", "walk up", "walk back"), WALK_FRONT),
    (("move down", "walk down", "walk front", "move", "walk", "run"), WALK_FRONT),
    (("attack", "strike", "swing"), ATTACK),
    (("damage", "hurt", "hit", "burning", "burn"), HURT),
    (("idle", "stand"), IDLE),
]


def cycle_for(action: str):
    """Pick the pose cycle for an action name, or None if nothing matches.

    Returning None matters: an unrecognised action should fall back to plain
    img2img rather than being forced into a walk cycle it never asked for.
    """
    a = (action or "").lower()
    for keys, cycle in POSE_LIBRARY:
        if any(k in a for k in keys):
            return cycle
    return None


def render_skeleton(keypoints, size, box=None):
    """Draw one COCO-18 skeleton as an OpenPose control image.

    `box` is the (x0, y0, x1, y1) region of the output the character occupies -
    normally the core sprite's own alpha bounds. The skeleton is fitted to it so
    the pose lands where the character actually is; a skeleton centred on the
    canvas while the sprite sits low and left conditions for a pose the model
    then has to reconcile with the init image, and the result is a stretched or
    doubled figure.
    """
    w, h = (size, size) if isinstance(size, int) else size
    if box is None:
        box = (0, 0, w, h)
    bx0, by0, bx1, by1 = box
    bw, bh = max(1, bx1 - bx0), max(1, by1 - by0)

    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    def xy(i):
        p = keypoints.get(i)
        return None if p is None else (bx0 + p[0] * bw, by0 + p[1] * bh)

    # OpenPose renders limbs at roughly 4px on a 512px image and joints slightly
    # thicker. Scale with the canvas so a 256px control image is not a smear.
    stick = max(2, round(min(w, h) / 128))
    joint = max(2, round(min(w, h) / 110))

    for n, (a, b) in enumerate(LIMB_SEQ):
        pa, pb = xy(a), xy(b)
        if pa and pb:
            draw.line([pa, pb], fill=COLORS[n % len(COLORS)], width=stick)

    # Joints go on top of the limbs, as OpenPose draws them.
    for i in range(18):
        p = xy(i)
        if p:
            draw.ellipse([p[0] - joint, p[1] - joint, p[0] + joint, p[1] + joint],
                         fill=COLORS[i % len(COLORS)])
    return canvas


def control_images(action: str, count: int, size, box=None):
    """Control images for `count` frames of `action`, or None if unposed.

    The cycle is sampled cyclically, so asking for 6 frames of a 4-phase walk
    continues the cycle rather than stopping dead or padding with copies.
    """
    cycle = cycle_for(action)
    if not cycle:
        return None
    return [render_skeleton(cycle[f % len(cycle)], size, box) for f in range(count)]
