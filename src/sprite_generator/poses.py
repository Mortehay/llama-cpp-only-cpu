"""OpenPose control images for sprite animation frames.

WHY THIS EXISTS

img2img cannot animate a character. It preserves composition, and pose *is*
composition, so the one `strength` knob controls both identity and pose
together: low strength keeps the character but freezes the pose, high strength
frees the pose but returns a different character. Measured on a 4-frame walk,
that trade never had a usable middle - see the step 2 section of the README.

ControlNet breaks the tie by giving pose its own conditioning channel. Identity
still comes from the core image through img2img; the pose comes from a skeleton
drawn here. The two stop competing for one parameter.

WHY THE SKELETONS ARE HAND-AUTHORED

The usual route is `controlnet_aux`'s OpenPose *detector*, run over a reference
photo. That is the wrong tool twice over: there is no reference photo of a
sprite mid-walk, and a detector trained on photographs does poorly on a 64px
stylised character with four-pixel limbs. Authoring the keypoints directly is
deterministic, needs no extra model, costs nothing at runtime, and animation
poses are a solved problem anyway - contact, passing, contact, passing.

TWO THINGS THAT HAVE TO BE RIGHT

*Proportion.* The skeleton sets the body plan the model draws to. A default
OpenPose figure is roughly 7.5 heads tall; these pixel-art cores are roughly 4,
with a big head and short legs. Conditioning a chibi sprite on an adult
skeleton makes ControlNet and img2img pull in different directions, and the
result is a stretched or doubled figure. `Y` below is the single place that
proportion lives - retune it there if the art style changes.

*Consistent limb length.* Elbows and knees are SOLVED from the endpoints rather
than typed in, so every phase of a cycle has arms and legs of exactly the same
length. Hand-placed mid-joints drift a few percent per frame, and a skeleton
whose thigh grows between frame 2 and frame 3 conditions for a body that grows
with it - which reads as the character morphing, the exact failure ControlNet
was brought in to remove.

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

import math

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

# Vertical landmarks, as a fraction of the character's height. Tuned for the
# chibi proportions the pixel-art cores actually have - big head, short legs,
# roughly 4 heads tall - NOT the ~7.5-head adult figure an OpenPose skeleton
# defaults to. This is the one place proportion is defined.
Y = {
    "eye": 0.15, "nose": 0.17, "ear": 0.17,
    "neck": 0.30, "shoulder": 0.33,
    "hip": 0.57, "ankle": 0.95,
}

# Half-limb lengths, i.e. upper arm = forearm = ARM, thigh = shin = LEG.
# Derived from the landmarks: a hanging arm reaches 2*ARM below the shoulder,
# a straight leg spans hip to ankle.
LEG = (Y["ankle"] - Y["hip"]) / 2      # 0.19
ARM = 0.14


def _solve_joint(a, b, seg, bend):
    """Place the mid-joint of a two-segment limb running from `a` to `b`.

    Both segments are `seg` long, so the joint sits on the perpendicular
    bisector of ab, offset by sqrt(seg^2 - (d/2)^2). Two solutions exist -
    knee forward or knee backward - and `bend` (a direction hint) picks one.

    When the endpoints are further apart than the limb can reach the limb is
    simply straight, and the joint goes at the midpoint rather than raising a
    domain error on a negative square root. That case means the pose data asks
    for a stride longer than the leg; it is clamped instead of rejected so a
    slightly over-reaching pose degrades to a straight leg instead of crashing
    a generation job.
    """
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    d = math.hypot(dx, dy)
    mx, my = (ax + bx) / 2, (ay + by) / 2
    if d < 1e-6 or d >= 2 * seg:
        return (mx, my)

    h = math.sqrt(seg * seg - (d / 2) ** 2)
    # Unit perpendicular to ab.
    px, py = -dy / d, dx / d
    if px * bend[0] + py * bend[1] < 0:
        px, py = -px, -py
    return (mx + px * h, my + py * h)


def _head(facing_side, face_dir=1):
    """Head and neck keypoints.

    In a side view the eyes and nose sit toward the facing direction and the
    far ear is hidden behind the skull; head-on, both ears show.
    """
    if not facing_side:
        return {0: (0.50, Y["nose"]), 14: (0.47, Y["eye"]), 15: (0.53, Y["eye"]),
                16: (0.43, Y["ear"]), 17: (0.57, Y["ear"])}
    # Only the NEAR eye, and the ear BEHIND it. This is the whole profile cue.
    #
    # Showing both eyes was enough to defeat the side view entirely: shoulders
    # and hips already sit on one x, but a skeleton with two eyes and limbs
    # spread symmetrically left and right is, as a 2D image, indistinguishable
    # from a front view with the legs apart - and that is exactly what the
    # model drew, every frame, for every strength and conditioning scale tried.
    # One eye plus a trailing ear is what says "this figure is turned".
    f = face_dir
    return {0: (0.50 + 0.09 * f, Y["nose"]),
            14: (0.50 + 0.07 * f, Y["eye"]), 15: None,
            16: (0.50 - 0.02 * f, Y["ear"]), 17: None}


def _build(side, wrists, ankles, lift=0.0, arm_bend=None, leg_bend=None):
    """Assemble one frame from the four endpoints that actually vary.

    Everything else - head, neck, shoulders, hips - is fixed by the body plan,
    and the elbows and knees are solved. A phase therefore specifies only where
    the hands and feet are, which is how animation is described anyway.

    `lift` raises the whole body, modelling the vertical bob of a walk: highest
    at the passing pose, lowest at contact. Without it a walk reads as a
    character sliding along on rails.
    """
    if side:
        sh_r = sh_l = (0.50, Y["shoulder"])
        hip_r = hip_l = (0.50, Y["hip"])
        arm_bend = arm_bend or ((-1, 0), (-1, 0))
        leg_bend = leg_bend or ((1, 0), (1, 0))
        head = _head(True)
    else:
        sh_r, sh_l = (0.40, Y["shoulder"]), (0.60, Y["shoulder"])
        hip_r, hip_l = (0.45, Y["hip"]), (0.55, Y["hip"])
        arm_bend = arm_bend or ((-1, 0), (1, 0))
        leg_bend = leg_bend or ((-1, 0), (1, 0))
        head = _head(False)

    wr_r, wr_l = wrists
    an_r, an_l = ankles

    kp = dict(head)
    kp[1] = (0.50, Y["neck"])
    kp[2], kp[5] = sh_r, sh_l
    kp[8], kp[11] = hip_r, hip_l
    kp[4], kp[7] = wr_r, wr_l
    kp[10], kp[13] = an_r, an_l
    kp[3] = _solve_joint(sh_r, wr_r, ARM, arm_bend[0])
    kp[6] = _solve_joint(sh_l, wr_l, ARM, arm_bend[1])
    kp[9] = _solve_joint(hip_r, an_r, LEG, leg_bend[0])
    kp[12] = _solve_joint(hip_l, an_l, LEG, leg_bend[1])

    if lift:
        kp = {i: (p[0], p[1] - lift) if p else None for i, p in kp.items()}
    return kp


# Walk, side view, facing right. Contact / passing / contact / passing, with
# arms swinging opposite the legs - which is what makes a walk read as a walk
# rather than a shuffle. The forward foot rises as it reaches: at full stride a
# foot planted flat at ankle height would be further from the hip than the leg
# is long.
WALK_SIDE = [
    _build(True, ((0.38, 0.56), (0.62, 0.55)), ((0.64, 0.92), (0.38, 0.93))),
    # The swing foot tucks up UNDER the body, ankle near 0.82 - not merely
    # lowered to 0.85. With 0.19 segments a hip-to-ankle gap of only 0.28 puts
    # the solved knee level with, or below, the ankle: a bird's leg. Lifting
    # the foot properly swings the knee forward and up, which is what a passing
    # pose actually looks like from the side.
    _build(True, ((0.47, 0.60), (0.53, 0.58)), ((0.50, 0.95), (0.52, 0.82)), lift=0.025),
    _build(True, ((0.62, 0.55), (0.38, 0.56)), ((0.38, 0.93), (0.64, 0.92))),
    _build(True, ((0.53, 0.58), (0.47, 0.60)), ((0.52, 0.82), (0.50, 0.95)), lift=0.025),
]

# Walk seen head-on. Deliberately shallow: a step toward the camera is
# foreshortened, so its 2D segments get SHORTER, but the solver holds them
# constant. Lifting a foot to 0.87 therefore threw the knee 0.12 out sideways -
# a leg bent out at the hip rather than a leg stepping forward. Keep the feet
# near the ground and let the body bob carry the movement, which is how a
# front-facing walk is drawn in pixel art anyway.
WALK_FRONT = [
    _build(False, ((0.36, 0.59), (0.64, 0.59)), ((0.43, 0.94), (0.56, 0.93))),
    _build(False, ((0.37, 0.60), (0.63, 0.60)), ((0.45, 0.95), (0.55, 0.90)), lift=0.025),
    _build(False, ((0.36, 0.59), (0.64, 0.59)), ((0.44, 0.93), (0.56, 0.94))),
    _build(False, ((0.37, 0.60), (0.63, 0.60)), ((0.45, 0.90), (0.55, 0.95)), lift=0.025),
]

# Idle: a breathing loop. Deliberately tiny - an idle that moves as much as a
# walk looks like a nervous tic.
#
# Phases 1 and 3 used to be the SAME _build call with the same arguments, so
# render_skeleton returned two byte-identical control images. Every other input
# to a frame is already shared - same core, same prompt, same seed - so the
# skeleton is the only thing that distinguishes one frame from the next, and
# duplicating it made frames 2 and 4 pixel-identical. A four frame idle was
# therefore two frames padded out to four, which is visible as a stutter the
# moment the strip is played back.
#
# Now a real four-beat breath: settle, inhale, hold, exhale. Phase 3 differs
# from phase 1 in both lift and shoulder width, so no two phases can collapse.
IDLE = [
    _build(False, ((0.37, 0.60), (0.63, 0.60)), ((0.45, 0.95), (0.55, 0.95))),
    _build(False, ((0.37, 0.59), (0.63, 0.59)), ((0.45, 0.95), (0.55, 0.95)), lift=0.012),
    _build(False, ((0.38, 0.60), (0.62, 0.60)), ((0.45, 0.95), (0.55, 0.95))),
    _build(False, ((0.375, 0.595), (0.625, 0.595)), ((0.45, 0.95), (0.55, 0.95)), lift=0.006),
]

# Attack: wind the right arm up behind the head, then swing down and through.
# Feet stay in a braced stance - the weight shifts, the stance does not walk.
ATTACK = [
    _build(True, ((0.36, 0.22), (0.57, 0.58)), ((0.40, 0.93), (0.60, 0.93))),
    _build(True, ((0.55, 0.17), (0.58, 0.58)), ((0.40, 0.93), (0.60, 0.93))),
    _build(True, ((0.74, 0.36), (0.60, 0.56)), ((0.42, 0.93), (0.61, 0.92))),
    # 0.68/0.53, not 0.70/0.55: the further reach exceeded 2*ARM, so the solver
    # clamped the arm straight and the follow-through lost its elbow.
    _build(True, ((0.68, 0.53), (0.58, 0.58)), ((0.42, 0.94), (0.60, 0.93))),
]

# --- Ranged and magic attacks -------------------------------------------
#
# All four are `side=False`. ATTACK above is a side view, which suits a sword
# swing because the whole arc is visible in profile - but it is the wrong
# choice for these. What makes a bow, a sling or a cast legible is the SHAPE
# the two arms make relative to each other, and a profile hides one arm behind
# the other. Front view also matches the cores, which face the camera; see the
# note on BURN for what happens when the skeleton and the core disagree.
#
# Feet stay braced throughout all four. These are stances, not walks: the
# weight shifts between phases but the feet do not travel, so a sheet can cut
# between an attack and an idle without the character appearing to teleport.

# Bow: nock, draw, hold at full draw, release. The bow arm stays extended and
# still - it is the draw hand travelling back to the cheek and then snapping
# away that carries the whole action.
BOW = [
    _build(False, ((0.48, 0.42), (0.72, 0.36)), ((0.42, 0.94), (0.58, 0.94))),
    _build(False, ((0.44, 0.28), (0.78, 0.33)), ((0.42, 0.93), (0.58, 0.94))),
    _build(False, ((0.40, 0.26), (0.80, 0.33)), ((0.41, 0.94), (0.59, 0.93))),
    # Release: the draw hand flies BACK past the face, not forward. Drawn
    # forward it reads as a punch and the arrow appears to be thrown.
    _build(False, ((0.34, 0.30), (0.79, 0.34)), ((0.42, 0.93), (0.58, 0.94))),
]

# Sling: wind down and back, whirl up and over, release across the body,
# follow through. The overhead sweep is the recognisable part.
SLING = [
    _build(False, ((0.30, 0.45), (0.56, 0.44)), ((0.42, 0.94), (0.58, 0.93))),
    _build(False, ((0.34, 0.18), (0.58, 0.42)), ((0.42, 0.93), (0.58, 0.94)),
           lift=0.012),
    _build(False, ((0.52, 0.14), (0.60, 0.44)), ((0.41, 0.93), (0.59, 0.94)),
           lift=0.018),
    # 0.66/0.30 keeps the follow-through inside 2*ARM. Reaching further clamps
    # the arm straight and the release loses its elbow, exactly as ATTACK's
    # phase 3 did before it was pulled back in.
    _build(False, ((0.66, 0.30), (0.58, 0.46)), ((0.42, 0.94), (0.58, 0.93))),
]

# Close magic: gather at the chest, wind back, then throw both hands out low
# and wide. Hands stay near the body - that proximity is what separates a
# short-range burst from the overhead channel below.
MAGIC_CLOSE = [
    _build(False, ((0.48, 0.44), (0.52, 0.44)), ((0.43, 0.94), (0.57, 0.94))),
    _build(False, ((0.44, 0.40), (0.56, 0.40)), ((0.42, 0.93), (0.58, 0.94)),
           lift=0.010),
    _build(False, ((0.36, 0.36), (0.64, 0.36)), ((0.41, 0.94), (0.59, 0.93))),
    _build(False, ((0.30, 0.42), (0.70, 0.42)), ((0.42, 0.94), (0.58, 0.94))),
]

# Distant magic: raise both arms, bring the hands together overhead to channel,
# then sweep down as it is released. Symmetric on purpose - unlike BURN, where
# symmetry read as surrender, a deliberate two-handed cast is *supposed* to
# look composed and controlled.
MAGIC_FAR = [
    _build(False, ((0.36, 0.30), (0.64, 0.30)), ((0.43, 0.94), (0.57, 0.94))),
    _build(False, ((0.32, 0.18), (0.68, 0.18)), ((0.42, 0.93), (0.58, 0.94)),
           lift=0.010),
    _build(False, ((0.38, 0.12), (0.62, 0.12)), ((0.42, 0.94), (0.58, 0.93)),
           lift=0.016),
    _build(False, ((0.34, 0.24), (0.66, 0.24)), ((0.43, 0.94), (0.57, 0.94))),
]

# Recoil: arms fly up, weight drives onto the back foot. Bending the elbows
# outward and up is what separates "hit" from "surrender". Feet stay inside
# leg reach - a stagger wide enough to over-extend gets clamped to a straight
# leg, which reads as stiff at exactly the moment it should read as buckling.
# FRONT-facing, not side. It was authored `side=True`, and that was the same
# defect as burning: the cores face the camera, so a profile skeleton asks for a
# turn the init image forbids, and the two signals fight. `got damage` declares
# view "front" in action_prompts.json, and scripts/validate-actions.py now fails
# the build when a declared view and its skeleton disagree - which is how this
# one was caught, having survived the burning fix that should have caught it.
#
# Every wrist is inside 2*ARM (0.28) of its shoulder and every ankle inside
# 2*LEG (0.38) of its hip, so no limb hits the straight-line clamp.
HURT = [
    _build(False, ((0.34, 0.22), (0.66, 0.22)), ((0.44, 0.94), (0.58, 0.93)),
           arm_bend=((0, -1), (0, -1))),
    _build(False, ((0.30, 0.18), (0.70, 0.20)), ((0.42, 0.92), (0.60, 0.94)),
           lift=0.010, arm_bend=((0, -1), (0, -1))),
    _build(False, ((0.32, 0.24), (0.68, 0.26)), ((0.43, 0.93), (0.59, 0.93)),
           arm_bend=((0, -1), (0, -1))),
    _build(False, ((0.36, 0.30), (0.64, 0.32)), ((0.45, 0.94), (0.57, 0.94)),
           arm_bend=((0, -1), (0, -1))),
]

# Burning: writhing in flames. This used to share HURT, and sharing was wrong
# twice over.
#
# HURT is authored `side=True`, which collapses both shoulders and both hips
# onto x=0.50 and gives the head one eye plus a trailing ear - the deliberate,
# heavily-commented profile cue in _head. The cores are drawn facing the
# camera, so conditioning one on HURT asks ControlNet for a profile while
# img2img holds a front view. They disagree, and what comes back is neither:
# a different character, in different clothes, at whatever angle won that
# frame. Front view is the only thing that matches the cores.
#
# The second problem is semantic. A recoil is one impact read four ways -
# symmetric, both arms doing the same thing at once. Burning is continuous and
# asymmetric: the body twists away from itself, one arm crosses the chest while
# the other flails wide. Symmetry is what made the old row read as "surrender"
# rather than "on fire".
#
# Beat structure: flinch up, twist one way, twist back the other, then hunch.
# No phase is a mirror of another - a mirrored pair reads as a metronome, which
# is the one thing a panic animation must not do.
#
# Every wrist here sits within 2*ARM (0.28) of its shoulder and every ankle
# within 2*LEG (0.38) of its hip, so no limb hits the straight-line clamp in
# _solve_joint. Elbows and knees are real bends, not degenerate midpoints. If
# you retune these, keep that invariant or the limb silently goes stiff.
BURN = [
    # Flinch: arms fly up and out, weight back, one knee already giving.
    _build(False, ((0.32, 0.20), (0.68, 0.22)), ((0.42, 0.93), (0.57, 0.90))),
    # Twist left: right arm whips across the chest, left flails wide.
    _build(False, ((0.52, 0.26), (0.76, 0.40)), ((0.40, 0.91), (0.56, 0.94)),
           lift=0.015),
    # Twist right: not the mirror of phase 1 - the arms swap roles but the
    # heights and the stagger do not match, so the loop never reads as a rock.
    _build(False, ((0.26, 0.38), (0.50, 0.22)), ((0.44, 0.94), (0.59, 0.91)),
           lift=0.008),
    # Hunch: hands clutch in at the torso, both knees buckle, body sinks.
    # Negative lift, because this is the beat where the character drops.
    _build(False, ((0.46, 0.44), (0.56, 0.46)), ((0.43, 0.89), (0.57, 0.89)),
           lift=-0.012),
]


def _mirror(cycle):
    """Flip a cycle horizontally, swapping left and right keypoints.

    A character walking left is a character walking right, mirrored - there is
    no reason to author the coordinates twice, and a hand-authored mirror would
    drift out of sync with its original on the first edit. Mirroring after the
    joints are solved keeps the solved elbows and knees correct for free.
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


# Matched against the action text in order, so the specific entries come first:
# "move right" has to win before a bare "move" would.
POSE_LIBRARY = [
    # Diagonals alias onto the horizontal cardinal, and the keys live in the
    # horizontal entries because matching is ordered and substring-based:
    # "move up right" CONTAINS "move up", so a separate diagonal entry placed
    # after these would never be reached, while one placed before would have to
    # stay before forever. Position here is the guarantee.
    #
    # Aliasing rather than authoring four more cycles is deliberate. A
    # side-facing sprite reads correctly moving diagonally; a front or back one
    # does not, so the horizontal cardinal is the right fallback and the
    # vertical component is the one to drop. These MUST agree with
    # action_prompts.json - scripts/validate-actions.py asserts that they do.
    (("move up right", "move down right", "up right", "down right",
      "northeast", "southeast",
      "move right", "walk right", "run right"), WALK_SIDE),
    (("move up left", "move down left", "up left", "down left",
      "northwest", "southwest",
      "move left", "walk left", "run left"), _mirror(WALK_SIDE)),
    (("move up", "walk up", "walk back"), WALK_FRONT),
    (("move down", "walk down", "walk front", "move", "walk", "run"), WALK_FRONT),
    # Weapon and magic variants BEFORE the generic attack entry: every one of
    # them contains the word "attack", so a bare ("attack",) placed first would
    # swallow all five and hand a sword swing to an archer. Close/distant magic
    # must likewise precede the generic magic keys.
    (("bow", "arrow", "archery"), BOW),
    (("sling", "slingshot"), SLING),
    (("close magic", "close magick", "melee magic"), MAGIC_CLOSE),
    (("distant magic", "distant magick", "ranged magic", "magic", "magick",
      "spell", "cast"), MAGIC_FAR),
    (("attack", "strike", "swing", "melee"), ATTACK),
    # Before HURT, and burning is no longer one of HURT's keys: "burning"
    # contains "burn", so whichever entry comes first wins outright.
    (("burning", "burn", "on fire"), BURN),
    (("damage", "hurt", "hit"), HURT),
    (("idle", "stand"), IDLE),
]


def is_side_view(cycle) -> bool:
    """True when a cycle is authored in profile rather than head-on.

    _build(side=True) collapses both shoulders onto x=0.50, so the two shoulder
    keypoints being identical is an exact structural marker - no threshold and
    no separate flag to keep in sync with the coordinates.

    This matters because img2img cannot rotate a character. A profile skeleton
    over a camera-facing core asks for a turn that the init image forbids, and
    what comes back is a front-facing character shuffling in place. Callers use
    this to swap in a side-facing init instead of fighting it.
    """
    if not cycle:
        return False
    kp = cycle[0]
    return kp.get(2) is not None and kp.get(2) == kp.get(5)


def faces_left(cycle) -> bool:
    """True when a profile cycle looks left. Meaningless for front views.

    _head(True) puts the nose at 0.50 + 0.09*face_dir, so a nose left of centre
    is a figure turned left.
    """
    nose = cycle[0].get(0) if cycle else None
    return bool(nose and nose[0] < 0.50)


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
