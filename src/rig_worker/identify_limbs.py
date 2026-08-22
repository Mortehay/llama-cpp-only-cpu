"""Can a skeleton's limb ends be labelled AUTOMATICALLY - which leaf is the left leg?

WHY THIS IS THE DECIDING TEST
compare_skeletons established that UniRig gives two different characters the
same FIGURE (3 branch points, 5 limb ends) but different joint counts along the
limbs. That kills a bone-name retarget map and leaves IK retargeting, which
drives limb ENDS rather than matching bones one-to-one. IK only helps if the
ends can be named without a human looking at each character - otherwise the
manual work simply moves one step downstream and the automation case still
fails.

So this labels the leaves from GEOMETRY ALONE and reports what it inferred. No
bone names are used: UniRig emits bone_0..bone_N, which carry no meaning.

METHOD
  up       - the axis with the largest spread across bone positions. Detected,
             never assumed: the FBX importer may rotate the rig relative to the
             source mesh, and hardcoding Y-up already caused a false verdict
             once in this project (see inspect_mesh.up_axis).
  lateral  - of the two remaining axes, the one across which the leaves spread
             more. A biped's arms and legs separate left-to-right far more than
             front-to-back, so this finds the shoulder-to-shoulder axis without
             knowing which way the character faces.
  head     - the highest leaf.
  legs     - the two lowest leaves.
  arms     - whatever is left, which for a 5-leaf figure is exactly two.
  L/R      - sign along `lateral`. Which sign means "left" is a convention, and
             it is only consistent across characters if the meshes share an
             orientation - which is checked by comparing two rigs, not assumed.

WHAT WOULD FALSIFY THIS
A leaf count other than 5; legs that are not clearly below the arms; or a pair
that does not separate cleanly along `lateral`. Each is reported rather than
silently smoothed over, because a confident-looking wrong label here would
propagate into every animation the conveyor produces.
"""
import sys


def load_bones(path):
    """[(name, head_xyz, tail_xyz, parent_or_None)] for the first armature."""
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    lower = path.lower()
    if lower.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=path)
    elif lower.endswith((".glb", ".gltf")):
        bpy.ops.import_scene.gltf(filepath=path)
    else:
        raise SystemExit(f"unsupported format: {path}")

    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if not arms:
        raise SystemExit(f"no armature in {path}")
    arm = arms[0]
    out = []
    for b in arm.data.bones:
        out.append((b.name,
                    tuple(arm.matrix_world @ b.head_local),
                    tuple(arm.matrix_world @ b.tail_local),
                    b.parent.name if b.parent else None))
    return out


def classify(bones):
    names = [b[0] for b in bones]
    parents = {b[0]: b[3] for b in bones}
    tails = {b[0]: b[2] for b in bones}

    has_child = set(p for p in parents.values() if p)
    leaves = [n for n in names if n not in has_child]

    pts = [t for t in tails.values()]
    spread = [max(p[i] for p in pts) - min(p[i] for p in pts) for i in range(3)]
    leaf_pts = [tails[n] for n in leaves]

    # SEARCH the six (up, lateral) orientations instead of guessing from extent.
    #
    # "up is the axis with the largest spread" is wrong for a T-pose, where the
    # arm span exceeds the height: on the box control it picked X (arm span
    # 1.860) over the true up Z (1.578) and mislabelled a textbook-correct
    # skeleton. The exported orientation also varies between files here - the
    # control comes back Z-up while the TripoSR-derived rigs come back Y-up - so
    # there is no fixed convention to hardcode either.
    #
    # Scoring each orientation against what makes a biped a biped is both more
    # robust and more honest: if NO orientation satisfies those constraints, the
    # skeleton genuinely is not a biped, and that is the answer rather than an
    # artefact of the axis guess.
    def score(up, lateral):
        """Lower is better. Sum of how badly the biped constraints are broken."""
        if len(leaves) != 5:
            return float("inf")
        ranked = sorted(leaves, key=lambda n: tails[n][up])
        legs, head = ranked[:2], ranked[-1]
        arms = [n for n in leaves if n not in legs and n != head]
        up_span = spread[up] or 1.0
        centre = (max(p[lateral] for p in leaf_pts)
                  + min(p[lateral] for p in leaf_pts)) / 2.0
        penalty = 0.0
        for pair in (legs, arms):
            a, b = pair
            # Pairs should sit at the same height...
            penalty += abs(tails[a][up] - tails[b][up]) / up_span
            # ...and straddle the centre line.
            la, lb = tails[a][lateral] - centre, tails[b][lateral] - centre
            if la * lb > 0:
                penalty += 1.0
            else:
                penalty += max(0.0, 0.25 - abs(la - lb) / up_span)
        # The head should stand clear above the arms, not sit among them.
        penalty += max(0.0, 0.15 - (tails[head][up]
                                    - max(tails[n][up] for n in arms)) / up_span)
        return penalty

    # The search must not be free to pick ANY axis as up. Given five points and
    # six orientations it will almost always find one that satisfies loose
    # biped constraints: on charB2 it chose the figure's SHORTEST axis (spread
    # 0.468) as up for a figure 2.199 tall, and declared a broken rig clean.
    #
    # A real biped's height is always comparable to its widest span - arms out
    # to the sides at most match it. So the up axis must be one of the dominant
    # extents. 0.6 admits the T-pose case (the box control is 1.578 tall against
    # a 1.860 arm span, a ratio of 0.85) while excluding the degenerate ones.
    max_spread = max(spread) or 1.0
    candidates = [i for i in range(3) if spread[i] >= 0.6 * max_spread]

    best = None
    for u in candidates:
        for lat in range(3):
            if lat == u:
                continue
            s = score(u, lat)
            if best is None or s < best[0]:
                best = (s, u, lat)
    _, up, lateral = best
    depth = next(i for i in range(3) if i not in (up, lateral))

    ranked = sorted(leaves, key=lambda n: tails[n][up])
    result = {
        "up": "XYZ"[up], "lateral": "XYZ"[lateral], "depth": "XYZ"[depth],
        "leaves": leaves, "n_leaves": len(leaves), "labels": {},
        "warnings": [],
    }
    if len(leaves) != 5:
        result["warnings"].append(
            f"{len(leaves)} limb ends, expected 5 (head + 2 arms + 2 legs) - "
            "the labelling below is not trustworthy")
        return result, tails, up, lateral

    legs = ranked[:2]
    head = ranked[-1]
    arms = [n for n in leaves if n not in legs and n != head]

    def lr(pair):
        a, b = pair
        if tails[a][lateral] <= tails[b][lateral]:
            return {"left": a, "right": b}
        return {"left": b, "right": a}

    result["labels"] = {
        "head": head,
        "leg": lr(legs),
        "arm": lr(arms),
    }

    # Sanity, reported rather than assumed.
    leg_top = max(tails[n][up] for n in legs)
    arm_bot = min(tails[n][up] for n in arms)
    if leg_top >= arm_bot:
        result["warnings"].append(
            f"legs are not clearly below arms (highest leg {leg_top:.3f} vs "
            f"lowest arm {arm_bot:.3f}) - the low/high split may be wrong")
    # A limb PAIR has two defining properties, and both must be checked. An
    # earlier version tested only "some lateral separation" and passed charB2,
    # whose two "arms" sat 1.14 apart in height (over half the figure) on the
    # SAME side - a confident, wrong labelling of a skeleton that simply has no
    # arm pair. Weak checks are worse than none here: they launder a bad rig.
    up_spread = spread[up] or 1.0
    for label, pair in (("legs", legs), ("arms", arms)):
        a, b = pair
        # 1. Roughly level. Left and right limbs of one figure sit at similar
        #    heights; a big gap means these two leaves are not a pair at all.
        dh = abs(tails[a][up] - tails[b][up]) / up_spread
        if dh > 0.25:
            result["warnings"].append(
                f"the two {label} are not level: {dh * 100:.0f}% of the "
                f"figure's height apart ({tails[a][up]:.3f} vs "
                f"{tails[b][up]:.3f}) - they are probably not a {label[:-1]} "
                "pair")
        # 2. On OPPOSITE sides. Straddling the centre line is what makes
        #    left/right meaningful; two leaves on one side cannot be labelled.
        la, lb = tails[a][lateral], tails[b][lateral]
        centre = (max(p[lateral] for p in leaf_pts)
                  + min(p[lateral] for p in leaf_pts)) / 2.0
        if (la - centre) * (lb - centre) > 0:
            result["warnings"].append(
                f"both {label} are on the SAME side of centre "
                f"({la:.3f}, {lb:.3f}, centre {centre:.3f}) - left/right is "
                "not recoverable from geometry")
        elif abs(la - lb) < 0.05 * up_spread:
            result["warnings"].append(
                f"the two {label} barely separate along {'XYZ'[lateral]} "
                f"({abs(la - lb):.4f}) - left/right assignment is a coin flip")
    return result, tails, up, lateral


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    codes = []
    for path in sys.argv[1:]:
        bones = load_bones(path)
        res, tails, up, lateral = classify(bones)
        print(f"\n=== {path} ===")
        print(f"bones {len(bones)}   limb ends {res['n_leaves']}   "
              f"up={res['up']} lateral={res['lateral']} depth={res['depth']}")
        if res["labels"]:
            lab = res["labels"]
            order = [("head", lab["head"]),
                     ("arm.L", lab["arm"]["left"]),
                     ("arm.R", lab["arm"]["right"]),
                     ("leg.L", lab["leg"]["left"]),
                     ("leg.R", lab["leg"]["right"])]
            print(f"  {'role':8s} {'bone':10s} "
                  f"{'up':>8s} {'lateral':>8s} {'depth':>8s}")
            d = "XYZ".index(res["depth"])
            for role, bone in order:
                t = tails[bone]
                print(f"  {role:8s} {bone:10s} {t[up]:8.3f} "
                      f"{t[lateral]:8.3f} {t[d]:8.3f}")
        for w in res["warnings"]:
            print(f"  WARNING: {w}")
        codes.append(0 if res["labels"] and not res["warnings"] else 1)

    print()
    if all(c == 0 for c in codes):
        print("Every limb end was labelled from geometry alone, with no "
              "warnings. An IK retarget can be authored against these roles "
              "once and applied to any character the same way.")
    else:
        print("At least one skeleton could not be labelled cleanly - see the "
              "warnings. Read them before trusting an IK retarget: a confident "
              "wrong label propagates into every animation.")
    return max(codes)


if __name__ == "__main__":
    raise SystemExit(main())
