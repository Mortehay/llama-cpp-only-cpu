"""Do two characters get the SAME skeleton? That decides the whole conveyor.

THE QUESTION

UniRig predicts a skeleton per mesh rather than fitting a standard humanoid
template, and it does not claim Mixamo compatibility. Retargeting a stock
animation onto a non-standard skeleton is a solved problem in Blender - what is
NOT solved is whether the mapping has to be authored once or once per character:

  * Same bone count and hierarchy across characters -> author the retarget map
    ONCE -> the conveyor is automatic, which is the whole point.
  * A differently-shaped skeleton per mesh -> per-character manual work -> the
    automation argument collapses and pre-rigged source models become the
    honest fallback.

So this compares STRUCTURE, not geometry. Two characters of different build
should still yield the same bone TOPOLOGY if the model is template-like; joint
POSITIONS are expected to differ and are not the failure signal.

Run with Blender's Python (bpy is a dependency of this image):
    python compare_skeletons.py rigged_a.fbx rigged_b.fbx
"""
import sys


def load_armature(path):
    """Return (bone_names, parent_map) for the first armature in `path`."""
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    lower = path.lower()
    if lower.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=path)
    elif lower.endswith((".glb", ".gltf")):
        bpy.ops.import_scene.gltf(filepath=path)
    elif lower.endswith(".obj"):
        raise SystemExit(f"{path} is an OBJ - it carries no armature. "
                         "Rig it first and pass the FBX/GLB.")
    else:
        raise SystemExit(f"unsupported format: {path}")

    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if not arms:
        raise SystemExit(f"no armature found in {path} - rigging produced "
                         "geometry but no skeleton")
    arm = arms[0]
    names = [b.name for b in arm.data.bones]
    parents = {b.name: (b.parent.name if b.parent else None)
               for b in arm.data.bones}
    return names, parents


def limb_structure(parents):
    """(branch_points, leaves) - the figure's shape, ignoring joint resolution.

    Bone count and depth profile answer "is this the same rig". This answers the
    weaker but more useful question "is this the same FIGURE": how many places
    the skeleton splits, and how many limb ends it has. Two characters can score
    19-vs-10 bones and depth 6-vs-3 while both being a 5-limbed biped, which is
    exactly what the zombie and the upright zombie do here.

    It matters because the two failures have different fixes. Different joint
    COUNTS along matching limbs can be retargeted by IK - drive the 5 limb ends
    and let the solver distribute rotation over however many joints each
    character has. A different limb STRUCTURE cannot: there is no correspondence
    to drive.
    """
    children = {}
    for name, parent in parents.items():
        children.setdefault(parent, []).append(name)
    branch = sum(1 for n, k in children.items() if n is not None and len(k) > 1)
    leaves = sum(1 for n in parents if n not in children)
    return branch, leaves


def depth_profile(parents):
    """Bone counts per depth from the root.

    Compared instead of names because names may be generated per mesh while the
    TOPOLOGY is still identical - and topology is what a retarget map needs.
    """
    prof = {}
    for name in parents:
        d, cur = 0, name
        seen = set()
        while parents.get(cur) and cur not in seen:
            seen.add(cur)
            cur = parents[cur]
            d += 1
        prof[d] = prof.get(d, 0) + 1
    return dict(sorted(prof.items()))


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    a_path, b_path = sys.argv[1], sys.argv[2]

    a_names, a_par = load_armature(a_path)
    b_names, b_par = load_armature(b_path)

    a_prof, b_prof = depth_profile(a_par), depth_profile(b_par)

    print(f"{'':22s} {'A':>28s} {'B':>28s}")
    print(f"{'bones':22s} {len(a_names):>28d} {len(b_names):>28d}")
    print(f"{'depth profile':22s} {str(a_prof):>28s} {str(b_prof):>28s}")
    print(f"{'root':22s} "
          f"{[n for n in a_par if not a_par[n]][:1]!s:>28s} "
          f"{[n for n in b_par if not b_par[n]][:1]!s:>28s}")

    a_limbs, b_limbs = limb_structure(a_par), limb_structure(b_par)

    same_count = len(a_names) == len(b_names)
    same_names = a_names == b_names
    same_topo = a_prof == b_prof
    same_limbs = a_limbs == b_limbs

    print()
    print(f"same bone count:     {same_count}")
    print(f"same bone names:     {same_names}")
    print(f"same topology:       {same_topo}")
    print(f"same limb structure: {same_limbs}   "
          f"(A: {a_limbs[0]} branch pts / {a_limbs[1]} ends, "
          f"B: {b_limbs[0]} / {b_limbs[1]})")

    if not same_names:
        only_a = sorted(set(a_names) - set(b_names))[:6]
        only_b = sorted(set(b_names) - set(a_names))[:6]
        if only_a or only_b:
            print(f"  only in A: {only_a}")
            print(f"  only in B: {only_b}")

    print()
    if same_names and same_topo:
        print("VERDICT: template-like. One retarget map serves every character "
              "-> the conveyor can be automatic.")
        return 0
    if same_topo and same_count:
        print("VERDICT: same topology, different names. A retarget map keyed on "
              "HIERARCHY POSITION rather than name is authored once and still "
              "serves every character. Workable.")
        return 0
    if same_limbs:
        print("VERDICT: same FIGURE, different joint resolution. Both skeletons "
              f"have {a_limbs[0]} branch points and {a_limbs[1]} limb ends, so "
              "the characters correspond limb-for-limb, but the joint counts "
              f"along those limbs differ ({len(a_names)} bones vs "
              f"{len(b_names)}).")
        print()
        print("  A bone-NAME or hierarchy-POSITION retarget map will not "
              "transfer - the chains are different lengths, so there is no "
              "one-to-one bone correspondence to author.")
        print("  An IK retarget can: drive the limb ends as targets and let the "
              "solver distribute rotation over whatever joints each character "
              "has. That is authored once against the limb roles, not per "
              "character.")
        print()
        print("  NEXT TEST: whether the limb ends can be identified "
              "AUTOMATICALLY (which leaf is the left leg?). If that needs "
              "eyeballing per character, the automation argument still fails - "
              "it just fails one step later.")
        return 0
    print("VERDICT: skeletons differ structurally - not even the same figure "
          f"(A: {a_limbs[0]} branch points / {a_limbs[1]} limb ends; "
          f"B: {b_limbs[0]} / {b_limbs[1]}). A stock animation cannot be "
          "retargeted without per-character bone mapping, which is manual work "
          "per character - the automation case for the 3D conveyor fails here. "
          "Fallback: pre-rigged source models (see decisions/0004).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
