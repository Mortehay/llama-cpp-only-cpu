"""Print one skeleton's bone hierarchy, for reading a rig rather than judging it.

compare_skeletons.py answers "do two match". This answers "what IS this" - which
is what you need when a verdict looks wrong and you want to see whether the
skeleton is a plausible humanoid or a degenerate chain. Bone POSITIONS are
included because a topology can look right while the joints sit nowhere near
the limbs they are supposed to drive.
"""
import sys

from compare_skeletons import load_armature


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]

    from identify_limbs import load_bones

    bones = load_bones(path)
    names = [b[0] for b in bones]
    parents = {b[0]: b[3] for b in bones}
    tails = {b[0]: b[2] for b in bones}

    children = {}
    for name, parent in parents.items():
        children.setdefault(parent, []).append(name)

    print(f"{path}: {len(names)} bones")

    def walk(name, depth=0):
        kids = children.get(name, [])
        marker = "" if kids else "  (leaf)"
        pos = tails.get(name)
        coords = (f"  [{pos[0]:6.3f} {pos[1]:6.3f} {pos[2]:6.3f}]"
                  if pos else "")
        print(f"{'  ' * depth}{name}{marker}{coords}")
        for k in sorted(kids):
            walk(k, depth + 1)

    for root in children.get(None, []):
        walk(root)

    # A humanoid should branch: a chain with no branching means the model saw a
    # blob rather than a figure with limbs.
    branch_points = sum(1 for n, k in children.items()
                        if n is not None and len(k) > 1)
    leaves = sum(1 for n in names if n not in children)
    print(f"\nbranch points: {branch_points}   leaves (limb ends): {leaves}")
    if branch_points == 0:
        print("WARNING: no branching - this is a straight chain, not a figure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
