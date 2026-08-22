"""Strip a fused ground/shadow patch from a core image before 3D reconstruction.

WHY THIS IS NEEDED FOR THE 3D PATH SPECIFICALLY
In 2D a shadow ellipse under the feet is cosmetic. In 3D it is geometry:
TripoSR reconstructs it as a disc FUSED to the soles, which inflates the depth
extent and - measured here on charB - drops silhouette IoU from 0.908 to 0.412
and makes UniRig emit a 6-bone straight chain instead of a branching humanoid
skeleton. A cosmetic 2D flaw becomes a structural 3D failure.

rembg does NOT remove it: it segments the shadow as part of the subject, which
is arguably correct for a photo and wrong for us. Neither of the existing
cleanups can see it either - remove_background only clears what is reachable
from the border, and _isolate_largest_sprite keeps the largest blob, of which
the patch is part. Hence strip_ground_patch, which detects by WIDTH.

Usage:  python clean_for_mesh.py in.png out.png
"""
import sys

from PIL import Image

from tasks import strip_ground_patch


def opaque_box(img):
    """Bounding box of pixels that are actually opaque, for a before/after read."""
    a = img.convert("RGBA").getchannel("A")
    return a.point(lambda v: 255 if v > 16 else 0).getbbox()


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    src, dst = sys.argv[1], sys.argv[2]

    img = Image.open(src).convert("RGBA")
    before = opaque_box(img)
    cleaned = strip_ground_patch(img)
    after = opaque_box(cleaned)
    cleaned.save(dst)

    def describe(box):
        if not box:
            return "empty"
        w, h = box[2] - box[0], box[3] - box[1]
        return f"{w}x{h} at {box[:2]}  aspect {h / w:.2f}"

    print(f"in   {src}")
    print(f"  before  {describe(before)}")
    print(f"  after   {describe(after)}")
    if before and after and before[3] != after[3]:
        print(f"  removed {before[3] - after[3]}px from the bottom")
    else:
        print("  NOTHING REMOVED - the patch was not detected by width. "
              "Check the image; a soft shadow wider than the stance should "
              "trip it, one narrower than width_ratio will not.")
    print(f"out  {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
