"""Is a generated mesh good enough to RIG? Not "is it pretty".

The 3D conveyor's value depends on the mesh surviving auto-rigging, and the
properties that decide that are structural, not aesthetic. Judging a mesh by
eye - or worse, by a screenshot - would repeat the mistake this project has
already made several times: measuring something adjacent to the question and
believing the number.

WHAT IS MEASURED, AND WHY EACH ONE MATTERS FOR RIGGING

  components      A character must be ONE connected shape. Fragments mean the
                  reconstruction broke up, and a rigger will weight only one
                  piece, leaving limbs behind when the skeleton moves.
  watertight      Manifold geometry. Skinning solvers assume a closed surface;
                  holes produce vertices no bone claims.
  aspect          Height/width of the projected silhouette. A humanoid is
                  roughly 2-3. Well under 2 means the reconstruction spread
                  sideways instead of standing up.
  silhouette IoU  Overlap between the mesh's front projection and the source
                  image's alpha. This is the honest test of "did it reconstruct
                  THIS character" - a plausible humanoid that does not match the
                  input is still a failure for a conveyor.
  degenerate      Zero-area faces. They survive export and then break normals
                  and weight transfer downstream.
"""
import argparse
import sys

import numpy as np
import trimesh
from PIL import Image


def _fit_mask(mask, res):
    """Fit a boolean mask into res x res, preserving aspect, bottom-centred.

    Both the reference alpha and the mesh projection go through this, so they
    are normalised identically. Anything that scales the two differently is
    measuring its own distortion.
    """
    h, w = mask.shape
    scale = min(res / w, res / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    small = np.array(Image.fromarray(mask.astype(np.uint8) * 255)
                     .resize((nw, nh), Image.Resampling.NEAREST)) > 127
    out = np.zeros((res, res), dtype=bool)
    x0 = (res - nw) // 2
    y0 = res - nh          # stand on the bottom edge, as sprites do
    out[y0:y0 + nh, x0:x0 + nw] = small
    return out


def up_axis(mesh):
    """Which axis the character stands along. DETECTED, never assumed.

    An earlier version hardcoded Y-up and reported this mesh as aspect 0.46 -
    "not humanoid" - when TripoSR had produced a Z-up mesh whose real aspect is
    2.16. The verdict was wrong on two of four criteria and would have condemned
    a working route. A standing figure's longest bbox extent IS its height, so
    detect it rather than trusting a convention.
    """
    ext = mesh.bounds[1] - mesh.bounds[0]
    return int(np.argmax(ext))


def silhouette_iou(mesh, image_path, res=256, up=1):
    """IoU between the mesh's front projection and the image's alpha.

    `up` selects the vertical axis. The horizontal one is chosen by TRYING BOTH
    remaining axes and keeping the better match; `silhouette_axis` in the report
    says which won.

    It used to assume the horizontal axis was simply the WIDER of the two.
    Best-of-two replaced that as cheap insurance: the axis is a nuisance
    parameter, since the question is whether the mesh depicts the character, not
    how TripoSR happened to orient it. Note it fixed nothing when introduced -
    every mesh measured here scores best on Y under either rule - so it is a
    guard, not a bug fix.

    PASS THE KEYED SOURCE IMAGE, NOT A PREPARED ONE. The reference silhouette is
    the image's ALPHA. mesh.py writes a *_input.png with the character
    composited onto an opaque grey background for TripoSR, and that image has no
    transparency at all, so the reference becomes the whole rectangle and IoU
    collapses. Measured: charB2 scored 0.383 against its prepared image and
    0.844 against its keyed source; charB, 0.412 against prepared and 0.895
    against keyed. Both meshes were fine, and both looked broken. `report`
    now refuses an image that is almost entirely opaque rather than returning
    that number.
    """
    img = Image.open(image_path).convert("RGBA")
    alpha = np.array(img)[:, :, 3] > 0

    # An image that is opaque nearly everywhere is not a keyed sprite - it is a
    # prepared/composited one, and its "silhouette" is the whole frame. Refuse
    # it: silently scoring against a full rectangle produces a low number that
    # reads as a broken mesh, which is exactly how two good meshes were
    # condemned here. 0.95 rather than 1.0 so a sprite that genuinely fills the
    # frame edge to edge still errors rather than sneaking through.
    if alpha.mean() > 0.95:
        raise ValueError(
            f"{image_path} is {alpha.mean() * 100:.0f}% opaque - it looks like "
            "a prepared image (character composited on a solid background), "
            "not a keyed source. Its alpha carries no silhouette. Pass the "
            "original core PNG with transparency instead."
        )

    ys, xs = np.where(alpha)
    if not len(ys):
        return None
    alpha = alpha[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    # Fit ASPECT-PRESERVING, not stretched to square. Resizing the crop to
    # res x res distorts a 195x423 character into a square while the mesh
    # projection below is scaled uniformly - comparing those two measures the
    # distortion, not the shape. That bug reported 0.284 for a mesh whose
    # aspect (2.16) actually matches the reference (2.17) almost exactly.
    ref = _fit_mask(alpha, res)

    def project(horiz):
        """Rasterise the mesh onto the (horiz, up) plane by binning vertices.

        Crude next to a real renderer, but it needs no GL context in a headless
        container and the question here is coverage, not shading.
        """
        v = mesh.vertices[:, [horiz, up]].copy()
        v -= v.min(axis=0)
        span = v.max()
        if span <= 0:
            return None
        # Rasterise at native aspect first, then hand it to the SAME _fit_mask
        # the reference goes through, so both are normalised identically.
        big = res * 2
        vi = (v / span * (big - 1)).astype(int)
        raw = np.zeros((big, big), dtype=bool)
        raw[big - 1 - vi[:, 1], vi[:, 0]] = True
        from scipy import ndimage
        raw = ndimage.binary_dilation(raw, np.ones((3, 3), bool), iterations=2)
        raw = ndimage.binary_fill_holes(raw)
        ys2, xs2 = np.where(raw)
        if not len(ys2):
            return None
        raw = raw[ys2.min():ys2.max() + 1, xs2.min():xs2.max() + 1]
        return _fit_mask(raw, res)

    # Score BOTH candidate horizontal axes and keep the better. See the
    # docstring: picking by extent silently compared a side view against a
    # front-facing reference on two of the three meshes tested here.
    best_score, best_axis = None, None
    for horiz in (i for i in range(3) if i != up):
        proj = project(horiz)
        if proj is None:
            continue
        union = np.logical_or(proj, ref).sum()
        if not union:
            continue
        score = float(np.logical_and(proj, ref).sum()) / float(union)
        if best_score is None or score > best_score:
            best_score, best_axis = score, horiz
    return best_score, best_axis


def report(mesh_path, image_path=None):
    mesh = trimesh.load(mesh_path, force="mesh")
    ext = mesh.bounds[1] - mesh.bounds[0]
    up = up_axis(mesh)
    others = [i for i in range(3) if i != up]
    height = float(ext[up])
    width = float(max(ext[others[0]], ext[others[1]]))
    aspect = height / width if width else 0.0

    areas = mesh.area_faces
    degenerate = int((areas <= 1e-12).sum())

    # Count only components big enough to matter. A marching-cubes surface
    # routinely carries specks - this mesh had one of 32 faces against 47,828 -
    # and reporting "2 disconnected pieces" for that reads as a broken limb when
    # it is a speck that any cleanup pass removes.
    comps = mesh.split(only_watertight=False)
    sizes = sorted((len(c.faces) for c in comps), reverse=True)
    significant = [n for n in sizes if n >= max(50, 0.01 * len(mesh.faces))]

    rows = {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "up_axis": "XYZ"[up],
        "components_total": len(sizes),
        "components_significant": len(significant),
        "component_sizes": sizes[:5],
        "watertight": mesh.is_watertight,
        "winding_consistent": mesh.is_winding_consistent,
        "degenerate_faces": degenerate,
        "bbox_extent": [round(float(x), 3) for x in ext],
        "aspect_h_over_w": round(aspect, 2),
    }
    ref_aspect = None
    if image_path:
        # The character's OWN aspect, from the source alpha. This is the right
        # thing to hold the mesh to - see the aspect check below.
        _a = np.array(Image.open(image_path).convert("RGBA"))[:, :, 3] > 0
        _ys, _xs = np.where(_a)
        if len(_ys):
            _h = _ys.max() - _ys.min() + 1
            _w = _xs.max() - _xs.min() + 1
            ref_aspect = float(_h) / float(_w)
            rows["source_aspect"] = round(ref_aspect, 2)

        iou, axis = silhouette_iou(mesh, image_path, up=up)
        rows["silhouette_iou"] = round(iou, 3) if iou is not None else "n/a"
        # Which plane actually matched. If this is not the axis you expect the
        # character to face along, the mesh is turned - the score alone hides
        # that, since the best-of-two search will happily score a rotated mesh
        # highly on the other axis.
        rows["silhouette_axis"] = "XYZ"[axis] if axis is not None else "n/a"

    width_col = max(len(k) for k in rows)
    for k, v in rows.items():
        print(f"{k:<{width_col}}  {v}")

    # A verdict, so the result is not left to interpretation.
    print()
    problems = []
    if len(significant) > 1:
        problems.append(f"{len(significant)} significant disconnected pieces "
                        f"{significant} - rigging will weight only one")
    # Hold the mesh to the CHARACTER's proportions, not to a fixed humanoid
    # range. The absolute 1.6-3 rule was calibrated on one slim zombie and
    # flagged charB at 1.48 as "not humanoid" - but charB is a stocky figure
    # with an arm held out, its source image measures 1.40, and its silhouette
    # IoU is 0.895. The mesh was right and the yardstick was wrong. What
    # actually matters is whether reconstruction CHANGED the proportions.
    if ref_aspect:
        drift = abs(aspect - ref_aspect) / ref_aspect
        if drift > 0.30:
            problems.append(
                f"aspect {aspect:.2f} differs from the source character's "
                f"{ref_aspect:.2f} by {drift * 100:.0f}% - reconstruction "
                "changed the proportions")
    elif aspect < 1.6:
        # No source image to compare against, so fall back to the absolute
        # rule - and say plainly that it is the weaker test.
        problems.append(f"aspect {aspect:.2f} is low for a humanoid (expect "
                        "2-3); pass --image to check against the character's "
                        "own proportions instead")
    if not mesh.is_watertight:
        problems.append("not watertight - skinning solvers assume closed")
    if image_path and isinstance(rows.get("silhouette_iou"), float) \
            and rows["silhouette_iou"] < 0.5:
        problems.append(f"silhouette IoU {rows['silhouette_iou']} - the mesh "
                        "does not match the input character")
    if problems:
        print("NOT RIG-READY:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("Structurally rig-ready. Whether UniRig produces a CONSISTENT "
              "skeleton across characters is a separate question.")
    return 0 if not problems else 1


def main():
    ap = argparse.ArgumentParser(description="Judge a mesh for rigging fitness")
    ap.add_argument("mesh")
    ap.add_argument("-i", "--image", default=None,
                    help="source image, to compute silhouette IoU")
    args = ap.parse_args()
    return report(args.mesh, args.image)


if __name__ == "__main__":
    raise SystemExit(main())
