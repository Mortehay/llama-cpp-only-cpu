#!/usr/bin/env python3
"""A job's 2D turnaround -> a 3D mesh -> a 48x64 eight-direction sprite sheet.

WHAT THIS IS FOR, AND WHAT IT IS NOT
It produces a STATIC eight-direction turnaround. It does not produce action
frames: that needs a rigged mesh, and the rigging gate failed 0 of 2 with a
clean synthetic control (decisions/0016). Do not read this module as a revival
of the 3D conveyor.

What it IS good for is the thing the 2D lane cannot do. The 2D turnaround's
front quarters collapse onto the front view (|s-se| 12.8) and its west rows are
flipped east rows with handedness swapped. Here a direction is a camera
position, so all eight are structurally correct and cannot drift.

WHY THE COLOUR COMES FROM THE VIEWS AND NOT FROM THE RECONSTRUCTOR
Measured 2026-10-24 (0016, "Colour projection"): a reconstructor EXTRAPOLATES
colour for surfaces it could not see, and that - not palette, and not geometry
- is what made cells unusable at 48x64. TripoSR's back geometry is fine when
normal-shaded; its back colour is not. The 2D lane already emits a turnaround,
so those surfaces ARE observed, just not by the reconstructor. Projecting the
observed views onto the mesh replaces extrapolated colour with measured
colour, and it is reconstructor-agnostic.

EVERY ORIENTATION HERE IS DERIVED, NEVER ASSUMED
Four orientation bugs were hit in one session before this rule went in: a +90
offset calibrated on one mesh, a 180 on another, Z-up assumed for a Y-up mesh,
and inverted winding after a mirror. Each presented as "the model produced
something bad" rather than "the camera was pointed wrong". TripoSR emits Z-up,
Hunyuan3D-2mv emits Y-up, and their azimuth origins differ too, so there is no
convention to hardcode. Up comes from the tallest extent; azimuth comes from
matching a rasterised silhouette against the source view, which IS the ground
truth for where that camera was.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np
import trimesh
from PIL import Image

# src/sprite_generator is mounted read-only at /sprite; pixelate and sheet are
# standalone (numpy + PIL only), so the whole chain runs in this one container.
for _c in ("/sprite", os.path.join(os.path.dirname(__file__), "..",
                                   "sprite_generator")):
    if os.path.isfile(os.path.join(_c, "pixelate.py")):
        sys.path.insert(0, _c)
        break

import pixelate  # noqa: E402
import sheet as sheet_mod  # noqa: E402

log = logging.getLogger("turnaround")

DIRECTIONS = ["s", "se", "e", "ne", "n", "nw", "w", "sw"]

# Views fed to a MULTIVIEW reconstructor, in the mirrored frame. dir_w is
# deliberately absent: measured across 11 jobs, w is a mirror-ish re-render of
# e in most (|e-mirror(w)| 20-50) but in two it is a near-DUPLICATE of e facing
# the same way (|e-w| 7.02), the documented left-side failure. Mirroring the
# whole problem instead puts the one genuinely rendered profile into the slot
# the model expects, with one handedness across all three inputs.
RECON_VIEWS = [("front", "dir_s.png", True),
               ("left", "dir_e.png", False),
               ("back", "dir_n.png", True)]

# Views used only as COLOUR sources, in the true frame. The quarters are
# included here but never for reconstruction: colour projection does not care
# that se collapses toward s, because it samples whatever the pixels show, and
# more cameras means fewer surfaces falling back to the median fill.
COLOUR_VIEWS = [("front", "dir_s.png", False),
                ("left", "dir_e.png", True),
                ("back", "dir_n.png", False),
                ("q1", "dir_se.png", True),
                ("q2", "dir_ne.png", True)]


# --- view preparation ---------------------------------------------------

def _key(path: str, tolerance: int, engine: str):
    """Background removal. Flood fill by default; BiRefNet on request.

    The flood fill is the default because this lane's whole value is that it
    never asks for the card, and BiRefNet in this codebase is CUDA-only
    (`cutout.enabled` requires it and `cutout.mask` places the model on cuda).
    But the fill cannot clear an ENCLOSED backdrop, which is what produced the
    wolf medal, so `--engine birefnet` is offered for sources that need it and
    accepts the GPU cost knowingly.
    """
    if engine == "birefnet":
        import cutout
        if cutout.weights_dir() is None:
            raise SystemExit(
                "--engine birefnet: no BiRefNet weights under "
                f"{cutout.MODELS_DIR}. `make provision` fetches them.")
        return cutout.cut_out(Image.open(path))
    return pixelate.key_background(Image.open(path), tolerance=tolerance)


def _keyed(path: str, size: int, tolerance: int, engine: str = "floodfill"):
    """Background-removed, bbox-cropped, centred on a square canvas.

    Raw dir_*.png carry an opaque backdrop (~69%), and consistent framing
    matters as much as keying: these views are NOT uniformly framed - a
    close-up `dir_w` cut off at the knees beside a full-body `dir_e` is a
    documented failure mode - and anything downstream reads inconsistent scale
    as inconsistent geometry.
    """
    im = _key(path, tolerance, engine)
    a = np.asarray(im)[..., 3] > 8
    ys, xs = np.where(a)
    if not len(ys):
        raise SystemExit(f"{path}: keyed to nothing at tolerance {tolerance}")
    im = im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    pad = 0.92
    h = int(size * pad)
    w = max(1, round(im.width * h / im.height))
    if w > int(size * pad):
        w = int(size * pad)
        h = max(1, round(im.height * w / im.width))
    im = im.resize((w, h), Image.LANCZOS)
    sq = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    sq.paste(im, ((size - w) // 2, (size - h) // 2))
    return sq


def stage_views(a):
    plan = RECON_VIEWS if a.frame == "mirrored" else COLOUR_VIEWS
    os.makedirs(a.out, exist_ok=True)
    made, clears = [], {}
    for name, src, mirror in plan:
        p = os.path.join(a.job, src)
        if not os.path.isfile(p):
            log.warning("missing %s - skipping", p)
            continue
        im = _keyed(p, a.size, a.tolerance, engine=a.engine)
        if mirror:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        im.save(os.path.join(a.out, f"{name}.png"))
        clear = 100.0 * (np.asarray(im)[..., 3] == 0).mean()
        clears[name] = clear
        made.append(name)
        print(f"  {name:5s} <- {src:11s} mirror={int(mirror)} "
              f"transparent={clear:5.1f}%")
    if not made:
        raise SystemExit(f"no usable views in {a.job}")
    _gate_views(clears, a)
    print(f"wrote {len(made)} view(s) to {a.out} (frame={a.frame})")


def _gate_views(clears: dict, a) -> None:
    """Refuse a source whose background did not come off. REFUSE, not warn.

    THE WOLF MEDAL, 2026-10-24. Job f841ec70's turnaround renders the subject
    inside a cream disc with a dark rim - the editor's own vignette, present
    in the SOURCE, not added here. `key_background` is a corner flood fill and
    structurally cannot clear a region the rim encloses, so TripoSR was handed
    a wolf-inside-a-disc and faithfully reconstructed the disc: eight views of
    a medal with a wolf on it.

    That is ADR 0004's core-82 failure exactly ("the character was fine; the
    border destroyed it"), and it had TWO warnings it did not act on - four
    views at ~37% transparent against one at 68%, and a front/back separation
    of 30 degrees. Both printed. Neither stopped anything, and three minutes
    of CPU produced a plate.

    The SPREAD is the stronger signal. Views of one subject at one framing
    should clear similar fractions; one view keying cleanly while the rest do
    not means the fill reached the border on that one alone.
    """
    if not clears:
        return
    lo, hi = min(clears.values()), max(clears.values())
    worst = min(clears, key=clears.get)
    problems = []
    if lo < a.min_clear:
        problems.append(f"{worst} cleared only {lo:.1f}% (need "
                        f"{a.min_clear:.0f}%)")
    if hi - lo > a.max_spread:
        problems.append(f"views disagree by {hi - lo:.1f} points "
                        f"({lo:.1f}-{hi:.1f}%, max {a.max_spread:.0f})")
    if not problems:
        return
    cause = ("The background did not come off. The usual cause is an ENCLOSED "
             "backdrop - a vignette, disc or frame whose rim reaches the "
             "image border - which a corner flood fill cannot clear by "
             "construction. Reconstructing from this produces geometry of "
             "the BACKDROP, not the subject (ADR 0004, core 82).")
    if a.engine == "birefnet":
        # Do not recommend the engine already in use. Measured on the wolf
        # medal: BiRefNet cleared the disc on 3 of 5 views and kept the whole
        # medallion as the subject on the other 2 (36.8/64.8/65.0/35.6/73.3%),
        # making the SPREAD worse than the flood fill's. A segmenter asked
        # "what is salient here" can reasonably answer "the medal", and no
        # keying choice makes five views agree about that.
        fix = ("BiRefNet was already used and disagreed with itself across "
               "views, so this source cannot be rescued by keying. Regenerate "
               "the turnaround on a plain field - the vignette is in the 2D "
               "output, so every consumer of it inherits the disc. --no-gate "
               "proceeds anyway and gets a plate.")
    else:
        fix = ("Options: --engine birefnet (needs CUDA and the BiRefNet "
               "weights; it clears enclosed regions, which is why the sprite "
               "path uses it), a source job whose turnaround is on a plain "
               "field, or --no-gate to proceed anyway and get a plate.")
    msg = "; ".join(problems) + ".\n" + cause + "\n" + fix
    if a.gate:
        raise SystemExit("REFUSED: " + msg)
    print("WARNING (gate disabled): " + msg)


# --- geometry helpers ---------------------------------------------------

def _up_axis(v):
    """Fallback only: the tallest extent. Prefer `_derive_pose`.

    True for a standing biped and FALSE for anything longer than it is tall -
    a quadruped, a prop lying down, or a reconstruction that came out as a
    disc. `_derive_pose` searches instead of guessing, and is what the stages
    use when views are available; this remains for the no-views case.
    """
    return int(np.argmax(np.ptp(v, axis=0)))


def _derive_pose(v, views, res, sweep):
    """Search (up axis, azimuth) jointly against the source views.

    WHY THIS REPLACES A HEURISTIC
    Up was `argmax(ptp)` - the tallest extent. That is a guess, correct for a
    standing humanoid and wrong for a quadruped (a wolf is longer than it is
    tall) or for a reconstruction that came out as a plate. It cost a session
    to learn that orientation guesses fail silently and look like bad models.

    The source views are upright by construction and they are the ground
    truth for where the cameras were, so BOTH unknowns come from the same
    match. 3 axes x 36 azimuths is 108 silhouettes - cheap, and it makes the
    quadruped case correct by construction rather than by luck.

    Returns (up, {view: (azimuth, iou)}).
    """
    refs = {n: _fit_mask(np.asarray(im)[..., 3] > 8, res)
            for n, im in views.items()}
    cand = list(range(0, 360, sweep))

    # THE VIEWS ARE NOT INDEPENDENT. They come from one turnaround, so their
    # RELATIVE angles are known - front, left and back are 0, 90 and 180 apart,
    # and the quarters sit at 45 and 135. Only the global offset and the
    # handedness are unknown.
    #
    # Searching each view's azimuth freely let them land in geometrically
    # impossible arrangements: on the wolf, front 0 and back 260 - 100 degrees
    # apart - because a quadruped's front and back silhouettes are both narrow
    # and the matcher cannot tell them apart. Three free parameters invited an
    # answer that cannot exist. One free parameter cannot produce it.
    OFFSETS = {"front": 0, "left": 90, "back": 180, "q1": 45, "q2": 135}
    best = (None, None, None, -1.0)          # up, offset, hand, score
    for up in range(3):
        a, b = [i for i in range(3) if i != up]
        sil = {d: _silhouette(v, up, a, b, d, res) for d in cand}
        for hand in (1, -1):
            for off in cand:
                total, n_used = 0.0, 0
                for n, ref in refs.items():
                    if ref is None or n not in OFFSETS:
                        continue
                    d = (off + hand * OFFSETS[n]) % 360
                    d = min(cand, key=lambda c: min(abs(c - d),
                                                    360 - abs(c - d)))
                    s = sil[d]
                    if s is None:
                        continue
                    u = np.logical_or(s, ref).sum()
                    if u:
                        total += np.logical_and(s, ref).sum() / u
                        n_used += 1
                if n_used and (score := total / n_used) > best[3]:
                    best = (up, off, hand, score)
    best_up, off, hand, best_score = best
    if best_up is None:
        raise SystemExit("could not match any view against the mesh")
    best_assign = {}
    for n, ref in refs.items():
        if n not in OFFSETS:
            continue
        d = (off + hand * OFFSETS[n]) % 360
        d = min(cand, key=lambda c: min(abs(c - d), 360 - abs(c - d)))
        a, b = [i for i in range(3) if i != best_up]
        s = _silhouette(v, best_up, a, b, d, res)
        iou = 0.0
        if s is not None and ref is not None:
            u = np.logical_or(s, ref).sum()
            iou = float(np.logical_and(s, ref).sum() / u) if u else 0.0
        best_assign[n] = (d, iou)
    print(f"  up axis derived as {'xyz'[best_up]}, offset {off} deg, "
          f"handedness {'+' if hand > 0 else '-'} (mean IoU {best_score:.3f})")
    for n, (d, iou) in sorted(best_assign.items()):
        print(f"  {n:5s} -> azimuth {d:3d} deg (IoU {iou:.3f})")
    _report_distinct(best_assign)
    _gate_separation(best_assign)
    return best_up, best_assign


def _project(v, up, a, b, deg, elev=0.0):
    """Orthographic projection: azimuth about `up`, then camera elevation.

    ELEVATION EXISTS BECAUSE OF THE WOLF. At 0 degrees - a dead-level camera -
    a quadruped seen along its own length shows no body at all: head, chest and
    two front legs stacked vertically, which reads as a figure standing upright.
    The mesh is correct; the camera is wrong for the subject. From above, a
    wolf facing the camera still shows its back, which is why every reference
    sheet this project was built against (Diablo, Fallout, Age of Empires) uses
    a raised three-quarter camera.

    Default stays 0 because the 2D lane's sheets are eye-level by deliberate
    choice (0005, "Final character, eye-level"), and a humanoid reads fine
    there - the mage fills 89% of its cell at elevation 0.
    """
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    x = v[:, a] * c - v[:, b] * s
    y = v[:, up]
    d = v[:, a] * s + v[:, b] * c
    if elev:
        e = np.radians(elev)
        ce, se = np.cos(e), np.sin(e)
        # Tilt the scene back about the screen-x axis: the camera rises, the
        # ground plane opens out, and what was hidden behind the subject comes
        # into view above it.
        y, d = y * ce + d * se, -y * se + d * ce
    return x, y, d


def _fit_mask(m, res):
    ys, xs = np.where(m)
    if not len(ys):
        return None
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    im = Image.fromarray((m * 255).astype(np.uint8))
    h, w = res, max(1, round(im.width * res / im.height))
    if w > res:
        w, h = res, max(1, round(im.height * res / im.width))
    im = im.resize((w, h), Image.NEAREST)
    out = np.zeros((res, res), bool)
    y0, x0 = (res - h) // 2, (res - w) // 2
    out[y0:y0 + h, x0:x0 + w] = np.asarray(im) > 127
    return out


def _silhouette(v, up, a, b, deg, res):
    x, y, _ = _project(v, up, a, b, deg)
    sx, sy = x - x.min(), y - y.min()
    span = max(float(np.ptp(x)), float(np.ptp(y)))
    if span <= 0:
        return None
    big = res * 2
    xi = (sx / span * (big - 1)).astype(int)
    yi = (sy / span * (big - 1)).astype(int)
    m = np.zeros((big, big), bool)
    m[big - 1 - yi, xi] = True
    from scipy import ndimage
    m = ndimage.binary_dilation(m, np.ones((3, 3), bool), iterations=2)
    return _fit_mask(ndimage.binary_fill_holes(m), res)


def _report_distinct(out: dict, min_sep: int = 30) -> None:
    """How many genuinely different angles did the views resolve to?

    REPORTS, never refuses. Duplicates are normal and usually harmless: on the
    mage `q1` landed on `back`'s azimuth and `q2` on `left`'s, and that was the
    best run measured - 94.3% colour coverage - because the remaining views
    still covered the hemisphere between them.

    It exists because the WOLF collapsed `front` and `left` onto 260 degrees:
    the 2D turnaround gives a quadruped two profiles rather than a front and a
    side, so one of three primary views was redundant. That cost nothing
    measurable (94% coverage, the quarters filled in), but it was only
    discoverable by reading a log closely, which is how a real version of this
    would get missed.

    A collapse to ONE angle is the degenerate case, and `_gate_separation`
    already refuses that - a mesh whose front and back look alike is a disc.
    """
    angles = sorted({d for d, _ in out.values() if d is not None})
    if not angles:
        return
    groups: list[list[int]] = []
    for d in angles:
        for g in groups:
            if min(abs(d - x) % 360 for x in g) <= min_sep or \
                    min(360 - (abs(d - x) % 360) for x in g) <= min_sep:
                g.append(d)
                break
        else:
            groups.append([d])
    dupes = {}
    for name, (d, _iou) in out.items():
        if d is None:
            continue
        dupes.setdefault(d, []).append(name)
    shared = {d: ns for d, ns in dupes.items() if len(ns) > 1}
    print(f"  {len(out)} view(s) -> {len(groups)} distinct angle(s) "
          f"(>{min_sep} deg apart)")
    for d, ns in sorted(shared.items()):
        print(f"    NOTE: {', '.join(ns)} all resolved to {d} deg - that view "
              f"adds no new aspect, only more samples of one")


def _gate_separation(out: dict) -> None:
    """Front and back must resolve to roughly opposite azimuths."""
    if "front" in out and "back" in out and None not in (out["front"][0],
                                                         out["back"][0]):
        gap = abs(out["front"][0] - out["back"][0]) % 360
        gap = min(gap, 360 - gap)
        # Catches a gross mis-assignment. It does NOT catch a clean front/back
        # swap - silhouette IoU is front/back ambiguous for a robed figure, and
        # the giveaway there is a face on the back of the head. Look.
        ok = abs(gap - 180) <= 35
        print(f"  front/back separation {gap} deg"
              + ("" if ok else "   <-- SUSPECT"))
        if not ok:
            # RAISES now. On the wolf medal this printed "30 deg <-- SUSPECT"
            # and the job carried on to render eight views of a plate. A
            # rotationally symmetric reconstruction - which a disc is - cannot
            # tell its front from its back, so a collapsed separation means
            # the thing reconstructed is not the subject.
            raise SystemExit(
                f"REFUSED: front and back resolved to azimuths {gap} degrees "
                f"apart; they must be roughly opposite. The reconstruction is "
                f"probably rotationally symmetric - a disc, plate or slab "
                f"rather than a figure - which happens when an enclosed "
                f"backdrop survives keying. Check the view stage's "
                f"transparency numbers, and see ADR 0004 core 82.")


# --- colour -------------------------------------------------------------

def _cut_pedestal(mesh, up: int, band: float, ratio: float):
    """Drop a reconstructed ground plate fused to the subject's feet.

    Hunyuan3D-2mv reconstructs the ground the subject stands on - a flat slab
    under the wolf, visible as a grey plinth in every cell. Component removal
    cannot help: it is FUSED, so it is one component with the animal.

    This is the 3D analogue of pixelate.strip_ground_patch's width rule, which
    works in 2D for the same shape of defect. A plinth is much wider than the
    legs standing on it, so the horizontal cross-section jumps sharply at the
    top of the slab. Cut there.

    `cut_below.py` is NOT this: it zeroes alpha below a row read off a width
    profile BY HAND, and its own docstring says hand-picking does not
    generalise and is not meant to.
    """
    v = np.asarray(mesh.vertices, float)
    lo, hi = float(v[:, up].min()), float(v[:, up].max())
    height = hi - lo
    if height <= 0:
        return mesh, 0.0
    # Measure the footprint in horizontal slices through the bottom `band`.
    edges = np.linspace(lo, lo + band * height, 13)
    ax, bx = [i for i in range(3) if i != up]
    areas = []
    for i in range(len(edges) - 1):
        m = (v[:, up] >= edges[i]) & (v[:, up] < edges[i + 1])
        if m.sum() < 8:
            areas.append(0.0)
            continue
        areas.append(float(np.ptp(v[m, ax]) * np.ptp(v[m, bx])))
    areas = np.asarray(areas)
    # THE REFERENCE IS THE SHINS, NOT THE BAND. Measuring the band against its
    # own median cannot work: if the plinth fills most of the band then the
    # median IS the plinth and nothing exceeds it. CLAUDE.md records the same
    # correction for strip_ground_patch - "that compared the patch against the
    # body median, so a wide stance looked like ground; the shin reference
    # moves with the legs".
    ref_edges = np.linspace(lo + band * height, lo + 2.2 * band * height, 9)
    refs = []
    for i in range(len(ref_edges) - 1):
        m = (v[:, up] >= ref_edges[i]) & (v[:, up] < ref_edges[i + 1])
        if m.sum() >= 8:
            refs.append(float(np.ptp(v[m, ax]) * np.ptp(v[m, bx])))
    ref = float(np.median(refs)) if refs else 0.0
    if ref <= 0:
        return mesh, 0.0
    # The slab is the lowest run whose footprint exceeds `ratio` x the median
    # of the band. Cut at the top of that run, not at its bottom, or the
    # plinth's upper surface survives as a disc under the feet.
    fat = [i for i, x in enumerate(areas) if x > ratio * ref]
    if not fat or fat[0] != 0:
        return mesh, 0.0           # nothing fat at the very bottom: no plinth
    top = 0
    for i in fat:
        if i == top:
            top = i + 1
    cut = edges[top]
    keep = v[:, up] >= cut
    fmask = keep[np.asarray(mesh.faces)].all(axis=1)
    if not fmask.any():
        return mesh, 0.0
    out = mesh.submesh([np.where(fmask)[0]], append=True)
    return out, 100.0 * (cut - lo) / height


def stage_colour(a):
    mesh = trimesh.load(a.mesh, force="mesh")
    if a.repair:
        # Hunyuan3D-2mv output is RAW: not watertight, and carrying stray
        # components that render as a slab across the sprite. TripoSR's path
        # already does this, which is most of why its coloured result looked
        # cleaner in the 0016 comparison.
        before = len(mesh.faces)
        parts = mesh.split(only_watertight=False)
        if len(parts) > 1:
            big = max(parts, key=lambda m: len(m.faces))
            keep = [m for m in parts
                    if len(m.faces) >= a.min_component * len(big.faces)]
            mesh = trimesh.util.concatenate(keep) if len(keep) > 1 else big
            print(f"  dropped {len(parts) - len(keep)} stray component(s), "
                  f"{before - len(mesh.faces)} faces")
    v = np.asarray(mesh.vertices, float)
    print(f"{os.path.basename(a.mesh)}: {len(v)} verts")

    views = {}
    for n in ("front", "left", "back", "q1", "q2"):
        p = os.path.join(a.views, f"{n}.png")
        if os.path.isfile(p):
            views[n] = Image.open(p).convert("RGBA")
    if not views:
        raise SystemExit(f"no views in {a.views}")
    up, assign = _derive_pose(v, views, a.res, a.sweep)

    # Cut the plinth AFTER the pose is known, then derive again.
    #
    # The order was the other way round and it silently did nothing: it used
    # the `_up_axis` heuristic, which for a quadruped returns nose-to-tail -
    # measured on this wolf, up=x with a footprint profile of tiny/wide/tiny,
    # the shape of an animal measured along its length. "Below" then meant
    # "behind the nose" and no plinth could be found.
    #
    # Cutting first was meant to stop the plinth distorting the silhouettes
    # the azimuth match uses. That is a real concern, so the pose is derived
    # TWICE: once to learn which way is up, and again on the cut mesh so the
    # azimuths are matched against the shape that will actually be rendered.
    # A few seconds against a 160s job.
    if a.cut_pedestal:
        pre = len(mesh.faces)
        mesh, frac = _cut_pedestal(mesh, up, a.pedestal_band,
                                   a.pedestal_ratio)
        if frac > 0:
            print(f"  cut a pedestal: bottom {frac:.1f}% of the height, "
                  f"{pre - len(mesh.faces)} faces - re-deriving the pose")
            v = np.asarray(mesh.vertices, float)
            up, assign = _derive_pose(v, views, a.res, a.sweep)

    ax, bx = [i for i in range(3) if i != up]

    vn = np.asarray(mesh.vertex_normals, float)
    acc = np.zeros((len(v), 3))
    wsum = np.zeros(len(v))
    for n, (deg, _iou) in assign.items():
        if deg is None:
            continue
        arr = np.asarray(views[n]).astype(float)
        vis = arr[..., 3] > 8
        ys, xs = np.where(vis)
        if not len(ys):
            continue
        x, y, _ = _project(v, up, ax, bx, deg)
        # Map the mesh's projected bbox onto the view's FIGURE bbox: the views
        # were cropped to alpha and re-fitted, so raw image coordinates would
        # sample the padding.
        sx = (x - x.min()) / max(float(np.ptp(x)), 1e-9)
        sy = (y - y.min()) / max(float(np.ptp(y)), 1e-9)
        xi = np.clip((xs.min() + sx * (xs.max() - xs.min())).astype(int),
                     0, arr.shape[1] - 1)
        yi = np.clip((ys.max() - sy * (ys.max() - ys.min())).astype(int),
                     0, arr.shape[0] - 1)
        cam = np.zeros(3)
        cam[ax], cam[bx] = -np.sin(np.radians(deg)), -np.cos(np.radians(deg))
        w = np.clip(vn @ cam, 0, 1) ** 2 * vis[yi, xi]
        acc += arr[yi, xi, :3] * w[:, None]
        wsum += w
        print(f"  sampled {n}: {int((w > 0).sum())}/{len(v)} verts")

    hit = wsum > 1e-6
    col = np.full((len(v), 4), 255, np.uint8)
    col[hit, :3] = np.clip(acc[hit] / wsum[hit, None], 0, 255).astype(np.uint8)
    if hit.any() and (~hit).any():
        # Median, never white: unobserved pockets (under the arms, between the
        # legs) would otherwise read as holes punched in the sprite.
        col[~hit, :3] = np.median(col[hit, :3], axis=0).astype(np.uint8)
    print(f"  coloured {int(hit.sum())}/{len(v)} from views, "
          f"{int((~hit).sum())} median-filled")
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=col)
    mesh.export(a.out)
    print(f"wrote {a.out} ({len(np.unique(col[:, :3], axis=0))} distinct RGB)")


# --- render -------------------------------------------------------------

def _rasterise(v, f, col, up, ax, bx, deg, res, elev=0.0):
    x, y, d = _project(v, up, ax, bx, deg, elev)
    span = max(np.ptp(v[:, 0]), np.ptp(v[:, 1]), np.ptp(v[:, 2]))
    if span <= 0:
        raise SystemExit("degenerate mesh")
    # One span for every azimuth AND every elevation, so the subject keeps a
    # constant size as it turns. Taken from the 3D extent rather than the
    # projected one for exactly that reason.
    pad = 0.06
    px = (((x - (x.max() + x.min()) / 2) / span * (1 - pad) + 0.5) * (res - 1))
    py = ((0.5 - (y - (y.max() + y.min()) / 2) / span * (1 - pad)) * (res - 1))

    img = np.zeros((res, res, 3))
    zbuf = np.full((res, res), np.inf)
    hit = np.zeros((res, res), bool)
    fx, fy, fd = px[f], py[f], d[f]
    if col is None:
        p = np.stack([x, y, d], axis=1)[f]
        n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        n /= np.where(ln == 0, 1, ln)
        light = np.array([-0.3, 0.45, -1.0])
        light /= np.linalg.norm(light)
        # Two-sided: winding is not trustworthy after a mirror, and a one-sided
        # clamp renders the whole figure at the ambient floor, which looks like
        # a black mesh rather than a flipped normal.
        fcol = np.repeat((0.25 + 0.75 * np.abs(n @ light))[:, None], 3, axis=1)
    else:
        fcol = col[f].mean(axis=1)

    for i in np.argsort(fd.mean(axis=1)):
        axx, bxx, cxx = fx[i]
        ayy, byy, cyy = fy[i]
        x0 = max(int(np.floor(min(axx, bxx, cxx))), 0)
        x1 = min(int(np.ceil(max(axx, bxx, cxx))), res - 1)
        y0 = max(int(np.floor(min(ayy, byy, cyy))), 0)
        y1 = min(int(np.ceil(max(ayy, byy, cyy))), res - 1)
        if x1 < x0 or y1 < y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        det = (byy - cyy) * (axx - cxx) + (cxx - bxx) * (ayy - cyy)
        if abs(det) < 1e-12:
            mask = np.ones(gx.shape, bool)
            depth = np.full(gx.shape, fd[i].mean())
        else:
            l1 = ((byy - cyy) * (gx - cxx) + (cxx - bxx) * (gy - cyy)) / det
            l2 = ((cyy - ayy) * (gx - cxx) + (axx - cxx) * (gy - cyy)) / det
            l3 = 1.0 - l1 - l2
            mask = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
            if not mask.any():
                continue
            depth = l1 * fd[i][0] + l2 * fd[i][1] + l3 * fd[i][2]
        sub = zbuf[y0:y1 + 1, x0:x1 + 1]
        closer = mask & (depth < sub)
        if not closer.any():
            continue
        sub[closer] = depth[closer]
        img[y0:y1 + 1, x0:x1 + 1][closer] = fcol[i]
        hit[y0:y1 + 1, x0:x1 + 1] |= closer

    out = np.zeros((res, res, 4), np.uint8)
    out[..., :3] = np.clip(img * 255, 0, 255).astype(np.uint8)
    out[..., 3] = np.where(hit, 255, 0)
    return out


def stage_render(a):
    mesh = trimesh.load(a.mesh, force="mesh")
    v = np.asarray(mesh.vertices, float)
    f = np.asarray(mesh.faces, int)
    vc = getattr(mesh.visual, "vertex_colors", None)
    col = None
    if vc is not None and len(vc) == len(v):
        rgb = np.asarray(vc)[:, :3]
        # Degenerate colour counts as NO colour: a shape-only reconstructor
        # still writes a flat vertex_colors array, and taking it at face value
        # renders a silhouette rather than a shape.
        if len(np.unique(rgb, axis=0)) > 4:
            col = rgb.astype(float) / 255.0
    print(f"{os.path.basename(a.mesh)}: {len(v)} verts, "
          f"{'coloured' if col is not None else 'normal-shaded'}, "
          f"elevation {a.elevation:g} deg")

    # The FRONT azimuth is derived, so the eight emitted directions are
    # labelled against the source view rather than against a guess.
    # Both unknowns from the same match when views are available. Without
    # them, fall back to the tallest-extent guess and say so - a caller that
    # renders without views is asking for an unverified orientation.
    front = 0
    views = {}
    for n in ("front", "left", "back"):
        fp = os.path.join(a.views, f"{n}.png") if a.views else None
        if fp and os.path.isfile(fp):
            views[n] = Image.open(fp).convert("RGBA")
    if views:
        up, assign = _derive_pose(v, views, 192, a.sweep)
        if assign.get("front", (None,))[0] is not None:
            front = assign["front"][0]
    else:
        up = _up_axis(v)
        print(f"  no views given - up GUESSED as {'xyz'[up]} from the tallest "
              f"extent, azimuth 0. Pass --views to derive both.")
    ax, bx = [i for i in range(3) if i != up]

    os.makedirs(a.out, exist_ok=True)
    for i, d in enumerate(DIRECTIONS):
        deg = (front + i * 45) % 360
        img = _rasterise(v, f, col, up, ax, bx, deg, a.res, a.elevation)
        p = os.path.join(a.out, f"dir_{d}.png")
        Image.fromarray(img, "RGBA").save(p)
        print(f"  dir_{d:2s} ({deg:3d} deg) covered="
              f"{100.0 * (img[..., 3] > 0).mean():.1f}%")


def stage_sheet(a):
    b = sheet_mod.SheetBuilder(cell=tuple(int(x) for x in a.cell.split("x")),
                               frames=1, directions=DIRECTIONS,
                               n_colors=a.colors)
    for d in DIRECTIONS:
        p = os.path.join(a.renders, f"dir_{d}.png")
        if not os.path.isfile(p):
            log.warning("missing %s", p)
            continue
        b.add("turn", d, 0, Image.open(p))
    out, atlas = b.save(a.out, allow_missing=True)
    im = Image.open(a.out).convert("RGBA")
    h = im.getchannel("A").histogram()
    print(f"wrote {a.out} {out.width}x{out.height} scale={atlas.get('scale')}")
    print(f"  transparent={100.0 * h[0] / (im.width * im.height):.1f}% "
          f"partial_alpha={sum(h[1:255])}px")
    _report_fit(im, a.cell)
    with open(os.path.splitext(a.out)[0] + ".atlas.json", "w") as fh:
        json.dump(atlas, fh, indent=2)


def _report_fit(sheet: Image.Image, cell: str) -> None:
    """How much of each cell the sprite actually uses, and whether the cell is
    the wrong way round for this subject.

    A QUADRUPED IS WIDE. Measured on the wolf: a 46x43 sprite in a 48x64 cell
    - it fills the width and wastes a third of the height, because 48x64 is a
    portrait cell chosen for standing humanoids. Nothing was WRONG with that
    sheet; it just spent a third of its pixels on empty space, which at this
    size is a real cost.

    Reported rather than auto-corrected: the cell size is part of what a
    consumer expects, so transposing it silently would change the contract.
    `--cell 64x48` is the lever.
    """
    cw, ch = (int(x) for x in cell.lower().split("x"))
    rows = sheet.height // ch
    widest = tallest = 0
    for i in range(rows):
        bb = sheet.crop((0, i * ch, cw, (i + 1) * ch)).getchannel("A").getbbox()
        if bb:
            widest = max(widest, bb[2] - bb[0])
            tallest = max(tallest, bb[3] - bb[1])
    if not widest or not tallest:
        return
    used = 100.0 * (widest * tallest) / (cw * ch)
    wfill, hfill = 100.0 * widest / cw, 100.0 * tallest / ch
    print(f"  cell fit: widest sprite {widest}x{tallest} of {cw}x{ch} "
          f"({used:.0f}% of the cell; {wfill:.0f}% of width, "
          f"{hfill:.0f}% of height)")
    # Compare the FILL on each axis, not the fitted sprite's own aspect. The
    # fitted sprite has already been squeezed to the cell, which hides the
    # subject's shape: the wolf measured 46x43 - almost square - purely
    # because 46 was all the width there was. What actually gives it away is
    # filling 96% of the width against 67% of the height.
    if wfill - hfill > 20:
        print(f"    NOTE: the sprite fills the width and leaves "
              f"{100 - hfill:.0f}% of the height empty - this subject is "
              f"wider than the cell is shaped for. --cell {ch}x{cw} fits it "
              f"at a larger scale (measured on a wolf: 64% -> 73% of the "
              f"cell, scale 0.190 -> 0.204).")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="stage", required=True)

    v = sub.add_parser("views", help="job turnaround -> keyed square views")
    v.add_argument("job")
    v.add_argument("out")
    v.add_argument("--frame", choices=("mirrored", "true"), default="true",
                   help="mirrored = fed to a multiview reconstructor; "
                        "true = colour sources for the final mesh")
    v.add_argument("--size", type=int, default=512)
    v.add_argument("--tolerance", type=int, default=24)
    v.add_argument("--engine", choices=("floodfill", "birefnet"),
                   default="floodfill",
                   help="birefnet clears ENCLOSED backdrops (vignettes, "
                        "discs, frames) that a corner flood fill cannot, at "
                        "the cost of needing CUDA and the weights")
    v.add_argument("--min-clear", type=float, default=45.0,
                   help="refuse if any view clears less than this percent")
    v.add_argument("--max-spread", type=float, default=15.0,
                   help="refuse if views disagree by more than this many "
                        "points - the stronger signal that keying failed")
    v.add_argument("--no-gate", dest="gate", action="store_false",
                   default=True,
                   help="proceed on a source that failed the checks. You will "
                        "get geometry of the BACKDROP; see the wolf medal")
    v.set_defaults(fn=stage_views)

    c = sub.add_parser("colour", help="project observed views onto a mesh")
    c.add_argument("mesh")
    c.add_argument("views")
    c.add_argument("out")
    c.add_argument("--res", type=int, default=192)
    c.add_argument("--sweep", type=int, default=10)
    c.add_argument("--no-repair", dest="repair", action="store_false",
                   default=True)
    c.add_argument("--min-component", type=float, default=0.02)
    c.add_argument("--no-cut-pedestal", dest="cut_pedestal",
                   action="store_false", default=True,
                   help="keep a reconstructed ground slab. It is FUSED to the "
                        "feet, so component removal cannot drop it, and it "
                        "renders as a grey plinth under every cell")
    c.add_argument("--pedestal-band", type=float, default=0.22,
                   help="fraction of the height searched for a plinth")
    c.add_argument("--pedestal-ratio", type=float, default=1.4,
                   help="how much wider than the SHINS a bottom slice must be "
                        "to count as plinth. 1.4 is validated on two "
                        "positives and two negatives across both body plans "
                        "and both reconstructors - see decisions/0016")
    c.set_defaults(fn=stage_colour)

    r = sub.add_parser("render", help="8 orthographic azimuths")
    r.add_argument("mesh")
    r.add_argument("out")
    r.add_argument("--views", default=None,
                   help="view dir; front.png fixes the azimuth labels")
    r.add_argument("--res", type=int, default=256)
    r.add_argument("--sweep", type=int, default=10)
    r.add_argument("--elevation", type=float, default=0.0,
                   help="camera degrees above horizontal. 0 is eye-level, "
                        "which the 2D lane uses and a humanoid reads fine at. "
                        "A QUADRUPED NEEDS ELEVATION: seen along its own "
                        "length at 0 it shows head, chest and front legs "
                        "stacked vertically and reads as standing upright. "
                        "30-45 is the range the reference sheets use")
    r.set_defaults(fn=stage_render)

    s = sub.add_parser("sheet", help="renders -> 48x64 sheet + atlas")
    s.add_argument("renders")
    s.add_argument("out")
    s.add_argument("--cell", default="48x64")
    s.add_argument("--colors", type=int, default=24)
    s.set_defaults(fn=stage_sheet)

    a = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
