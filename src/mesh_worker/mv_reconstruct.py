#!/usr/bin/env python3
"""Hunyuan3D-2mv: three observed views -> a mesh. Run in the hy3d venv.

    PYTHONPATH=/opt/Hunyuan3D-2 HF_HUB_CACHE=/models /opt/hy3d/bin/python \\
        /app/mv_reconstruct.py <views-dir> <out.glb> [device] [steps] [octree]

Set up the venv first with /app/setup_hy3dgen.sh, which explains why this is
not baked into the image and when multiview is worth running at all.

WHY THE VIEWS ARE MIRRORED, AND WHY dir_w IS NOT USED
The model wants {front, left, back}. The obvious mapping is
{dir_s, dir_w, dir_n} and it is wrong twice:

  - `dir_w` is not reliable. Measured across 11 jobs: in most it is a
    mirror-ish re-render of `dir_e` (|e-mirror(w)| 20-50), but in TWO
    (95486f75, c71047a1) it is a near-DUPLICATE of `dir_e` facing the same
    way (|e-w| 7.02 against |e-mirror(w)| 27.15) - the documented left-side
    failure. Feeding that as "left" hands the model a right-side view
    labelled left.
  - Even when it does mirror, handedness flips, so an asymmetric prop appears
    in opposite hands between front and left. To a reconstructor those are
    contradictory observations of one object.

So mirror the whole problem instead:

    {front: M(dir_s), left: dir_e, back: M(dir_n)} -> mesh -> M(mesh)

In the mirrored world `dir_e` IS a left-side view and all three inputs share
one handedness. Mirroring the mesh back on X restores the true subject. Same
M(render(M(C))) identity the 2D turnaround already relies on, and it uses only
genuinely RENDERED views.

`turnaround.py views --frame mirrored` produces exactly this set.
"""
import os
import sys
import time

from PIL import Image

SRC = sys.argv[1]
OUT = sys.argv[2]
DEVICE = sys.argv[3] if len(sys.argv) > 3 else "cuda"
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 30
OCTREE = int(sys.argv[5]) if len(sys.argv) > 5 else 256

views = {}
for n in ("front", "left", "back"):
    p = os.path.join(SRC, f"{n}.png")
    if not os.path.isfile(p):
        raise SystemExit(f"missing {p} - run `turnaround.py views "
                         f"--frame mirrored` first")
    views[n] = Image.open(p).convert("RGBA")
    print(f"  {n}: {views[n].size}", flush=True)

print(f"=== loading on {DEVICE} ===", flush=True)
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa: E402

t0 = time.time()
pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    "tencent/Hunyuan3D-2mv",
    subfolder="hunyuan3d-dit-v2-mv",
    variant="fp16",
    use_safetensors=True,
    device=DEVICE,
)
print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

t1 = time.time()
meshes = pipe(image=views, num_inference_steps=STEPS,
              octree_resolution=OCTREE, enable_pbar=False)
print(f"=== reconstructed in {time.time() - t1:.1f}s ===", flush=True)

mesh = meshes[0]
print(f"  raw: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
      f"watertight={mesh.is_watertight}", flush=True)

# Mirror back on X: the reconstruction is of the MIRRORED subject.
mesh.vertices[:, 0] *= -1.0
# Flipping one axis inverts winding. Without reversing the faces every normal
# points inward, and a back-face-culling renderer shows an empty frame - which
# reads as "the model produced nothing" rather than "the normals are inside
# out". turnaround.py's renderer shades two-sided for the same reason.
mesh.faces = mesh.faces[:, ::-1]

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
mesh.export(OUT)
print(f"wrote {OUT}", flush=True)
print("  NOTE: this output is RAW - not repaired, and it carries a ground "
      "slab fused to the feet. `turnaround.py colour` cuts that; see "
      "--cut-pedestal.", flush=True)
