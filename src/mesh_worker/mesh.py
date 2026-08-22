"""Stage 2 of the 3D conveyor: a character image -> a 3D mesh, via TripoSR.

Deliberately a plain function plus a thin CLI rather than a Celery task. The
mesh step has one unverified question in front of it - whether a generated,
stylised character yields a usable mesh at all - and wiring it into the queue
before that is answered would be building on an assumption. `image_to_mesh` is
the seam a task would call later.

WHY TripoSR AND NOT A BIGGER MODEL
Reviews mark TripoSR down against TRELLIS and Hunyuan3D for weak textures and
fine detail. That criticism does not apply here: the final cell is a 48-128px
sprite, so 4096-square PBR and 8K textures are destroyed by the pixelation step.
What survives at 128px is SILHOUETTE and VOLUME. TripoSR is 1.68GB and MIT;
Hunyuan3D-2 is 74.9GB and TRELLIS reportedly needs ~24GB peak VRAM against this
card's 12GB. Pick the smallest model that gets the silhouette right.
See .ai/decisions/0004-pivot-to-3d-conveyor.md
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

MODEL_REPO = "stabilityai/TripoSR"
CACHE_DIR = "/models"


def load_model(device: str):
    from huggingface_hub import hf_hub_download
    from tsr.system import TSR

    # Resolve the files ourselves and hand TSR a DIRECTORY.
    #
    # TSR.from_pretrained(path, config_name, weight_name) takes no cache_dir and
    # calls hf_hub_download without one, so it would honour HF_HOME and fetch
    # into /models/hub/... - a second 1.68GB copy alongside the one already
    # cached at /models/models--stabilityai--TripoSR by cache_dir=/models.
    # Two cache layouts under one volume is how a 12GB card ends up with a full
    # disk. Passing a local directory takes from_pretrained's os.path.isdir
    # branch instead, which reads the files directly and downloads nothing.
    cfg = hf_hub_download(MODEL_REPO, "config.yaml", cache_dir=CACHE_DIR)
    ckpt = hf_hub_download(MODEL_REPO, "model.ckpt", cache_dir=CACHE_DIR)
    snapshot = os.path.dirname(ckpt)
    if os.path.dirname(cfg) != snapshot:
        raise RuntimeError(f"config and weights in different dirs: {cfg} {ckpt}")

    model = TSR.from_pretrained(
        snapshot,
        config_name=os.path.basename(cfg),
        weight_name=os.path.basename(ckpt),
    )
    # Chunk size trades speed for peak VRAM during the triplane query. 8192 is
    # TripoSR's own default for constrained cards; this one shares 12GB with a
    # diffusion stack that may still be resident.
    model.renderer.set_chunk_size(8192)
    model.to(device)
    return model


def prepare_image(path: str, remove_bg: bool, size: int = 512):
    """Load a character image as TripoSR expects it: RGB, foreground on grey.

    The grey background is not decoration. TripoSR was trained on renders keyed
    this way, and the alpha edge is what it reads as silhouette - which is the
    one property that has to survive to a 128px sprite.
    """
    img = Image.open(path)

    if remove_bg:
        import rembg
        img = rembg.remove(img, session=rembg.new_session())
    elif img.mode != "RGBA":
        # A sprite from step 1 is already keyed; anything else without alpha has
        # no silhouette to read and would be reconstructed as a flat slab.
        print(f"WARNING: {path} has no alpha and --remove-bg was not passed. "
              "TripoSR needs a keyed foreground; expect a poor mesh.")

    img = img.convert("RGBA")
    arr = np.array(img).astype(np.float32) / 255.0
    rgb, alpha = arr[:, :, :3], arr[:, :, 3:4]
    composited = rgb * alpha + 0.5 * (1 - alpha)
    out = Image.fromarray((composited * 255.0).astype(np.uint8))

    # Fit the character into a square canvas with margin. TripoSR expects the
    # subject centred and roughly filling the frame; a sprite that sits low and
    # left in its canvas reconstructs off-axis.
    bbox = img.getbbox()
    if bbox:
        crop = out.crop(bbox)
        side = int(max(crop.size) * 1.15)
        canvas = Image.new("RGB", (side, side), (127, 127, 127))
        canvas.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
        out = canvas
    return out.resize((size, size), Image.Resampling.LANCZOS)


def repair_for_rigging(mesh):
    """Make a marching-cubes surface acceptable to a skinning solver.

    Auto-riggers assume a closed, single-shell surface: skinning weights are
    solved over the volume, and an open surface leaves vertices no bone claims.
    TripoSR's raw output is neither, but only barely - measured on the zombie
    mesh, 100 boundary edges out of 71,768 (0.14%), with the boundary vertices
    sitting in the interior rather than on the volume faces. That means small
    holes, not a reconstruction clipped by the grid, so filling is the right
    tool. A clipped mesh would need re-extraction at a larger bound instead.

    Order matters. Specks are dropped FIRST: a 32-face fragment floating beside
    a 47,828-face body counts as its own shell, so hole-filling would dutifully
    close it and leave a sealed pebble inside the rig.
    """
    import trimesh

    before = {"faces": len(mesh.faces), "watertight": mesh.is_watertight}

    # 1. Keep only components big enough to be part of the character.
    comps = mesh.split(only_watertight=False)
    if len(comps) > 1:
        comps = sorted(comps, key=lambda c: len(c.faces), reverse=True)
        dropped = sum(len(c.faces) for c in comps[1:])
        mesh = comps[0]
        logger_print(f"  dropped {len(comps) - 1} stray component(s), {dropped} faces")

    # 2. Standard degeneracy cleanup before filling, so fill_holes is not asked
    #    to reason about zero-area faces or duplicated vertices.
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()

    # 3. Close the remaining holes and make the winding consistent, so normals
    #    point outward - a solver reading inverted normals weights the inside.
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_normals(mesh)

    return mesh, {
        "faces_before": before["faces"],
        "faces_after": len(mesh.faces),
        "watertight_before": before["watertight"],
        "watertight_after": bool(mesh.is_watertight),
    }


def logger_print(msg):
    print(msg, flush=True)


def image_to_mesh(image_path: str, out_path: str, device: str = None,
                  remove_bg: bool = False, mc_resolution: int = 256,
                  bake_texture: bool = False):
    """Reconstruct `image_path` into a mesh at `out_path`. Returns a report."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    model = load_model(device)
    t_load = time.time() - t0

    img = prepare_image(image_path, remove_bg)
    prepped = os.path.splitext(out_path)[0] + "_input.png"
    img.save(prepped)

    t1 = time.time()
    with torch.no_grad():
        codes = model([img], device=device)
    t_infer = time.time() - t1

    t2 = time.time()
    # has_vertex_color=False: colour is irrelevant at sprite scale and the
    # rigging step that follows cares only about geometry.
    meshes = model.extract_mesh(codes, has_vertex_color=bake_texture,
                                resolution=mc_resolution)
    t_mesh = time.time() - t2

    mesh = meshes[0]
    mesh, fixes = repair_for_rigging(mesh)
    mesh.export(out_path)

    return {
        "out": out_path,
        **fixes,
        "prepared_input": prepped,
        "device": device,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "load_s": round(t_load, 1),
        "infer_s": round(t_infer, 1),
        "mesh_s": round(t_mesh, 1),
    }


def main():
    ap = argparse.ArgumentParser(description="Character image -> 3D mesh")
    ap.add_argument("image")
    ap.add_argument("-o", "--out", default="/app/images/mesh.obj")
    ap.add_argument("--remove-bg", action="store_true",
                    help="run rembg first; unnecessary for a keyed sprite")
    ap.add_argument("--mc-resolution", type=int, default=256,
                    help="marching cubes grid; higher = finer and slower")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.image):
        print(f"No such image: {args.image}", file=sys.stderr)
        return 2

    report = image_to_mesh(args.image, args.out, device=args.device,
                           remove_bg=args.remove_bg,
                           mc_resolution=args.mc_resolution)
    for k, v in report.items():
        print(f"{k:16s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
