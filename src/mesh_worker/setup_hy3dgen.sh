#!/bin/sh
# Multiview reconstruction (Hunyuan3D-2mv) inside mesh_worker, on demand.
#
#   docker compose ... --profile mesh exec -T mesh-worker sh /app/setup_hy3dgen.sh
#
# WHY A SCRIPT AND NOT A DOCKERFILE LAYER
# `make turnaround3d` uses single-image TripoSR: 1.68 GB, MIT, ~45 s on CPU,
# and it is the right default. Multiview is 5.36 GB of weights plus this
# dependency set, under a licence with territorial exclusions, and 0018 made
# disk a standing retention policy after the 0016 spike's images took C: from
# 108.7 GB free to 16.3 in a day. Baking in a path the default lane does not
# use would be paying that cost permanently.
#
# WHEN MULTIVIEW IS ACTUALLY WORTH IT (measured, 0016)
# It depends on how the subject sits in its CONCEPT, which is a conditional the
# first version of 0016 missed:
#   - concept faces the camera (the mage): TripoSR observes the front and
#     invents only the back. Multiview did not clearly earn its cost.
#   - concept is a PROFILE (the wolf): TripoSR observes one flank and invents
#     the chest, the face-on geometry and the far flank entirely. Four of eight
#     cells then read as a figure standing upright, because the head-on
#     direction has no observed data at all. Multiview is the fix.
#
# This is a venv in the CONTAINER, so it dies on `--force-recreate`. That is
# the trade for not baking it; re-run this script, it is a couple of minutes.
set -e

# --system-site-packages shares torch/torchvision (~3 GB, already present).
# The venv still SHADOWS transformers, so TripoSR keeps the 4.35 it is pinned
# to in system site-packages - that pin is why rig_worker is a separate image
# and it must not be disturbed.
test -d /opt/hy3d || python3 -m venv --system-site-packages /opt/hy3d
/opt/hy3d/bin/pip install -q --upgrade pip

# PINS ARE LOAD-BEARING, both learned by failure:
#   transformers 4.49.0 - newer versions renamed DINOv2's attention
#     projections (query/key/value -> q_proj/k_proj/v_proj) and the MLP to
#     gate/up/down_proj, so every projection weight in a 2025 checkpoint reads
#     as MISSING. The error is a wall of "Missing key(s) in state_dict" that
#     looks like corrupt weights.
#   diffusers 0.32.2 - 0.42 requires huggingface-hub >= 1.32, which the pinned
#     transformers will not install.
/opt/hy3d/bin/pip install -q \
    einops omegaconf trimesh pymeshlab pygltflib accelerate \
    transformers==4.49.0 diffusers==0.32.2

# hy3dgen is a repo, not a package - its inference lives in hy3dgen/ and is
# imported by path, same as TripoSR's tsr/ and UniRig's src/.
test -d /opt/Hunyuan3D-2 \
    || git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git \
         /opt/Hunyuan3D-2

echo "--- versions ---"
/opt/hy3d/bin/python -c "import transformers, diffusers; print('venv: transformers', transformers.__version__, 'diffusers', diffusers.__version__)"
python3 -c "import transformers; print('system (TripoSR): transformers', transformers.__version__)"
/opt/hy3d/bin/python -c "import sys; sys.path.insert(0, '/opt/Hunyuan3D-2'); from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline; print('hy3dgen imports OK')"

cat <<'USAGE'

Ready. The multiview path, from a job that has a turnaround:

  # 1. views in the MIRRORED frame - dir_w never participates, see 0016
  python /app/turnaround.py views <job> <job>/t3d/views_mv --frame mirrored

  # 2. reconstruct (GPU, ~105 s; weights are 5.36 GB under models.toml)
  PYTHONPATH=/opt/Hunyuan3D-2 HF_HUB_CACHE=/models /opt/hy3d/bin/python \
      <your mv_run script> <job>/t3d/views_mv <job>/t3d/mesh_mv.glb cuda 30 256

  # 3. colour and render as usual - turnaround.py handles the rest
USAGE
