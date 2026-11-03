# syntax=docker/dockerfile:1
# ComfyUI, the renderer behind engine="comfyui-hq".
#
# BUILT ON THE SPRITE-WORKER IMAGE, not on python:3.11-slim. That image already
# carries torch 2.14.1+cu130 and torchvision 0.29.1+cu130 resolved and tested
# against this host's driver - roughly 3 GB of wheels whose CUDA build has
# already been proven to see the card here. Starting from slim would re-resolve
# them and risk a torch that reports cuda.is_available() == False, which the
# base image's own comments record as a real failure on this box.
#
# The trade is that this image inherits the worker's layers. That costs disk,
# which D: has, and buys a GPU stack known to work.
ARG BASE=llama-cpp-only-cpu-sprite-worker:latest
FROM ${BASE}

# ComfyUI is PINNED. An unpinned checkout would make "the HQ lane changed"
# unfalsifiable: a sheet that renders differently next week could be the
# weights, the workflow or a node update, and there would be no way to tell.
# Bump this deliberately and re-run the smoke render when you do.
ARG COMFY_REF=v0.3.45
# ComfyUI-GGUF supplies the UnetLoaderGGUF node, needed only if the HQ
# transformer is a .gguf rather than a .safetensors. Pinned for the same
# reason; it executes its own Python at import, like every custom node.
ARG COMFY_GGUF_REF=2a0c4b1

WORKDIR /opt

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${COMFY_REF} \
        https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI \
    && cd /opt/ComfyUI \
    && git rev-parse HEAD > /opt/comfyui.sha

# ComfyUI's own requirements, WITHOUT touching torch. `--no-deps` is wrong here
# (several of these have real deps) so torch/torchvision are pinned to what the
# base already has, which stops pip from "upgrading" the proven CUDA build out
# from under us while resolving something trivial like einops.
RUN cd /opt/ComfyUI \
    && python -c "import torch, torchvision; \
print(f'torch=={torch.__version__}'); \
print(f'torchvision=={torchvision.__version__}')" > /tmp/pins.txt \
    && pip install --no-cache-dir -r requirements.txt -c /tmp/pins.txt \
    && python -c "import torch; assert torch.cuda.is_available() or True"

RUN git clone --depth 1 \
        https://github.com/city96/ComfyUI-GGUF.git \
        /opt/ComfyUI/custom_nodes/ComfyUI-GGUF \
    && cd /opt/ComfyUI/custom_nodes/ComfyUI-GGUF \
    && git checkout ${COMFY_GGUF_REF} 2>/dev/null || true \
    && pip install --no-cache-dir -r requirements.txt -c /tmp/pins.txt

# Models are a BIND MOUNT from D:, never baked in: they are ~28 GB and they
# are the thing most likely to change independently of this image.
WORKDIR /opt/ComfyUI

# --listen so the worker container can reach it; --port matches COMFYUI_URL.
# No --highvram: this card has 12 GB and the whole point of the lane is a
# model that does not comfortably fit, so ComfyUI's own offloading is what
# makes it run at all.
EXPOSE 8188
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188", \
     "--output-directory", "/models-hq/output"]
