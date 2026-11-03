# 0020 - Nunchaku INT4 cannot run the sheet conveyor: no custom LoRA

Date: 2026-11-03
Status: **Rejected for the sheet path, on a documented blocker, without
installing anything.** Still open as a candidate for the single-image
`/sdapi/v1/txt2img` lane, which needs no custom adapter. Nothing on the box
was changed by this spike.

**Result: Nunchaku's own Qwen-Image documentation states "Custom LoRA support
is under development". The sheet conveyor's camera conditioning IS a custom
LoRA (`angles`), so the conveyor cannot run on Nunchaku today at any quant
level, independently of whether it fits or how fast it is.**

## Why this exists

something2 asked for SOMET-690: replace the GGUF Qwen-Image-Edit runtime with
Nunchaku SVDQuant INT4 (`github.com/nunchaku-ai/nunchaku`), reporting the
rank-32 vs rank-128 trade-off and WSL2/CUDA compatibility, keeping GGUF
selectable as a fallback until proven. The motivation was build time: ~51 s
per cell, ~4 h 45 m for a 336-cell sheet.

It was chosen over SOMET-689 ("keep the image model resident during a sheet")
because 689's premise turned out to be false - `qwen_edit.denoise_cells`
already renders "every cell with ONE transformer load", so the transformer is
resident across a whole stage and only the 5 transformer-loading stages of 8
reload it, capping that idea at roughly 7.5 minutes of a 4 h 45 m build.

## The blocker

`denoise_cells` fuses **two** adapters onto the transformer:

- `lightning` - the step distillation. Nunchaku ships official Qwen-Image-Edit
  INT4 checkpoints with the Lightning 4-step and 8-step LoRAs **pre-merged at
  strength 1.0**, so this one is covered without custom LoRA support.
- `angles` - trained here, and the thing that pins the camera for each cell.
  CLAUDE.md's prop-concepts tripwire records what happens without it: "the
  angles LoRA pins pose, so a subject with no pose has a body invented for
  it" - a tree came back as a green humanoid.

Nunchaku's Qwen-Image usage page says, verbatim, **"Custom LoRA support is
under development."** Its LoRA documentation is FLUX-only: the worked examples
are FLUX.1-dev and the conversion modules live under `nunchaku.lora.flux`.
The project's own Qwen-Image launch note (2025-08-15) said "ComfyUI, LoRA, and
CPU offloading support are coming soon!"; offloading has since shipped and
custom LoRA has not.

So the ranks, the VRAM arithmetic and the WSL2 question are all moot for the
conveyor. There is no configuration in which it renders a correctly-angled
cell.

## What was established anyway, for whoever revisits this

- **Architecture: compatible in principle.** Nunchaku supports sm_80 / sm_86 /
  sm_89. This card is sm_86 (RTX 3060 12 GB). The README names the 3090 and
  A6000 as its sm_86 examples, not the 3060, so this is family-level support
  rather than a tested configuration.
- **Size: it does not fit resident.** The Qwen-Image-Edit INT4 builds measure
  about **11.5 GB at rank 32** and **12.7 GB at rank 128** - against 12 GB of
  card, before the text encoder and VAE. A Hugging Face discussion on that
  repo reports a 3060 12 GB with 24 GB of system RAM failing to load it even
  with a GGUF CLIP and VAE.
- **Offload exists and is the intended answer.** Nunchaku's per-layer
  asynchronous offload is documented to bring VRAM "to approximately 3GB",
  with the project claiming "no performance loss". Its docs warn to exclude
  `transformer` from diffusers' `enable_sequential_cpu_offload`, because
  Nunchaku's mechanism is its own. That shape is already proven on this box
  for GGUF: group offload took the transformer from 9408 MB to 2028 MB
  resident for 1.57x the denoise time, so "offload works here" is not a
  guess - but note that the GGUF measurement contradicts "no performance
  loss" for this card's PCIe bandwidth, and Nunchaku's claim should be
  measured rather than believed.
- **A 2511 build exists, unofficially.** The box's editor is
  Qwen-Image-Edit-**2511** (0015). Nunchaku publishes `qwen-image-edit` and
  `qwen-image-edit-2509`; a community repo, `QuantFunc/Nunchaku-Qwen-Image-
  EDIT-2511`, carries 2511. Using a different base version is its own risk:
  an adapter only fuses onto the base it was trained against.
- Python/torch/CUDA requirements are not stated on the README and were not
  pursued, because the LoRA blocker lands first.

## Where it could still pay

`/sdapi/v1/txt2img` - something2's single-image lane - runs Qwen-Image with
**only** the Lightning LoRA, and a pre-fused Lightning build removes the need
for custom LoRA support entirely. That path currently measures ~54 s warm
(encode ~6 s, place ~2.8 s, denoise ~45 s) on Q2_K. INT4 r32 would be a
quality improvement over Q2_K as well as a possible speed one.

That is a separate, smaller experiment than the one asked for, and it does not
touch the conveyor. It is not started.

## Consequences

- SOMET-690 as scoped is **not actionable**; say so rather than spending a
  session proving it with an install.
- The lever on the 51 s/cell stays where 0019 left it: fewer cells (mirroring
  already takes 336 to 210), step count (which the owner excluded), or the
  cell-size decision - not the runtime.
- **Re-open this when Nunchaku ships custom LoRA for Qwen-Image.** That single
  sentence in their docs is the whole gate. Check it before re-deriving any of
  the above.
- Nothing was installed, so there is nothing to roll back. The worker image is
  untouched.

## Sources

- https://github.com/nunchaku-ai/nunchaku
- https://nunchaku.tech/docs/nunchaku/usage/qwen-image.html
- https://nunchaku.tech/docs/nunchaku/usage/lora.html
- https://huggingface.co/nunchaku-ai/nunchaku-qwen-image-edit
- https://huggingface.co/nunchaku-ai/nunchaku-qwen-image-edit-2509
- https://huggingface.co/QuantFunc/Nunchaku-Qwen-Image-EDIT-2511
- https://huggingface.co/nunchaku-ai/nunchaku-qwen-image-edit/discussions/4
