# 0011 - A switchable "brain" tier: Colibri spike before any switcher

Date: 2026-09-27
Status: **Proposed, awaiting measurements.** Direction decided; nothing is
switchable until the spike below fills its table. Tickets 12-14.

## Why this exists

The ask: the host was upgraded (Ryzen 5 5500 6c/12t, 48 GB RAM, same RTX 3060
12 GB), and the owner wanted to run much bigger LLMs (27-40B Qwen3 class or
larger) via [Colibri](https://github.com/JustVugg/colibri), switchable beside
the current small one, expecting that to produce better pixel art and
sprites.

## The finding that reshaped the ask

**The LLM does not draw, and in this codebase it never touches a sprite.**
`llm-server` (Qwen3-8B-Q8_0, llama.cpp, GPU, `--sleep-idle-seconds 120`) is
called only by `regions.py` (map region graphs), `worlds.py` (biome plans) and
the planned audio style picker (ticket 08). Sprite quality is bounded by the
**image model** - SDXL + LoRA, the seed, CLIP's 77 tokens, no negation in a
positive prompt - none of which a smarter text model changes.

The observed symptom ("sometimes generates something not expected at all")
has a measured cause instead: **SDXL-Turbo**, distilled, guidance 0, negatives
inert, which lost the 2026-08-21 bench (36.9% / 25.2% palette / blockiness
against 86.6% / 74.7% for SDXL + `nerijs/pixel-art-xl`) and was dropped from
the roster - while `main.py` form defaults and a `generator.js` fallback kept
serving it to any caller that omitted `llm_name`. The owner reported all four
entry points in use (React, legacy UI, something2, scripts). Ticket 12.

The naming made this easy to miss: **`llm_name` is the image checkpoint**, not
a language model. See `domain.md`.

## What Colibri is (read 2026-09-27 from its README; not yet run here)

- Its own C runtime, not llama.cpp. Runs large mixture-of-experts models by
  keeping dense weights resident in RAM (int4) and **streaming routed experts
  from disk** with an LRU / pinned hot-store / one-layer-ahead prefetch.
  OpenAI-compatible gateway, Docker images, WSL2 listed. Apache 2.0.
  Current version v1.12.1 - there is no "Colibri 2".
- **Language models only.** Output is text. GLM-5.3-Flash and DeepSeek V4.1
  Flash accept image *input*. No diffusion, no Qwen-Image - so "switch between
  SDXL+LoRA and Colibri Qwen-Image" is not a thing; the image-model switcher
  and the brain switcher are different switchers.
- Its own figures: 128 GB CPU-only ~1.8 tok/s warm; single RTX 5070 Ti 1.07
  tok/s; 25 GB box 0.05-0.1 tok/s cold. Assumes NVMe.
- "Brio" mode scores a closed set of options instead of generating - the
  shape a judge verdict wants.

## Directions considered

| | optimises for | gives up |
|---|---|---|
| A. MoE ~30-35B resident in RAM on CPU, beside the GPU 8B | no VRAM contention, always warm | weaker than dense 32B; slow prefill |
| B. Dense 27-32B with partial GPU offload | most intelligence per answer | 3-6 tok/s and it takes the card - reintroduces the zero-headroom state behind the 2026-09-04 faults. Rejected |
| C. Vision judge picks best-of-N / rejects item sheets | the only LLM direction that moves sprite quality | slow image prefill on CPU; unproven on 64-256 px sprites |
| D. LLM fills structured prompt slots (ticket 08 pattern) | consistent prompts | 77-token / no-negation ceiling |
| E. Spend the RAM on the image side (Qwen-Image, FLUX-class offload) | attacks image quality head-on | reopens 0002/0005; **parked, not rejected** - **taken 2026-09-28, see [0012](0012-qwen-image-default-lane.md)** (resident on the card, not RAM offload) |
| F. A model switcher | the literal ask | plumbing, not quality |
| G. Colibri for models that do not fit RAM | GLM/DeepSeek-class vision | SATA SSDs, ~200 GB disk, second runtime |

## Decisions

### D1. Turbo fallback removed first (ticket 12)

A precondition, not a quality feature: any comparison run while Turbo leaks
in through a default measures the wrong model.

### D2. Colibri spike before any switcher (ticket 13)

The owner chose to start from Colibri and investigate before implementing.
Two lanes, with pass criteria **fixed before measuring**:

| lane | model | passes if |
|---|---|---|
| smart text brain | Qwen3.6-35B (35B / 3B active, fully RAM-resident, ~20 GB int4) | decode >= 8 tok/s; a ~1.5k-token prompt answered in <= 30 s; >= 12 GB RAM left for the worker; no VRAM held |
| vision judge | GLM-5.3-Flash (321B / 40B active, experts streamed from SSD, ~195 GB) | one sprite verdict as a closed choice (single / sheet / wrong subject) <= 60 s warm for per-candidate use, <= 5 min for batch-only; **and >= 10 / 12** on the labelled set |
| control (not a gate) | same Qwen3.6 as GGUF in the existing llama.cpp | if it matches Colibri, the text lane does not need a second runtime |

Why these numbers: below ~8 tok/s a plan request reads as hung; the worker's
measured host peak is ~10 GiB (`qwen_edit.py` split encode/denoise), so 12 GB
keeps it safe; a fast wrong judge is worse than none, hence the accuracy gate.
Qwen3.6 has no listed vision, so the judge lane rests on GLM-5.3-Flash
(DeepSeek V4.1 Flash as fallback).

### D3. Switchers only for lanes that pass (ticket 14)

The brain switcher follows the existing roster pattern (one list, surfaced to
UI and callers) and never lets a brain emit a diffusion prompt directly.

### D4. The GPU stays the image model's

No brain lane may hold VRAM by default. The 12 GB card's fault history
(project-context "Known-broken") is the reason.

## Assumptions, and what breaks if they are wrong

- CPU decode is memory-bandwidth bound (dual-channel DDR4 ~40-50 GB/s). If a
  3B-active MoE measures far below 8 tok/s, the text lane fails and D3 builds
  nothing for it.
- Colibri's expert streaming tolerates SATA (~0.5 GB/s). DeepSeek V4.1 Flash
  reads ~4.5 GB of experts per token by its own README - on SATA that is
  seconds per cache-missed token, which is why GLM is the primary judge.
- A vision LLM's verdict is meaningful on small pixel sprites. Unproven; the
  accuracy gate exists for this.
- Pre-converted GLM-5.3-Flash weights are obtainable; converting from source
  needs source + output on disk at once, which D: (223 GB) cannot hold.

## Measured (to be filled by ticket 13)

| lane | engine | model | decode tok/s | prefill (1.5k tok) | RAM peak | RAM left | VRAM | verdict s (cold / warm) | accuracy /12 | pass? |
|---|---|---|---|---|---|---|---|---|---|---|
| text | Colibri | Qwen3.6-35B | | | | | | n/a | n/a | |
| text | llama.cpp | Qwen3.6-35B-A3B UD-IQ4_XS, hybrid `--n-cpu-moe 24` (2026-10-01, [0013](0013-gated-brain.md)) | 18.1-18.8 | 10.6 s | page cache only (mmapped experts) | ~31.7 of 36 GB | ~8.3 GB (while active, gateway-exclusive) | n/a | n/a | **passes the text gates** - on the GPU via the gateway, not RAM-only as this lane assumed; see 0013 for why D4 does not apply |
| judge | Colibri | GLM-5.3-Flash | | | | | | | | |
