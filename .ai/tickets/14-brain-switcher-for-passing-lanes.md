# Brain switcher, for the lanes that passed ticket 13

## What To Build

A roster of **brains** (text/vision models that plan and judge - never the
diffusion checkpoint) so callers of `regions.py`, `worlds.py` and the audio
style picker can pick the fast GPU 8B or a slower, smarter tier. Only lanes
that passed 0011 D2 get an entry.

## Blocked By

13.

## Scope

Shaped after ticket 13's result; outline only:

- One brain roster (mirroring `core_models.CORE_MODELS`): id, engine
  (`llama.cpp` / `colibri`), base URL, capabilities (`text`, `vision`), and a
  per-entry timeout, because a CPU tier is minutes where the GPU tier is seconds.
- Lift `worlds._llm_model()` / the chat-completions call into a shared helper
  that takes a brain id; keep the rules fallback and honest `author` recording
  that ticket 08 specifies.
- A second service in compose for the CPU tier with no GPU reservation (D4).
- UI: a brain select beside the existing image-model select - labelled so the
  two cannot be confused (`llm_name` is the image model; see `domain.md`).

## Out Of Scope

An image-model switcher to Qwen-Image / FLUX-class (0011 direction E, parked).
Letting any brain write diffusion prompts free-form.

## Acceptance Criteria

To be written from ticket 13's numbers.

## Suggested Route

`/plan-feature` once 13 is done.
