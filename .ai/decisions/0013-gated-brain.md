# 0013 - The brain goes through the model gateway; `llm_engine` leaves the GPU

Date: 2026-10-01
Status: **Accepted. Built and verified 2026-10-01** (measurements below).
Contract:
[specs/something2-text/contract.md](../specs/something2-text/contract.md).

## Context

something2 wants a text provider (`/api/text`) for its audio prompt writer:
JSON-schema answers (a music style from an enum, an SFX entity) and free text.
The rule from the owner: the brain goes through the model gateway and is
**never resident next to an image or audio pipeline** - the WSL2 out-of-memory
wedge is what that co-residence produces on a 12 GB RTX 3060.

The existing brain does not meet that rule. `llm_engine` (llama.cpp router,
`Qwen3-8B-Q8_0`, 8.2 GB, `-ngl 99`) sits **outside** the gateway: any call -
`worlds.py` biome plans, or a chat - loads it onto the card, and only
`--sleep-idle-seconds 120` gives the card back. A gated `/api/text` beside it
would honour the rule's letter and leave its failure reachable.

## Options considered

| | Optimises for | Gives up |
|---|---|---|
| **A. Worker-owned `llama-server` child, gated like `qwen_server`** | one arbiter for the card; grammar-guaranteed schemas; a pattern already debugged here | a CUDA llama.cpp binary in the worker image; one more child process |
| B. Keep `llm_engine`, unload it from the gateway | least code | eviction depends on another container cooperating; a stray call re-creates the wedge |
| C. transformers + 4-bit in the worker | same evict-on-miss path as diffusers | new deps, slower decode, weaker schema enforcement |
| D. CPU-only brain | never touches VRAM | breaks the stated rule; competes for host RAM, the other half of the wedge |

## Decision

1. **A.** The worker owns a `llama-server` child for the brain, under gateway
   label `brain:<id>` in `model_gateway.GATED`. A gateway switch kills it; it
   is resident only while it is the active model (normal idle hold).
2. **Two brains, one runtime** (owner, 2026-10-01: "lean to bigger"):
   - **default `qwen3.6-35b-a3b`** - Qwen3.6-35B-A3B, unsloth `UD-IQ4_XS`
     (17.7 GB), MoE with 3B active. **Hybrid:** attention + KV on the card,
     experts in host RAM. Thinking is ON by default in this model and is
     turned off per request (`chat_template_kwargs.enable_thinking=false`).
     IQ4_XS rather than Q4_K_M (22.1 GB) because only it leaves 0011's
     ">= 12 GB RAM for the worker" inside WSL's 36 GB.
   - **fast `qwen3-vl-8b`** - Qwen3-VL-8B-Instruct Q4_K_M (5.0 GB, already
     at `/models/image-gguf/qwen-image-2.1/`), fully on the card, an Instruct
     release with no thinking mode. Header read 2026-10-01: `qwen3vl`, 36
     blocks, 4096 wide, output head + chat template, no vision tensors (the
     mmproj is not on disk), so it loads text-only.
   - **Correction:** the `Qwen3-8B-Q8_0` that `project-context.md` and 0011
     name is **not on disk** (checked 2026-10-01). `llm_engine` has only
     `Qwen2.5-3B-Instruct-Q4_K_M` to serve - plausibly the model behind the
     40/40 `medieval_fantasy` canary.
   - Dense 27-32B stays rejected for 0011's reason (3-6 tok/s split across
     card and RAM).
   - WSL `processors` 4 -> 10 so the experts have cores (host has 12 threads).
   - **This overrides 0011 D4 ("no brain lane may hold VRAM") for the text
     lane.** D4's reason was a brain *beside* the image model; under the
     gateway the brain holds the card exclusively or not at all.
   - Host RAM is the other half of the wedge: the 35B must never be active
     beside the ~17 GiB `qwen_server`. Every gateway switch already stops it;
     the brain also preflights free RAM before loading.
3. **Every brain caller is rerouted to the gated brain, and `llm_engine` stops
   holding the card.** One brain runtime, not two. There are three callers -
   `worlds.py`, `audio_styles.py` and `regions.py` (the last runs inside a map
   job, *after* the tiles, with the image pipeline still cached: a live wedge
   path before this change). Each keeps its existing rule fallback for when
   the brain is refused. Details in
   [plan.md](../specs/something2-text/plan.md).
4. **Busy is refused at once** (503 + `Retry-After` via `admit_sync`), never
   queued or borrowed - something2 has no retries and a 5-minute budget, and
   falls back on 409/503.
5. Every call writes a `generations` row (`kind='text'`, reply in `params`,
   capped). It is a ledger, never read back as a cache.

## Consequences

- **World specs get a new failure frequency, not a new failure mode.** They
  fall back to deterministic biomes whenever the card is held by another model.
  `WORLD_LLM_TIMEOUT` is 45 s, shorter than a cold brain load (target <= 2 min),
  so a world spec that wins the card cold also falls back - and the load
  continues for the next caller.
- For up to `MODEL_SWITCH_IDLE_S` (300 s) after a model is picked in the UI,
  text requests are refused. Correct by the gateway's rules; if it hurts, lower
  that knob rather than special-casing the brain.
- The CLAUDE.md tripwire "Qwen3-8B and a diffusion pipeline cannot both hold
  the card; `--sleep-idle-seconds 120` lets them share it" becomes obsolete
  when this ships - update it then, not before.

## Unverified at time of writing

- That llama-server honours `chat_template_kwargs.enable_thinking=false` for
  Qwen3.6 (check raw output for `<think>`); `--reasoning-budget 0` is the
  server-side fallback.
- The 35B against 0011's text-lane gates (>= 8 tok/s decode, ~1.5k-token
  prompt <= 30 s, >= 12 GB RAM left) - at 4 and at 10 vCPUs. T0 fills 0011's
  table with the llama.cpp row.
- Whether something2's actual schemas convert to a grammar (`$ref`,
  `pattern` are the usual gaps).
- Cold-load time and resident VRAM - to be measured, then recorded here.

## Measured (2026-10-01, RTX 3060 12 GB, WSL 36 GB / 10 vCPUs)

Through the real stack (`/api/text`, gateway, worker child), host-side VRAM
from `Get-Counter` minus the ~1.0 GB desktop baseline.

| | Qwen3.6-35B-A3B IQ4_XS, `--n-cpu-moe 24` | Qwen3-VL-8B Q4_K_M, all on card |
|---|---|---|
| load, file not in page cache | **107-143 s** (112.8 s wall for the first request after a WSL restart) | 36.6 s |
| load, file in page cache | 4-13 s | 2-28 s |
| ~200-240-token JSON, warm | **13.3-14.8 s** | **3.7-3.9 s** |
| decode | 18.1-18.3 tok/s (18.8 at 4 vCPUs - more cores did NOT help) | 52.9-55.7 tok/s |
| 1.5k-token prompt | 10.6 s | 4.5 s |
| brain VRAM | **~8.3 GB** (~3 GB of card left) | ~5.1 GB |
| host RAM | experts are mmapped page cache; `MemAvailable` stayed ~31.7 of 36 GB | negligible |
| thinking | off (`enable_thinking=false` honoured; no `<think>`) | none |

Expert split, 35B, measured at 4 vCPUs: n=40 10.7 tok/s; n=24 ~8.3 GB / 18.8;
n=20 ~9.7 GB / 21.7; n=16 ~10.9 GB / 27.4 with 0.3 GB of card left - rejected
for the same zero-headroom reason as 0011 option B. 24 chosen.

**The cold 35B load is over the 2-minute target** (107-143 s) when the 17.7 GB
file has left page cache - after a WSL restart, or after a Qwen-Image run
evicted it. Inside something2's 5-minute budget and `TEXT_GENERATE_TIMEOUT_S`
240, so it answers; it does not meet the owner's "about 2 minutes". The 8B is
the answer when that matters.

Verified end to end: 401 without a key; 422 for an impossible schema and an
unknown model before any GPU time; **503 in 0.06 s with `Retry-After: 60`
while ACE-Step held the card, the music job completed, `gpu-health` OK**;
worlds / audio propose / regions each succeed with the brain admitted and fall
back to rules (with the reason) when refused; the regions in-job brain is
stopped before the job continues; every call is a `generations` row.

### Style variety - the canary is NOT a model problem

10 identical Vale music requests at temperature 0.15: **10/10
`medieval_fantasy` on both brains** - determinism, as expected at 0.15.

8 subjects x 2 shuffled enum orders: the 35B used 3 of 5 styles, the 8B 2 of
5. **Shuffling the enum never changed a pick** - position bias is ruled out.
The picks are often wrong on their face: the 35B chose `medieval_fantasy` for
an inn ("The Rusty Flagon") and `tavern` for a war camp; neither brain ever
chose `battle`, `village` or (8B) `tavern`. `medieval_fantasy` behaves as the
genre, not as one option - every subject IS medieval fantasy. A bigger brain
does not fix a vocabulary in which one value describes all the others; the
lever is the style list or the client's prompt (describe each style, or drop
the catch-all), which is something2's side and the generator's roster.
