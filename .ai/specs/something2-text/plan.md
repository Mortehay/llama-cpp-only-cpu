# Plan - gated brain, `/api/text`, callers rerouted

Implements [0013](../../decisions/0013-gated-brain.md) and
[contract.md](contract.md). Written 2026-10-01 against `4378feb`.

**Status 2026-10-01: T0-T5 done** (uncommitted at time of writing). Open:
`make lan-expose` after the WSL restart (UAC declined/timed out once, so LAN
clients cannot reach the API until it runs); the real SFX-entity schema;
the style-variety result in 0013, which is a vocabulary question, not a
brain one.

## What the code says (read before planning, 2026-10-01)

- **Three callers use `llm_engine`, not one:** `worlds._llm_biome_plan`
  (API process), `audio_styles._llm_style_plan` (API process - both
  `/api/audio/propose` and the music generation path, `audio.py:321`, when no
  style is given), and `regions.propose` (**inside the solo worker**, from
  `map_tasks.py:563`). All three already fall back to rules when the brain does
  not answer.
- `regions.propose` runs **after** the map's tiles were drawn, with the image
  pipeline still cached on the card, and then wakes an 8.2 GB GGUF beside it.
  That is a live wedge path today, not a hypothetical one.
- `regions.py` already sends `response_format: json_schema` (strict) to
  llama.cpp - the grammar path works on this llama.cpp; only the packaging is new.
- `audio_styles` asks at **temperature 0**, with "medieval fantasy pixel-art
  RPG" in its own prompt and `medieval_fantasy` listed first. Whether that is
  where the 40/40 canary came from is unverified (something2 may have its own
  prompt), but it is a candidate cause independent of the model.
- A worker task cannot block on another Celery task: `--pool=solo` would
  deadlock. So the worker-side caller needs an in-process path.
- `collector/bridge.py` also calls `LLM_URL`; legacy, out of scope.

## Tickets

Order: T0 gates everything. T3 must land in ONE change (reroute + `llm_engine`
off the card) or the callers lose the brain in between. T4 can run beside T3.

### T0 - Spike: CUDA `llama-server` inside the worker image

Worker is `python:3.11-slim` + pip torch. Try in order, stop at the first that
works:

1. Multi-stage `COPY --from=ghcr.io/ggml-org/llama.cpp:server-cuda@<digest>`
   the server binary + `libggml*`/`libllama*`, and the CUDA runtime libs it
   links (`libcudart`, `libcublas`, `libcublasLt`) into `/opt/llama`;
   `LD_LIBRARY_PATH` scoped to the child process only. Watch glibc: the image's
   Ubuntu base must not be newer than Debian bookworm's 2.36.
2. Fall back: build llama.cpp with `-DGGML_CUDA=ON` in a builder stage.

Exit criteria (recorded in 0013):
- `llama-server --version` in the worker; `ldd` clean.
- Both brains load (0013 D2), from two levels below `/models`:
  `brain-gguf/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` (hybrid: all
  layers on the card, routed experts kept in RAM - `--n-cpu-moe`/`-ot exps=CPU`;
  tune how many experts fit on the card) and
  `image-gguf/qwen-image-2.1/Qwen3VL-8B-Instruct-Q4_K_M.gguf` (`-ngl 99`).
- Thinking off on the 35B via `chat_template_kwargs.enable_thinking=false`,
  checked by reading raw output for `<think>`; `--reasoning-budget 0` if not.
- Per brain: cold load s (cold and warm page cache), resident VRAM host-side
  (`Get-Counter`), RAM peak and RAM left in WSL, decode tok/s, a ~1.5k-token
  prompt's wall time, a `json_schema` request honouring an enum.
- The 35B against 0011's gates (>= 8 tok/s, <= 30 s for 1.5k tokens, >= 12 GB
  RAM left) at `processors=4`, then again at 10 (`.wslconfig`, then
  `wsl --shutdown` and the CLAUDE.md post-reboot steps - only with no job
  running). Fill 0011's llama.cpp row.
- Card fully released after the child is killed (host counter back to baseline).

### T1 - Brain in the worker, gated

- `brain_engine.py` (pattern: `audio_engine.py` + the qwen_server helpers):
  start / alive / stop / `complete(messages, schema, temperature, max_tokens)`
  against the child on localhost. Startup preflight: refuse if free VRAM <
  `BRAIN_MIN_FREE_MB` instead of OOMing half way. Idle exit `BRAIN_IDLE_S`.
  Every child call has a wall-clock timeout (a hung child must not wedge the
  solo worker - `audio_engine.ACESTEP_TIMEOUT_S` lesson).
- A brain roster (`brain_engine.BRAINS`: id, path, launch args, default) -
  the single list `/api/text/models` and the gateway read.
- `tasks.generate_text_task(model, ...)`, label `"brain:" + model` in
  `GATED`, each label in `FIXED_LABELS`, the task in `SYNC_TASKS` (a deferral
  returns at once). Startup preflight also checks host `MemAvailable` (the
  35B's experts live in RAM) - the `EDIT_HOST_RAM_NEEDED` pattern.
- `_gateway_switch`: every switch away from the brain stops the child, as it
  already does for qwen_server. Switch *to* the brain evicts pipelines first.
- CUDA-fault breaker: a child crash/OOM maps to `error_kind`, never a retry loop.
- Tests: `model_of` for the task -> BRAIN; `decide` with the brain active /
  pinned image model; request validation is pure and tested separately.

### T2 - `/api/text` and `/api/text/models`

- `text.py` router, mounted in `main.py`. Follows `audio.py`'s facade shape:
  `auth.require` (`generate` / `read`) -> validate (422) ->
  `worker_busy_reason()` + `admit_sync(BRAIN)` (503 + `Retry-After`) ->
  `generations.begin(kind="text", route="/api/text")` -> enqueue -> wait
  `TEXT_GENERATE_TIMEOUT_S` (240) -> 503 `building` **without revoke** on
  overshoot -> `finish` with reply in `params` (capped ~32 KB).
- 422 cases: missing prompt, params out of range, schema that fails
  `jsonschema` meta-validation, empty `enum`, schema llama.cpp rejects,
  `finish_reason == "length"` with a schema (truncated JSON).
- `/api/text/models`: static, no worker round trip.
- Mint `something2-text` (`read,generate`) with `scripts/mint-key.py`.
- Contract checklist items 1, 2, 3.

### T3 - Reroute the three callers; `llm_engine` leaves the card (atomic)

- One client helper, two paths:
  - API process (`worlds`, `audio_styles`): admit check, enqueue
    `generate_text_task`, wait the caller's existing timeout; refusal or
    timeout -> existing rule fallback with a note saying why.
  - Worker process (`regions` inside a map job): in-process
    `brain_engine.complete` after `_evict_pipelines` + `release_vram_cache`,
    then stop the child before returning to the job. Never brain + pipeline
    at once. Costs one image pipeline reload for the next job.
- Remove the router-specific code (`_llm_model`, `/v1/models` lookups,
  cold-start double attempts) from the three callers.
- Compose: `llm-server` moves to a profile so `make up` does not start it on
  the GPU host. `LLM_URL` dropped from the API/worker env.
- Verify each caller twice - brain admitted (note says LLM) and brain refused
  (note says rules): a world spec, `/api/audio/propose`, a music generation
  with context, a map with `regions`.

### T4 - React "Text" tab

`frontend/src/tabs/Text.tsx`, registered in `App.tsx`. Prompt, optional
system prompt, optional schema (JSON textarea), temperature, max tokens ->
`/api/text`. Shows reply (pretty JSON when a schema was sent), model, timings,
and on 503 the reason + `Retry-After`. History is the Activity tab; no new
storage.

### T5 - Verification run and docs

- Contract checklist items 4 and 5, both variety tests.
- Fill measured numbers into 0013 and contract.md; replace the CLAUDE.md
  tripwire "Qwen3-8B and a diffusion pipeline cannot both hold the card" and the
  project-context model table (it lists a Qwen3-8B that is not on disk); add
  the text row to `.ai/project-context.md`'s model table.
- `make gpu-health` after the busy-card test.

## Scope

Included: the above. Excluded: something2-side changes; streaming; a second
brain or vision; caching; a history read API beyond the Activity tab;
`collector/` (legacy); changing `audio_styles`' prompt or temperature (a
variety fix is a separate decision, after T5's numbers).

## Assumptions (each checked in T0 or T3)

- The worker mounts the models volume (both GGUFs live there). The
  `Qwen3-8B-Q8_0` the older docs name is NOT on disk (checked 2026-10-01).
- A llama.cpp CUDA binary runs on the worker's base and the host driver.
- The 35B's hybrid split leaves card headroom for an 8k context; WSL's 36 GB
  holds its experts with >= 12 GB to spare (IQ4_XS chosen for this).

## Risks

- **Packaging (T0)** - the one unknown that can force a builder stage.
- **Audio generation with context** now pays brain load + evict + audio load
  inside something2's 300 s budget. If T3 measures it near the budget, the
  generation path passes a short timeout and falls back to rules.
- **More rule fallbacks** for world specs (45 s timeout < cold load) and while
  any other model holds the card. Correct by design; visible in the notes.
- **Map jobs with regions** gain one pipeline reload each.
- The solo worker means text is refused for the full length of any long job.

## Acceptance (user-visible)

- something2's Text provider, configured from contract.md's table, gets
  schema-valid JSON, 401 without a key, 422 for an impossible schema, and
  503 (never a CUDA fault) while an image or audio job holds the card.
- The Text tab answers free prompts; every call appears on the Activity tab.
- World specs, audio propose and maps with regions still succeed, with the
  brain or with rules, and `make gpu-health` stays clean throughout.
