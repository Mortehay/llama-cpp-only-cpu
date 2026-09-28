# CLAUDE.md

Routing index. Detail lives in `.ai/`; this file exists so a session does not
have to discover the repo before it can work. Inlined below are only the
tripwires — things that cause a *wrong change*, not merely a slow one.

## What this is

Despite the repo name, this is **not** a CPU LLM cluster. The live component is
`src/sprite_generator/` — FastAPI + Celery generating pixel-art sprite sheets,
with a Jinja/vanilla-JS UI, a React app under `frontend/`, and Postgres history.
`collector/`, `orchestrator/`, `mesh_worker/`, `rig_worker/` are legacy.

**`README.md` is stale — it describes the old project. Do not trust it.**
`.ai/project-context.md` is the current one.

## Where to look

| Question | File |
|---|---|
| Current state, hardware, environment, known-broken | `.ai/project-context.md` |
| A word means three things — "core", "task", "map" | `.ai/domain.md` |
| Why a hard-to-reverse choice was made | `.ai/decisions/` (0001–0009) |
| Model choice, LoRA rules, trigger words | `.ai/decisions/0002` |
| The something2 integration contract | `.ai/specs/something2-provider/contract.md` |
| World specs (the one synchronous surface) | `.ai/specs/worlds/contract.md` |
| Why both trained adapters failed | `.ai/decisions/0009` |
| Bigger LLMs / Colibri; `llm_name` is the *image* model | `.ai/decisions/0011` |
| Qwen-Image-2512 GGUF lane; the downloaded-GGUF verdicts | `.ai/decisions/0012` |

Read the decision before changing anything it covers. They are long because they
record measurements; several correct an earlier claim *in the same document*, so
a passage read alone can be the superseded version. Read the whole file, not a
fragment of it.

## Tripwires

**Environment**
- Docker Engine runs **inside WSL2 Ubuntu**; Docker Desktop is not used. Run
  `make` from inside WSL.
- `.py` edits do **not** hot-reload — neither the API nor the worker, despite
  `uvicorn --reload`. inotify does not cross the 9p mount. Restart the
  container. Jinja templates *are* the exception and reload per request.
- `df` inside WSL reports the VHDX's nominal size, not free disk. Check
  `Get-PSDrive C` on the Windows side before pulling weights.
- After any reboot or `wsl --shutdown`, three things must be re-run:
  `scripts/setup-models-vhd.ps1 -AttachOnly`, `scripts/lan-expose.ps1`
  (elevated), `scripts/wsl-keepalive.ps1`. Symptoms of forgetting: models
  "vanished", LAN 404s, containers exiting 0 with clean logs.
  **something2 reports an error but the API log has no `sdapi` lines? Run
  `make lan-check`** — the portproxy forwards are gone or stale. `make
  lan-expose` re-runs the script elevated (UAC prompt) and re-checks.
- The repo path holds Cyrillic and a space. Quote every path; prefer
  `-LiteralPath` in PowerShell; keep `scripts/*.ps1` **ASCII-only** (5.1 reads
  ANSI without a BOM and a curly quote becomes a parse error).

**GPU — 12 GB, and it is the binding constraint**
- **Generation failing while the API answers 200? Run `make gpu-health` FIRST.**
  A faulted CUDA context in the worker produces three *different* errors
  (allocator assert at 1024, a `c10::Half` dtype error at 512, "device not
  ready"), takes the same wall-clock time as a success, and leaves `/docs`,
  the models route and `make gpu-check` all green. **Stop sending traffic and
  re-probe before restarting** — measured 2026-09-04, one fault cleared on its
  own in under two minutes once a retry storm stopped. Restart only if the
  probe still says FAULTED with the card quiet. The dtype variant invites a
  `pipe.to(float16)` "fix" that would silently produce **black PNGs**. Full
  signature in `.ai/project-context.md`.
- **This card used to have no spare VRAM in normal operation — that is fixed,
  and the old numbers are no longer the baseline.** It was 0 MB free guest-side
  while healthy, because the caching allocator held the whole card and
  `empty_cache()` ran only on pipeline *eviction*, so a run of same-model
  requests never released anything. Two changes closed it (2026-09-10):
  `release_vram_cache()` after each generation, and VAE **tiling** for the
  decode. Steady state now measures ~6.8 GB reserved / ~4.4 GB headroom
  guest-side and ~7.0 GB host-side, down from 11.6 GB.
  - **`enable_vae_slicing` was never the mitigation it was documented as.**
    Slicing splits the *batch*; this service decodes one image per call, so it
    saved nothing. Tiling is the spatial one. Enabling it is not enough on its
    own: the guard is `latent > tile_latent_min_size` and SDXL ships 128,
    exactly a 1024² latent, so **both** thresholds must be lowered. Measured
    peak 11,588 MB → 8,148 MB, no seams, no slowdown. `VAE_TILING=0` disables.
  - Callers still need backoff, but the worker no longer depends on their
    manners: a CUDA fault trips a breaker (`GPU_FAULT_COOLDOWN_S`, default 90s)
    that refuses work in ~0.2 ms and re-probes in-process before resuming. The
    API turns that into **503 + `Retry-After`**, not 500.
- **`nvidia-smi` inside WSL does not see host VRAM**, so guest and host
  readings diverge — once measured 1342 MiB guest against 11,601 MiB host on
  the same 12 GB card. Still true; but a high host figure is no longer the
  normal warm state (see above), so treat ~11.6 GB as worth investigating
  rather than as the healthy baseline.
  Check `(Get-Counter '\GPU Process Memory(*)\Dedicated Usage')` on Windows.
- The Qwen3-8B GGUF and a diffusion pipeline **cannot both hold the card**.
  `--sleep-idle-seconds 120` is what lets them share it.
- Celery runs `--pool=solo`. `tasks.py` must not touch CUDA at import —
  `torch.cuda.is_available()` is fork-safe, `get_device_properties()` is not.

**Models — easy to get silently wrong**
- Step 2 is **locked to SD1.5**. `poses.py` authors COCO-18 skeletons against
  `control_v11p_sd15_openpose`; the SDXL ControlNet measured 3.6x weaker.
- Anything matching `DISTILLED_MARKERS` (`turbo`, `schnell`, `lightning`,
  `lcm`) runs at guidance 0, which makes **every negative prompt a silent
  no-op**. Prefer non-distilled checkpoints.
- Family detection reads `model_index.json` (`_is_sdxl_checkpoint`,
  `tasks.py`), **not** the repo name — which is why
  `stable-diffusion-xl-base-1.0` loads at all. The name heuristic survives only
  as a *fallback* for an unreadable config (offline, cold cache), so it is not
  dead code. `0002` states the old rule at line ~95 and corrects itself at
  ~199; read both before quoting either.
- `"<base>+<lora>"` is a valid model string. A LoRA only fuses onto the base it
  was trained against — check `base_model` in the repo card first.
- Each checkpoint needs **its own** trigger word. A foreign trigger is inert at
  best; `PixelartFSS` actively requests a 4-character sheet. Triggers live in
  **two** places — `core_models.trigger_for()` and `tasks.CORE_TRIGGERS` — and
  nothing keeps them in sync.
- Adding a checkpoint means `core_models.CORE_MODELS` (the roster the UI and
  `a1111` both read) and, if something2 should see it, `a1111.KNOWN_MODELS`.
  The old "two dropdowns hardcoded in `templates/index.html`" is gone — the
  template iterates the roster, and `core_models.py` also reports whether a
  checkpoint is actually on disk, which archiving to cold storage makes matter.
- `POSED_STRENGTH` below ~0.75 silently disables pose conditioning while every
  log line still says "pose-conditioned".
- **The UI's preselected model and `core_models.default_model()` differ on
  purpose** (0012). The UI defaults to Qwen-Image-2512 (~144 s/image);
  `default_model()` is also `main.py`'s `Form` default for every script that
  omits `llm_name`, so it stays SDXL + nerijs. Do not "fix" them to match. Qwen
  is also not in `a1111.KNOWN_MODELS` - it cannot meet something2's 240 s.
- **A GGUF's `general.architecture` label is not evidence.** Four of the
  2026-09 downloads were mislabelled (`wan`, `qwen_image`, `pig`). Read the
  tensor prefixes and block count before wiring a file.
- **Never put an image-model GGUF directly in `/models` or one folder below
  it.** `llm_engine` (llama.cpp, `--models-dir /models`) lists every `.gguf`
  there, and every folder directly holding one, as a *chat* model, and
  `worlds._llm_model()` takes the first entry. Image GGUFs live two levels deep
  in `/models/image-gguf/transformers/` (0012). The router scans only at
  startup - restart `llm_engine` after moving files, then check `/v1/models`.
- **"no frame, no border, no card" in a POSITIVE prompt asks for a frame.** A
  CLIP text encoder has no negation operator, so each of those nouns lands in
  the conditioning. something2's entity prompts arrive carrying nine of them;
  `split_negations()` moves them to the negative prompt on the cutout path,
  where classifier-free guidance can actually subtract (SDXL base at guidance
  7 — not a distilled checkpoint at 0). Measured on 12 fixed seeds: 3/12
  multi-object → 2/12, with every affected case improving. It reduces the
  problem, it does not solve it — some subjects still come back as item sheets
  and `_isolate_largest_sprite` keeps the largest, so "obsidian boots" can
  return a knight.
- Two fixes for that were measured and **rejected**: adding `NEGATIVE_SINGLE`
  changed 3/12 → 3/12, and dropping `lora_scale` to 0.7 made it markedly worse
  (2/12 → 5/12). The pixel-art adapter at full strength holds the subject
  together; it is not what imposes the grid. Do not reach for that dial.
- **A contact sheet is a property of the SEED, not the prompt** — so the cutout
  path regenerates on a new seed instead of refusing (`ENTITY_CUTOUT_ATTEMPTS`,
  default 3). Measured on the exact prompt+negative+seed something2 kept
  resending: 1 of 7 seeds rejected, 6 clean. something2 pins its seed and
  retries the *same* one — six times per item — so identical input gave
  identical rejection forever, ~108s of GPU each. Replaying all 8 stuck
  requests after the change: **8/8 succeeded**, ~29s average.
  - A retry deliberately **overrides a caller-pinned seed**; the seed actually
    used is returned, and the facade forwards it in `info`. A pinned seed that
    yields an item sheet can never yield anything else.
  - Only `cutout_failed` is retried. A `gpu_faulted` result must NOT be —
    that is the storm the breaker exists to stop.
  - `ENTITY_CUTOUT_BUDGET_S` (180s) bounds the loop by wall clock, not just by
    attempt count, and refuses to *start* an attempt it expects to overshoot.
    Attempts are not the same size: two warm ones measured 31.3s, but the same
    two after a cold load measured 242.2s — past something2's 240s ceiling.
  - Still imperfect: an accepted cutout can be the largest item *of* a sheet
    (measured 24.6–53.8% kept on some passes), so a subject can be the wrong
    item. Retrying toward a higher kept-target would trade latency for asset
    quality; the guard's own floor stays at the measured 20%.

**The something2 contract**
- **Synchronous only.** Submit/poll is explicitly unsupported on their side, so
  anything 202-and-poll is not connectable today. The A1111 facade must stay a
  blocking read. World specs are the one surface that is genuinely sync.

**Training data — three verdict columns, and they answer different questions**
- `usable` gates style-profile derivation (measurement-grade). `trainable` is
  `measure.judge_trainable`, deliberately permissive (blank, tiny,
  extreme-strip). `audit_trainable` is `audit-character-refs.py --apply`,
  deliberately strict (contact sheets, baked checkers, near-duplicates), and
  covers `sprite`/`core` only — NULL elsewhere means "not judged", not "failed".
- **Training reads `trainable AND audit_trainable IS NOT FALSE`.** Filtering on
  `trainable` alone silently re-admits everything 0009 rejected.
- Those were **one column until migration 016**, and whichever writer ran last
  won: on 2026-09-04 a single `remeasure-all` un-rejected 131 core and 84 sprite
  references with no warning. Split now, so remeasure is safe to run.
- **`remeasure-all` outruns its client.** A full pass over 2,565 references
  exceeds 900s; curl returns `HTTP 000` and **the handler keeps running**. Read
  a timeout as "still going", scope with `?kind=`, and don't click twice.
- The gates for the other two kinds do not discriminate: `tile` marks
  2001 of 2001 `usable`, `map` rejects 112 of 114. For tiles, `trainable` is the
  only number that means anything; for maps, see the terrain-separation note in
  `.ai/specs/maps/`.

**Auth — enforcement is ON, and this is no longer a future cliff**
- **Enforcement is already ON.** `GET /api/auth/mode` returned
  `{"enforced": true, "active_keys": 2}` on 2026-09-04. Anything
  unauthenticated is 401 **now**, not "the moment a key exists" — plain
  `<img src="/api/jobs/{id}/sheet">` tags and every helper script without a
  bearer included. Mint and revoke out-of-band with `scripts/mint-key.py`.
- So a bare `curl` against the API fails, and that is the expected answer, not a
  broken service. `.ai/specs/api-auth-lockdown/plan.md` has the background, but
  **recount before trusting any figure in it**: its route counts predate
  `auth.py` gaining coverage.
- `a1111.py` no longer reads the legacy shared secret; `auth.py` owns it, and an
  unset `SPRITE_API_TOKEN` no longer means "no auth". The `if not API_TOKEN:
  return` silent-open bug is fixed — the constant was deleted so the shortcut
  cannot be reinstated by accident. Do not reintroduce it.

## Conventions

- Durable knowledge goes in `.ai/`, in git, reviewable — not into chat memory.
- Generated reports stay summarised in Markdown; the per-row data belongs in the
  `--json` sibling. `.ai/specs/training-data/reference-audit.md` is the pattern.
- Prefer the narrowest verification that actually runs. Say what you did not run.
