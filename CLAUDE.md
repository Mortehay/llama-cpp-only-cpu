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
- The repo path holds Cyrillic and a space. Quote every path; prefer
  `-LiteralPath` in PowerShell; keep `scripts/*.ps1` **ASCII-only** (5.1 reads
  ANSI without a BOM and a curly quote becomes a parse error).

**GPU — 12 GB, and it is the binding constraint**
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
  `tasks.py`), **not** the repo name. `0002` still describes the old name
  heuristic in places — that constraint is gone, and
  `stable-diffusion-xl-base-1.0` loads because of it.
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

**The something2 contract**
- **Synchronous only.** Submit/poll is explicitly unsupported on their side, so
  anything 202-and-poll is not connectable today. The A1111 facade must stay a
  blocking read. World specs are the one surface that is genuinely sync.

**Training data — two flags, and one of them gets silently reverted**
- `usable` and `trainable` are **different verdicts** and `training.py` filters
  on `trainable` only. `measure.judge_trainable` is deliberately permissive
  (blank, tiny, extreme-strip); `scripts/audit-character-refs.py --apply` writes
  a much stricter verdict over the top for `sprite` and `core`.
- **`POST /api/references/remeasure-all` reverts that audit.** It recomputes
  `trainable` from the permissive gate with no knowledge that `--apply` ran.
  Measured 2026-09-04: one call un-rejected 131 core and 84 sprite references.
  **Always follow it with `make audit-refs-apply`**, and check the counts.
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
