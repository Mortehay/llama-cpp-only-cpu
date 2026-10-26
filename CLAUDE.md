# CLAUDE.md

Routing index. Detail lives in `.ai/`; this file exists so a session does not
have to discover the repo before it can work. Inlined below are only the
tripwires — things that cause a *wrong change*, not merely a slow one.

## What this is

Despite the repo name, this is **not** a CPU LLM cluster. The live component is
`src/sprite_generator/` — FastAPI + Celery generating pixel-art sprite sheets,
with a Jinja/vanilla-JS UI, a React app under `frontend/`, and Postgres history.
`collector/`, `orchestrator/`, `mesh_worker/`, `rig_worker/` are legacy.

**`README.md` is accurate now, and this note has been wrong in both
directions** (re-checked 2026-10-23 by reading the file, not by trusting the
note). First it warned the README "describes the old project", which was
stale. Then it listed four wrong parts — and **all four have since been
fixed**: the `HF_TOKEN` line names only `stabilityai/stable-audio-open-1.0`,
"First run" step 5 says *"NOTHING ELSE DOES THIS — not `make up`, not `make
build`"*, the legacy services sit in their own table under a "Was" column
stating none of them is on the sprite path, and the `models.txt` rows are
marked legacy. **Verify against the file before repeating any complaint from
here.** `.ai/project-context.md` remains the authority on current state.

## Where to look

| Question | File |
|---|---|
| Current state, hardware, environment, known-broken | `.ai/project-context.md` |
| A word means three things — "core", "task", "map"; also the four kinds of "model" on disk and the four answers to "what models does this have" | `.ai/domain.md` |
| Why a hard-to-reverse choice was made | `.ai/decisions/` (0001–0019) |
| Model choice, LoRA rules, trigger words | `.ai/decisions/0002` |
| The something2 integration contract | `.ai/specs/something2-provider/contract.md` |
| World specs (the one synchronous surface) | `.ai/specs/worlds/contract.md` |
| Why both trained adapters failed | `.ai/decisions/0009` |
| Bigger LLMs / Colibri; `llm_name` is the *image* model | `.ai/decisions/0011` |
| Qwen-Image-2512 GGUF lane; the downloaded-GGUF verdicts | `.ai/decisions/0012` |
| How weights get onto a machine; `make provision` | `.ai/decisions/0014`, `.ai/specs/model-provisioning/` |
| `/api/text` brain for something2; why `llm_engine` leaves the GPU | `.ai/decisions/0013`, `.ai/specs/something2-text/contract.md` |
| `/api/edit`: which editor runs, `EDIT_ENABLED`, and what editing destroys | `.ai/decisions/0015` |
| A concept that is not a character (tree, crate, banner) | `.ai/specs/prop-concepts/` |
| Per-frame sheet checks, redraws, the frames store, mirrored west rows | `.ai/specs/sheet-quality/contract.md` |
| Why the 3D conveyor stays retired; T-pose feedstock measured; what runs off-card | `.ai/decisions/0016` |
| C: is full again; what actually fills it; why compacting is not the fix | `.ai/decisions/0018`, `compact-wsl.md` |
| Why a sprite looks nothing like its concept; the 48x64 cell | `.ai/decisions/0019`, `scripts/cell-size-cost.py` |

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
  - **`scripts/build-sheet.py` is the opposite trap: it reloads whether you
    want it to or not.** Each conveyor stage is a fresh `python` subprocess
    reading the file off disk, so editing it while a job is running changes
    what the NEXT stage runs. A half-finished edit killed a job at
    `actions-denoise` with `NameError: name 'stage_actions_explicit_redraw'
    is not defined` — the dispatch table is built inside `main()`, so a name
    that is briefly missing breaks *every* stage, not just the one being
    edited. Check `docker top sprite_worker | grep build-sheet` first.
- `df` inside WSL reports the VHDX's nominal size, not free disk. Check
  `Get-PSDrive C` on the Windows side before pulling weights.
  - **"Docker usage" means `/var/lib/containerd` + `/var/lib/docker`, summed.**
    This box uses the containerd image store, so images and build cache live
    under `containerd` and `/var/lib/docker` holds little but volumes. Reading
    the latter alone makes Docker look negligible — that is how
    `compact-wsl.md` came to claim weights dominated at "126 GB against 22 GB
    of Docker" and nearly sent a session to migrate 126 GB of weights to D:
    for nothing. Measured 2026-10-23: **containerd 151 + docker 22 = 173 GB**
    against **149 GB** for all of `/home`. Docker wins, and it is the part
    that churns.
  - **Compaction is not always available as a fix, and it is not the fix.**
    A 46-minute compaction that morning left C: with 108.7 GB free; the 0016
    rig-gate retest put it back to **16.3 GB free the same day** (+90 GB, from
    two 20-GB-class experiment images plus the build cache regrowing to
    81.22 GB). By then ext4 held 325 GB in a 342.8 GB file — ~18 GB of slack,
    so there was nothing to reclaim. **Delete inside first; delete dead
    images BEFORE `builder prune`**, or the prune nets out layers still shared
    with them (measured 56 GB freed one way, 13.66 GB reclaimable the other).
    Retention rules in `0018`.
  - **Use `wsl -d Ubuntu -u root`, not `sudo`.** `sudo` here requires a
    password, and with no tty it **hangs forever** instead of failing — a `du`
    over `/var/lib` that never returns is this, not a slow disk.
  - A second project, `aod-gpubox`, shares this dockerd and holds ~32 GB
    (`aod_ollama_models` is 22.65 GB). Subtract it before blaming the sprite
    stack for disk use.
- After any reboot or `wsl --shutdown`, re-run `scripts/wsl-keepalive.ps1`
  and `make lan-expose` (UAC prompt; it runs `lan-check` after), then
  `make up`. Symptoms of forgetting: LAN 404s, containers exiting 0 with clean
  logs. `setup-models-vhd.ps1 -AttachOnly` no longer applies: since the drive
  reshuffle there is no `D:\wsl-models.vhdx`, and `MODELS_DIR` is on the
  distro root (checked 2026-09-28). Do NOT run it without `-AttachOnly` "to
  fix" that - it would create an empty disk.
  **something2 reports an error but the API log has no `sdapi` lines? Run
  `make lan-check`** — the portproxy forwards are gone or stale. `make
  lan-expose` re-runs the script elevated (UAC prompt) and re-checks.
  - **The keepalive is not only an after-reboot step, and your own `wsl.exe`
    command is not a substitute for it.** WSL2 sweeps the distro once no
    `wsl.exe` client is attached, so a session that merely runs commands
    holds it up only while a command is in flight. Hit 2026-10-23 with no
    reboot involved: **every container exited 255 at the same second**,
    mid-16 GB download, and `uptime` inside the distro read `up 0 min`.
    Simultaneous 255 across all services is this, not an OOM — check
    `Get-Process wsl` (none running is the tell) before reading any
    container log. Recovery is `wsl-keepalive.ps1` then `make up`; it needs
    no admin, unlike *registering* it with `-Install`.
  - The portproxy forwards may well still be fine afterwards, because they
    target the distro's IP and that often survives the restart. **Run
    `make lan-check` (no UAC) and only reach for `lan-expose` if it fails** —
    on 2026-10-23 the IP was unchanged and all five forwards still answered.
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
- **The brain is a gateway model, not a service** (0013, since 2026-10-01).
  `brain_engine.py` runs llama-server as a child of the worker, label
  `brain:<id>`; every switch away and every pipeline eviction stops it. Brain
  + pipeline on the card at once is the wedge - so **never call llama-server
  from a new place without going through `text.run_text`** (API process) or
  `brain_engine.complete_in_job` (inside a worker job, evicts first). The
  default 35B is hybrid (`--n-cpu-moe 24`, ~8.3 GB VRAM, experts in RAM);
  `--n-cpu-moe 16` measured 0.3 GB of card left - do not "speed it up" there.
  `llm_engine` is retired to the `legacy-llm` compose profile.
- **A model gateway decides who gets the card** (`model_gateway.py`, spec
  `.ai/specs/model-gateway/`). A job for a non-active model is *deferred* — a
  Celery retry of the same task id every `MODEL_GATEWAY_RECHECK_S`, row text
  "Deferred - ..." — and something2 gets 503 + `Retry-After`. A model picked in
  the UI holds the card until `MODEL_SWITCH_IDLE_S` (300) idle. Neither is a
  hang or an outage. Every task goes through `GatedTask` (`task_cls`), so a
  **new GPU task must be added to `model_gateway.GATED`** or it bypasses the
  gateway silently.
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
  omits `llm_name`, so it stays SDXL + nerijs. Do not "fix" them to match.
  Qwen **is** offered to something2 (0012 D2, reversed 2026-09-28 at the
  owner's request), square sizes only. `KNOWN_MODELS[0]` is the fast Q2_K +
  Lightning variant (~107-113 s cold). The Q3_K_M entry is ~250 s against the
  285 s facade budget, with no room for a cutout retry - accepted, not a bug.
- **A GGUF's `general.architecture` label is not evidence.** Four of the
  2026-09 downloads were mislabelled (`wan`, `qwen_image`, `pig`). Read the
  tensor prefixes and block count before wiring a file —
  `python3 scripts/gguf-head.py <path-or-url>` does it in a few MB, remote
  files included, so a candidate can be identified before it is fetched and
  compared against one already on disk. Check `general.file_type` too: that is
  the quant level.
- **Keep GGUFs two levels below `/models`** (`image-gguf/...`,
  `brain-gguf/...`). The retired `llm_engine` router listed every `.gguf` one
  level deep as a *chat* model; it only starts with `--profile legacy-llm` now,
  but anyone reviving it gets that trap back. Brains are wired by explicit path
  in `brain_engine.BRAINS`, never discovered.
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
- **"transparent background" in a POSITIVE prompt makes the model PAINT the
  transparency checkerboard** into the pixels, and the flood-fill cutout
  (one corner colour) cannot clear a two-tone grid. Measured 2026-09-29 on
  fast Qwen, production path: 9/9 opaque "successes" (3–18% transparent) vs
  12/12 clean with "plain white background". `rewrite_transparency()` swaps
  the phrase before generating, and asking for transparency makes the request
  a cutout. `CUTOUT_MIN_CLEAR` (25%) refuses what still comes back opaque,
  which triggers a retry on a new seed (entities AND UI cores). Do not
  colour-match the checker instead: white subjects are the same colour as its
  light squares (`scripts/key-checkerboard.py` learned that). Spec
  `.ai/specs/transparent-cutouts/`.
- **Object cutouts use BiRefNet, not the flood fill**, on the CUDA worker
  (`cutout.py`, called from `remove_background`). The flood fill cannot clear
  walls, scenes or enclosed pockets; it is only the fallback (API process,
  `CUTOUT_ENGINE=floodfill`, load failure). Its soft mask is snapped to 0/255
  for pixel art. It is parked in RAM between calls **on purpose** - resident,
  it would break the slow Qwen core's 10 GB free-VRAM check. Anything that
  ends up in the web RPG game must be transparent; tiles/terrain are the
  deliberate exception.
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

**Provisioning — `models.toml`, and the gaps it exposed**
- **`models.toml` is the manifest; `make provision` is the only thing that
  fetches weights.** Not `make up`, not `make build` (two containers on the
  same `/models` cache deadlock on HF file locks). Tiers: `required` by
  default (56.4 GB), `tier=all` for 103 GB, `dry=1` to plan, `audit=1` for real sizes. Spec
  `.ai/specs/model-provisioning/`, decision `0014`.
- **`models.txt` has no live consumers** (checked 2026-10-13). Its one active
  entry, `Qwen2.5-3B-Instruct-Q4_K_M.gguf`, was superseded by the brain lane in
  0013 — `worlds.py` calls `text.run_text`, not llama-server — so the file now
  serves only the retired `legacy-llm` profile. **Do not add models to it.**
  `up`/`build`/`rebuild`/`rebuild-clean` used to run its downloader on every
  invocation, which is why `make up` printed *"[SKIPPED]
  Qwen2.5-3B-Instruct-Q4_K_M.gguf already present"* while fetching none of the
  56.4 GB that matters. Removed 2026-10-13; only `make sync-models` calls it
  now. **If you see that line again, something re-added the step.**
- **The Models tab is where weights are seen and queued**
  (`frontend/src/tabs/Models.tsx`, `GET /api/models/status`). Brains, image
  models and audio are three separate tables **on purpose** — tokens exist
  only for brains, and audio rows record no model at all. Its chart palette
  was validated against the dark surface, not picked; the day/token charts are
  deliberately two charts because the measures sit 124x apart. Details in
  `.ai/specs/model-provisioning/`.
- **The `downloader` container is no longer idle — it is the background
  download lane.** `scripts/provision-daemon.py` is the **single** consumer of
  `model_downloads` (migration 020), fed by `POST /api/models/download`. Never
  add a second writer to `/models`: both it and `make provision` take
  `flock(/models/.provision.lock)`, and that lock is the only thing standing
  between you and the HF file-lock deadlock. A download is **not** a Celery
  task on purpose — `sprite-worker` is `--pool=solo`, so one would block every
  generation for its whole duration.
  - **A hand-rolled `docker exec` download outlives the client that started
    it.** Ctrl-C, a killed background task or a dropped `exec -T` does not
    stop the python inside the container. Starting the "same" download again
    therefore gives you two streams fighting for the link. Measured
    2026-10-23: **1.0 MiB/s with two running, 9.0 MiB/s with one** — so a
    slow download is evidence of contention long before it is evidence of a
    slow link. `ps aux | grep hf_hub_download` inside the container first;
    do not conclude anything about bandwidth until it prints exactly one.
  - **Point `hf_hub_download(local_dir=...)` INSIDE `/models`.** A staging
    dir on the container's own filesystem (`/tmp/dl`) makes the final
    `shutil.move` a cross-filesystem **copy** of the whole file, not a
    rename — a 16 GB second pass during which the destination exists at the
    wrong size. Anything watching for the file to appear fires early; watch
    for the *size* or for the process to exit.
- **`make up` warns instead of downloading.** A `models-warn` step compares
  `models.toml` against `MODELS_DIR` — host-only, no network — and prints the
  missing *required* entries plus "Run: make provision". Silent when complete,
  and never fatal. It exists because the failure this whole lane targets is a
  stack that comes up healthy with an empty `/models`.
- **Adding a model means adding a `models.toml` stanza.** `make
  test-models-manifest` walks the source for repo ids, `gguf:` stems and brain
  paths and fails if one is neither declared nor in `[[dead]]`. That test is
  the point — it is the thing `models.txt` never had.
- **Two entries save under a different filename than the Hub publishes** (the
  VL-8B brain, the Q3_K_M image quant), via the `as` field. Fetching either by
  hand with `hf download` leaves a file that exists and is never found.
- **"Model" means four things on disk, each needing a different fetch.** An
  `hf-cache` repo must land in `HF_HUB_CACHE` layout (`models--org--name/`) or
  diffusers cannot resolve it — `hf download --local-dir` produces the wrong
  shape. An `hf-file` GGUF needs an explicit destination two levels deep. The
  acestep checkpoint tree has its own layout. `loras/*.safetensors` were
  trained here and **cannot be downloaded at all**. See `.ai/domain.md`.
- **Both openpose ControlNets ARE cached - the "absent" note above it was
  wrong** (re-checked 2026-10-22 by listing the snapshots, not by
  `is_cached`). `lllyasviel/control_v11p_sd15_openpose` holds
  `diffusion_pytorch_model.safetensors` (1.44 GB) plus fp16 variants, and
  `thibaud/controlnet-openpose-sdxl-1.0` holds `OpenPoseXL2.safetensors`
  (5.0 GB) and the rank-256 control-lora. Both were fetched 2026-10-12, the
  day before the note claiming they were missing. So **pose-conditioned step 2
  is not blocked on weights**; if it fails, read the actual error rather than
  assuming the repo is absent. Step 2 is still locked to SD1.5 for the reason
  above (the SDXL ControlNet measured 3.6x weaker), and nothing here has been
  run end to end to confirm the pose path works - only that the weights exist.
- **`stabilityai/sdxl-turbo` was a live default and is not cached — fixed
  2026-10-13.** `generate_sheet`'s `Form` default, `main.py`'s retry fallback,
  `model_gateway.GATED` and both `tasks.py` signatures named it, though 0002
  dropped it for cause, so a request omitting `llm_name` could only fail. All
  five now resolve `default_model()`. The name survives **only** in a legacy
  cache-directory normalisation (`models--stabilityai--sdxl-turbo`); do not
  read that as the model being available. The `default_model()` claim above now
  holds for sheets too.
- **The D: cold archive is gone.** `/mnt/d/wsl-model-archive` does not exist,
  so `scripts/archive-models.sh restore` recovers nothing and the "28 GB parked
  2026-08-23" is not retrievable. Missing weights must come from the Hub again.

**Editing — `/api/edit` repairs nothing, and the flag that gates it has a trap**
- **`EDIT_ENABLE` (no `D`) is read by nothing.** The code reads `EDIT_ENABLED`.
  Set the short spelling and `env` inside the container shows it while
  `/api/edit` keeps answering 503 `enable_with: EDIT_ENABLED=1`. It must be
  set on **both** `sprite-generator` and `sprite-worker` - the API answers that
  503 from the same constant the worker reads - and the containers
  **recreated**, not restarted, for compose to re-read the environment.
- **The editor is Qwen-Image-Edit-2511 (GGUF), not FLUX Kontext** (0015, since
  2026-10-21). `tasks.EDIT_ENGINE` selects it. The Kontext weights are
  `[[dead]]` in `models.toml` and `make provision` does not fetch them, so
  `EDIT_ENGINE=flux-nf4` can only fail offline - which is exactly how the old
  default behaved once the flag was finally spelled right.
- **An edit re-renders the WHOLE image. It is not a masked inpaint**, so it
  cannot be used to repair a concept. Measured: a 39%-coverage subject held
  its framing, but 8-14% ones were recomposed 2-3x and one invented content
  that was never in the source. The VAE round trip also replaces the pixel
  grid with gradients (26,847 -> 74,579 distinct colours on one prop), and
  cropping to the subject first fixes the hallucination but not that. It is
  **seed-dependent**: the same request on the same source returned 61.3%
  transparent on one seed and an unusable 12.7% zoom on another.
- **It returns RGB.** Without the cutout step an edit of a 60.8%-transparent
  core comes back 0.0% transparent, and a sheet built from it carries that
  backdrop into the game. `_edit_qwen` matches the source's alpha by default;
  the result reports `transparent`, and a caller that needs it should gate on
  that number rather than assume.

**Turnaround rows — the west rows are answered by mirroring; the front quarters are not**
- **`left side` returns the same facing as `right side`.** Measured on two
  subjects and two seeds (`qwen_edit`, by `AZIMUTH_HINTS`): on a humanoid the
  `w` panel is a near-duplicate of `e` (|e−w| 4.7 against |e−mirror(w)| 24.6).
  No prompt wording fixes it; three were tried and the best left the pair
  still unmirrored. Dropping the generic hint is worse — both panels go
  near-frontal.
- **FIXED for the turnaround stage** (0015-era work, `MIRRORED_AZIMUTHS`):
  `w`/`nw`/`sw` are rendered from a *mirrored source* and flipped back, since
  `M(render(M(C), east)) == render(C, west)` exactly. Costs nothing — the
  cells API already takes a per-cell image. Verified through `POST /api/jobs`:
  `dir_e` faces screen-right, `dir_w` faces screen-left, |e−mirror(w)| 6.27.
- **ANSWERED for the action stage, by not rendering those rows at all**
  (2026-10-22). `_action_cell` re-conditioned on `dir_{d}.png` with
  `angle_prompt(d)` and the western rows came back facing east — measured
  frame 0: |e−w| 9.91 against |e−mirror(w)| 25.28. Nothing fixed that, so
  compose now **flips the finished `e`/`ne`/`se` cells** into `w`/`nw`/`sw`
  (`build-sheet.MIRROR_WEST`, default on; `SHEET_MIRROR_WEST=0` renders all
  eight the old, wrong way). A 336-cell sheet renders 210.
  - The cost is the handedness the owner rejected this for before: a
    one-handed sword changes hands between the `e` row and the `w` row.
    Accepted because something2 already mirrors `e`/`ne`/`se` in game. Every
    flipped row carries `mirrored_from` and `handedness: "swapped"` in the
    atlas so a consumer that does NOT mirror can see it.
  - **Flip the FITTED cell, never the 512px render.** `pixelate` centres with
    `(cell_w - width) // 2`, so flipping before the fit rounds the sprite one
    pixel left in both cells and the pair is off by one — in game, a sprite
    that jumps sideways when the character turns around. Caught by
    `scripts/check-sheet-assembly.py`, which exists to keep catching it.
- **`se` and `sw` are still the FRONT view, and it starts at the turnaround.**
  Measured on both evidence jobs: |s−se| 12.8/16.4 against |s−e| 52.2/60.4,
  and |se−sw| 18.1/33.9 against |se−mirror(sw)| 43.6/53.7 — the front quarters
  collapse onto the front view and are near-duplicates of each other rather
  than mirrors. The mirrored-source trick cannot help here, which is why `sw`
  already uses it and still equals `se`: the collapse is toward a
  near-symmetric front view, so the mirror maps it onto itself. The back
  quarters are fine (`nw` is a correct mirror of `ne`). Unfixed - but a
  candidate hint now exists, see below.
- **The quarter-view hint was validated and REJECTED - it overshoots**
  (`scripts/try-quarter-hints.py`; candidate run 2026-10-22, validation
  2026-10-24 on 2 subjects x 2 seeds, default Q2_K). Appending
  `", halfway between front and profile"` to the front-quarter azimuths moves
  `se` off the front view on **4 of 4** subject/seed pairs - the direction is
  not in doubt. It lands in the wrong place.

  | subject/seed | `se_frac` control -> hint | `sw` a true mirror? |
  |---|---|---|
  | 212d56d7 @1916017834 | 0.526 -> 0.754 | yes -> **no** |
  | 212d56d7 @4242 | 0.348 -> **0.859** | no -> yes |
  | 2ff8af36 @1916017834 | 0.272 -> 0.642 | no -> no |
  | 2ff8af36 @4242 | 0.268 -> 0.682 | no -> yes |

  A three-quarter view is ~0.5 on the front->profile path. The hint lands
  **0.642-0.859**, so at the top of that range `se` is effectively a PROFILE
  and reads as a duplicate of `e` - trading "se looks like the front" for
  "se looks like the side". A hint is unconditional, and the first row shows
  the cost: that pair was **already a correct three-quarter at 0.526** and
  the hint pushed it to 0.754, breaking its `sw` mirror in the process.
  Mirroring is not fixed either: 1 of 4 before, 2 of 4 after, which is noise
  at this sample size.
  - **The earlier figures in this note were wrong.** It recorded
    "0.22 -> 0.45"; the run's own JSON says **0.272 -> 0.642**, and a
    byte-identical re-run in 2026-10-24 reproduced 38.72 / 0.642 exactly. The
    pipeline is deterministic here, so a number that does not reproduce is a
    transcription error, not noise.
  - **The collapse is NOT universal**, which one subject could never show:
    control `se_frac` ranges **0.268-0.526** across subject and seed. Some
    sheets already have a usable front quarter. Any fix has to not damage
    those, which is precisely what this wording fails.
  - Next wording to try should aim **weaker**, targeting 0.5 rather than
    exceeding it. Two earlier wordings failed the other way: one merely
    ZOOMED (figure 1.62x, a close-up the `close_up` check would flag) while
    scoring the *largest* raw pixel difference of all - which is why
    **`|s-se|` alone must never be the metric**; read it beside the figure
    fraction and the path position.
- **The camera clause is NOT the cause, and the obvious fixes are all dead.**
  Measured on one known-good left-facing `dir_w` at the action stage's own
  `angles_scale=0.4`, with the real side-family pose, four variants: the
  current `angle_prompt("w")`, the eastern `angle_prompt("e")`, no camera
  clause at all, and the mirrored-source trick that fixes the turnaround.
  **All four came back facing right.** So do not bother dropping, swapping or
  weakening the camera clause, and do not re-try mirroring here - it was
  implemented and reverted, and the revert also removed a pose-text swap that
  would have desynchronised the `w` gait from the `e` gait for nothing.
  `qwen_edit.swap_lr` and `flip` are kept for whoever tries again.
- **Next hypothesis, untested: the POSE text sets the facing.** The side
  family's frame 0 is "left leg forward heel down, **right arm forward**,
  contact", and the renders put the arm out to screen-right regardless of
  view. A front-family pose ("right knee raised high, left foot well off the
  ground") did NOT break the facing in the same harness - A and B held left -
  which is what points at the lateralised words rather than the camera.
  Beware: that earlier run also used `angles_scale=1.0` by mistake, so it is
  suggestive, not evidence.
- **Do not "fix" this by mirroring the finished east render.** It swaps the
  hands of anything asymmetric, so a sword changes sides between rows.

**Group offload DOES work with a GGUF transformer — sequential is the one that doesn't**
- `denoise_only` refuses `enable_sequential_cpu_offload` because the hook
  moves parameters to the meta device and a `GGUFParameter` loses its
  `quant_type`. **Do not generalise that to all offload.** Measured
  2026-10-23 (`scripts/probe-gguf-offload.py`): block-level
  `enable_group_offload` returns valid latents and drops the transformer's
  resident footprint from **9408 MB to 2028 MB**, leaving ~10.9 GB free, for
  **1.57x** the denoise time (30.1 s → 47.1 s per cell).
- That is enough headroom for Q4_K_M, Q5_K_M or Q6_K. The limits become host
  RAM (41 GB in WSL) and PCIe bandwidth, not VRAM — and the Lightning LoRA's
  4 steps bound the streaming cost, because weights cross the bus four times
  per image rather than fifty.
- **No longer a projection — Q6_K was run through the real conveyor**
  (2026-10-23, `QWEN_GROUP_OFFLOAD=1`, blocks=4). A **15.70 GiB** quant on a
  12 GB card: **10.7 GiB free after placement**, 10.9 GiB at the end of a
  48-cell pass, **68.8 s/cell all-in** against ~30.1 s/cell for Q2_K
  resident. VRAM is not the constraint on this lane any more; wall clock is.
  Quality verdict in the editor-quant section below.
- **It is wired into both `denoise_cells` and `denoise_only`** (`_place`), so
  a quant chosen with `QWEN_EDIT_GGUF_FILE` loads on the sheet path AND on
  `/api/edit`. Without the flag a 15.70 GiB file OOMs on load, and that
  failure reads like a broken quant rather than a missing flag.
- **It is NOT CPU compute, and the brain's trick does not transfer.**
  Group offload keeps weights in host RAM and streams them to the card; every
  multiply still happens on the GPU. `--n-cpu-moe` works for the brain
  because that model is **MoE** — sparse, a few experts per token. The editor
  is a **dense DiT**: 1920 tensors across 60 `transformer_blocks`, no expert
  tensors in the file, so every block runs for every step. Moving any share
  of it to CPU runs that share at CPU speed with no sparsity to recover it.
  Do not reach for a hybrid split here.
- **`blocks` is a SPEED dial, not a memory dial**: peak VRAM is identical at
  2, 4 and 8, because the resident set is activations + VAE + LoRAs rather
  than held blocks. It saturates at 4; `blocks=2` is 50% slower for nothing.
  `enable_model_cpu_offload` saves 26 MB and is not worth using.
- **Encode before the transformer exists** when measuring this. A probe that
  encoded after placing the pipeline asked for a ~6 GB text encoder on top of
  a ~7 GB transformer and failed reproducibly with `CUDA driver error: device
  not ready` — which reads as "resident is broken" and was really "this asked
  for 13 GB". Every offload mode passed only because the transformer was not
  fully resident at that moment.

**The editor quant — three have been measured and Q2_K is still the default**
- **Q6_K ran a full sheet and is NOT adopted** (2026-10-24, needs
  `QWEN_GROUP_OFFLOAD=1`; it is 15.70 GiB). Against the Q2_K baseline
  `2ff8af36`, same concept and seed, 210 cells, **post-redraw**: flagged
  27 → **31**, identity 22 → **10**, facing 8 → **21**, close_up 3 → 3,
  border 1 → 0, build 3h07 → **6h51**, `sheet_scale` 0.089844 → **0.092184**.
  - **It does not make sprites bigger**, which was the reason to want it.
    First-attempt `close_up` 26 → 4 was expected to lift the pinned
    sheet-wide scale; it moved **2.6% linear**, because that scale is the
    MINIMUM over every cell and the 3 surviving bloated cells still bind it.
  - **A 48-cell gate overstated it in both directions.** `close_up` read
    14 → 1, but the redraw loop takes Q2_K to 3 as well — so that was a
    speed saving, not a quality gain. And subset `facing` read flat (9 → 8)
    against the sheet's 8 → 21. Full reasoning in the `models.toml` stanza
    and `scripts/quant-ab.py`.
  - **Identity halving (22 → 10) is the one durable win** and survives the
    redraws. Worth revisiting **after** a cell-size change (`0019`), because
    at 96x128 that gain is visible where at 48x64 it is not.
- **"facing got worse" may mean the CHECK woke up.** It fires only when the
  cell's lean AND its turnaround reference both exceed `LEAN_MIN` (0.04); a
  weak reference leaves it **dormant** and the cell passes whichever way it
  faces. A sharper quant strengthens references (median |view_lean| 0.1816 →
  0.2026) and starts judging cells that were never judged. Pairing the two
  sheets on first attempts: 141 cells active in both (17 → 25 fails), **24
  dormant under Q2_K became active under Q6_K** (9 fail), 12 went the other
  way (4 had failed). So the real regression is ~+8, not +13. **57 of 210
  baseline cells had a dormant facing check** — "facing: 0" has never meant
  "every cell faces correctly".

**Q3_K_S exists, fits, and is NOT the default either**
- The OOM on record (~24 MiB short, mid-denoise) is **Q3_K_M's**, a larger
  file. **Q3_K_S fits**: 806 MiB headroom through a full 7-action sheet. Do
  not read the Q3_K_M measurement as ruling out every Q3.
- It buys the 512px render, not the sprite: masked colour distance from the
  concept 0.0336 → **0.0108** at 512, but 0.1322 → 0.1196 at 48x64, which is
  no visible change. Whatever is lost at 48px is spatial, and no quant
  restores it.
- A full sheet (`d3083b62` vs `2ff8af36`, same concept and seed) came out
  **worse overall**: 27 → 45 flagged, because identity improved (22 → 15,
  median 0.0103 → 0.0069) while **facing got worse (8 → 31)** and the build
  took 29% longer. The facing regression is not noise — the turnaround
  anchors got STRONGER under it (`dir_ne` −0.047 → −0.105) — but it is
  unexplained. Opt in with the **ABSOLUTE** path in `QWEN_EDIT_GGUF_FILE`; a
  bare name goes to the Hub and fails offline with a message that reads like
  a missing download.
- **`identity` is a colour histogram and is spatially blind.** It scored
  48x64 and 96x128 as equally far from the concept while the 96px render is
  plainly more recognisable. It catches drift, which is what it was built
  for; it cannot see "the shape became mush". A passing `identity` is not
  evidence a sprite is usable at its cell size — `.ai/specs/sheet-quality/`
- **"The sprite looks nothing like the concept" is a CELL-SIZE problem, and
  it is the INTERIOR that is lost, not the outline**
  (`scripts/cell-size-cost.py`, all 210 cells of `2ff8af36`, 2026-10-24, CPU
  only - it reads the alpha already baked into `cut_*.png`, so it is safe to
  run while a build holds the card).

  | cell | silhouette IoU | interior detail vs 512 | sprite px |
  |---|---|---|---|
  | **48x64** | 0.6121 | **0.1665** | 1499 |
  | 64x86 | 0.6035 | 0.2346 | 2745 |
  | 96x128 | 0.6025 | 0.3582 | 6216 |
  | 128x170 | 0.6028 | 0.4566 | 10960 |

  At the production cell **~17% of interior detail survives**; 96x128 more
  than doubles it (+115%). **Silhouette IoU is FLAT across every size**, so
  resolution buys nothing for the outline and a silhouette metric cannot see
  this - which is why the first version of that script, built expecting the
  outline to be the casualty, found nothing. No quant touches it either:
  Q3_K_S tripled closeness at 512 (0.0336 -> 0.0108) and moved 48x64 by
  0.1322 -> 0.1196.
  - Those figures fit each sprite **alone**. Production pins the sheet-wide
    MINIMUM fit (0.089844 on that sheet), which is smaller still, so real
    cells sit **below** the table. Cell size is a something2 contract
    question - their atlas consumer and in-game scale both change with it -
    so raise it rather than tuning the generator against it.
  §7.

**Sheet cells are checked per frame now — and the checks measure the RAW render**
- `sheet_qc.py` scores every cell before it is fitted into 48x64, because the
  fitted cell cannot answer the question. Pre-fit figure height is **478-512
  of 512 on every cell** of both evidence jobs (the LoRA frames edge to edge),
  so a chest close-up and a full body are the same height and any
  height-based rule passes everything. **Area** against that direction's
  turnaround view is the signal: clean 1.07-1.09x, close-ups 1.85-1.98x.
- **The sheet-wide scale is the MINIMUM fit over all cells**, bound by the
  widest one — so one bloated cell shrinks all 336, and measuring after the
  fit measures the contamination instead of the cause. It is pinned in
  `config.json` after the first compose so a redraw cannot resize the whole
  sheet; when a redrawn cell genuinely needs a smaller scale the sheet refits
  and the atlas says `refit: true`. Do not "fix" the min-scale rule.
- **The sheet path keys with BiRefNet now, not the corner flood fill.** That
  was the last place still using it, and it structurally cannot clear a glow,
  a scene or a painted transparency checkerboard — measured 9-39% of frame
  removed on the cells the owner complained about, with no redraw. Compose
  reads `cut_*.png` and falls back to `key_background(cell_*.png)` only for
  old work directories or a box without the weights.
- **Backdrop is a SET DIFFERENCE, not a difference of fractions.** BiRefNet
  also keeps pixels the flood fill dropped, so subtracting the two percentages
  nets them off: on `attack se f4` that reads 7.3% and passes an 8% cut while
  the real backdrop-and-not-figure area is 9.0%. That cell is one the owner
  named, and the first version of the module let it through on that
  arithmetic.
- **`scripts/score-sheet.py --anti-goals` is the gate that stops the checks
  becoming a check that always passes.** It asserts the five cells the owner
  named still fire for the right reason and exits non-zero otherwise. Run it
  after touching a threshold. Scoring 672 cells takes ~25 min, almost all
  BiRefNet at ~1.8 s/cell — **not while a sheet job is on the card.**
- **Redraws are batched by round, not row by row**, and that is forced by the
  hardware: a cell is checked from decoded pixels and the VAE cannot share the
  card with a resident transformer, so checking after each row would cost a
  ~90 s transformer reload per row (~35 rows). Rounds reuse the embeddings,
  because a seed re-roll changes neither the prompt nor the image.
- A new GPU task still has to go in `model_gateway.GATED` —
  `tasks.redraw_frames_job` is registered there under the same label as
  `build_sheet_job`, because it is the same transformer.
- **A sheet stays redrawable only while `images/_job_<id>/` survives, and
  nothing deletes it today** (checked 2026-10-22). `POST
  /api/jobs/{id}/redraw` re-reads `config.json`, every `cut_*.png`,
  `dir_<d>.png` and `scores.json`; without them it answers 410 and the sheet
  can only be rebuilt. **Adding a cleanup sweep is therefore deciding how
  long a sheet stays repairable** — say so rather than letting something2's
  admin UI discover it. They are not small: 490 MB for a 7-action sheet,
  4.3 GB across all jobs. **71% of that is `embeds_*.pt`, which nothing reads
  once the build is finished** — pruning those plus `att*_*.png` would take a
  sheet to ~105 MB with redraw intact. `make prune-job-cache` does it, opt-in
  and dry by default; see `.ai/specs/sheet-quality/contract.md` §5d.
  - **Never prune while a build is on the card.** "Nothing reads
    `embeds_*.pt` once the build is finished" is the operative clause: a
    RUNNING build reads them at every denoise and redraw stage, so
    `apply=1` mid-build deletes the conditioning out from under it. Check
    `docker top sprite_worker | grep build-sheet` first, same as for editing
    `build-sheet.py`.
  - **Nor compact the VHDX** (`compact-wsl.md`, `0018`). Compaction requires
    `wsl --shutdown`, which stops dockerd and every container — so it does
    not merely risk the build, it ends it. And `actions-denoise` holds all
    210 latents in memory and writes cells only when the stage finishes, so
    a 4-hour stage interrupted at hour 3 leaves **nothing**; the first Q6_K
    attempt lost exactly that when C: filled at ~2h10. Disk maintenance on
    this box is a between-builds job, and the same `docker top` check
    answers it.
- **Do not edit `action_prompts.json` while a build is on the card either**,
  for the same reason the `build-sheet.py` tripwire exists: encode and
  denoise are separate processes. The pose text is baked into `embeds_act.pt`
  at `actions-encode`, so changing it mid-build leaves every later stage
  rendering from embeddings that no longer match the file — and the scores,
  the atlas and anyone reading the JSON afterwards all disagree about what
  was asked for. Queue pose-text work for between builds.
- **The checks cannot see a row that does not ANIMATE** — every one of them
  scores a cell against a reference, never against its neighbours, so six
  near-identical frames pass all five. `atlas.rows[].motion` reports
  `mean_step`, `min_step` and `duplicates` instead. **Reported, not checked,
  on purpose**: motion is a property of a ROW and a static row needs
  different pose text, not another seed — a redraw structurally cannot fix
  it, so wiring this into the redraw loop would only spend GPU.
- **`back` pose lists were verbatim copies of `front` for every action**
  (fixed 2026-10-22). From behind the torso occludes anything held in front
  of it, so those frames had no silhouette to change. Measured: back rows
  moved least of the three families (12.20 against 19.08 for side), `walk n`
  — a walk cycle — managed 4.40, and `attack n` had two frames 0.39 apart.
  Also **never repeat a pose inside one list**: the conveyor renders one
  image per distinct (direction, pose) and copies it, so the old walk list
  naming "both feet flat..." at frames 2 AND 5 froze a third of the cycle.
  Rules in `action_prompts.json` `_readme.sheet_frames.back_is_not_front`.
  **`idle` was fixed the same way on 2026-10-24**; `sway` got a `back`
  family on 2026-10-26 and **keeps its original `side` list, on the
  measurement** - see the sway entry at the end of this block.
  - It was broken in BOTH ways at once, and worse than "the `n` row is
    frozen". Its one `side` list named `"standing still, arms at sides"` at
    frames **0 AND 2** - byte-identical strings, so one image was rendered
    and copied - and with no `back` family, `n`/`ne`/`nw` drove a
    back-facing camera from side-view pose text. Six entries, **five
    distinct**.
  - Rewritten `side` (a breath cycle plus weight moving between the feet)
    and added `back` (shoulder blades, the back arching and rounding, heels
    and hips - nothing that needs the chest or hands to be visible).
    Verified by rendering at the baseline's concept and seed, scored the way
    `row_motion` does: **`idle n` 0.93 -> 14.68 mean step, 4 duplicate pairs
    -> 0**; `idle e` 2.29 -> 19.59. 0 of 12 cells flagged.
  - **Those figures are on 512px `cut_*.png`, NOT on fitted 48x64 cells**,
    so they are comparable to each other and to nothing else. The family
    averages above (side 19.08, front 14.84, back 12.20) and the atlas's
    own `motion` are measured on FITTED cells. Do not read "idle e 19.59"
    as idle moving more than a walk - that mistake was made and caught.
  - **No head turns in any idle pose, deliberately.** `sheet_qc.facing`
    reads head lean against the direction's turnaround reference, so a pose
    that moves the head per frame trips the facing check for a cosmetic
    gain - and facing is already the noisiest check on these sheets.
- **`sway` looked like the same bug and was NOT** (2026-10-26). Its `side`
  list also had a byte-identical pair - frames 1 and 3, both `"upright,
  settling back through centre"` - so it read as idle's twin. It is not:
  - **`duplicates` counts CONSECUTIVE pairs only**, and 1 and 3 are not
    adjacent, so identical non-adjacent frames never register. Both baseline
    sway rows reported 0 duplicates while genuinely containing two copies of
    one image. Do not treat `duplicates: 0` as "no repeated frames".
  - Baseline `sway n` already moved at **10.88** - near the back-family
    average of 12.20 - against `idle n`'s 0.93. The row was not frozen, so
    the string comparison predicted a defect that was not there.
  - Adding `back` helped: **`sway n` 10.88 -> 17.58**. Rewriting `side` HURT:
    **`sway e` 16.84 -> 14.01**, so the original list was restored and only
    `back` was kept. A sway is a pendulum and its centre frames SHOULD look
    alike; that duplicate is defensible where a walk cycle's is not.
  - The lesson is the general one: **a pose list that looks wrong is a
    hypothesis, not a defect.** Render and measure before rewriting, and
    measure the families separately - `n` exercises `back`, `e` exercises
    `side`, so one run scores both independently.
- **`/api/action-catalog` now reports what a redraw actually does**
  (`redraw.frame_context`, `.ordering`, `.max_frames`). It exists because a
  consumer UI captioned a redraw "redrawn in order, context: f0, f3" and the
  box does neither — frames are rendered in ONE batched pass and neighbours
  condition nothing. Keep it honest if that ever changes.
- **The sheet conveyor has NO negative prompt and cannot be given one.**
  `denoise_cells` passes `true_cfg_scale=1.0`, so there is no
  classifier-free guidance to subtract with. Spell glow and invented props
  came from `action_prompts.json` asking for them by name ("a glowing orb
  held at the fingertips", "holding an object out in front"); the fix was to
  rewrite the pose text, stripped from `cast` and `use` on 2026-10-22. Adding
  "no glow, no aura, no particles" to the positive prompt is the "no frame,
  no border, no card" trap and makes it worse. The rule and the measured
  2026-08-25 fix it supersedes are in that file's
  `_readme.sheet_frames.effects_stay_on_the_body`.

**Prop concepts — do not reword `_core_prompt` to fix them**
- `_core_prompt` asks for "one character alone, full body, standing" whatever
  the subject is, and that wording **is** the isolation. A "prop mode" that
  replaced it was built and measured over 12 fixed seeds on 2026-10-21: three
  replacements all scored 6-7/12 forests and contact sheets against the
  current wording's 2-3/12. Reverted; the numbers are in `_core_prompt` and
  `.ai/specs/prop-concepts/`. The token "character" is carrying it.
- The working route for a non-character subject is `/sdapi/v1/txt2img`
  (`generate_raw_task`), which rewrites nothing, with `cutout: true`. Verified
  2026-10-21: one generation, one isolated tree, transparent, no person. **A
  prop ends there - it is a finished static asset.**
- **Never send a prop through `POST /api/jobs`.** Measured 2026-10-21: that
  tree, `actions: ["idle"]`, one frame, seven minutes of GPU - and the sheet
  came back as **a green humanoid in a dark shirt and trousers**. The
  conveyor's turnaround and action stages drive the editor with pose and
  camera prompts, and the angles LoRA pins pose, so a subject with no pose has
  a body invented for it. `concept.judge` will NOT catch this: it measures
  isolation, not species, and a clean cut-out prop passes it. The gate also
  runs in the worker, so `/api/jobs` answers 202 either way.

**The something2 contract**
- **NOT synchronous only — this entry was the superseded claim** (caught
  2026-10-24). It said submit/poll "is explicitly unsupported on their side,
  so anything 202-and-poll is not connectable today". The contract itself
  **corrected that on 2026-08-23**: the original read their published docs
  and SOMET-334 with an explicit caveat that their calling code had never
  been reviewed, and the caveat was the accurate part. The owner states
  something2 **queues the task, does not wait, and polls later**.
- **There are TWO surfaces.** `/api/jobs` is async submit/poll and is what
  character sheets go through; the A1111 facade (`/sdapi/v1/txt2img`) is the
  synchronous one, for single images inside a 240 s budget, and *it* must
  stay a blocking read. World specs are sync too. Read
  `.ai/specs/something2-provider/contract.md` before repeating anything from
  here - it opens with that correction.
- **`cell` is already in the documented submit body**, so a larger sprite
  cell needs no API work at all: `{"cell": "96x128"}` instead of `"48x64"`
  on the request something2 already sends (`0019`, and `JobSpec.cell` is
  live on `/openapi.json`).

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
