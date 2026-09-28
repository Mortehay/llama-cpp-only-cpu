# Model gateway - plan

Status: **approved and implemented 2026-09-28** (backend; UIs in progress).
Two owner answers changed the draft below: D4 = release after idle (as
recommended), and **every GPU job is gated**, including sheets, edit, audio
and training - the "Non-goals" entry on fixed-model jobs is superseded. See
"As built" at the end.

## Goal

One **active image model** at a time, chosen from the UI. Jobs for that model
run immediately; jobs for other models wait instead of forcing a reload
mid-queue. Both UIs show the active model and a "switching, please wait"
popup; something2 gets an immediate, readable 503 instead of a timeout.

## Decisions taken (owner, 2026-09-28)

| Question | Answer |
|---|---|
| Model semantics | Global active model, set in the UI |
| something2 when its model is not active | Immediate `503` + `Retry-After`, detail names the active model |
| "5-10 s" | What the popup shows as the expected switch time |
| UIs | Both - React `frontend/` and Jinja (`templates/` + `generator.js`) |
| Deferred-job release | `MODEL_SWITCH_IDLE_S` (default 300) in `.env` - see D4 |

## Behaviour

- **Normal switch** (UI "Switch"): allowed only when no job for the current
  active model is queued or running. Otherwise refused with the count.
- **Forced switch** (UI "Force switch"): takes effect immediately. Jobs already
  queued for the old model are marked **Deferred** and wait (D4).
- **A job for a non-active model**:
  - UI routes: accepted and queued; the worker defers it (row shows
    "Deferred - waiting for <model>").
  - something2 (`/sdapi/v1/txt2img`, tile and entity facades): if the active
    model is idle and not force-locked, the gateway switches to the requested
    model and the request proceeds (switch time counts inside the 285 s budget).
    Otherwise `503` + `Retry-After` immediately.
- **In-flight something2 request hit by a forced switch**: returned as `503`
  at once rather than held until its 285 s timeout.

## Design

**`model_gateway.py`** - Redis only, safe to import in the API and the worker
(no CUDA). Keys:
- `gw:active` - `{model, since, forced, last_activity}`
- `gw:switching` - `{from, to, started, expected_s}` while the new model loads
- `gw:pending` - hash `task_id -> {model, queued_at}`
- `gw:load_s:<model>` - last measured load time, for the popup estimate

**Decision logic is a pure function** (`decide(state, pending, model, now)`),
so it is unit-tested without Redis or a GPU. Redis I/O is a thin shell.

**Enqueue**: the gated `.delay()` sites go through
`model_gateway.enqueue(task, model, ...)`, which registers the task in
`gw:pending`. Gated tasks are the ones with a **caller-chosen image model**:

| Task | Site |
|---|---|
| `generate_core_task` | `main.py` `/api/generate_core`, retry |
| `generate_raw_task` | `a1111.py` txt2img |
| `build_tile_job` | `tiles.py` `queue_tile` (UI and tile facade) |
| `build_map_job`, `resolve_map_props` | `maps.py`, `map_tasks.py` |
| `generate_spritesheet_task` | `main.py` (no UI caller, gated anyway) |
| `warm_model_task` | `main.py` `/api/warm` - also the switch's loader |

**Defer in the worker**: each gated task calls `model_gateway.defer_reason()`
first. If it should wait, it updates its row and does
`raise self.retry(countdown=MODEL_GATEWAY_RECHECK_S, max_retries=None)`.
Celery `retry` keeps the **same task id**, so UI polling and `task.get()`
are unaffected. A `task_postrun` signal removes the id from `gw:pending`; the
boot reaper clears ids older than a ceiling so a crashed worker cannot pin
the gateway.

**Switch execution**: setting `gw:active` enqueues `warm_model_task(new)`;
`gw:switching` is set until it finishes and records the measured load time.
For a `gguf:` model there is nothing resident to load - the switch only evicts
the SDXL pipeline, so it completes in about a second.

**Status endpoint**: `GET /api/model-gateway` -> active, switching (+ expected
and elapsed seconds), pending per model, deferred count. `POST
/api/model-gateway/switch {model, force}` (scope: admin/generate - match
`auth.py`). `sd-models` gains a non-A1111 `active: true` field on the active
entry.

**UI**:
- React: an active-model pill in `App.tsx` nav (polls the status endpoint every
  3 s), a switch control, and a blocking modal modelled on `CropModal` while
  `switching` is set.
- Jinja: the same pill in `base.html` nav and a modal reusing `.modal-overlay`,
  driven from `generator.js`.

**Env**: `MODEL_SWITCH_IDLE_S=300`, `MODEL_GATEWAY_RECHECK_S=5`, added to
`.env.example` **and** both services' `environment:` lists in
`docker-compose.yml` (the API and worker do not use `env_file`, so a key only
in `.env` would be invisible).

## D4 - when deferred jobs run (confirm)

Recommended: deferred jobs are released when the active model has had **no
queued or running work for `MODEL_SWITCH_IDLE_S`** (300 s). The gateway then
switches back to their model on its own.

Why not "as soon as the new model's queue empties": someone generating by hand
leaves gaps of tens of seconds between clicks. The queue empties in every gap,
a deferred job for the old model sneaks in, the card reloads it, and the next
click reloads the new one - the thrash the gateway exists to stop.

## Honest numbers for the popup

"5-10 s" holds only for a warm switch between LoRAs on an already-resident
SDXL base. A cold SDXL load has measured minutes. Switching *to* Qwen is about
1 s, but each Qwen image still reloads its model (~55 s) inside the job. The
popup shows the last measured load for that model, falling back to 10 s.

## Non-goals (v1)

- **Fixed-model jobs are not gated**: sheet builds (Qwen-Edit), `/api/edit`
  (FLUX Kontext), audio, training. They run as they do today and evict what
  they need; the next gated job reloads the active model. Gating them would
  mean audio and sheets wait behind an SDXL lock, which nobody asked for.
- No priorities, no reordering beyond deferral.
- something2's own UI cannot be changed; what it shows is our 503 detail text.

## Verification

- Unit: `decide()` table - normal switch refused and allowed, forced switch,
  deferral, release after idle, something2 auto-switch vs 503, stale pending.
  Run as `make test-model-gateway` in the worker image.
- Live: a forced switch with jobs queued (rows show Deferred, then run after
  the idle window); a normal switch refused while busy; popup visible in both
  UIs; `/sdapi/v1/txt2img` for a non-active model returns 503 with a
  `Retry-After` header (needs a minted key).

## Risks

- `self.retry` holds ETA messages in the solo worker's memory; a worker
  restart re-delivers them, and the boot reaper must not fail them.
- 11 enqueue sites. Missing one means that path silently bypasses the gateway.
  The test greps for bare `.delay(` on the gated task names.

## As built (2026-09-28)

- **Gate placement.** Not at the 16 enqueue sites. `tasks.GatedTask` is the
  Celery app's `task_cls`, so every task inherits it; its `__call__` asks
  `model_gateway` before the body for any task named in `model_gateway.GATED`
  (12 tasks). Registration is the `before_task_publish` signal, which also
  catches `send_task` by name and the worker's own enqueue of
  `resolve_map_props`. Removal is `task_postrun` unless the state is `RETRY`.
- **Labels for fixed-model jobs:** `qwen-image-edit` (sheets),
  `flux-kontext-edit` (`/api/edit`), `audio:ace-step` (music),
  `audio:stable-audio` (ambience, sfx), `training:sdxl-lora`.
- **Rules** (`model_gateway.decide`, pure): run on the active model; otherwise
  defer while the active model has queued or running work; a pinned model (UI
  switch) also holds until `MODEL_SWITCH_IDLE_S` idle; an unpinned model yields
  to a job that has waited `MODEL_SWITCH_IDLE_S` (starvation cap); among
  waiters the oldest job's model goes next.
- **Normal switch** is refused (409) while ANY job is pending - the literal
  "only if no pending tasks". Force always proceeds.
- **Sync callers.** `generate_raw_task` (txt2img, entity) returns
  `error_kind: model_deferred` instead of retrying, and the facade maps it to
  503 + `Retry-After`. The tile facade and both audio routes check
  `admit_sync` before queueing. Audio tasks are NOT sync-returning: a forced
  switch after admission makes the audio facade wait until its own timeout,
  and the build still completes later and closes its rows (the existing D5
  no-revoke design).
- **Worker restart.** `_gateway_reap` drops pending entries marked running.
  Deferred entries stay: their retry messages are redelivered by the broker,
  which on Redis can take up to the visibility timeout (1 h). The existing
  stranded-task reaper will already have failed their `sprite_images` rows;
  a later successful run overwrites that.
- **Tests:** `make test-model-gateway` - 15 `decide` cases, `retry_after_s`,
  `model_of` labels, and a static check that every GATED name is declared.
