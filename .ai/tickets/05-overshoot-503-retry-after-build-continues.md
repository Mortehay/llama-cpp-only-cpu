# An over-budget build is not revoked: `503` + `Retry-After`, and the task closes its own row

## What To Build

Contract D5's overshoot branch, which is the one place the audio facade
deliberately differs from the tile facade (`_serve_tile` revokes the task and
returns `504`). When `AsyncResult.get` times out: leave the task running,
leave the ledger row `running` with its `celery_task_id`, return `503` with a
`Retry-After` estimate and a body that names the artefact - and prove the row
still closes when the task finishes after the request has gone.

## Blocked By

04.

## Scope

- `audio.py` timeout branch: no `revoke`; `503`, `Retry-After: <remaining
  estimate>` (warm seconds-per-track from 0010 minus elapsed, floor 30),
  JSON body `{"kind","name","detail","generation_id","retry_after_s",
  "reason":"building"}`.
- Distinguish the two `503`s in the body: `"reason":"building"` vs
  `"reason":"gpu_faulted"` (from ticket 04's breaker path) vs
  `"reason":"busy"` (`_long_job_ahead`). Same status, three bodies, each with
  `Retry-After`.
- The task's own `finish`/`fail` via `gen_id` (built in 04) is what closes
  the orphaned row; add a defensive path: if the task finds its `gen_id` row
  already closed (e.g. a second request for the same name raced), it does not
  reopen it.
- Duplicate-request handling: a `POST` for a name whose row is `running`
  with a live task does **not** queue a second build; it blocks on the
  existing task id (`AsyncResult(celery_task_id).get(timeout=...)`) and
  returns the same result or the same `503`. Something2 sends no retries,
  but the tab and a second map load might.
- `verify-audio-api.py --overshoot`: run with `AUDIO_GENERATE_TIMEOUT_S=1`
  exported on the API container (restart it), `POST` -> expect `503` with
  `reason: building` and a `Retry-After`; poll `GET /api/audio?name=` until
  the row is `done`; `GET /api/audio/music/<name>` -> `200`; assert the
  worker log shows exactly one load.
- Never a placeholder: assert no `2xx` ever carries empty/silent audio.

## Out Of Scope

Any UI; tuning the `Retry-After` estimate beyond the simple formula.

## Acceptance Criteria

- [ ] `verify-audio-api.py --overshoot` passes: `503` + `Retry-After`, then the row closes itself and the file is served.
- [ ] A second `POST` during the build does not start a second task (worker log, one load; ledger, one `running` row).
- [ ] The three `503` bodies are distinguishable and documented in `contract.md`.
- [ ] `Activity` never shows a permanent "on the worker" row after the run.

## Test Seam

`POST /api/audio` under a 1-second budget, via `verify-audio-api.py
--overshoot`.

## Verification

    # API container with AUDIO_GENERATE_TIMEOUT_S=1
    SPRITE_API_KEY=... python scripts/verify-audio-api.py --overshoot
    docker logs sprite_worker --since 10m | grep -c "Loading.*acestep"   # == 1

## Implementation Notes

- The tile facade's `celery_app.control.revoke(task_id, terminate=True)`
  must NOT be copied. Read `_serve_tile`'s `except` block to see what is
  being deliberately not done.
- Uvicorn runs sync `def` routes in a threadpool; a blocked request holds a
  thread. That is the existing tile behaviour; ticket 10 measures it.

## Review Focus

Review-gated. The orphaned-row close path, the duplicate-request join, and
that no branch ever revokes. This is the code that reads fine and fails
quietly - a reviewer should try to construct the sequence that leaves a row
`running` forever.

## Suggested Route

`/implement`, then `/review-code` with the review focus above quoted.

## Status 2026-09-28 (read from code; no box ticked without the named check)

The no-revoke 503 path and the join rule were verified on **ambience** with `AUDIO_GENERATE_TIMEOUT_S=1` on 2026-09-13 (contract.md, the facade section): one running row, the abandoned build closed its own row, next request a 0.08 s cache read. Not yet via `verify-audio-api.py --overshoot`, and not on music.
