# something2's admin has the values to register the audio provider, and a LAN machine verifies it

## What To Build

The connector-facing half: the "values to enter" table for something2's
(future) audio provider kind, the exact request template and response
pointer, the timeout they must set, and a verifier that runs from another
LAN machine through the portproxy - plus one measurement of how many API
threads a burst pins.

## Blocked By

05, 09.

## Scope

- `.ai/specs/audio/contract.md` gains "Values to enter in the admin":
  base URL `http://192.168.0.217:8001/api/audio`, auth header
  `Authorization` with the **whole** value `Bearer <key>` (their
  `authHeaders` sends it verbatim - see the something2 contract), request
  template with their placeholders (`{{prompt}}`, `{{seed}}`, plus literal
  `kind`, `name`, `style`, `duration_s`), response pointer `audio[0]`,
  models/styles discovery path `/api/audio/styles` with pointer
  `$[*].id`, and `AI_PROVIDER_GENERATE_TIMEOUT_MS` = (0010's cold
  seconds-per-track + one SDXL eviction) x 1.5, stated as a number.
- The three `503` bodies and the `404` documented from their side ("what
  their operator sees" table, like the something2 contract's troubleshooting
  table).
- `verify-audio-api.py --lan <ip>`: same checks as `--submit` against the
  Windows LAN address from a second machine (this is what exercises
  `scripts/lan-expose.ps1`), plus `--burst N`: N concurrent `POST`s for N
  new names; assert exactly one build at a time on the worker, N-1 `503
  busy` or queued-and-served per the 05 join rule, and report how long each
  API thread was held.
- A `read`-scoped key for something2 minted with `scripts/mint-key.py
  --name something2-audio --scopes read,generate` (documented, not
  committed).

## Out Of Scope

Any code on something2's side; their audio provider kind is their ticket.
Rate limiting or quotas here.

## Acceptance Criteria

- [ ] The values table is complete enough that someone with no context can fill their admin form.
- [ ] `verify-audio-api.py --lan 192.168.0.217` passes from a second LAN machine.
- [ ] `--burst 5` completes without a `500`, without a GPU fault, and the thread-holding numbers are recorded in `contract.md`.
- [ ] The timeout number handed to something2 is written down with the measurement it came from.

## Test Seam

The public surface over the LAN, via `verify-audio-api.py --lan`.

## Verification

    # from another LAN machine, with SPRITE_API_KEY set
    python scripts/verify-audio-api.py --lan 192.168.0.217 --burst 5

## Implementation Notes

- After any WSL restart `scripts/lan-expose.ps1` (elevated) must be re-run
  or the LAN check fails for reasons unrelated to this ticket.
- Their provider has **no retries**; the `503` + `Retry-After` is advisory
  to their operator until their audio kind implements a wait.

## Review Focus

That the template uses their placeholder syntax (quoted numbers are how they
substitute) and that the timeout number is derived, not guessed.

## Suggested Route

`/implement`, then `/review-code` on the contract table.

## Status 2026-09-28

Written: `contract.md` "For something2's operator" - admin values for music,
ambience and sfx (base URLs, auth, discovery pointers, `audio[0]` vs
`audio[*]`, request templates), the timeout to keep (300000 ms; ours answers
503 at 240 s with the build still running), and the operator error table.
It says plainly that **their side has no audio provider kind yet** - the
image provider decodes `images[0]` as a PNG and cannot carry audio.

`verify-audio-api.py` gained `--kind sfx` (one-shot checks: a base64 OGG per
variant, no loop tags, <= 3.5 s, onset <= 10 ms, cache hit, retro-footstep
refusal) and `--burst N`. Dry-run confirmed it reaches the API and gets 401
without a key.

- [ ] Every check needs a key: `scripts/mint-key.py --name something2-audio
      --scopes read,generate` - to be run by the owner, not the agent.
- [ ] `--lan` from a second LAN machine; `--burst 5`.
