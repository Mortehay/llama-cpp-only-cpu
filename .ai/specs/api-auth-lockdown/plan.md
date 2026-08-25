# Phase 1 - close the API, prove a cross-machine tile render

Status: in progress, started 2026-08-25.

## Why now

The LAN exposure landed before the authentication did. Measured 2026-08-25:
`netsh portproxy` forwards `0.0.0.0:8001` to WSL, this host is
**192.168.0.217** on Wi-Fi, and **two independent machines on the LAN** fetched
`/api/auth/mode` unauthenticated and got
`{"enforced": false, "active_keys": 0, "legacy_token": false}` back. Anyone on
the Wi-Fi could queue GPU work.

`auth.py` is already good - DB-backed keys, scopes, fail-closed on DB error,
honest open-mode reporting. The defect is **coverage**, not design:

- `main.py` has 18 routes and **zero** `authorization` references.
- `a1111.py` still uses the legacy `SPRITE_API_TOKEN` with the
  `if not API_TOKEN: return` silent-open bug that `auth.py` exists to replace.

## The two problems that look like one

`imageUrl()` (`frontend/src/api.ts:369`) resolves to `/images/<name>` - the
`NoStoreStaticFiles` **mount**, not an API route. So there are two distinct
issues, and only one is in scope:

**A. `<img>` against an authed API route - MUST FIX.**
`SheetGenerator.tsx:283` and `Tiles.tsx:200` render
`<img src="/api/jobs/{id}/sheet">` as plain tags. That route already calls
`auth.require`. Browsers send no `Authorization` header on `<img src>`, so
**these 401 the moment the first key exists** - a latent break sitting in the
repo today, invisible only because `enforced` is false.

**B. `<img>` against the `/images` static mount - OUT OF SCOPE, deliberately.**
Roughly 6 sites. Left open, and the reasoning is evidentiary rather than
optimistic:

- names are `uuid4().hex[:12]` (48 bits) or job UUIDs - not guessable;
- `StaticFiles` performs no directory listing;
- every endpoint that *enumerates* names (`/api/cores`, `/api/tasks/recent`,
  `/api/jobs`) becomes authed in this plan.

So after phase 1 an unauthenticated client can neither guess an image name nor
discover one. Protecting `/images` would mean replacing `NoStoreStaticFiles`,
which exists to fix a real stale-sheet caching bug - a bad trade for confidentiality
we get another way.

**The exception that forced a deletion.** `GET /gallery` calls
`fetch_gallery_rows()` and renders image paths and prompts into HTML,
unauthenticated. That *is* the enumeration leak that would break argument B.
React reached gallery parity on 2026-08-25 (commit `3ce440f`), so the route and
`gallery.html` are deleted rather than gated - gating an HTML page a browser
navigates to does not work anyway.

## The ordering property that makes this safe

`auth.require` returns a synthetic principal while `is_enforced()` is false.
**Every code change in slice 1 is inert until the first key exists.** Land it
all, verify nothing broke *in open mode*, then flip.

Rollback at any point: revoke every key, and the API returns to open mode.

## Slice 1 - land everything, still open (no user-visible change)

| Change | Where |
|---|---|
| `auth.require(authorization, scope)` on 14 API routes | `main.py` |
| Delete `_require_auth`, call `auth.require(..., "generate")` | `a1111.py:96` |
| Delete the route and `templates/gallery.html` | `main.py:225` |
| Blob-fetch the 2 `<img>` sites | `SheetGenerator.tsx`, `Tiles.tsx` |
| Read `SPRITE_API_KEY` from env and send it | 9 helper scripts |

### Scope assignment

**`read`** - `/api/compute-info`, `GET /api/settings`, `/api/core-models`,
`/api/edit-capabilities`, `/api/cores`, `/api/task-status/{task_id}`,
`/api/tasks/recent`

**`generate`** - `/api/warm`, `/api/generate_core`, `/api/edit`,
`/api/generate_sheet`, `/api/crop`, `POST /api/task/{id}/retry`,
`DELETE /api/task/{id}`

**`admin`** - `POST /api/settings`

**Open by design** - `/` (a `FileResponse`, embeds no data), `/legacy` (model
names only; kept because `static/app` is gitignored and a fresh clone would
otherwise serve a blank page), `/static`, `/images`, `/api/auth/mode`
(a client must be able to discover that it needs a token without having one).

### Verification

Full UI exercise, `make test-flow`, and
`scripts/verify-something2-contract.sh` - all with `enforced` still false.
Everything must behave exactly as before.

## Slice 2 - flip enforcement

Mint the bootstrap key **out-of-band** with `scripts/mint-key.py`, not over
HTTP: `create_key()` force-adds `admin` to the first key by design, and doing
this over HTTP in open mode works but is the more fragile path.

Paste it into Settings -> "This browser's token". Re-run slice 1's verification.
Now everything must work *with* a token and 401 without one.

## Slice 3 - the cross-machine render

Mint `something2-dev` scoped `read,generate` - **not** `admin`. The something2
machine calls `POST /api/tiles` and polls `GET /api/jobs/{id}` as an
authenticated client.

Green light confirmed 2026-08-25 by the Claude session in that checkout:
`ai_providers` has **0 rows** and `SPRITE_GEN_URL` is `http://sprite-gen:8100`
(in-compose). Nothing over there points at this host, so enforcement breaks
nothing on their side.

### Acceptance criteria

1. A tile PNG rendered on this GPU, requested from another machine, with a
   scoped bearer.
2. The same request with no token returns **401**.
3. The same request with a `read`-only key returns **403**.
4. The browser UI works end to end with a pasted token, images included.
5. `/api/auth/mode` reports `enforced: true`.

## Excluded from phase 1

- Bearer protection for `/images` (argued above).
- A **synchronous tile route** on the A1111 facade so something2-the-app can
  consume tiles. Their `/api/tile-jobs` exists and routes through
  `resolveGenerationTarget()`, but they cannot poll a `202`. Separate ADR.
- The **integration fork**: whether this service should be something2's
  DB-configured remote provider (`ai_providers.base_url` + `auth_token`, set in
  their admin UI) or its env-configured local sprite-gen
  (`SPRITE_GEN_URL` + `SPRITE_GEN_SHARED_SECRET`). Two different credential
  stores. Separate decision.
- Any Plane REST code in this repo. Plane is a **record, not a trigger**:
  tickets go to workspace `something2` / project `SOMET` via the Plane MCP on
  the other machine, driven by human-initiated `SendMessage`. No poller, no
  auto-execute - an agent that executes ticket text is a remote-code path, and
  filing a ticket is a lower bar than being on the Wi-Fi.

## Assumptions

- `/legacy` is acceptable collateral once enforced: its XHR calls all 401, so
  it becomes a dead page, kept only so a fresh clone is not blank.
- Deleting `/gallery` is acceptable given React parity (commit `3ce440f`).
- Helper scripts read a token from `SPRITE_API_KEY`, matching the three that
  already send one.

## Risks

- **The portproxy target is a WSL guest IP and goes stale silently on restart.**
  If slice 3 fails with a connection error, re-run `scripts/lan-expose.ps1`
  elevated *before* debugging auth.
- **`active_key_count()` returns 1 on DB failure**, so a Postgres outage fails
  closed. The UI then looks broken-with-a-valid-token rather than obviously
  down. This is the correct trade, but it is a confusing symptom - check
  Postgres before suspecting the key.
- **Minting flips `is_enforced()` globally and instantly.** Slice 1 must be
  complete and verified before slice 2, or the UI breaks with no way to reach
  Settings to fix it.
