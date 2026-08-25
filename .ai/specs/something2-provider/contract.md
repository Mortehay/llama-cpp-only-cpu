# something2 provider registration

> **CORRECTED 2026-08-23. This document previously said something2 is
> synchronous-only and cannot poll**, citing their `docs/ai-providers.md` and
> their SOMET-334. That was read from published docs, with an explicit caveat
> that their actual calling code had never been reviewed — and the caveat was
> the accurate part. **The project owner states something2 queues the task, does
> not wait, and polls later** — either for one task or for every task it has
> sent. All it needs back immediately is a task number.
>
> That is now implemented as the **job API** below. The A1111 façade is kept for
> single-image txt2img, which genuinely fits in one request.

There are therefore TWO surfaces, for two different jobs:

| Surface | For | Shape |
|---|---|---|
| **Job API** (`/api/jobs`) | Character sheets | Async: submit, get an id, poll |
| A1111 façade (`/sdapi/v1/txt2img`) | One image | Synchronous, 240 s budget |

**Use the job API for anything with actions or directions.** A full character is
~2 hours of GPU time on this hardware (measured, ADR 0005); no HTTP timeout
makes that synchronous, which is why the façade cannot serve it whatever the
timeout is set to.

## Job API

    POST   /api/jobs                 -> 202 {"job_id", "status":"queued",
                                             "cells", "estimated_seconds", "poll"}
    GET    /api/jobs/{id}            -> one job
    GET    /api/jobs?ids=a,b,c       -> a known set        (poll what you sent)
    GET    /api/jobs?since=<iso>     -> anything changed   (poll for updates)
    GET    /api/jobs?status=running  -> by state
    GET    /api/jobs/{id}/sheet      -> the PNG   (409 until done)
    GET    /api/jobs/{id}/atlas      -> the JSON  (409 until done)
    DELETE /api/jobs/{id}            -> cooperative cancel

Submit body (every field optional except `concept_image`):

```json
{
  "concept_image": "meshtest_core.png",
  "actions": ["walk", "attack"],
  "directions": ["s", "se", "e", "ne", "n", "nw", "w", "sw"],
  "frames": 4,
  "cell": "48x64",
  "colors": 24,
  "seed": 0
}
```

Three details worth knowing on the consumer side:

- **`sheet_url` appears only when the job is done.** Treat its presence as
  "ready" rather than string-matching on `status`.
- **Asking for the sheet early returns 409, not 404.** The job exists, it is
  just not finished. A client that conflates those will abandon live jobs.
- **List responses carry `server_time`.** Use it as the next `since` rather than
  the client's own clock.

Verify the whole contract with `scripts/verify-jobs-api.py --submit`.

## A1111 façade (single images only)

Source of truth for this half remains something2's `docs/ai-providers.md`.

## Values to enter in the admin

| Field | Value |
|---|---|
| Base URL | `http://<windows-lan-ip>:8001/sdapi/v1/txt2img` |
| Models path | `/sdapi/v1/sd-models` |
| Models pointer | `$[*].model_name` |
| Image pointer | `images[0]` |
| Auth header | `Authorization` *(only if `SPRITE_API_TOKEN` is set)* |
| Auth token | `Bearer <SPRITE_API_TOKEN>` |

Use the **Windows** LAN address, not the WSL address. WSL2 is NAT'd; the
Windows host forwards these ports only after `scripts/lan-expose.ps1` has been
run elevated. See the README's "Home-network exposure".

## Request template

```json
{
  "prompt": "{{prompt}}",
  "negative_prompt": "",
  "steps": 4,
  "cfg_scale": 0,
  "width": "{{width}}",
  "height": "{{height}}",
  "seed": "{{seed}}",
  "frames": "{{frames}}",
  "override_settings": { "sd_model_checkpoint": "{{model}}" }
}
```

Notes on why this differs from something2's documented example:

- **`steps: 4`, `cfg_scale: 0`** — their example uses A1111's conventional
  `steps: 20, cfg_scale: 7`. Those are wrong for distilled checkpoints like
  SDXL-Turbo, which expect 1-4 steps at zero guidance and produce over-guided
  output otherwise. The façade clamps this defensively for any model whose name
  contains `turbo`/`schnell`/`lightning`, but setting it correctly here is
  clearer. For a non-distilled checkpoint, 20/7 is right.
- **`frames`** is not an A1111 field. It is accepted so a sheet request does not
  422; when `frames > 1` the canvas is widened to `width * frames` and returned
  as one horizontal grid, which something2 then slices itself.
- Quoted numbers (`"{{width}}"`) are intentional — that is how something2
  substitutes. The façade coerces strings, and falls back to per-field defaults
  when a placeholder arrives unsubstituted as `""`.

## Sprite sheet settings

something2 slices grids itself and requires the image to divide **evenly** into
the declared grid, or the job fails. For a 4-frame strip from a single action:

| Setting | Value |
|---|---|
| Sprite sheet | `flat` |
| Columns | `4` |
| Rows | `1` |

With `frames=4` and `width=128`, this service returns a 512x128 PNG.

## Verifying before touching the admin

```bash
# 1. Model discovery — must be a bare JSON array with model_name on each entry.
curl -s http://<host>:8001/sdapi/v1/sd-models | head -c 400

# 2. A generation round-trip. Blocks until done; returns base64 at images[0].
curl -s -X POST http://<host>:8001/sdapi/v1/txt2img \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"green zombie, pixel art","width":"512","height":"512","seed":"-1",
       "override_settings":{"sd_model_checkpoint":"stabilityai/sdxl-turbo"}}' \
  | python3 -c "import sys,json,base64; d=json.load(sys.stdin); \
      open('out.png','wb').write(base64.b64decode(d['images'][0])); \
      print('wrote out.png', d['info'])"
```

Run these from another LAN machine, not just localhost — that is what actually
exercises the portproxy.

## Timeouts

something2's `AI_PROVIDER_GENERATE_TIMEOUT_MS` defaults to 5 minutes and it does
**not** support submit/poll queues (their SOMET-334), so this service blocks
until the image is ready. `A1111_GENERATE_TIMEOUT_S` (default 240) is set below
their limit deliberately, so a slow job surfaces as our 504 with a message
rather than their opaque timeout. If jobs legitimately need longer, raise both.

## Troubleshooting, mapped to their error text

| Their error | Cause here |
|---|---|
| `no image found at response_image_pointer` | Image pointer is not `images[0]` |
| `models_pointer selected objects rather than names` | Pointer must be `$[*].model_name`, not `$[*]` |
| `sheet is NxM px, which does not divide evenly` | `frames` and their columns/rows disagree |
| `refusing to call …: scheme file: is not allowed` | Base URL missing `http://` |
| Connection refused / timeout from another machine | `scripts/lan-expose.ps1` not run, or WSL IP changed since it was |
| 401 | `SPRITE_API_TOKEN` set here but the admin's auth header is missing or malformed |

## Verified against their code (2026-08-25)

Everything above this section was written from something2's published
`docs/ai-providers.md`. Their **calling code has now been read**, at
`Mortehay/something2@main`, and reported back by the Claude session working in
that checkout. Facts below are from the implementation, not the doc.

### The auth token is sent VERBATIM - nothing prepends "Bearer"

`authHeaders(provider)` in both `services/remoteImageProvider.js` and
`services/providerDiscovery.js` is literally
`{ [provider.auth_header_name]: provider.auth_token }`, applied only when both
are non-empty. So the "Auth token" table row above is correct as written: the
admin types the **whole** value including the word `Bearer`. If this service
ever moves to a bare-key header, the admin types a bare key.

### base_url and auth_token are NOT in something2's .env

They are **columns on the `ai_providers` table** - `base_url`,
`auth_header_name`, `auth_token`, plus `model`, `models_path`,
`models_pointer`, `response_image_pointer`, `request_template`, `sheet_*` -
written through their admin Settings UI. `auth_token` is stored in plaintext and
is redacted on every read by `services/aiProviders.js`.

Only these remote-provider *tunables* are env, and all have defaults:
`AI_PROVIDER_GENERATE_TIMEOUT_MS`, `AI_PROVIDER_DISCOVERY_TIMEOUT_MS`,
`AI_PROVIDER_MAX_IMAGE_BYTES`, `AI_PROVIDER_MAX_DISCOVERY_BYTES`,
`AI_PROVIDER_JOB_TTL_MS`, `AI_PROVIDER_MAX_JOBS`.

**There is a second, separate integration path that IS env-configured:**
`SPRITE_GEN_URL`, `SPRITE_GEN_TIMEOUT_MS`, `SPRITE_GEN_SHARED_SECRET`, reached
via `spriteGen.postGenerate` as the fallback when no remote provider resolves.
Which of the two paths this service should occupy is **an open decision** - the
remote-provider path is what this whole spec describes, but the shared-secret
path is the one that matches "put the token in something2's .env".

### Real timeouts and retries

| | Measured |
|---|---|
| Generate | `AbortSignal.timeout(GENERATE_TIMEOUT_MS())` - **5 min default**, not the 240 s assumed above. **Negotiable:** read at call time from `AI_PROVIDER_GENERATE_TIMEOUT_MS` on their backend container, so a longer budget is a config change on their side - not a code change and not a lost job. Tell them the number you need. |
| Discovery / reachability | separate **10 s** budget |
| Retries | **none, anywhere** - one attempt, no backoff |
| Image body cap | 32 MB, streamed and abandoned mid-stream if exceeded |
| Discovery body cap | 2 MB |
| Redirects | max 3 hops, each re-validated, **auth header dropped on a cross-origin hop** |

A non-2xx becomes job error `provider answered <status>`; a transport failure
becomes `could not reach <redacted url>: <msg>`.

### Tiles already have a route on their side

`index.js:2683 startGenerationJob()` is shared by `POST /api/sprite-jobs`,
`/api/entity-jobs` and **`/api/tile-jobs`** (index.js:2897). It resolves
local-vs-remote through `services/generationTarget.js resolveGenerationTarget()`
(precedence: request body -> per-type pin `ai_provider_mode`/`ai_provider_id` ->
active provider -> local sprite-gen), then calls
`remoteImageProvider.startGeneration`. A `sprite_sets` row is inserted either
way; remote rows carry `backend = remote:<provider name>`.

So something2 can already *ask* for a tile. What it cannot do is wait on a
`202` + poll: **SOMET-334 is still unsupported** - no submit/poll path exists,
only the header comment at `remoteImageProvider.js:11` and
`docs/ai-providers.md:268` ("Sync services only"). One POST, one response.

Precision on that claim: what was verified is **the code**, not the Plane
ticket. No submit/poll/fetch path exists on their calling side. The ticket's own
status was never queried - if it matters to phase 2, check `SOMET` directly
rather than trusting "still unsupported" here.

`POST /api/tiles` here is `202` + poll, so **the tile path is not connectable
today**. Closing it means a synchronous tile route on the facade - a cache
reader over the background job - which is its own design decision.

### Multi-frame works now (their SOMET-346)

`frames > 1` expects **one image containing a grid**, cut by
`services/spriteSheet.js manifestForSheet()` using the provider columns
`sheet_layout` (`flat`|`directional`), `sheet_columns`, `sheet_rows`,
`sheet_directions`. The grid must divide the PNG evenly or the job errors before
anything is stored; the result is `atlas.png` + `atlas.json` rather than
`static.png`.

Their own code comment claiming `frames > 1` "fails loudly" against a remote
provider is **stale** - the published doc's "Animated sprites" section is the
correct one.

### Their "job" is not our job

`remoteImageProvider.startGeneration()` creates an **in-memory** job with an
`rmt_` prefix and returns immediately; `runGeneration()` does the POST on a
floating promise and writes status into a registry. That is a bookkeeping handle
around one blocking HTTP call - not a queue. See the "Task" section in
`../../domain.md`.
