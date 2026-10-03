# something2 text provider - `/api/text` contract

Status: **Live 2026-10-01**, verified end to end with the `something2-text`
key. Decision and measurements: [0013](../../decisions/0013-gated-brain.md).
Typical: 35B ~14 s for a ~200-token JSON answer warm, 107-143 s cold from
disk; 8B ~4 s warm, ~37 s cold. The SFX-entity check ran against a stand-in
schema (entity / cue enum / prompt) - the client's real one was not available
here; re-run item 1 with it.

## Who calls it

- something2's **AI Providers "Text" entry** - the same generic remote-provider
  client as image and audio (`ai_providers` row: `base_url`, auth header sent
  verbatim, `models_path` + `models_pointer`, `request_template`, response
  pointer). See [something2-provider/contract.md](../something2-provider/contract.md)
  "Verified against their code" for how that client behaves: **5 min** generate
  timeout, **10 s** discovery timeout, **no retries**.
- The generator's own React **Text** tab (free prompts, same endpoint).
- In-process (not over HTTP), rerouted from `llm_engine` - see 0013 and
  [plan.md](plan.md): `worlds.py` biome plans, `audio_styles.py` style plans,
  `regions.py` region graphs (inside map jobs).

The text model is a **brain** (`.ai/domain.md`). No field here is `llm_name`;
that word means the image model.

## Values for AI Providers -> Text

| Field | Value |
|---|---|
| Base URL | `http://<windows-lan-ip>:8001` |
| Generate path | `/api/text` |
| Models path | `/api/text/models` |
| Models pointer | `$.data[*].id` (default first) |
| Auth header | `Authorization` |
| Auth token | `Bearer <key>` - the whole value; their client prepends nothing |
| Key name | **`something2-text`**, scopes `read,generate`, minted with `scripts/mint-key.py` |
| Response pointer | `$.json` with a schema, `$.text` without |

## `POST /api/text`

Request:

```json
{
  "prompt": "Subject: world \"Vale\"; regions: Meadow, Deep Forest; levels 1-10; slot: music",
  "system": "optional system prompt",
  "schema": { "...": "optional JSON Schema; omitted = free text" },
  "temperature": 0.15,
  "max_tokens": 512
}
```

- `prompt` required, non-empty. `system`, `schema`, `temperature`
  (0-2, default 0.7), `max_tokens` (1-4096, default 512) optional.
- With `schema`, decoding is **grammar-constrained** (llama.cpp
  `response_format: json_schema`): the output cannot leave the schema, so an
  `enum` is always honoured. The parsed result is still validated before it is
  returned.
- Thinking is off. No request-body cache: identical requests run again.

Response `200`:

```json
{
  "text": "{\"style\":\"tavern\", ...}",
  "json": { "style": "tavern" },
  "model": "qwen3.6-35b-a3b",
  "usage": { "prompt_tokens": 61, "completion_tokens": 18 },
  "timings": { "load_s": 0.0, "generate_s": 1.9, "total_s": 2.1 }
}
```

`json` is present only when `schema` was sent.

## `GET /api/text/models`

Static, never loads the brain, answers well inside 10 s:

```json
{ "data": [
  { "id": "qwen3.6-35b-a3b", "default": true,  "thinking": false },
  { "id": "qwen3-vl-8b",     "default": false, "thinking": false }
] }
```

`POST /api/text` takes an optional `"model"` (one of these ids; unknown ->
422). Omitted -> the default. Each brain is its own gateway label, so asking
for the other one while one is loaded is a switch like any other.

## Status codes

| Code | Meaning | Client should |
|---|---|---|
| 200 | done | use it |
| 401 / 403 | no/invalid key, or missing scope | fix config (own bug) |
| 422 | can never succeed as sent: invalid or unsupported schema, impossible schema (e.g. empty `enum`), out-of-range params, output truncated by `max_tokens` | **terminal** - do not retry the same body |
| 409 | card is mid-switch | fall back |
| 503 + `Retry-After` | model gateway: another model holds the card, or GPU breaker open (`gpu_faulted`) | fall back |
| other 5xx | brain process failed | fall back |

Busy is refused **at once**, never queued: the caller has no retries and a
5-minute budget, so waiting behind a long job would only spend it.

## Verification checklist (to report after build)

1. curl with the `something2-text` key: no schema; SFX-entity schema; music
   schema with `enum = ["medieval_fantasy","tavern","dungeon","battle","village"]`
   and the Vale prompt - `json.style` in the enum every time.
2. No token -> 401. Impossible schema -> 422.
3. Text request while an image or audio job holds the card -> 503/409, no CUDA
   fault (`make gpu-health` clean afterwards).
4. Cold and warm latency for a ~200-token JSON answer; VRAM while loaded
   (host-side counter, not `nvidia-smi` in WSL); model id.
5. Variety, two numbers - they answer different questions:
   - 10 identical music requests at temperature 0.15: how many pick the same
     style. Expect ~10/10 from any brain; this measures determinism.
   - Varied subjects (dungeon world, tavern level, boss region, ...) with the
     enum order shuffled: distinct styles returned and whether each fits. This
     is the test for the 40/40 `medieval_fantasy` canary (subject vs. enum
     position bias vs. model collapse).
