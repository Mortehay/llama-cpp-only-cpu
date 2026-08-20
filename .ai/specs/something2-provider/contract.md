# something2 provider registration

How to point the something2 admin panel at this service. No code changes are
needed on the something2 side — this service impersonates Automatic1111, which
its provider system already supports as a stock preset.

Source of truth for the consumer side is something2's `docs/ai-providers.md`.
**Caveat: that doc was read from `main`; their actual calling code has not been
reviewed.** Verify against a real job before trusting these values.

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
