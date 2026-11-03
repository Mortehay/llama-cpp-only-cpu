# The high-quality sprite lane (`engine: "comfyui-hq"`)

A second renderer beside the existing one, selected per job. The fast lane is
unchanged and remains the default.

## Why it exists

The diffusers lane runs a **Q2_K** transformer with the **4-step Lightning
LoRA**, and that distillation forces `true_cfg_scale=1.0`. At CFG 1.0 there is
no classifier-free guidance, so **the sheet conveyor has no negative prompt and
cannot be given one** — every "no extra limbs" instruction lands in the
positive conditioning instead, which is the `no frame, no border` trap
recorded in `action_prompts.json`. Spell glow had to be removed from pose text
by hand for exactly this reason.

This lane drops Lightning. That buys real step counts, real CFG, and a
negative prompt — the standard tool for the defects the owner reports (extra
limbs, melted faces) — plus a far less aggressive quant.

**Measured attribution before building it**, so the lane is not asked to fix
things it cannot:

| defect | attributed to | evidence |
|---|---|---|
| profile close-ups | **neither** — already fixed | `ISO_DISTANCE=wide` took reference cuts 2/8 → 0/8 and cell cuts 21/30 → 4/30 |
| pose-text ignored | the model / angles LoRA | both better quants made `facing` *worse* (8 → 21, 8 → 31) |
| melted faces | the quant | Q6_K halved identity failures 22 → 10, and it survived the redraws |
| extra limbs | untested; points at Lightning | it is the classic negative-prompt defect, and this lane is the only one that can have one |

## The contract

```
JobSpec.engine : "diffusers" (default) | "comfyui-hq"
```

- **Declared in Pydantic**, never read with `.get`. An undeclared field is
  dropped silently and `model_dump()` is what reaches the worker. For this
  field that failure is severe: a caller who asks for the quality lane and is
  quietly served the fast one waits hours for a sheet identical to the one
  they already had, with nothing in the response to say which ran.
- **Default stays `diffusers` permanently.** Every existing integration omits
  the field; changing what silence means would move all of them onto a lane
  costing hours per sheet.

### Errors, and why they differ

| condition | status | header | retry? |
|---|---|---|---|
| unknown engine name | 422 (validation) | — | no |
| known engine, this box cannot serve it | **422** | `X-Error-Kind: engine_unavailable` | **no** |
| card busy | 503 | `X-Error-Kind: vram_busy` + `Retry-After` | yes |

`engine_unavailable` carries **no `Retry-After`**: waiting cannot help, and a
caller retrying it burns requests on an answer that only an operator can
change. Availability is checked **before the row is written**, so an HQ job on
an unprovisioned box fails in milliseconds rather than hours in. The message
names which weight files are missing, and states that nothing was queued and
no concept consumed.

The header is `X-Error-Kind` — the capitalisation already used for
`vram_busy` and `gpu_faulted`. HTTP header names are case-insensitive; match
accordingly rather than adding a second spelling.

### Provenance

The atlas carries `engine`, and `GET /api/jobs/{id}` carries both:

- `engine` — what was **asked for**, from the spec.
- `engine_info` — what actually **ran**, read from the atlas on disk.

```json
{"name": "comfyui-hq", "model_file": "...", "precision": "fp8mixed",
 "steps": 20, "cfg": 4.0,
 "loras": [{"name": "...multiple-angles-lora.safetensors", "scale": 0.4}]}
```

Two fields because the **comparison between them** is what a consumer needs.
`engine_info` is read from the file, never recomputed: a default evaluated at
response time would describe the box's current configuration rather than the
sheet in hand.

`loras` is a list of **objects with a scale**, never bare names — the scale is
load-bearing here, because the angles LoRA pins *pose* as well as camera, so a
name alone discards the number most likely to explain a bad sheet. `steps` and
`cfg` are the values **actually used**, so a job retried at other settings
describes the pixels that exist.

### It never falls back

If ComfyUI is down, an HQ job **fails and says so**. `engines.py` has no
fallback path and must not grow one — a lane that silently degrades to
diffusers is indistinguishable from the lane not working.
`test-engine-contract` asserts the absence.

## What is shared, and what is not

**Shared code, and deliberately unaware of the engine:** the action plan, the
per-pose dedup, the mirrored-view pairing, the cutout, every per-frame check,
`scores.json`, the redraw bookkeeping, `pixelate`, and the atlas.

> The moment a check behaves differently per lane, comparing the two lanes
> stops meaning anything.

**Engine-specific:** exactly one line at each of four render sites — the
turnaround, the action cells, the explicit redraw and the round-based redraw.
`_hq_render_cells` returns `{key: image}`, the same contract as
`denoise_cells`, so the write-and-score loop is untouched.

All four sites dispatch, including both redraws: repairing an HQ sheet with
diffusers would leave one Q2_K frame in an otherwise high-precision row, and
the atlas echo would then be a lie about those pixels. The engine is recorded
in `config.json` at `turnaround-encode`, so a redraw days later reads it off
the work directory rather than a job row that may be gone.

The **per-pose dedup matters more here**, not less: a repeated pose costs
minutes on this lane rather than ~30 s.

Camera prompts are reused verbatim, **including the mirrored western
azimuths** — that trick is a property of the angles LoRA, not of the engine, so
the west rows keep facing correctly and the lanes stay comparable.

## Operational

- **Separate gateway label.** `comfyui-hq`, not a variant of
  `qwen-image-edit`. Both lanes render the same family but as separate
  processes holding separate copies on a 12 GB card — ComfyUI's ~19 GB
  transformer against the worker child's Q2_K. A shared label would let the
  gateway think the card was already granted and run them together, which is
  the wedge it exists to prevent. `_sheet_model` reads the engine from the
  jobs row so a **redraw of an HQ sheet gates as HQ too**.
- **Models on D:** `/mnt/d/comfy-hq/models`, ~28 GB. `/models` is on a C: that
  has filled twice. The host tree is mounted **twice** — at
  `/opt/ComfyUI/models` where ComfyUI looks and `/models-hq` where the worker
  checks availability — so the two answers cannot drift; there is no second
  copy to go stale.
- **`profiles: ["hq"]`** keeps the service stopped on a box without the
  weights, which is exactly when `engines` answers 422.
- **Pinned.** ComfyUI `v0.43.2`. The first pin tried was `v0.3.45`, which
  **cannot run this model at all** — no `nodes_qwen.py`, no
  `TextEncodeQwenImageEditPlus`. Unpinned, "the lane renders differently this
  week" would be unfalsifiable.
- **The graph is validated, not trusted.** `comfy_client.validate_graph`
  checks every node class and input name against the server's own
  `/object_info` before submitting, and names the wrong one. Node names move
  between versions.
- **`denoised N/M`** is printed per cell because `job_runner.run_stage` parses
  exactly that for the progress bar *and* the per-cell cancel — the only thing
  that lets a cancel land mid-stage rather than after hours.
- **Celery visibility timeout** was raised to 24 h (`fe8b12d`). It was 3600 s
  by default while sheets take 3–7 h, so Redis had been redelivering running
  builds; the solo worker queued the duplicate and rebuilt the whole sheet.
  That was a defect in the **current** lane, found while scoping this one.

## Not yet established

Written before the lane has rendered anything. **Unknown until the smoke
render:** seconds per cell, peak VRAM and host RAM, and whether the angles
LoRA holds the camera through ComfyUI. If s/cell lands far above the
diffusers lane's ~30 s, a 210-cell sheet stops being an overnight job and the
lane's practicality is a question for the owner, not a given.

The **FP8-mixed vs Q8_0** choice is also open; FP8 was fetched first because
it is ComfyUI-native and needs no GGUF node, which removes a custom-node
dependency from the critical path. This card is **sm_86** and has no native
FP8 tensor cores, so FP8 buys memory rather than speed — as does Q8_0
dequantising on the fly.
