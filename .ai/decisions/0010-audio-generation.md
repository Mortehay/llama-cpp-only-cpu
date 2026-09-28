# 0010 - Music and ambience for something2 maps: neural audio, tile-shaped facade

Date: 2026-09-12
Status: **Proposed, awaiting measurements.** The direction is decided; the
model is not. Sections marked *to be measured* are empty on purpose and this
ADR is not final until they hold numbers from this card. Surface in
[specs/audio/contract.md](../specs/audio/contract.md).

## Why this exists

The ask: AI-generated music plus surrounding sound for something2 levels,
requestable from their side through their AI connector, visible in the
Activity tab, listenable there, loopable endlessly on a map, no sample shorter
than 1-2 minutes, with a local tab to generate for testing, and an LLM
somewhere in the loop. Format was an open question (WAV, MP3, MIDI, other).

Nothing in this repo touched audio before. What it did have shaped every
choice below: a 12 GB card that is already the binding constraint, a solo
worker with pipeline eviction, an LLM that already authors *categories* for
worlds, a request ledger feeding Activity, and two facade behaviours (`tile:`
builds within a budget, `map:` only cache-reads) that something2 already
speaks.

## Directions considered

| | optimises for | gives up |
|---|---|---|
| **A. Long-form neural text-to-music** (ACE-Step / DiffRhythm), loop-mastered in post | realism, prompt-driven style, native 2-4 min | ~8 GB VRAM tenancy; the loop is a crossfade, not a guarantee; licenses and instrumental quality unverified |
| B. MusicGen with continuation stitching | maturity, smaller VRAM, melody conditioning | 30 s windows so 2 min is 4+ stitched continuations that drift; CC-BY-NC weights |
| C. Symbolic: LLM picks parameters, rule-based composer writes MIDI, FluidSynth renders | zero GPU, loops perfect by construction, seconds per track, MIDI export, on-brand chip aesthetic | sounds like 1998; ambience cannot come from MIDI; bespoke composer to maintain |
| D. Hybrid: C for music, short-clip neural for ambience | each requirement solved where it is cheap | two pipelines |
| E. Cloud API | best quality, zero GPU | not local, costs money, this repo becomes a proxy |
| F. Curate a CC0 pack, LLM selects | beats a 3060 today | not what was asked |

The recommendation going in was D, because the two hard requirements - endless
loop, >= 1-2 min - are *structural* under C and *post-hoc* under A, and because
C spends none of the scarcest resource. **The owner chose A: the music must
sound like recorded instruments.** That is a quality bar, not an engineering
error, and it is recorded here so the costs it carries are not later mistaken
for defects.

## Decisions

### D1. Neural audio (A). The loop is measured, not assumed.

Consequence: the loop is a beat-tracked, bar-aligned cut plus crossfade. It is
expected to be clean on ambient and drone pieces and audible on strongly
rhythmic ones. Therefore the acceptance bar is a **fixed ~10-track set** kept
across changes (the entity-cutout pattern: 12 fixed seeds) and a stated pass
rate, not a listen-and-nod. A loop-mastering rejection may override a
caller-pinned seed; the seed used is returned - the same rule the cutout path
already follows, for the same reason: a pinned seed that fails can never
succeed.

### D2. Two kinds, `music` and `ambience`, named by map, layered by their side.

Rejected: one mixed file per map. Ambience is stationary and loops at almost
any cut; music is where seams go wrong. Splitting puts the hard property on the
artefact that needs it, lets something2 swap ambience per area, and lets the
ambience model be the tiny one (Stable Audio Open Small class, whose ~47 s cap
is a fit for a texture, not a limit). Cost: a second model to verify and a
second mastering path.

### D3. OGG Vorbis delivered, WAV master kept, MP3 and MIDI rejected.

MP3 encoder delay/padding gaps a loop. MIDI exists only under the symbolic
direction that was set aside. Browser, desktop-first, so Vorbis is safe; iOS
is a non-goal. Loop points travel twice - as `LOOPSTART`/`LOOPLENGTH` tags and
in `info` - because a Web Audio player reads `info` and an engine reads tags.

### D4. The LLM selects from a style roster. It never composes and never free-forms.

The `worlds.py` pattern: the LLM picks biomes, never counts. Here it picks a
roster entry and fills its slots, records what it chose and what it invented
and dropped, and the rules fallback covers it being asleep (it sleeps after
120 s idle) or wrong. `medieval_fantasy` is the default entry. Rejected: a
free-form LLM prompt straight into the model - unreproducible, so a re-roll
can never be compared to its predecessor.

### D5. something2 requests; the facade is tile-shaped; not-ready is never 2xx.

The owner wants something2 to request audio the way it requests images. Their
image call is one blocking POST, 5-minute default cap, no retries - and a
music build is minutes plus up to three model swaps. So: cache first; build
within `AUDIO_GENERATE_TIMEOUT_S`; over budget -> **`503` + `Retry-After` with
the build continuing**, so the next request cache-reads. Rejected: the `map:`
behaviour of returning a placeholder with `cached: false` - silence stored as
final is a silent map forever. Rejected: read-only integration - it was the
safer default, but it is not what was asked, and the ledger records who asked
either way.

### D6. One worker. No second GPU process.

SDXL ~6.8 GB warm, `llm-server` 8-9 GB until it sleeps, the audio model on top.
A second process on the card reproduces the `dxgkio_make_resident: -12` fault
that cost two sessions a day. The audio model loads through the existing
eviction path in the solo worker; a track costs up to three swaps, which is
why D5 is cache-first and why the tab exists to pre-warm a map before it is
published.

### D7. A third kind, `sfx`: one-shot cues, two engines. (2026-09-28)

Added after the original ask: action sounds for an RPG pixel-art game - a
slash, a hit, a pickup, a spell, a footstep, a UI click. Unlike `music` and
`ambience` a cue is **0.1-3 s, played once on an event, never looped**; what
makes one good is a hard onset (no leading silence), a clean tail and
consistent loudness, and a game needs dozens of them with a few variants each
so repeats do not sound identical. Roster entries are **cues**, not
"actions" - *action* already means a sprite-sheet row (`domain.md`).

Both looks were asked for, so there are two **engines**:

| engine | how | optimises for | gives up |
|---|---|---|---|
| `realistic` | Stable Audio Open 1.0, already on disk and verified in product for ambience; one-shot mastering (trim onset, fade tail, loudness-normalise) | reuse, no rebuild, the model card's stated strength | sub-second quality unmeasured; a ~21 s load, so it favours batches |
| `retro` | sfxr-style procedural synthesis, the brain or rules pick a preset and fill its parameters | zero GPU, milliseconds, deterministic, the genre-default pixel-art sound | only 8-bit; some cues (footstep) do not translate |

Rejected for now: Stable Audio Open **Small** - its repo is
`stable-audio-tools` format (`model.safetensors` + `model_config.json`, no
`model_index.json`), so diffusers' `StableAudioPipeline` cannot load it; it
would need another library and possibly a third environment to save a load
that batches amortise. Revisit only if 1.0 measures poor on sub-second cues.

### D8. The sfx engine is resolved by precedence, and the answer says who decided.

Most specific wins; a level applies only when every level above it is silent:

1. the request's `engine` field;
2. the world's `sfx_engine`, read from `<world>.gen.json` (the worlds sidecar
   their seeder never reads - no migration, no change to the spec they
   consume) when the request names a `world`;
3. the cue's own default engine in the roster;
4. `realistic`.

`info.engine_from` names the level (`request` | `world` | `cue` | `default`),
the `author` convention. The engine is part of the cache key, so flipping a
world's engine takes effect on the next request instead of serving stale
cues. A level that picks an engine the cue has no recipe for is **refused
with a message, not silently passed down** - falling through would recreate
the mixed look within one game that level 2 exists to prevent. Rejected: cue
default above world, for the same reason.

## Sourced (2026-09-12) vs measured on this card

Two columns of truth. **Sourced** is what the model's own README, model card
or paper says, fetched on 2026-09-12; it is not evidence about this card in
WSL. **Measured** is empty until someone runs it here, and this ADR is not
final until the music row holds a number.

| | ACE-Step 1.5 | DiffRhythm v1.2 | MusicGen medium | Stable Audio Open 1.0 | Stable Audio Open Small |
|---|---|---|---|---|---|
| license | **MIT** (code and weights) | Apache 2.0 | code MIT, **weights CC-BY-NC 4.0 - out** | Stability Community License: free under USD 1M/yr revenue, outputs owned by the user | same |
| size | DiT 2B (turbo / sft / base), XL 4B; optional LM 0.6B / 1.7B / 4B | ~1B class | 1.5B | 1B | 0.5B |
| VRAM, sourced | turbo **<=6 GB** ("INT8 + full CPU offload" row for <=6 GB); sft/base 8-16 GB; XL >=12 GB *with offload* | "minimum of 8G" | unstated | unstated (fp16 example) | unstated |
| output | **48 kHz stereo** | 44.1 kHz stereo | 32 kHz **mono** | 44.1 kHz stereo | 44.1 kHz stereo |
| native max length | **10 s - 600 s** | full 4m45s, base 1m35s | 30 s | **47 s** | 11 s |
| instrumental | `lyrics` empty or `[Instrumental]` / `[inst]` as the only lyric | "Instrumental Mode" | trained on instrumental only, vocals removed | "not able to generate realistic vocals" | same |
| negative prompt / CFG | `guidance_scale` on sft/base; **turbo has no CFG** (8 steps) | not documented | not exposed | **`negative_prompt` yes** (diffusers example) | not documented |
| speed, sourced | "under 10 s" per song on RTX 3090; 1.74-4.70 s per minute (v1) | unstated | ~real-time class | unstated | unstated |
| structure controls | **`bpm`, `key_scale`, `time_signature`, `audio_duration`, `seed`** in the API | text style prompt or reference audio | text | text | text |
| Python | **3.11-3.12** - the worker image is `python:3.10-slim` | unstated | 3.9+ | 3.9+ | 3.9+ |
| **measured on this card** | | | n/a | | |

Sources: ACE-Step 1.5 README and `docs/en/API.md`, arXiv 2602.00744;
DiffRhythm README; `facebook/musicgen-medium`, `stabilityai/stable-audio-open-1.0`
and `-small` model cards; `stability.ai/license`.

What the sourced row already settles:

- **ACE-Step 1.5 is the music candidate.** MIT, 2B, 48 kHz stereo, ten
  minutes native so the 60-240 s band needs no stitching, and it takes
  `bpm` + `time_signature` as inputs - so the loop cut is **bar arithmetic on
  known values**, not beat tracking on unknown ones. That removes the least
  reliable step from D1. The turbo variant fits the card but has **no CFG**,
  so "no vocals" is carried by the `[Instrumental]` lyric, not a negative
  prompt; sft/base have CFG at 8-16 GB, which on this card means offload.
- **MusicGen is out on license.** Kept in the table so nobody re-proposes it.
- **Stable Audio Open 1.0, not Small, for ambience**: 47 s covers the 20-45 s
  band, 11 s does not; it has a real `negative_prompt`; the card itself says it
  is better at "sound effects and field recordings than music", which is the
  ambience job exactly. Gated download (accept terms once).
- **A Python version wall - resolved 2026-09-12 by moving the worker image
  to 3.11.** ACE-Step wants 3.11-3.12; the worker was `python:3.10-slim`.
  The alternative - ACE-Step as its own container behind its async REST API
  (`/release_task`, `/query_result`, default port 8001, colliding with ours) -
  was rejected because it is a **second GPU process** with no documented idle
  unload, which breaks D6. The bump is one `FROM` line in each sprite
  Dockerfile; with nothing pinned, the re-resolve also moved torch 2.13.0 ->
  2.14.0+cu130, torchvision 0.28 -> 0.29, transformers 5.16.1 -> 5.17.0,
  numpy 2.2.6 -> 2.4.6 (2.2 was the last for 3.10), scipy 1.15 -> 1.17,
  cuDNN 9.20 -> 9.24, triton 3.7 -> 3.8; diffusers 0.40.0, peft, bitsandbytes
  unchanged. Verified on the new image: 19 migrations current, `gpu-health`
  OK (11.2 GB headroom cold, 6800 MB reserved / 4.4 GB headroom warm - the
  documented normal state), `smoke-world-gen.py` 34/34, and an SDXL base +
  pixel-art-xl txt2img at 512/20 steps produced a real sprite in 11.6 s
  denoise (71 s wall, cold load). The build took ~55 min, almost all of it
  re-downloading the cp311 CUDA wheels at ~0.6 MB/s.

The 240 s music maximum and the 120 s default in the contract are
placeholders that the seconds-per-track measurement replaces. The contract's
`sample_rate: 44100` example is wrong for ACE-Step; expect 48000.

### Spike (ticket 01), 2026-09-12 - what is settled before any run

**Fork 1 is decided by ACE-Step's `pyproject.toml`, not by a build.** Read
at `ace-step/ACE-Step-1.5@ca1e85f` (2026-08-29): `requires-python
>=3.11,<3.13`; on linux x86_64 **torch 2.10.0+cu128** (pinned, from the
PyTorch index via `uv.lock`), **transformers >=4.51,<4.58**, diffusers
>=0.37, plus gradio 6.2, lightning, torchao 0.16.x, torchcodec, a vendored
`nano-vllm` (local path source, uv-only), and `mlx` for macOS. The worker
runs transformers 5.17.0 and torch 2.14.0+cu130, and diffusers' Qwen-Edit
path depends on that transformers. One site-packages cannot satisfy both,
so **in-image install is rejected without trying it**. The product shape is
**an isolated venv inside the worker image, invoked as a subprocess after
`pipes` eviction** - `compose/develop/sprite_generator/Dockerfile.audio-spike`
builds exactly that (`uv sync --frozen` from their lockfile into
`/opt/acestep/.venv`) and asserts both stacks import at their own versions.
Their own `Dockerfile` does the same `uv sync --frozen` on a CUDA 12.8 base;
ours needs no CUDA base because the wheels vendor the runtime, the same
argument `Dockerfile.cuda` makes.

**The calling surface is the library, not the API server.** From their
`run_generate_test.py`: `AceStepHandler().initialize_service(project_root,
config_path="acestep-v15-turbo", device="auto", offload_to_cpu=False)`,
then `generate_music(dit_handler, llm_handler, GenerationParams(...),
GenerationConfig(batch_size=1, audio_format="wav"), save_dir)`.
`GenerationParams` carries `caption`, `lyrics` (`"[Instrumental]"`), an
explicit **`instrumental: bool`**, `bpm`, `keyscale`, `timesignature`,
`duration`, `inference_steps` (8 for turbo), `guidance_scale`, `seed`, and
`thinking` - with `thinking=False` the LM handler is `None` and no LM is
loaded. `audio_format="wav"` exists, so the WAV master is available without
transcoding.

**The main model repo is 10.1 GB and all of it is required on disk**:
`ACE-Step/Ace-Step1.5` = turbo DiT 4.79 GB + `Qwen3-Embedding-0.6B` text
encoder 1.19 GB + VAE 0.34 GB + `acestep-5Hz-lm-1.7B` 3.71 GB.
`initialize_service` -> `_ensure_models_present` -> `check_main_model_exists`
requires all four directories, so the LM is pulled even though the spike
never loads it. Layout is `<checkpoints>/<component>/`, selected by
`ACESTEP_CHECKPOINTS_DIR`. Note the roster also lists
`acestep-v15-turbo-fix-inst-*` variants - "fix-inst" reads as an
instrumental fix; if turbo leaks vocals, try one of those before sft.
**Corrected 2026-09-28: the fix-inst variants are not downloadable.** They
appear only in `model_downloader.py`'s `_CHECKPOINT_TO_VARIANT` (code sync),
not in its `SUBMODEL_REGISTRY`, and none of the 20 repos under `ACE-Step` on
Hugging Face is one. The real vocal escalation is **`ACE-Step/acestep-v15-sft`**
(4.79 GB DiT, has CFG), fetched 2026-09-28 into
`MODELS_DIR/acestep/acestep-v15-sft/` beside turbo; it reuses the text
encoder and VAE already there. Its VRAM on this card is unmeasured.

**Where the bytes land, and it is not where the docs say.** `MODELS_DIR`
(`/home/markunn/sprite-data/models`) is on `/dev/sdd`, the distro's own
ext4 VHDX on C:, and the existing 27 GB of weights are there too. The D:
models VHD is attached as `/dev/sde` (ext4, 1 TB nominal) but **mounted
nowhere**; `project-context.md`'s "D: now holds the models" describes an
arrangement that is not in effect. C: had 39.7 GB free before the pull;
this spike adds ~10 GB of weights plus ~8 GB of image to the VHDX.

**Build measured 2026-09-12: the two stacks coexist.** `Dockerfile.audio-spike`
built (exit 0) and its assertion printed
`worker : 2.14.0+cu130 0.40.0 5.17.0` beside
`acestep: 2.10.0+cu128 12.8 4.57.6` - the worker's torch/diffusers/
transformers untouched, ACE-Step's own torch and transformers 4.57.6 in
`/opt/acestep/.venv`. **The image is 25 GB** (worker 10.6 GB + ~14 GB of
venv, most of it a second vendored CUDA runtime). That is the cost of the
venv shape; the alternative that avoids it is a separate container, which
D6 rejects. Wheel download took ~95 min of the build on this link.

**Run measured 2026-09-12 (`scripts/spike-acestep.py`, turbo, bf16, 8
steps, `guidance_scale=1.0`, `[Instrumental]` + `instrumental=True`,
`thinking=False`, sprite-worker stopped, guest free 11,245 MB before load;
weights on ext4 under `MODELS_DIR/acestep`):**

| | measured |
|---|---|
| weights download (HF CDN) | 10.1 GB in ~27 min - the CDN is fast; PyPI was the slow link |
| `initialize_service`, cold from ext4 | **23.3 s**; 6,033 MB allocated / 7,488 MB reserved after load; dtype bfloat16 |
| 120 s track, first (cold) | **12.9 s** |
| 120 s track, warm | **7.2-7.4 s**, 10 of 10 succeeded - about 16x real time |
| VRAM peak | 6,546 MB allocated / **7,488 MB reserved** (cold), 6,890 MB warm; the VAE decode reported 5.04 GB free at its start |
| output | **48 kHz stereo**, exactly 120.0 s, 23.0 MB WAV each |
| worker afterwards | restarted, `gpu-health` OK, 11.2 GB headroom |

What this settles:

- **Turbo is the model, and speed is not a constraint anywhere.** A cold
  request is load 23 s + 13 s = under 40 s before eviction cost; a warm one
  is 7 s. The facade budget and something2's timeout are dominated by the
  SDXL evict/reload around it, not by ACE-Step. The contract's 240 s
  maximum is no longer a budget placeholder: at 16x real time a 600 s
  track (the model's native ceiling) would take ~40 s. Set the v1 max by
  product choice, not by time.
- **VRAM fits, barely beside nothing.** 7.5 GB reserved means SDXL (6.8 GB
  warm) must be evicted first - the same rule as the ambience model. The
  two audio models (7.5 GB and 6.0 GB) also cannot be resident together;
  the `pipes` "exactly one" rule extends to them.
- **bpm / time signature / duration are honoured exactly** (120.0 s on all
  ten), which is what the bar-arithmetic loop cut needs.
- **The 1.7B LM is on disk and unused.** `thinking=False` cost nothing
  visible in success rate; whether `thinking=True` (LM 3.5 GB more, would
  not fit beside the DiT here) improves quality is an open question, not a
  v1 one.
- **Not measured**: host-side `Get-Counter` during the run; the venv's
  `pip freeze` diff is moot (separate environment). **Vocal leakage out of
  10 is a listening verdict** and is recorded by the owner in ticket 01 -
  the fix-inst turbo variants are the first escalation if it fails.

### Spike (ticket 02), 2026-09-12 - Stable Audio Open 1.0, MEASURED

Run with `scripts/spike-stable-audio.py` in the worker image (diffusers
0.40.0 `StableAudioPipeline`, fp16, `HF_HUB_OFFLINE=1`), sprite-worker
stopped, card otherwise idle (guest free 11,245 MB before load). Three
30 s clips, 100 steps, seed 7, `negative_prompt="vocals, singing, music,
melody, low quality"`:

| | measured |
|---|---|
| gated download, one-time, via the API container's own `HF_TOKEN` | 25 files, 4.7 GB, 3643 s on this link |
| load from ext4 | **20.9 s**; 2,578 MB allocated after load |
| 30 s clip | **29.6 s warm** (32.7 s cold) - roughly 1x real time |
| VRAM peak | **5,572 MB allocated / 5,988 MB reserved** |
| output | 44.1 kHz stereo, exactly 30.0 s |
| worker afterwards | restarted, `gpu-health` OK, 11.2 GB headroom |

So the ambience model fits beside nothing: at ~6 GB reserved it needs SDXL
(6.8 GB warm) evicted first, exactly as the music model will. Host-side
`Get-Counter` was not read during the run. Listening verdicts (texture vs
music) are recorded by the owner in the ticket, not here.

**The worker image has no `soundfile` and no `libsndfile`** - the spike had
to fall back to `scipy.io.wavfile`. Ticket 03's OGG writer therefore adds a
dependency to the image (soundfile + libsndfile1, or ffmpeg), and that is a
rebuild. Done 2026-09-13: apt `libsndfile1`, pip `soundfile` + `mutagen` in
both sprite images (soundfile 0.14.0 / libsndfile 1.2.2 / mutagen 1.48.1).
There is still **no ffmpeg** in the image; anything that shells out to
`ffprobe` will not run there.

### In product, measured 2026-09-13 (ambience path end to end)

`POST /api/audio {"kind":"ambience"}` through the real facade, worker warm,
SDXL not resident:

| | measured |
|---|---|
| whole request, cold model load included | **37-41 s** for a 30 s loop |
| the same name again | **0.07-0.08 s**, `cached: true`, no GPU |
| output | 44.1 kHz stereo OGG ~420 KiB + a 5.3 MB WAV master |
| seam after mastering, real material | **0.77 / 1.02 / 2.38 dB** across three takes |
| worker afterwards | `gpu-health` OK, 11.2 GB headroom |

The seam figures are the first real numbers for the acceptance bar, and they
already say the synthetic threshold was optimistic: **2.38 dB on a texture**
is above the 2.0 dB the smoke asserts on its fixture. The smoke's bound is a
statement about the fixture, not about output; ticket 11's corpus is what
sets the real one, and it should be set from a distribution, not from three
takes.

**Two ledger defects were found by reading the rows back, not by the tests.**
Both are fixed and both were invisible from the API's own success response:
`generations.finish` did not carry the producer's `params` (so `info` served
nulls for duration, sample rate, loop points and seam to something2), and it
did not carry the rendered `prompt` (so Activity showed "(no prompt)" for
every track, because the request names a STYLE and the text is rendered in
the worker). `/api/audio` also listed finished cache-read rows as artefacts
with null metadata; it now lists a row only if it owns a file or is still
in flight.

### sfx, measured 2026-09-28 (realistic engine, ticket 16)

Stable Audio Open 1.0, 100 steps, run from the audio worktree in a one-off
worker container with the stock worker stopped (never two processes on the
card):

| | measured |
|---|---|
| one cue, 1 variant, pipeline defaults | 29.3 s model time for a 0.6 s hit |
| one cue, 3 variants, pipeline's batched decode | **GPU fault, twice, deterministic**: `dxgkio_make_resident -12` inside `autoencoder_oobleck.decode`, surfacing as `CUDA driver error: device not ready` |
| 3 variants, latents decoded one at a time, full window | fixed the fault; but 142 s - batching was SLOWER than 3 separate calls |
| one cue, denoised window cut to 128 frames (5.9 s) | **8.0 s** (64 frames: 6.9 s, but raw peak 2.36 and a hit that rang to 1.1 s) |
| product: pack of 3 cues / 8 variants, 128 frames, sequential variants | **79.2 s** incl. a 3.4 s load; onset 0-5 ms on all 8; peak 2,851 MB |
| facade, through Celery | single cold 21.7 s (2 variants); repeat 0.0 s from cache; pack of 1 cached + 2 new 35.8 s |

What it settles:

- **The pipeline always denoises the model's full ~47.6 s window** (1024
  latent frames) whatever `audio_end_in_s` asks for, and trims after
  decoding. That is why a cue cost as much as a 30 s ambience, and why its
  own decode of 3 variants (3 x 47 s of audio) faulted the card. The engine
  now asks for latents, cuts them to the cue plus slack, decodes one at a
  time, and shrinks the window to `SFX_LATENT_FRAMES` (128).
- Variants are sequential calls with seeds `seed..seed+n-1`, each recorded,
  so any one variant can be regenerated alone.
- **Not yet judged by ear**: whether 128 frames costs quality against the
  full window. `audio/sfx/_experiment/hit-slime_frames{1024,128,64}.wav`
  are the same prompt and seed at each size, for that comparison.
- The same full-window cost applies to AMBIENCE (a 30 s clip denoises 47.6 s);
  shrinking it there is a possible later saving, not measured.

### In product, measured 2026-09-28 (music path end to end)

Worker image rebuilt with the ACE-Step venv (ticket 15), turbo, bf16,
`medieval_fantasy` in the smooth house style, SDXL not resident:

| | measured |
|---|---|
| whole task, cold (venv spawn + load + take + master + write) | **56.0 s** |
| ACE-Step load / model time | 28.9 s / 11.3 s |
| loop | 125.714 s = 44 bars at 84 bpm, 48 kHz stereo |
| seam after mastering | 0.90 dB |
| VRAM peak | 7,459 MB allocated; all of it returned when the child exits |
| after, then an SDXL + pixel-art-xl txt2img | `gpu-health` OK both times; SDXL warm state normal |

**libsndfile segfaults on a large single Vorbis write.** The first attempt
killed the worker (exit 139, `segfault in libsndfile_x86_64.so`) after the
take was already on disk: soundfile 0.14.0's bundled libsndfile 1.2.2
overflows its stack writing >= 60 s of 48 kHz stereo in one call. Fixed by
block writes in `audio_master.write_ogg`. A segfault bypasses every Python
handler and the GPU breaker, and leaves the ledger row `running`.

## State on 2026-09-28 (read from the code, not from the tickets)

- ~~Music cannot run in product~~ - **fixed the same day by ticket 15**
  (measured above): `Dockerfile.cuda` now builds the venv, pinned to
  `ca1e85f`.
- ~~`subprocess.run` with no timeout~~ - `ACESTEP_TIMEOUT_S`, default 300 s.
- The **listening verdicts** for the ten spike WAVs
  (`IMAGES_DIR/audio-spike/`) were never recorded; that is the gate before
  ticket 15 (see ticket 01).
- Ambience is end to end (above). Frontend (06, 07), the brain path (08),
  connector (10) and corpus (11) are not started. Nothing is committed.

## Open

- Whether something2 plays audio at all today. If not, milestone 1 is
  download-and-drop, and the connector is milestone 2.
- Worker image rebuild cost and the "no CUDA at import" rule against whatever
  the chosen model's library does at import time.
