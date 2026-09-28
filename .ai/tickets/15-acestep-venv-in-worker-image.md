# The worker image carries ACE-Step's venv, and a hung ACE-Step child cannot wedge the worker

## What To Build

Move the isolated venv from the deleted spike image into the product worker
image, so `audio_engine._generate_music` has `/opt/acestep/.venv/bin/python`
to spawn. Give that subprocess a timeout.

## Blocked By

01's listening gate (pass, or fail-on-vocals - see 01). Not a fail on quality.

## Scope

- `compose/develop/sprite_generator/Dockerfile.cuda`: a stage copied from
  `Dockerfile.audio-spike` - `uv` binary, `git clone` of
  `ace-step/ACE-Step-1.5` **pinned to `ca1e85f`** (the commit 0010 read; not
  `main`), `uv sync --frozen --no-dev --python /usr/local/bin/python3.11`
  into `/opt/acestep/.venv`, and the two-stack import assertion. Put it
  **before** the `requirements*.txt` layers so a routine requirements change
  hits the layer cache instead of re-downloading ~14 GB at ~0.6 MB/s.
- `ffmpeg` only if the venv needs it at runtime (the spike image had it for
  torchcodec); the worker's own writer is soundfile.
- `audio_engine._generate_music`: `subprocess.run(..., timeout=...)` with
  `ACESTEP_TIMEOUT_S` (default: measured warm 7.4 s + load 23.3 s, times a
  generous factor - state the number). On `TimeoutExpired` raise
  `AudioGenerationError` naming the timeout; the child is killed by `run`.
- Compose: `ACESTEP_CHECKPOINTS_DIR=/models/acestep` on the worker.
- Model choice `ACESTEP_CONFIG` (`acestep-v15-turbo` default,
  `acestep-v15-sft` available) if 01 failed on vocals.

## Out Of Scope

UI (06, 07), the brain path (08), sfx (16, 17).

## Acceptance Criteria

- [ ] The worker image builds; the assertion prints both stacks at their own versions.
- [ ] `verify-audio-api.py --submit` (kind music) passes; the OGG is in `audio/music/`.
- [ ] `make gpu-health` OK afterwards, and an SDXL txt2img still works (eviction both ways - ticket 04's check).
- [ ] With `ACESTEP_TIMEOUT_S=1`, a music request fails with the timeout message and the next image job runs.
- [ ] Image size recorded in 0010 (expected ~25 GB).

## Verification

    make gpu-health
    SPRITE_API_KEY=... python scripts/verify-audio-api.py --submit --kind music
    make gpu-health

## Suggested Route

`/implement`, then `/review-code`.

## Status 2026-09-28 - done except the two unticked checks named below

- [x] Image builds; assertion printed `worker : 2.14.0+cu130 0.40.0 5.17.0`
      and `acestep: 2.10.0+cu128 12.8 4.57.6` - identical to the spike.
      Build 2,676 s (after LSO was disabled - see project-context); both
      sprite images 25 GB.
- [x] Music end to end through the worker (`smooth-medieval-1`,
      medieval_fantasy): 56.0 s wall cold (load 28.9 s, model 11.3 s),
      125.714 s loop = 44 bars at 84 bpm, 48 kHz, seam 0.90 dB, peak
      7,459 MB allocated. File in `audio/music/`.
- [x] `gpu-health` OK after music (11.2 GB free: the child exits, so VRAM
      returns fully) and after an SDXL + pixel-art-xl 512^2 txt2img that
      followed it (6,800 MB reserved / 4.4 GB headroom, the normal warm state).
- [ ] `verify-audio-api.py --submit --kind music` over HTTP - not run (no
      bearer in this session); the task path was driven directly.
- [ ] `ACESTEP_TIMEOUT_S=1` path - not run.

**Found on the first run: the worker SEGFAULTED (exit 139) after ACE-Step had
finished.** Kernel log: `segfault ... in libsndfile_x86_64.so`, fault address
= stack pointer. libsndfile 1.2.2 (bundled with soundfile 0.14.0) overflows
its stack on a single large Vorbis write: 30 s of 48 kHz stereo wrote, 60 s
and up died - reproduced outside the worker. Ambience never hit it (30 s at
44.1 kHz). `audio_master.write_ogg` now writes in 64k-frame blocks; smoke
case "a full-length music loop writes to OGG" covers it in a child process.
The crash killed the Celery process itself, so no handler ran and the ledger
row stayed `running` until closed by hand - the breaker cannot see a
segfault.
