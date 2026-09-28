# ACE-Step 1.5 runs on this card in a throwaway image; 0010's measured row is filled

## What To Build

A one-off spike, not product code. Prove that ACE-Step 1.5 (turbo DiT) can
generate a 120 s instrumental WAV on the RTX 3060 inside WSL, from an image
derived from the current worker image, and write the numbers into
`.ai/decisions/0010-audio-generation.md`'s **measured** row. Decide the two
forks the plan leaves open: in-image install vs isolated venv + subprocess,
and turbo vs sft.

## Blocked By

None.

## Scope

- `compose/develop/sprite_generator/Dockerfile.audio-spike`: `FROM
  llama-cpp-only-cpu-sprite-worker:latest`, install ACE-Step 1.5 from its repo
  (`https://github.com/ace-step/ACE-Step-1.5`), then `pip check` and a
  `pip freeze` diff against the worker (`/tmp/pip-after.txt` from 2026-09-12
  is the baseline: torch 2.14.0+cu130, transformers 5.17.0, diffusers 0.40.0).
- Download `acestep-v15-turbo` (MIT, ungated) into `MODELS_DIR`. Check
  `Get-PSDrive C, D` on Windows first; `df` inside WSL lies.
- A `scripts/spike-acestep.py` that generates the **fixed 10-prompt set**
  (5 styles x 2 seeds: medieval_fantasy, tavern, dungeon, battle, village;
  `[Instrumental]` as the only lyric; `bpm` and `time_signature` set;
  `audio_duration=120`) and prints per track: seconds (cold for the first,
  warm after), `torch.cuda.max_memory_allocated`, output sample rate and
  channels, output path. Run it with the compose GPU reservation
  (`docker compose ... run --rm --gpus all` on the spike image), with the
  sprite worker **stopped** so the two never share the card.
- Host-side VRAM during a run: `(Get-Counter '\GPU Process Memory(*)\Dedicated
  Usage')` on Windows, one reading mid-generation.
- Listen to all 10; note any vocal leakage per track.
- Confirm WAV output (or a tensor the caller can write); MP3-only is a fail.
- Try `--cpu_offload` / INT8 only if the plain fp16 turbo load OOMs.
- Write the measured row and the two decisions into 0010. Delete the spike
  image afterwards (`docker rmi`).

## Out Of Scope

Any product code, roster, mastering, facade, UI. sft/base variants beyond
one load attempt if turbo leaks vocals on >2 of 10. DiffRhythm (only if
ACE-Step does not install at all - then stop and report).

## Acceptance Criteria

- [ ] `pip check` result and the `pip freeze` diff are recorded in 0010 (even if empty).
- [ ] 10 WAV files exist, each ~120 s, 48 kHz stereo (or whatever the model actually emits, recorded).
- [ ] 0010's measured row has: VRAM peak (guest and host), seconds cold, seconds warm, vocal-leak count out of 10, native max length confirmed.
- [ ] 0010 states the decision: in-image vs venv-subprocess, turbo vs sft, and the resulting `duration_s` default/max.
- [ ] Worker restarted afterwards and `make gpu-health` reports OK.

## Test Seam

Not applicable - a spike; its output is measurements in a decision record.

## Verification

    make gpu-health                       # before: OK, worker idle
    docker compose -f compose/develop/docker-compose.yml -f compose/develop/docker-compose.cuda.yml --env-file compose/develop/.env stop sprite-worker
    docker build -f compose/develop/sprite_generator/Dockerfile.audio-spike -t audio-spike .
    docker run --rm --gpus all -v <MODELS_DIR>:/models -v <IMAGES_DIR>:/app/images audio-spike python /app/scripts/spike-acestep.py
    docker compose ... start sprite-worker && make gpu-health

## Implementation Notes

- Read `.ai/decisions/0010` "Sourced" table and `.ai/specs/audio/plan.md`
  slice 0 first. ACE-Step's API server (`acestep-api`, port 8001) is NOT the
  shape wanted - library or CLI call only.
- Never run the spike while `sprite_worker` or `llm_engine` hold the card;
  `llm-server` sleeps after 120 s idle, check `nvidia-smi` host-side.
- The `MSYS_NO_PATHCONV=1` and "put `$VAR` in a script file" rules apply to
  every command run through `wsl --` from the Windows side.
- The repo path has Cyrillic and a space; quote everything.

## Review Focus

Whether the numbers were measured under the conditions stated (worker
stopped, card otherwise idle), and whether the dependency diff was read
honestly - a downgrade of `transformers` is the single most consequential
finding this ticket can produce.

## Suggested Route

`/implement`, then `/review-code` on the 0010 edits.

## Listening gate (added 2026-09-28) - blocks ticket 15

The ten WAVs from the run are at `audio/spike/spike_<style>_<seed>.wav`
(moved from `sprite-data/images/audio-spike/`). Verdicts were never recorded.
Record one row per track, then apply the gate:

| file | vocals (y/n) | quality (ok/weak) | note |
|---|---|---|---|
| spike_medieval_fantasy_11 | | | |
| spike_medieval_fantasy_42 | | | |
| spike_tavern_11 | | | |
| spike_tavern_42 | | | |
| spike_dungeon_11 | | | |
| spike_dungeon_42 | | | |
| spike_battle_11 | | | |
| spike_battle_42 | | | |
| spike_village_11 | | | |
| spike_village_42 | | | |

- **Pass**: no vocals on either `medieval_fantasy` (the default something2
  gets), at most 1 of 10 with vocals, at least 7 of 10 "sounds like recorded
  instruments". -> ticket 15.
- **Fail on vocals**: ticket 15 anyway (it is what can run anything), then the
  same 10 prompts x seeds on `acestep-v15-sft` (downloaded 2026-09-28; the
  `fix-inst` variants 0010 mentioned are not published) with CFG and a
  negative prompt; its VRAM is unmeasured.
- **Fail on quality**: stop. Neural music was chosen for sound quality
  (0010 "The owner chose A"); if it does not deliver, revisit direction D
  (hybrid) rather than engineer further.

### Owner listening, 2026-09-28

Listened. Verdict as given: make everything **smooth and medieval** - read as
a quality/style fail on part of the set, not on the default. Acted on in
`audio_styles.py` (contract "House style"). Note the ten spike tracks used
`spike-acestep.py`'s own prompts, not the roster, so they cannot be
re-judged against the new templates; the first product music run after
ticket 15 is the real test. **Vocals per track were not reported** - still
needed for the gate's vocal rule.

### Gate outcome, 2026-09-28

Superseded by the corpus pass (ticket 11): the owner judged the house-style
product output "normal for now". Per-track vocal counts were never given;
the fallback model (acestep-v15-sft) stays on disk, unused, as the
escalation if vocals are ever reported.
