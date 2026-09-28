# Stable Audio Open 1.0 produces a 30 s ambience clip on this card

## What To Build

A one-off spike proving the ambience model loads and generates here through
the diffusers already in the worker image, and that the gated download can be
done once and served offline afterwards. Fills the Stable Audio Open column of
0010's measured row.

## Blocked By

None. Independent of ACE-Step; runs on the existing worker image (diffusers
0.40.0 ships `StableAudioPipeline`).

## Scope

- One-time gated access: accept the model terms on Hugging Face, then fetch
  `stabilityai/stable-audio-open-1.0` into the HF cache under `MODELS_DIR`
  with a token, **out of band** (the running services have
  `HF_HUB_OFFLINE=1`). Record in `downloader/models.txt` or its equivalent
  how it was fetched so the next machine can repeat it; the token itself is
  never committed.
- `scripts/spike-stable-audio.py`: three prompts (`forest wind and birds`,
  `dripping cave, distant echo`, `village square, day, crowd murmur`),
  `negative_prompt="vocals, music, melody, singing"`, 30 s, seed fixed;
  prints seconds cold/warm, `max_memory_allocated`, sample rate, channels;
  writes WAV.
- Run with the worker stopped, GPU reserved, as in ticket 01.
- Listen to all three; note whether any contains music rather than texture.
- Write the column in 0010.

## Out Of Scope

Mastering, looping, roster, any product code. Stable Audio Open *Small*
(11 s cap - already rejected in 0010).

## Acceptance Criteria

- [ ] The model loads with `HF_HUB_OFFLINE=1` from the local cache after the one-time fetch.
- [ ] Three 30 s WAVs exist at 44.1 kHz stereo.
- [ ] 0010's measured row has VRAM peak, seconds cold/warm, and a per-clip note on music-vs-texture.
- [ ] The fetch procedure (not the token) is documented where the other weights are.

## Test Seam

Not applicable - a spike.

## Verification

    make gpu-health
    docker compose ... stop sprite-worker
    docker compose ... run --rm --gpus all sprite-worker python /app/scripts/spike-stable-audio.py
    docker compose ... start sprite-worker && make gpu-health

## Implementation Notes

- diffusers example: `StableAudioPipeline.from_pretrained(..., torch_dtype=torch.float16).to("cuda")`, `audio_end_in_s=30.0`, `negative_prompt=...`. `num_waveforms_per_prompt=1`.
- The worker's `pipes` eviction is not in play in a spike; in product code (ticket 09) it is.

## Review Focus

That the offline load really is offline (unset nothing; the run must succeed
with the same env the worker has), and that the license note in 0010 matches
what the user actually accepted.

## Suggested Route

`/implement`, then `/review-code` on the 0010 edits.
