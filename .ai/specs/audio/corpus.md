# Audio acceptance corpus

The fixed before/after set decisions/0010 D1 refers to. Per-row data lives in
`audio/corpus/corpus.json` (gitignored with the rest of `audio/`); this file is
the summary, in the `reference-audit.md` pattern. Rebuild or extend with
`scripts/build-audio-corpus.py` - fixed names, so existing rows are cache hits.

## Built 2026-09-28

Through the real authenticated `POST /api/audio`, house style (smooth,
medieval), worker otherwise idle. 15/15 built in 723 s wall; `gpu-health` OK
after.

| name | kind | seed | gen time | seam jump |
|---|---|---|---|---|
| corpus-medieval_fantasy-11 | music | 11 | 57.8 s (cold) | 1.41 dB |
| corpus-medieval_fantasy-42 | music | 42 | 29.5 s | 1.15 dB |
| corpus-tavern-11 | music | 11 | 26.3 s | 1.42 dB |
| corpus-tavern-42 | music | 42 | 27.5 s | 1.62 dB |
| corpus-dungeon-11 | music | 11 | 28.1 s | 0.42 dB |
| corpus-dungeon-42 | music | 42 | 27.2 s | 0.56 dB |
| corpus-battle-11 | music | 11 | 27.1 s | 1.31 dB |
| corpus-battle-42 | music | 42 | 26.3 s | 0.98 dB |
| corpus-village-11 | music | 11 | 27.0 s | 1.38 dB |
| corpus-village-42 | music | 42 | 26.0 s | 1.78 dB |
| corpus-forest | ambience | 11 | 34.2 s | 0.53 dB |
| corpus-cave | ambience | 11 | 33.8 s | **5.57 dB** |
| corpus-village_day | ambience | 11 | 33.3 s | **4.50 dB** |
| corpus-night | ambience | 11 | 43.6 s | 0.52 dB |
| corpus-rain | ambience | 11 | 51.6 s | 0.78 dB |

What the numbers say, before anyone listens:

- **Music: 0.42-1.78 dB across all ten**, mean 1.20. Warm generation 26-29 s
  for a 2-minute loop; the first (cold) 57.8 s. The ~100-120 s figures from
  the verifier earlier the same day were contention on a busy host, not the
  steady state.
- **Two ambience seams stand out: cave 5.57 dB, village_day 4.50 dB**,
  against 0.52-0.78 dB for the other three. A seam metric this high usually
  means an event (a drip, a voice, a cart) sits across the cut. Listen to
  these first with the Seam button; if they click, that is the first measured
  case for a longer crossfade or a quieter cut point, not a reason to change
  the bar.
- The metric is RMS continuity at the join. It is evidence, not the verdict:
  the acceptance bar is the LISTENING pass below.

## Listening pass - owner, 2026-09-28

Verdict as given: **"sounds normal for now"** - an overall pass on the set,
taken as meeting the bar for v1. Recorded exactly as said: it is NOT a
per-track verdict, so `corpus.json` rows are still unfilled, no vocal count
exists, and the two high-seam ambience loops (cave, village_day) were not
singled out as clicking. If a later change is compared against this corpus,
the per-row fields are where a finer verdict goes.

### The bar, for reference

Bar (contract "Acceptance"): the seam inaudible on >= 8 of 10 music tracks,
"audible but not a click" tolerated on the rest; **0 vocals** on
`medieval_fantasy`; all music >= 60 s (met: all 125.7 s).

Fill `verdict` (`inaudible` | `audible` | `click`), `vocals` (true/false) and
`note` per row in `corpus.json` - re-running the builder preserves them. Then
this section gets the counts and 0010 moves to accepted, or says which bar was
missed and names the next dial.
