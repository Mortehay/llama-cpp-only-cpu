# Colibri runs on this host; 0011's measured table is filled

## What To Build

A one-off spike, not product code. Run Colibri (https://github.com/JustVugg/colibri)
on this box in two lanes and write the numbers into
`.ai/decisions/0011-brain-tiers-and-colibri.md`'s measured table against the
pass criteria **already fixed in D2** - do not move them after measuring.

## Blocked By

None (ticket 12 should land first so no concurrent generation runs turbo, but
the spike does not depend on it).

## Scope

0. **Prerequisites, before downloading anything:**
   - Drive letters moved with the upgrade: the old 111.8 GB D: is now E:
     (69 GB free), and D: is a new empty 223 GB SATA SSD. Confirm
     `scripts/setup-models-vhd.ps1 -AttachOnly` still attaches the models VHD;
     fix its path if not. Symptom of forgetting: every model "vanished".
   - `.wslconfig`: `processors=4` -> 12, `memory=24GB` -> ~36GB, and replace the
     i3/15.9 GB comments. `wsl --shutdown`, then the three post-restart scripts.
     Measuring with 4 threads would understate the CPU lane by ~2-3x.
   - Find out whether Colibri publishes **pre-converted** GLM-5.3-Flash
     weights. If only conversion from source exists, source + ~195 GB output
     does not fit D:; stop and report rather than filling C:.
1. **Text lane:** Qwen3.6-35B (int4, ~20 GB, RAM-resident), CPU only.
   Record decode tok/s (short and long answers), time to answer a fixed
   ~1.5k-token prompt (reuse the `worlds._llm_biome_plan` system prompt), RAM
   peak, and RAM left inside WSL with the sprite worker idle-but-loaded.
2. **Control:** the same Qwen3.6-35B as a GGUF in the existing `llm-server`
   image with `-ngl 0 --threads 12`. Same prompt, same numbers.
3. **Judge lane:** GLM-5.3-Flash with vision, experts on D:. Build a
   labelled set of **12 sprites** from `.ai/specs/entity-cutout/findings.md`
   cases (single object / item sheet / wrong subject, 4 each). Ask as a closed
   choice (Brio mode if it works through the gateway). Record cold and warm
   seconds per verdict and the score out of 12.
4. Write the table, mark each lane pass/fail against D2, and add one paragraph
   on whether the control makes the Colibri text lane redundant.

## Out Of Scope

Any product code, switcher, UI, or compose service. DeepSeek V4.1 Flash beyond
one load attempt if GLM will not run. Putting any brain on the GPU.

## Acceptance Criteria

- [ ] 0011's measured table has every cell filled or marked "could not run: <why>".
- [ ] Each lane has an explicit pass/fail against the D2 thresholds.
- [ ] The labelled 12-sprite set and prompts are saved (`scripts/spike-colibri.py`
      plus a small manifest) so the judge can be re-scored later.
- [ ] `project-context.md` hardware table reflects what was measured, not assumed.
- [ ] Sprite worker restarted afterwards and `make gpu-health` reports OK.

## Test Seam

Not applicable - a spike; its output is measurements in a decision record.

## Implementation Notes

- The sprite worker must stay usable: measure "RAM left" with it loaded, since
  that is the condition the text lane would run under.
- Colibri's own speed claims assume NVMe; every disk here is SATA. Expect the
  judge lane to be the slow one and do not extrapolate from its README.
- Paths: Cyrillic + space in the repo path; quote everything; run `make` from
  WSL; ASCII-only `.ps1`.
- Auth is enforced: any script calling the sprite API needs a bearer.

## Review Focus

Whether conditions were as stated (threads, RAM cap, worker loaded, card
idle), and whether the accuracy score was graded blind to which answer was
expected.

## Suggested Route

`/implement`, then `/review-code` on the 0011 edits.
