# 0014 - A model manifest, so a second machine is possible

Date: 2026-10-13
Status: **Accepted. Built 2026-10-13.** Verified against the development host;
**not yet run on clean hardware** - see "Unverified at time of writing".
Spec: [specs/model-provisioning/contract.md](../specs/model-provisioning/contract.md).

## Context

The request was "add all present models to `models.txt` and fix the download
process so this can be deployed on another PC". Inspection turned that premise
over: `models.txt` is not the file that would do it, and "all present models"
is not the right set.

**`models.txt` has no live consumers.** It feeds the llama.cpp `llm-server`
router, which retired to the `legacy-llm` compose profile on 2026-10-01 (0013).
Its one remaining active entry, `Qwen2.5-3B-Instruct-Q4_K_M.gguf` (1.8 GB), was
superseded the same day: `worlds.py` calls `text.run_text`, which reaches the
brain, not the router. The file had been drifting for months while the brain
lane, the GGUF lane and the Edit lane grew up beside it.

**Nothing fetched the live weights.** All 112 GB - both brains, the image
GGUFs, the ACE-Step checkpoints, every `models--*` cache tree - was placed by
hand. `make up` and `make build` download none of it, despite README "First
run" step 5 saying otherwise. A fresh machine following the README got a
healthy stack with an empty `/models` and discovered it one 500 at a time.

**`models.txt`'s schema cannot express what is needed.** `<repo_id> <file>`
fed to `hf download --local-dir` produces a flat tree. Three of the four kinds
of thing under `MODELS_DIR` need something else, and getting the kind wrong
does not fail loudly - it produces files in a layout nothing looks in.

**The disk is not a trustworthy source of truth either.** Taking inventory
surfaced drift in both directions:

| | |
|---|---|
| Present, referenced by nothing | `Qwen2.5-3B` (1.8 GB), `joelseytre/toonout`, the sd.cpp bench pair in `image-gguf/qwen-image-2.1/` (8.4 GB) |
| Referenced by live code, absent | **both** openpose ControlNets, `stabilityai/sdxl-turbo` |

The ControlNet gap is not cosmetic. `use_pose` turns on automatically for any
action with a cycle (`tasks.py`), and `from_pretrained` under
`HF_HUB_OFFLINE=1` cannot recover a missing repo - so pose-conditioned step 2
has been broken on the development host, for both families, with no error
anywhere that names the cause. The cold archive those weights were presumed to
be in (`/mnt/d/wsl-model-archive`) **no longer exists**.

## Options considered

1. **Extend `models.txt`** with destination and mode columns. Rejected: its
   consumer is POSIX `sh` + `read` in an Alpine image, and `snapshot_download`
   with glob filters is not expressible there. It is hand-rolling a package
   manager in shell, to extend a file nothing reads.
2. **Two lanes** - keep `models.txt` for flat GGUFs, generalise
   `fetch-qwen-edit.py` into a cache fetcher. Rejected: two places to forget an
   entry. This repo already documents three "same fact in two places, nothing
   keeps them in sync" bugs (`trigger_for`/`CORE_TRIGGERS`,
   `CORE_MODELS`/`KNOWN_MODELS`, `model_gateway.GATED`); a fourth is not a
   neutral cost.
3. **One declarative manifest plus one provisioner.** Chosen.

## Decision

### D1. `models.toml` is the manifest. `models.txt` is left alone.

Not deleted and not migrated - it still works for anyone reviving
`legacy-llm`, and rewriting it would imply it matters. It is marked legacy in
the Makefile, in `list-models.sh` and in the README.

TOML over JSON because every entry carries prose - `why`, `notes`, `skipping` -
and this repo's files earn their length by recording *why*. A commentless
format would have lost that, and the prose is most of the value.

### D2. Four kinds, because there are four fetches

Recorded in `.ai/domain.md`. `hf-cache` must land in HF cache layout and be
resolved by repo id; `hf-file` goes to an explicit destination two levels down;
`checkpoint-tree` is a layout another project owns; `local` cannot be
downloaded at all.

The depth rule for `hf-file` is load-bearing: the retired router advertised
every `.gguf` one level below `/models` as a *chat* model, which is how a
Qwen-Image transformer once appeared in the chat roster named "gguf".

### D3. `tier`, not VRAM tiers

One manifest, pinning what the development host runs. Target hosts may have a
bigger card, but everything declared **runs** on one - the 12 GB choices
(Q2_K, `--n-cpu-moe 24`, Q3_K_M Edit) are conservative, not incompatible, and
every `min_free_mb` gate passes more easily with more VRAM. Tuning for a larger
card already has a mechanism: `BRAIN_35B_CPU_MOE`, `BRAIN_35B_MIN_FREE_MB`,
`QWEN_EDIT_GGUF_FILE`. Building a VRAM-tiered manifest now would be
configurability for hardware that does not exist.

`required` is the default lane. `optional` carries a `skipping` field naming
exactly what breaks, so a thin deploy is an informed choice rather than a
discovery.

### D4. The source of truth is the code, not the disk

Entries derive from `CORE_MODELS`, `BRAINS`, `GGUF_COMPANION_REPOS`,
`cutout.py`, `audio_engine.py`, `qwen_edit.py` and `tasks.py`. This is the only
option that *surfaces* the missing ControlNets instead of propagating them: a
disk-derived manifest would have faithfully reproduced the hole.

Hence `missing_on_source_host = true`, which marks an entry declared because
the **code** needs it, not because it was present when the manifest was
written.

### D5. The drift test is the deliverable

`make test-models-manifest` walks the source for repo ids, `gguf:` stems and
brain paths, and fails if one is neither declared nor listed under `[[dead]]`.

This is the part that matters. `models.txt` did not rot because it was written
badly; it rotted because nothing ever compared it to the code. Without the
test, `models.toml` is the same file with better syntax and a later rot date.
Run against the real tree it reports exactly the two absent required models and
nothing else.

`[[dead]]` is the other half: 14 entries that are referenced or present and
deliberately not fetched, each with a reason. An omission and a decision are
otherwise indistinguishable.

### D5a. "Present" is stricter than `core_models.is_cached()`

Found by running the provisioner during a real download: huggingface_hub
populates `snapshots/` with symlinks early, so a repo **still downloading**
answered "present" to the snapshot test, and a resumed `make provision` would
have skipped it - leaving a model that exists on disk and cannot load. The
shared `have_cache()` therefore also rejects any repo with a `.incomplete`
blob.

Deliberately divergent from `core_models.is_cached()`, which keeps the looser
test. For the runtime, a half-downloaded repo nothing has tried to load yet is
harmless; for a provisioner it is the worst possible answer.

Two other disk facts the same run established, both recorded in the manifest:

- `fetch_file` uses `local_dir=`, not `cache_dir=`. Via the cache the file
  lands in the blob store and must be copied out, costing its size **twice** -
  34 GB of peak disk for the 17 GB brain, on a host with ~60 GB free.
- `du` on a `models--*` directory is not a size. The freshly fetched SD1.5
  ControlNet measured **824 KB** as a directory while holding 1.44 GB + 722 MB,
  because snapshots symlink into a top-level xet `blobs/` store.

### D5b. The drift test scans `scripts/` too

Added after the first version, which scanned only `src/sprite_generator`,
missed that **`scripts/two-step-test.sh` pinned both dead models** in shell
variables - `stabilityai/sdxl-turbo` as `CORE_MODEL` and
`Onodofthenorth/SD_PixelArt_SpriteSheet_Generator` as `SHEET_MODEL`. Neither
is on disk, so `make test-flow`, the end-to-end harness, could only fail. Both
now default to the roster default and stay overridable by environment.

`scripts/smoke-test.sh` had the same defect from the other direction: it
pinned `sd_model_checkpoint: stabilityai/sdxl-turbo` in its `txt2img`
`override_settings`. The override is gone - omitting it lets the facade pick
its own default, which is what a real something2 request does. Its heading
also claimed "first run also downloads the model"; the worker is offline and
downloads nothing.

So **three** of this repo's end-to-end harnesses were pinned to models that
cannot load, which is the clearest evidence that the drift this ADR is about
had already spread beyond `models.txt`. A hardcoded dead model in a test
harness fails exactly as badly as one in the app and is less likely to be
noticed, so repo ids are now checked in `.py` and `.sh` under both roots. `gguf:` stems and `BRAINS` paths stay app-only: they
are roster concepts, and collecting them from `scripts/` picks up fixtures
(the gateway test's short label `"gguf:qwen"` is not a model).

### D6. `make provision` stays out of `make up`; `make up` warns instead

Two containers downloading into the same `/models` cache deadlocked on
HuggingFace's file locks, which is why the downloader service idles on
`tail -f /dev/null`. Blocking a start on a 56-103 GB fetch would also be hostile.

**Corrected 2026-10-13, after the owner noticed the symptom.** The first pass
of this work wrote "`make up` downloads nothing" into the README and CLAUDE.md
and *did not check whether it was true*. It was not: `up`, `build`, `rebuild`
and `rebuild-clean` each ran `run --rm downloader download_models.sh`, so every
start printed

    Reading models to download from /usr/local/bin/models.txt...
    [SKIPPED] Qwen2.5-3B-Instruct-Q4_K_M.gguf already present.

- a 1.8 GB model nothing loads, while fetching **none** of the 56.4 GB the
stack needs. Documenting the behaviour I wanted instead of the behaviour
present was the mistake; the lesson is that a claim about a Makefile target is
checkable in one grep.

Those four call sites are gone. `make sync-models` is now the only caller, and
it is explicitly the legacy lane. In their place each target ends with
`models-warn`: a host-only, network-free comparison of `models.toml` against
`MODELS_DIR` that lists the missing **required** entries and says
`Run: make provision`. Silent when complete, never fatal. It exists because
the exact failure this lane targets is a stack that comes up healthy with an
empty `/models`.

`build` also ran `docker compose build` twice - once without `--env-file` -
and that duplicate went with it.

### D7. The `sdxl-turbo` defaults are repointed at `default_model()`

Five sites: `generate_sheet`'s `Form` default, `main.py`'s retry fallback,
`model_gateway.GATED`, `get_sd_pipeline`'s signature and
`generate_core_task`'s. 0002 dropped turbo for cause - distilled, so guidance 0,
so every negative prompt a silent no-op - and it is not on disk, so those
defaults could only fail.

This is a **behaviour change**, not a cleanup: a caller that omitted
`llm_name` previously got an error and now gets the roster default. Resolved at
call time, not in the signature, so it follows the roster rather than import
order, and so the gateway's fallback and the task's default cannot disagree.
The name survives only in a legacy cache-directory normalisation; the entry is
in `[[dead]]` so the drift test does not reopen the question.

## Consequences

- A second machine is `make provision` plus the README's new "Deploying to
  another machine" section, instead of tribal knowledge.
- Adding a model means adding a stanza. The drift test enforces it, which is a
  small tax on every new checkpoint and the entire point.
- `make download` no longer silently appends to `models.txt`. It prints the
  stanza to paste, because `kind`, `tier`, `used_by` and `why` cannot be
  inferred from a repo id.
- The downloader image now builds from the repo root (`context: ../../`), like
  `sprite-generator`, because the provisioner lives in `scripts/`.
- `scripts/models_manifest.py` holds the parse-and-presence logic that the
  provisioner and the checker share. Underscored so both can import it. Two
  copies of "is this model present" would be exactly the drift this ADR exists
  to prevent.
- Three facts about the host that were wrong in the docs are now corrected:
  the D: cold archive is gone, there is no models VHD, and `README.md` was
  never as stale as `CLAUDE.md` claimed.

## Unverified at time of writing

- **The clean-machine walkthrough has not been run.** No second machine exists
  to test on. This is the largest remaining risk and the README section says
  so in its first line. Partially mitigated: the provisioner was driven from
  an **empty** models directory (all three kinds, correct layouts, idempotent
  resume), so what is untested is the surrounding runbook - WSL, Docker, the
  CUDA build, migrations, `make api-key` - not the fetch itself.
- ~~Only the `hf-cache` fetch path has been executed.~~ **Closed 2026-10-13.**
  Both ControlNets were really fetched through `make provision` (hf-cache),
  and all three kinds were then exercised from an **empty** models directory
  using a throwaway manifest of tiny files. Each landed in the right shape:
  the hf-file two levels down under its `as` name with **no cache duplicate**
  (confirming the `local_dir` fix), the hf-cache under
  `models--org--name/snapshots/`, the tree flat at `dest`. A second run
  fetched nothing, so resume is idempotent. What remains untested is a real
  multi-GB `hf-file` and the gated 15 GB download; their code paths are the
  ones just exercised, only the size differs.
- ~~`model_gateway` could not be imported on the host.~~ **Closed
  2026-10-13**: `make test-model-gateway` passes in the container, and its
  `model_of tasks.generate_core_task: sdxl+nerijs` assertion is D7's gateway
  half verified by running.
- **`Qwen-Image-2512-Q3_K_M` cannot be reproduced byte-for-byte**, but the
  substitute is **verified equivalent** (2026-10-13). The local build is
  9,686,940,192 bytes and 0012 line 27 records it as explicitly "not unsloth's
  9.93 GB build", from a publisher never written down. Rather than trust a
  label - 0012's own rule is that `general.architecture` is not evidence,
  since four of that batch were mislabelled - both GGUF headers were read and
  compared (the local file directly, unsloth's via an HTTP Range request for
  its first 4 MB, which covers the KV block and tensor index):

  | | local | unsloth |
  |---|---|---|
  | tensors | 1933 | 1933 |
  | architecture | `qwen_image` | `qwen_image` |
  | `general.file_type` | 12 (Q3_K_M) | 12 (Q3_K_M) |
  | transformer blocks | 60 (0-59) | 60 (0-59) |
  | tensor prefixes | identical | identical |

  A different quantisation *run* of the same model at the same quant level,
  not a different model. What is still unmeasured is **speed and fit**: 0012's
  253.5 s at 1024x1024 was measured on the local build, and 246 MB more on a
  12 GB card is not free. Re-measure; Q2_K is the documented fallback.
- ~~Sizes for `hf-cache` entries are approximate.~~ **Closed by `--audit`.**
  Every `approx_gb` is now the Hub's own sum over the files that entry's globs
  select: 56.4 GB required, 103.0 GB for everything. This had to come from the
  Hub rather than from disk, because HF snapshots symlink into a top-level
  xet `blobs/` store and `du` on a repo directory badly under-reports - the
  freshly fetched SD1.5 ControlNet measured 824 KB as a directory while
  holding 1.44 GB + 722 MB of weights.
- ~~Most `allow` lists are unexercised.~~ **Closed by `--audit`** (added
  2026-10-13). It applies each entry's globs through `filter_repo_objects` -
  what `snapshot_download` itself uses - against the repo's real file list, so
  a filter can be checked without a fetch. This matters because a bad filter
  downloads cleanly and fails at *load* time, which watching a fetch succeed
  would never catch. `sdxl-base`'s fp16 list was the worry and is correct: 22
  files, all five fp16 component weights plus every config and tokenizer file.
  The three subtractive filters added after that audit were then **load-tested
  for real**: provisioned into an empty directory so only the filtered files
  existed, then loaded with `HF_HUB_OFFLINE=1` so nothing could quietly fetch
  a missing piece. 5.4 GB instead of ~17.5 GB, and all four loads passed -
  the CLIP tokenizer, the SD1.5 ControlNet (361M params), the All-In-One
  pipeline (unet 860M, every component present), and the **real step-2 shape**,
  `StableDiffusionControlNetImg2ImgPipeline` over both filtered repos.
- **`stable-audio-open`'s gating is no longer unknown.** `--audit` resolved it
  against the Hub with the configured `HF_TOKEN`, so the terms are accepted
  for this account. The 15 GB download itself still has not been run.

## Verified (2026-10-13, development host)

| Check | Result |
|---|---|
| `check-models-manifest.py` against `src/sprite_generator` | passes; 20 repo refs, 2 gguf stems, 2 brain paths all accounted for |
| …with `--models-dir` | 2 advisories, both the known-absent ControlNets |
| `provision --tier required --dry-run` | 11 present, 2 to fetch, ~4.0 GB |
| `provision --tier all --dry-run` | 21 present, 1 manual, 2 to fetch |
| Downloader image rebuilt with `context: ../../` | builds clean |
| `make provision` end to end, through compose | **both ControlNets fetched**; required tier now 13/13 present, 0 to fetch |
| `ignore` globs | took effect: 6 files fetched instead of 9, demo images skipped |
| All three fetch KINDS, from an **empty** models dir (throwaway manifest, tiny files) | hf-file landed two levels down under its `as` name with no cache duplicate; hf-cache under `models--org--name/snapshots/`; tree flat at `dest`. Re-run fetched nothing |
| `--audit` over all 24 entries | real sizes from the Hub: **56.4 GB required, 103.0 GB all**. Found 12.2 GB of dead weight, now filtered |
| `sdxl-base` fp16 allow-list | correct: 22 files, all five fp16 component weights plus every config/tokenizer/vocab/merges |
| `stable-audio-open` gating | resolves with the configured `HF_TOKEN`, so its terms are accepted for this account |
| `make test-model-gateway` | all checks pass, including `model_of tasks.generate_core_task: sdxl+nerijs` - D7's gateway half |
| Drift test, negative control | a bogus repo planted in a `.sh` is caught and named |
| Fresh `.env` from `.env.example` | `docker compose config` resolves clean, no warnings; the `scripts` mount the provisioner needs is present, downloader gets `HF_HUB_OFFLINE=0` while the services get `1` |
| Both brain GGUFs, by tensor evidence (`scripts/gguf-head.py`) | 35B: `qwen35moe`, 40 blocks, 256 experts / 8 used - matches `--n-cpu-moe 24`'s 40-layer MoE. VL-8B: `qwen3vl`, 36 blocks. Independent of the byte-size match |
| The three subtractive filters, **load-tested offline** from a directory holding only the filtered files | 4/4 pass: CLIP tokenizer; SD1.5 ControlNet (361M params); All-In-One pipeline (unet 860M, all components); `StableDiffusionControlNetImg2ImgPipeline` over both - the real step-2 shape. 5.4 GB vs ~17.5 GB unfiltered |
| Q3_K_M substitute, by GGUF header comparison | structurally identical to the local build: 1933 tensors, `qwen_image`, `file_type` 12, 60 blocks. A different quantisation run, not a different model |
| **Pose-conditioned step 2, end to end** | **WORKS.** `./scripts/two-step-test.sh` passed: core 512x512 / 85.5% transparent, sheet 4 frames x 2 actions / 64.8% transparent, real content. Worker log confirms it took the ControlNet path, not an unposed fallback: *"Pose cycles found; loading ControlNet"*, the SDXL substitution, then `'move right': pose-conditioned, strength 0.75` and `'idle': pose-conditioned, strength 0.75`. First working run since the cold archive was lost |
| The SDXL ControlNet's missing safetensors | that same log shows diffusers probing for `diffusion_pytorch_model.safetensors`, not finding one, and falling back to the `.bin` - so the decision NOT to filter that repo to safetensors/fp16 is proven, not merely reasoned |
| `make test-flow`, `make smoke` | both were pinned to dead models and could only fail; fixed, and test-flow now passes |
| Provenance, 35B brain | `unsloth/Qwen3.6-35B-A3B-GGUF`, 17,730,509,792 bytes - exact match |
| Provenance, Q2_K image | `unsloth/Qwen-Image-2512-GGUF`, 7,333,837,344 bytes - exact match, and matches 0012's recorded sha256 prefix |
| Provenance, VL-8B brain | `unsloth/Qwen3-VL-8B-Instruct-GGUF`, 5,027,785,568 vs 5,027,784,800 local - 768-byte metadata difference, same quant |
| Both openpose ControlNets | HTTP 200 on the Hub; fetchable |
| `py_compile` on all changed modules | clean |
