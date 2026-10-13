# Model provisioning - the manifest contract

Decision: [0014](../../decisions/0014-model-provisioning-manifest.md).
Manifest: `compose/develop/downloader/models.toml`.

The manifest answers one question: **what weights does a machine need, and how
does each one get there.** It is the only place that answers it. `models.txt`
answers a different, legacy question (what the retired llama.cpp router loads)
and is not part of this contract.

## The pieces

| File | Role |
|---|---|
| `compose/develop/downloader/models.toml` | The manifest. Declarative, commented, hand-edited |
| `scripts/models_manifest.py` | Shared reader: parsing, and the "is it present" question |
| `scripts/provision-models.py` | Fetches. Runs in the `downloader` container |
| `scripts/check-models-manifest.py` | Drift test. Host-only, no containers, no network |
| `scripts/list-models.sh` | Reports all four answers side by side |

`models_manifest.py` is underscored because the other two import it. Presence
logic lives there and **only** there: two copies of "is this model present"
would be the exact drift this whole mechanism exists to prevent.

## Commands

    make provision                  # required tier (56.4 GB)
    make provision tier=all         # every optional feature too (103 GB)
    make provision dry=1            # plan: what is MISSING, and how big
    make provision only=birefnet    # one entry
    make provision tier=all audit=1 # resolve EVERY entry against the Hub,
                                    #   present or not - true sizes, and proof
                                    #   each glob selects what it should
    make provision audit=1 files=1 only=X   # list the files X would fetch
    make test-models-manifest       # does the code reference anything undeclared?
    make models                     # declared / code-ref / on disk / served

`--audit` is the glob review. A filter's failure mode is a clean download that
dies at load time, so it cannot be caught by watching a fetch succeed; `audit`
applies each entry's `allow`/`ignore` through `filter_repo_objects` - the same
function `snapshot_download` uses - against the repo's real file list, and
prints exactly what would land. **Run it after changing any filter.** It found
12.2 GB of dead weight on its first run (see below).

`make provision` is deliberately **not** part of `make up`. Two containers
downloading into the same `/models` cache deadlocked on HuggingFace's file
locks; the downloader service idles on `tail -f /dev/null` for that reason.

Provisioning is **idempotent and resumable**. Anything present is skipped,
nothing is ever deleted, and one failure does not strand the rest - a 401 on
the single gated repo still leaves every other weight on disk.

## Entry schema

Required on every entry: `name`, `kind`, `tier`, `used_by`, `why`.

| Field | Applies to | Meaning |
|---|---|---|
| `name` | all | Manifest-local id. What `--only` takes |
| `kind` | all | `hf-cache` \| `hf-file` \| `checkpoint-tree` \| `local` |
| `tier` | all | `required` \| `optional` |
| `used_by` | all | The code that asks for it. **Enforced** by the drift test |
| `why` | all | Why this model and not another. Usually a measurement |
| `repo` | not `local` | Hub repo id |
| `file` | `hf-file` | Filename **in the repo** |
| `as` | `hf-file` | Save under **this** name instead. See below |
| `dest` | not `hf-cache` | Directory under the cache root |
| `allow` / `ignore` | Hub kinds | Glob filters. Not cosmetic - see below |
| `gated` | Hub kinds | Terms must be accepted on the Hub first |
| `fetch = "self"` | `checkpoint-tree` | Another project downloads it; only verify presence |
| `bytes` / `local_bytes` | any | Exact measured size; `local_bytes` when it differs from the Hub |
| `approx_gb` | any | Hand-entered estimate, used when `bytes` is unknown |
| `provenance` | any | `"substitute"`: **not** the file that was benched |
| `missing_on_source_host` | any | Declared because the code needs it, not because it was present |
| `skipping` | `optional` | What breaks without it. Required in spirit for optional entries |
| `files` | `local` | The filenames expected under `dest` |
| `notes` | any | Traps |

### The four kinds

Each is a different fetch, and getting it wrong does not fail loudly - it
produces files in a layout nothing looks in. Vocabulary in
[domain.md](../../domain.md).

- **`hf-cache`** -> `snapshot_download(cache_dir=root)`. Lands as
  `models--<org>--<name>/`, resolved by **repo id**. `hf download --local-dir`
  makes the wrong shape and diffusers will not find it.
- **`hf-file`** -> `hf_hub_download`, then a real copy placed at
  `dest/<as or file>`. A copy, not a symlink: the archive scripts and
  `ls -l /models/**/*.gguf` both expect real files.
- **`checkpoint-tree`** -> `snapshot_download(local_dir=dest)`, or nothing at
  all when `fetch = "self"`.
- **`local`** -> never fetched. Reported as MANUAL in the plan.

### `as` is not optional where it appears

Two entries are saved under a different name than the Hub publishes:

| Hub | Saved as | Because |
|---|---|---|
| `Qwen3-VL-8B-Instruct-Q4_K_M.gguf` | `Qwen3VL-8B-Instruct-Q4_K_M.gguf` | `BRAIN_8B_PATH` - no hyphen after `Qwen3` |
| `qwen-image-2512-Q3_K_M.gguf` | `Qwen-Image-2512-Q3_K_M.gguf` | `CORE_MODELS` capitalises the `gguf:<stem>` |

`gguf:<stem>` and `BRAINS[...]["path"]` resolve by **exact filename**.
Downloading under the Hub's own name leaves a file that exists, looks right in
a directory listing, and is never found. The drift test checks `gguf:` stems
and brain paths against the **saved** name for this reason.

### `allow` / `ignore` are load-bearing

Measured with `--audit` on 2026-10-13. Filtering takes the full set from
115.2 GB to **103.0 GB**, and every removed byte was verified unread:

| Entry | Unfiltered | Filtered | What was dropped |
|---|---|---|---|
| `clip-vit-large` | 6.38 GB | 0.005 GB | the same weights in four frameworks (`.h5`, `.bin`, `.safetensors`, `.msgpack`); only the tokenizer is used |
| `openpose-sd15` | 4.04 GB | 1.35 GB | the same weights four ways (.bin/.safetensors x fp32/fp16) plus demo PNGs |
| `all-in-one-pixel` | 7.09 GB | 3.97 GB | a 1.99 GB single-file `.ckpt` duplicate, and a 1.13 GB safety checker that `tasks.py` disables |
| `qwen-edit-lightning` | ~107 GB | 0.79 GB | a 20 GB fp8 full checkpoint and the variants not used |
| `qwen-companions-4bit` | +11.59 GB | 5.00 GB | a transformer this card cannot run |
| `openpose-sdxl` | 10.04 GB | 10.04 GB | only demo images; see below |

**Filters are per-repo, not a house rule.** The two openpose ControlNets look
identical and need opposite treatment: the SD1.5 repo publishes
`diffusion_pytorch_model.safetensors`, which diffusers prefers, so the `.bin`
can go; the SDXL repo publishes **no** such file, so there the `.bin` is what
loads and the same filter would break it. Decide per repo, with `--audit`.

`sdxl-base`'s fp16 allow-list was verified the same way: 22 files, all five
fp16 component weights plus every config, tokenizer, vocab and merges file.

Mind the case: `unsloth/Qwen-Image-Edit-2511-GGUF` publishes
`qwen-image-edit-2511-*.gguf`. A wrong-cased pattern does not 404 - it matches
nothing and reports success.

## The drift test

`make test-models-manifest`. Walks `src/sprite_generator/**.py` and fails when:

1. A Hub repo id is neither declared nor in `[[dead]]`. Case-insensitive -
   `CORE_TRIGGERS` and the step-2 tables are keyed by lowercased ids.
2. A `gguf:<stem>` from `CORE_MODELS` has no declared `hf-file` saving
   `<stem>.gguf`.
3. A `{MODELS}/...gguf` path in `BRAINS` has no declared entry.
4. The manifest contradicts itself: duplicate `name`, missing `used_by`, bad
   `tier` or `kind`, or a repo both declared and dead.
5. An entry is wrong for its kind. These are checked because each one, left
   out, yields a provisioner that fetches to the **wrong place** rather than
   one that errors:
   - `hf-file` without `repo`/`file`/`dest`, or with a `dest` only one level
     deep (two levels is the rule that keeps the llama.cpp router from
     advertising a GGUF as a chat model);
   - `hf-cache` **with** a `dest` - it lands in cache layout, by repo id, and
     a `dest` means someone has the wrong mental model;
   - `checkpoint-tree` without `repo`/`dest`;
   - `local` without `files` (its absence then cannot be detected) or **with**
     a `repo` (it cannot be downloaded);
   - any `optional` entry without `skipping`.

Scanned roots are `src/sprite_generator` and `scripts`, `.py` and `.sh`.
Shell is included because `two-step-test.sh` and `smoke-test.sh` were both
found pinning dead models.

Advisory, never fatal: declared-but-absent (per kind, via the shared `have()`)
and on-disk-but-undeclared. A thin deploy is a choice.

**`[[dead]]` is a decision, not an omission.** 14 entries, each with a reason.
Without it there is no way to distinguish "we considered this and said no"
from "nobody noticed".

## Adding a model

1. Add the stanza. Pick `kind` by *how it is loaded*, not by file extension -
   a `.safetensors` resolved by repo id is `hf-cache`; a `.gguf` opened by
   path is `hf-file`.
2. Fill `used_by` and `why`. The test fails without `used_by`; `why` is what
   stops the next person re-litigating it.
3. For `optional`, write `skipping`.
4. `make test-models-manifest`, then `make provision only=<name> dry=1`.

## Operating notes

- **Do not wrap `make provision` in a short `timeout`.** A full required-tier
  fetch on a cold machine is tens of GB; the SDXL ControlNet alone took over
  15 minutes here. An interrupted run is safe - re-running resumes - but the
  interruption looks like `Error 130` and leaves `.incomplete` blobs, which is
  precisely what the presence check now looks for.
- **`HF_DEBUG` is off by default.** At `1`, huggingface_hub prints a full curl
  line per request and buries the plan. Set it only when diagnosing.
- **`du` on a `models--*` directory is not a size.** Snapshots are symlinks
  into a top-level xet `blobs/` store; a complete 2.1 GB repo measured 824 KB.
  Use `audit=1`, or `ls -lL` through the symlinks.
- **Provisioned files are root-owned.** The downloader runs as root, so
  everything it writes under `MODELS_DIR` is `root:root`. Deleting or moving a
  model afterwards needs `sudo`, or a throwaway container
  (`docker compose run --rm --entrypoint sh downloader -c 'rm -rf /models/...'`).
  Harmless at runtime - the services read as root too - but it surprises
  anyone tidying up by hand.

## Known limits

- ~~Sizes for `hf-cache` entries are hand-entered.~~ **Closed 2026-10-13 by
  `--audit`.** Every `approx_gb` in the manifest is now the Hub's own sum over
  the files that entry's globs actually select. Re-derive with
  `make provision tier=all audit=1`.
- **One entry cannot be reproduced.** `Qwen-Image-2512-Q3_K_M` on the source
  host is a third-party 9.69 GB build whose publisher was never recorded;
  unsloth's 9.93 GB build is the declared substitute. See 0012 line 27.
- **The clean-machine path is unverified.** No second machine has run it.
- **The manifest does not cover the database, API keys or the ACE-Step venv.**
  The venv and `llama-server` are baked into `Dockerfile.cuda`; keys come from
  `make api-key`; a fresh deploy's Postgres starts empty by design.
