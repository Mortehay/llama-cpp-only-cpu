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
| `scripts/gguf-head.py` | What model is this GGUF *really*: header, KV pairs, tensor index. Takes a local path or a URL (4-128 MB Range request), so a candidate quant can be identified before it is fetched |
| `scripts/provision_fetch.py` | The fetching itself, one function per kind, shared by the CLI and the daemon. Reporting is injected; locking is the caller's job |
| `scripts/provision-daemon.py` | The `downloader` service's command. Single consumer of `model_downloads` |
| `src/sprite_generator/models_status.py` | `GET /api/models/status`, and the download queue the UI posts to |

## The background download lane

    POST   /api/models/download              {"entry": "<models.toml name>"}
    GET    /api/models/downloads
    POST   /api/models/downloads/{id}/cancel
    GET    /api/models/status?days=30

The API **never fetches**. It inserts a `model_downloads` row; the `downloader`
container's daemon is the single consumer. Why not Celery, why not the `jobs`
table, and what each guard is for: `.ai/decisions/0014` D6a.

Rules worth knowing before changing any of it:

- **One writer per cache, enforced twice.** The daemon is a single consumer
  *and* every fetch - its own and `make provision`'s - is taken under
  `flock(/models/.provision.lock)`. The CLI holds it for a whole tier; the
  daemon per entry.
- **A queued entry cannot be queued twice** (partial unique index on `entry`
  where status is queued/running) - 409.
- **A running download cannot be cancelled**, only a queued one. Interrupting
  leaves `.incomplete` blobs, which the presence check reads as absent, so the
  entry would restart rather than resume.
- **`bytes_done` is measured from disk, following symlinks.** An HF snapshot is
  symlinks into a shared blob store; measuring the links reported **17 KB for a
  42 MB repo**. Inodes are deduped, and it can still slightly *exceed*
  `bytes_total` (43.2 vs 41.0 MiB measured) because the on-disk tree carries HF
  metadata the Hub's file sum does not - so the API serves a clamped
  `progress_pct` and the raw bytes stay honest.
- **`local` and `fetch = "self"` entries are refused with an explanation**,
  not a download button: one was trained here, the other is fetched by the
  project that owns it.

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

Instead `up`, `build`, `rebuild` and `rebuild-clean` end with **`models-warn`**:
a host-only, network-free comparison of the manifest against `MODELS_DIR` that
lists missing *required* entries and says `Run: make provision`. Silent when
complete, never fatal. Until 2026-10-13 those targets instead ran the legacy
`download_models.sh`, which printed `[SKIPPED] Qwen2.5-3B-Instruct-Q4_K_M.gguf`
on every start and fetched nothing that mattered - `make sync-models` is now
its only caller.

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
- **`hf-file`** -> `hf_hub_download(local_dir=dest)`, landing a real file at
  `dest/<as or file>`, renamed in place when `as` differs. `local_dir`, not
  `cache_dir`: via the cache the file is written to the blob store and then
  copied out, costing its size **twice** (34 GB of peak disk for the 17 GB
  brain). A real file, not a symlink, because the archive scripts and
  `ls -l /models/**/*.gguf` both expect one.
  Side effect: `local_dir` leaves a small `.cache/huggingface/` beside the
  file, holding HF's resume metadata. Harmless - nothing enumerates these
  directories, `core_models.gguf_file()` builds exact paths - but it is why a
  provisioned machine's tree looks slightly different from one populated by
  hand. Leave it; it is what lets a half-finished download resume.
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

1. **If it is a GGUF, identify it first:**
   `python3 scripts/gguf-head.py <url>`. The architecture label is not
   evidence - 0012 found four mislabelled files - so check the tensor prefixes,
   the block count and `general.file_type` (the quant level). This works on a
   remote URL, so do it *before* spending the download.
2. Add the stanza. Pick `kind` by *how it is loaded*, not by file extension -
   a `.safetensors` resolved by repo id is `hf-cache`; a `.gguf` opened by
   path is `hf-file`.
3. Fill `used_by` and `why`. The test fails without `used_by`; `why` is what
   stops the next person re-litigating it.
4. For `optional`, write `skipping` - the test fails without it.
5. `make test-models-manifest`, then
   `make provision only=<name> audit=1 files=1` to see exactly which files the
   entry selects, and only then `make provision only=<name>`.

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

## The Models tab

`frontend/src/tabs/Models.tsx`, reading `GET /api/models/status`. Four stat
tiles, three state tables (required / optional / downloads), three statistics
tables (brains / image models / audio), and two time charts.

Design decisions worth not re-litigating:

- **Three families, three shapes, never one table.** Tokens and tok/s exist
  only for brains; an image model draws and never tokenises; audio rows carry
  **no model at all**, so audio is per-kind totals with that stated on the
  page. One table with half the columns blank would recreate the exact
  `llm_name`-is-not-an-LLM confusion `.ai/domain.md` records.
- **Two charts, not one dual-axis chart.** Calls and tokens are different
  measures: measured 2,516 calls/day against 312,776 tokens/day, **124x
  apart**. One shared y-axis would flatten the calls line to the baseline and
  make any crossing point meaningless.
- **The chart palette is validated, not chosen.** `#1fa5c4` teal (image),
  `#8577e8` violet (text), `#d96a93` pink (audio), in that fixed order,
  passing all five computable checks against the dark card surface `#212138`:
  lightness band, chroma floor, CVD separation (worst adjacent ΔE 9.3 deutan),
  normal-vision floor (16.5), contrast. **Re-run the validator if you change
  one** - the ordering matters too, because violet sits between teal and pink
  so the weakest CVD pair is never adjacent.
- **`--success`/`--warn`/`--danger` are reserved for state** and are not used
  as series colours: a pink "audio" line is identity, an amber one would imply
  a warning. Every status pill carries an icon and a word, never colour alone.
- **Progress bars use the server's clamped `progress_pct`**, never the raw
  bytes - those can exceed the declared total (measured 43.2 against
  41.0 MiB) because the on-disk tree carries HF metadata the Hub's file sum
  does not.
- Polling runs only while a download is queued or running, as the Activity tab
  does: an idle machine does not need a request every three seconds.

Verified: `tsc -b` clean; chart geometry replayed over the real 30-day payload
(finite coordinates, everything inside the viewBox, absent series-days reading
0 rather than NaN, date labels 654px apart); and the **authenticated page was
rendered and looked at**, with no console errors.

Seeing it needed a scripted browser, because the app reads its token from
`localStorage` only and a plain headless `--screenshot` shows the signed-out
error state. The way that does NOT involve putting a credential in a file the
API serves: run Chromium with puppeteer **inside the compose network**, talk to
`sprite-generator:8001` directly, and pass the token as an env var to a
`page.evaluate` that seeds storage. No host ports, nothing written to
`static/`.

Looking at it paid for itself - five defects that every automated check had
passed:

| Symptom | Cause |
|---|---|
| `0.00 GB` beside several models | one formatter for every magnitude; a 1 MB config and a 16 GB brain cannot share it |
| Descriptions ending in a comma | `why` is a wrapped multi-line TOML string, and taking line one cuts mid-clause |
| …then a whole size breakdown in one cell | the first-sentence guard fell through when the first sentence was short |
| `0.00 GB / 0.04 GB · 100%` | a *finished* row showing a disk measurement against a declared total - three numbers disagreeing |
| `hf-/cache`, `16.51 / GB` | narrow columns wrapping mid-token, which reads as corruption |

None is a logic error; all five are legibility, and none would have surfaced
without rendering the page.

## Known limits

Sizes and filters are no longer on this list: `--audit` makes every
`approx_gb` the Hub's own sum over the selected files (re-derive with
`make provision tier=all audit=1`), and the three subtractive filters were
load-tested offline from a directory holding only their filtered files.

- **One entry is not byte-reproducible**, though it is verified equivalent.
  `Qwen-Image-2512-Q3_K_M` on the source host is a third-party 9.69 GB build
  whose publisher was never recorded (0012 line 27). unsloth's 9.93 GB build
  is the declared substitute, and `scripts/gguf-head.py` shows the two are
  structurally identical - 1933 tensors, `qwen_image`, `file_type` 12,
  60 blocks - so it is a different quantisation run, not a different model.
  **Speed and fit on a 12 GB card are unmeasured** for the substitute; 0012's
  253.5 s was the local build, and 246 MB more is not free.
- **The clean-machine path is unverified end to end.** No second machine has
  run it. What *has* been checked here: the provisioner driven from an empty
  directory for all three kinds, idempotent resume, and a fresh `.env` from
  `.env.example` resolving through `docker compose config` with the right
  mounts and online/offline split. What has not: WSL, Docker, the CUDA image
  build, migrations, and the first `make api-key` on real fresh hardware.
- **A real multi-GB `hf-file` fetch has not been run**, only a tiny one. The
  path is size-independent (`hf_hub_download(local_dir=...)` plus an
  `os.replace` rename on one filesystem), and all three real entries match the
  Hub byte-for-byte, so a fresh fetch would reproduce what is on disk - but
  that is an argument, not a measurement.
- **The gated 15 GB download has not been run**, only its authorisation
  (`--audit` resolves `stable-audio-open` with the configured token, so the
  terms are accepted). The fetch itself is the same `snapshot_download` path
  exercised twice on the ControlNets.
- **The manifest does not cover the database, API keys or the ACE-Step venv.**
  The venv and `llama-server` are baked into `Dockerfile.cuda`; keys come from
  `make api-key`; a fresh deploy's Postgres starts empty by design.
