-- --------------------------------------------------------------------------
-- Background model downloads, queued from the UI.
-- --------------------------------------------------------------------------
--
-- WHY A TABLE OF ITS OWN, AND NOT `jobs`
--
-- `jobs` is sprite-shaped: it carries `sheet_path`, `atlas_path`, `stage` and
-- a `spec` describing a concept prompt, actions, directions and cell size. A
-- download has none of those. More to the point, `.ai/domain.md` already
-- records that "task" means four different things in this repo and that the
-- confusion has cost real time; putting weights into `jobs` would make "job"
-- mean a fifth thing, in the one table whose rows the queue panel reads.
--
-- WHY NOT CELERY
--
-- `sprite-worker` runs `--concurrency=1 --pool=solo` on the default queue, so
-- one task at a time. A 16 GB download submitted there would hold that single
-- slot and block EVERY generation for the length of the download. The
-- downloader container consumes this table instead - see
-- scripts/provision-daemon.py and .ai/decisions/0014.
--
-- SINGLE WRITER, AND IT IS NOT NEGOTIABLE
--
-- Two processes downloading into the same /models cache deadlocked on
-- HuggingFace's file locks ("Still waiting to acquire lock on
-- /models/.cache/huggingface/download/..."), which is why the downloader
-- service used to idle rather than fetch on startup. One consumer of this
-- table gives serialisation for free; `make provision` is a SEPARATE
-- container and does not, so both paths take an advisory lock at
-- /models/.provision.lock. See scripts/provision-models.py.
CREATE TABLE IF NOT EXISTS model_downloads (
    id              UUID PRIMARY KEY,

    -- The manifest entry's `name` (models.toml), not a repo id: the entry is
    -- what carries kind, dest, allow/ignore and `as`, and the repo id alone
    -- is not enough to fetch correctly. Two manifest entries legitimately
    -- share one repo (the Q2_K and Q3_K_M image quants), so a repo id would
    -- also be ambiguous.
    entry           TEXT NOT NULL,

    -- queued -> running -> done | failed | cancelled
    --
    -- `cancelled` is set by the API on a row that has not started yet. A
    -- RUNNING download is not interrupted: huggingface_hub would leave
    -- .incomplete blobs, and the presence check treats those as absent, so a
    -- half-cancelled entry silently re-downloads from scratch later.
    status          TEXT NOT NULL DEFAULT 'queued',

    -- What the manifest declared, so the UI can draw a bar before any byte
    -- has landed. Authoritative since `make provision audit=1` started
    -- summing the Hub's own sizes over the files each glob selects.
    bytes_total     BIGINT,

    -- Measured from disk by the daemon while it runs. NOT a progress callback:
    -- huggingface_hub exposes none that survives its own retries, and for an
    -- hf-cache entry the bytes land in a shared blob store where they cannot
    -- be attributed to one repo. So this is exact for `hf-file` and a floor
    -- for everything else - the UI must not present it as a precise percent.
    bytes_done      BIGINT NOT NULL DEFAULT 0,

    error           TEXT,

    -- Who asked. auth.require returns the principal; a download is minutes of
    -- bandwidth and tens of GB of disk, so it is worth knowing.
    requested_by    TEXT NOT NULL DEFAULT 'unknown',

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at      TIMESTAMPTZ,
    finished_at     TIMESTAMPTZ
);

-- The daemon's hot query: oldest queued row. Partial, because the table is
-- append-only history and the queued set is tiny by comparison.
CREATE INDEX IF NOT EXISTS model_downloads_queued_idx
    ON model_downloads (created_at)
    WHERE status = 'queued';

-- One in-flight or waiting row per entry. Enqueuing the same model twice
-- while the first is still going is the obvious double-click bug, and the
-- consequence is two writers on one cache - the deadlock above. A finished or
-- failed row does not participate, so a retry after a failure is allowed.
CREATE UNIQUE INDEX IF NOT EXISTS model_downloads_active_entry_idx
    ON model_downloads (entry)
    WHERE status IN ('queued', 'running');

CREATE INDEX IF NOT EXISTS model_downloads_recent_idx
    ON model_downloads (created_at DESC);
