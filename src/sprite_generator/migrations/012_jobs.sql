-- Async sheet jobs for external consumers (something2).
--
-- Deliberately NOT folded into sprite_images. That table is one row per IMAGE;
-- a job is a REQUEST that may produce a sheet, an atlas and a pile of
-- intermediate cells, and it has to stay queryable after the images are gone.
--
-- It is also not Celery's result backend. That expires on a TTL and does not
-- survive a broker flush, and something2's whole model is "hand me an id now,
-- ask about it whenever" - which needs the id to still resolve tomorrow.

CREATE TABLE IF NOT EXISTS jobs (
    -- The id handed back to the caller. A UUID rather than a serial so it is
    -- safe to expose and cannot be enumerated.
    id              UUID PRIMARY KEY,

    -- queued -> running -> done | failed | cancelled
    status          TEXT NOT NULL DEFAULT 'queued',

    -- The request exactly as received: concept prompt or source image,
    -- actions, directions, frames, cell size. Kept verbatim so a job can be
    -- replayed and so a caller can be told what it actually asked for.
    spec            JSONB NOT NULL DEFAULT '{}'::jsonb,

    progress_pct    INTEGER NOT NULL DEFAULT 0,
    progress_msg    TEXT,
    -- Which of the five build stages is running. Separate from progress_msg
    -- because it is a fixed vocabulary a client can branch on.
    stage           TEXT,

    sheet_path      TEXT,
    atlas_path      TEXT,
    error           TEXT,

    celery_task_id  TEXT,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at      TIMESTAMPTZ,
    finished_at     TIMESTAMPTZ
);

-- something2 polls "what changed since I last asked", so updated_at is the
-- access path that matters, not created_at.
CREATE INDEX IF NOT EXISTS jobs_updated_at_idx ON jobs (updated_at DESC);
CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs (status);

-- Keep updated_at honest without every writer having to remember it.
CREATE OR REPLACE FUNCTION jobs_touch_updated_at() RETURNS trigger AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS jobs_touch ON jobs;
CREATE TRIGGER jobs_touch BEFORE UPDATE ON jobs
    FOR EACH ROW EXECUTE FUNCTION jobs_touch_updated_at();
