-- 013: reference assets, style profiles, training runs, API keys, job kinds.
--
-- Four tables and one view, added together because they are one feature: the
-- system learns a style from examples the user supplies, and the examples,
-- what was measured from them, what was trained on them, and who may ask for
-- generation are all parts of that loop.
--
-- The view is the important part for the UI. See the note above assets_v.

-- --------------------------------------------------------------------------
-- jobs.kind - a job is no longer always a spritesheet.
-- --------------------------------------------------------------------------
--
-- Defaulted to 'sheet' so the 20 existing rows keep their meaning and every
-- current caller keeps working without sending a new field.
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'sheet';

-- Soft delete, so the gallery can hide a sheet without destroying the job
-- record something2 may still poll for.
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS deleted BOOLEAN NOT NULL DEFAULT false;
CREATE INDEX IF NOT EXISTS jobs_kind_idx ON jobs (kind, updated_at DESC);

-- --------------------------------------------------------------------------
-- api_keys - per-client credentials.
-- --------------------------------------------------------------------------
--
-- Replaces the single shared API_TOKEN, which was also a no-op when unset:
-- forgetting to set an environment variable silently disabled authentication
-- entirely, which is the worst possible default.
--
-- The token itself is never stored. `key_hash` is a SHA-256 of it; `key_prefix`
-- is the first 8 characters, kept in clear ONLY so the UI can show which key a
-- row refers to without being able to reconstruct it.
CREATE TABLE IF NOT EXISTS api_keys (
    id           UUID PRIMARY KEY,
    name         TEXT NOT NULL,
    key_hash     TEXT NOT NULL UNIQUE,
    key_prefix   TEXT NOT NULL,

    -- What this key may do: any of 'generate', 'read', 'admin'. An array
    -- rather than a single role because something2 needs generate+read but
    -- must never be able to mint further keys.
    scopes       TEXT[] NOT NULL DEFAULT ARRAY['generate', 'read'],

    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at TIMESTAMPTZ,

    -- Revocation is a timestamp, not a delete: a revoked key must stay
    -- resolvable so its past jobs still attribute to something.
    revoked_at   TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS api_keys_hash_idx ON api_keys (key_hash) WHERE revoked_at IS NULL;

-- --------------------------------------------------------------------------
-- reference_assets - the examples the user uploads.
-- --------------------------------------------------------------------------
--
-- Deliberately separate from sprite_images. That table is "things this system
-- generated"; this one is "things the target art looks like". Mixing them
-- would make the gallery lie and would risk training a model on its own
-- output, which is how a style collapses.
CREATE TABLE IF NOT EXISTS reference_assets (
    id          UUID PRIMARY KEY,

    -- core | sprite | tile, matching the three UI tabs. Kind decides which
    -- measurements apply: a tile yields a projection angle, a sprite yields a
    -- cell grid, a core yields an isolation verdict.
    kind        TEXT NOT NULL,

    file_path   TEXT NOT NULL,
    label       TEXT,

    -- Everything measured on upload, shape depending on kind. JSONB because
    -- the measurement set is still being discovered and a column per metric
    -- would mean a migration per insight.
    metrics     JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Whether this example is fit to learn from. NULL means not yet measured.
    usable      BOOLEAN,
    why         TEXT,

    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted     BOOLEAN NOT NULL DEFAULT false
);
CREATE INDEX IF NOT EXISTS reference_assets_kind_idx
    ON reference_assets (kind, created_at DESC) WHERE deleted = false;

-- --------------------------------------------------------------------------
-- style_profiles - what was concluded from the references.
-- --------------------------------------------------------------------------
--
-- The bridge between "examples" and "generation". A profile is the set of hard
-- constraints the conveyor applies: which colours exist, what a cell measures,
-- what camera angle the world uses. Those are MEASURED, and work from three
-- examples.
--
-- `lora_path` and `trigger_token` are the trained half, filled in later and
-- optional: a profile without a LoRA is still useful, which is the point of
-- doing measurement before training.
CREATE TABLE IF NOT EXISTS style_profiles (
    id            UUID PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,

    palette       JSONB,     -- ["#1a1a2e", ...] locked palette
    cell_w        INTEGER,
    cell_h        INTEGER,
    colors        INTEGER,
    outline       JSONB,     -- {"width": 1, "color": "#000000"} or null

    -- The camera. `projection_ratio` is a tile's width:height (2.0 for classic
    -- 2:1 dimetric); `elevation` is the Qwen vocabulary term derived from it.
    projection_ratio REAL,
    elevation     TEXT,

    derived_from  UUID[],    -- reference_assets ids this was measured from
    lora_path     TEXT,
    trigger_token TEXT,

    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- --------------------------------------------------------------------------
-- training_runs - one row per LoRA training attempt.
-- --------------------------------------------------------------------------
--
-- Kept beside `jobs` rather than inside it because a run has metrics a
-- generation job has no concept of (loss, steps, dataset size) and because a
-- failed run is a record worth keeping: on a 12 GB card, WHICH settings OOMed
-- is the most valuable thing the run produced.
CREATE TABLE IF NOT EXISTS training_runs (
    id           UUID PRIMARY KEY,
    job_id       UUID REFERENCES jobs (id) ON DELETE SET NULL,
    profile_id   UUID REFERENCES style_profiles (id) ON DELETE SET NULL,

    base_model   TEXT NOT NULL,
    config       JSONB NOT NULL DEFAULT '{}'::jsonb,
    dataset_size INTEGER,

    status       TEXT NOT NULL DEFAULT 'queued',
    steps_done   INTEGER NOT NULL DEFAULT 0,
    steps_total  INTEGER,
    loss         REAL,

    output_path  TEXT,
    error        TEXT,

    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at   TIMESTAMPTZ,
    finished_at  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS training_runs_status_idx ON training_runs (status, created_at DESC);

-- --------------------------------------------------------------------------
-- assets_v - ONE list of everything this system has produced.
-- --------------------------------------------------------------------------
--
-- The gallery read only sprite_images, which on 2026-08-25 held 2 undeleted
-- rows while `jobs` held 13 finished sheets. Thirteen finished spritesheets had
-- never been visible in the UI - not because of a rendering bug, but because
-- nothing ever asked for them.
--
-- A view rather than a table so there is no synchronisation to get wrong: the
-- two producers keep writing where they already write.
CREATE OR REPLACE VIEW assets_v AS
    SELECT
        'image'::text          AS source,
        si.id::text            AS id,
        si.file_path           AS file_path,
        si.prompt              AS title,
        COALESCE(si.image_type, 'core') AS kind,
        si.timestamp           AS created_at,
        NULL::uuid             AS job_id,
        si.llm_name            AS model
    FROM sprite_images si
    WHERE si.deleted = false AND si.file_path IS NOT NULL
UNION ALL
    SELECT
        'job'::text,
        j.id::text,
        j.sheet_path,
        COALESCE(NULLIF(j.spec ->> 'prompt', ''), 'sheet ' || left(j.id::text, 8)),
        COALESCE(j.kind, 'sheet'),
        COALESCE(j.finished_at, j.updated_at),
        j.id,
        NULL::text
    FROM jobs j
    WHERE j.status = 'done' AND j.sheet_path IS NOT NULL AND j.deleted = false;

-- Reuse the trigger function 012 already defined; it is generic.
DROP TRIGGER IF EXISTS style_profiles_touch ON style_profiles;
CREATE TRIGGER style_profiles_touch BEFORE UPDATE ON style_profiles
    FOR EACH ROW EXECUTE FUNCTION jobs_touch_updated_at();
