-- --------------------------------------------------------------------------
-- generations.image_id - the third way a request can own its artefact.
-- --------------------------------------------------------------------------
--
-- 017 gave a ledger row two possible relationships to a picture: it owns the
-- file (`file_path`), or a job owns it (`job_id`). That covered the API facade
-- and nothing else, which left the browser's own generation endpoints -
-- /api/generate_core, /api/edit, /api/generate_sheet - outside the ledger
-- entirely. They appeared on the Activity tab through the `sprite_images` arm,
-- labelled "browser (local UI)" no matter which key made the call, with no
-- address and no principal.
--
-- This is the third case: the request is recorded here, `sprite_images` owns
-- the row and the file. The invariant from 017 is unchanged and now reads:
--
--     file_path set              this row owns the PNG
--     job_id set                 a `jobs` row owns it
--     image_id set               a `sprite_images` row owns it
--
-- At most one of the three, and `assets_v` shows only the first - otherwise the
-- same picture is in the gallery twice.
ALTER TABLE generations ADD COLUMN IF NOT EXISTS image_id INTEGER;
CREATE INDEX IF NOT EXISTS generations_image_idx ON generations (image_id)
    WHERE image_id IS NOT NULL;

-- --------------------------------------------------------------------------
-- assets_v - exclude the rows whose artefact belongs to sprite_images.
-- --------------------------------------------------------------------------
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
    WHERE j.status = 'done' AND j.sheet_path IS NOT NULL AND j.deleted = false
UNION ALL
    SELECT
        'generation'::text,
        g.id::text,
        g.file_path,
        COALESCE(NULLIF(g.name, ''), NULLIF(g.prompt, ''),
                 'api ' || left(g.id::text, 8)),
        g.kind,
        COALESCE(g.finished_at, g.created_at),
        NULL::uuid,
        g.model
    FROM generations g
    WHERE g.status = 'done' AND g.file_path IS NOT NULL
      AND g.job_id IS NULL AND g.image_id IS NULL AND g.deleted = false;

-- --------------------------------------------------------------------------
-- activity_v - resolve each ledger row against whatever owns its artefact.
-- --------------------------------------------------------------------------
--
-- Two problems with 017's version, both visible on the tab:
--
-- 1. NO THUMBNAIL for a row that does not own its file. Every tile and map the
--    facade served showed an empty image cell, because `file_path` is NULL on
--    exactly those rows by design.
--
-- 2. A LEDGER ROW COULD BE STUCK AT 'running' FOREVER. The worker writes to
--    `sprite_images` and `jobs`; it does not know this table exists, and giving
--    it that knowledge would mean editing `tasks.py` and restarting the GPU
--    worker for a bookkeeping change. So the READ side resolves it instead: a
--    row still marked running takes the terminal state of the artefact it
--    points at. That also rescues the case in `.ai/project-context.md` where
--    the API answered 200 for six hours while nothing generated - the request
--    row would otherwise sit at 'running' with no way to learn otherwise.
--
-- The joins are LEFT and the COALESCE order is fixed: the ledger's own value
-- wins where it has one, because a facade call that blocked on its own result
-- knows more than the artefact row does.
--
-- DROPPED, NOT REPLACED. `CREATE OR REPLACE VIEW` may append columns but may
-- not insert one in the middle - it matches by position and reports the
-- attempt as "cannot change name of view column created_at to forwarded_for",
-- which reads like a rename and is not one. Nothing else selects from this
-- view, so dropping it costs nothing; the whole migration is one transaction,
-- so there is no window where it is missing.
DROP VIEW IF EXISTS activity_v;
CREATE VIEW activity_v AS
    SELECT
        'api'::text            AS source,
        g.id::text             AS id,
        g.kind                 AS kind,
        g.name                 AS name,
        CASE
            WHEN g.status <> 'running' THEN g.status
            WHEN si.id IS NOT NULL THEN
                CASE
                    WHEN COALESCE(si.error, '') <> '' THEN 'failed'
                    WHEN si.file_path IS NOT NULL     THEN 'done'
                    ELSE 'running'
                END
            WHEN j.id IS NOT NULL THEN j.status
            ELSE g.status
        END                    AS status,
        g.served_from          AS served_from,
        COALESCE(NULLIF(g.prompt, ''), '(no prompt)') AS title,
        COALESCE(g.model, si.llm_name) AS model,
        COALESCE(g.file_path, si.file_path, j.sheet_path) AS file_path,
        g.job_id               AS job_id,
        COALESCE(g.error, NULLIF(si.error, ''), j.error) AS error,
        COALESCE(g.duration_ms, si.duration_ms::integer) AS duration_ms,
        g.principal_name       AS requested_by,
        g.client_addr          AS client_addr,
        -- Caller-supplied and UNVERIFIED. There is no proxy in front of this
        -- service, so nothing sets this header except the caller itself; it is
        -- shown as a hint, never as proof of origin.
        g.forwarded_for        AS forwarded_for,
        g.created_at           AS created_at,
        g.finished_at          AS finished_at
    FROM generations g
    LEFT JOIN sprite_images si ON si.id = g.image_id
    LEFT JOIN jobs j           ON j.id = g.job_id
    WHERE g.deleted = false
UNION ALL
    SELECT
        'job'::text,
        j.id::text,
        COALESCE(j.kind, 'sheet'),
        NULLIF(j.spec ->> 'name', ''),
        j.status,
        'generated'::text,
        COALESCE(NULLIF(j.spec ->> 'prompt', ''), 'job ' || left(j.id::text, 8)),
        NULL::text,
        j.sheet_path,
        j.id,
        j.error,
        NULL::integer,
        'unattributed (queued before the ledger existed, or from the UI)'::text,
        NULL::text,
        NULL::text,
        j.created_at,
        j.finished_at
    FROM jobs j
    WHERE j.deleted = false
      -- A facade-built tile is already the `api` row above. Without this the
      -- Activity tab shows every something2 tile request twice.
      AND NOT EXISTS (SELECT 1 FROM generations g2
                      WHERE g2.job_id = j.id AND g2.deleted = false)
UNION ALL
    SELECT
        'ui'::text,
        si.id::text,
        COALESCE(si.image_type, 'core'),
        NULL::text,
        CASE
            WHEN COALESCE(si.error, '') <> '' THEN 'failed'
            WHEN si.file_path IS NOT NULL     THEN 'done'
            ELSE 'running'
        END,
        'generated'::text,
        COALESCE(NULLIF(si.prompt, ''), 'task ' || si.id::text),
        si.llm_name,
        si.file_path,
        NULL::uuid,
        NULLIF(si.error, ''),
        si.duration_ms::integer,
        'browser (local UI)'::text,
        NULL::text,
        NULL::text,
        si.timestamp,
        NULL::timestamptz
    FROM sprite_images si
    WHERE si.deleted = false
      -- An instrumented UI generation is already the `api` row above, with the
      -- key that made it. Rows from before this migration have no ledger row
      -- and keep the unattributed label, which is the truth about them.
      AND NOT EXISTS (SELECT 1 FROM generations g3
                      WHERE g3.image_id = si.id AND g3.deleted = false);
