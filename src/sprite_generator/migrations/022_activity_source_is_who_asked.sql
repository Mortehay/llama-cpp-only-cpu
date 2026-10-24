-- --------------------------------------------------------------------------
-- activity_v - `source` becomes WHO ASKED; `lane` carries what ran it.
-- --------------------------------------------------------------------------
--
-- See .ai/decisions/0017. In 017 and 018 `source` was really "which arm of the
-- UNION this row came from", i.e. WHICH TABLE OWNS THE ARTEFACT - not a
-- property of the caller at all. Three defects followed from that directly:
--
-- 1. THIS BROWSER WAS FILED UNDER "API". It holds an API key, so it writes a
--    ledger row, so it matched the first arm: 3101 rows under principal
--    `browser - 2026-08-26 10:24`.
--
-- 2. THE "ui" ARM WAS A FROZEN ARCHIVE. It matches `sprite_images` rows with
--    NO ledger row, which since 018 means pre-018 history and nothing else:
--    264 rows, newest 2026-09-04, a count that can never grow.
--
-- 3. WORK SUBMITTED TO `POST /api/jobs` HAD NO CALLER. That route wrote no
--    ledger row, so it reached the tab only through the job arm's
--    "unattributed" sentence - and the browser posts to the same route, so one
--    bucket held two askers and could distinguish neither.
--
-- So the two questions get two columns:
--
--     source   api | internal | unknown     who asked
--     lane     sync | job                   whether a `jobs` row owns the work
--
-- `ui` IS GONE AS A SOURCE. "Which browser" is not answerable here - the
-- browser authenticates with an ordinary API key and `api_keys` carries no type
-- column - and `requested_by` already names the key, which is the finest
-- identity this service has. The UI filters on that instead.
--
-- DROPPED, NOT REPLACED, for the reason 018 records: `CREATE OR REPLACE VIEW`
-- matches columns by position and reports an inserted column as "cannot change
-- name of view column created_at to forwarded_for", which reads like a rename
-- and is not one. `generations.py` is the only consumer (checked 2026-10-23),
-- and the whole migration is one transaction, so there is no window where the
-- view is missing.
DROP VIEW IF EXISTS activity_v;
CREATE VIEW activity_v AS
    SELECT
        -- WHO ASKED. The route test comes FIRST: an internal call carries a
        -- synthesized caller with no `principal_id`, so testing the id first
        -- would file every bench and every `internal:*` brain call as
        -- "unknown". The legacy shared token is a real credential with no row
        -- in `api_keys`, so it is named explicitly rather than left to fall
        -- through; `auth.require` returns it with `id` NULL.
        CASE
            WHEN g.route LIKE 'internal:%' OR g.route LIKE 'bench%' THEN 'internal'
            WHEN g.principal_id IS NOT NULL THEN 'api'
            WHEN g.principal_name = 'legacy SPRITE_API_TOKEN' THEN 'api'
            ELSE 'unknown'
        END                    AS source,
        -- WHAT RAN IT. `job` means a `jobs` row owns the work, with its
        -- 202/poll contract - the "Job" sense in .ai/domain.md. `sync` means
        -- no `jobs` row; it does NOT promise the HTTP call blocked, because
        -- the three `image_id` routes queue a Celery task and return.
        CASE WHEN g.job_id IS NOT NULL THEN 'job' ELSE 'sync' END AS lane,
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
    -- A `jobs` row with no ledger row. Since this migration that means one
    -- queued before `create_job` recorded a principal, so it is honestly
    -- `unknown` - not "api", which is the guess that started all of this.
    SELECT
        'unknown'::text,
        'job'::text,
        j.id::text,
        -- `jobs.kind` is NULL for sheets AND for redraws; `redraw_frames` puts
        -- its kind in the spec JSON. Reading the column alone displayed all 5
        -- existing redraws as ordinary sheets, so a 2-frame repair and a
        -- 336-cell build looked identical.
        COALESCE(j.kind, NULLIF(j.spec ->> 'kind', ''), 'sheet'),
        NULLIF(j.spec ->> 'name', ''),
        j.status,
        'generated'::text,
        COALESCE(NULLIF(j.spec ->> 'prompt', ''), 'job ' || left(j.id::text, 8)),
        NULL::text,
        j.sheet_path,
        j.id,
        j.error,
        NULL::integer,
        'unattributed (queued before the ledger covered this route)'::text,
        NULL::text,
        NULL::text,
        j.created_at,
        j.finished_at
    FROM jobs j
    WHERE j.deleted = false
      -- A ledgered job is already the row above, with the key that asked for
      -- it. Without this every something2 tile request shows twice.
      AND NOT EXISTS (SELECT 1 FROM generations g2
                      WHERE g2.job_id = j.id AND g2.deleted = false)
UNION ALL
    -- A `sprite_images` row with no ledger row: pre-018 history only, frozen
    -- at 264 rows. This arm used to be called `ui`, which was the claim that
    -- it meant "this browser". It never did - it meant "we have no ledger row
    -- for this", which is `unknown`.
    SELECT
        'unknown'::text,
        'sync'::text,
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
        'unattributed (generated before the ledger existed)'::text,
        NULL::text,
        NULL::text,
        si.timestamp,
        NULL::timestamptz
    FROM sprite_images si
    WHERE si.deleted = false
      AND NOT EXISTS (SELECT 1 FROM generations g3
                      WHERE g3.image_id = si.id AND g3.deleted = false);
