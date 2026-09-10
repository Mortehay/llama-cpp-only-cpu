-- --------------------------------------------------------------------------
-- Name the recovered rows honestly.
-- --------------------------------------------------------------------------
--
-- The 731 rows `scripts/backfill-generations.py` created have no prompt: it
-- was never written down for those files and is not recoverable. Checked again
-- on 2026-09-10 before accepting that - the worker's own logs reach back only
-- as far as its last restart and never carried prompt text at all.
--
-- The consequence was cosmetic and then not: with an empty prompt the views
-- fell through to `'api ' || left(id, 8)`, so a third of the gallery was
-- titled with meaningless hex that LOOKS like an identifier worth reading.
-- Worse, there was no way to filter them out - "show me the images that
-- actually know what they are" had no query.
--
-- So a backfilled row is titled after the one fact about it that is true: its
-- filename. `recovered · raw_002d4a9f.png` sorts together, reads as what it
-- is, and makes `?q=recovered` the filter that was missing.
--
-- Column names, types and order are unchanged in both views, which is what
-- lets these be REPLACE rather than DROP - see the note in 018 about why that
-- distinction bites.
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
        COALESCE(
            NULLIF(g.name, ''),
            NULLIF(g.prompt, ''),
            CASE WHEN g.params ->> 'backfill' = 'true'
                 THEN 'recovered · ' || regexp_replace(g.file_path, '^.*/', '')
                 ELSE 'api ' || left(g.id::text, 8) END),
        g.kind,
        COALESCE(g.finished_at, g.created_at),
        NULL::uuid,
        g.model
    FROM generations g
    WHERE g.status = 'done' AND g.file_path IS NOT NULL
      AND g.job_id IS NULL AND g.image_id IS NULL AND g.deleted = false;

CREATE OR REPLACE VIEW activity_v AS
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
        COALESCE(
            NULLIF(g.prompt, ''),
            CASE WHEN g.params ->> 'backfill' = 'true'
                 THEN 'recovered · ' || regexp_replace(
                          COALESCE(g.file_path, ''), '^.*/', '')
                 ELSE '(no prompt)' END) AS title,
        COALESCE(g.model, si.llm_name) AS model,
        COALESCE(g.file_path, si.file_path, j.sheet_path) AS file_path,
        g.job_id               AS job_id,
        COALESCE(g.error, NULLIF(si.error, ''), j.error) AS error,
        COALESCE(g.duration_ms, si.duration_ms::integer) AS duration_ms,
        g.principal_name       AS requested_by,
        g.client_addr          AS client_addr,
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
      AND NOT EXISTS (SELECT 1 FROM generations g3
                      WHERE g3.image_id = si.id AND g3.deleted = false);
