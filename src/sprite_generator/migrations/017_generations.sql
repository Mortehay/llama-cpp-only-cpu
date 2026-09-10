-- --------------------------------------------------------------------------
-- generations - the request ledger for the API surface.
-- --------------------------------------------------------------------------
--
-- WHY THIS EXISTS. On 2026-09-10 `images/` held 731 `raw_*.png` files and
-- `sprite_images` held ZERO rows pointing at any of them. Every image
-- something2 has ever pulled through /sdapi/v1/txt2img was written to disk by
-- `generate_raw_task` and then forgotten: invisible in the gallery (assets_v
-- reads sprite_images and jobs, neither of which knew), unaddressable, and
-- un-cacheable, so an identical repeat request re-spent the GPU.
--
-- Named tiles and maps were already exempt - they go through `jobs` with
-- kind='tile' and spec->>'name', which is why the tile facade can cache-read.
-- Entities had no equivalent. This table is that equivalent, plus the
-- provenance the jobs table never carried.
--
-- IT IS A LEDGER, NOT AN ARTEFACT TABLE. A row records one REQUEST: who asked,
-- what they asked for, what happened. When the request produced a file of its
-- own, `file_path` is set and `job_id` is NULL - that row owns the image and
-- assets_v surfaces it. When the request was served by (or built) a job - the
-- tile and map facades - `job_id` points at it and `file_path` stays NULL,
-- because `jobs` already owns that image and counting it twice would show the
-- same tile twice in the gallery.
CREATE TABLE IF NOT EXISTS generations (
    id             UUID PRIMARY KEY,

    -- What was produced: entity | tile | map. NOT the model family and not the
    -- reference kinds - this says which facade served the call.
    kind           TEXT NOT NULL DEFAULT 'entity',

    -- The addressable handle, or NULL for a one-off. Deliberately not unique,
    -- exactly as tile names are not: re-rolling an entity makes a second row
    -- with the same name and the newest finished one wins, so a better result
    -- takes effect without anyone editing configuration on the calling machine.
    name           TEXT,

    -- running -> done | failed. `cached` is NOT a status: a cache read is a
    -- finished request that happens to have cost no GPU, and `served_from`
    -- records that without splitting the terminal states.
    status         TEXT NOT NULL DEFAULT 'running',

    -- 'generated' | 'cache'. A caller measuring throughput off duration_ms
    -- needs to know which of the two it is looking at.
    served_from    TEXT NOT NULL DEFAULT 'generated',

    route          TEXT NOT NULL DEFAULT '/sdapi/v1/txt2img',
    prompt         TEXT NOT NULL DEFAULT '',
    negative_prompt TEXT NOT NULL DEFAULT '',
    model          TEXT,
    seed           BIGINT,

    -- Everything else the request asked for: width, height, steps, cfg, frames,
    -- cutout, lora_scale. JSONB rather than columns because the A1111 template
    -- system is the thing that decides what arrives, and it is not ours.
    params         JSONB NOT NULL DEFAULT '{}'::jsonb,

    file_path      TEXT,
    job_id         UUID,
    error          TEXT,
    duration_ms    INTEGER,

    -- WHO ASKED. `principal_id` is the api_keys row, and it is the only one of
    -- these three that discriminates reliably - see client_addr.
    principal_id   UUID,
    principal_name TEXT NOT NULL DEFAULT 'unknown',

    -- THIS IS THE GATEWAY, NOT THE CLIENT, and the column is kept anyway so
    -- the UI can say so out loud rather than leaving a blank that invites
    -- someone to "fix" it. Measured 2026-09-10 in `docker logs
    -- sprite_generator`: every external request arrives as 172.18.0.1, the
    -- docker bridge gateway. There is no reverse proxy in the compose file,
    -- and LAN traffic additionally crosses `netsh interface portproxy`, which
    -- SNATs a second time. So every machine on the Wi-Fi collapses to one
    -- address. Per-consumer API KEYS are the identity that actually separates
    -- callers on this topology; this is corroboration at best.
    client_addr    TEXT,
    forwarded_for  TEXT,
    user_agent     TEXT,

    celery_task_id TEXT,
    deleted        BOOLEAN NOT NULL DEFAULT false,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS generations_created_at_idx ON generations (created_at DESC);
CREATE INDEX IF NOT EXISTS generations_status_idx ON generations (status, created_at DESC);
-- The cache-read access path: newest finished row with this name and kind.
CREATE INDEX IF NOT EXISTS generations_name_idx
    ON generations (kind, lower(name), created_at DESC)
    WHERE name IS NOT NULL AND status = 'done';
CREATE INDEX IF NOT EXISTS generations_job_idx ON generations (job_id)
    WHERE job_id IS NOT NULL;

-- --------------------------------------------------------------------------
-- assets_v - third arm.
-- --------------------------------------------------------------------------
--
-- Same reasoning as when the `jobs` arm was added in 013: the producer keeps
-- writing where it writes and the view does the joining. `job_id IS NULL` is
-- what stops a facade-built tile appearing twice - once as its job row and
-- once as the ledger row that requested it.
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
      AND g.job_id IS NULL AND g.deleted = false;

-- --------------------------------------------------------------------------
-- activity_v - one feed of everything that has asked this machine to generate.
-- --------------------------------------------------------------------------
--
-- Three producers, and the Activity tab needs all three or "pending" lies: an
-- API call, a queued job (sheet, tile, map, training) and a browser-initiated
-- single-image task all compete for the same one-at-a-time GPU worker. Showing
-- only the ledger would report an idle machine while a two-hour sheet holds the
-- card.
--
-- `requested_by` is deliberately a sentence, not an id. Only the ledger arm can
-- name a principal; the other two predate the ledger and were never attributed,
-- so they say so rather than borrowing a plausible-looking name.
CREATE OR REPLACE VIEW activity_v AS
    SELECT
        'api'::text            AS source,
        g.id::text             AS id,
        g.kind                 AS kind,
        g.name                 AS name,
        g.status               AS status,
        g.served_from          AS served_from,
        COALESCE(NULLIF(g.prompt, ''), '(no prompt)') AS title,
        g.model                AS model,
        g.file_path            AS file_path,
        g.job_id               AS job_id,
        g.error                AS error,
        g.duration_ms          AS duration_ms,
        g.principal_name       AS requested_by,
        g.client_addr          AS client_addr,
        g.created_at           AS created_at,
        g.finished_at          AS finished_at
    FROM generations g
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
        si.timestamp,
        NULL::timestamptz
    FROM sprite_images si
    WHERE si.deleted = false;
