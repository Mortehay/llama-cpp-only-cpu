# 0017 - Activity `source` means who asked, not which table owns the artefact

Date: 2026-10-23
Status: accepted, NOT YET IMPLEMENTED
Related: [0013](0013-gated-brain.md) (`/api/text` and the brain's ledger rows),
migration [017](../../src/sprite_generator/migrations/017_generations.sql)
(`generations`, `activity_v`), migration
[018](../../src/sprite_generator/migrations/018_generations_image_link.sql)
(`image_id`, the read-side status resolution)

## What changed

The Activity tab's `source` column becomes **the identity of the caller**. A
second, derived axis - `lane` - carries what 017 and 018 were actually
encoding.

    source   api | internal | unknown     who asked
    lane     sync | job                   whether a `jobs` row owns the work

`ui` is **dropped as a source**. "Who asked" is answered by `requested_by` (the
API key name), which is already populated on 99% of rows, and the source filter
is replaced by a principal dropdown built the same way the model dropdown
already is.

## Why

`source` never meant what its three values suggest. In 017 it is *which arm of
the UNION a row came from*, which is *which table owns the artefact* - and that
is not a property of the caller at all. Three defects follow from it directly,
all three measured against the live database on 2026-10-23.

**1. This browser is filed under "API (something2 and scripts)".** The browser
holds an API key, so its generations write a ledger row, so they match the
first arm. Row counts by principal:

| principal | rows | routes |
|---|---|---|
| `browser - 2026-08-26 10:24` | 3101 | 3 |
| `something2-audio` | 2817 | 2 |
| `something2-text` | 2072 | 2 |
| `unknown (backfilled)` | 731 | 1 |
| `DESKTOP-DF4MGDC` | 77 | 4 |
| `bench: ...`, `... (internal)`, `claude ...` | ~140 | - |

**2. The "This browser" filter is a frozen archive.** The `ui` arm matches only
`sprite_images` rows with NO ledger row, which since migration 018 means
pre-018 history and nothing else. 264 rows, newest **2026-09-04**. That number
can never grow. A filter that cannot ever gain a row is not a filter.

**3. Work submitted over the API to `POST /api/jobs` has no caller at all.**
`jobs.py` contains zero references to `generations` - no `Request` parameter, no
principal recorded, though `_require_auth` already *returns* the principal and
the route discards it. Such a job reaches the tab only through the second arm,
labelled *"unattributed (queued before the ledger existed, or from the UI)"*.
61 sheet rows, newest 2026-10-23 02:02. And the browser posts to the same route
(`frontend/src/tabs/SheetGenerator.tsx`), so that one bucket holds two
different askers and can distinguish neither.

That third one is the reported symptom, and it is the only one of the three
that is a missing write. **The other API routes are all instrumented and
working** - this was checked rather than assumed, because the obvious reading
of the complaint is that the ledger had stopped recording:

| route | rows | newest |
|---|---|---|
| `/sdapi/v1/txt2img` | 3869 | 2026-10-23 14:53 |
| `/api/audio` + `/api/audio/sfx` | 2854 | 2026-10-11 |
| `/api/text` | 2071 | 2026-10-11 |
| `/api/generate_core` | 29 | 2026-10-22 |
| `/api/edit` | 18 | 2026-10-21 |

018's read-side status resolution is also working: the 48 rows sitting at
`running` on the three `image_id` routes all resolve to terminal through the
join, and the pending list measured **zero** rows.

## What this costs

**"Show me the long-running queued work" stops being a source filter.** That is
a real question and it is what the three-arm design was for. It moves to
`lane`, derived in the view - `job` when `job_id IS NOT NULL`, else `sync` -
which matches [domain.md](../domain.md)'s definition of **Job** exactly: a row
in `jobs` plus its 202/poll contract. No stored column.

**Rows that cannot be attributed get `source = 'unknown'`, and keep saying so.**
The 731 backfilled rows, the 264 pre-018 `sprite_images` rows and the 81 `jobs`
rows predate attribution and can never acquire it. The existing sentence does
two jobs - *we do not know who* and *this predates the ledger* - and both stay
true for exactly those rows.

## Alternatives rejected

**Add `api_keys.kind` (`consumer` | `browser` | `internal`).** This would keep
`ui` as a first-class source. Rejected: it spends a migration to re-encode a
fact the key name already carries, and `rotate_key` copies only name and
scopes, so a missed line there silently reclassifies the browser as a consumer
on the next rotation. A 3-value enum also cannot answer "which of my five
consumers", which is the question that actually gets asked - something2 already
holds three separate keys.

**Attribute inside the `job` arm instead** - add `principal_name` /
`client_addr` columns to `jobs` and fill them in `create_job`. Smaller change,
keeps the taxonomy. Rejected after the owner confirmed `source` is meant to be
who asked: this preserves the conflation and gives `requested_by` two
independent sources of truth.

**Keep `source`, add `asked_by_kind` alongside.** Two overlapping taxonomies on
one tab is how this state was reached.

## Implementation notes

- `create_job` and `redraw_frames` write `generations.begin(job_id=...)`.
  Order: `INSERT jobs` -> `begin` -> `.delay()`. After the insert so `job_id`
  never dangles; before `delay` so the row exists while the work is queued. A
  `begin` that returns `None` falls through to the job arm - degraded, not
  broken, which is the trade the whole ledger makes.
- A redraw already creates its **own** `jobs` row (child id, `parent_job` in
  the spec), so there is no double-attachment to the parent and no status
  collision. Checked, because ledgering it against the parent would have had
  both.
- **`jobs.kind` is NULL for sheets and for redraws.** `tile`, `map` and
  `map_props` set it; `create_job` does not, and `redraw_frames` puts
  `kind: "redraw"` inside the spec JSON instead. `activity_v` reads the
  column, so all 5 existing redraw jobs (4 done, 1 failed) are displayed as
  ordinary sheets - a 2-frame repair and a 336-cell build look identical. Set
  the column at insert for both.
- A redraw's ledger row needs a synthesized title: `RedrawSpec` carries no
  prompt, so the view's `COALESCE(NULLIF(g.prompt,''), '(no prompt)')` would
  title every one of them "(no prompt)". Use the frame count and parent id.
- **Rebuild, do not replace, the view.** 018 records why: `CREATE OR REPLACE
  VIEW` matches columns by position and reports an inserted column as
  *"cannot change name of view column created_at to forwarded_for"*, which
  reads like a rename and is not one. `DROP` then `CREATE`, in one
  transaction.
- The model dropdown is shared between brains and image checkpoints and must
  be grouped, not relabelled: `qwen3.6-35b-a3b` (2060 rows) is its single
  largest entry while the column's tooltip reads *"the image model that drew
  it (llm_name) - not a language model"*. Group using
  `models_status.TEXT_KINDS` / `IMAGE_KINDS` - but note `IMAGE_KINDS` omits
  `edit`, `prop`, `sheet` and `map_props`, which `activity_v` all produce, so
  four kinds fall into neither group as those tuples stand.

## Not verified

Nothing here is implemented. The row counts and the empty pending list were
read from `activity_v` and `generations` on 2026-10-23 with three read-only
`SELECT`s; no migration has been written and no UI change has been made. The
claim that `lane` answers the queued-work question has not been tested against
anyone actually using the tab.
