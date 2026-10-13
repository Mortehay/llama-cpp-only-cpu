-- --------------------------------------------------------------------------
-- The download queue also deletes. Renamed to match.
-- --------------------------------------------------------------------------
--
-- `model_downloads` was the right name for one day. Freeing disk is the
-- natural counterpart to fetching - a 103 GB full tier on a host with ~57 GB
-- free means "remove the one I am not using" is not an optional feature - and
-- a table called model_downloads holding delete rows is the kind of name that
-- makes the next person read the code to find out what it does.
--
-- Renamed rather than added-to: a second table would need its own queue, its
-- own consumer and its own lock discipline, and the whole point of routing
-- deletes through the same daemon is that a delete CANNOT run while a fetch
-- holds the cache lock. One queue, one consumer, one lock.
--
-- Safe to rename: 020 shipped the same day and the table holds only the rows
-- from testing the lane.
ALTER TABLE IF EXISTS model_downloads RENAME TO model_ops;

-- 'download' | 'delete'. Defaulted so the existing rows - all downloads - stay
-- correct without a backfill.
ALTER TABLE model_ops ADD COLUMN IF NOT EXISTS op TEXT NOT NULL DEFAULT 'download';

-- How much disk the operation actually freed, measured. NOT the declared size:
-- an HF repo's blobs live in a shared, xet-backed store at the root of
-- MODELS_DIR (17 GB of it on this host), so removing a repo can reclaim less
-- than its manifest size when another entry still references the same blobs.
-- Reporting the declared number would overstate what the user got back.
ALTER TABLE model_ops ADD COLUMN IF NOT EXISTS bytes_freed BIGINT;

-- The indexes from 020 follow the rename automatically, but their names still
-- say "downloads" and the uniqueness rule has to change: one entry may have a
-- queued delete AND have had downloads before. Keep "one active op per entry"
-- - a delete racing a download on the same model is the deadlock this whole
-- lane exists to prevent.
ALTER INDEX IF EXISTS model_downloads_queued_idx RENAME TO model_ops_queued_idx;
ALTER INDEX IF EXISTS model_downloads_recent_idx RENAME TO model_ops_recent_idx;
ALTER INDEX IF EXISTS model_downloads_active_entry_idx
    RENAME TO model_ops_active_entry_idx;
