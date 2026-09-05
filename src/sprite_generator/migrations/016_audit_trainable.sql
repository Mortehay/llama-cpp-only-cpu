-- 016: stop two independent judges writing to one `trainable` column.
--
-- This is 014's problem again, one level down. 014 split "can I measure this?"
-- from "can I train on this?" and that split was right. What it did not
-- anticipate is that TWO different things would end up writing `trainable`:
--
--   * `measure.judge_trainable` - deliberately permissive. Rejects only blank,
--     tiny, or extreme-strip images, because a JPEG reference board with a grey
--     backdrop still teaches palette and shading to a STYLE adapter.
--   * `scripts/audit-character-refs.py --apply` - deliberately strict. Rejects
--     contact sheets, baked transparency checkers, non-pixel-art and
--     near-duplicates, because those are what taught both failed adapters to
--     draw a lattice of framed cells (see decisions/0009).
--
-- Both are correct. They answer different questions. Sharing one column means
-- whichever ran last wins, and the loser is silent.
--
-- MEASURED 2026-09-04. One call to `POST /api/references/remeasure-all`
-- recomputed `trainable` from the permissive gate and un-rejected **131 core
-- and 84 sprite** references that the audit had marked false. Nothing warned;
-- `usable` was untouched, the UI looked identical, and the next training run
-- would have read 215 images the audit had already judged unable to teach.
-- `make audit-refs-apply` put them back (237 marked not trainable), but only
-- because someone happened to check the counts.
--
-- That is the same failure as the one recorded in commit d05bbe5 - "a missing
-- verdicts file silently un-rejected three bad references" - at seventy times
-- the scale. Fixing it by making remeasure "not widen" was rejected: `--apply`
-- only ever writes false, so nothing would ever be able to re-enable a
-- reference again, and a file rejected for being under the 160px floor would
-- stay rejected even if the floor changed. A ratchet is not a fix.
--
-- So each judge gets its own column and training reads the conjunction. Both
-- stay independently recomputable, and neither can silently overwrite the
-- other.
ALTER TABLE reference_assets
    ADD COLUMN IF NOT EXISTS audit_trainable BOOLEAN,
    ADD COLUMN IF NOT EXISTS audit_trainable_why TEXT;

-- Deliberately left NULL rather than backfilled to true.
--
-- NULL means "this judge has not looked", which is the truth for `tile` and
-- `map` (the character audit covers only sprite and core) and for anything
-- uploaded since the last audit run. Training therefore tests
-- `audit_trainable IS NOT FALSE`, so an unaudited reference is not blocked by a
-- verdict that was never rendered. Backfilling true would have been a lie that
-- reads identically to a considered pass.
--
-- Existing `trainable = false` rows keep that value until the next remeasure
-- recomputes them from the permissive gate. That is conservative - some are
-- false only because the audit wrote them - so the set can only shrink, never
-- silently grow, while the two columns converge.

-- Mirrors 014's partial index, widened to the conjunction training actually
-- filters on. Without this the added predicate turns the lookup back into a
-- scan as the reference set grows (2001 tile rows already).
CREATE INDEX IF NOT EXISTS reference_assets_trainable_audited_idx
    ON reference_assets (kind)
    WHERE deleted = false AND trainable = true AND audit_trainable IS NOT FALSE;
