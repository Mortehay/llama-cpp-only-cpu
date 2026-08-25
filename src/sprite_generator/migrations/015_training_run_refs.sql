-- 015: remember which references each training run actually consumed.
--
-- So "train on the new images only" has a real answer instead of a guess.
-- Without this the system can tell you 199 references are trainable but not
-- which of them a given adapter has already seen, and re-training from scratch
-- on the full set every time is the only safe option.
--
-- A join table rather than an array column on reference_assets: the question
-- goes both ways ("what did this run use?" and "what has this image been used
-- in?"), and a deleted run should drop its rows without rewriting every
-- reference.
CREATE TABLE IF NOT EXISTS training_run_refs (
    run_id       UUID NOT NULL REFERENCES training_runs (id) ON DELETE CASCADE,
    reference_id UUID NOT NULL REFERENCES reference_assets (id) ON DELETE CASCADE,
    PRIMARY KEY (run_id, reference_id)
);

-- The hot query is "references NOT yet used by any SUCCESSFUL run", which
-- probes by reference_id.
CREATE INDEX IF NOT EXISTS training_run_refs_reference_idx
    ON training_run_refs (reference_id);

-- Rows are written at SUBMIT time, before the run has proven anything, so the
-- "already trained" test must join back to training_runs and require
-- status = 'done'. A run that failed or was purged taught the adapter nothing,
-- and its images must stay eligible - otherwise one crashed run permanently
-- removes images from every future dataset.
