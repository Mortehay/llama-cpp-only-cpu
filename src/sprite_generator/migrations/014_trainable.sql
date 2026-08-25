-- 014: separate "can I measure this?" from "can I train on this?".
--
-- `reference_assets.usable` was doing both jobs and doing one of them badly.
-- Against 227 real references it rejected 100 of 106 sprites and 60 of 90
-- characters, and the rejections were almost all wrong:
--
--   * sprites failed on "32,000 distinct colours". They were JPEG reference
--     boards, not palette-locked art. You genuinely cannot read a PALETTE from
--     them - and you can perfectly well train a style on them.
--   * characters failed on "fills the frame; reaches the border", with an
--     average aspect of 1.36 - correctly character-shaped, simply CROPPED
--     TIGHTLY, which is what good reference art looks like. That rule came
--     from concept.judge, which asks "did the GENERATOR produce a scene?" -
--     the opposite expectation from a reference a human picked on purpose.
--
-- So the two verdicts are stored separately. `usable` keeps its strict meaning
-- and still gates style-profile derivation, because a palette read off a JPEG
-- collage would be garbage that silently poisons every sheet built afterwards.
-- `trainable` is permissive and gates training.
ALTER TABLE reference_assets
    ADD COLUMN IF NOT EXISTS trainable BOOLEAN,
    ADD COLUMN IF NOT EXISTS trainable_why TEXT;

-- Training counts this constantly; the partial index keeps that a lookup
-- rather than a scan as the reference set grows.
CREATE INDEX IF NOT EXISTS reference_assets_trainable_idx
    ON reference_assets (kind) WHERE deleted = false AND trainable = true;
