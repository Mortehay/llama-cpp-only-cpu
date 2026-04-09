-- Add seed column to preserve character identity
ALTER TABLE sprite_images ADD COLUMN IF NOT EXISTS seed BIGINT;
