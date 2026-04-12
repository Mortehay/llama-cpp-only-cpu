-- Add 'deleted' column to sprite_images table for soft deletion support
ALTER TABLE sprite_images ADD COLUMN IF NOT EXISTS deleted BOOLEAN DEFAULT false;
