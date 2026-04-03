-- Add columns for detailed task progress tracking
ALTER TABLE sprite_images ADD COLUMN progress_pct INTEGER DEFAULT 0;
ALTER TABLE sprite_images ADD COLUMN progress_msg TEXT;
