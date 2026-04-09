-- Add llm_name and step_number columns to sprite_images table
ALTER TABLE sprite_images ADD COLUMN IF NOT EXISTS llm_name VARCHAR(255);
ALTER TABLE sprite_images ADD COLUMN IF NOT EXISTS step_number INTEGER;
