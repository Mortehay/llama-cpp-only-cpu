-- Add support for tracking core vs spritesheet types and component parts
ALTER TABLE sprite_images ADD COLUMN IF NOT EXISTS image_type TEXT DEFAULT 'spritesheet';
ALTER TABLE sprite_images ADD COLUMN IF NOT EXISTS parent_id INTEGER REFERENCES sprite_images(id) ON DELETE SET NULL;
ALTER TABLE sprite_images ADD COLUMN IF NOT EXISTS components JSONB DEFAULT '[]'::jsonb;
ALTER TABLE sprite_images ADD COLUMN IF NOT EXISTS requested_actions JSONB DEFAULT '[]'::jsonb;
