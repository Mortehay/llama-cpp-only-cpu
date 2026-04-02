-- Relax NOT NULL on file_path to allow storing queued tasks before generation
ALTER TABLE sprite_images ALTER COLUMN file_path DROP NOT NULL;

-- Add task_id column to track Celery tasks directly in the records
ALTER TABLE sprite_images ADD COLUMN task_id TEXT;
