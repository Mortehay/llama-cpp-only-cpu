-- Add sub_task_ids column to track distributed components of a chord task
ALTER TABLE sprite_images ADD COLUMN sub_task_ids JSONB;
