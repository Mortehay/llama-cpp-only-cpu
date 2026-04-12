-- Add 'cropped_from' column to track source of cropped images
ALTER TABLE sprite_images ADD COLUMN IF NOT EXISTS cropped_from INTEGER REFERENCES sprite_images(id);
