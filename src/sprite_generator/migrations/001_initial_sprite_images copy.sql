CREATE TABLE IF NOT EXISTS sprite_images (
    id          SERIAL PRIMARY KEY,
    timestamp   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    prompt      TEXT      NOT NULL,
    file_path   TEXT      NOT NULL,
    duration_ms FLOAT
);
