CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Initialize default compute mode
INSERT INTO app_settings (key, value) 
VALUES ('compute_mode', '"cpu"')
ON CONFLICT (key) DO NOTHING;
