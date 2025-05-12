CREATE TABLE IF NOT EXISTS user_feedback (
    created_at timestamp default now(),
    user_id INTEGER,
    playlist_id INTEGER NOT NULL,
    score INTEGER
);

