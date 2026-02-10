CREATE TABLE IF NOT EXISTS episodes (
  id SERIAL PRIMARY KEY,
  source_url TEXT UNIQUE,
  title TEXT,
  guests TEXT[] DEFAULT '{}',
  published_date DATE,
  raw_html TEXT,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS segments (
  id SERIAL PRIMARY KEY,
  episode_id INTEGER REFERENCES episodes(id),
  speaker TEXT,
  t_start_sec INTEGER,
  t_end_sec INTEGER,
  youtube_url TEXT,
  text TEXT NOT NULL,
  hash TEXT UNIQUE NOT NULL,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS topics (
  id SERIAL PRIMARY KEY,
  episode_id INTEGER REFERENCES episodes(id),
  name TEXT NOT NULL,
  summary TEXT NOT NULL,
  start_seg_id INTEGER,
  end_seg_id INTEGER
);

CREATE TABLE IF NOT EXISTS entities (
  id SERIAL PRIMARY KEY,
  type TEXT NOT NULL,
  canonical_name TEXT UNIQUE NOT NULL,
  aliases TEXT[] DEFAULT '{}',
  first_seen_episode_id INTEGER,
  last_seen_episode_id INTEGER
);

CREATE TABLE IF NOT EXISTS assertions (
  id SERIAL PRIMARY KEY,
  episode_id INTEGER REFERENCES episodes(id),
  entity_id INTEGER REFERENCES entities(id),
  assertion_type TEXT NOT NULL,
  statement TEXT NOT NULL,
  speaker TEXT,
  confidence REAL DEFAULT 0.5,
  verify_priority INTEGER DEFAULT 0,
  segment_ids INTEGER[] NOT NULL,
  evidence_quote VARCHAR(240) NOT NULL,
  hash TEXT UNIQUE NOT NULL,
  created_at TIMESTAMP DEFAULT NOW(),
  verification_status TEXT
);

CREATE TABLE IF NOT EXISTS tech_cards (
  entity_id INTEGER PRIMARY KEY REFERENCES entities(id),
  short_definition TEXT NOT NULL,
  key_points TEXT[] DEFAULT '{}',
  comparisons TEXT[] DEFAULT '{}',
  recent_summary TEXT NOT NULL,
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS embeddings (
  id SERIAL PRIMARY KEY,
  object_type TEXT NOT NULL CHECK (object_type IN ('segment', 'card', 'assertion')),
  object_id INTEGER NOT NULL,
  vector JSONB
);
