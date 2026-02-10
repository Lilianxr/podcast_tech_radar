CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
  id SERIAL PRIMARY KEY,
  episode_id INTEGER REFERENCES episodes(id),
  topic_id INTEGER REFERENCES topics(id),
  start_seg_id INTEGER,
  end_seg_id INTEGER,
  t_start_sec INTEGER,
  t_end_sec INTEGER,
  chunk_text TEXT NOT NULL,
  chunk_hash TEXT UNIQUE NOT NULL,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunk_segments (
  chunk_id INTEGER REFERENCES chunks(id),
  segment_id INTEGER REFERENCES segments(id),
  ord INTEGER NOT NULL,
  PRIMARY KEY (chunk_id, segment_id),
  UNIQUE (chunk_id, ord)
);

ALTER TABLE embeddings RENAME COLUMN vector TO embedding_json;
ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS model_name TEXT;
ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS dims INTEGER;
ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS embedding vector(1536);
ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW();

ALTER TABLE embeddings DROP CONSTRAINT IF EXISTS embeddings_object_type_check;
ALTER TABLE embeddings ADD CONSTRAINT embeddings_object_type_check CHECK (object_type IN ('segment', 'card', 'assertion', 'chunk'));

CREATE UNIQUE INDEX IF NOT EXISTS embeddings_unique ON embeddings (object_type, object_id, model_name, dims);
CREATE INDEX IF NOT EXISTS chunks_episode_id_idx ON chunks (episode_id);
CREATE INDEX IF NOT EXISTS chunks_topic_id_idx ON chunks (topic_id);

CREATE INDEX IF NOT EXISTS embeddings_hnsw_idx ON embeddings USING hnsw (embedding vector_cosine_ops);
