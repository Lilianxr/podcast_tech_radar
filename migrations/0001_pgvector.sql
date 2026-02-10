CREATE EXTENSION IF NOT EXISTS vector;

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='embeddings' AND column_name='vector') THEN
    ALTER TABLE embeddings ALTER COLUMN vector TYPE vector(1536) USING vector::vector(1536);
  END IF;
END $$;
