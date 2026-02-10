# Podcast Tech Card Library

A LangGraph-powered pipeline that turns long-form podcast transcripts (Lex Fridman or local text) into an accumulative library of Tech Cards. It stores segments, topics, entities, and assertions in PostgreSQL and provides timestamped Q&A over the library.

## Features

- Ingest Lex Fridman transcript URLs or local text/markdown/json files
- Parse speaker + timestamp segments and keep YouTube timestamp links when available
- Extract topics, entities, and assertions with evidence quotes
- Upsert tech cards over time (idempotent re-runs per episode)
- Optional pgvector embeddings for semantic search
- CLI for ingest, report, and Q&A

## Requirements

- Python 3.11+
- PostgreSQL running locally
- `DATABASE_URL` set in your environment

Optional:
- pgvector extension for embeddings

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Set `DATABASE_URL` (for example):

```
postgresql+psycopg://localhost:5432/tech_radar
```

Initialize the database schema:

```bash
python -m tech_radar.migrations init
```

Enable pgvector (optional but recommended):

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## CLI

Ingest a Lex transcript URL:

```bash
python -m tech_radar ingest --url https://lexfridman.com/ai-sota-2026-transcript
```

Ingest local files:

```bash
python -m tech_radar ingest --files examples/sample_transcript.txt
```

Generate a markdown report:

```bash
python -m tech_radar report --episode 1 --out out/episode.md
```

Ask a question:

```bash
python -m tech_radar ask --q "What did they say about inference cost?" --episode 1 --mode fast
```

Query optimizer (debug output):

```bash
python -m tech_radar ask --q "What did they say about inference cost?" --episode 1 --mode fast --debug
```

Build chunks + embeddings for an episode:

```bash
python -m tech_radar build-chunks --episode-id 1
```

Vector search demo:

```bash
python -m tech_radar search --q "inference cost small batch" --top-k 5 --episode-id 1
```

Export the Tech Card library to Markdown (Obsidian/Git):

```bash
python -m tech_radar export-cards --out out/cards
```

Example output structure:

```
out/cards/
  ai.md
  deepseek-r1.md
```

## Smoke Test

```bash
python -m tech_radar.smoke_test
```

This runs a tiny transcript through ingest → cards → ask and prints a short answer with timestamp citations.

## Notes

- Assertions always include transcript evidence (segment IDs + quote).
- Re-running the same episode is idempotent due to unique hashes.
- This phase does not do decision support or project fit scoring.
