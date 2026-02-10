# Repository Guidelines

## Project Structure & Module Organization

- `src/tech_radar/` contains the LangGraph workflow, CLI, parsers, and storage.
  - `graph.py` defines the ingest and QA workflows.
  - `nodes.py` holds LangGraph node implementations.
  - `parsers/` contains the Lex transcript and generic text parsers.
  - `models.py` and `storage.py` handle PostgreSQL persistence.
  - `cli.py` provides the CLI entry point.
- `migrations/` stores SQL migrations.
- `examples/` includes a small sample transcript for smoke testing.

## Build, Test, and Development Commands

- `python -m venv .venv` - create virtual environment.
- `pip install -e .` - install in editable mode.
- `python -m tech_radar.migrations init` - initialize the database schema.
- `python -m tech_radar ingest --url <lex_url>` - ingest a transcript.
- `python -m tech_radar report --episode <id> --out out/episode.md` - generate notes.
- `python -m tech_radar ask --episode <id> --q "..."` - query the library.
- `python -m tech_radar.smoke_test` - run a stubbed end-to-end check.

## Coding Style & Naming Conventions

- Python code is formatted with `ruff format` and linted with `ruff`.
- Use `snake_case` for modules and functions, `PascalCase` for classes.
- Keep node logic focused and reusable; avoid side effects outside storage helpers.

## Testing Guidelines

No formal test suite yet. When adding tests:

- Use `tests/` with `test_*.py` naming.
- Focus on parser behavior, idempotent upserts, and graph wiring.
- Document how to run tests locally (e.g., `pytest`).

## Commit & Pull Request Guidelines

- Use concise, imperative commit messages (e.g., "Add assertion extractor").
- Keep PRs scoped and include a short summary and testing notes.
- Link related issues and add logs/screenshots when useful.

## Configuration & Security Notes

- Do not commit secrets. Add keys to `.env` and keep `.env.example` updated.
- Set `DATABASE_URL` for PostgreSQL and enable `USE_PGVECTOR=true` only if the extension is installed.
