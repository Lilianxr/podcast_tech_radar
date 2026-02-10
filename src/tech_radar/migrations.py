from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import get_settings
from .db import run_sql_file


def run_init() -> None:
    root = Path(__file__).resolve().parents[2]
    base = root / "migrations" / "0001_init.sql"
    run_sql_file(str(base))
    settings = get_settings()
    if settings.use_pgvector:
        pgvector_sql = root / "migrations" / "0001_pgvector.sql"
        run_sql_file(str(pgvector_sql))
    extra = root / "migrations" / "0002_chunks_embeddings.sql"
    if extra.exists():
        run_sql_file(str(extra))


def main() -> None:
    parser = argparse.ArgumentParser(description="Database migrations for Tech Radar.")
    parser.add_argument("command", choices=["init"], help="Migration command")
    args = parser.parse_args()

    if args.command == "init":
        run_init()


if __name__ == "__main__":
    main()
