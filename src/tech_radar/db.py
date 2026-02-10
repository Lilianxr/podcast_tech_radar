from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from .config import get_settings


_settings = get_settings()
_engine = create_engine(_settings.database_url, future=True)
SessionLocal = sessionmaker(bind=_engine, class_=Session, expire_on_commit=False)


@contextmanager
def get_session() -> Session:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def run_sql_file(path: str) -> None:
    with open(path, "r", encoding="utf-8") as handle:
        sql = handle.read()
    with _engine.begin() as conn:
        conn.execute(text(sql))


def engine():
    return _engine
