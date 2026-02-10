from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    model: str
    temperature: float
    database_url: str
    use_pgvector: bool
    vector_dim: int
    stub_llm: bool
    embedding_model: str
    embedding_dims: int


def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.3")),
        database_url=os.getenv("DATABASE_URL", "postgresql+psycopg://localhost:5432/tech_radar"),
        use_pgvector=os.getenv("USE_PGVECTOR", "false").lower() == "true",
        vector_dim=int(os.getenv("VECTOR_DIM", "1536")),
        stub_llm=os.getenv("TECH_RADAR_STUB_LLM", "false").lower() == "true",
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_dims=int(os.getenv("EMBEDDING_DIMS", os.getenv("VECTOR_DIM", "1536"))),
    )
