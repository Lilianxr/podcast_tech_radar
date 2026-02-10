from __future__ import annotations

from typing import Iterable

from .config import get_settings
from .models import Chunk
from .storage import fetch_embeddings, upsert_embeddings
from .utils import compact_spaces


def _embedding_client():
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for embeddings")
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=settings.embedding_model)


def embed_chunks(chunks: Iterable[Chunk]) -> int:
    settings = get_settings()
    items = [chunk for chunk in chunks if chunk.id is not None]
    if not items:
        return 0
    existing = fetch_embeddings("chunk", [chunk.id for chunk in items], settings.embedding_model, settings.embedding_dims)
    missing = [chunk for chunk in items if chunk.id not in existing]
    if not missing:
        return 0
    client = _embedding_client()
    texts = [compact_spaces(chunk.chunk_text) for chunk in missing]
    vectors = client.embed_documents(texts)
    upsert_embeddings(
        "chunk",
        [chunk.id for chunk in missing],
        vectors,
        model_name=settings.embedding_model,
        dims=settings.embedding_dims,
    )
    return len(missing)


def embed_query(text: str) -> list[float]:
    client = _embedding_client()
    return client.embed_query(text)
