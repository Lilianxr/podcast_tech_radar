from __future__ import annotations

from typing import Iterable

from sqlalchemy import select, text
from pgvector.psycopg import Vector as PgVector

from .db import get_session
from .models import Assertion, Chunk, ChunkSegment, Embedding, Entity, Episode, Segment, TechCard, Topic
from .schemas import Assertion as AssertionSchema
from .schemas import Entity as EntitySchema
from .schemas import Chunk as ChunkSchema
from .schemas import EpisodeInput, Segment as SegmentSchema, TechCard as TechCardSchema
from .schemas import Topic as TopicSchema
from .utils import hash_text, normalize_name


def upsert_episode(episode: EpisodeInput) -> Episode:
    with get_session() as session:
        existing = None
        if episode.source_url:
            existing = session.execute(
                select(Episode).where(Episode.source_url == episode.source_url)
            ).scalar_one_or_none()
        if existing:
            if episode.title:
                existing.title = episode.title
            if episode.guests:
                existing.guests = episode.guests
            if episode.published_date:
                existing.published_date = episode.published_date
            if episode.raw_html:
                existing.raw_html = episode.raw_html
            return existing
        created = Episode(
            source_url=episode.source_url,
            title=episode.title,
            guests=episode.guests,
            published_date=episode.published_date,
            raw_html=episode.raw_html,
        )
        session.add(created)
        session.flush()
    return created


def fetch_episode(episode_id: int) -> Episode | None:
    with get_session() as session:
        return session.execute(select(Episode).where(Episode.id == episode_id)).scalar_one_or_none()


def upsert_segments(episode_id: int, segments: Iterable[SegmentSchema]) -> list[Segment]:
    stored: list[Segment] = []
    with get_session() as session:
        for seg in segments:
            hash_value = seg.hash or hash_text(seg.text)
            existing = session.execute(
                select(Segment).where(Segment.hash == hash_value)
            ).scalar_one_or_none()
            if existing:
                stored.append(existing)
                continue
            created = Segment(
                episode_id=episode_id,
                speaker=seg.speaker,
                t_start_sec=seg.t_start_sec,
                t_end_sec=seg.t_end_sec,
                youtube_url=seg.youtube_url,
                text=seg.text,
                hash=hash_value,
            )
            session.add(created)
            session.flush()
            stored.append(created)
    return stored


def upsert_entities(episode_id: int, entities: Iterable[EntitySchema]) -> list[Entity]:
    stored: list[Entity] = []
    with get_session() as session:
        for ent in entities:
            canonical = normalize_name(ent.canonical_name)
            existing = session.execute(
                select(Entity).where(Entity.canonical_name == canonical)
            ).scalar_one_or_none()
            if existing:
                existing.last_seen_episode_id = episode_id
                existing.aliases = sorted(set(existing.aliases + ent.aliases))
                stored.append(existing)
                continue
            created = Entity(
                type=ent.type,
                canonical_name=canonical,
                aliases=ent.aliases,
                first_seen_episode_id=episode_id,
                last_seen_episode_id=episode_id,
            )
            session.add(created)
            session.flush()
            stored.append(created)
    return stored


def upsert_assertions(assertions: Iterable[AssertionSchema]) -> list[Assertion]:
    stored: list[Assertion] = []
    with get_session() as session:
        for assertion in assertions:
            hash_value = hash_text(
                f"{assertion.episode_id}|{assertion.statement}|{assertion.speaker}|{assertion.segment_ids}"
            )
            existing = session.execute(
                select(Assertion).where(Assertion.hash == hash_value)
            ).scalar_one_or_none()
            if existing:
                stored.append(existing)
                continue
            created = Assertion(
                episode_id=assertion.episode_id,
                entity_id=assertion.entity_id,
                assertion_type=assertion.assertion_type,
                statement=assertion.statement,
                speaker=assertion.speaker,
                confidence=assertion.confidence,
                verify_priority=assertion.verify_priority,
                segment_ids=assertion.segment_ids,
                evidence_quote=assertion.evidence_quote,
                hash=hash_value,
            )
            session.add(created)
            session.flush()
            stored.append(created)
    return stored


def upsert_cards(cards: Iterable[TechCardSchema]) -> list[TechCard]:
    stored: list[TechCard] = []
    with get_session() as session:
        for card in cards:
            existing = session.execute(
                select(TechCard).where(TechCard.entity_id == card.entity_id)
            ).scalar_one_or_none()
            if existing:
                existing.short_definition = card.short_definition
                existing.key_points = card.key_points
                existing.comparisons = card.comparisons
                existing.recent_summary = card.recent_summary
                stored.append(existing)
                continue
            created = TechCard(
                entity_id=card.entity_id,
                short_definition=card.short_definition,
                key_points=card.key_points,
                comparisons=card.comparisons,
                recent_summary=card.recent_summary,
            )
            session.add(created)
            session.flush()
            stored.append(created)
    return stored


def upsert_chunks(episode_id: int, chunks: Iterable[ChunkSchema]) -> list[Chunk]:
    stored: list[Chunk] = []
    with get_session() as session:
        for chunk in chunks:
            hash_value = hash_text(chunk.chunk_text)
            existing = session.execute(select(Chunk).where(Chunk.chunk_hash == hash_value)).scalar_one_or_none()
            if existing:
                stored.append(existing)
                continue
            created = Chunk(
                episode_id=episode_id,
                topic_id=chunk.topic_id,
                start_seg_id=chunk.start_seg_id,
                end_seg_id=chunk.end_seg_id,
                t_start_sec=chunk.t_start_sec,
                t_end_sec=chunk.t_end_sec,
                chunk_text=chunk.chunk_text,
                chunk_hash=hash_value,
            )
            session.add(created)
            session.flush()
            for ord_idx, seg_id in enumerate(chunk.segment_ids):
                session.add(ChunkSegment(chunk_id=created.id, segment_id=seg_id, ord=ord_idx))
            stored.append(created)
    return stored


def fetch_chunks_for_episode(episode_id: int) -> list[Chunk]:
    with get_session() as session:
        return list(session.execute(select(Chunk).where(Chunk.episode_id == episode_id)).scalars().all())


def fetch_segment_ids_for_chunk(chunk_id: int) -> list[int]:
    with get_session() as session:
        rows = session.execute(
            select(ChunkSegment.segment_id).where(ChunkSegment.chunk_id == chunk_id).order_by(ChunkSegment.ord)
        ).scalars()
        return list(rows)


def fetch_embeddings(
    object_type: str, object_ids: list[int], model_name: str, dims: int
) -> set[int]:
    if not object_ids:
        return set()
    with get_session() as session:
        rows = session.execute(
            select(Embedding.object_id).where(
                Embedding.object_type == object_type,
                Embedding.object_id.in_(object_ids),
                Embedding.model_name == model_name,
                Embedding.dims == dims,
            )
        ).scalars()
        return set(rows)


def upsert_embeddings(
    object_type: str, object_ids: list[int], embeddings: list[list[float]], model_name: str, dims: int
) -> None:
    with get_session() as session:
        for object_id, vector in zip(object_ids, embeddings):
            existing = session.execute(
                select(Embedding).where(
                    Embedding.object_type == object_type,
                    Embedding.object_id == object_id,
                    Embedding.model_name == model_name,
                    Embedding.dims == dims,
                )
            ).scalar_one_or_none()
            if existing:
                existing.embedding = vector
                continue
            session.add(
                Embedding(
                    object_type=object_type,
                    object_id=object_id,
                    model_name=model_name,
                    dims=dims,
                    embedding=vector,
                )
            )


def similarity_search_chunks(
    query_embedding: list[float],
    top_k: int = 5,
    episode_id: int | None = None,
    topic_id: int | None = None,
    model_name: str | None = None,
    dims: int | None = None,
) -> list[tuple[Chunk, float]]:
    filters = []
    params = {
        "query": PgVector(query_embedding).to_text(),
        "top_k": top_k,
    }
    if episode_id is not None:
        filters.append("c.episode_id = :episode_id")
        params["episode_id"] = episode_id
    if topic_id is not None:
        filters.append("c.topic_id = :topic_id")
        params["topic_id"] = topic_id
    if model_name is not None:
        filters.append("e.model_name = :model_name")
        params["model_name"] = model_name
    if dims is not None:
        filters.append("e.dims = :dims")
        params["dims"] = dims
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    sql = text(
        (
            "SELECT c.*, (e.embedding <=> CAST(:query AS vector)) AS distance "
            "FROM chunks c "
            "JOIN embeddings e ON e.object_type = 'chunk' AND e.object_id = c.id "
            f"{where} "
            "ORDER BY e.embedding <=> CAST(:query AS vector) "
            "LIMIT :top_k"
        )
    )
    with get_session() as session:
        rows = session.execute(sql, params).mappings().all()
        results = []
        for row in rows:
            chunk = Chunk(
                id=row["id"],
                episode_id=row["episode_id"],
                topic_id=row["topic_id"],
                start_seg_id=row["start_seg_id"],
                end_seg_id=row["end_seg_id"],
                t_start_sec=row["t_start_sec"],
                t_end_sec=row["t_end_sec"],
                chunk_text=row["chunk_text"],
                chunk_hash=row["chunk_hash"],
            )
            results.append((chunk, float(row["distance"])))
        return results


def upsert_topics(episode_id: int, topics: Iterable[TopicSchema]) -> list[Topic]:
    stored: list[Topic] = []
    with get_session() as session:
        for topic in topics:
            created = Topic(
                episode_id=episode_id,
                name=topic.name,
                summary=topic.summary,
                start_seg_id=topic.start_seg_id,
                end_seg_id=topic.end_seg_id,
            )
            session.add(created)
            session.flush()
            stored.append(created)
    return stored


def fetch_topics_for_episode(episode_id: int) -> list[Topic]:
    with get_session() as session:
        return list(session.execute(select(Topic).where(Topic.episode_id == episode_id)).scalars().all())


def fetch_episode_segments(episode_id: int) -> list[Segment]:
    with get_session() as session:
        return list(
            session.execute(select(Segment).where(Segment.episode_id == episode_id)).scalars().all()
        )


def fetch_cards() -> list[TechCard]:
    with get_session() as session:
        return list(session.execute(select(TechCard)).scalars().all())


def fetch_entities() -> list[Entity]:
    with get_session() as session:
        return list(session.execute(select(Entity)).scalars().all())


def fetch_assertions_for_episode(episode_id: int) -> list[Assertion]:
    with get_session() as session:
        return list(
            session.execute(select(Assertion).where(Assertion.episode_id == episode_id)).scalars().all()
        )


def fetch_assertions_for_entity(entity_id: int) -> list[Assertion]:
    with get_session() as session:
        return list(session.execute(select(Assertion).where(Assertion.entity_id == entity_id)).scalars().all())


def fetch_segments_by_ids(segment_ids: list[int]) -> list[Segment]:
    if not segment_ids:
        return []
    with get_session() as session:
        return list(session.execute(select(Segment).where(Segment.id.in_(segment_ids))).scalars().all())
