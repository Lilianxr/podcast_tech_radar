from __future__ import annotations

import datetime as dt

from sqlalchemy import CheckConstraint, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass


class Episode(Base):
    __tablename__ = "episodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_url: Mapped[str | None] = mapped_column(String, unique=True)
    title: Mapped[str | None] = mapped_column(String)
    guests: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    published_date: Mapped[dt.date | None] = mapped_column(Date)
    raw_html: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    segments: Mapped[list[Segment]] = relationship("Segment", back_populates="episode")


class Segment(Base):
    __tablename__ = "segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    episode_id: Mapped[int] = mapped_column(ForeignKey("episodes.id"), index=True)
    speaker: Mapped[str | None] = mapped_column(String)
    t_start_sec: Mapped[int | None] = mapped_column(Integer)
    t_end_sec: Mapped[int | None] = mapped_column(Integer)
    youtube_url: Mapped[str | None] = mapped_column(String)
    text: Mapped[str] = mapped_column(Text)
    hash: Mapped[str] = mapped_column(String, unique=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    episode: Mapped[Episode] = relationship("Episode", back_populates="segments")


class Topic(Base):
    __tablename__ = "topics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    episode_id: Mapped[int] = mapped_column(ForeignKey("episodes.id"), index=True)
    name: Mapped[str] = mapped_column(String)
    summary: Mapped[str] = mapped_column(Text)
    start_seg_id: Mapped[int | None] = mapped_column(Integer)
    end_seg_id: Mapped[int | None] = mapped_column(Integer)


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    episode_id: Mapped[int] = mapped_column(ForeignKey("episodes.id"), index=True)
    topic_id: Mapped[int | None] = mapped_column(ForeignKey("topics.id"), index=True)
    start_seg_id: Mapped[int | None] = mapped_column(Integer)
    end_seg_id: Mapped[int | None] = mapped_column(Integer)
    t_start_sec: Mapped[int | None] = mapped_column(Integer)
    t_end_sec: Mapped[int | None] = mapped_column(Integer)
    chunk_text: Mapped[str] = mapped_column(Text)
    chunk_hash: Mapped[str] = mapped_column(String, unique=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class ChunkSegment(Base):
    __tablename__ = "chunk_segments"

    chunk_id: Mapped[int] = mapped_column(ForeignKey("chunks.id"), primary_key=True)
    segment_id: Mapped[int] = mapped_column(ForeignKey("segments.id"), primary_key=True)
    ord: Mapped[int] = mapped_column(Integer)


class Entity(Base):
    __tablename__ = "entities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    type: Mapped[str] = mapped_column(String)
    canonical_name: Mapped[str] = mapped_column(String, unique=True)
    aliases: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    first_seen_episode_id: Mapped[int | None] = mapped_column(Integer)
    last_seen_episode_id: Mapped[int | None] = mapped_column(Integer)


class Assertion(Base):
    __tablename__ = "assertions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    episode_id: Mapped[int] = mapped_column(ForeignKey("episodes.id"), index=True)
    entity_id: Mapped[int | None] = mapped_column(ForeignKey("entities.id"), index=True)
    assertion_type: Mapped[str] = mapped_column(String)
    statement: Mapped[str] = mapped_column(Text)
    speaker: Mapped[str | None] = mapped_column(String)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    verify_priority: Mapped[int] = mapped_column(Integer, default=0)
    segment_ids: Mapped[list[int]] = mapped_column(ARRAY(Integer))
    evidence_quote: Mapped[str] = mapped_column(String(240))
    hash: Mapped[str] = mapped_column(String, unique=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    verification_status: Mapped[str | None] = mapped_column(String)


class TechCard(Base):
    __tablename__ = "tech_cards"

    entity_id: Mapped[int] = mapped_column(ForeignKey("entities.id"), primary_key=True)
    short_definition: Mapped[str] = mapped_column(Text)
    key_points: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    comparisons: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    recent_summary: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class Embedding(Base):
    __tablename__ = "embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    object_type: Mapped[str] = mapped_column(String)
    object_id: Mapped[int] = mapped_column(Integer)
    model_name: Mapped[str] = mapped_column(String)
    dims: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[list[float]] = mapped_column(Vector(1536))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    __table_args__ = (
        CheckConstraint("object_type IN ('segment', 'card', 'assertion', 'chunk')"),
    )
