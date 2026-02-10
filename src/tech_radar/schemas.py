from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EpisodeInput(BaseModel):
    source_url: str | None = None
    title: str | None = None
    guests: list[str] = Field(default_factory=list)
    published_date: str | None = None
    raw_html: str | None = None


class Segment(BaseModel):
    id: int | None = None
    episode_id: int | None = None
    speaker: str | None = None
    t_start_sec: int | None = None
    t_end_sec: int | None = None
    youtube_url: str | None = None
    text: str
    hash: str | None = None


class Topic(BaseModel):
    id: int | None = None
    episode_id: int | None = None
    name: str
    summary: str
    start_seg_id: int | None = None
    end_seg_id: int | None = None


EntityType = Literal[
    "model",
    "company",
    "framework",
    "hardware",
    "benchmark",
    "paper",
    "product",
    "concept",
]


class Entity(BaseModel):
    id: int | None = None
    type: EntityType
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    first_seen_episode_id: int | None = None
    last_seen_episode_id: int | None = None


AssertionType = Literal["fact", "opinion", "prediction", "recommendation", "anecdote"]


class Assertion(BaseModel):
    id: int | None = None
    episode_id: int
    entity_id: int | None = None
    assertion_type: AssertionType
    statement: str
    speaker: str | None = None
    confidence: float = 0.5
    verify_priority: int = 0
    segment_ids: list[int]
    evidence_quote: str


class TechCard(BaseModel):
    entity_id: int
    short_definition: str
    key_points: list[str] = Field(default_factory=list)
    comparisons: list[str] = Field(default_factory=list)
    recent_summary: str


class TranscriptParseResult(BaseModel):
    episode: EpisodeInput
    toc: list[dict] = Field(default_factory=list)
    segments: list[Segment]


class TopicResult(BaseModel):
    topics: list[Topic]


class EntityResult(BaseModel):
    entities: list[Entity]


class AssertionResult(BaseModel):
    assertions: list[Assertion]


class CardResult(BaseModel):
    cards: list[TechCard]
    needs_refine: bool = False


class QAAnswer(BaseModel):
    answer: str
    citations: list[dict]


class Chunk(BaseModel):
    id: int | None = None
    episode_id: int | None = None
    topic_id: int | None = None
    start_seg_id: int | None = None
    end_seg_id: int | None = None
    t_start_sec: int | None = None
    t_end_sec: int | None = None
    segment_ids: list[int] = Field(default_factory=list)
    chunk_text: str


class RetrievalPlan(BaseModel):
    use_tech_cards: bool = True
    use_chunks: bool = True
    use_assertions: bool = False
    top_k_chunks: int = 8
    top_k_cards: int = 5
    filters: dict | None = None


class QueryOptimizationResult(BaseModel):
    intent: Literal["definition", "comparison", "opinion", "trend", "how-to", "fact"]
    entities: list[str] = Field(default_factory=list)
    time_hint: dict
    queries: list[str]
    retrieval_plan: RetrievalPlan
    answer_style: Literal["concise", "detailed", "bullet"] = "concise"
