from __future__ import annotations

from typing import TypedDict

from .chunking import build_chunks_from_topics
from .schemas import Chunk as ChunkSchema
from .schemas import Segment as SegmentSchema
from .schemas import Topic as TopicSchema
from .storage import fetch_chunks_for_episode, upsert_chunks


class ChunkState(TypedDict, total=False):
    episode_id: int
    topics: list
    segments: list
    chunks: list


def chunk_builder(state: ChunkState) -> ChunkState:
    topics = state.get("topics", [])
    segments = state.get("segments", [])
    chunks = build_chunks_from_topics(topics, segments)
    return {**state, "chunks": chunks}


def chunk_persist(state: ChunkState) -> ChunkState:
    episode_id = state["episode_id"]
    chunks = state.get("chunks", [])
    stored = upsert_chunks(episode_id, chunks)
    return {**state, "chunks": stored}
