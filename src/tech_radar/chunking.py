from __future__ import annotations

from typing import Iterable

from .schemas import Chunk, Topic, Segment
from .utils import seconds_to_hms, estimate_tokens


def build_chunks_from_topics(
    topics: Iterable[Topic],
    segments: Iterable[Segment],
    max_tokens: int = 800,
    min_segs: int = 2,
    max_segs: int = 6,
    overlap: int = 1,
) -> list[Chunk]:
    ordered_segments = [seg for seg in segments if seg.id is not None]
    ordered_segments.sort(key=lambda s: s.id or 0)
    if not ordered_segments:
        return []

    topic_ranges = _topic_ranges(list(topics), ordered_segments)
    chunks: list[Chunk] = []
    for topic, segs in topic_ranges:
        segs_to_use = segs if len(segs) >= min_segs else ordered_segments
        chunks.extend(
            _build_topic_chunks(
                topic,
                segs_to_use,
                max_tokens=max_tokens,
                min_segs=min_segs,
                max_segs=max_segs,
                overlap=overlap,
            )
        )
    return chunks


def _topic_ranges(topics: list[Topic], segments: list[Segment]) -> list[tuple[Topic | None, list[Segment]]]:
    if not topics:
        return [(None, segments)]
    seg_ids = [seg.id for seg in segments if seg.id is not None]
    ranges: list[tuple[Topic | None, list[Segment]]] = []
    for topic in topics:
        if topic.start_seg_id in seg_ids and topic.end_seg_id in seg_ids:
            start_idx = seg_ids.index(topic.start_seg_id)
            end_idx = seg_ids.index(topic.end_seg_id)
            if start_idx <= end_idx:
                ranges.append((topic, segments[start_idx : end_idx + 1]))
                continue
        ranges.append((topic, segments))
    return ranges


def _build_topic_chunks(
    topic: Topic | None,
    segments: list[Segment],
    max_tokens: int,
    min_segs: int,
    max_segs: int,
    overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0
    total = len(segments)
    while idx < total:
        window = segments[idx : min(total, idx + max_segs)]
        if len(window) < min_segs:
            break
        chunk_text = _build_chunk_text(topic, window)
        while estimate_tokens(chunk_text) > max_tokens and len(window) > min_segs:
            window = window[:-1]
            chunk_text = _build_chunk_text(topic, window)
        if len(window) < min_segs:
            break
        chunks.append(_make_chunk(topic, window, chunk_text))
        idx += max(1, len(window) - overlap)
    return chunks


def _build_chunk_text(topic: Topic | None, segments: list[Segment]) -> str:
    lines = []
    if topic:
        lines.append(f"[Topic: {topic.name}]")
    for seg in segments:
        timestamp = seconds_to_hms(seg.t_start_sec)
        link = f"[{seg.youtube_url}]" if seg.youtube_url else ""
        lines.append(f"{seg.speaker or 'Unknown'} ({timestamp}){link}")
        lines.append(seg.text)
    return "\n".join(lines).strip()


def _make_chunk(topic: Topic | None, segments: list[Segment], chunk_text: str) -> Chunk:
    start = segments[0]
    end = segments[-1]
    return Chunk(
        topic_id=topic.id if topic else None,
        start_seg_id=start.id,
        end_seg_id=end.id,
        t_start_sec=start.t_start_sec,
        t_end_sec=end.t_end_sec,
        segment_ids=[seg.id for seg in segments if seg.id is not None],
        chunk_text=chunk_text,
    )
