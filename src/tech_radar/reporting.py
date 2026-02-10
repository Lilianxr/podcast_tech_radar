from __future__ import annotations

import json
from pathlib import Path

from .models import Assertion, Chunk, Episode, Segment, Topic
from .storage import fetch_cards, fetch_chunks_for_episode
from .utils import first_sentence, take_quote


def build_json_payload(
    episode: Episode,
    segments: list[Segment],
    topics: list[Topic],
    assertions: list[Assertion],
) -> dict:
    return {
        "episode": {
            "id": episode.id,
            "source_url": episode.source_url,
            "title": episode.title,
            "guests": episode.guests,
        },
        "segments": [
            {
                "id": seg.id,
                "speaker": seg.speaker,
                "t_start_sec": seg.t_start_sec,
                "youtube_url": seg.youtube_url,
                "text": seg.text,
            }
            for seg in segments
        ],
        "topics": [
            {
                "id": topic.id,
                "name": topic.name,
                "summary": topic.summary,
                "start_seg_id": topic.start_seg_id,
                "end_seg_id": topic.end_seg_id,
            }
            for topic in topics
        ],
        "assertions": [
            {
                "id": assertion.id,
                "assertion_type": assertion.assertion_type,
                "statement": assertion.statement,
                "speaker": assertion.speaker,
                "segment_ids": assertion.segment_ids,
                "evidence_quote": assertion.evidence_quote,
            }
            for assertion in assertions
        ],
        "cards": [
            {
                "entity_id": card.entity_id,
                "short_definition": card.short_definition,
                "key_points": card.key_points,
                "comparisons": card.comparisons,
                "recent_summary": card.recent_summary,
            }
            for card in fetch_cards()
        ],
    }


def render_markdown(
    episode: Episode,
    segments: list[Segment],
    topics: list[Topic],
    assertions: list[Assertion],
) -> str:
    lines = [f"# Episode {episode.id}: {episode.title or 'Untitled'}", ""]
    if episode.source_url:
        lines.append(f"Source: {episode.source_url}")
        lines.append("")

    lines.append("## Topics")
    for topic in topics:
        lines.append(f"- **{topic.name}**: {topic.summary}")
    if not topics:
        lines.append("- (No topics extracted)")

    lines.append("")
    lines.append("## Assertions")
    for assertion in assertions:
        lines.append(f"- ({assertion.assertion_type}) {assertion.statement}")
        lines.append(f"  - Evidence: \"{assertion.evidence_quote}\"")
        lines.append(f"  - Segments: {assertion.segment_ids}")
    if not assertions:
        lines.append("- (No assertions extracted)")

    lines.append("")
    lines.append("## Key Segments")
    chunks = fetch_chunks_for_episode(episode.id)
    if chunks:
        chunks_sorted = sorted(chunks, key=lambda c: c.t_start_sec or 0)[:5]
        shown = 0
        for chunk in chunks_sorted:
            from .storage import fetch_segment_ids_for_chunk, fetch_segments_by_ids

            seg_ids = fetch_segment_ids_for_chunk(chunk.id)
            segs = fetch_segments_by_ids(seg_ids)[:3]
            for seg in segs:
                link = f" ({seg.youtube_url})" if seg.youtube_url else ""
                lines.append(f"- {seg.speaker or 'Unknown'} @ {seg.t_start_sec}s{link}")
                lines.append(f"  - {first_sentence(seg.text, 360)}")
                shown += 1
                if shown >= 10:
                    break
            if shown >= 10:
                break
    else:
        for seg in segments[:10]:
            link = f" ({seg.youtube_url})" if seg.youtube_url else ""
            lines.append(f"- {seg.speaker or 'Unknown'} @ {seg.t_start_sec}s{link}")
            lines.append(f"  - {first_sentence(seg.text, 360)}")

    lines.append("")
    lines.append("## Chunks (Top)")
    chunks = fetch_chunks_for_episode(episode.id)
    for chunk in chunks[:5]:
        lines.append(f"- Topic {chunk.topic_id} @ {chunk.t_start_sec}s–{chunk.t_end_sec}s:")
        lines.append(f"  - {chunk.chunk_text}")
    if not chunks:
        lines.append("- (No chunks yet)")

    return "\n".join(lines)


def write_report(
    episode: Episode,
    segments: list[Segment],
    topics: list[Topic],
    assertions: list[Assertion],
    out_path: str,
    json_out: str | None = None,
) -> None:
    markdown = render_markdown(episode, segments, topics, assertions)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(markdown, encoding="utf-8")

    if json_out:
        payload = build_json_payload(episode, segments, topics, assertions)
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
