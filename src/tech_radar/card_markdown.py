from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - fallback if PyYAML is not available
    yaml = None

from .models import Assertion, Entity, Segment, TechCard
from .storage import (
    fetch_assertions_for_entity,
    fetch_entities,
    fetch_segments_by_ids,
    similarity_search_chunks,
)
from .embeddings import embed_query
from .config import get_settings
from .utils import seconds_to_hms, slugify, take_quote


def render_tech_card_markdown(
    card: TechCard,
    entity: Entity | None,
    assertions: list[Assertion],
    segments: list[Segment],
    existing_markdown: str | None = None,
) -> str:
    frontmatter = _build_frontmatter(card, entity, assertions, segments)
    evidence_map = _build_evidence_index(assertions, segments)
    key_points = _format_key_points(card, evidence_map)
    comparisons = card.comparisons or []
    open_questions = _derive_open_questions(assertions)
    change_log = _merge_change_log(existing_markdown, _build_change_entry(assertions))

    body_lines = [
        "---",
        _dump_yaml(frontmatter).strip(),
        "---",
        "",
        "## Summary",
        card.short_definition.strip() or "(No summary yet)",
        "",
        "## Key Points",
    ]
    if key_points:
        body_lines.extend([f"- {line}" for line in key_points])
    else:
        body_lines.append("- (No key points yet)")

    body_lines.append("")
    body_lines.append("## Comparisons")
    if comparisons:
        body_lines.extend([f"- {item}" for item in comparisons])
    else:
        body_lines.append("- (No comparisons yet)")

    body_lines.append("")
    body_lines.append("## Recent Summary")
    body_lines.append(card.recent_summary.strip() or "(No recent summary yet)")

    body_lines.append("")
    body_lines.append("## Evidence Index")
    if evidence_map:
        for entry in evidence_map.values():
            body_lines.append(f"- {entry['label']}")
            body_lines.append(f"  - Quote: \"{entry['quote']}\"")
    else:
        body_lines.append("- (No evidence yet)")

    body_lines.append("")
    body_lines.append("## Related Chunks")
    related = _related_chunks(card)
    if related:
        for entry in related:
            body_lines.append(f"- {entry}")
    else:
        body_lines.append("- (No related chunks found)")

    body_lines.append("")
    body_lines.append("## Open Questions")
    if open_questions:
        body_lines.extend([f"- {item}" for item in open_questions])
    else:
        body_lines.append("- (None)")

    body_lines.append("")
    body_lines.append("## Change Log")
    body_lines.extend(change_log)

    return "\n".join(body_lines).strip() + "\n"


def write_card_markdown(
    card: TechCard,
    entity: Entity | None,
    assertions: list[Assertion],
    segments: list[Segment],
    out_dir: str,
) -> str:
    slug = slugify(entity.canonical_name if entity else str(card.entity_id))
    path = Path(out_dir) / f"{slug}.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else None
    rendered = render_tech_card_markdown(card, entity, assertions, segments, existing)
    if existing:
        rendered = _merge_markdown(existing, rendered)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    return str(path)


def export_cards(out_dir: str, entity_id: int | None = None) -> list[str]:
    from .storage import fetch_cards

    entities = fetch_entities()
    entity_map = {entity.id: entity for entity in entities if entity.id is not None}
    cards = fetch_cards()
    card_map = {card.entity_id: card for card in cards}
    if entity_id is not None:
        entity_map = {entity_id: entity_map.get(entity_id)}
        card_map = {entity_id: card_map.get(entity_id)}
    paths: list[str] = []
    for eid, entity in entity_map.items():
        if eid is None:
            continue
        card = card_map.get(eid)
        if not card:
            continue
        assertions = fetch_assertions_for_entity(eid)
        segment_ids = sorted({sid for assertion in assertions for sid in assertion.segment_ids})
        segments = fetch_segments_by_ids(segment_ids)
        paths.append(write_card_markdown(card, entity, assertions, segments, out_dir))
    return paths


def _build_frontmatter(
    card: TechCard,
    entity: Entity | None,
    assertions: list[Assertion],
    segments: list[Segment],
) -> dict[str, Any]:
    canonical = entity.canonical_name if entity else str(card.entity_id)
    aliases = entity.aliases if entity else []
    entity_type = entity.type if entity else "unknown"
    first_seen = _infer_first_seen(entity, assertions, segments)
    confidence = _infer_confidence(assertions)
    return {
        "entity_id": card.entity_id,
        "canonical_name": canonical,
        "entity_type": entity_type or "unknown",
        "aliases": aliases or [],
        "first_seen": first_seen,
        "last_updated": dt.date.today().isoformat(),
        "confidence_level": confidence,
    }




def _infer_first_seen(
    entity: Entity | None,
    assertions: list[Assertion],
    segments: list[Segment],
) -> str | None:
    episode_id = entity.first_seen_episode_id if entity else None
    if not episode_id:
        return None
    seg_times = [seg.t_start_sec for seg in segments if seg.t_start_sec is not None]
    if not seg_times:
        return f"episode {episode_id}"
    timestamp = seconds_to_hms(min(seg_times))
    return f"episode {episode_id} @ {timestamp}"


def _infer_confidence(assertions: list[Assertion]) -> str:
    count = len(assertions)
    if count >= 3:
        return "high"
    if count == 0:
        return "low"
    return "medium"


def _build_evidence_index(
    assertions: list[Assertion],
    segments: list[Segment],
) -> dict[int, dict[str, str]]:
    segment_lookup = {seg.id: seg for seg in segments if seg.id is not None}
    evidence: dict[int, dict[str, str]] = {}
    for assertion in assertions:
        for seg_id in assertion.segment_ids:
            seg = segment_lookup.get(seg_id)
            if not seg:
                continue
            timestamp = seconds_to_hms(seg.t_start_sec)
            link = f" ({seg.youtube_url})" if seg.youtube_url else ""
            label = f"{seg.speaker or 'Unknown'} @ {timestamp}{link}"
            evidence[seg_id] = {
                "label": label,
                "quote": take_quote(assertion.evidence_quote or seg.text),
            }
    return evidence


def _format_key_points(card: TechCard, evidence_map: dict[int, dict[str, str]]) -> list[str]:
    key_points = card.key_points or []
    if not key_points:
        return []
    if not evidence_map:
        return key_points
    evidence_labels = list(evidence_map.values())
    rendered: list[str] = []
    for idx, point in enumerate(key_points):
        cite = evidence_labels[min(idx, len(evidence_labels) - 1)]["label"]
        rendered.append(f"{point} — {cite}")
    return rendered


def _derive_open_questions(assertions: list[Assertion]) -> list[str]:
    questions = []
    for assertion in assertions:
        if assertion.assertion_type == "prediction" or assertion.verify_priority >= 2:
            questions.append(f"Validate: {assertion.statement}")
    return questions


def _build_change_entry(assertions: list[Assertion]) -> str:
    episode_ids = sorted({assertion.episode_id for assertion in assertions})
    today = dt.date.today().isoformat()
    if episode_ids:
        return f"- {today}: updated from episodes {', '.join(str(i) for i in episode_ids)}"
    return f"- {today}: updated"


def _related_chunks(card: TechCard) -> list[str]:
    try:
        settings = get_settings()
        query_text = " ".join(
            [
                card.short_definition or "",
                " ".join(card.key_points or []),
                " ".join(card.comparisons or []),
                card.recent_summary or "",
            ]
        ).strip()
        if not query_text:
            return []
        vec = embed_query(query_text)
        results = similarity_search_chunks(
            vec, top_k=3, model_name=settings.embedding_model, dims=settings.embedding_dims
        )
        lines = []
        for chunk, _distance in results:
            label = f"Topic {chunk.topic_id} @ {seconds_to_hms(chunk.t_start_sec)}–{seconds_to_hms(chunk.t_end_sec)}"
            lines.append(f"{label}: {chunk.chunk_text}")
        return lines
    except Exception:
        return []


def _merge_change_log(existing_markdown: str | None, new_entry: str) -> list[str]:
    existing_entries = []
    if existing_markdown:
        _extract_frontmatter(existing_markdown)
        existing_entries = _extract_change_log(existing_markdown)
    entries = list(existing_entries)
    if new_entry not in entries:
        entries.append(new_entry)
    return entries if entries else [new_entry]


def _extract_change_log(markdown: str) -> list[str]:
    lines = markdown.splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if line.strip().lower() == "## change log":
            start_idx = idx + 1
            break
    if start_idx is None:
        return []
    entries = []
    for line in lines[start_idx:]:
        if line.startswith("## "):
            break
        if line.strip().startswith("-"):
            entries.append(line.strip())
    return entries


def _merge_markdown(existing: str, generated: str) -> str:
    existing_sections = _split_sections(existing)
    generated_sections = _split_sections(generated)
    merged_sections = []
    used = set()

    for title, content in generated_sections:
        merged_sections.append((title, content))
        used.add(title)

    for title, content in existing_sections:
        if title in used:
            continue
        merged_sections.append((title, content))

    frontmatter = _extract_frontmatter(generated)
    preamble = _extract_preamble(existing)
    lines = ["---", _dump_yaml(frontmatter).strip(), "---", ""]
    if preamble:
        lines.append(preamble.strip())
        lines.append("")
    for title, content in merged_sections:
        lines.append(f"## {title}")
        lines.append(content.rstrip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _split_sections(markdown: str) -> list[tuple[str, str]]:
    content = _strip_frontmatter(markdown)
    lines = content.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_title = None
    current_lines: list[str] = []
    for line in lines:
        if line.startswith("## "):
            if current_title is not None:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = line[3:].strip()
            current_lines = []
            continue
        if current_title is None:
            continue
        current_lines.append(line)
    if current_title is not None:
        sections.append((current_title, "\n".join(current_lines).strip()))
    return sections


def _extract_preamble(markdown: str) -> str:
    content = _strip_frontmatter(markdown)
    lines = content.splitlines()
    preamble_lines = []
    for line in lines:
        if line.startswith("## "):
            break
        preamble_lines.append(line)
    return "\n".join(preamble_lines).strip()


def _strip_frontmatter(markdown: str) -> str:
    if not markdown.startswith("---"):
        return markdown
    parts = markdown.split("---", 2)
    if len(parts) < 3:
        return markdown
    return parts[2].lstrip()


def _extract_frontmatter(markdown: str) -> dict[str, Any]:
    if not markdown.startswith("---"):
        return {}
    parts = markdown.split("---", 2)
    if len(parts) < 3:
        return {}
    raw = parts[1].strip()
    if not raw:
        return {}
    if yaml:
        try:
            return yaml.safe_load(raw) or {}
        except Exception:
            return {}
    return _yaml_fallback_parse(raw)


def _dump_yaml(data: dict[str, Any]) -> str:
    if yaml:
        return yaml.safe_dump(data, sort_keys=False).strip()
    return "\n".join(_yaml_fallback_lines(data))


def _yaml_fallback_lines(data: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{key}: {value if value is not None else ''}")
    return lines


def _yaml_fallback_parse(raw: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    current_key = None
    for line in raw.splitlines():
        if not line.strip():
            continue
        if re.match(r"^[A-Za-z0-9_\\-]+:", line):
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if value == "":
                result[key] = []
                current_key = key
            else:
                result[key] = value
                current_key = None
            continue
        if current_key and line.strip().startswith("-"):
            item = line.strip()[1:].strip()
            result.setdefault(current_key, []).append(item)
    return result
