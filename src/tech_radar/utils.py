from __future__ import annotations

import hashlib
import re
from typing import Iterable


def hash_text(value: str) -> str:
    return hashlib.sha256(value.strip().encode("utf-8")).hexdigest()


def normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).lower()


def to_seconds(hms: str) -> int:
    parts = [int(p) for p in hms.split(":")]
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    else:
        h, m, s = 0, 0, parts[0]
    return h * 3600 + m * 60 + s


def seconds_to_hms(seconds: int | None) -> str:
    if seconds is None:
        return "00:00:00"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def build_youtube_url(base: str | None, seconds: int | None) -> str | None:
    if not base or seconds is None:
        return None
    if "?" in base:
        return f"{base}&t={seconds}s"
    return f"{base}?t={seconds}s"


def compact_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def slugify(value: str) -> str:
    normalized = normalize_name(value)
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    return normalized.strip("-") or "unknown-entity"


def expand_query(question: str) -> str:
    base = compact_spaces(question.lower())
    synonyms = {
        "ai": ["artificial intelligence", "llm", "model"],
        "agent": ["agents", "agentic", "autonomous agent", "autonomy", "tool use"],
        "china": ["chinese", "prc", "china ai", "china labs"],
        "gpu": ["gpus", "graphics processor", "hardware"],
        "cost": ["inference cost", "price per token", "compute cost"],
    }
    tokens = re.findall(r"[a-z0-9]+", base)
    expanded = [base]
    for token in tokens:
        if token in synonyms:
            expanded.extend(synonyms[token])
    return " | ".join(dict.fromkeys(expanded))


def take_quote(text: str, limit: int = 240) -> str:
    trimmed = compact_spaces(text)
    if len(trimmed) <= limit:
        return trimmed
    return trimmed[: max(0, limit - 3)].rstrip() + "..."


def first_sentence(text: str, limit: int = 240) -> str:
    trimmed = compact_spaces(text)
    if not trimmed:
        return ""
    for sep in [". ", "? ", "! "]:
        parts = trimmed.split(sep)
        if len(parts) > 1:
            sentence = parts[0] + sep.strip()
            return sentence if len(sentence) <= limit else take_quote(sentence, limit)
    return take_quote(trimmed, limit)


def chunk_lines(lines: Iterable[str], max_chars: int = 1200) -> list[str]:
    chunks: list[str] = []
    buffer: list[str] = []
    count = 0
    for line in lines:
        if count + len(line) > max_chars and buffer:
            chunks.append("\n".join(buffer))
            buffer = []
            count = 0
        buffer.append(line)
        count += len(line)
    if buffer:
        chunks.append("\n".join(buffer))
    return chunks


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)
