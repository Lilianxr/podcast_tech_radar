from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Type

from langchain_core.messages import HumanMessage, SystemMessage

from .config import get_settings
from .schemas import CardResult, EntityResult, AssertionResult, TopicResult
from .utils import take_quote


def get_chat_model():
    settings = get_settings()
    if not settings.openai_api_key and not settings.stub_llm:
        raise RuntimeError("OPENAI_API_KEY is required unless TECH_RADAR_STUB_LLM=true")
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=settings.model, temperature=settings.temperature)


def call_json(task: str, system_prompt: str, user_prompt: str, schema: Type[Any], context: dict) -> Any:
    settings = get_settings()
    if settings.stub_llm:
        return _stub(task, context, schema)

    model = get_chat_model()
    last_raw = None
    for _attempt in range(3):
        try:
            structured = model.with_structured_output(schema, method="function_calling")
            return structured.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        except Exception:
            message = model.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
            raw = message.content
            last_raw = raw
            cleaned = _strip_code_fences(raw)
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                continue
            data = _normalize_payload(task, data, context)
            return schema.model_validate(data)
    raise ValueError(f"LLM returned non-JSON for {task}: {last_raw}")


def _stub(task: str, context: dict, schema: Type[Any]) -> Any:
    if task == "topics":
        segments = context.get("segments", [])
        summary = segments[0].text if segments else "Short discussion."
        return TopicResult(topics=[{"name": "Overview", "summary": summary, "start_seg_id": None, "end_seg_id": None}])
    if task == "entities":
        return EntityResult(entities=[{"type": "concept", "canonical_name": "AI", "aliases": ["artificial intelligence"]}])
    if task == "assertions":
        segments = context.get("segments", [])
        if not segments:
            return AssertionResult(assertions=[])
        seg = segments[0]
        return AssertionResult(
            assertions=[
                {
                    "episode_id": context.get("episode_id", 0),
                    "entity_id": None,
                    "assertion_type": "opinion",
                    "statement": take_quote(seg.text, 200),
                    "speaker": seg.speaker or "Unknown",
                    "confidence": 0.5,
                    "verify_priority": 1,
                    "segment_ids": [seg.id or 0],
                    "evidence_quote": take_quote(seg.text, 200),
                }
            ]
        )
    if task == "cards":
        entities = context.get("entities", [])
        cards = []
        for entity in entities:
            cards.append(
                {
                    "entity_id": entity.id or 0,
                    "short_definition": f"{entity.canonical_name} is a key concept discussed.",
                    "key_points": ["Appears in the transcript"],
                    "comparisons": [],
                    "recent_summary": "No recent updates in this episode.",
                }
            )
        return CardResult(cards=cards, needs_refine=False)
    raise ValueError(f"Unknown stub task {task}")


def _strip_code_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return text


def _normalize_payload(task: str, data: Any, context: dict) -> Any:
    if task == "topics" and isinstance(data, list):
        return {"topics": data}
    if task == "entities" and isinstance(data, list):
        return {"entities": data}
    if task == "assertions" and isinstance(data, list):
        return {"assertions": data}
    if task == "cards" and isinstance(data, list):
        return {"cards": data, "needs_refine": False}
    if task == "query_optimizer":
        if isinstance(data, dict):
            time_hint = data.get("time_hint")
            if isinstance(time_hint, str):
                data["time_hint"] = {"value": time_hint}
            if "time_hint" not in data:
                data["time_hint"] = {"value": "none"}
            if "queries" not in data or not isinstance(data["queries"], list) or len(data["queries"]) < 3:
                question = context.get("question", "")
                entities = data.get("entities", []) if isinstance(data.get("entities"), list) else []
                queries = []
                if question:
                    queries.append(question)
                for ent in entities:
                    queries.append(f"{ent} {question}".strip())
                while len(queries) < 3:
                    queries.append(question or "AI podcast transcript")
                data["queries"] = list(dict.fromkeys(queries))[:6]
            if "retrieval_plan" not in data:
                data["retrieval_plan"] = {
                    "use_tech_cards": True,
                    "use_chunks": True,
                    "use_assertions": False,
                    "top_k_chunks": 8,
                    "top_k_cards": 5,
                    "filters": context.get("filters"),
                }
        return data
    return data


def load_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")
