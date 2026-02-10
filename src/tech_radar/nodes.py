from __future__ import annotations

from typing import TypedDict
import json

from langgraph.graph import END

from .llm import call_json
from .parsers.generic import GenericTextParser
from .parsers.lex import LexTranscriptParser
from .schemas import Assertion as AssertionSchema
from .schemas import AssertionResult, CardResult, EntityResult, TopicResult, QueryOptimizationResult
from .schemas import Segment as SegmentSchema
from .schemas import TranscriptParseResult
from .storage import (
    fetch_episode_segments,
    fetch_segment_ids_for_chunk,
    fetch_segments_by_ids,
    upsert_assertions,
    upsert_cards,
    upsert_entities,
    upsert_episode,
    upsert_segments,
    upsert_topics,
)
from .utils import take_quote


class GraphState(TypedDict, total=False):
    source_url: str
    file_texts: list[str]
    episode_id: int
    episode_title: str | None
    segments: list[SegmentSchema]
    topics: list
    chunks: list
    entities: list
    assertions: list
    cards: list
    refine_count: int
    needs_refine: bool
    question: str
    mode: str
    answer: dict
    optimized: QueryOptimizationResult | None
    debug: bool


def ingest_url(state: GraphState) -> GraphState:
    return state


def ingest_files(state: GraphState) -> GraphState:
    return state


def parse_and_segment(state: GraphState) -> GraphState:
    if state.get("source_url"):
        result = LexTranscriptParser().parse(state["source_url"])
    else:
        texts = state.get("file_texts", [])
        merged = "\n\n".join(texts)
        result = GenericTextParser().parse(merged)
    episode = upsert_episode(result.episode)
    stored_segments = upsert_segments(episode.id, result.segments)
    return {
        **state,
        "episode_id": episode.id,
        "episode_title": episode.title,
        "segments": [
            SegmentSchema(
                id=seg.id,
                episode_id=seg.episode_id,
                speaker=seg.speaker,
                t_start_sec=seg.t_start_sec,
                t_end_sec=seg.t_end_sec,
                youtube_url=seg.youtube_url,
                text=seg.text,
                hash=seg.hash,
            )
            for seg in stored_segments
        ],
    }


def topic_threader(state: GraphState) -> GraphState:
    segments = state["segments"]
    lines = [f"{seg.id} | {seg.speaker or 'Unknown'} | {seg.text}" for seg in segments]
    system = (
        "You cluster transcript segments into topics. Return JSON with topics: "
        "[{name, summary, start_seg_id, end_seg_id}]."
    )
    user = "\n".join(lines)
    result: TopicResult = call_json("topics", system, user, TopicResult, {"segments": segments})
    stored = upsert_topics(state["episode_id"], result.topics)
    return {**state, "topics": stored}


def entity_extractor(state: GraphState) -> GraphState:
    segments = state["segments"]
    lines = [f"{seg.id} | {seg.speaker or 'Unknown'} | {seg.text}" for seg in segments]
    system = (
        "Extract technology entities and return JSON {entities:[{type, canonical_name, aliases}]}. "
        "Entity types: model, company, framework, hardware, benchmark, paper, product, concept."
    )
    user = "\n".join(lines)
    result: EntityResult = call_json("entities", system, user, EntityResult, {"segments": segments})
    stored = upsert_entities(state["episode_id"], result.entities)
    return {**state, "entities": stored}


def assertion_extractor(state: GraphState) -> GraphState:
    segments = state["segments"]
    lines = [f"{seg.id} | {seg.speaker or 'Unknown'} | {seg.text}" for seg in segments]
    system = (
        "Extract assertions with evidence. Return JSON {assertions:[{episode_id, entity_id, "
        "assertion_type, statement, speaker, confidence, verify_priority, segment_ids, evidence_quote}]}. "
        "evidence_quote must be <= 240 chars and copied from the transcript."
    )
    user = "\n".join(lines)
    result: AssertionResult = call_json(
        "assertions",
        system,
        user,
        AssertionResult,
        {"segments": segments, "episode_id": state["episode_id"]},
    )
    seg_lookup = {seg.id: seg for seg in segments if seg.id is not None}
    assertions: list[AssertionSchema] = []
    for assertion in result.assertions:
        assertion.episode_id = state["episode_id"]
        assertion.segment_ids = [sid for sid in assertion.segment_ids if sid in seg_lookup]
        if not assertion.segment_ids:
            continue
        quote = assertion.evidence_quote or ""
        if not quote or len(quote) > 240:
            first_seg = seg_lookup.get(assertion.segment_ids[0])
            if first_seg:
                quote = take_quote(first_seg.text, 240)
            else:
                quote = take_quote(assertion.statement, 240)
        assertion.evidence_quote = quote
        assertions.append(assertion)
    stored = upsert_assertions(assertions)
    return {**state, "assertions": stored}


def card_upserter(state: GraphState) -> GraphState:
    entities = state.get("entities", [])
    assertions = state.get("assertions", [])
    system = (
        "Build tech cards. Return JSON {cards:[{entity_id, short_definition, key_points, "
        "comparisons, recent_summary}], needs_refine}."
    )
    user = f"Entities: {[(e.id, e.canonical_name, e.type) for e in entities]}\n"
    user += f"Assertions: {[(a.entity_id, a.statement) for a in assertions]}"
    result: CardResult = call_json(
        "cards", system, user, CardResult, {"entities": entities, "assertions": assertions}
    )
    if result.needs_refine:
        return {**state, "needs_refine": True, "refine_count": state.get("refine_count", 0) + 1}
    stored = upsert_cards(result.cards)
    needs_refine = any(not card.short_definition for card in result.cards)
    return {
        **state,
        "cards": stored,
        "needs_refine": needs_refine,
        "refine_count": state.get("refine_count", 0),
    }


def indexer(state: GraphState) -> GraphState:
    from .embeddings import embed_chunks
    from .storage import fetch_chunks_for_episode

    chunks = state.get("chunks") or fetch_chunks_for_episode(state["episode_id"])
    embed_chunks(chunks)
    return state


def qa_chain(state: GraphState) -> GraphState:
    question = state["question"]
    mode = state.get("mode", "fast")
    optimized = state.get("optimized")
    segments = state.get("segments") or fetch_episode_segments(state["episode_id"])
    hits = []
    try:
        from .embeddings import embed_query
        from .storage import similarity_search_chunks
        from .config import get_settings

        settings = get_settings()
        queries = [question]
        top_k = 8
        if optimized and optimized.queries:
            queries = optimized.queries
            top_k = optimized.retrieval_plan.top_k_chunks
        chunk_scores: dict[int, float] = {}
        for q in queries:
            query_vec = embed_query(q)
            chunk_results = similarity_search_chunks(
                query_vec,
                top_k=top_k,
                episode_id=state.get("episode_id"),
                model_name=settings.embedding_model,
                dims=settings.embedding_dims,
            )
            for chunk, distance in chunk_results:
                score = 1.0 / (1.0 + distance)
                chunk_scores[chunk.id] = max(chunk_scores.get(chunk.id, 0.0), score)
        chunk_ids = sorted(chunk_scores.keys(), key=lambda cid: chunk_scores[cid], reverse=True)
        chunk_segment_ids = []
        for chunk_id in chunk_ids:
            chunk_segment_ids.extend(fetch_segment_ids_for_chunk(chunk_id))
        chunk_segment_ids = list(dict.fromkeys(chunk_segment_ids))
        hits = fetch_segments_by_ids(chunk_segment_ids) if chunk_segment_ids else []
    except Exception:
        hits = []
    if not hits:
        for seg in segments:
            if question.lower() in seg.text.lower():
                hits.append(seg)
        if not hits:
            hits = segments[:3]
    citations = [
        {
            "segment_id": seg.id,
            "speaker": seg.speaker,
            "timestamp": seg.t_start_sec,
            "youtube_url": seg.youtube_url,
            "quote": take_quote(seg.text, 360),
        }
        for seg in hits[:8]
    ]
    context_lines = [
        f"{c['speaker']} @ {c['timestamp']}s: {c['quote']}" for c in citations if c["quote"]
    ]
    answer_text = "\n".join([c["quote"] for c in citations])
    try:
        from .llm import get_chat_model
        from langchain_core.messages import HumanMessage, SystemMessage

        if context_lines:
            model = get_chat_model()
            system = (
                "Answer the question using only the provided transcript snippets. "
                "If the answer is not in the snippets, say you don't know."
            )
            user = f"Question: {question}\n\nSnippets:\n" + "\n".join(context_lines)
            message = model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
            answer_text = message.content.strip() or answer_text
    except Exception:
        pass
    if answer_text.strip().lower() in {"i don't know", "i don't know.", "i do not know"}:
        answer_text = "Based on retrieved snippets:\n" + "\n".join([c["quote"] for c in citations])
    if mode == "verify" and any(token in question.lower() for token in ["benchmark", "cost", "numbers"]):
        answer_text += "\n\nNote: verification mode is not yet implemented in this phase."
    return {**state, "answer": {"answer": answer_text, "citations": citations}}


def should_refine(state: GraphState) -> str:
    if state.get("needs_refine") and state.get("refine_count", 0) < 2:
        return "refine"
    return END


def optimize_query_node(state: GraphState) -> GraphState:
    from .llm import load_prompt
    from pathlib import Path

    prompt_path = Path(__file__).resolve().parent / "prompts" / "query_optimizer.txt"
    system = load_prompt(str(prompt_path))
    user = {
        "question": state["question"],
        "filters": {
            "episode_id": state.get("episode_id"),
        },
    }
    result: QueryOptimizationResult = call_json(
        "query_optimizer", system, json.dumps(user, ensure_ascii=False), QueryOptimizationResult, user
    )
    return {**state, "optimized": result}
