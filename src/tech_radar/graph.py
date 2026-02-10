from __future__ import annotations

from langgraph.graph import END, StateGraph

from .chunk_nodes import chunk_builder, chunk_persist
from .nodes import (
    GraphState,
    assertion_extractor,
    card_upserter,
    entity_extractor,
    ingest_files,
    ingest_url,
    indexer,
    parse_and_segment,
    qa_chain,
    should_refine,
    topic_threader,
    optimize_query_node,
)


def build_ingest_graph() -> StateGraph:
    graph = StateGraph(GraphState)
    graph.add_node("ingest_url", ingest_url)
    graph.add_node("ingest_files", ingest_files)
    graph.add_node("parse", parse_and_segment)
    graph.add_node("topics", topic_threader)
    graph.add_node("chunk_builder", chunk_builder)
    graph.add_node("chunk_persist", chunk_persist)
    graph.add_node("entities", entity_extractor)
    graph.add_node("assertions", assertion_extractor)
    graph.add_node("cards", card_upserter)
    graph.add_node("index", indexer)

    graph.set_entry_point("parse")
    graph.add_edge("parse", "topics")
    graph.add_edge("topics", "chunk_builder")
    graph.add_edge("chunk_builder", "chunk_persist")
    graph.add_edge("chunk_persist", "entities")
    graph.add_edge("entities", "assertions")
    graph.add_edge("assertions", "cards")
    graph.add_conditional_edges("cards", should_refine, {"refine": "assertions", END: "index"})
    graph.add_edge("index", END)

    return graph.compile()


def build_qa_graph() -> StateGraph:
    graph = StateGraph(GraphState)
    graph.add_node("optimize_query", optimize_query_node)
    graph.add_node("qa", qa_chain)
    graph.set_entry_point("optimize_query")
    graph.add_edge("optimize_query", "qa")
    graph.add_edge("qa", END)
    return graph.compile()
