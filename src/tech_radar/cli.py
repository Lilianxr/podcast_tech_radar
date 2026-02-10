from __future__ import annotations

import argparse
import json
from pathlib import Path

from .card_markdown import export_cards
from .graph import build_ingest_graph, build_qa_graph
from .reporting import write_report
from .storage import (
    fetch_assertions_for_episode,
    fetch_chunks_for_episode,
    fetch_episode,
    fetch_episode_segments,
    fetch_topics_for_episode,
    similarity_search_chunks,
)
from .embeddings import embed_chunks, embed_query
from .chunking import build_chunks_from_topics
from .storage import upsert_chunks


def _load_file_text(path: Path) -> str:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        segments = None
        if isinstance(payload, dict) and "segments" in payload:
            segments = payload["segments"]
        elif isinstance(payload, list):
            segments = payload
        if segments:
            lines = []
            for seg in segments:
                speaker = seg.get("speaker", "Unknown")
                t = seg.get("timestamp") or seg.get("time") or seg.get("t_start")
                if isinstance(t, str) and ":" in t:
                    timestamp = t
                elif isinstance(t, (int, float)):
                    timestamp = f"00:00:{int(t):02d}"
                else:
                    timestamp = "00:00:00"
                text = seg.get("text", "")
                lines.append(f"{speaker} ({timestamp}): {text}")
            return "\n".join(lines)
        if isinstance(payload, dict) and "text" in payload:
            return str(payload["text"])
        return json.dumps(payload)
    return path.read_text(encoding="utf-8")


def ingest_command(args: argparse.Namespace) -> None:
    graph = build_ingest_graph()
    if args.url:
        state = {"source_url": args.url, "refine_count": 0}
    else:
        texts = [_load_file_text(Path(p)) for p in args.files]
        state = {"file_texts": texts, "refine_count": 0}
    result = graph.invoke(state)
    episode_id = result.get("episode_id")
    episode_title = result.get("episode_title") or "Untitled"
    if episode_id:
        print(f"Ingested episode {episode_id}: {episode_title}")
        print("If you'd like a markdown report, run:")
        print(f"python -m tech_radar report --episode {episode_id} --out out/episode_{episode_id}.md")
        print("Optional JSON:")
        print(
            f"python -m tech_radar report --episode {episode_id} "
            f"--out out/episode_{episode_id}.md --json-out out/episode_{episode_id}.json"
        )


def report_command(args: argparse.Namespace) -> None:
    episode = fetch_episode(args.episode)
    if not episode:
        raise SystemExit(f"Episode {args.episode} not found")
    segments = fetch_episode_segments(args.episode)
    topics = fetch_topics_for_episode(args.episode)
    assertions = fetch_assertions_for_episode(args.episode)
    write_report(episode, segments, topics, assertions, args.out, args.json_out)


def ask_command(args: argparse.Namespace) -> None:
    from .config import get_settings

    settings = get_settings()
    if not settings.openai_api_key:
        print("Warning: OPENAI_API_KEY is not set; vector search will be disabled.")
    graph = build_qa_graph()
    state = {"episode_id": args.episode, "question": args.q, "mode": args.mode, "debug": args.debug}
    result = graph.invoke(state)
    if args.debug and result.get("optimized"):
        print("Query optimizer output:")
        print(result["optimized"].model_dump())
    answer = result["answer"]
    print(answer["answer"])
    print("\nCitations:")
    for cite in answer["citations"]:
        print(f"- {cite['speaker']} @ {cite['timestamp']}s: {cite['quote']}")
        if cite["youtube_url"]:
            print(f"  {cite['youtube_url']}")


def export_cards_command(args: argparse.Namespace) -> None:
    paths = export_cards(args.out, entity_id=args.entity_id)
    if not paths:
        print("No cards exported.")
        return
    print("Exported cards:")
    for path in paths:
        print(f"- {path}")


def build_chunks_command(args: argparse.Namespace) -> None:
    segments = fetch_episode_segments(args.episode_id)
    topics = fetch_topics_for_episode(args.episode_id)
    chunks = build_chunks_from_topics(topics, segments)
    stored = upsert_chunks(args.episode_id, chunks)
    print(f"Built {len(stored)} chunks for episode {args.episode_id}.")
    embedded = embed_chunks(stored)
    print(f"Embedded {embedded} chunks.")


def search_command(args: argparse.Namespace) -> None:
    query_vec = embed_query(args.q)
    results = similarity_search_chunks(query_vec, top_k=args.top_k, episode_id=args.episode_id)
    if not results:
        print("No chunks found.")
        return
    for chunk, distance in results:
        print(f"- (distance {distance:.4f}) Episode {chunk.episode_id} | Topic {chunk.topic_id}")
        print(chunk.chunk_text.splitlines()[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Podcast Tech Card Library CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Ingest a transcript URL or local files")
    ingest.add_argument("--url", help="Lex transcript URL")
    ingest.add_argument("--files", nargs="*", default=[], help="Local files (.md/.txt/.json)")
    ingest.set_defaults(func=ingest_command)

    report = sub.add_parser("report", help="Generate markdown/json report")
    report.add_argument("--episode", type=int, required=True)
    report.add_argument("--out", required=True)
    report.add_argument("--json-out")
    report.set_defaults(func=report_command)

    ask = sub.add_parser("ask", help="Ask questions over the library")
    ask.add_argument("--episode", type=int, required=True)
    ask.add_argument("--q", required=True)
    ask.add_argument("--mode", choices=["fast", "verify"], default="fast")
    ask.add_argument("--debug", action="store_true")
    ask.set_defaults(func=ask_command)

    export_cards_parser = sub.add_parser("export-cards", help="Export tech cards to markdown")
    export_cards_parser.add_argument("--out", required=True, help="Output directory, e.g. out/cards")
    export_cards_parser.add_argument("--entity-id", type=int, help="Only export a single entity")
    export_cards_parser.set_defaults(func=export_cards_command)

    build_chunks = sub.add_parser("build-chunks", help="Build chunks for an episode")
    build_chunks.add_argument("--episode-id", type=int, required=True)
    build_chunks.set_defaults(func=build_chunks_command)

    search = sub.add_parser("search", help="Vector search chunks")
    search.add_argument("--q", required=True)
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--episode-id", type=int)
    search.set_defaults(func=search_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
