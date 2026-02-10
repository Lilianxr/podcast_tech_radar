from __future__ import annotations

import os
from pathlib import Path

from .graph import build_ingest_graph, build_qa_graph
from .migrations import run_init


def main() -> None:
    os.environ["TECH_RADAR_STUB_LLM"] = "true"
    run_init()

    sample = Path(__file__).resolve().parents[2] / "examples" / "sample_transcript.txt"
    text = sample.read_text(encoding="utf-8")

    ingest_graph = build_ingest_graph()
    ingest_graph.invoke({"file_texts": [text], "refine_count": 0})

    qa_graph = build_qa_graph()
    result = qa_graph.invoke({"episode_id": 1, "question": "What did they say about cost?"})
    answer = result["answer"]

    print("Answer:")
    print(answer["answer"])
    print("\nCitations:")
    for cite in answer["citations"]:
        print(f"- {cite['speaker']} @ {cite['timestamp']}s: {cite['quote']}")


if __name__ == "__main__":
    main()
