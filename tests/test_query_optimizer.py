from tech_radar.schemas import QueryOptimizationResult


def test_query_optimizer_schema_defaults():
    payload = {
        "intent": "opinion",
        "entities": ["inference cost"],
        "time_hint": {"value": "episode_specific"},
        "queries": ["inference cost", "LLM inference cost", "compute cost"],
        "retrieval_plan": {
            "use_tech_cards": True,
            "use_chunks": True,
            "use_assertions": False,
            "top_k_chunks": 8,
            "top_k_cards": 5,
            "filters": {"episode_id": 1},
        },
        "answer_style": "concise",
    }
    result = QueryOptimizationResult.model_validate(payload)
    assert result.intent in {"opinion", "fact"}
    assert len(result.queries) >= 3


def test_query_optimizer_chinese_question():
    payload = {
        "intent": "comparison",
        "entities": ["DeepSeek", "OpenAI"],
        "time_hint": {"value": "episode_specific"},
        "queries": [
            "DeepSeek inference cost vs OpenAI",
            "inference cost comparison DeepSeek OpenAI",
            "DeepSeek R1 inference cost",
        ],
        "retrieval_plan": {
            "use_tech_cards": True,
            "use_chunks": True,
            "use_assertions": False,
            "top_k_chunks": 8,
            "top_k_cards": 5,
            "filters": {"episode_id": 1},
        },
        "answer_style": "concise",
    }
    result = QueryOptimizationResult.model_validate(payload)
    assert result.intent == "comparison"
    query_text = " ".join(result.queries)
    assert "inference cost" in query_text
    assert "DeepSeek" in query_text
    assert "OpenAI" in query_text
