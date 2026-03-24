import json
import math
import pytest
from pathlib import Path
from agents.memory.memory_stream import MemoryObject, MemoryStream
from agents.memory.retrieval import Retriever, score_recency, score_importance, score_relevance

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "retrieval_cases.json"

def make_memory(d: dict, embed_fn) -> MemoryObject:
    m = MemoryObject(
        content=d["content"],
        memory_type="observation",
        created_at=d["created_at"],
        importance=d["importance"],
        embedding=embed_fn(d["content"]),
        source_agent=None,
        credibility=1.0,
    )
    if "id" in d:
        m.id = d["id"]
    return m

def test_score_recency_decay():
    score_recent = score_recency(elapsed_steps=0)
    score_old = score_recency(elapsed_steps=200)
    assert score_recent == pytest.approx(1.0)
    assert score_old < score_recent
    assert 0.0 <= score_old <= 1.0

def test_score_importance_normalized():
    assert score_importance(10.0) == pytest.approx(1.0)
    assert score_importance(1.0) == pytest.approx(0.1)
    assert score_importance(5.0) == pytest.approx(0.5)

def test_score_relevance_range():
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [0.0, 1.0, 0.0]
    vec_same = [1.0, 0.0, 0.0]
    assert score_relevance(vec_a, vec_same) == pytest.approx(1.0, abs=1e-5)
    assert score_relevance(vec_a, vec_b) == pytest.approx(0.0, abs=1e-5)

def test_all_scores_in_unit_interval():
    r = score_recency(50)
    i = score_importance(7.0)
    v = score_relevance([0.5, 0.5], [0.5, 0.5])
    for s in [r, i, v]:
        assert 0.0 <= s <= 1.0

def test_retrieval_fixture_top3_hit_rate():
    """Top-3 hit rate >= 80% on labeled test cases."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embed_fn = lambda text: model.encode(text).tolist()

    with open(FIXTURE_PATH) as f:
        cases = json.load(f)

    retriever = Retriever(embed_fn=embed_fn)
    hits = 0
    total_expected = 0

    for case in cases:
        ms = MemoryStream()
        for d in case["memories"]:
            ms.add(make_memory(d, embed_fn))
        results = retriever.retrieve(ms.all(), query=case["query"], current_step=12, top_k=3)
        result_ids = {m.id for m in results}
        expected_ids = set(case["expected_top3_ids"])
        hits += len(result_ids & expected_ids)
        total_expected += len(expected_ids)

    hit_rate = hits / total_expected
    assert hit_rate >= 0.80, f"Hit rate {hit_rate:.2f} < 0.80"
