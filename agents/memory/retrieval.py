"""Three-component memory retrieval: recency + importance + relevance."""
from __future__ import annotations
import math
import numpy as np
from agents.memory.memory_stream import MemoryObject

__all__ = ["Retriever", "score_recency", "score_importance", "score_relevance"]

_DECAY_LAMBDA = 0.005


def score_recency(elapsed_steps: int) -> float:
    """Exponential decay; elapsed_steps since last access."""
    return math.exp(-_DECAY_LAMBDA * elapsed_steps)


def score_importance(importance: float) -> float:
    """Normalize 1-10 importance to [0, 1]."""
    return max(0.0, min(1.0, importance / 10.0))


def score_relevance(query_emb: list[float], mem_emb: list[float]) -> float:
    """Cosine similarity clipped to [0, 1]. Orthogonal -> 0.0, identical -> 1.0."""
    a = np.array(query_emb, dtype=float)
    b = np.array(mem_emb, dtype=float)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cosine = float(np.dot(a, b) / (norm_a * norm_b))
    return max(0.0, min(1.0, cosine))


class Retriever:
    def __init__(self, embed_fn) -> None:
        self._embed = embed_fn

    def retrieve(
        self,
        memories: list[MemoryObject],
        query: str,
        current_step: int,
        top_k: int = 10,
    ) -> list[MemoryObject]:
        if not memories:
            return []
        query_emb = self._embed(query)
        scored = []
        for m in memories:
            r = score_recency(current_step - m.last_accessed)
            i = score_importance(m.importance)
            v = score_relevance(query_emb, m.embedding)
            scored.append((r + i + v, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]
