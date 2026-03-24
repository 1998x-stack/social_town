"""Memory stream: unified store for observations, reflections, and plans."""
from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from typing import Callable

__all__ = ["MemoryObject", "MemoryStream"]


@dataclass
class MemoryObject:
    content: str
    memory_type: str          # "observation" | "reflection" | "plan"
    created_at: int
    importance: float         # 1.0-10.0
    embedding: list[float]
    source_agent: str | None  # None = direct observation
    credibility: float        # 0.0-1.0
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    last_accessed: int = field(init=False)

    def __post_init__(self) -> None:
        self.last_accessed = self.created_at


class MemoryStream:
    """Append-only memory store with reflection accumulator."""

    def __init__(self) -> None:
        self._memories: list[MemoryObject] = []
        self._reflection_accumulator: float = 0.0

    def add(self, memory: MemoryObject) -> None:
        self._memories.append(memory)
        self._reflection_accumulator += memory.importance

    def all(self) -> list[MemoryObject]:
        return list(self._memories)

    def recent(self, n: int) -> list[MemoryObject]:
        return self._memories[-n:]

    def touch(self, memory_id: str, current_step: int) -> None:
        for m in self._memories:
            if m.id == memory_id:
                m.last_accessed = current_step
                return

    def should_reflect(self, threshold: float) -> bool:
        return self._reflection_accumulator >= threshold

    def reset_reflection_accumulator(self) -> None:
        self._reflection_accumulator = 0.0

    def to_dict_list(self) -> list[dict]:
        """Serialise for snapshot (no embedding — recomputed on load)."""
        return [
            {
                "id": m.id,
                "content": m.content,
                "memory_type": m.memory_type,
                "created_at": m.created_at,
                "last_accessed": m.last_accessed,
                "importance": m.importance,
                "source_agent": m.source_agent,
                "credibility": m.credibility,
            }
            for m in self._memories
        ]

    @classmethod
    def from_dict_list(cls, data: list[dict], embed_fn: Callable[[str], list[float]]) -> "MemoryStream":
        """Restore from snapshot; embed_fn recomputes embeddings."""
        ms = cls()
        for d in data:
            obj = MemoryObject(
                content=d["content"],
                memory_type=d["memory_type"],
                created_at=d["created_at"],
                importance=d["importance"],
                embedding=embed_fn(d["content"]),
                source_agent=d["source_agent"],
                credibility=d["credibility"],
            )
            obj.id = d["id"]
            obj.last_accessed = d["last_accessed"]
            ms.add(obj)
        return ms
