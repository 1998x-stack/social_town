"""Experimental event injection interface."""
from __future__ import annotations
import logging
from typing import Callable
from agents.memory.memory_stream import MemoryObject, MemoryStream

__all__ = ["EventInjector"]

logger = logging.getLogger(__name__)


class EventInjector:
    def __init__(self, agent_streams: dict[str, MemoryStream]) -> None:
        self._streams = agent_streams

    def inject_event(
        self,
        event_type: str,
        content: str,
        seed_agents: list[str],
        credibility: float,
        step: int,
        embed_fn: Callable[[str], list[float]],
        importance_fn: Callable[[str], float],
    ) -> None:
        for agent_id in seed_agents:
            if agent_id not in self._streams:
                logger.warning(f"[Injector] Agent {agent_id} not found, skipping")
                continue
            importance = importance_fn(content)
            embedding = embed_fn(content)
            memory = MemoryObject(
                content=f"[{event_type}] {content}",
                memory_type="observation",
                created_at=step,
                importance=importance,
                embedding=embedding,
                source_agent=None,
                credibility=credibility,
            )
            self._streams[agent_id].add(memory)
            logger.info(
                f"[Injector] {event_type} injected to {agent_id} (cred={credibility:.2f})"
            )
