"""Reflection engine: synthesise high-level insights from raw memories."""
from __future__ import annotations
import logging
from typing import Any, Callable
from agents.memory.memory_stream import MemoryObject, MemoryStream
from llm.prompts import reflect_questions_prompt, reflect_insights_prompt

__all__ = ["ReflectionEngine"]

logger = logging.getLogger(__name__)


class ReflectionEngine:
    def __init__(self, llm: Any) -> None:
        self._llm = llm

    async def reflect(
        self,
        memory_stream: MemoryStream,
        embed_fn: Callable[[str], Any],
        current_step: int,
    ) -> list[MemoryObject]:
        recent = memory_stream.recent(100)
        if not recent:
            return []
        mem_text = "\n".join(f"- {m.content}" for m in recent)

        # Step 1: generate questions
        questions_raw = await self._llm.generate(
            reflect_questions_prompt(mem_text),
            fallback="What is this person focused on?\nWhat relationships matter?\nWhat goals do they have?"
        )
        questions = [q.strip() for q in questions_raw.strip().splitlines() if q.strip()][:3]
        logger.info(f"[Reflect] Generated {len(questions)} reflection questions")

        # Step 2: generate insights
        insights_raw = await self._llm.generate(
            reflect_insights_prompt(mem_text),
            fallback="This person values community connections."
        )
        insights = [i.strip() for i in insights_raw.strip().splitlines() if i.strip()][:5]

        # Step 3: store as reflection memories
        reflection_objects = []
        for insight in insights:
            embedding = await embed_fn(insight)
            m = MemoryObject(
                content=insight,
                memory_type="reflection",
                created_at=current_step,
                importance=7.0,
                embedding=embedding,
                source_agent=None,
                credibility=1.0,
            )
            reflection_objects.append(m)
        logger.info(f"[Reflect] Generated {len(reflection_objects)} reflections at step={current_step}")
        return reflection_objects
