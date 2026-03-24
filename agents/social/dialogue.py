"""Dialogue generation between two agents."""
from __future__ import annotations
import logging
from typing import Any
from llm.prompts import dialogue_prompt

__all__ = ["DialogueEngine"]

logger = logging.getLogger(__name__)


class DialogueEngine:
    def __init__(self, llm: Any) -> None:
        self._llm = llm

    async def generate_utterance(
        self,
        speaker: str,
        listener: str,
        relationship: str,
        context: str,
        topic: str,
    ) -> str:
        prompt = dialogue_prompt(speaker, listener, relationship, context, topic)
        fallback = f"{speaker} nods and continues the conversation."
        try:
            result = await self._llm.generate(prompt, fallback=fallback)
            return result if result.strip() else fallback
        except Exception as exc:
            logger.warning(
                f"[Dialogue] {speaker}->{listener} LLM failed, using fallback: {exc}"
            )
            return fallback
