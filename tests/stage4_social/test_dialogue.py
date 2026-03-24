from __future__ import annotations
import pytest
from unittest.mock import AsyncMock
from agents.social.dialogue import DialogueEngine


@pytest.mark.asyncio
async def test_dialogue_returns_nonempty_string(dialogue_engine):
    utterance = await dialogue_engine.generate_utterance(
        speaker="Alice", listener="Bob",
        relationship="acquaintances",
        context="Alice met Bob at the library",
        topic="the upcoming election",
    )
    assert isinstance(utterance, str)
    assert len(utterance.strip()) >= 5


@pytest.mark.asyncio
async def test_dialogue_fallback_on_error():
    erroring_llm = AsyncMock()
    erroring_llm.generate = AsyncMock(side_effect=Exception("LLM down"))
    engine = DialogueEngine(llm=erroring_llm)
    utterance = await engine.generate_utterance(
        speaker="Alice", listener="Bob",
        relationship="acquaintances", context="", topic="",
    )
    assert utterance != ""
