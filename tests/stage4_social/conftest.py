"""Shared fixtures for Stage 4 social tests."""
from __future__ import annotations
import pytest
from unittest.mock import AsyncMock
from world.social_graph import SocialGraph
from agents.social.dialogue import DialogueEngine
from agents.memory.memory_stream import MemoryStream

__all__ = ["graph", "mock_llm", "dialogue_engine", "memory_stream"]


@pytest.fixture
def graph() -> SocialGraph:
    g = SocialGraph()
    g.add_agent("a1")
    g.add_agent("a2")
    g.add_agent("a3")
    return g


@pytest.fixture
def mock_llm() -> AsyncMock:
    client = AsyncMock()
    client.generate = AsyncMock(return_value="Hello, how are you today?")
    return client


@pytest.fixture
def dialogue_engine(mock_llm: AsyncMock) -> DialogueEngine:
    return DialogueEngine(llm=mock_llm)


@pytest.fixture
def memory_stream() -> MemoryStream:
    return MemoryStream()
