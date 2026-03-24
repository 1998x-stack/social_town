from __future__ import annotations
import pytest
from unittest.mock import AsyncMock
from agents.persona import Persona
from agents.memory.memory_stream import MemoryObject


@pytest.mark.asyncio
async def test_persona_perceive_adds_memory(mock_llm):
    p = Persona(agent_id="a1", name="Alice", description="A student", llm=mock_llm)
    await p.perceive("Bob is reading at the library", current_step=1)
    assert len(p.memory.all()) == 1


@pytest.mark.asyncio
async def test_persona_act_returns_string(mock_llm):
    p = Persona(agent_id="a1", name="Alice", description="A student", llm=mock_llm)
    # conftest mock_llm.generate already returns L1 plan format
    await p.start_day(day=1, current_step=0)
    mock_llm.generate.return_value = "Alice opens her notebook."
    action = await p.act(current_step=1, perceived_top3="Bob is nearby")
    assert isinstance(action, str) and len(action) > 0


@pytest.mark.asyncio
async def test_persona_snapshot_roundtrip(mock_llm):
    p = Persona(agent_id="a1", name="Alice", description="A student", llm=mock_llm)
    await p.perceive("Alice went to the library", current_step=5)
    snap = p.to_snapshot()
    assert snap["id"] == "a1"
    assert snap["memory_count"] == 1
    assert len(snap["memories"]) == 1
