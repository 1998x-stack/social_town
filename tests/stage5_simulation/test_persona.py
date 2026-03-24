from __future__ import annotations
import pytest
from unittest.mock import AsyncMock
from agents.persona import Persona
from agents.memory.memory_stream import MemoryObject

def make_mock_llm():
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value="Alice sits quietly and reads.")
    llm.score_importance = AsyncMock(return_value=5.0)
    llm.embed = AsyncMock(return_value=[0.1] * 384)
    return llm

@pytest.mark.asyncio
async def test_persona_perceive_adds_memory():
    llm = make_mock_llm()
    p = Persona(agent_id="a1", name="Alice", description="A student", llm=llm)
    await p.perceive("Bob is reading at the library", current_step=1)
    assert len(p.memory.all()) == 1

@pytest.mark.asyncio
async def test_persona_act_returns_string():
    llm = make_mock_llm()
    p = Persona(agent_id="a1", name="Alice", description="A student", llm=llm)
    llm.generate.return_value = "Morning|Study|Library\nAfternoon|Lunch|Cafe\nEvening|Walk|Park\nNight|Sleep|Home"
    await p.start_day(day=1, current_step=0)
    llm.generate.return_value = "Alice opens her notebook."
    action = await p.act(current_step=1, perceived_top3="Bob is nearby")
    assert isinstance(action, str) and len(action) > 0

@pytest.mark.asyncio
async def test_persona_snapshot_roundtrip():
    llm = make_mock_llm()
    p = Persona(agent_id="a1", name="Alice", description="A student", llm=llm)
    await p.perceive("Alice went to the library", current_step=5)
    snap = p.to_snapshot()
    assert snap["id"] == "a1"
    assert snap["memory_count"] == 1
    assert len(snap["memories"]) == 1
