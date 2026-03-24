from __future__ import annotations
import pytest
from unittest.mock import AsyncMock
from core.simulation import Simulation

def make_mock_llm():
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value="Morning|Rest|Home\nAfternoon|Rest|Home\nEvening|Rest|Home\nNight|Sleep|Home")
    llm.score_importance = AsyncMock(return_value=3.0)
    llm.embed = AsyncMock(return_value=[0.1]*384)
    return llm

@pytest.mark.asyncio
async def test_simulation_runs_50_steps_no_crash():
    sim = Simulation(n_agents=5, llm=make_mock_llm(), steps_per_day=10)
    for _ in range(50):
        await sim.step()
    assert sim.current_step == 50

@pytest.mark.asyncio
async def test_simulation_memories_accumulate():
    sim = Simulation(n_agents=3, llm=make_mock_llm(), steps_per_day=10)
    for _ in range(10):
        await sim.step()
    total_memories = sum(len(p.memory.all()) for p in sim.personas)
    assert total_memories >= 10

@pytest.mark.asyncio
async def test_simulation_snapshot_and_resume():
    import json, tempfile, os
    sim = Simulation(n_agents=3, llm=make_mock_llm(), steps_per_day=10)
    for _ in range(20):
        await sim.step()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sim.to_snapshot(), f)
        snap_path = f.name
    try:
        sim2 = Simulation.from_snapshot(snap_path, llm=make_mock_llm())
        assert sim2.current_step == 20
        await sim2.step()
        assert sim2.current_step == 21
    finally:
        os.unlink(snap_path)
