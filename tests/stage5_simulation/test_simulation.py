from __future__ import annotations
import json
import os
import tempfile
import pytest
from core.simulation import Simulation


@pytest.mark.asyncio
async def test_simulation_runs_50_steps_no_crash(mock_llm):
    sim = Simulation(n_agents=5, llm=mock_llm, steps_per_day=10)
    for _ in range(50):
        await sim.step()
    assert sim.current_step == 50


@pytest.mark.asyncio
async def test_simulation_memories_accumulate(mock_llm):
    sim = Simulation(n_agents=3, llm=mock_llm, steps_per_day=10)
    for _ in range(10):
        await sim.step()
    total_memories = sum(len(p.memory.all()) for p in sim.personas)
    assert total_memories >= 10


@pytest.mark.asyncio
async def test_simulation_snapshot_and_resume(mock_llm):
    sim = Simulation(n_agents=3, llm=mock_llm, steps_per_day=10)
    for _ in range(20):
        await sim.step()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sim.to_snapshot(), f)
        snap_path = f.name
    try:
        sim2 = Simulation.from_snapshot(snap_path, llm=mock_llm)
        assert sim2.current_step == 20
        await sim2.step()
        assert sim2.current_step == 21
    finally:
        os.unlink(snap_path)
