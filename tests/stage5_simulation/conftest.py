"""Shared fixtures for Stage 5 simulation tests."""
from __future__ import annotations
import pytest
from unittest.mock import AsyncMock
from agents.persona import Persona
from core.simulation import Simulation

__all__ = ["mock_llm", "persona", "simulation"]


@pytest.fixture
def mock_llm() -> AsyncMock:
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value="Morning|Rest|Home\nAfternoon|Rest|Home\nEvening|Rest|Home\nNight|Sleep|Home")
    llm.score_importance = AsyncMock(return_value=3.0)
    llm.embed = AsyncMock(return_value=[0.1] * 384)
    return llm


@pytest.fixture
def persona(mock_llm: AsyncMock) -> Persona:
    return Persona(agent_id="a1", name="Alice", description="A student", llm=mock_llm)


@pytest.fixture
def simulation(mock_llm: AsyncMock) -> Simulation:
    return Simulation(n_agents=3, llm=mock_llm, steps_per_day=10)
