"""Shared fixtures for Stage 8 webapp tests."""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock
from fastapi import FastAPI
from webapp.server import create_app

__all__ = ["mock_sim", "mock_sim_with_agents", "app"]


@pytest.fixture
def mock_sim() -> MagicMock:
    sim = MagicMock()
    sim.current_step = 5
    sim.personas = []
    sim.agents = []
    sim.social_graph.edge_count.return_value = 0
    sim.social_graph.density.return_value = 0.0
    sim.diffusion_rate = 0.0
    return sim


@pytest.fixture
def mock_sim_with_agents() -> MagicMock:
    sim = MagicMock()
    sim.current_step = 5
    # Create a mock persona with opinion values
    persona = MagicMock()
    persona.name = "Alice"
    persona.opinion = {"economy": 0.5, "health": -0.3}
    persona.current_action = "reading"
    persona.location = "Library"
    persona.memory.memories = []
    # Set up location
    sim.town.agent_location.return_value = "Library"
    sim.personas = [persona]
    sim.agents = [persona]
    sim._event_aware_agents = set()
    sim.diffusion_rate = 0.0
    sim.social_graph.density.return_value = 0.0
    sim.inject_event = MagicMock()
    return sim


@pytest.fixture
def app(mock_sim: MagicMock) -> FastAPI:
    return create_app(sim=mock_sim)
