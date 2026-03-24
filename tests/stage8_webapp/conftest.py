"""Shared fixtures for Stage 8 webapp tests."""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock
from fastapi import FastAPI
from webapp.server import create_app

__all__ = ["mock_sim", "app"]


@pytest.fixture
def mock_sim() -> MagicMock:
    sim = MagicMock()
    sim.current_step = 5
    sim.personas = []
    sim.agents = []
    sim.social_graph.edge_count.return_value = 0
    sim.social_graph.density.return_value = 0.0
    return sim


@pytest.fixture
def app(mock_sim: MagicMock) -> FastAPI:
    return create_app(sim=mock_sim)
