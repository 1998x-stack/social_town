"""Shared fixtures for Stage 3 planning tests."""
from __future__ import annotations
import pytest
from unittest.mock import AsyncMock
from agents.cognitive.plan import PlanningEngine

FAKE_L1 = "Morning|Breakfast|Home\nAfternoon|Study|Library\nEvening|Walk|Park\nNight|Read|Home"


@pytest.fixture
def mock_llm() -> AsyncMock:
    client = AsyncMock()
    client.generate = AsyncMock(return_value=FAKE_L1)
    return client


@pytest.fixture
def engine(mock_llm: AsyncMock) -> PlanningEngine:
    return PlanningEngine(
        name="Alice",
        description="A diligent economics student",
        llm=mock_llm,
    )
