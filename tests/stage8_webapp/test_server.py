from __future__ import annotations
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import MagicMock


@pytest.mark.asyncio
async def test_root_returns_html(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_metrics_endpoint_returns_json(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "step" in data
    assert "network_density" in data


@pytest.mark.asyncio
async def test_inject_event_endpoint(mock_sim, app):
    mock_sim.inject_event = MagicMock()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/inject", json={
            "event_type": "rumor",
            "content": "The mayor is resigning",
            "seed_agent_ids": ["agent_00"],
            "credibility": 0.8,
        })
    assert resp.status_code == 200
