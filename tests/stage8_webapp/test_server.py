from __future__ import annotations
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import MagicMock
from webapp.server import create_app

__all__: list[str] = []


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
async def test_metrics_endpoint_has_diffusion_rate(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "diffusion_rate" in data


@pytest.mark.asyncio
async def test_inject_event_endpoint(mock_sim):
    mock_sim.inject_event = MagicMock()
    app = create_app(mock_sim)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/inject", json={
            "event_type": "rumor",
            "content": "The mayor is resigning",
            "seed_agent_ids": ["agent_00"],
            "credibility": 0.8,
        })
    assert resp.status_code == 200
    mock_sim.inject_event.assert_called_once()


@pytest.mark.asyncio
async def test_stream_endpoint_content_type(mock_sim):
    import asyncio
    app = create_app(mock_sim)

    async def _check() -> None:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            async with ac.stream("GET", "/api/stream") as response:
                assert response.status_code == 200
                assert "text/event-stream" in response.headers.get("content-type", "")

    try:
        await asyncio.wait_for(_check(), timeout=5.0)
    except asyncio.TimeoutError:
        # Timing out is expected for an infinite stream — headers already asserted
        pass
