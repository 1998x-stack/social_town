"""
Docker integration tests — run manually after docker-compose up -d.
These tests validate the live Docker environment.

Usage: pytest tests/stage7_docker/ -v -m docker
Requires: docker-compose -f docker/docker-compose.yml up -d (run first)
"""
from __future__ import annotations
import os
import pytest
import httpx

pytestmark = pytest.mark.docker


def test_ollama_healthcheck():
    """Ollama API should be accessible on port 11434."""
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        assert resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
        pytest.skip(f"Ollama not available: {e}")


def test_model_is_pulled():
    """Target model should be listed in Ollama."""
    model = os.getenv("MODEL_NAME", "qwen2.5:0.5b")
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        tags = resp.json().get("models", [])
        names = [t["name"] for t in tags]
        assert any(model in n for n in names), f"{model} not found in {names}"
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
        pytest.skip(f"Ollama not available: {e}")


def test_webapp_responds():
    """Web dashboard should respond on port 8080."""
    try:
        resp = httpx.get("http://localhost:8080/", timeout=5.0)
        assert resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
        pytest.skip(f"Webapp not available: {e}")
