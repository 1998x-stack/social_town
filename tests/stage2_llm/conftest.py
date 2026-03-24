"""Shared fixtures for Stage 2 LLM tests."""
from __future__ import annotations
import pytest
from llm.client import OllamaClient


@pytest.fixture
def client() -> OllamaClient:
    return OllamaClient(host="http://localhost:11434", model="qwen2.5:0.5b", timeout=5.0)
