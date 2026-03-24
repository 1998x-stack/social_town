"""Shared fixtures for Stage 2 LLM tests."""
from __future__ import annotations
import pytest
from config.params import MODEL_NAME, OLLAMA_HOST, LLM_TIMEOUT
from llm.client import OllamaClient


@pytest.fixture(scope="module")
def client() -> OllamaClient:
    return OllamaClient(host=OLLAMA_HOST, model=MODEL_NAME, timeout=LLM_TIMEOUT)
