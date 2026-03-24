"""Shared fixtures for Stage 6 evaluation tests."""
from __future__ import annotations
import pytest
from evaluation.reporter import Reporter

__all__ = ["reporter"]


@pytest.fixture
def reporter() -> Reporter:
    return Reporter()
