"""Project-wide custom exception hierarchy."""
from __future__ import annotations

__all__ = ["SocialTownError", "LLMError"]


class SocialTownError(Exception):
    """Base exception for all Social Town errors."""


class LLMError(SocialTownError):
    """Raised when an LLM call fails without a fallback."""
