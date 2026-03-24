"""Ollama LLM client with timeout protection and fallback."""
from __future__ import annotations
import inspect
import re
from typing import Optional
import httpx
from sentence_transformers import SentenceTransformer
from config.params import OLLAMA_HOST, MODEL_NAME, LLM_TIMEOUT, EMBEDDING_MODEL

__all__ = ["OllamaClient", "LLMError"]


class LLMError(Exception):
    pass


class OllamaClient:
    def __init__(
        self,
        host: str = OLLAMA_HOST,
        model: str = MODEL_NAME,
        timeout: float = LLM_TIMEOUT,
    ) -> None:
        self._host = host
        self._model = model
        self._timeout = timeout
        self._st_model = SentenceTransformer(EMBEDDING_MODEL)

    @staticmethod
    async def _parse_json(resp: httpx.Response) -> dict:
        """Call resp.json(), awaiting if necessary (handles AsyncMock in tests)."""
        result = resp.json()
        if inspect.isawaitable(result):
            result = await result
        return result

    async def generate(
        self,
        prompt: str,
        fallback: Optional[str] = None,
    ) -> str:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._host}/api/generate",
                    json={"model": self._model, "prompt": prompt, "stream": False},
                )
                resp.raise_for_status()
                data = await self._parse_json(resp)
                return data["response"].strip()
        except Exception as exc:
            if fallback is not None:
                return fallback
            raise LLMError(str(exc)) from exc

    async def score_importance(self, content: str) -> float:
        """Ask LLM to rate importance 1-10; fallback=5.0 on parse failure."""
        from llm.prompts import importance_prompt
        prompt = importance_prompt(content)
        try:
            raw = await self.generate(prompt, fallback="5")
            numbers = re.findall(r"\b([1-9]|10)\b", raw)
            if numbers:
                return float(numbers[0])
            return 5.0
        except LLMError:
            return 5.0

    async def embed(self, text: str) -> list[float]:
        """Local sentence-transformers embedding (no network call)."""
        return self._st_model.encode(text).tolist()
