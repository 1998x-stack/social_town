"""Ollama LLM client with timeout protection and fallback."""
from __future__ import annotations
import inspect
import logging
import re
from typing import Optional
import httpx
from sentence_transformers import SentenceTransformer
from config.params import OLLAMA_HOST, MODEL_NAME, LLM_TIMEOUT, EMBEDDING_MODEL
from core.errors import LLMError

__all__ = ["OllamaClient", "LLMError"]

logger = logging.getLogger(__name__)


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
                data = resp.json()
                if inspect.isawaitable(data):
                    data = await data
                return data["response"].strip()
        except httpx.TimeoutException as exc:
            if fallback is not None:
                logger.warning(f"[LLM] 调用超时，使用 fallback: {fallback}")
                return fallback
            logger.error(f"[LLM] 调用超时: {exc}")
            raise LLMError(str(exc)) from exc
        except Exception as exc:
            if fallback is not None:
                logger.warning(f"[LLM] 调用失败，使用 fallback: {fallback}")
                return fallback
            logger.error(f"[LLM] 调用失败: {exc}")
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
            logger.warning("[LLM] 重要性评分 fallback=5.0")
            return 5.0
        except LLMError:
            logger.warning("[LLM] 重要性评分 fallback=5.0")
            return 5.0

    async def embed(self, text: str) -> list[float]:
        """Local sentence-transformers embedding (no network call)."""
        return self._st_model.encode(text).tolist()
