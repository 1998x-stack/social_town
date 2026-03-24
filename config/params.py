"""Global simulation parameters — all tunable via environment variables."""
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()


def _int_env(key: str, default: int) -> int:
    """Read an int environment variable with a clear error on bad value."""
    raw = os.getenv(key, str(default))
    try:
        return int(raw)
    except ValueError:
        raise ValueError(
            f"Environment variable {key}={raw!r} must be an integer (got {raw!r})"
        ) from None


NUM_AGENTS: int = _int_env("NUM_AGENTS", 10)
MODEL_NAME: str = os.getenv("MODEL_NAME", "qwen2.5:0.5b")
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
SIMULATION_DAYS: int = _int_env("SIMULATION_DAYS", 3)

STEPS_PER_DAY: int = 288            # 5min × 288 = 24h
REFLECTION_THRESHOLD: float = 100.0
MEMORY_TOP_K: int = 10
SOCIAL_GRAPH_DECAY: float = 0.01
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
METRICS_UPDATE_INTERVAL: int = 10
LLM_TIMEOUT: float = 30.0
SNAPSHOT_INTERVAL: int = 10
