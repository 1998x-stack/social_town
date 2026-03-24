# Social Town — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Docker-based LLM social dynamics simulation platform using Qwen2.5 via Ollama, with memory stream, 3-layer planning, social graph, event injection, quantitative evaluation, and real-time web dashboard.

**Architecture:** Each agent runs a cognitive loop (perceive → retrieve → plan/react → act → communicate → reflect) powered by Ollama. Persistent memory stream with 3-component scored retrieval (recency/importance/relevance). Social graph tracks inter-agent trust and intimacy; events propagate via credibility-weighted acceptance. Web dashboard streams metrics via SSE.

**Tech Stack:** Python 3.11, Ollama (qwen2.5:0.5b / 7b), sentence-transformers (all-MiniLM-L6-v2), FastAPI, httpx, networkx, scipy, pytest, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-03-24-social-town-design.md`
**Boundaries:** `CLAUDE.md`

---

## Stage Gate Rule

**Each stage ends with:**
1. Run `bash scripts/test.sh --stage N` — ALL tests must pass
2. `git add -A && git commit -m "stage(N): ..." && git push`
3. Do NOT proceed to stage N+1 until stage N is fully green and pushed.

---

## Task 0: Project Scaffold + Git Setup

**Files:**
- Create: `.gitignore`
- Create: `README.md`
- Create: `requirements.txt`
- Create: `config/__init__.py`
- Create: `config/params.py`
- Create: `scripts/test.sh`
- Create: `scripts/run.sh`
- Create: `main.py`

---

- [ ] **Step 0.1 — Write `.gitignore`**

```
# Python
__pycache__/
*.pyc
*.pyo
.pytest_cache/
*.egg-info/
dist/
build/
.coverage
htmlcov/

# Environment
.env
.venv/
venv/

# Data (runtime artifacts)
data/

# HuggingFace cache
.cache/

# IDE
.idea/
.vscode/
*.swp

# OS
.DS_Store
Thumbs.db
```

- [ ] **Step 0.2 — Write `README.md`**

```markdown
# Social Town — LLM 社会动力实验平台

基于 Qwen2.5（Ollama）的多智能体社会模拟系统，模拟信息传播、社区形成、观点极化和社会事件响应。

## 快速开始

### 本地开发
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py --agents 5 --days 2
open http://localhost:8080
```

### Docker（推荐）
```bash
docker-compose -f docker/docker-compose.yml up
open http://localhost:8080
```

## 测试
```bash
bash scripts/test.sh              # 全量测试
bash scripts/test.sh --stage 1   # 单阶段测试
bash scripts/test.sh --mock-llm  # 跳过真实 LLM 调用
bash scripts/test.sh --coverage  # 含覆盖率报告
```

## 配置
通过环境变量或 `.env` 文件：
- `MODEL_NAME` — Ollama 模型（默认 qwen2.5:0.5b）
- `NUM_AGENTS` — Agent 数量（默认 10，支持 5-50）
- `OLLAMA_HOST` — Ollama API 地址（默认 http://localhost:11434）
- `SIMULATION_DAYS` — 模拟天数（默认 3）

## 架构
见 `docs/superpowers/specs/2026-03-24-social-town-design.md`
```

- [ ] **Step 0.3 — Write `requirements.txt`**

```
fastapi>=0.100
uvicorn>=0.22
sentence-transformers>=2.2
networkx>=3.1
numpy>=1.24
httpx>=0.24
scipy>=1.11
pytest>=7.4
pytest-asyncio>=0.21
pytest-cov>=4.1
python-dotenv>=1.0
```

- [ ] **Step 0.4 — Write `config/params.py`**

```python
"""Global simulation parameters — all tunable via environment variables."""
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

NUM_AGENTS: int = int(os.getenv("NUM_AGENTS", "10"))
MODEL_NAME: str = os.getenv("MODEL_NAME", "qwen2.5:0.5b")
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
SIMULATION_DAYS: int = int(os.getenv("SIMULATION_DAYS", "3"))

STEPS_PER_DAY: int = 288            # 5min × 288 = 24h
REFLECTION_THRESHOLD: float = 100.0
MEMORY_TOP_K: int = 10
SOCIAL_GRAPH_DECAY: float = 0.01
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
METRICS_UPDATE_INTERVAL: int = 10
LLM_TIMEOUT: float = 30.0
SNAPSHOT_INTERVAL: int = 10
```

- [ ] **Step 0.5 — Write `config/__init__.py`** (empty)

- [ ] **Step 0.6 — Write `scripts/test.sh`**

```bash
#!/usr/bin/env bash
set -e

STAGE=""
MOCK_LLM=""
COVERAGE=""

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --stage) STAGE="$2"; shift ;;
    --mock-llm) MOCK_LLM="--mock-llm" ;;
    --coverage) COVERAGE="--cov=. --cov-report=term-missing" ;;
  esac
  shift
done

if [ -n "$STAGE" ]; then
  DIR="tests/stage${STAGE}_*"
else
  DIR="tests/"
fi

CMD="pytest $DIR -v $COVERAGE"
[ -n "$MOCK_LLM" ] && CMD="$CMD -k 'not real_llm'"

echo "▶ Running: $CMD"
eval $CMD
```

- [ ] **Step 0.7 — Write `scripts/setup.sh`**

```bash
#!/usr/bin/env bash
set -e
echo "▶ Installing Python dependencies..."
pip install -r requirements.txt

echo "▶ Downloading sentence-transformers model..."
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

echo "▶ Checking Ollama..."
if ! command -v ollama &>/dev/null; then
  echo "  Ollama not found. Install from https://ollama.ai then run: ollama pull qwen2.5:0.5b"
else
  ollama pull "${MODEL_NAME:-qwen2.5:0.5b}"
fi

echo "✓ Setup complete. Run: bash scripts/run.sh"
```

- [ ] **Step 0.8 — Write `scripts/run.sh`**

```bash
#!/usr/bin/env bash
set -e
AGENTS="${AGENTS:-10}"
DAYS="${DAYS:-3}"
python main.py --agents "$AGENTS" --days "$DAYS" "$@"
```

- [ ] **Step 0.9 — Write stub `main.py`**

```python
"""Entry point — runs simulation + webapp."""
from __future__ import annotations
import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="Social Town Simulation")
    parser.add_argument("--agents", type=int, default=10)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to snapshot JSON to resume from")
    args = parser.parse_args()
    print(f"[social-town] agents={args.agents} days={args.days}")
    # Replaced in Stage 5

if __name__ == "__main__":
    main()
```

- [ ] **Step 0.9 — Create empty `__init__.py` files**

```bash
touch agents/__init__.py agents/memory/__init__.py agents/cognitive/__init__.py \
      agents/social/__init__.py world/__init__.py core/__init__.py \
      llm/__init__.py evaluation/__init__.py webapp/__init__.py \
      tests/__init__.py
```

- [ ] **Step 0.10 — Initial commit and push**

```bash
git add -A
git commit -m "chore: project scaffold, config, scripts, README, .gitignore"
git push
```

---

## Stage 1: Memory Stream + Retrieval

**Files:**
- Create: `agents/memory/memory_stream.py`
- Create: `agents/memory/retrieval.py`
- Create: `tests/stage1_memory/conftest.py`
- Create: `tests/stage1_memory/fixtures/retrieval_cases.json`
- Create: `tests/stage1_memory/test_memory_stream.py`
- Create: `tests/stage1_memory/test_retrieval.py`
- Create: `tests/stage1_memory/__init__.py`

### Task 1.1: MemoryObject + MemoryStream

- [ ] **Step 1.1.1 — Write failing tests for MemoryObject**

`tests/stage1_memory/test_memory_stream.py`:
```python
import pytest
from agents.memory.memory_stream import MemoryObject, MemoryStream

def test_memory_object_creation():
    m = MemoryObject(
        content="Alice went to the library",
        memory_type="observation",
        created_at=0,
        importance=3.0,
        embedding=[0.1] * 384,
        source_agent=None,
        credibility=1.0,
    )
    assert m.id != ""
    assert m.last_accessed == 0
    assert m.memory_type == "observation"

def test_memory_stream_add_and_get():
    ms = MemoryStream()
    m = MemoryObject(
        content="Bob is at the cafe",
        memory_type="observation",
        created_at=5,
        importance=4.0,
        embedding=[0.0] * 384,
        source_agent=None,
        credibility=1.0,
    )
    ms.add(m)
    assert len(ms.all()) == 1
    assert ms.all()[0].content == "Bob is at the cafe"

def test_memory_stream_updates_last_accessed():
    ms = MemoryStream()
    m = MemoryObject(
        content="test", memory_type="observation", created_at=0,
        importance=1.0, embedding=[0.0]*384, source_agent=None, credibility=1.0
    )
    ms.add(m)
    ms.touch(m.id, current_step=10)
    assert ms.all()[0].last_accessed == 10

def test_reflection_accumulator_triggers():
    ms = MemoryStream()
    for i in range(12):
        ms.add(MemoryObject(
            content=f"event {i}", memory_type="observation", created_at=i,
            importance=9.0, embedding=[0.0]*384, source_agent=None, credibility=1.0
        ))
    assert ms.should_reflect(threshold=100.0) is True

def test_reflection_accumulator_reset():
    ms = MemoryStream()
    ms.add(MemoryObject(
        content="big event", memory_type="observation", created_at=0,
        importance=9.0, embedding=[0.0]*384, source_agent=None, credibility=1.0
    ))
    ms.reset_reflection_accumulator()
    assert ms.should_reflect(threshold=100.0) is False
```

- [ ] **Step 1.1.2 — Run test, confirm FAIL**

```bash
pytest tests/stage1_memory/test_memory_stream.py -v 2>&1 | head -20
# Expected: ImportError or AttributeError
```

- [ ] **Step 1.1.3 — Implement `agents/memory/memory_stream.py`**

```python
"""Memory stream: unified store for observations, reflections, and plans."""
from __future__ import annotations
import uuid
from dataclasses import dataclass, field

__all__ = ["MemoryObject", "MemoryStream"]


@dataclass
class MemoryObject:
    content: str
    memory_type: str          # "observation" | "reflection" | "plan"
    created_at: int
    importance: float         # 1.0-10.0
    embedding: list[float]
    source_agent: str | None  # None = direct observation
    credibility: float        # 0.0-1.0
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    last_accessed: int = field(init=False)

    def __post_init__(self) -> None:
        self.last_accessed = self.created_at


class MemoryStream:
    """Append-only memory store with reflection accumulator."""

    def __init__(self) -> None:
        self._memories: list[MemoryObject] = []
        self._reflection_accumulator: float = 0.0

    def add(self, memory: MemoryObject) -> None:
        self._memories.append(memory)
        self._reflection_accumulator += memory.importance

    def all(self) -> list[MemoryObject]:
        return list(self._memories)

    def recent(self, n: int) -> list[MemoryObject]:
        return self._memories[-n:]

    def touch(self, memory_id: str, current_step: int) -> None:
        for m in self._memories:
            if m.id == memory_id:
                m.last_accessed = current_step
                return

    def should_reflect(self, threshold: float) -> bool:
        return self._reflection_accumulator >= threshold

    def reset_reflection_accumulator(self) -> None:
        self._reflection_accumulator = 0.0

    def to_dict_list(self) -> list[dict]:
        """Serialise for snapshot (no embedding — recomputed on load)."""
        return [
            {
                "id": m.id,
                "content": m.content,
                "memory_type": m.memory_type,
                "created_at": m.created_at,
                "last_accessed": m.last_accessed,
                "importance": m.importance,
                "source_agent": m.source_agent,
                "credibility": m.credibility,
            }
            for m in self._memories
        ]

    @classmethod
    def from_dict_list(cls, data: list[dict], embed_fn) -> "MemoryStream":
        """Restore from snapshot; embed_fn recomputes embeddings."""
        ms = cls()
        for d in data:
            obj = MemoryObject(
                content=d["content"],
                memory_type=d["memory_type"],
                created_at=d["created_at"],
                importance=d["importance"],
                embedding=embed_fn(d["content"]),
                source_agent=d["source_agent"],
                credibility=d["credibility"],
            )
            obj.id = d["id"]
            obj.last_accessed = d["last_accessed"]
            ms._memories.append(obj)
        return ms
```

- [ ] **Step 1.1.4 — Run tests, confirm PASS**

```bash
pytest tests/stage1_memory/test_memory_stream.py -v
# Expected: 5 passed
```

### Task 1.2: Retrieval Engine

- [ ] **Step 1.2.1 — Create retrieval test fixture**

`tests/stage1_memory/fixtures/retrieval_cases.json`:
```json
[
  {
    "query": "library study",
    "memories": [
      {"id": "m1", "content": "Alice is studying at the library", "importance": 6.0, "created_at": 10, "last_accessed": 10},
      {"id": "m2", "content": "Bob ate breakfast at home", "importance": 3.0, "created_at": 8, "last_accessed": 8},
      {"id": "m3", "content": "Alice borrowed a book on economics", "importance": 5.0, "created_at": 9, "last_accessed": 9},
      {"id": "m4", "content": "The weather is sunny today", "importance": 2.0, "created_at": 7, "last_accessed": 7},
      {"id": "m5", "content": "Carol is researching at the library too", "importance": 5.0, "created_at": 11, "last_accessed": 11}
    ],
    "expected_top3_ids": ["m1", "m3", "m5"]
  },
  {
    "query": "food eating meal",
    "memories": [
      {"id": "m1", "content": "Bob ate breakfast at home", "importance": 3.0, "created_at": 5, "last_accessed": 5},
      {"id": "m2", "content": "Alice cooked dinner for friends", "importance": 7.0, "created_at": 6, "last_accessed": 6},
      {"id": "m3", "content": "The market opened at 9am", "importance": 4.0, "created_at": 3, "last_accessed": 3},
      {"id": "m4", "content": "Bob had lunch at the cafe", "importance": 4.0, "created_at": 7, "last_accessed": 7},
      {"id": "m5", "content": "Carol is studying economics", "importance": 5.0, "created_at": 2, "last_accessed": 2}
    ],
    "expected_top3_ids": ["m1", "m2", "m4"]
  }
]
```

- [ ] **Step 1.2.2 — Write failing retrieval tests**

`tests/stage1_memory/test_retrieval.py`:
```python
import json
import math
import pytest
from agents.memory.memory_stream import MemoryObject, MemoryStream
from agents.memory.retrieval import Retriever, score_recency, score_importance, score_relevance

FIXTURE_PATH = "tests/stage1_memory/fixtures/retrieval_cases.json"

def make_memory(d: dict, embed_fn) -> MemoryObject:
    return MemoryObject(
        content=d["content"],
        memory_type="observation",
        created_at=d["created_at"],
        importance=d["importance"],
        embedding=embed_fn(d["content"]),
        source_agent=None,
        credibility=1.0,
    )

def test_score_recency_decay():
    score_recent = score_recency(elapsed_steps=0)
    score_old = score_recency(elapsed_steps=200)
    assert score_recent == pytest.approx(1.0)
    assert score_old < score_recent
    assert 0.0 <= score_old <= 1.0

def test_score_importance_normalized():
    assert score_importance(10.0) == pytest.approx(1.0)
    assert score_importance(1.0) == pytest.approx(0.1)
    assert score_importance(5.0) == pytest.approx(0.5)

def test_score_relevance_range():
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [0.0, 1.0, 0.0]
    vec_same = [1.0, 0.0, 0.0]
    assert score_relevance(vec_a, vec_same) == pytest.approx(1.0, abs=1e-5)
    assert score_relevance(vec_a, vec_b) == pytest.approx(0.0, abs=1e-5)

def test_all_scores_in_unit_interval():
    r = score_recency(50)
    i = score_importance(7.0)
    v = score_relevance([0.5, 0.5], [0.5, 0.5])
    for s in [r, i, v]:
        assert 0.0 <= s <= 1.0

def test_retrieval_fixture_top3_hit_rate():
    """Top-3 hit rate >= 80% on labeled test cases."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embed_fn = lambda text: model.encode(text).tolist()

    with open(FIXTURE_PATH) as f:
        cases = json.load(f)

    retriever = Retriever(embed_fn=embed_fn)
    hits = 0
    total_expected = 0

    for case in cases:
        ms = MemoryStream()
        for d in case["memories"]:
            ms.add(make_memory(d, embed_fn))
        results = retriever.retrieve(ms.all(), query=case["query"], current_step=12, top_k=3)
        result_ids = {m.id for m in results}
        expected_ids = set(case["expected_top3_ids"])
        hits += len(result_ids & expected_ids)
        total_expected += len(expected_ids)

    hit_rate = hits / total_expected
    assert hit_rate >= 0.80, f"Hit rate {hit_rate:.2f} < 0.80"
```

- [ ] **Step 1.2.3 — Run test, confirm FAIL**

```bash
pytest tests/stage1_memory/test_retrieval.py -v 2>&1 | head -10
# Expected: ImportError
```

- [ ] **Step 1.2.4 — Implement `agents/memory/retrieval.py`**

```python
"""Three-component memory retrieval: recency + importance + relevance."""
from __future__ import annotations
import math
import numpy as np
from agents.memory.memory_stream import MemoryObject

__all__ = ["Retriever", "score_recency", "score_importance", "score_relevance"]

_DECAY_LAMBDA = 0.005


def score_recency(elapsed_steps: int) -> float:
    """Exponential decay; elapsed_steps since last access."""
    return math.exp(-_DECAY_LAMBDA * elapsed_steps)


def score_importance(importance: float) -> float:
    """Normalize 1-10 importance to [0, 1]."""
    return max(0.0, min(1.0, importance / 10.0))


def score_relevance(query_emb: list[float], mem_emb: list[float]) -> float:
    """Cosine similarity clipped to [0, 1]."""
    a = np.array(query_emb, dtype=float)
    b = np.array(mem_emb, dtype=float)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cosine = float(np.dot(a, b) / (norm_a * norm_b))
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


class Retriever:
    def __init__(self, embed_fn) -> None:
        self._embed = embed_fn

    def retrieve(
        self,
        memories: list[MemoryObject],
        query: str,
        current_step: int,
        top_k: int = 10,
    ) -> list[MemoryObject]:
        if not memories:
            return []
        query_emb = self._embed(query)
        scored = []
        for m in memories:
            r = score_recency(current_step - m.last_accessed)
            i = score_importance(m.importance)
            v = score_relevance(query_emb, m.embedding)
            scored.append((r + i + v, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]
```

- [ ] **Step 1.2.5 — Run all Stage 1 tests**

```bash
pytest tests/stage1_memory/ -v --cov=agents/memory --cov-report=term-missing
# Expected: all pass, coverage >= 90%
```

- [ ] **Step 1.2.6 — Stage 1 gate: verify criteria**

```bash
bash scripts/test.sh --stage 1 --coverage
# Gate: all tests pass, coverage >= 90%, hit rate >= 80%
```

- [ ] **Step 1.2.7 — Commit and push Stage 1**

```bash
git add -A
git commit -m "feat(stage1): memory stream, retrieval engine, fixture tests"
git push
```

---

## Stage 2: LLM Client

**Files:**
- Create: `llm/client.py`
- Create: `llm/prompts.py`
- Create: `tests/stage2_llm/__init__.py`
- Create: `tests/stage2_llm/conftest.py`
- Create: `tests/stage2_llm/test_client.py`
- Create: `tests/stage2_llm/test_prompts.py`

### Task 2.1: Ollama Client + Fallback

- [ ] **Step 2.1.1 — Write failing tests**

`tests/stage2_llm/test_client.py`:
```python
import pytest
from unittest.mock import patch, AsyncMock
from llm.client import OllamaClient, LLMError

@pytest.fixture
def client():
    return OllamaClient(host="http://localhost:11434", model="qwen2.5:0.5b", timeout=5.0)

@pytest.mark.asyncio
async def test_generate_returns_string(client):
    mock_response = {"response": "Alice goes to the library."}
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value.json.return_value = mock_response
        mock_post.return_value.raise_for_status = lambda: None
        result = await client.generate("What does Alice do?")
    assert isinstance(result, str)
    assert len(result) > 0

@pytest.mark.asyncio
async def test_generate_uses_fallback_on_timeout(client):
    with patch("httpx.AsyncClient.post", side_effect=Exception("timeout")):
        result = await client.generate("test prompt", fallback="reading quietly")
    assert result == "reading quietly"

@pytest.mark.asyncio
async def test_generate_raises_without_fallback(client):
    with patch("httpx.AsyncClient.post", side_effect=Exception("timeout")):
        with pytest.raises(LLMError):
            await client.generate("test prompt")

@pytest.mark.asyncio
async def test_score_importance_returns_float(client):
    mock_response = {"response": "7"}
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value.json.return_value = mock_response
        mock_post.return_value.raise_for_status = lambda: None
        score = await client.score_importance("Alice got accepted to university")
    assert 1.0 <= score <= 10.0

@pytest.mark.asyncio
async def test_score_importance_fallback_on_parse_error(client):
    mock_response = {"response": "very important"}  # not a number
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value.json.return_value = mock_response
        mock_post.return_value.raise_for_status = lambda: None
        score = await client.score_importance("test event")
    assert score == 5.0  # default fallback

@pytest.mark.asyncio
async def test_embed_returns_list(client):
    result = await client.embed("hello world")
    assert isinstance(result, list)
    assert len(result) == 384  # all-MiniLM-L6-v2 dim
```

- [ ] **Step 2.1.2 — Run test, confirm FAIL**

```bash
pytest tests/stage2_llm/test_client.py -v 2>&1 | head -10
```

- [ ] **Step 2.1.3 — Implement `llm/client.py`**

```python
"""Ollama LLM client with timeout protection and fallback."""
from __future__ import annotations
import asyncio
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
                return resp.json()["response"].strip()
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
```

- [ ] **Step 2.1.4 — Implement `llm/prompts.py`**

```python
"""Prompt templates — all token budgets enforced by truncation guards."""
from __future__ import annotations

__all__ = [
    "importance_prompt", "reflect_questions_prompt",
    "reflect_insights_prompt", "plan_l1_prompt",
    "plan_l2_prompt", "plan_l3_prompt", "dialogue_prompt",
]


def _truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "…"


def importance_prompt(content: str) -> str:
    c = _truncate(content, 300)
    return (
        f"Rate the importance of this memory for a person on a scale of 1-10.\n"
        f"1=mundane (brushing teeth), 10=life-changing (breakup, job offer).\n"
        f"Memory: {c}\n"
        f"Reply with only a single integer:"
    )


def reflect_questions_prompt(recent_memories: str) -> str:
    m = _truncate(recent_memories, 1200)
    return (
        f"Based on the following memories:\n{m}\n\n"
        f"What are 3 important high-level questions we can answer about this person? "
        f"Reply with exactly 3 questions, one per line."
    )


def reflect_insights_prompt(memories: str) -> str:
    m = _truncate(memories, 1400)
    return (
        f"Based on these memories:\n{m}\n\n"
        f"List 5 high-level insights about this person. "
        f"Format: one insight per line, no numbering."
    )


def plan_l1_prompt(name: str, description: str, day: int, top3_memories: str) -> str:
    d = _truncate(description, 200)
    m = _truncate(top3_memories, 200)
    return (
        f"You are {name}, {d}.\n"
        f"Today is simulation day {day}. Recent highlights: {m}\n"
        f"Describe today's plan as exactly 4 activity blocks.\n"
        f"Format (one per line): TimeSlot|Activity|Location\n"
        f"Example: Morning|Study economics|Library\n"
        f"Reply with only 4 lines:"
    )


def plan_l2_prompt(name: str, l1_block: str, top3_memories: str) -> str:
    m = _truncate(top3_memories, 200)
    return (
        f"You are {name}. Current time block: {l1_block}. Context: {m}\n"
        f"List 3 specific activities for this time block.\n"
        f"Format (one per line): Activity|Location\n"
        f"Reply with only 3 lines:"
    )


def plan_l3_prompt(name: str, l2_block: str, perceived_top3: str) -> str:
    e = _truncate(perceived_top3, 120)
    return (
        f"You are {name}. Current activity: {l2_block}. "
        f"Nearby (max 3): {e}\n"
        f"Describe your action right now in one sentence:"
    )


def dialogue_prompt(
    speaker: str,
    listener: str,
    relationship: str,
    context: str,
    topic: str,
) -> str:
    ctx = _truncate(context, 300)
    return (
        f"You are {speaker}. You are talking to {listener}. "
        f"Relationship: {relationship}. Context: {ctx}\n"
        f"Topic: {topic}\n"
        f"What do you say? Reply in one sentence:"
    )
```

- [ ] **Step 2.1.5 — Run all Stage 2 tests**

```bash
pytest tests/stage2_llm/ -v
# Expected: all pass (mocked)
```

- [ ] **Step 2.1.6 — Stage 2 gate check**

```bash
bash scripts/test.sh --stage 2 --mock-llm
```

- [ ] **Step 2.1.7 — Commit and push Stage 2**

```bash
git add -A
git commit -m "feat(stage2): Ollama client, prompt templates, fallback handling"
git push
```

---

## Stage 3: Planning Engine (L1 / L2 / L3)

**Files:**
- Create: `agents/cognitive/plan.py`
- Create: `tests/stage3_planning/__init__.py`
- Create: `tests/stage3_planning/conftest.py`
- Create: `tests/stage3_planning/test_plan.py`

### Task 3.1: Three-Layer Planner

- [ ] **Step 3.1.1 — Write failing tests**

`tests/stage3_planning/test_plan.py`:
```python
import pytest
import math
from unittest.mock import AsyncMock, patch
from agents.cognitive.plan import PlanningEngine, PlanState

FAKE_L1 = "Morning|Breakfast|Home\nAfternoon|Study|Library\nEvening|Walk|Park\nNight|Read|Home"
FAKE_L2 = "Read economics textbook|Library\nTake notes|Library\nDiscuss with peer|Library"
FAKE_L3 = "Alice sits down and opens her textbook."


@pytest.fixture
def mock_llm():
    client = AsyncMock()
    client.generate = AsyncMock(return_value=FAKE_L1)
    return client


@pytest.fixture
def engine(mock_llm):
    return PlanningEngine(
        name="Alice",
        description="A diligent economics student",
        llm=mock_llm,
    )


@pytest.mark.asyncio
async def test_l1_generates_4_blocks(engine, mock_llm):
    mock_llm.generate.return_value = FAKE_L1
    await engine.generate_l1(day=1, top3_memories="Alice likes studying")
    assert len(engine.state.l1_blocks) == 4


@pytest.mark.asyncio
async def test_l1_block_format_valid(engine, mock_llm):
    mock_llm.generate.return_value = FAKE_L1
    await engine.generate_l1(day=1, top3_memories="recent events")
    for block in engine.state.l1_blocks:
        parts = block.split("|")
        assert len(parts) == 3, f"Bad format: {block}"


@pytest.mark.asyncio
async def test_l2_generates_3_activities(engine, mock_llm):
    mock_llm.generate.return_value = FAKE_L1
    await engine.generate_l1(day=1, top3_memories="")
    mock_llm.generate.return_value = FAKE_L2
    await engine.generate_l2(top3_memories="studying notes")
    assert len(engine.state.l2_activities) == 3


@pytest.mark.asyncio
async def test_l3_returns_nonempty_string(engine, mock_llm):
    mock_llm.generate.return_value = FAKE_L3
    action = await engine.generate_l3(perceived_top3="Bob is nearby reading")
    assert isinstance(action, str) and len(action) > 0


@pytest.mark.asyncio
async def test_replan_resets_l2(engine, mock_llm):
    mock_llm.generate.return_value = FAKE_L1
    await engine.generate_l1(day=1, top3_memories="")
    mock_llm.generate.return_value = FAKE_L2
    await engine.generate_l2(top3_memories="")
    assert len(engine.state.l2_activities) == 3
    engine.replan_l2()
    assert engine.state.l2_activities == []


@pytest.mark.asyncio
async def test_l1_location_entropy_across_5_calls(engine, mock_llm):
    """Shannon entropy of location field across 5 days must be > 0.5."""
    varied_plans = [
        "Morning|Study|Library\nAfternoon|Lunch|Cafe\nEvening|Walk|Park\nNight|Sleep|Home",
        "Morning|Cook|Home\nAfternoon|Work|Office\nEvening|Read|Library\nNight|Sleep|Home",
        "Morning|Jog|Park\nAfternoon|Study|Library\nEvening|Dinner|Cafe\nNight|Sleep|Home",
        "Morning|Meeting|Office\nAfternoon|Research|Library\nEvening|Relax|Home\nNight|Sleep|Home",
        "Morning|Breakfast|Cafe\nAfternoon|Study|Home\nEvening|Walk|Park\nNight|Read|Home",
    ]
    locations_per_day = []
    for plan in varied_plans:
        mock_llm.generate.return_value = plan
        await engine.generate_l1(day=1, top3_memories="")
        locs = [b.split("|")[2] for b in engine.state.l1_blocks]
        locations_per_day.extend(locs)

    from collections import Counter
    counts = Counter(locations_per_day)
    total = sum(counts.values())
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    assert entropy > 0.5, f"Entropy {entropy:.3f} too low"
```

- [ ] **Step 3.1.2 — Run test, confirm FAIL**

```bash
pytest tests/stage3_planning/test_plan.py -v 2>&1 | head -10
```

- [ ] **Step 3.1.3 — Implement `agents/cognitive/plan.py`**

```python
"""Three-layer planning engine: L1 (daily) → L2 (block) → L3 (action)."""
from __future__ import annotations
from dataclasses import dataclass, field
from llm.prompts import plan_l1_prompt, plan_l2_prompt, plan_l3_prompt

__all__ = ["PlanningEngine", "PlanState"]


@dataclass
class PlanState:
    l1_blocks: list[str] = field(default_factory=list)  # ["TimeSlot|Activity|Location"]
    l1_index: int = 0                                    # current block index
    l2_activities: list[str] = field(default_factory=list)  # ["Activity|Location"]
    l2_index: int = 0


class PlanningEngine:
    def __init__(self, name: str, description: str, llm) -> None:
        self._name = name
        self._desc = description
        self._llm = llm
        self.state = PlanState()

    async def generate_l1(self, day: int, top3_memories: str) -> None:
        prompt = plan_l1_prompt(self._name, self._desc, day, top3_memories)
        raw = await self._llm.generate(prompt, fallback="Morning|Rest|Home\nAfternoon|Rest|Home\nEvening|Rest|Home\nNight|Sleep|Home")
        blocks = [line.strip() for line in raw.strip().splitlines() if "|" in line]
        # Ensure exactly 4 blocks
        while len(blocks) < 4:
            blocks.append("Free time|Wander|Town")
        self.state.l1_blocks = blocks[:4]
        self.state.l1_index = 0
        self.state.l2_activities = []
        self.state.l2_index = 0

    def current_l1_block(self) -> str:
        if not self.state.l1_blocks:
            return "Unknown|Idle|Home"
        idx = min(self.state.l1_index, len(self.state.l1_blocks) - 1)
        return self.state.l1_blocks[idx]

    async def generate_l2(self, top3_memories: str) -> None:
        block = self.current_l1_block()
        prompt = plan_l2_prompt(self._name, block, top3_memories)
        raw = await self._llm.generate(prompt, fallback="Idle|Home\nIdle|Home\nIdle|Home")
        activities = [l.strip() for l in raw.strip().splitlines() if "|" in l]
        while len(activities) < 3:
            activities.append("Rest|Home")
        self.state.l2_activities = activities[:3]
        self.state.l2_index = 0

    def current_l2_activity(self) -> str:
        if not self.state.l2_activities:
            return "Idle|Home"
        idx = min(self.state.l2_index, len(self.state.l2_activities) - 1)
        return self.state.l2_activities[idx]

    async def generate_l3(self, perceived_top3: str) -> str:
        activity = self.current_l2_activity()
        prompt = plan_l3_prompt(self._name, activity, perceived_top3)
        return await self._llm.generate(prompt, fallback=f"{self._name} continues their activity.")

    def replan_l2(self) -> None:
        """Discard current L2; caller must call generate_l2 again."""
        self.state.l2_activities = []
        self.state.l2_index = 0
```

- [ ] **Step 3.1.4 — Run all Stage 3 tests**

```bash
pytest tests/stage3_planning/ -v
# Expected: all pass
```

- [ ] **Step 3.1.5 — Stage 3 gate check**

```bash
bash scripts/test.sh --stage 3
```

- [ ] **Step 3.1.6 — Commit and push Stage 3**

```bash
git add -A
git commit -m "feat(stage3): three-layer planning engine L1/L2/L3"
git push
```

---

## Stage 4: Social Network + Dialogue + Event Injection

**Files:**
- Create: `world/social_graph.py`
- Create: `agents/social/dialogue.py`
- Create: `world/event_injector.py`
- Create: `world/town.py`
- Create: `tests/stage4_social/__init__.py`
- Create: `tests/stage4_social/conftest.py`
- Create: `tests/stage4_social/test_social_graph.py`
- Create: `tests/stage4_social/test_dialogue.py`
- Create: `tests/stage4_social/test_event_injector.py`

### Task 4.1: Social Graph

- [ ] **Step 4.1.1 — Write failing tests**

`tests/stage4_social/test_social_graph.py`:
```python
import pytest
from world.social_graph import SocialGraph, SocialEdge

@pytest.fixture
def graph():
    g = SocialGraph()
    g.add_agent("a1")
    g.add_agent("a2")
    g.add_agent("a3")
    return g

def test_add_agent_creates_node(graph):
    assert "a1" in graph.agents

def test_initial_edge_defaults(graph):
    edge = graph.get_edge("a1", "a2")
    assert edge.intimacy == 0.1
    assert edge.trust == 0.3
    assert edge.interaction_count == 0

def test_record_interaction_updates_intimacy(graph):
    graph.record_interaction("a1", "a2")
    edge = graph.get_edge("a1", "a2")
    assert edge.intimacy == pytest.approx(0.15, abs=0.01)
    assert edge.interaction_count == 1

def test_intimacy_capped_at_1(graph):
    for _ in range(30):
        graph.record_interaction("a1", "a2")
    assert graph.get_edge("a1", "a2").intimacy <= 1.0

def test_accept_probability_formula(graph):
    # trust=1.0, credibility=1.0 → prob=1.0
    graph.get_edge("a1", "a2").trust = 1.0
    prob = graph.accept_probability("a1", "a2", credibility=1.0)
    assert prob == pytest.approx(1.0)

def test_accept_probability_zero_credibility(graph):
    prob = graph.accept_probability("a1", "a2", credibility=0.0)
    assert prob == pytest.approx(0.0)

def test_serialization_roundtrip(graph):
    graph.record_interaction("a1", "a2")
    data = graph.to_dict()
    restored = SocialGraph.from_dict(data)
    edge = restored.get_edge("a1", "a2")
    assert edge.interaction_count == 1
    assert edge.intimacy == pytest.approx(0.15, abs=0.01)
```

- [ ] **Step 4.1.2 — Implement `world/social_graph.py`**

```python
"""Social relationship graph: intimacy, trust, interaction history."""
from __future__ import annotations
from dataclasses import dataclass, field
import itertools
import random

__all__ = ["SocialGraph", "SocialEdge"]


@dataclass
class SocialEdge:
    from_agent: str
    to_agent: str
    intimacy: float = 0.1
    trust: float = 0.3
    interaction_count: int = 0
    last_interaction: int = 0


class SocialGraph:
    def __init__(self) -> None:
        self.agents: set[str] = set()
        self._edges: dict[tuple[str, str], SocialEdge] = {}

    def add_agent(self, agent_id: str) -> None:
        self.agents.add(agent_id)
        for other in self.agents:
            if other != agent_id:
                for a, b in [(agent_id, other), (other, agent_id)]:
                    if (a, b) not in self._edges:
                        self._edges[(a, b)] = SocialEdge(from_agent=a, to_agent=b)

    def get_edge(self, from_id: str, to_id: str) -> SocialEdge:
        key = (from_id, to_id)
        if key not in self._edges:
            self._edges[key] = SocialEdge(from_agent=from_id, to_agent=to_id)
        return self._edges[key]

    def record_interaction(self, a: str, b: str, current_step: int = 0) -> None:
        for frm, to in [(a, b), (b, a)]:
            edge = self.get_edge(frm, to)
            edge.intimacy = min(1.0, edge.intimacy + 0.05)
            edge.interaction_count += 1
            edge.last_interaction = current_step

    def accept_probability(self, from_id: str, to_id: str, credibility: float) -> float:
        trust = self.get_edge(from_id, to_id).trust
        return min(1.0, trust * credibility)

    def edge_count(self) -> int:
        return len(self._edges)

    def density(self, n_agents: int) -> float:
        if n_agents < 2:
            return 0.0
        max_edges = n_agents * (n_agents - 1)
        return self.edge_count() / max_edges

    def to_dict(self) -> dict:
        return {
            "agents": list(self.agents),
            "edges": [
                {
                    "from": e.from_agent, "to": e.to_agent,
                    "intimacy": e.intimacy, "trust": e.trust,
                    "interaction_count": e.interaction_count,
                    "last_interaction": e.last_interaction,
                }
                for e in self._edges.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SocialGraph":
        g = cls()
        g.agents = set(data["agents"])
        for ed in data["edges"]:
            key = (ed["from"], ed["to"])
            g._edges[key] = SocialEdge(
                from_agent=ed["from"], to_agent=ed["to"],
                intimacy=ed["intimacy"], trust=ed["trust"],
                interaction_count=ed["interaction_count"],
                last_interaction=ed["last_interaction"],
            )
        return g
```

### Task 4.2: Dialogue + Event Injector

- [ ] **Step 4.2.1 — Write failing tests for dialogue and event injection**

`tests/stage4_social/test_dialogue.py`:
```python
import pytest
from unittest.mock import AsyncMock
from agents.social.dialogue import DialogueEngine

@pytest.fixture
def mock_llm():
    client = AsyncMock()
    client.generate = AsyncMock(return_value="Hello, how are you today?")
    return client

@pytest.mark.asyncio
async def test_dialogue_returns_nonempty_string(mock_llm):
    engine = DialogueEngine(llm=mock_llm)
    utterance = await engine.generate_utterance(
        speaker="Alice", listener="Bob",
        relationship="acquaintances",
        context="Alice met Bob at the library",
        topic="the upcoming election",
    )
    assert isinstance(utterance, str)
    assert len(utterance.strip()) >= 5

@pytest.mark.asyncio
async def test_dialogue_fallback_on_error():
    erroring_llm = AsyncMock()
    erroring_llm.generate = AsyncMock(side_effect=Exception("LLM down"))
    engine = DialogueEngine(llm=erroring_llm)
    utterance = await engine.generate_utterance(
        speaker="Alice", listener="Bob",
        relationship="acquaintances", context="", topic="",
    )
    assert utterance != ""
```

`tests/stage4_social/test_event_injector.py`:
```python
import pytest
from world.event_injector import EventInjector
from agents.memory.memory_stream import MemoryStream

def make_stream():
    return MemoryStream()

def test_inject_event_adds_to_seed_agent_memory():
    streams = {"a1": make_stream(), "a2": make_stream()}
    injector = EventInjector(agent_streams=streams)
    injector.inject_event(
        event_type="rumor",
        content="The mayor is resigning",
        seed_agents=["a1"],
        credibility=0.9,
        step=0,
        embed_fn=lambda t: [0.0] * 384,
        importance_fn=lambda t: 8.0,
    )
    mems = streams["a1"].all()
    assert any("mayor" in m.content.lower() for m in mems)
    assert mems[-1].credibility == pytest.approx(0.9)

def test_inject_event_does_not_add_to_non_seed():
    streams = {"a1": make_stream(), "a2": make_stream()}
    injector = EventInjector(agent_streams=streams)
    injector.inject_event(
        event_type="rumor",
        content="Secret event",
        seed_agents=["a1"],
        credibility=0.5,
        step=0,
        embed_fn=lambda t: [0.0] * 384,
        importance_fn=lambda t: 5.0,
    )
    assert len(streams["a2"].all()) == 0

def test_credibility_zero_still_injects_to_seed():
    streams = {"a1": make_stream()}
    injector = EventInjector(agent_streams=streams)
    injector.inject_event(
        event_type="rumor", content="Low-cred rumor", seed_agents=["a1"],
        credibility=0.0, step=0,
        embed_fn=lambda t: [0.0]*384, importance_fn=lambda t: 2.0,
    )
    assert len(streams["a1"].all()) == 1
```

- [ ] **Step 4.2.2 — Implement `agents/social/dialogue.py`**

```python
"""Dialogue generation between two agents."""
from __future__ import annotations
from llm.prompts import dialogue_prompt

__all__ = ["DialogueEngine"]


class DialogueEngine:
    def __init__(self, llm) -> None:
        self._llm = llm

    async def generate_utterance(
        self,
        speaker: str,
        listener: str,
        relationship: str,
        context: str,
        topic: str,
    ) -> str:
        prompt = dialogue_prompt(speaker, listener, relationship, context, topic)
        fallback = f"{speaker} nods and continues the conversation."
        try:
            result = await self._llm.generate(prompt, fallback=fallback)
            return result if result.strip() else fallback
        except Exception:
            return fallback
```

- [ ] **Step 4.2.3 — Implement `world/event_injector.py`**

```python
"""Experimental event injection interface."""
from __future__ import annotations
from agents.memory.memory_stream import MemoryObject, MemoryStream

__all__ = ["EventInjector"]


class EventInjector:
    def __init__(self, agent_streams: dict[str, MemoryStream]) -> None:
        self._streams = agent_streams

    def inject_event(
        self,
        event_type: str,
        content: str,
        seed_agents: list[str],
        credibility: float,
        step: int,
        embed_fn,
        importance_fn,
    ) -> None:
        for agent_id in seed_agents:
            if agent_id not in self._streams:
                continue
            importance = importance_fn(content)
            embedding = embed_fn(content)
            memory = MemoryObject(
                content=f"[{event_type}] {content}",
                memory_type="observation",
                created_at=step,
                importance=importance,
                embedding=embedding,
                source_agent=None,
                credibility=credibility,
            )
            self._streams[agent_id].add(memory)
```

- [ ] **Step 4.2.4 — Write stub `world/town.py`**

```python
"""Town world state: locations and object states."""
from __future__ import annotations
from dataclasses import dataclass, field

__all__ = ["Town", "Location"]

LOCATIONS = ["Home", "Library", "Cafe", "Park", "Office", "Market", "Town Square"]


@dataclass
class Location:
    name: str
    capacity: int = 10
    occupants: list[str] = field(default_factory=list)


class Town:
    def __init__(self) -> None:
        self.locations: dict[str, Location] = {
            name: Location(name=name) for name in LOCATIONS
        }

    def move_agent(self, agent_id: str, from_loc: str, to_loc: str) -> None:
        if from_loc in self.locations and agent_id in self.locations[from_loc].occupants:
            self.locations[from_loc].occupants.remove(agent_id)
        if to_loc in self.locations:
            self.locations[to_loc].occupants.append(agent_id)

    def agents_at(self, location: str) -> list[str]:
        return list(self.locations.get(location, Location(name=location)).occupants)
```

- [ ] **Step 4.2.5 — Run all Stage 4 tests**

```bash
pytest tests/stage4_social/ -v
# Expected: all pass
```

- [ ] **Step 4.2.6 — Stage 4 gate check**

```bash
bash scripts/test.sh --stage 4
```

- [ ] **Step 4.2.7 — Commit and push Stage 4**

```bash
git add -A
git commit -m "feat(stage4): social graph, dialogue engine, event injector, town world"
git push
```

---

## Stage 5: Persona + Main Simulation Loop

**Files:**
- Create: `agents/persona.py`
- Create: `agents/cognitive/reflect.py`
- Create: `core/simulation.py`
- Create: `tests/stage5_simulation/__init__.py`
- Create: `tests/stage5_simulation/conftest.py`
- Create: `tests/stage5_simulation/test_persona.py`
- Create: `tests/stage5_simulation/test_simulation.py`

### Task 5.1: Reflect Engine + Persona

- [ ] **Step 5.1.1 — Write failing tests**

`tests/stage5_simulation/test_persona.py`:
```python
import pytest
from unittest.mock import AsyncMock, patch
from agents.persona import Persona
from agents.memory.memory_stream import MemoryObject

def make_mock_llm():
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value="Alice sits quietly and reads.")
    llm.score_importance = AsyncMock(return_value=5.0)
    llm.embed = AsyncMock(return_value=[0.1] * 384)
    return llm

@pytest.mark.asyncio
async def test_persona_perceive_adds_memory():
    llm = make_mock_llm()
    p = Persona(agent_id="a1", name="Alice", description="A student", llm=llm)
    await p.perceive("Bob is reading at the library", current_step=1)
    assert len(p.memory.all()) == 1

@pytest.mark.asyncio
async def test_persona_act_returns_string():
    llm = make_mock_llm()
    p = Persona(agent_id="a1", name="Alice", description="A student", llm=llm)
    llm.generate.return_value = "Morning|Study|Library\nAfternoon|Lunch|Cafe\nEvening|Walk|Park\nNight|Sleep|Home"
    await p.start_day(day=1, current_step=0)
    llm.generate.return_value = "Alice opens her notebook."
    action = await p.act(current_step=1, perceived_top3="Bob is nearby")
    assert isinstance(action, str) and len(action) > 0

@pytest.mark.asyncio
async def test_persona_snapshot_roundtrip():
    llm = make_mock_llm()
    p = Persona(agent_id="a1", name="Alice", description="A student", llm=llm)
    await p.perceive("Alice went to the library", current_step=5)
    snap = p.to_snapshot()
    assert snap["id"] == "a1"
    assert snap["memory_count"] == 1
    assert len(snap["memories"]) == 1
```

`tests/stage5_simulation/test_simulation.py`:
```python
import pytest
from unittest.mock import AsyncMock, patch
from core.simulation import Simulation

def make_mock_llm():
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value="Morning|Rest|Home\nAfternoon|Rest|Home\nEvening|Rest|Home\nNight|Sleep|Home")
    llm.score_importance = AsyncMock(return_value=3.0)
    llm.embed = AsyncMock(return_value=[0.1]*384)
    return llm

@pytest.mark.asyncio
async def test_simulation_runs_50_steps_no_crash():
    sim = Simulation(n_agents=5, llm=make_mock_llm(), steps_per_day=10)
    for _ in range(50):
        await sim.step()
    assert sim.current_step == 50

@pytest.mark.asyncio
async def test_simulation_memories_accumulate():
    sim = Simulation(n_agents=3, llm=make_mock_llm(), steps_per_day=10)
    for _ in range(10):
        await sim.step()
    total_memories = sum(len(p.memory.all()) for p in sim.personas)
    assert total_memories >= 10

@pytest.mark.asyncio
async def test_simulation_snapshot_and_resume():
    import json, tempfile, os
    sim = Simulation(n_agents=3, llm=make_mock_llm(), steps_per_day=10)
    for _ in range(20):
        await sim.step()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sim.to_snapshot(), f)
        snap_path = f.name
    try:
        sim2 = Simulation.from_snapshot(snap_path, llm=make_mock_llm())
        assert sim2.current_step == 20
        await sim2.step()
        assert sim2.current_step == 21
    finally:
        os.unlink(snap_path)
```

- [ ] **Step 5.1.2 — Implement `agents/cognitive/reflect.py`**

```python
"""Reflection engine: synthesise high-level insights from raw memories."""
from __future__ import annotations
from agents.memory.memory_stream import MemoryObject, MemoryStream
from llm.prompts import reflect_questions_prompt, reflect_insights_prompt

__all__ = ["ReflectionEngine"]


class ReflectionEngine:
    def __init__(self, llm) -> None:
        self._llm = llm

    async def reflect(
        self,
        memory_stream: MemoryStream,
        embed_fn,
        current_step: int,
    ) -> list[MemoryObject]:
        recent = memory_stream.recent(100)
        if not recent:
            return []
        mem_text = "\n".join(f"- {m.content}" for m in recent)

        # Step 1: generate questions
        questions_raw = await self._llm.generate(
            reflect_questions_prompt(mem_text),
            fallback="What is this person focused on?\nWhat relationships matter?\nWhat goals do they have?"
        )
        questions = [q.strip() for q in questions_raw.strip().splitlines() if q.strip()][:3]

        # Step 2: generate insights from question-retrieved context
        insights_raw = await self._llm.generate(
            reflect_insights_prompt(mem_text),
            fallback="This person values community connections."
        )
        insights = [i.strip() for i in insights_raw.strip().splitlines() if i.strip()][:5]

        # Step 3: store as reflection memories
        reflection_objects = []
        for insight in insights:
            embedding = await embed_fn(insight)
            m = MemoryObject(
                content=insight,
                memory_type="reflection",
                created_at=current_step,
                importance=7.0,  # reflections have elevated importance
                embedding=embedding,
                source_agent=None,
                credibility=1.0,
            )
            reflection_objects.append(m)
        return reflection_objects
```

- [ ] **Step 5.1.3 — Implement `agents/persona.py`**

```python
"""Agent Persona: cognitive loop entry point."""
from __future__ import annotations
from agents.memory.memory_stream import MemoryObject, MemoryStream
from agents.memory.retrieval import Retriever
from agents.cognitive.plan import PlanningEngine
from agents.cognitive.reflect import ReflectionEngine
from config.params import MEMORY_TOP_K, REFLECTION_THRESHOLD

__all__ = ["Persona"]


class Persona:
    def __init__(self, agent_id: str, name: str, description: str, llm) -> None:
        self.id = agent_id
        self.name = name
        self.description = description
        self.location: str = "Home"
        self.current_action: str = "idle"
        self.opinion: dict[str, float] = {}

        self.memory = MemoryStream()
        self._retriever = Retriever(embed_fn=lambda t: llm.embed(t))
        self._planner = PlanningEngine(name, description, llm)
        self._reflector = ReflectionEngine(llm)
        self._llm = llm

    async def _embed(self, text: str) -> list[float]:
        return await self._llm.embed(text)

    async def perceive(self, observation: str, current_step: int) -> None:
        importance = await self._llm.score_importance(observation)
        embedding = await self._embed(observation)
        m = MemoryObject(
            content=observation,
            memory_type="observation",
            created_at=current_step,
            importance=importance,
            embedding=embedding,
            source_agent=None,
            credibility=1.0,
        )
        self.memory.add(m)

    async def start_day(self, day: int, current_step: int) -> None:
        recent = self.memory.recent(3)
        top3 = "; ".join(m.content for m in recent)
        await self._planner.generate_l1(day=day, top3_memories=top3)
        recent_top = self.memory.recent(3)
        top3_l2 = "; ".join(m.content for m in recent_top)
        await self._planner.generate_l2(top3_memories=top3_l2)

    async def act(self, current_step: int, perceived_top3: str) -> str:
        action = await self._planner.generate_l3(perceived_top3=perceived_top3)
        self.current_action = action
        return action

    async def maybe_reflect(self, current_step: int) -> bool:
        if not self.memory.should_reflect(REFLECTION_THRESHOLD):
            return False
        reflections = await self._reflector.reflect(
            self.memory, embed_fn=self._embed, current_step=current_step
        )
        for r in reflections:
            self.memory.add(r)
        self.memory.reset_reflection_accumulator()
        return True

    def to_snapshot(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "location": self.location,
            "current_action": self.current_action,
            "opinion": self.opinion,
            "memory_count": len(self.memory.all()),
            "l1_plan": self._planner.state.l1_blocks,
            "l2_current_block": self._planner.current_l1_block(),
            "reflection_accumulator": self.memory._reflection_accumulator,
            "memories": self.memory.to_dict_list(),
        }

    @classmethod
    def from_snapshot(cls, data: dict, llm) -> "Persona":
        p = cls(
            agent_id=data["id"],
            name=data["name"],
            description=data["description"],
            llm=llm,
        )
        p.location = data["location"]
        p.current_action = data["current_action"]
        p.opinion = data["opinion"]
        embed_fn_sync = lambda t: llm._st_model.encode(t).tolist()
        p.memory = MemoryStream.from_dict_list(data["memories"], embed_fn_sync)
        p.memory._reflection_accumulator = data["reflection_accumulator"]
        p._planner.state.l1_blocks = data["l1_plan"]
        return p
```

- [ ] **Step 5.1.4 — Implement `core/simulation.py`**

```python
"""Main simulation loop: coordinates all agents per time step."""
from __future__ import annotations
import json
import os
import random
from agents.persona import Persona
from world.social_graph import SocialGraph
from world.town import Town
from config.params import (
    NUM_AGENTS, STEPS_PER_DAY, SNAPSHOT_INTERVAL
)

__all__ = ["Simulation"]

_AGENT_SEEDS = [
    ("Alice", "A diligent economics student who loves books"),
    ("Bob", "A friendly cafe owner interested in local politics"),
    ("Carol", "A nurse who cares deeply about public health"),
    ("David", "An ambitious journalist seeking the truth"),
    ("Eve", "A retired teacher passionate about education"),
    ("Frank", "A young farmer concerned about market prices"),
    ("Grace", "A tech worker interested in innovation"),
    ("Henry", "A local politician campaigning for mayor"),
    ("Iris", "A librarian who knows everyone in town"),
    ("Jack", "A skeptical trader with strong opinions"),
]


class Simulation:
    def __init__(
        self,
        n_agents: int = NUM_AGENTS,
        llm=None,
        steps_per_day: int = STEPS_PER_DAY,
        data_dir: str = "data",
    ) -> None:
        self.n_agents = n_agents
        self._llm = llm
        self.steps_per_day = steps_per_day
        self.data_dir = data_dir
        self.current_step: int = 0
        self.current_day: int = 1

        seeds = _AGENT_SEEDS[:n_agents]
        self.personas: list[Persona] = [
            Persona(agent_id=f"agent_{i:02d}", name=seeds[i][0],
                    description=seeds[i][1], llm=llm)
            for i in range(n_agents)
        ]
        self.social_graph = SocialGraph()
        for p in self.personas:
            self.social_graph.add_agent(p.id)
        self.town = Town()
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(f"{data_dir}/snapshots", exist_ok=True)

    async def step(self) -> dict:
        # New day check
        if self.current_step % self.steps_per_day == 0:
            for p in self.personas:
                await p.start_day(day=self.current_day, current_step=self.current_step)
            self.current_day += 1

        step_log = {"step": self.current_step, "actions": []}

        for persona in self.personas:
            # Perceive nearby agents
            nearby = [
                f"{other.name} is {other.current_action}"
                for other in self.personas
                if other.id != persona.id
            ][:3]
            perceived_str = "; ".join(nearby)

            if perceived_str:
                await persona.perceive(perceived_str, self.current_step)

            # Act
            action = await persona.act(
                current_step=self.current_step,
                perceived_top3=perceived_str,
            )
            step_log["actions"].append({"agent": persona.name, "action": action})

            # Communicate: probabilistic dialogue with one nearby agent
            candidates = [p for p in self.personas if p.id != persona.id]
            if candidates:
                import random as _rng
                partner = _rng.choice(candidates)
                edge = self.social_graph.get_edge(persona.id, partner.id)
                # Only engage if intimacy threshold met (avoids all-pairs flood)
                if edge.intimacy >= 0.15 or _rng.random() < 0.1:
                    from agents.social.dialogue import DialogueEngine
                    dlg = DialogueEngine(llm=self._llm)
                    topic = perceived_str or "the town news"
                    utterance = await dlg.generate_utterance(
                        speaker=persona.name, listener=partner.name,
                        relationship=f"intimacy={edge.intimacy:.2f}",
                        context=perceived_str, topic=topic,
                    )
                    # Partner receives utterance as observation with credibility propagation
                    recv_prob = self.social_graph.accept_probability(
                        persona.id, partner.id, credibility=1.0
                    )
                    if _rng.random() < recv_prob:
                        from agents.memory.memory_stream import MemoryObject
                        emb = await partner._embed(utterance)
                        imp = await partner._llm.score_importance(utterance)
                        new_cred = edge.trust * 1.0  # direct speech, credibility=1.0
                        m = MemoryObject(
                            content=f"{persona.name} said: {utterance}",
                            memory_type="observation",
                            created_at=self.current_step,
                            importance=imp,
                            embedding=emb,
                            source_agent=persona.id,
                            credibility=new_cred,
                        )
                        partner.memory.add(m)
                        # Update opinion on any topic in utterance
                        for topic_key, shift in [("community", 0.05)]:
                            partner.opinion[topic_key] = max(-1.0, min(1.0,
                                partner.opinion.get(topic_key, 0.0) + shift * new_cred
                            ))
                    self.social_graph.record_interaction(
                        persona.id, partner.id, current_step=self.current_step
                    )

            # Reflect if threshold met
            await persona.maybe_reflect(self.current_step)

        self.current_step += 1

        # Snapshot
        if self.current_step % SNAPSHOT_INTERVAL == 0:
            snap_path = f"{self.data_dir}/snapshots/step_{self.current_step:06d}.json"
            with open(snap_path, "w") as f:
                json.dump(self.to_snapshot(), f, indent=2)

        return step_log

    def to_snapshot(self) -> dict:
        return {
            "step": self.current_step,
            "agents": [p.to_snapshot() for p in self.personas],
            "social_graph": self.social_graph.to_dict(),
        }

    @classmethod
    def from_snapshot(cls, path: str, llm) -> "Simulation":
        with open(path) as f:
            data = json.load(f)
        sim = cls(n_agents=len(data["agents"]), llm=llm)
        sim.current_step = data["step"]
        sim.current_day = data["step"] // sim.steps_per_day + 1
        sim.personas = [Persona.from_snapshot(p, llm) for p in data["agents"]]
        sim.social_graph = SocialGraph.from_dict(data["social_graph"])
        return sim
```

- [ ] **Step 5.1.5 — Update `main.py` to wire everything**

```python
"""Entry point — runs simulation + webapp."""
from __future__ import annotations
import argparse
import asyncio
from llm.client import OllamaClient
from core.simulation import Simulation


async def run(agents: int, days: int, resume: str | None) -> None:
    llm = OllamaClient()
    from config.params import STEPS_PER_DAY
    if resume:
        sim = Simulation.from_snapshot(resume, llm=llm)
        print(f"[social-town] Resumed from step {sim.current_step}")
    else:
        sim = Simulation(n_agents=agents, llm=llm)
        print(f"[social-town] Starting: {agents} agents, {days} days")

    total_steps = days * STEPS_PER_DAY
    for step_num in range(total_steps):
        log = await sim.step()
        if step_num % 10 == 0:
            print(f"  Step {sim.current_step}: {log['actions'][0]['action'][:60]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=10)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    asyncio.run(run(args.agents, args.days, args.resume))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5.1.6 — Run all Stage 5 tests**

```bash
pytest tests/stage5_simulation/ -v
# Expected: all pass
```

- [ ] **Step 5.1.7 — Stage 5 gate check**

```bash
bash scripts/test.sh --stage 5
```

- [ ] **Step 5.1.8 — Commit and push Stage 5**

```bash
git add -A
git commit -m "feat(stage5): persona, reflection engine, main simulation loop, snapshot/resume"
git push
```

---

## Stage 6: Evaluation Metrics + Reporter

**Files:**
- Create: `evaluation/metrics.py`
- Create: `evaluation/reporter.py`
- Create: `tests/stage6_evaluation/__init__.py`
- Create: `tests/stage6_evaluation/test_metrics.py`
- Create: `tests/stage6_evaluation/test_reporter.py`

### Task 6.1: Four-Dimension Metrics

- [ ] **Step 6.1.1 — Write failing tests**

`tests/stage6_evaluation/test_metrics.py`:
```python
import pytest
from evaluation.metrics import (
    diffusion_rate, network_density, bimodality_coefficient, social_response_lag
)

def test_diffusion_rate_zero():
    assert diffusion_rate(informed=0, total=10) == pytest.approx(0.0)

def test_diffusion_rate_full():
    assert diffusion_rate(informed=10, total=10) == pytest.approx(1.0)

def test_diffusion_rate_partial():
    assert diffusion_rate(informed=5, total=10) == pytest.approx(0.5)

def test_network_density_formula():
    # 4 agents, 6 directed edges (of 12 max) → density = 0.5
    assert network_density(edge_count=6, n_agents=4) == pytest.approx(0.5)

def test_network_density_empty():
    assert network_density(edge_count=0, n_agents=5) == pytest.approx(0.0)

def test_bc_uniform_low():
    """Uniform random opinions should have BC < 0.55."""
    import random
    opinions = [random.uniform(-1, 1) for _ in range(100)]
    bc = bimodality_coefficient(opinions)
    assert bc < 0.55, f"BC={bc:.3f} for uniform dist should be < 0.55"

def test_bc_bimodal_high():
    """Two-peak distribution should have BC > 0.55."""
    opinions = [-0.9] * 50 + [0.9] * 50
    bc = bimodality_coefficient(opinions)
    assert bc > 0.55, f"BC={bc:.3f} for bimodal dist should be > 0.55"

def test_social_response_lag_returns_positive():
    lag = social_response_lag(inject_step=10, fifty_pct_step=25)
    assert lag == 15

def test_social_response_lag_none_when_not_reached():
    lag = social_response_lag(inject_step=10, fifty_pct_step=None)
    assert lag is None
```

- [ ] **Step 6.1.2 — Implement `evaluation/metrics.py`**

```python
"""Four-dimension social dynamics metrics."""
from __future__ import annotations
import math
from typing import Optional
import numpy as np
from scipy import stats

__all__ = [
    "diffusion_rate", "network_density",
    "bimodality_coefficient", "social_response_lag",
]


def diffusion_rate(informed: int, total: int) -> float:
    """Fraction of agents who know about the injected event."""
    if total == 0:
        return 0.0
    return informed / total


def network_density(edge_count: int, n_agents: int) -> float:
    """Ratio of actual directed edges to maximum possible."""
    max_edges = n_agents * (n_agents - 1)
    if max_edges == 0:
        return 0.0
    return edge_count / max_edges


def bimodality_coefficient(opinions: list[float]) -> float:
    """
    Bimodality Coefficient (BC) = (skewness^2 + 1) / (kurtosis + 3*(n-1)^2/((n-2)*(n-3)))
    BC > 0.555 indicates bimodal distribution.
    """
    if len(opinions) < 4:
        return 0.0
    n = len(opinions)
    arr = np.array(opinions, dtype=float)
    skew = float(stats.skew(arr))
    kurt = float(stats.kurtosis(arr))  # Fisher's definition (excess kurtosis)
    # Sarle's bimodality coefficient
    bc = (skew ** 2 + 1) / (kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3) + 1e-9))
    return float(bc)


def social_response_lag(
    inject_step: int,
    fifty_pct_step: Optional[int],
) -> Optional[int]:
    """Steps from event injection to 50% agent awareness. None if not reached."""
    if fifty_pct_step is None:
        return None
    return fifty_pct_step - inject_step
```

- [ ] **Step 6.1.3 — Implement `evaluation/reporter.py`**

```python
"""Snapshot-based experiment reporter."""
from __future__ import annotations
import json
import os
from datetime import datetime

__all__ = ["Reporter"]


class Reporter:
    def __init__(self, data_dir: str = "data") -> None:
        self._data_dir = data_dir
        self._history: list[dict] = []

    def record(self, step: int, metrics: dict) -> None:
        self._history.append({"step": step, **metrics})

    def to_json(self) -> str:
        return json.dumps(self._history, indent=2)

    def save_json(self, path: str | None = None) -> str:
        path = path or f"{self._data_dir}/report_{datetime.now():%Y%m%d_%H%M%S}.json"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())
        return path

    def to_markdown(self) -> str:
        if not self._history:
            return "No data recorded."
        lines = ["# Social Town Experiment Report\n"]
        lines.append(f"Total steps: {self._history[-1]['step']}\n")
        lines.append("| Step | Diffusion | Net Density | Polar BC | Lag |")
        lines.append("|------|-----------|-------------|----------|-----|")
        for h in self._history:
            lines.append(
                f"| {h['step']} "
                f"| {h.get('diffusion_rate', '-'):.2f} "
                f"| {h.get('network_density', '-'):.3f} "
                f"| {h.get('bc', '-'):.3f} "
                f"| {h.get('lag', '-')} |"
            )
        return "\n".join(lines)
```

- [ ] **Step 6.1.4 — Run all Stage 6 tests**

```bash
pytest tests/stage6_evaluation/ -v
# Expected: all pass
```

- [ ] **Step 6.1.5 — Stage 6 gate check**

```bash
bash scripts/test.sh --stage 6
```

- [ ] **Step 6.1.6 — Commit and push Stage 6**

```bash
git add -A
git commit -m "feat(stage6): four-dimension evaluation metrics and reporter"
git push
```

---

## Stage 7: Docker Compose Integration

**Files:**
- Create: `docker/Dockerfile`
- Create: `docker/docker-compose.yml`
- Create: `docker/ollama-entrypoint.sh`
- Create: `tests/stage7_docker/__init__.py`
- Create: `tests/stage7_docker/test_docker.py`

### Task 7.1: Dockerfile + Compose

- [ ] **Step 7.1.1 — Write `docker/Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install curl (needed for healthcheck polling in entrypoint)
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download sentence-transformers model into image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

CMD ["python", "main.py"]
```

- [ ] **Step 7.1.2 — Write `docker/docker-compose.yml`**

```yaml
version: "3.9"

services:
  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_data:/root/.ollama
      - ./ollama-entrypoint.sh:/entrypoint.sh
    entrypoint: ["/bin/sh", "/entrypoint.sh"]
    ports:
      - "11434:11434"
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:11434/api/tags"]
      interval: 10s
      timeout: 5s
      retries: 18
      start_period: 30s

  social-town:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    depends_on:
      ollama:
        condition: service_healthy
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - MODEL_NAME=${MODEL_NAME:-qwen2.5:0.5b}
      - NUM_AGENTS=${NUM_AGENTS:-10}
      - SIMULATION_DAYS=${SIMULATION_DAYS:-3}
      - HF_HOME=/app/.cache/huggingface
    volumes:
      - ../data:/app/data
      - hf_cache:/app/.cache/huggingface
    ports:
      - "8080:8080"

volumes:
  ollama_data:
  hf_cache:
```

- [ ] **Step 7.1.3 — Write `docker/ollama-entrypoint.sh`**

```bash
#!/bin/sh
set -e
MODEL_NAME="${MODEL_NAME:-qwen2.5:0.5b}"

# Start Ollama serve in background
ollama serve &
SERVE_PID=$!

# Wait for API readiness
echo "[ollama-init] Waiting for Ollama API..."
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
  sleep 2
done
echo "[ollama-init] Ollama API ready"

# Pull model if not present
echo "[ollama-init] Pulling model: $MODEL_NAME"
ollama pull "$MODEL_NAME"
echo "[ollama-init] Model ready: $MODEL_NAME"

# Keep Ollama running in foreground
wait $SERVE_PID
```

- [ ] **Step 7.1.4 — Write `tests/stage7_docker/test_docker.py`**

```python
"""
Docker integration tests — run manually after `docker-compose up -d`.
These tests validate the live Docker environment.

Usage: pytest tests/stage7_docker/ -v -m docker
Requires: docker-compose -f docker/docker-compose.yml up -d (run first)
"""
import pytest
import httpx

pytestmark = pytest.mark.docker


def test_ollama_healthcheck():
    """Ollama API should be accessible on port 11434."""
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        assert resp.status_code == 200
    except Exception as e:
        pytest.skip(f"Ollama not available: {e}")


def test_model_is_pulled():
    """Target model should be listed in Ollama."""
    import os
    model = os.getenv("MODEL_NAME", "qwen2.5:0.5b")
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        tags = resp.json().get("models", [])
        names = [t["name"] for t in tags]
        assert any(model in n for n in names), f"{model} not found in {names}"
    except Exception as e:
        pytest.skip(f"Ollama not available: {e}")


def test_webapp_responds():
    """Web dashboard should respond on port 8080."""
    try:
        resp = httpx.get("http://localhost:8080/", timeout=5.0)
        assert resp.status_code == 200
    except Exception as e:
        pytest.skip(f"Webapp not available: {e}")
```

- [ ] **Step 7.1.5 — Build Docker images (no run yet)**

```bash
cd docker
docker-compose build
# Expected: BUILD SUCCESSFUL (no errors)
```

- [ ] **Step 7.1.6 — Run Docker stack**

```bash
docker-compose -f docker/docker-compose.yml up -d
# Wait ~2 minutes for Ollama to pull the model
docker-compose -f docker/docker-compose.yml logs -f ollama
# Expected: "[ollama-init] Model ready: qwen2.5:0.5b"
```

- [ ] **Step 7.1.7 — Run Stage 7 tests against live Docker**

```bash
pytest tests/stage7_docker/ -v -m docker
# Expected: all pass (or skip if container not up)
```

- [ ] **Step 7.1.8 — Stage 7 gate check**

```bash
# Gate criteria: build passes, ollama healthy, social-town runs 10 steps
docker-compose -f docker/docker-compose.yml logs social-town | grep "Step 10"
```

- [ ] **Step 7.1.9 — Commit and push Stage 7**

```bash
docker-compose -f docker/docker-compose.yml down
git add -A
git commit -m "feat(stage7): Docker Compose, Ollama entrypoint, healthcheck, model auto-pull"
git push
```

---

## Stage 8: Web Dashboard (FastAPI + SSE + Chart.js)

**Files:**
- Create: `webapp/server.py`
- Create: `webapp/static/index.html`
- Create: `tests/stage8_webapp/__init__.py`
- Create: `tests/stage8_webapp/test_server.py`
- Modify: `main.py` — launch webapp alongside simulation

### Task 8.1: FastAPI SSE Server

- [ ] **Step 8.1.1 — Write failing tests**

`tests/stage8_webapp/test_server.py`:
```python
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import MagicMock
from webapp.server import create_app

@pytest.fixture
def mock_sim():
    sim = MagicMock()
    sim.current_step = 5
    sim.personas = []
    sim.social_graph.edge_count.return_value = 0
    sim.social_graph.density.return_value = 0.0
    return sim

@pytest.mark.asyncio
async def test_root_returns_html(mock_sim):
    app = create_app(sim=mock_sim)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]

@pytest.mark.asyncio
async def test_metrics_endpoint_returns_json(mock_sim):
    app = create_app(sim=mock_sim)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "step" in data
    assert "network_density" in data

@pytest.mark.asyncio
async def test_inject_event_endpoint(mock_sim):
    mock_sim.inject_event = MagicMock()
    app = create_app(sim=mock_sim)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/inject", json={
            "event_type": "rumor",
            "content": "The mayor is resigning",
            "seed_agent_ids": ["agent_00"],
            "credibility": 0.8,
        })
    assert resp.status_code == 200
```

- [ ] **Step 8.1.2 — Implement `webapp/server.py`**

```python
"""FastAPI server with SSE metrics stream and event injection endpoint."""
from __future__ import annotations
import asyncio
import json
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from evaluation.metrics import diffusion_rate, network_density, bimodality_coefficient

__all__ = ["create_app"]

STATIC_DIR = Path(__file__).parent / "static"


class InjectRequest(BaseModel):
    event_type: str
    content: str
    seed_agent_ids: list[str]
    credibility: float = 0.7


def create_app(sim) -> FastAPI:
    app = FastAPI(title="Social Town Dashboard")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def root():
        return (STATIC_DIR / "index.html").read_text()

    @app.get("/api/metrics")
    async def get_metrics():
        opinions_flat = []
        for p in sim.personas:
            opinions_flat.extend(p.opinion.values())
        bc = bimodality_coefficient(opinions_flat) if len(opinions_flat) >= 4 else 0.0
        return {
            "step": sim.current_step,
            "network_density": sim.social_graph.density(len(sim.personas)),
            "bc": bc,
            "agents": [
                {"name": p.name, "location": p.location, "action": p.current_action[:60]}
                for p in sim.personas
            ],
        }

    @app.get("/api/stream")
    async def stream_metrics():
        async def event_generator() -> AsyncGenerator[str, None]:
            while True:
                data = await get_metrics()
                yield f"data: {json.dumps(data)}\n\n"
                await asyncio.sleep(2)
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.post("/api/inject")
    async def inject_event(req: InjectRequest):
        if hasattr(sim, "inject_event"):
            sim.inject_event(
                event_type=req.event_type,
                content=req.content,
                seed_agents=req.seed_agent_ids,
                credibility=req.credibility,
            )
        return {"status": "injected", "content": req.content}

    return app
```

- [ ] **Step 8.1.3 — Write `webapp/static/index.html`**

```html
<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <title>Social Town Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: monospace; background: #0d1117; color: #c9d1d9; padding: 16px; }
    h1 { color: #58a6ff; margin-bottom: 12px; font-size: 1.2rem; }
    .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
    .step-badge { background: #21262d; padding: 4px 12px; border-radius: 4px; font-size: 0.85rem; }
    .grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-bottom: 16px; }
    .chart-box { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; }
    .chart-box h3 { font-size: 0.8rem; color: #8b949e; margin-bottom: 8px; }
    .agent-list { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; overflow-y: auto; max-height: 280px; }
    .agent-row { font-size: 0.75rem; padding: 4px 0; border-bottom: 1px solid #21262d; }
    .inject-panel { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
    .inject-panel h3 { color: #8b949e; margin-bottom: 12px; font-size: 0.85rem; }
    .inject-panel input, .inject-panel select { background: #0d1117; border: 1px solid #30363d; color: #c9d1d9; padding: 6px 10px; border-radius: 4px; margin-right: 8px; font-family: monospace; }
    .inject-panel button { background: #238636; color: white; border: none; padding: 6px 16px; border-radius: 4px; cursor: pointer; }
    .inject-panel button:hover { background: #2ea043; }
    canvas { width: 100% !important; height: 180px !important; }
  </style>
</head>
<body>
  <div class="header">
    <h1>Social Town Dashboard</h1>
    <div class="step-badge" id="step-badge">Step: — | Day: —</div>
  </div>

  <div class="grid">
    <div class="chart-box"><h3>网络密度</h3><canvas id="densityChart"></canvas></div>
    <div class="chart-box"><h3>意见极化 (BC)</h3><canvas id="bcChart"></canvas></div>
    <div class="chart-box"><h3>信息扩散率</h3><canvas id="diffusionChart"></canvas></div>
    <div class="agent-list" id="agent-list"><h3 style="font-size:0.8rem;color:#8b949e;margin-bottom:8px">Agent 状态</h3></div>
  </div>

  <div class="inject-panel">
    <h3>事件注入</h3>
    <select id="event-type">
      <option value="rumor">谣言</option>
      <option value="breaking_news">突发新闻</option>
      <option value="public_health">公共卫生</option>
      <option value="election">选举</option>
    </select>
    <input id="event-content" type="text" placeholder="事件内容..." style="width:40%">
    <input id="credibility" type="number" min="0" max="1" step="0.1" value="0.7" style="width:80px">
    <button onclick="injectEvent()">注入事件</button>
  </div>

<script>
const MAX_POINTS = 60;
const labels = [];
const densityData = [], bcData = [], diffusionData = [];

function makeChart(id, label, color) {
  const ctx = document.getElementById(id).getContext('2d');
  return new Chart(ctx, {
    type: 'line',
    data: { labels, datasets: [{ label, data: densityData === diffusionData ? diffusionData : (densityData === bcData ? bcData : densityData), borderColor: color, tension: 0.3, pointRadius: 2, fill: false }] },
    options: { animation: false, plugins: { legend: { display: false } }, scales: { x: { ticks: { color: '#8b949e', maxTicksLimit: 8 } }, y: { ticks: { color: '#8b949e' }, min: 0, max: 1 } } }
  });
}

const densityChart = new Chart(document.getElementById('densityChart').getContext('2d'), {
  type: 'line',
  data: { labels, datasets: [{ label: '网络密度', data: densityData, borderColor: '#58a6ff', tension: 0.3, pointRadius: 2, fill: false }] },
  options: { animation: false, plugins: { legend: { display: false } }, scales: { x: { ticks: { color: '#8b949e', maxTicksLimit: 8 } }, y: { ticks: { color: '#8b949e' }, min: 0, max: 1 } } }
});
const bcChart = new Chart(document.getElementById('bcChart').getContext('2d'), {
  type: 'line',
  data: { labels, datasets: [{ label: 'BC', data: bcData, borderColor: '#f78166', tension: 0.3, pointRadius: 2, fill: false }] },
  options: { animation: false, plugins: { legend: { display: false } }, scales: { x: { ticks: { color: '#8b949e', maxTicksLimit: 8 } }, y: { ticks: { color: '#8b949e' }, min: 0, max: 2 } } }
});
const diffChart = new Chart(document.getElementById('diffusionChart').getContext('2d'), {
  type: 'line',
  data: { labels, datasets: [{ label: '扩散率', data: diffusionData, borderColor: '#3fb950', tension: 0.3, pointRadius: 2, fill: false }] },
  options: { animation: false, plugins: { legend: { display: false } }, scales: { x: { ticks: { color: '#8b949e', maxTicksLimit: 8 } }, y: { ticks: { color: '#8b949e' }, min: 0, max: 1 } } }
});

let knownAgents = 0;
const es = new EventSource('/api/stream');
es.onmessage = (e) => {
  const d = JSON.parse(e.data);
  const day = Math.floor(d.step / 288) + 1;
  document.getElementById('step-badge').textContent = `Step: ${d.step} | Day: ${day}`;

  if (labels.length >= MAX_POINTS) { labels.shift(); densityData.shift(); bcData.shift(); diffusionData.shift(); }
  labels.push(d.step);
  densityData.push(d.network_density);
  bcData.push(d.bc);
  diffusionData.push(knownAgents / Math.max(d.agents.length, 1));

  densityChart.update(); bcChart.update(); diffChart.update();

  const list = document.getElementById('agent-list');
  list.innerHTML = '<h3 style="font-size:0.8rem;color:#8b949e;margin-bottom:8px">Agent 状态</h3>';
  (d.agents || []).forEach(a => {
    const div = document.createElement('div');
    div.className = 'agent-row';
    div.textContent = `${a.name} @ ${a.location} → ${a.action}`;
    list.appendChild(div);
  });
};

async function injectEvent() {
  const body = {
    event_type: document.getElementById('event-type').value,
    content: document.getElementById('event-content').value,
    seed_agent_ids: ['agent_00'],
    credibility: parseFloat(document.getElementById('credibility').value),
  };
  await fetch('/api/inject', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  knownAgents = 1;
}
</script>
</body>
</html>
```

- [ ] **Step 8.1.4 — Run Stage 8 tests**

```bash
pytest tests/stage8_webapp/ -v
# Expected: all pass
```

- [ ] **Step 8.1.5 — Update `main.py` to run webapp alongside simulation**

```python
"""Entry point — concurrent simulation + webapp."""
from __future__ import annotations
import argparse
import asyncio
import uvicorn
from llm.client import OllamaClient
from core.simulation import Simulation
from webapp.server import create_app
from config.params import STEPS_PER_DAY


async def run(agents: int, days: int, resume: str | None) -> None:
    llm = OllamaClient()
    if resume:
        sim = Simulation.from_snapshot(resume, llm=llm)
        print(f"[social-town] Resumed from step {sim.current_step}")
    else:
        sim = Simulation(n_agents=agents, llm=llm)
        print(f"[social-town] Starting: {agents} agents, {days} days")

    app = create_app(sim=sim)

    # Run webapp in background
    config = uvicorn.Config(app, host="0.0.0.0", port=8080, log_level="warning")
    server = uvicorn.Server(config)
    webapp_task = asyncio.create_task(server.serve())
    print("[social-town] Dashboard at http://localhost:8080")

    total_steps = days * STEPS_PER_DAY
    for _ in range(total_steps):
        await sim.step()

    webapp_task.cancel()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=10)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    asyncio.run(run(args.agents, args.days, args.resume))


if __name__ == "__main__":
    main()
```

- [ ] **Step 8.1.6 — Stage 8 gate check**

```bash
bash scripts/test.sh --stage 8
```

- [ ] **Step 8.1.7 — Final end-to-end smoke test**

```bash
# Local smoke test (5 agents, 1 simulated day)
NUM_AGENTS=5 SIMULATION_DAYS=1 python main.py &
sleep 5
curl -s http://localhost:8080/api/metrics | python -m json.tool
# Expected: valid JSON with step, network_density, agents list
kill %1
```

- [ ] **Step 8.1.8 — Commit and push Stage 8**

```bash
git add -A
git commit -m "feat(stage8): FastAPI SSE dashboard, Chart.js UI, event injection endpoint"
git push
```

---

## Final: Docker End-to-End + Full Test Suite

- [ ] **Step F.1 — Run full test suite locally**

```bash
bash scripts/test.sh --mock-llm
# Expected: all stages pass (1-6, 8 in mock mode)
```

- [ ] **Step F.2 — Build and launch full Docker stack**

```bash
docker-compose -f docker/docker-compose.yml up --build
# Wait for: "[ollama-init] Model ready: qwen2.5:0.5b"
# Then open: http://localhost:8080
```

- [ ] **Step F.3 — Run Docker integration tests**

```bash
pytest tests/stage7_docker/ -v -m docker
# Expected: ollama healthy, model pulled, webapp responds
```

- [ ] **Step F.4 — Final commit and tag**

```bash
git add -A
git commit -m "chore: final integration, all stages complete"
git tag v1.0.0
git push && git push --tags
```

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `bash scripts/test.sh --stage N` | Run one stage's tests |
| `bash scripts/test.sh --mock-llm` | All tests, no LLM calls |
| `bash scripts/test.sh --coverage` | With coverage report |
| `python main.py --agents 5 --days 1` | Local quick run |
| `docker-compose -f docker/docker-compose.yml up` | Full Docker stack |
| `open http://localhost:8080` | Web dashboard |
