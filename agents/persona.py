"""Agent Persona: cognitive loop entry point."""
from __future__ import annotations
import inspect
import logging
from typing import Any
from agents.memory.memory_stream import MemoryObject, MemoryStream
from agents.memory.retrieval import Retriever
from agents.cognitive.plan import PlanningEngine
from agents.cognitive.reflect import ReflectionEngine
from config.params import MEMORY_TOP_K, REFLECTION_THRESHOLD

__all__ = ["Persona"]

logger = logging.getLogger(__name__)


class Persona:
    def __init__(self, agent_id: str, name: str, description: str, llm: Any) -> None:
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
        logger.info(f"[Step {current_step}] Agent {self.name}: perceive={observation[:50]}")

    async def start_day(self, day: int, current_step: int) -> None:
        recent = self.memory.recent(3)
        top3 = "; ".join(m.content for m in recent)
        await self._planner.generate_l1(day=day, top3_memories=top3)
        recent_top = self.memory.recent(3)
        top3_l2 = "; ".join(m.content for m in recent_top)
        await self._planner.generate_l2(top3_memories=top3_l2)
        logger.info(f"[Plan] Agent {self.name} started day {day}")

    async def act(self, current_step: int, perceived_top3: str) -> str:
        action = await self._planner.generate_l3(perceived_top3=perceived_top3)
        self.current_action = action
        logger.info(f"[Step {current_step}] Agent {self.name}: {action}")
        return action

    async def maybe_reflect(self, current_step: int) -> bool:
        if not self.memory.should_reflect(REFLECTION_THRESHOLD):
            return False
        logger.info(f"[Reflect] {self.name}: 触发反思（累积阈值已达）")
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
    def from_snapshot(cls, data: dict, llm: Any) -> "Persona":
        p = cls(
            agent_id=data["id"],
            name=data["name"],
            description=data["description"],
            llm=llm,
        )
        p.location = data["location"]
        p.current_action = data["current_action"]
        p.opinion = data["opinion"]
        st_model = getattr(llm, "_st_model", None)
        _use_real_model = False
        if st_model is not None and not isinstance(st_model, inspect.Parameter.__class__):
            encode_fn = getattr(st_model, "encode", None)
            if encode_fn is not None and callable(encode_fn):
                # Check without actually calling (avoid unawaited coroutine warning)
                # Real SentenceTransformer.encode is a regular sync function
                _coro = None
                try:
                    _coro = encode_fn("test")
                    if not inspect.isawaitable(_coro):
                        _use_real_model = True
                finally:
                    if inspect.isawaitable(_coro):
                        _coro.close()  # cleanly discard mock coroutine
        if _use_real_model:
            embed_fn_sync = lambda t: st_model.encode(t).tolist()
        else:
            embed_fn_sync = lambda t: [0.0] * 384
        p.memory = MemoryStream.from_dict_list(data["memories"], embed_fn_sync)
        p.memory._reflection_accumulator = data["reflection_accumulator"]
        p._planner.state.l1_blocks = data["l1_plan"]
        return p
