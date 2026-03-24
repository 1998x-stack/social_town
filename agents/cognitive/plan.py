"""Three-layer planning engine: L1 (daily) → L2 (block) → L3 (action)."""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from llm.prompts import plan_l1_prompt, plan_l2_prompt, plan_l3_prompt

__all__ = ["PlanningEngine", "PlanState"]

logger = logging.getLogger(__name__)


@dataclass
class PlanState:
    l1_blocks: list[str] = field(default_factory=list)  # ["TimeSlot|Activity|Location"]
    l1_index: int = 0                                    # current block index
    l2_activities: list[str] = field(default_factory=list)  # ["Activity|Location"]
    l2_index: int = 0


class PlanningEngine:
    def __init__(self, name: str, description: str, llm: object) -> None:
        self._name = name
        self._desc = description
        self._llm = llm
        self.state = PlanState()

    async def generate_l1(self, day: int, top3_memories: str) -> None:
        """Generate daily schedule (4 activity blocks)."""
        prompt = plan_l1_prompt(self._name, self._desc, day, top3_memories)
        fallback = "Morning|Rest|Home\nAfternoon|Rest|Home\nEvening|Rest|Home\nNight|Sleep|Home"
        raw = await self._llm.generate(prompt, fallback=fallback)
        blocks = [line.strip() for line in raw.strip().splitlines() if "|" in line]
        while len(blocks) < 4:
            blocks.append("Free time|Wander|Town")
        self.state.l1_blocks = blocks[:4]
        self.state.l1_index = 0
        self.state.l2_activities = []
        self.state.l2_index = 0
        logger.debug(f"[Plan] {self._name} L1 day={day}: {self.state.l1_blocks}")

    def current_l1_block(self) -> str:
        if not self.state.l1_blocks:
            return "Unknown|Idle|Home"
        idx = min(self.state.l1_index, len(self.state.l1_blocks) - 1)
        return self.state.l1_blocks[idx]

    async def generate_l2(self, top3_memories: str) -> None:
        """Generate activities for the current L1 block (3 activities)."""
        block = self.current_l1_block()
        prompt = plan_l2_prompt(self._name, block, top3_memories)
        raw = await self._llm.generate(prompt, fallback="Idle|Home\nIdle|Home\nIdle|Home")
        activities = [line.strip() for line in raw.strip().splitlines() if "|" in line]
        while len(activities) < 3:
            activities.append("Rest|Home")
        self.state.l2_activities = activities[:3]
        self.state.l2_index = 0
        logger.debug(f"[Plan] {self._name} L2 block={block}: {self.state.l2_activities}")

    def current_l2_activity(self) -> str:
        if not self.state.l2_activities:
            return "Idle|Home"
        idx = min(self.state.l2_index, len(self.state.l2_activities) - 1)
        return self.state.l2_activities[idx]

    async def generate_l3(self, perceived_top3: str) -> str:
        """Generate the current micro-action sentence."""
        activity = self.current_l2_activity()
        prompt = plan_l3_prompt(self._name, activity, perceived_top3)
        fallback = f"{self._name} continues their current activity."
        action = await self._llm.generate(prompt, fallback=fallback)
        logger.info(f"[Plan] {self._name} L3 action: {action}")
        return action

    def replan_l2(self) -> None:
        """Discard current L2 plan; caller must call generate_l2 again."""
        self.state.l2_activities = []
        self.state.l2_index = 0
        logger.info(f"[Plan] {self._name} L2 replanned (reset)")
