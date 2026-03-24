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
