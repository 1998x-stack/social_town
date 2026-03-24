import pytest
import math
from unittest.mock import AsyncMock
from agents.cognitive.plan import PlanningEngine, PlanState

FAKE_L1 = "Morning|Breakfast|Home\nAfternoon|Study|Library\nEvening|Walk|Park\nNight|Read|Home"
FAKE_L2 = "Read economics textbook|Library\nTake notes|Library\nDiscuss with peer|Library"
FAKE_L3 = "Alice sits down and opens her textbook."


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
