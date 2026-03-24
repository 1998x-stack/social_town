from __future__ import annotations
import pytest
from world.event_injector import EventInjector
from agents.memory.memory_stream import MemoryStream


def test_inject_event_adds_to_seed_agent_memory():
    streams = {"a1": MemoryStream(), "a2": MemoryStream()}
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
    streams = {"a1": MemoryStream(), "a2": MemoryStream()}
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
    streams = {"a1": MemoryStream()}
    injector = EventInjector(agent_streams=streams)
    injector.inject_event(
        event_type="rumor", content="Low-cred rumor", seed_agents=["a1"],
        credibility=0.0, step=0,
        embed_fn=lambda t: [0.0]*384, importance_fn=lambda t: 2.0,
    )
    assert len(streams["a1"].all()) == 1
