from __future__ import annotations
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
