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

def test_memory_stream_recent():
    ms = MemoryStream()
    for i in range(5):
        ms.add(MemoryObject(
            content=f"event {i}", memory_type="observation", created_at=i,
            importance=1.0, embedding=[0.0]*384, source_agent=None, credibility=1.0
        ))
    recent = ms.recent(3)
    assert len(recent) == 3
    assert recent[-1].content == "event 4"

def test_memory_stream_serialization_roundtrip():
    ms = MemoryStream()
    m = MemoryObject(
        content="serialize me", memory_type="reflection", created_at=5,
        importance=7.0, embedding=[0.5]*384, source_agent="agent_1", credibility=0.8
    )
    ms.add(m)
    data = ms.to_dict_list()
    assert len(data) == 1
    assert data[0]["content"] == "serialize me"
    assert data[0]["source_agent"] == "agent_1"
    assert data[0]["id"] == m.id

    embed_fn = lambda text: [0.5] * 384
    ms2 = MemoryStream.from_dict_list(data, embed_fn)
    restored = ms2.all()[0]
    assert restored.content == "serialize me"
    assert restored.id == m.id
    assert restored.last_accessed == 5
    assert restored.source_agent == "agent_1"
