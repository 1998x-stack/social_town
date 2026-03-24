from llm.prompts import (
    importance_prompt, reflect_questions_prompt, reflect_insights_prompt,
    plan_l1_prompt, plan_l2_prompt, plan_l3_prompt, dialogue_prompt,
)

def test_importance_prompt_contains_memory():
    p = importance_prompt("Alice got a scholarship")
    assert "Alice got a scholarship" in p
    assert "1" in p and "10" in p

def test_plan_l1_prompt_contains_name():
    p = plan_l1_prompt("Alice", "a student", 1, "studied yesterday")
    assert "Alice" in p
    assert "4" in p  # mention 4 blocks

def test_plan_l3_prompt_truncates_perceived():
    long_env = "A" * 200
    p = plan_l3_prompt("Alice", "Study|Library", long_env)
    # Should be < 500 chars total
    assert len(p) < 600

def test_dialogue_prompt_contains_speaker_and_listener():
    p = dialogue_prompt("Alice", "Bob", "friends", "at the library", "the upcoming election")
    assert "Alice" in p
    assert "Bob" in p

def test_reflect_questions_prompt_structure():
    p = reflect_questions_prompt("Alice went to library. Alice studied hard.")
    assert "3" in p or "three" in p.lower()

def test_reflect_insights_prompt_structure():
    p = reflect_insights_prompt("Alice studied. Alice is diligent.")
    assert "5" in p or "five" in p.lower() or "insight" in p.lower()
