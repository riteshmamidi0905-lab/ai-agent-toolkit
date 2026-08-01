import pytest

from aiagent import (
    default_tools, calculator, unit_convert, knowledge_base, text_stats,
    Agent, RuleBasedPlanner, Action, Final, run_benchmark, llm_available,
)


def test_calculator_safe_and_correct():
    assert calculator("12 * 7 + 5") == "89"
    assert calculator("(3 + 4) * 2 ** 3") == "56"
    with pytest.raises(Exception):
        calculator("__import__('os').system('ls')")   # no code execution


def test_unit_convert():
    assert unit_convert("10 km to mi").startswith("6.214")
    assert unit_convert("100 c to f").startswith("212")
    with pytest.raises(ValueError):
        unit_convert("10 km to lightyears")


def test_knowledge_base():
    assert knowledge_base("capital of France") == "Paris"
    assert knowledge_base("largest planet") == "Jupiter"
    assert knowledge_base("meaning of life") == "unknown"


def test_text_stats():
    assert text_stats("hello world") == "2 words, 11 chars"


@pytest.fixture(scope="module")
def agent():
    return Agent(default_tools(), RuleBasedPlanner())


def test_agent_single_step(agent):
    res = agent.run("What is 2 ** 10?")
    assert "1024" in res.answer
    assert res.succeeded and res.steps >= 1


def test_agent_knowledge(agent):
    res = agent.run("What is the capital of France?")
    assert "Paris" in res.answer


def test_agent_multistep_composition(agent):
    # conversion then arithmetic on the observation -> true ReAct feedback
    res = agent.run("Convert 5 km to mi then multiply by 2")
    assert "6.214" in res.answer
    assert res.steps == 2
    assert res.trace[0].tool == "unit_convert"
    assert res.trace[1].tool == "calculator"


def test_agent_trace_records_observations(agent):
    res = agent.run("Convert 10 km to mi and add 10")
    assert "16.21" in res.answer
    assert all(t.observation for t in res.trace)


def test_benchmark_all_pass(agent):
    bench = run_benchmark(agent)
    assert bench.success_rate == 1.0, [d for d in bench.details if not d["ok"]]


def test_unknown_tool_is_handled():
    planner = RuleBasedPlanner()
    agent = Agent(default_tools(), planner)
    # calculator with a malformed expression returns an error observation, not a crash
    res = agent.run("What is 5 plus")   # ambiguous; agent should still terminate
    assert isinstance(res.answer, str)


def test_llm_available_is_bool():
    assert isinstance(llm_available(), bool)
