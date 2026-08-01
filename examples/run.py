"""Run the agent on a benchmark, print traces, build a dashboard.

    python examples/run.py
"""

from pathlib import Path

from aiagent import default_tools, Agent, RuleBasedPlanner, run_benchmark, llm_available
from aiagent import viz

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    agent = Agent(default_tools(), RuleBasedPlanner())

    demo = "Convert 5 km to mi then multiply by 2"
    res = agent.run(demo)
    print(f"Task: {demo}")
    for h in res.trace:
        print(f"  💭 {h.thought}\n    {h.tool}({h.tool_input}) -> {h.observation}")
    print(f"  ✅ {res.answer}\n")

    bench = run_benchmark(agent)
    print(f"Benchmark: {bench.passed}/{bench.total} passed "
          f"({bench.success_rate*100:.0f}%)")
    print("LLM backend available:", llm_available())

    (ROOT / "docs").mkdir(exist_ok=True)
    out = viz.build_dashboard(bench, demo, res, ROOT / "docs" / "dashboard.html")
    print("Wrote", out)


if __name__ == "__main__":
    main()
