"""A small task suite to measure agent success rate end-to-end."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .agent import Agent

# (task, expected substring in the answer)
TASKS: List[Tuple[str, str]] = [
    ("What is 12 * 7 + 5?", "89"),
    ("Compute (3 + 4) * 2 ** 3", "56"),
    ("Convert 10 km to mi", "6.214"),
    ("Convert 100 c to f", "212"),
    ("Convert 5 km to mi then multiply by 2", "6.214"),   # 3.107*2
    ("Convert 26.2 mi to km", "42.16"),
    ("What is the capital of France?", "Paris"),
    ("What is the largest planet?", "Jupiter"),
    ("Convert 10 km to mi and add 10", "16.21"),
    ("What is 2 ** 10?", "1024"),
]


@dataclass
class BenchmarkResult:
    total: int
    passed: int
    details: List[dict]

    @property
    def success_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


def run_benchmark(agent: Agent) -> BenchmarkResult:
    details = []
    passed = 0
    for task, expected in TASKS:
        res = agent.run(task)
        ok = expected in res.answer
        passed += int(ok)
        details.append({"task": task, "expected": expected,
                        "answer": res.answer, "ok": ok, "steps": res.steps})
    return BenchmarkResult(len(TASKS), passed, details)
