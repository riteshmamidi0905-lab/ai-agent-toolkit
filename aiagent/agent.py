"""A ReAct-style agent loop: Thought → Action → Observation → … → Answer.

The agent is backend-agnostic: it asks a *policy* (any object with a
`.decide(task, history, tools)` method) for the next step, executes tool
actions, feeds the observation back, and repeats until the policy emits a final
answer. Swap the deterministic offline planner for a Gemini/OpenAI backend
without touching this loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Union

from .tools import Tool


@dataclass
class Action:
    thought: str
    tool: str
    tool_input: str


@dataclass
class Final:
    thought: str
    answer: str


Step = Union[Action, Final]


@dataclass
class TraceEntry:
    thought: str
    tool: Optional[str]
    tool_input: Optional[str]
    observation: Optional[str]


@dataclass
class Result:
    answer: str
    trace: List[TraceEntry] = field(default_factory=list)
    steps: int = 0
    succeeded: bool = True


class Policy(Protocol):
    def decide(self, task: str,
               history: List[TraceEntry],
               tools: Dict[str, Tool]) -> Step: ...


class Agent:
    def __init__(self, tools: Dict[str, Tool], policy: Policy,
                 max_steps: int = 6) -> None:
        self.tools = tools
        self.policy = policy
        self.max_steps = max_steps

    def run(self, task: str) -> Result:
        history: List[TraceEntry] = []
        for _ in range(self.max_steps):
            step = self.policy.decide(task, history, self.tools)
            if isinstance(step, Final):
                return Result(step.answer, history, len(history), True)
            # execute the tool
            if step.tool not in self.tools:
                obs = f"error: unknown tool '{step.tool}'"
            else:
                try:
                    obs = self.tools[step.tool](step.tool_input)
                except Exception as exc:
                    obs = f"error: {exc}"
            history.append(TraceEntry(step.thought, step.tool,
                                      step.tool_input, obs))
        return Result("(no answer: step limit reached)", history,
                      len(history), False)
