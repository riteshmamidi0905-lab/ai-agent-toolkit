"""Policy backends that drive the agent.

`RuleBasedPlanner` is a deterministic, offline policy: it parses the task,
plans a tool sequence, and — crucially — feeds real observations from earlier
tools into later steps (true ReAct behaviour), all with no network. It solves
the benchmark reliably, which makes the whole agent testable in CI.

For open-ended tasks, `aiagent.cloud` provides Gemini / OpenAI backends that
implement the same `.decide()` interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .agent import Action, Final, Step, TraceEntry
from .tools import Tool, _FACTS

_UNIT = r"(km|mi|kg|lb|m|ft|c|f)"
_NUM = r"-?\d+(?:\.\d+)?"

_WORD_OPS = {
    "double": ("*", 2), "doubled": ("*", 2), "triple": ("*", 3),
    "tripled": ("*", 3), "halve": ("/", 2), "halved": ("/", 2),
    "square": ("**", 2), "squared": ("**", 2),
}
_PHRASE_OPS = [
    (r"(?:multiply(?:\s+it)?\s+by|times)\s+(" + _NUM + r")", "*"),
    (r"(?:divide(?:\s+it)?\s+by)\s+(" + _NUM + r")", "/"),
    (r"(?:add|plus|increase(?:\s+it)?\s+by)\s+(" + _NUM + r")", "+"),
    (r"(?:subtract|minus|decrease(?:\s+it)?\s+by)\s+(" + _NUM + r")", "-"),
]


@dataclass
class _Plan:
    kind: str                      # "simple" | "compose"
    tool: str = ""
    tool_input: str = ""
    op: Optional[str] = None
    operand: Optional[float] = None


def _first_number(s: str) -> Optional[float]:
    m = re.search(_NUM, s)
    return float(m.group()) if m else None


class RuleBasedPlanner:
    """Deterministic offline policy — parses task, routes to tools."""

    # ---- parsing helpers ----
    def _trailing_op(self, task: str) -> Optional[Tuple[str, float]]:
        t = task.lower()
        for word, (op, operand) in _WORD_OPS.items():
            if re.search(rf"\b{word}\b", t):
                return op, float(operand)
        for pat, op in _PHRASE_OPS:
            m = re.search(pat, t)
            if m:
                return op, float(m.group(1))
        return None

    def _convert_expr(self, task: str) -> Optional[str]:
        m = re.search(rf"({_NUM})\s*{_UNIT}\s*(?:to|in)\s*{_UNIT}", task.lower())
        if m:
            return f"{m.group(1)} {m.group(2)} to {m.group(3)}"
        return None

    def _knowledge_query(self, task: str) -> Optional[str]:
        t = task.lower().strip("? ")
        for key in _FACTS:
            if key in t:
                return key
        if t.startswith(("what is the", "who is the", "how")) and \
                any(k.split()[0] in t for k in _FACTS):
            return t.replace("what is the", "").replace("who is the", "").strip()
        return None

    def _looks_arithmetic(self, task: str) -> Optional[str]:
        # extract a bare arithmetic expression if the task is a calculation
        m = re.search(r"[-+/*(). \d]+\d", task)
        if m and re.search(r"[-+*/]", m.group()) and re.search(r"\d", m.group()):
            expr = m.group().strip()
            # must be a balanced-ish arithmetic snippet
            if re.fullmatch(r"[\d+\-*/(). ]+", expr):
                return expr
        return None

    # ---- planning ----
    def _plan(self, task: str) -> _Plan:
        t = task.lower()
        mod = self._trailing_op(task)
        conv = self._convert_expr(task)
        if conv:
            if mod:
                return _Plan("compose", "unit_convert", conv, mod[0], mod[1])
            return _Plan("simple", "unit_convert", conv)
        kq = self._knowledge_query(task)
        if kq:
            return _Plan("simple", "knowledge_base", kq)
        if "word" in t or "character" in t or "how many words" in t:
            m = re.search(r"[:\"'](.+)[\"']?$", task)
            payload = m.group(1).strip("\"' ") if m else task
            return _Plan("simple", "text_stats", payload)
        expr = self._looks_arithmetic(task)
        if expr:
            return _Plan("simple", "calculator", expr)
        # fallback: try knowledge base
        return _Plan("simple", "knowledge_base", task)

    # ---- policy interface ----
    def decide(self, task: str, history: List[TraceEntry],
               tools: Dict[str, Tool]) -> Step:
        plan = self._plan(task)
        n = len(history)
        if plan.kind == "simple":
            if n == 0:
                return Action(f"I'll use {plan.tool} to answer this.",
                              plan.tool, plan.tool_input)
            return Final("I have the result.", history[-1].observation or "")
        # compose: base tool, then apply the arithmetic modifier to its result
        if n == 0:
            return Action("First run the base tool, then adjust the result.",
                          plan.tool, plan.tool_input)
        if n == 1:
            num = _first_number(history[-1].observation or "")
            expr = f"{num} {plan.op} {plan.operand}"
            return Action(f"Now apply {plan.op}{plan.operand} to {num}.",
                          "calculator", expr)
        return Final("I have the composed result.",
                     history[-1].observation or "")
