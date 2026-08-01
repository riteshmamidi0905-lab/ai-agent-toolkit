"""The tools an agent can call. Each is a small, safe, well-typed function.

Tools are the agent's hands: a safe calculator (AST-evaluated, no `eval`),
a unit converter, a tiny retrieval knowledge base, and text statistics. New
tools are registered by wrapping a function in `Tool`.
"""

from __future__ import annotations

import ast
import operator as op
from dataclasses import dataclass
from typing import Callable, Dict, List

# ---- safe arithmetic (no eval) ------------------------------------------
_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.USub: op.neg, ast.Mod: op.mod, ast.FloorDiv: op.floordiv,
}


def _eval(node):
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError("only numbers allowed")
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '12 * 7 + 5'."""
    tree = ast.parse(expression.strip(), mode="eval")
    result = _eval(tree.body)
    return f"{result:.4g}"


# ---- unit conversion -----------------------------------------------------
_CONV = {
    ("km", "mi"): lambda x: x * 0.621371, ("mi", "km"): lambda x: x / 0.621371,
    ("kg", "lb"): lambda x: x * 2.20462, ("lb", "kg"): lambda x: x / 2.20462,
    ("m", "ft"): lambda x: x * 3.28084, ("ft", "m"): lambda x: x / 3.28084,
    ("c", "f"): lambda x: x * 9 / 5 + 32, ("f", "c"): lambda x: (x - 32) * 5 / 9,
}


def unit_convert(query: str) -> str:
    """Convert a value between units, e.g. '10 km to mi'."""
    import re
    m = re.match(r"\s*(-?\d+(?:\.\d+)?)\s*([a-zA-Z]+)\s*(?:to|in)\s*([a-zA-Z]+)",
                 query.strip())
    if not m:
        raise ValueError("format: '<value> <unit> to <unit>'")
    val, src, dst = float(m.group(1)), m.group(2).lower(), m.group(3).lower()
    if (src, dst) not in _CONV:
        raise ValueError(f"no conversion {src}->{dst}")
    return f"{_CONV[(src, dst)](val):.4g} {dst}"


# ---- tiny retrieval knowledge base --------------------------------------
_FACTS = {
    "capital of france": "Paris", "capital of japan": "Tokyo",
    "capital of india": "New Delhi", "capital of italy": "Rome",
    "largest planet": "Jupiter", "speed of light": "299,792 km/s",
    "author of hamlet": "William Shakespeare", "chemical symbol for gold": "Au",
}


def knowledge_base(query: str) -> str:
    """Look up a fact by keyword, e.g. 'capital of France'."""
    q = query.lower().strip("? ")
    if q in _FACTS:
        return _FACTS[q]
    # fuzzy: highest keyword overlap
    best, score = None, 0
    qset = set(q.split())
    for k, v in _FACTS.items():
        s = len(qset & set(k.split()))
        if s > score:
            best, score = v, s
    if best and score >= 2:
        return best
    return "unknown"


def text_stats(text: str) -> str:
    """Return word and character counts for a piece of text."""
    words = len(text.split())
    return f"{words} words, {len(text)} chars"


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[[str], str]

    def __call__(self, arg: str) -> str:
        return self.func(arg)


def default_tools() -> Dict[str, Tool]:
    tools: List[Tool] = [
        Tool("calculator", "evaluate an arithmetic expression like '2*(3+4)'", calculator),
        Tool("unit_convert", "convert units, e.g. '10 km to mi'", unit_convert),
        Tool("knowledge_base", "look up a fact, e.g. 'capital of France'", knowledge_base),
        Tool("text_stats", "count words and characters in text", text_stats),
    ]
    return {t.name: t for t in tools}
