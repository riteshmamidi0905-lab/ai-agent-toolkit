"""aiagent — a compact ReAct-style tool-using agent framework.

A backend-agnostic agent loop (Thought → Action → Observation → Answer), a
registry of safe tools (calculator, unit convert, knowledge base, text stats),
a deterministic offline planner for reproducible testing, a benchmark suite,
and optional Gemini / OpenAI LLM backends.
"""

from .tools import (
    Tool, default_tools, calculator, unit_convert, knowledge_base, text_stats,
)
from .agent import Agent, Action, Final, Result, TraceEntry
from .backend import RuleBasedPlanner
from .benchmark import run_benchmark, BenchmarkResult, TASKS
from .cloud import GeminiBackend, OpenAIBackend, llm_available

__all__ = [
    "Tool", "default_tools", "calculator", "unit_convert", "knowledge_base",
    "text_stats", "Agent", "Action", "Final", "Result", "TraceEntry",
    "RuleBasedPlanner", "run_benchmark", "BenchmarkResult", "TASKS",
    "GeminiBackend", "OpenAIBackend", "llm_available",
]
__version__ = "0.1.0"
