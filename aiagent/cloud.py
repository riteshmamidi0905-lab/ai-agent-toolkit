"""LLM policy backends (Gemini / OpenAI) implementing the same `.decide()` API.

These generalise beyond the offline planner's patterns. Each prompts the model
in ReAct format ("Thought/Action/Action Input" or "Final Answer") and parses
the reply into an `Action` or `Final`. Guarded imports keep them optional.

    pip install google-generativeai        # Gemini
    export GEMINI_API_KEY=...
    from aiagent.cloud import GeminiBackend
    agent = Agent(default_tools(), GeminiBackend())
"""

from __future__ import annotations

import re
from typing import Dict, List

from .agent import Action, Final, Step, TraceEntry
from .tools import Tool

REACT_SYSTEM = """You are a tool-using agent. Given a task, respond with either:
Thought: <reasoning>
Action: <tool name>
Action Input: <input>
—or, when you can answer:
Thought: <reasoning>
Final Answer: <answer>
Available tools:
{tools}
"""


def _format_prompt(task: str, history: List[TraceEntry],
                   tools: Dict[str, Tool]) -> str:
    tool_desc = "\n".join(f"- {t.name}: {t.description}" for t in tools.values())
    lines = [REACT_SYSTEM.format(tools=tool_desc), f"Task: {task}"]
    for h in history:
        lines.append(f"Thought: {h.thought}")
        lines.append(f"Action: {h.tool}\nAction Input: {h.tool_input}")
        lines.append(f"Observation: {h.observation}")
    return "\n".join(lines)


def _parse(reply: str) -> Step:
    thought_m = re.search(r"Thought:\s*(.+)", reply)
    thought = thought_m.group(1).strip() if thought_m else ""
    final_m = re.search(r"Final Answer:\s*(.+)", reply, re.S)
    if final_m:
        return Final(thought, final_m.group(1).strip())
    tool_m = re.search(r"Action:\s*(.+)", reply)
    input_m = re.search(r"Action Input:\s*(.+)", reply)
    return Action(thought,
                  tool_m.group(1).strip() if tool_m else "",
                  input_m.group(1).strip() if input_m else "")


class GeminiBackend:
    def __init__(self, model: str = "gemini-1.5-flash") -> None:
        try:
            import google.generativeai as genai  # type: ignore
        except Exception as exc:  # pragma: no cover - needs SDK
            raise RuntimeError(
                "google-generativeai not installed. "
                "pip install google-generativeai and set GEMINI_API_KEY."
            ) from exc
        import os
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        self._model = genai.GenerativeModel(model)

    def decide(self, task: str, history: List[TraceEntry],
               tools: Dict[str, Tool]) -> Step:  # pragma: no cover - needs API
        reply = self._model.generate_content(
            _format_prompt(task, history, tools)).text
        return _parse(reply)


class OpenAIBackend:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover - needs SDK
            raise RuntimeError(
                "openai not installed. pip install openai and set OPENAI_API_KEY."
            ) from exc
        self._client = OpenAI()
        self._model = model

    def decide(self, task: str, history: List[TraceEntry],
               tools: Dict[str, Tool]) -> Step:  # pragma: no cover - needs API
        reply = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user",
                       "content": _format_prompt(task, history, tools)}],
        ).choices[0].message.content
        return _parse(reply)


def llm_available() -> bool:
    for mod in ("google.generativeai", "openai"):
        try:
            __import__(mod)
            return True
        except Exception:
            continue
    return False
