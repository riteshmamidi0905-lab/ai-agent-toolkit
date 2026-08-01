"""Render a ReAct trace + benchmark results as a self-contained HTML dashboard."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .agent import Agent, Result


def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _fig_success(passed: int, total: int):
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.pie([passed, total - passed], labels=["passed", "failed"],
           colors=["#34A853", "#EA4335"], autopct=lambda p: f"{p*total/100:.0f}",
           startangle=90, wedgeprops={"width": 0.42})
    ax.set_title(f"Benchmark — {passed}/{total} tasks")
    return fig


def _trace_html(task: str, result: Result) -> str:
    steps = ""
    for i, h in enumerate(result.trace, 1):
        steps += f"""<div class="step">
          <div class="thought">💭 {h.thought}</div>
          <div class="act"><span class="tool">{h.tool}</span>
             <code>{h.tool_input}</code></div>
          <div class="obs">→ {h.observation}</div></div>"""
    return f"""<div class="trace"><div class="task">🧩 {task}</div>{steps}
      <div class="final">✅ {result.answer}</div></div>"""


def build_dashboard(bench, example_task: str, example_result: Result,
                    out_path: Path) -> Path:
    pie = _b64(_fig_success(bench.passed, bench.total))
    rows = "".join(
        f"<tr><td>{d['task']}</td><td>{d['answer']}</td>"
        f"<td>{'✅' if d['ok'] else '❌'}</td><td>{d['steps']}</td></tr>"
        for d in bench.details
    )
    trace = _trace_html(example_task, example_result)
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>AI Agent Toolkit — Dashboard</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0d1117;color:#e6edf3}}
 header{{padding:28px 32px;background:linear-gradient(90deg,#4285F4,#34A853)}}
 header h1{{margin:0;font-size:22px}} header p{{margin:6px 0 0;opacity:.9}}
 .grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px;padding:24px 32px}}
 .card{{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:18px}}
 .card h2{{font-size:13px;margin:0 0 12px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px}}
 .wide{{grid-column:1/-1}} img{{max-width:100%;border-radius:8px;background:#fff}}
 table{{width:100%;border-collapse:collapse;font-size:13px}}
 td,th{{padding:6px 10px;border-bottom:1px solid #21262d;text-align:left}}
 th{{color:#8b949e;font-size:11px;text-transform:uppercase}}
 .trace{{font-size:14px}} .task{{font-weight:700;margin-bottom:10px}}
 .step{{border-left:2px solid #30363d;padding:6px 0 6px 14px;margin:8px 0}}
 .thought{{color:#c9d1d9}} .act{{margin:4px 0}} .tool{{background:#1f6feb33;color:#79c0ff;border-radius:6px;padding:2px 8px;font-size:12px;margin-right:6px}}
 code{{background:#0d1117;padding:2px 6px;border-radius:5px}}
 .obs{{color:#7ee787}} .final{{margin-top:12px;font-weight:700;color:#7ee787}}
</style></head><body>
<header><h1>AI Agent Toolkit</h1>
<p>ReAct tool-using agent · calculator · unit convert · knowledge base · pluggable LLM</p></header>
<div class="grid">
 <div class="card"><h2>Benchmark success</h2><img src="data:image/png;base64,{pie}"></div>
 <div class="card"><h2>Example reasoning trace</h2>{trace}</div>
 <div class="card wide"><h2>All tasks</h2>
   <table><tr><th>Task</th><th>Answer</th><th>OK</th><th>Steps</th></tr>{rows}</table></div>
</div></body></html>"""
    out_path.write_text(html, encoding="utf-8")
    return out_path
