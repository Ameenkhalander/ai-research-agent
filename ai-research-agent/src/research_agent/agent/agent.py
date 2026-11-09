"""The research agent: plan -> execute (LLM-driven tool selection via MCP) -> synthesise a cited report."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Callable

from research_agent.agent.models import AgentResult, ToolProvider
from research_agent.agent.prompts import EXECUTOR_SYSTEM, PLANNER_SYSTEM, SYNTH_SYSTEM
from research_agent.agent.sources import SourceRegistry
from research_agent.llm import text_of, tool_uses
from research_agent.reports.generator import report_to_markdown

OBSERVATION_LIMIT = 8000
EVIDENCE_LIMIT = 80_000

REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "executive_summary": {"type": "string", "description": "2-4 sentences with citations"},
        "key_findings": {"type": "array", "minItems": 1, "items": {
            "type": "object",
            "properties": {"heading": {"type": "string"},
                           "content": {"type": "string", "description": "Cited prose or bullet list"}},
            "required": ["heading", "content"]}},
        "conclusion": {"type": "string"},
        "limitations": {"type": "array", "items": {"type": "string"}},
        "follow_up_questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "executive_summary", "key_findings", "conclusion", "limitations"],
}


def plan_schema(tool_names: list[str], max_tasks: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "objective": {"type": "string"},
            "tasks": {"type": "array", "minItems": 1, "maxItems": max_tasks, "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "description": {"type": "string"},
                    "suggested_tool": {"type": "string", "enum": [*tool_names, "none"]},
                    "rationale": {"type": "string"},
                },
                "required": ["id", "description", "suggested_tool"]}},
        },
        "required": ["objective", "tasks"],
    }


def _preview(value, limit: int = 600) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} more chars]"


class ResearchAgent:
    def __init__(self, llm, tools: ToolProvider, settings, on_step: Callable[[list, list | None], None] | None = None):
        self.llm = llm
        self.tools = tools
        self.settings = settings
        self.on_step = on_step
        self.sources = SourceRegistry()
        self.steps: list[dict] = []
        self.plan: list[dict] | None = None
        self._evidence: list[str] = []

    # ------------------------------------------------------------------ tracing
    def _log(self, kind: str, title: str, **fields) -> None:
        step = {"n": len(self.steps) + 1, "kind": kind, "title": title,
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **fields}
        self.steps.append(step)
        if self.on_step:
            self.on_step(self.steps, self.plan)

    # ------------------------------------------------------------------ main entry
    async def run(self, question: str, *, style: str = "detailed", context: list[dict] | None = None,
                  available_documents: list[str] | None = None) -> AgentResult:
        t0 = time.perf_counter()
        self.sources, self.steps, self.plan, self._evidence = SourceRegistry(), [], None, []

        specs = await self.tools.list_tools()
        self._log("mcp", f"Connected to MCP server ({len(specs)} tools)", output=", ".join(s.name for s in specs))
        anthropic_tools = [{"name": s.name, "description": s.description, "input_schema": s.input_schema}
                           for s in specs]

        plan = await self._plan(question, specs, style, context, available_documents or [])
        findings: list[tuple[dict, str]] = []
        for task in plan:
            findings.append((task, await self._execute_task(task, len(plan), question, findings,
                                                             available_documents or [], anthropic_tools)))

        report = await self._synthesize(question, findings, style)
        sources = self.sources.as_list()
        markdown = report_to_markdown(report, sources, question)
        duration = time.perf_counter() - t0
        self._log("done", f"Report complete ({len(sources)} sources, {duration:.1f}s)")
        return AgentResult(
            question=question, style=style, plan=plan, steps=self.steps, report=report, markdown=markdown,
            sources=sources, evidence="\n\n".join(self._evidence)[:EVIDENCE_LIMIT], duration_s=round(duration, 2))

    # ------------------------------------------------------------------ planning
    async def _plan(self, question, specs, style, context, documents) -> list[dict]:
        max_tasks = 3 if style == "concise" else self.settings.max_plan_tasks
        tool_lines = "\n".join(f"- {s.name}: {s.description.splitlines()[0] if s.description else ''}" for s in specs)
        prompt = f"Research question: {question}\n"
        if documents:
            prompt += "Uploaded documents available: " + ", ".join(documents) + "\n"
        else:
            prompt += "No documents have been uploaded.\n"
        if context:
            history = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in context[-6:])
            prompt += f"\nConversation so far (resolve follow-up references with it):\n{history}\n"
        data = await self.llm.structured(
            system=PLANNER_SYSTEM.format(tools=tool_lines, max_tasks=max_tasks), prompt=prompt,
            tool_name="submit_plan", description="Submit the research plan",
            schema=plan_schema([s.name for s in specs], max_tasks), max_tokens=1500)
        names = {s.name for s in specs}
        plan = []
        for i, t in enumerate((data.get("tasks") or [])[:max_tasks], 1):
            tool = t.get("suggested_tool") if t.get("suggested_tool") in names else "none"
            plan.append({"id": i, "description": str(t.get("description", "")).strip() or question,
                         "suggested_tool": tool, "rationale": t.get("rationale", "")})
        if not plan:  # model returned nothing usable: fall back to one general task
            plan = [{"id": 1, "description": question, "suggested_tool": "web_search", "rationale": "fallback"}]
        self.plan = plan
        self._log("plan", f"Plan: {len(plan)} task(s) - {data.get('objective', '')}", output=plan)
        return plan

    # ------------------------------------------------------------------ execution
    async def _execute_task(self, task, n_tasks, question, findings, documents, anthropic_tools) -> str:
        tid = task["id"]
        self._log("task_start", f"Task {tid}/{n_tasks}: {task['description']}", task_id=tid,
                  tool=task["suggested_tool"])
        prior = "\n".join(f"- Task {t['id']} findings: {f[:1500]}" for t, f in findings) or "(none yet)"
        doc_line = f"Uploaded documents: {', '.join(documents)}\n" if documents else ""
        messages = [{"role": "user", "content":
                     f"Overall research question: {question}\n{doc_line}"
                     f"Your task ({tid}/{n_tasks}): {task['description']}\n"
                     f"Suggested tool: {task['suggested_tool']}\n\nFindings from earlier tasks:\n{prior}"}]
        final_text = ""
        rounds = self.settings.max_tool_rounds
        for round_no in range(1, rounds + 1):
            last = round_no == rounds
            resp = await self.llm.complete(
                system=EXECUTOR_SYSTEM, messages=messages, tools=anthropic_tools,
                tool_choice={"type": "none"} if last else None, max_tokens=2048)
            uses = tool_uses(resp)
            if not uses or last:
                final_text = text_of(resp)
                break
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for use in uses:
                output, is_error = await self._call_tool(use.name, dict(use.input or {}), tid)
                item = {"type": "tool_result", "tool_use_id": use.id, "content": output}
                if is_error:
                    item["is_error"] = True
                results.append(item)
            messages.append({"role": "user", "content": results})
        final_text = final_text or "(no findings produced for this task)"
        self._log("task_done", f"Task {tid} findings", task_id=tid, output=_preview(final_text, 1200))
        return final_text

    async def _call_tool(self, name: str, args: dict, task_id: int) -> tuple[str, bool]:
        shown = {k: (_preview(v, 200) if isinstance(v, str) else v) for k, v in args.items()}
        self._log("tool_call", f"Tool call: {name}", task_id=task_id, tool=name, input=shown)
        t0 = time.perf_counter()
        try:
            res = await self.tools.call(name, args)
            is_error, raw = res.is_error, res.text
        except Exception as exc:  # timeouts / transport errors become an observation the model can react to
            is_error, raw = True, f"{type(exc).__name__}: {exc}"
        observation = f"Tool error: {raw}" if is_error else self.sources.ingest_tool_output(name, raw)
        observation = observation[:OBSERVATION_LIMIT]
        if not is_error:
            self._evidence.append(observation)
        self._log("tool_result", f"{name} {'failed' if is_error else 'returned'}", task_id=task_id, tool=name,
                  output=_preview(observation), error=is_error, duration_ms=int((time.perf_counter() - t0) * 1000))
        return observation, is_error

    # ------------------------------------------------------------------ synthesis
    async def _synthesize(self, question, findings, style) -> dict:
        self._log("synthesis", "Writing the final report")
        findings_text = "\n\n".join(f"### Task {t['id']}: {t['description']}\n{f}" for t, f in findings)
        src_lines = []
        for s in self.sources.as_list():
            excerpt = " ".join(s["content"].split())[:500]
            src_lines.append(f"[S{s['id']}] {s['title']} ({s['locator']})\n    excerpt: {excerpt}")
        style_note = ("Write a CONCISE report: a 2-3 sentence executive summary and at most 3 key findings of "
                      "under 80 words each." if style == "concise" else
                      "Write a thorough, well-structured report with 3-6 key findings.")
        prompt = (f"Research question: {question}\n\n{style_note}\n\n## Findings from research tasks\n"
                  f"{findings_text}\n\n## Sources (cite by label)\n" + ("\n".join(src_lines) or "(no sources)"))
        report = await self.llm.structured(
            system=SYNTH_SYSTEM, prompt=prompt, tool_name="submit_report",
            description="Submit the final structured research report", schema=REPORT_SCHEMA,
            max_tokens=max(self.settings.max_tokens, 6000))
        report.setdefault("key_findings", [])
        report.setdefault("limitations", [])
        return report
