"""Turn the structured report (a dict) into Markdown and plain-text bodies."""
from __future__ import annotations

from research_agent.citations import extract_citation_ids


def report_body_text(report: dict) -> str:
    """The report content without the references section (this is what gets evaluated)."""
    parts = [report.get("executive_summary", "")]
    for finding in report.get("key_findings", []):
        parts.append(f"{finding.get('heading', '')}\n{finding.get('content', '')}")
    parts.append(report.get("conclusion", ""))
    return "\n\n".join(p for p in parts if p).strip()


def format_reference(src: dict) -> str:
    if src.get("kind") == "web":
        return f"[S{src['id']}] {src.get('title') or src.get('url')} - {src.get('url')}"
    return f"[S{src['id']}] Uploaded document: {src.get('title')}"


def report_to_markdown(report: dict, sources: list[dict], question: str) -> str:
    lines = [f"# {report.get('title') or 'Research Report'}", "", f"> **Research question:** {question}", ""]
    lines += ["## Executive summary", "", report.get("executive_summary", ""), ""]
    findings = report.get("key_findings", [])
    if findings:
        lines += ["## Key findings", ""]
        for f in findings:
            lines += [f"### {f.get('heading', '')}", "", f.get("content", ""), ""]
    if report.get("conclusion"):
        lines += ["## Conclusion", "", report["conclusion"], ""]
    if report.get("limitations"):
        lines += ["## Limitations and caveats", ""] + [f"- {x}" for x in report["limitations"]] + [""]
    if report.get("follow_up_questions"):
        lines += ["## Suggested follow-up questions", ""] + [f"- {x}" for x in report["follow_up_questions"]] + [""]

    cited = sorted(set(extract_citation_ids(report_body_text(report))))
    by_id = {s["id"]: s for s in sources}
    refs = [format_reference(by_id[i]) for i in cited if i in by_id]
    lines += ["## References", ""]
    lines += [f"{r}  " for r in refs] if refs else ["_No sources were cited._"]
    return "\n".join(lines).strip() + "\n"
