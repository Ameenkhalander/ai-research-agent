import pytest

from research_agent.reports.generator import report_body_text, report_to_markdown

REPORT = {"title": "MCP Overview", "executive_summary": "MCP standardises tool access [S1].",
          "key_findings": [{"heading": "Transport", "content": "- stdio [S1]\n- **HTTP** [S2] \u2014 remote"}],
          "conclusion": "Use MCP.", "limitations": ["Small sample"], "follow_up_questions": ["What about auth?"]}
SOURCES = [{"id": 1, "kind": "web", "title": "Spec", "locator": "https://spec.example", "url": "https://spec.example"},
           {"id": 2, "kind": "document", "title": "notes.txt, p.2", "locator": "notes.txt#chunk1", "url": None},
           {"id": 3, "kind": "web", "title": "Unused", "locator": "https://unused.example", "url": "https://unused.example"}]


def test_markdown_lists_only_cited_references():
    md = report_to_markdown(REPORT, SOURCES, "What is MCP?")
    assert md.startswith("# MCP Overview") and "## Key findings" in md and "### Transport" in md
    assert "[S1] Spec - https://spec.example" in md and "Uploaded document: notes.txt, p.2" in md
    assert "unused.example" not in md
    assert "References" not in report_body_text(REPORT)


def test_pdf_export_contains_content(tmp_path):
    pytest.importorskip("reportlab")
    pypdf = pytest.importorskip("pypdf")
    from research_agent.reports.pdf import render_pdf

    evaluation = {"overall": 0.82, "metrics": {"answer_relevance": {"score": 0.9, "method": "llm_judge", "applicable": True}}}
    pdf = render_pdf(question="What is MCP?", report=REPORT, sources=SOURCES, evaluation=evaluation,
                     created_at="2026-01-01T00:00:00+00:00")
    assert pdf.startswith(b"%PDF")
    path = tmp_path / "r.pdf"
    path.write_bytes(pdf)
    text = " ".join(p.extract_text() for p in pypdf.PdfReader(str(path)).pages)
    assert "MCP Overview" in text and "Executive summary" in text and "https://spec.example" in text
    assert "Automated evaluation" in text and "0.82" in text
