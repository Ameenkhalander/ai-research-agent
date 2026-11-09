"""PDF export of a research report (ReportLab)."""
from __future__ import annotations

import io
import re
from xml.sax.saxutils import escape

from research_agent.citations import extract_citation_ids
from research_agent.reports.generator import report_body_text

_UNICODE_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-", "\u2026": "...",
    "\u2192": "->", "\u2265": ">=", "\u2264": "<=", "\u00a0": " ", "\u2022": "-", "\u2212": "-",
})


def _clean(text: str) -> str:
    """Built-in PDF fonts only cover Latin-1/cp1252, so map or replace anything else."""
    text = (text or "").translate(_UNICODE_MAP)
    return text.encode("cp1252", "replace").decode("cp1252")


def _inline(text: str) -> str:
    out = escape(_clean(text))
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<i>\1</i>", out)
    out = re.sub(r"`([^`]+)`", r"<font face='Courier'>\1</font>", out)
    return re.sub(r"(\[S\d+\])", r"<font color='#1d4ed8'>\1</font>", out)


def render_pdf(*, question: str, report: dict, sources: list[dict], evaluation: dict | None = None,
               created_at: str | None = None) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    base = getSampleStyleSheet()
    title = ParagraphStyle("T", parent=base["Title"], fontSize=20, leading=24, spaceAfter=6, alignment=0)
    h2 = ParagraphStyle("H2", parent=base["Heading2"], fontSize=14, spaceBefore=14, spaceAfter=6,
                        textColor=colors.HexColor("#0f172a"))
    h3 = ParagraphStyle("H3", parent=base["Heading3"], fontSize=11.5, spaceBefore=8, spaceAfter=3)
    body = ParagraphStyle("B", parent=base["BodyText"], fontSize=10, leading=14.5, spaceAfter=5)
    small = ParagraphStyle("S", parent=body, fontSize=8.5, leading=11, textColor=colors.HexColor("#475569"))
    bullet = ParagraphStyle("Bu", parent=body, leftIndent=14, bulletIndent=4)

    def blocks(text: str) -> list:
        out = []
        for raw in (text or "").split("\n"):
            line = raw.strip()
            if not line:
                continue
            if line[:2] in ("- ", "* ") or re.match(r"^\d+[.)] ", line):
                out.append(Paragraph(_inline(re.sub(r"^(?:[-*]|\d+[.)]) ", "", line)), bullet, bulletText="\u2022"))
            else:
                out.append(Paragraph(_inline(line), body))
        return out

    story: list = [Paragraph(_inline(report.get("title") or "Research Report"), title),
                   Paragraph(_inline("Research question: " + question), small)]
    if created_at:
        story.append(Paragraph(escape(_clean(f"Generated: {created_at}")), small))
    story.append(Spacer(1, 8))

    story += [Paragraph("Executive summary", h2)] + blocks(report.get("executive_summary", ""))
    if report.get("key_findings"):
        story.append(Paragraph("Key findings", h2))
        for f in report["key_findings"]:
            story.append(Paragraph(_inline(f.get("heading", "")), h3))
            story += blocks(f.get("content", ""))
    if report.get("conclusion"):
        story += [Paragraph("Conclusion", h2)] + blocks(report["conclusion"])
    if report.get("limitations"):
        story += [Paragraph("Limitations and caveats", h2)] + blocks("\n".join("- " + x for x in report["limitations"]))

    cited = sorted(set(extract_citation_ids(report_body_text(report))))
    by_id = {s["id"]: s for s in sources}
    story.append(Paragraph("References", h2))
    refs = [by_id[i] for i in cited if i in by_id]
    if not refs:
        story.append(Paragraph("No sources were cited.", body))
    for s in refs:
        if s.get("kind") == "web" and s.get("url"):
            url = escape(s["url"], {'"': "&quot;"})
            label = f"[S{s['id']}] {escape(_clean(s.get('title') or ''))} - <link href=\"{url}\" color=\"blue\">{escape(_clean(s['url']))}</link>"
        else:
            label = f"[S{s['id']}] Uploaded document: {escape(_clean(s.get('title') or ''))}"
        story.append(Paragraph(label, small))

    if evaluation:
        story.append(Paragraph("Automated evaluation", h2))
        rows = [["Metric", "Score", "Method"]]
        for name, m in evaluation.get("metrics", {}).items():
            score = "n/a" if not m.get("applicable", True) else f"{m['score']:.2f}"
            rows.append([name.replace("_", " ").title(), score, m.get("method", "")])
        rows.append(["Overall", f"{evaluation.get('overall', 0):.2f}", ""])
        table = Table(rows, colWidths=[6.5 * cm, 2.5 * cm, 7 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(table)
        story.append(Paragraph("Scores are 0-1 (higher is better). Automated checks are indicators, "
                               "not proof of correctness.", small))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm,
                            bottomMargin=2 * cm, title=_clean(report.get("title") or "Research Report"),
                            author="AI Research Agent")

    def footer(canvas, d):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(2 * cm, 1.2 * cm, "AI Research Agent")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {d.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
