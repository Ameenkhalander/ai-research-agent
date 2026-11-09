"""Regenerate data/sample/ai_agents_overview.pdf (requires reportlab)."""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parents[1] / "data" / "sample" / "ai_agents_overview.pdf"

SECTIONS = [
    ("What is an AI agent?",
     "An AI agent is a system in which a language model decides which actions to take to reach a goal. Unlike a single "
     "prompt-and-response call, an agent runs in a loop: it observes its context, chooses an action such as calling a "
     "tool, receives the result, and repeats until it can produce a final answer. The model supplies the reasoning; "
     "the surrounding program supplies tools, memory and guard rails."),
    ("Planning and task decomposition",
     "Complex questions are usually broken into smaller sub-tasks before any tool is used. A planning step asks the model "
     "to list the sub-tasks, the order in which to tackle them and the tool best suited to each. Planning reduces wasted "
     "tool calls and makes the agent's behaviour inspectable, because the plan can be logged and reviewed."),
    ("Tool use",
     "Tools are described to the model by a name, a description and an input schema. When the model decides a tool is "
     "needed it emits a structured tool call; the application executes it and returns the result as an observation. Good "
     "tool descriptions matter as much as good code, because the description is how the model learns when to use a tool."),
    ("Failure modes",
     "Typical failures include choosing the wrong tool, looping without progress, trusting low-quality search results, "
     "and inventing facts when evidence is missing. Mitigations are step limits, forcing a final answer after a maximum "
     "number of rounds, requiring citations for every claim, and evaluating outputs automatically."),
    ("Memory and retrieval",
     "Agents combine short-term memory, the conversation history, with long-term memory in a vector database. Documents are "
     "split into overlapping chunks, embedded, and retrieved by semantic similarity. Retrieval-augmented generation grounds "
     "answers in the retrieved passages, which reduces hallucination when combined with citation requirements."),
]


def main() -> None:
    styles = getSampleStyleSheet()
    story = [Paragraph("An Introduction to AI Agents", styles["Title"]), Spacer(1, 12)]
    for i, (heading, text) in enumerate(SECTIONS):
        story += [Paragraph(heading, styles["Heading2"]), Paragraph(text, styles["BodyText"]), Spacer(1, 10)]
        if i == 2:
            story.append(PageBreak())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(OUT), pagesize=A4, title="An Introduction to AI Agents").build(story)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
