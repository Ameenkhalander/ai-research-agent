"""Regression benchmark: run the agent over evals/dataset.jsonl and score every answer.

    python -m research_agent.evals.benchmark --ingest-samples --limit 3

Requires ANTHROPIC_API_KEY and network access. Results are written to evals/results/<timestamp>.csv.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from research_agent.config import PROJECT_ROOT, Settings, get_settings
from research_agent.reports.generator import report_body_text

DATASET = PROJECT_ROOT / "evals" / "dataset.jsonl"
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"


def load_dataset(path: Path = DATASET) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def keyword_hit_rate(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    lowered = text.lower()
    return sum(1 for k in keywords if k.lower() in lowered) / len(keywords)


def tool_hit_rate(steps: list[dict], expected: list[str]) -> float:
    if not expected:
        return 1.0
    used = {s.get("tool") for s in steps if s.get("kind") == "tool_call"}
    return sum(1 for t in expected if t in used) / len(expected)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in ("id", "question", "error", "duration_s", "n_sources", "flags")]
    return df[cols].apply(pd.to_numeric, errors="coerce").mean().round(3).to_frame("mean")


async def run_benchmark(settings: Settings, items: list[dict], ingest_samples: bool = False) -> pd.DataFrame:
    from research_agent.agent.agent import ResearchAgent
    from research_agent.agent.mcp_client import MCPToolClient
    from research_agent.db import Database
    from research_agent.documents.processor import ingest_document
    from research_agent.documents.vectorstore import build_store
    from research_agent.evals.evaluator import Evaluator
    from research_agent.llm import LLMClient

    llm = LLMClient(settings)
    docs: list[str] = []
    if ingest_samples:
        db, store = Database(settings.sqlite_path), build_store(settings)
        for path in sorted(SAMPLE_DIR.glob("*")):
            if path.suffix.lower() in (".txt", ".pdf"):
                ingest_document(path.name, path.read_bytes(), db=db, store=store, settings=settings)
                docs.append(path.name)
    else:
        docs = [d["filename"] for d in Database(settings.sqlite_path).list_documents()]

    rows = []
    for item in items:
        print(f"[{item['id']}] {item['question'][:80]}", file=sys.stderr)
        row = {"id": item["id"], "question": item["question"]}
        try:
            async with MCPToolClient(settings) as tools:
                result = await ResearchAgent(llm, tools, settings).run(
                    item["question"], style="concise", available_documents=docs)
            body = report_body_text(result.report)
            ev = await Evaluator(llm).evaluate(question=item["question"], answer=body, sources=result.sources,
                                               evidence=result.evidence, report=result.report, style="concise")
            row.update({"overall": ev.overall, **{k: v.score for k, v in ev.metrics.items()},
                        "keyword_hit_rate": keyword_hit_rate(body, item.get("expected_keywords", [])),
                        "tool_hit_rate": tool_hit_rate(result.steps, item.get("expect_tools", [])),
                        "duration_s": result.duration_s, "n_sources": len(result.sources),
                        "flags": "; ".join(ev.flags)[:300]})
        except Exception as exc:  # keep going so one failure does not lose the whole benchmark
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    parser.add_argument("--ingest-samples", action="store_true", help="index data/sample/* before running")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.has_api_key:
        sys.exit("ANTHROPIC_API_KEY is not set (see .env.example)")
    items = load_dataset(args.dataset)[: args.limit]
    df = asyncio.run(run_benchmark(settings, items, args.ingest_samples))
    out_dir = PROJECT_ROOT / "evals" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"benchmark_{datetime.now():%Y%m%d_%H%M%S}.csv"
    df.to_csv(out, index=False)
    print(df.drop(columns=["flags"], errors="ignore").to_string(index=False))
    if "overall" in df:
        print("\nMeans:\n" + summarize_results(df).to_string())
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
