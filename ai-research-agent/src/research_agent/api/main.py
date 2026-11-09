"""FastAPI backend: document ingestion, research runs (background), reports, evaluations."""

import asyncio
import logging
from typing import Awaitable, Callable, Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from pydantic import BaseModel, Field

from research_agent import __version__
from research_agent.agent.models import AgentResult
from research_agent.config import Settings, get_settings
from research_agent.db import Database
from research_agent.documents.processor import ingest_document
from research_agent.evals.evaluator import Evaluator
from research_agent.reports.generator import report_body_text

log = logging.getLogger("research_agent.api")

Runner = Callable[[str, str, list, list, Callable], Awaitable[AgentResult]]


class ResearchRequest(BaseModel):
    question: str = Field(min_length=5, max_length=2000)
    style: Literal["detailed", "concise"] = "detailed"
    evaluate: bool = True


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=4, max_length=100)
    message: str = Field(min_length=2, max_length=2000)
    evaluate: bool = True


class AppContext:
    """Holds shared services. ``runner`` / ``evaluator_factory`` can be swapped (used by the tests)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.db = Database(settings.sqlite_path)
        self.db.mark_interrupted()
        self._store = None
        self.requires_key = True
        self.runner: Runner = self._default_runner
        self.evaluator_factory = lambda: Evaluator(self._llm())
        self.tasks: set[asyncio.Task] = set()

    @property
    def store(self):
        if self._store is None:
            from research_agent.documents.vectorstore import build_store
            self._store = build_store(self.settings)
        return self._store

    @store.setter
    def store(self, value):
        self._store = value

    def _llm(self):
        from research_agent.llm import LLMClient
        return LLMClient(self.settings)

    async def _default_runner(self, question, style, context, documents, on_step) -> AgentResult:
        from research_agent.agent.agent import ResearchAgent
        from research_agent.agent.mcp_client import MCPToolClient

        llm = self._llm()
        async with MCPToolClient(self.settings) as tools:
            agent = ResearchAgent(llm, tools, self.settings, on_step=on_step)
            return await agent.run(question, style=style, context=context, available_documents=documents)


async def _execute_run(ctx: AppContext, run_id: int, question: str, style: str, evaluate: bool,
                       session_id: str | None, context: list, documents: list) -> None:
    db = ctx.db
    try:
        result = await ctx.runner(question, style, context, documents,
                                  lambda steps, plan: db.update_run_progress(run_id, steps=steps, plan=plan))
        db.complete_run(run_id, plan=result.plan, steps=result.steps, report=result.report,
                        markdown=result.markdown, sources=result.sources, evidence=result.evidence,
                        duration=result.duration_s)
        if session_id:
            db.add_message(session_id, "assistant", result.markdown, run_id)
        if evaluate:
            try:
                evaluation = await ctx.evaluator_factory().evaluate(
                    question=question, answer=report_body_text(result.report), sources=result.sources,
                    evidence=result.evidence, report=result.report, style=style)
                db.save_evaluation(run_id, evaluation.to_dict())
            except Exception:
                log.exception("Evaluation failed for run %s", run_id)
                db.set_eval_status(run_id, "failed")
    except Exception as exc:
        log.exception("Run %s failed", run_id)
        db.fail_run(run_id, f"{type(exc).__name__}: {exc}")
        if session_id:
            db.add_message(session_id, "assistant", f"Sorry - the research run failed: {exc}", run_id)


def _launch(ctx: AppContext, question: str, style: str, evaluate: bool, session_id: str | None = None,
            context: list | None = None) -> int:
    if ctx.requires_key and not ctx.settings.has_api_key:
        raise HTTPException(503, "ANTHROPIC_API_KEY is not configured. Copy .env.example to .env and add your key.")
    run_id = ctx.db.create_run(question, style, evaluate, session_id)
    docs = [d["filename"] for d in ctx.db.list_documents()]
    task = asyncio.create_task(_execute_run(ctx, run_id, question, style, evaluate, session_id, context or [], docs))
    ctx.tasks.add(task)
    task.add_done_callback(ctx.tasks.discard)
    return run_id


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    ctx = AppContext(settings)
    app = FastAPI(title="AI Research Agent API", version=__version__,
                  description="LLM agent + MCP tools + RAG + evaluation")
    app.state.ctx = ctx

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__, "model": settings.claude_model,
                "api_key_configured": settings.has_api_key}

    @app.get("/stats")
    def stats():
        return ctx.db.stats()

    @app.get("/tools")
    async def tools():
        from research_agent.agent.mcp_client import MCPToolClient
        try:
            async with MCPToolClient(settings) as client:
                specs = await client.list_tools()
        except Exception as exc:
            raise HTTPException(502, f"Could not reach the MCP server: {exc}")
        return [{"name": s.name, "description": s.description, "input_schema": s.input_schema} for s in specs]

    # ------------------------------------------------------------ documents
    @app.get("/documents")
    def list_documents():
        return ctx.db.list_documents()

    @app.post("/documents", status_code=201)
    async def upload_document(file: UploadFile = File(...)):
        data = await file.read()
        try:
            return await run_in_threadpool(
                lambda: ingest_document(file.filename or "upload", data, db=ctx.db, store=ctx.store, settings=settings))
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.delete("/documents/{doc_id}")
    def delete_document(doc_id: int):
        if not ctx.db.get_document(doc_id):
            raise HTTPException(404, "Document not found")
        ctx.store.delete_document(doc_id)
        ctx.db.delete_document(doc_id)
        return {"deleted": doc_id}

    # ------------------------------------------------------------ research
    @app.post("/research", status_code=202)
    async def start_research(req: ResearchRequest):
        return {"run_id": _launch(ctx, req.question.strip(), req.style, req.evaluate), "status": "running"}

    @app.get("/runs")
    def list_runs(limit: int = Query(100, ge=1, le=500)):
        return ctx.db.list_runs(limit)

    @app.get("/runs/{run_id}")
    def get_run(run_id: int):
        run = ctx.db.get_run(run_id)
        if not run:
            raise HTTPException(404, "Run not found")
        run.pop("evidence", None)
        run["evaluation"] = ctx.db.get_evaluation(run_id)
        return run

    @app.post("/runs/{run_id}/evaluate")
    async def evaluate_run(run_id: int):
        run = ctx.db.get_run(run_id)
        if not run or run["status"] != "completed":
            raise HTTPException(404, "Completed run not found")
        if ctx.requires_key and not settings.has_api_key:
            raise HTTPException(503, "ANTHROPIC_API_KEY is not configured.")
        ctx.db.set_eval_status(run_id, "pending")
        try:
            ev = await ctx.evaluator_factory().evaluate(
                question=run["question"], answer=report_body_text(run["report"]), sources=run["sources"],
                evidence=run["evidence"] or "", report=run["report"], style=run["style"])
        except Exception as exc:
            ctx.db.set_eval_status(run_id, "failed")
            raise HTTPException(500, f"Evaluation failed: {exc}")
        ctx.db.save_evaluation(run_id, ev.to_dict())
        return ev.to_dict()

    @app.get("/runs/{run_id}/report")
    async def download_report(run_id: int, format: Literal["md", "pdf"] = "md"):
        run = ctx.db.get_run(run_id)
        if not run or run["status"] != "completed":
            raise HTTPException(404, "Completed run not found")
        name = f"research-report-{run_id}"
        if format == "md":
            return Response(run["markdown"], media_type="text/markdown; charset=utf-8",
                            headers={"Content-Disposition": f'attachment; filename="{name}.md"'})
        from research_agent.reports.pdf import render_pdf
        pdf = await run_in_threadpool(lambda: render_pdf(
            question=run["question"], report=run["report"], sources=run["sources"],
            evaluation=ctx.db.get_evaluation(run_id), created_at=run["completed_at"]))
        return Response(pdf, media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="{name}.pdf"'})

    @app.get("/evaluations")
    def list_evaluations():
        return ctx.db.list_evaluations()

    # ------------------------------------------------------------ chat
    @app.post("/chat", status_code=202)
    async def chat(req: ChatRequest):
        history = [{"role": m["role"], "content": m["content"]} for m in ctx.db.get_messages(req.session_id, 8)]
        message = req.message.strip()
        run_id = _launch(ctx, message, "concise", req.evaluate, session_id=req.session_id, context=history)
        ctx.db.add_message(req.session_id, "user", message, run_id)
        return {"run_id": run_id, "status": "running"}

    @app.get("/chat/{session_id}")
    def chat_history(session_id: str):
        return ctx.db.get_messages(session_id, 100)

    return app
