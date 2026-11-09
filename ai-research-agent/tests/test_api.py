import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from helpers import FakeStore
from research_agent.agent.models import AgentResult
from research_agent.api.main import create_app
from research_agent.config import Settings
from research_agent.evals.evaluator import Evaluator

REPORT = {"title": "T", "executive_summary": "MCP uses JSON-RPC [S1].", "key_findings": [
    {"heading": "H", "content": "Servers expose tools [S1]."}], "conclusion": "Done.", "limitations": []}
SOURCES = [{"id": 1, "kind": "web", "title": "Spec", "locator": "https://s.example", "url": "https://s.example",
            "content": "MCP uses JSON-RPC. Servers expose tools."}]


async def fake_runner(question, style, context, documents, on_step):
    steps = [{"n": 1, "kind": "plan", "title": "Plan", "ts": "now"}]
    on_step(steps, [{"id": 1}])
    return AgentResult(question=question, style=style, plan=[{"id": 1, "description": "d", "suggested_tool": "none"}],
                       steps=steps, report=REPORT, markdown="# T\n\nMCP uses JSON-RPC [S1].", sources=SOURCES,
                       evidence="MCP uses JSON-RPC. Servers expose tools.", duration_s=0.5)


def make_client(tmp_path, **kw):
    app = create_app(Settings(data_dir=tmp_path, chunk_size=200, chunk_overlap=20, **kw))
    ctx = app.state.ctx
    ctx.store, ctx.runner, ctx.requires_key = FakeStore(), fake_runner, False
    ctx.evaluator_factory = lambda: Evaluator(llm=None)
    return app, ctx


def wait(client, run_id, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        run = client.get(f"/runs/{run_id}").json()
        if run["status"] != "running" and run["eval_status"] != "pending":
            return run
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_health_and_document_lifecycle(tmp_path):
    app, _ = make_client(tmp_path)
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        text = ("The Model Context Protocol lets clients call tools on servers. " * 10).encode()
        r = client.post("/documents", files={"file": ("mcp.txt", text, "text/plain")})
        assert r.status_code == 201 and r.json()["n_chunks"] > 1
        assert client.post("/documents", files={"file": ("bad.docx", b"x", "application/octet-stream")}).status_code == 400
        docs = client.get("/documents").json()
        assert [d["filename"] for d in docs] == ["mcp.txt"]
        assert client.delete(f"/documents/{docs[0]['id']}").status_code == 200
        assert client.get("/documents").json() == []
        assert client.delete("/documents/999").status_code == 404


def test_research_run_report_and_evaluation(tmp_path):
    app, _ = make_client(tmp_path)
    with TestClient(app) as client:
        r = client.post("/research", json={"question": "What is MCP?", "style": "detailed", "evaluate": True})
        assert r.status_code == 202
        run = wait(client, r.json()["run_id"])
        assert run["status"] == "completed" and run["eval_status"] == "done"
        assert run["evaluation"]["overall"] > 0 and "evidence" not in run
        md = client.get(f"/runs/{run['id']}/report", params={"format": "md"})
        assert md.status_code == 200 and "JSON-RPC" in md.text
        pdf = client.get(f"/runs/{run['id']}/report", params={"format": "pdf"})
        assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
        assert client.get("/evaluations").json()[0]["run_id"] == run["id"]
        assert client.get("/stats").json()["completed"] == 1
        assert client.post(f"/runs/{run['id']}/evaluate").status_code == 200
        assert client.get("/runs/999").status_code == 404


def test_failed_run_is_recorded(tmp_path):
    app, ctx = make_client(tmp_path)

    async def boom(*a, **k):
        raise RuntimeError("kaboom")

    ctx.runner = boom
    with TestClient(app) as client:
        run = wait(client, client.post("/research", json={"question": "Will this fail?"}).json()["run_id"])
        assert run["status"] == "failed" and "kaboom" in run["error"]


def test_chat_keeps_history_and_passes_context(tmp_path):
    app, ctx = make_client(tmp_path)
    contexts = []

    async def runner(question, style, context, documents, on_step):
        contexts.append(context)
        return await fake_runner(question, style, context, documents, on_step)

    ctx.runner = runner
    with TestClient(app) as client:
        for msg in ("What is MCP?", "And how does it transport messages?"):
            wait(client, client.post("/chat", json={"session_id": "sess-1", "message": msg}).json()["run_id"])
        history = client.get("/chat/sess-1").json()
        assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
        assert contexts[0] == [] and len(contexts[1]) == 2


def test_validation_and_missing_api_key(tmp_path):
    app, ctx = make_client(tmp_path)
    with TestClient(app) as client:
        assert client.post("/research", json={"question": "hi"}).status_code == 422
        ctx.requires_key = True
        r = client.post("/research", json={"question": "A valid research question"})
        assert r.status_code == 503 and "ANTHROPIC_API_KEY" in r.json()["detail"]
