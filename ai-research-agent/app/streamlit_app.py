"""Streamlit front-end for the AI Research Agent."""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from api_client import Api, ApiError

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"
METRIC_LABELS = {
    "answer_relevance": "Answer relevance", "factual_consistency": "Factual consistency",
    "citation_validity": "Citation validity", "response_completeness": "Response completeness",
    "hallucination_indicators": "Hallucination indicators (1.0 = none found)",
}
STEP_ICONS = {"mcp": "🔌", "plan": "🗺️", "task_start": "▶️", "tool_call": "🛠️", "tool_result": "📥",
              "task_done": "✅", "synthesis": "📝", "done": "🏁"}

st.set_page_config(page_title="AI Research Agent", page_icon="🔎", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
[data-testid="stMetricValue"] {font-size: 1.8rem;}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def get_api() -> Api:
    return Api(BACKEND_URL)


api = get_api()


# ---------------------------------------------------------------------------- shared renderers
def score_color(v: float) -> str:
    return "🟢" if v >= 0.75 else "🟡" if v >= 0.5 else "🔴"


def render_steps_live(steps: list[dict]) -> str:
    lines = []
    for s in steps:
        extra = f" — `{s['tool']}`" if s.get("tool") and s["kind"] in ("tool_call", "tool_result") else ""
        ms = f" ({s['duration_ms']} ms)" if s.get("duration_ms") is not None else ""
        lines.append(f"{STEP_ICONS.get(s['kind'], '•')} **{s['n']}.** {s['title']}{extra}{ms}")
    return "\n\n".join(lines) or "Starting…"


def render_trace(run: dict) -> None:
    plan = run.get("plan") or []
    if plan:
        st.subheader("Plan")
        st.dataframe(pd.DataFrame(plan)[["id", "description", "suggested_tool"]], hide_index=True)
    st.subheader("Execution trace")
    for s in run.get("steps") or []:
        err = " ⚠️" if s.get("error") else ""
        with st.expander(f"{STEP_ICONS.get(s['kind'], '•')} {s['n']}. {s['title']}{err}"):
            st.caption(f"{s['ts']}" + (f" · {s['duration_ms']} ms" if s.get("duration_ms") is not None else ""))
            if s.get("input"):
                st.markdown("**Input**")
                st.json(s["input"], expanded=False)
            if s.get("output") is not None:
                st.markdown("**Output**")
                out = s["output"]
                st.json(out, expanded=False) if isinstance(out, (list, dict)) else st.code(out, language="markdown")


def render_sources(run: dict) -> None:
    sources = run.get("sources") or []
    if not sources:
        st.info("No sources were collected for this run.")
        return
    for s in sources:
        label = f"[S{s['id']}] {s['title']}"
        st.markdown(f"**{label}** — " + (f"[{s['url']}]({s['url']})" if s.get("url") else f"_{s['locator']}_"))
        if s.get("content"):
            with st.expander("Excerpt"):
                st.write(" ".join(s["content"].split())[:1200])


def render_evaluation(ev: dict | None, run_id: int | None = None) -> None:
    if not ev:
        st.info("This run has not been evaluated yet.")
        if run_id and st.button("Evaluate now", key=f"eval-{run_id}"):
            with st.spinner("Evaluating…"):
                try:
                    api.evaluate_run(run_id)
                    st.rerun()
                except ApiError as exc:
                    st.error(str(exc))
        return
    st.metric("Overall score", f"{ev['overall']:.2f}", help="Weighted mean of the applicable metrics (0-1).")
    for key, m in ev["metrics"].items():
        label = METRIC_LABELS.get(key, key)
        if not m.get("applicable", True):
            st.caption(f"{label}: not applicable — {m['explanation']}")
            continue
        st.markdown(f"{score_color(m['score'])} **{label}** — {m['score']:.2f} · _{m['method']}_")
        st.progress(min(1.0, max(0.0, m["score"])))
        if m.get("explanation"):
            st.caption(m["explanation"])
    if ev.get("flags"):
        st.markdown("**Flags**")
        for f in ev["flags"]:
            st.warning(f)
    with st.expander("Raw evaluation JSON"):
        st.json(ev, expanded=False)


def wait_for_run(run_id: int, label: str = "Agent is working…") -> dict:
    """Poll the backend until the run (and its evaluation) finish, showing live progress."""
    run = api.run(run_id)
    deadline = time.time() + 15 * 60
    if run["status"] == "running":
        with st.status(label, expanded=True) as status:
            box = st.empty()
            while run["status"] == "running" and time.time() < deadline:
                box.markdown(render_steps_live(run.get("steps") or []))
                time.sleep(1.5)
                run = api.run(run_id)
            status.update(label="Research complete" if run["status"] == "completed" else "Run failed",
                          state="complete" if run["status"] == "completed" else "error", expanded=False)
    if run["status"] == "completed" and run["eval_status"] == "pending":
        with st.spinner("Evaluating the report…"):
            while run["eval_status"] == "pending" and time.time() < deadline:
                time.sleep(1.5)
                run = api.run(run_id)
    return run


def show_run(run: dict) -> None:
    if run["status"] == "failed":
        st.error(f"Run failed: {run.get('error')}")
        return
    if run["status"] != "completed":
        st.info("Run still in progress.")
        return
    cols = st.columns(4)
    cols[0].metric("Duration", f"{run['duration']:.1f}s")
    cols[1].metric("Sources", len(run.get("sources") or []))
    cols[2].metric("Tool calls", sum(1 for s in run["steps"] if s["kind"] == "tool_call"))
    cols[3].metric("Eval score", f"{run['evaluation']['overall']:.2f}" if run.get("evaluation") else "—")
    tabs = st.tabs(["📄 Report", "🧭 Agent trace", "🔗 Sources", "🧪 Evaluation"])
    with tabs[0]:
        st.markdown(run["markdown"])
        c1, c2, _ = st.columns([1, 1, 4])
        c1.download_button("⬇️ Markdown", run["markdown"], f"research-report-{run['id']}.md", "text/markdown",
                           key=f"md-{run['id']}")
        try:
            c2.download_button("⬇️ PDF", api.report(run["id"], "pdf"), f"research-report-{run['id']}.pdf",
                               "application/pdf", key=f"pdf-{run['id']}")
        except ApiError as exc:
            c2.caption(f"PDF unavailable: {exc}")
    with tabs[1]:
        render_trace(run)
    with tabs[2]:
        render_sources(run)
    with tabs[3]:
        render_evaluation(run.get("evaluation"), run["id"])


# ---------------------------------------------------------------------------- pages
def page_dashboard() -> None:
    st.title("🔎 Research Dashboard")
    stats = api.stats()
    c = st.columns(5)
    c[0].metric("Documents", stats["documents"])
    c[1].metric("Indexed chunks", stats["chunks"])
    c[2].metric("Research runs", stats["runs"], f"{stats['failed']} failed" if stats["failed"] else None,
                delta_color="inverse")
    c[3].metric("Avg. duration", f"{stats['avg_duration_s']}s" if stats["avg_duration_s"] else "—")
    c[4].metric("Avg. eval score", f"{stats['avg_overall_score']:.2f}" if stats["avg_overall_score"] else "—")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Recent runs")
        runs = api.runs()[:10]
        if runs:
            df = pd.DataFrame(runs)[["id", "question", "status", "duration", "overall_score", "created_at"]]
            st.dataframe(df, hide_index=True)
        else:
            st.info("No research runs yet — head to **Research** or **Chat** to start.")
        evals = api.evaluations()
        if evals:
            st.subheader("Evaluation trend")
            trend = pd.DataFrame(evals).sort_values("run_id").set_index("run_id")[["overall"]]
            st.line_chart(trend)
    with right:
        st.subheader("MCP tools")
        try:
            for t in load_tools():
                st.markdown(f"**`{t['name']}`**  \n{t['description'].splitlines()[0] if t['description'] else ''}")
        except ApiError as exc:
            st.warning(f"Could not list MCP tools: {exc}")


@st.cache_data(ttl=300, show_spinner="Connecting to MCP server…")
def load_tools() -> list:
    return api.tools()


def page_research() -> None:
    st.title("🧪 Research")
    question = st.text_area("Research question", height=110, placeholder=(
        "e.g. Compare the stdio and HTTP transports of the Model Context Protocol and summarise my uploaded notes."))
    c1, c2, _ = st.columns([1, 1, 3])
    style = c1.selectbox("Report style", ["detailed", "concise"])
    evaluate = c2.toggle("Evaluate report", value=True)
    if st.button("Run research", type="primary", disabled=len(question.strip()) < 5):
        try:
            st.session_state["active_run"] = api.start_research(question.strip(), style, evaluate)
        except ApiError as exc:
            st.error(str(exc))
            return
    if run_id := st.session_state.get("active_run"):
        show_run(wait_for_run(run_id))


def page_chat() -> None:
    st.title("💬 Chat")
    st.caption("Follow-up questions use the conversation context. Each answer is a full agent run.")
    sid = st.session_state.setdefault("session_id", uuid.uuid4().hex)
    if "chat" not in st.session_state:
        st.session_state.chat = [{"role": m["role"], "content": m["content"], "run_id": m["run_id"]}
                                 for m in api.chat_history(sid)]
    if st.sidebar.button("🗑️ New conversation"):
        st.session_state.pop("chat", None)
        st.session_state["session_id"] = uuid.uuid4().hex
        st.rerun()
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    if prompt := st.chat_input("Ask a research question…"):
        st.session_state.chat.append({"role": "user", "content": prompt, "run_id": None})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                run = wait_for_run(api.chat(sid, prompt))
            except ApiError as exc:
                st.error(str(exc))
                return
            if run["status"] == "completed":
                st.markdown(run["markdown"])
                if run.get("evaluation"):
                    ev = run["evaluation"]
                    st.caption(f"Evaluation: overall {ev['overall']:.2f} · run #{run['id']} (see History for the trace)")
                st.session_state.chat.append({"role": "assistant", "content": run["markdown"], "run_id": run["id"]})
            else:
                st.error(f"The run failed: {run.get('error')}")


def page_documents() -> None:
    st.title("📚 Documents")
    st.caption("Upload PDF or TXT files. They are chunked, embedded and indexed in ChromaDB; "
               "the agent searches them through the `search_documents` MCP tool.")
    files = st.file_uploader("Upload documents", type=["pdf", "txt"], accept_multiple_files=True)
    c1, c2, _ = st.columns([1, 1.4, 4])
    if c1.button("Index files", type="primary", disabled=not files):
        for f in files:
            with st.spinner(f"Indexing {f.name}…"):
                try:
                    doc = api.upload(f.name, f.getvalue(), f.type or "application/octet-stream")
                    st.success(f"{f.name}: {doc['n_chunks']} chunks" + (" (already indexed)" if doc.get("duplicate") else ""))
                except ApiError as exc:
                    st.error(f"{f.name}: {exc}")
    if c2.button("Load sample documents", disabled=not SAMPLE_DIR.exists()):
        for p in sorted(SAMPLE_DIR.glob("*")):
            if p.suffix.lower() in (".pdf", ".txt"):
                with st.spinner(f"Indexing {p.name}…"):
                    try:
                        api.upload(p.name, p.read_bytes())
                    except ApiError as exc:
                        st.error(f"{p.name}: {exc}")
        st.rerun()
    docs = api.documents()
    st.subheader(f"Indexed documents ({len(docs)})")
    if not docs:
        st.info("Nothing indexed yet.")
    for d in docs:
        cols = st.columns([4, 1, 1, 1, 1])
        cols[0].markdown(f"**{d['filename']}**  \n<small>{d['uploaded_at']}</small>", unsafe_allow_html=True)
        cols[1].caption(f"{d['file_type'].upper()}")
        cols[2].caption(f"{d['n_chunks']} chunks")
        cols[3].caption(f"{d['size_bytes'] / 1024:.1f} KB")
        if cols[4].button("Delete", key=f"del-{d['id']}"):
            api.delete_document(d["id"])
            st.rerun()


def page_history() -> None:
    st.title("🕘 Agent execution history")
    runs = api.runs()
    if not runs:
        st.info("No runs yet.")
        return
    df = pd.DataFrame(runs)[["id", "question", "style", "status", "duration", "overall_score", "created_at"]]
    st.dataframe(df, hide_index=True)
    choice = st.selectbox("Inspect a run", [r["id"] for r in runs],
                          format_func=lambda i: f"#{i} — " + next(r["question"] for r in runs if r["id"] == i)[:90])
    show_run(api.run(choice))


def page_reports() -> None:
    st.title("📑 Research reports")
    done = [r for r in api.runs() if r["status"] == "completed"]
    if not done:
        st.info("No completed reports yet.")
        return
    choice = st.selectbox("Report", [r["id"] for r in done],
                          format_func=lambda i: f"#{i} — " + next(r["question"] for r in done if r["id"] == i)[:90])
    run = api.run(choice)
    c1, c2, _ = st.columns([1, 1, 4])
    c1.download_button("⬇️ Markdown", run["markdown"], f"research-report-{choice}.md", "text/markdown")
    c2.download_button("⬇️ PDF", api.report(choice, "pdf"), f"research-report-{choice}.pdf", "application/pdf")
    st.divider()
    st.markdown(run["markdown"])


def page_evaluation() -> None:
    st.title("🧪 Evaluation results")
    evals = api.evaluations()
    if not evals:
        st.info("No evaluations yet. Run research with “Evaluate report” enabled.")
        return
    df = pd.DataFrame([{"run_id": e["run_id"], "question": e["question"][:70], "overall": e["overall"],
                        **e["metrics"]} for e in evals])
    st.subheader("Average by metric")
    metric_cols = [c for c in df.columns if c in METRIC_LABELS]
    st.bar_chart(df[metric_cols].mean().rename(index=METRIC_LABELS))
    st.subheader("All evaluations")
    st.dataframe(df, hide_index=True)
    choice = st.selectbox("Details for run", df["run_id"].tolist())
    render_evaluation(api.run(choice).get("evaluation"), choice)


PAGES = {"Dashboard": page_dashboard, "Chat": page_chat, "Research": page_research, "Documents": page_documents,
         "History": page_history, "Reports": page_reports, "Evaluation": page_evaluation}

with st.sidebar:
    st.markdown("## 🔎 AI Research Agent")
    page = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.divider()
    try:
        health = api.health()
        st.success(f"Backend online · v{health['version']}")
        st.caption(f"Model: `{health['model']}`")
        if not health["api_key_configured"]:
            st.error("ANTHROPIC_API_KEY is missing. Add it to `.env` and restart the backend.")
    except ApiError as exc:
        st.error(str(exc))
        st.code("uvicorn research_agent.api.main:create_app --factory", language="bash")
        st.stop()

try:
    PAGES[page]()
except ApiError as exc:
    st.error(str(exc))
