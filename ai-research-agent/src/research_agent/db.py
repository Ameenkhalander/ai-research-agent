"""SQLite persistence: documents, research runs (with agent trace), evaluations and chat messages."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    size_bytes INTEGER NOT NULL,
    n_pages INTEGER NOT NULL DEFAULT 0,
    n_chunks INTEGER NOT NULL DEFAULT 0,
    n_chars INTEGER NOT NULL DEFAULT 0,
    uploaded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    style TEXT NOT NULL DEFAULT 'detailed',
    status TEXT NOT NULL DEFAULT 'running',
    eval_status TEXT NOT NULL DEFAULT 'skipped',
    plan TEXT,
    steps TEXT,
    report TEXT,
    markdown TEXT,
    sources TEXT,
    evidence TEXT,
    error TEXT,
    duration REAL,
    session_id TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL UNIQUE REFERENCES runs(id) ON DELETE CASCADE,
    scores TEXT NOT NULL,
    overall REAL NOT NULL,
    judge_model TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    run_id INTEGER,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""

_RUN_JSON_FIELDS = ("plan", "steps", "report", "sources")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dumps(value: Any) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False)


def _loads(value: str | None) -> Any:
    return None if value is None else json.loads(value)


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ documents
    def add_document(self, *, filename: str, file_type: str, sha256: str, size_bytes: int,
                     n_pages: int = 0, n_chunks: int = 0, n_chars: int = 0) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO documents (filename, file_type, sha256, size_bytes, n_pages, n_chunks, n_chars, uploaded_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (filename, file_type, sha256, size_bytes, n_pages, n_chunks, n_chars, _now()),
            )
            return int(cur.lastrowid)

    def update_document_chunks(self, doc_id: int, n_chunks: int) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE documents SET n_chunks=? WHERE id=?", (n_chunks, doc_id))

    def get_document(self, doc_id: int) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
            return dict(row) if row else None

    def get_document_by_hash(self, sha256: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM documents WHERE sha256=?", (sha256,)).fetchone()
            return dict(row) if row else None

    def list_documents(self) -> list[dict]:
        with self._conn() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM documents ORDER BY id DESC")]

    def delete_document(self, doc_id: int) -> bool:
        with self._conn() as conn:
            return conn.execute("DELETE FROM documents WHERE id=?", (doc_id,)).rowcount > 0

    # ------------------------------------------------------------------ runs
    def create_run(self, question: str, style: str = "detailed", evaluate: bool = True,
                   session_id: str | None = None) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO runs (question, style, status, eval_status, session_id, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (question, style, "running", "pending" if evaluate else "skipped", session_id, _now()),
            )
            return int(cur.lastrowid)

    def update_run_progress(self, run_id: int, *, steps: list | None = None, plan: list | None = None) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE runs SET steps=?, plan=? WHERE id=?", (_dumps(steps), _dumps(plan), run_id))

    def complete_run(self, run_id: int, *, plan: list, steps: list, report: dict, markdown: str,
                     sources: list, evidence: str, duration: float) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET status='completed', plan=?, steps=?, report=?, markdown=?, sources=?,"
                " evidence=?, duration=?, completed_at=? WHERE id=?",
                (_dumps(plan), _dumps(steps), _dumps(report), markdown, _dumps(sources), evidence,
                 duration, _now(), run_id),
            )

    def fail_run(self, run_id: int, error: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET status='failed', error=?, completed_at=?,"
                " eval_status=CASE WHEN eval_status='pending' THEN 'skipped' ELSE eval_status END WHERE id=?",
                (error[:2000], _now(), run_id),
            )

    def set_eval_status(self, run_id: int, status: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE runs SET eval_status=? WHERE id=?", (status, run_id))

    def mark_interrupted(self) -> int:
        """Runs left in 'running' by a previous server process can never finish."""
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE runs SET status='failed', error='Interrupted by server restart', completed_at=?"
                " WHERE status='running'", (_now(),))
            conn.execute("UPDATE runs SET eval_status='skipped' WHERE eval_status='pending' AND status='failed'")
            return cur.rowcount

    def get_run(self, run_id: int) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            return None
        run = dict(row)
        for key in _RUN_JSON_FIELDS:
            run[key] = _loads(run[key])
        return run

    def list_runs(self, limit: int = 100) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT r.id, r.question, r.style, r.status, r.eval_status, r.duration, r.error,"
                " r.created_at, r.completed_at, e.overall AS overall_score"
                " FROM runs r LEFT JOIN evaluations e ON e.run_id = r.id ORDER BY r.id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ evaluations
    def save_evaluation(self, run_id: int, result: dict) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM evaluations WHERE run_id=?", (run_id,))
            conn.execute(
                "INSERT INTO evaluations (run_id, scores, overall, judge_model, created_at) VALUES (?,?,?,?,?)",
                (run_id, _dumps(result), float(result["overall"]), result.get("judge_model"), _now()),
            )
            conn.execute("UPDATE runs SET eval_status='done' WHERE id=?", (run_id,))

    def get_evaluation(self, run_id: int) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM evaluations WHERE run_id=?", (run_id,)).fetchone()
        if not row:
            return None
        data = _loads(row["scores"])
        data["created_at"] = row["created_at"]
        return data

    def list_evaluations(self, limit: int = 200) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT e.run_id, e.scores, e.overall, e.judge_model, e.created_at, r.question"
                " FROM evaluations e JOIN runs r ON r.id = e.run_id ORDER BY e.id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out = []
        for row in rows:
            data = _loads(row["scores"])
            out.append({
                "run_id": row["run_id"], "question": row["question"], "overall": row["overall"],
                "judge_model": row["judge_model"], "created_at": row["created_at"],
                "metrics": {k: v["score"] for k, v in data.get("metrics", {}).items()},
            })
        return out

    # ------------------------------------------------------------------ chat
    def add_message(self, session_id: str, role: str, content: str, run_id: int | None = None) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO messages (session_id, role, content, run_id, created_at) VALUES (?,?,?,?,?)",
                (session_id, role, content, run_id, _now()),
            )
            return int(cur.lastrowid)

    def get_messages(self, session_id: str, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM (SELECT * FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?) ORDER BY id",
                (session_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict:
        with self._conn() as conn:
            docs = conn.execute("SELECT COUNT(*) c, COALESCE(SUM(n_chunks),0) ch FROM documents").fetchone()
            runs = conn.execute(
                "SELECT COUNT(*) total,"
                " SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) completed,"
                " SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) failed,"
                " AVG(CASE WHEN status='completed' THEN duration END) avg_duration FROM runs").fetchone()
            ev = conn.execute("SELECT AVG(overall) avg_overall, COUNT(*) n FROM evaluations").fetchone()
        return {
            "documents": docs["c"], "chunks": docs["ch"],
            "runs": runs["total"] or 0, "completed": runs["completed"] or 0, "failed": runs["failed"] or 0,
            "avg_duration_s": round(runs["avg_duration"], 1) if runs["avg_duration"] else None,
            "evaluations": ev["n"], "avg_overall_score": round(ev["avg_overall"], 3) if ev["avg_overall"] else None,
        }
