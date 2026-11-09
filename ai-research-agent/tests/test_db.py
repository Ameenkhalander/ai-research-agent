from research_agent.db import Database


def test_document_crud(tmp_path):
    db = Database(tmp_path / "t.db")
    doc_id = db.add_document(filename="a.txt", file_type="txt", sha256="abc", size_bytes=10, n_chunks=2)
    assert db.get_document_by_hash("abc")["id"] == doc_id
    assert [d["filename"] for d in db.list_documents()] == ["a.txt"]
    assert db.delete_document(doc_id) is True and db.delete_document(doc_id) is False


def test_run_lifecycle_and_evaluation(tmp_path):
    db = Database(tmp_path / "t.db")
    run_id = db.create_run("What is MCP?", "detailed", evaluate=True)
    assert db.get_run(run_id)["status"] == "running" and db.get_run(run_id)["eval_status"] == "pending"
    db.update_run_progress(run_id, steps=[{"n": 1}], plan=[{"id": 1}])
    assert db.get_run(run_id)["steps"] == [{"n": 1}]
    db.complete_run(run_id, plan=[{"id": 1}], steps=[{"n": 1}, {"n": 2}], report={"title": "T"}, markdown="# T",
                    sources=[{"id": 1}], evidence="ev", duration=1.5)
    run = db.get_run(run_id)
    assert run["status"] == "completed" and run["report"] == {"title": "T"} and run["duration"] == 1.5
    db.save_evaluation(run_id, {"overall": 0.8, "metrics": {"answer_relevance": {"score": 0.9}}, "judge_model": "m"})
    assert db.get_run(run_id)["eval_status"] == "done"
    assert db.get_evaluation(run_id)["overall"] == 0.8
    listed = db.list_evaluations()
    assert listed[0]["metrics"] == {"answer_relevance": 0.9} and listed[0]["question"] == "What is MCP?"
    assert db.list_runs()[0]["overall_score"] == 0.8
    stats = db.stats()
    assert stats["runs"] == 1 and stats["completed"] == 1 and stats["avg_overall_score"] == 0.8


def test_failed_and_interrupted_runs(tmp_path):
    db = Database(tmp_path / "t.db")
    a, b = db.create_run("q1 question"), db.create_run("q2 question")
    db.fail_run(a, "boom")
    assert db.get_run(a)["status"] == "failed" and db.get_run(a)["eval_status"] == "skipped"
    assert db.mark_interrupted() == 1
    assert db.get_run(b)["error"] == "Interrupted by server restart"


def test_messages_return_latest_in_order(tmp_path):
    db = Database(tmp_path / "t.db")
    for i in range(5):
        db.add_message("s1", "user" if i % 2 == 0 else "assistant", f"m{i}")
    db.add_message("s2", "user", "other")
    assert [m["content"] for m in db.get_messages("s1", limit=3)] == ["m2", "m3", "m4"]
