import asyncio

from helpers import FakeLLM, FakeTools, text_response, tool_response, web_search_json
from research_agent.agent.agent import ResearchAgent
from research_agent.config import Settings


def _plan(tool="web_search"):
    return {"objective": "obj", "tasks": [
        {"id": 1, "description": "Find facts", "suggested_tool": tool},
        {"id": 2, "description": "Compute total", "suggested_tool": "calculator"}]}


def _report():
    return {"title": "Report", "executive_summary": "MCP standardises tool access [S1].",
            "key_findings": [{"heading": "Facts", "content": "It uses JSON-RPC [S1][S2]."}],
            "conclusion": "Total cost is 6.", "limitations": ["Few sources"]}


def test_agent_end_to_end_with_scripted_llm_and_tools(tmp_path):
    tools = FakeTools({
        "web_search": lambda a: web_search_json(("MCP spec", "https://spec.example/mcp", "MCP uses JSON-RPC 2.0."),
                                                ("Blog", "https://blog.example/mcp", "Servers expose tools.")),
        "calculator": lambda a: '{"expression": "2*3", "result": "6"}',
    })
    llm = FakeLLM(
        structured=[_plan(), _report()],
        completions=[tool_response("web_search", {"query": "mcp"}, "t1"),
                     text_response("- MCP uses JSON-RPC [S1]\n- Servers expose tools [S2]"),
                     tool_response("calculator", {"expression": "2*3"}, "t2"),
                     text_response("- 2*3 = 6")])
    seen = []
    agent = ResearchAgent(llm, tools, Settings(data_dir=tmp_path), on_step=lambda steps, plan: seen.append(len(steps)))
    result = asyncio.run(agent.run("What is MCP and what is 2*3?", available_documents=["notes.txt"]))

    assert [c[0] for c in tools.calls] == ["web_search", "calculator"]
    assert [s["id"] for s in result.sources] == [1, 2] and result.sources[0]["url"] == "https://spec.example/mcp"
    kinds = [s["kind"] for s in result.steps]
    assert kinds[:2] == ["mcp", "plan"] and kinds.count("tool_call") == 2 and kinds[-1] == "done"
    assert "## References" in result.markdown and "https://spec.example/mcp" in result.markdown
    assert "MCP uses JSON-RPC 2.0." in result.evidence and "Calculator: 2*3 = 6" in result.evidence
    assert seen == sorted(seen) and seen[-1] == len(result.steps)  # progress was streamed incrementally
    plan_prompt = next(c for c in llm.calls if c[1] == "submit_plan")[2]["prompt"]
    assert "notes.txt" in plan_prompt
    # the model must have seen labelled observations in its second call
    second = [c for c in llm.calls if c[0] == "complete"][1][2]["messages"]
    assert "[S1] MCP spec" in second[-1]["content"][0]["content"]


def test_agent_forces_final_answer_after_max_rounds(tmp_path):
    tools = FakeTools({"web_search": lambda a: web_search_json(("T", "https://t.example", "snippet"))})

    def completion(kw):
        if kw.get("tool_choice") == {"type": "none"}:
            return text_response("Final findings [S1]")
        return tool_response("web_search", {"query": "again"})

    llm = FakeLLM(structured=[{"objective": "o", "tasks": [{"id": 1, "description": "d", "suggested_tool": "web_search"}]},
                              _report()], completions=[completion] * 3)
    agent = ResearchAgent(llm, tools, Settings(data_dir=tmp_path, max_tool_rounds=3))
    result = asyncio.run(agent.run("Some question?"))
    assert len(tools.calls) == 2  # rounds 1 and 2 call the tool, round 3 is forced to answer
    assert any(s["kind"] == "task_done" and "Final findings" in s["output"] for s in result.steps)


def test_agent_survives_tool_errors_and_bad_plan(tmp_path):
    def broken(args):
        raise RuntimeError("network down")

    tools = FakeTools({"web_search": broken})
    llm = FakeLLM(structured=[{"objective": "o", "tasks": []}, _report()],
                  completions=[tool_response("web_search", {"query": "x"}), text_response("Nothing found.")])
    result = asyncio.run(ResearchAgent(llm, tools, Settings(data_dir=tmp_path)).run("Question here?"))
    assert result.plan[0]["suggested_tool"] == "web_search"  # fallback plan
    err = [s for s in result.steps if s["kind"] == "tool_result"][0]
    assert err["error"] is True and "network down" in err["output"]
    tool_result_msg = [c for c in llm.calls if c[0] == "complete"][1][2]["messages"][-1]["content"][0]
    assert tool_result_msg["is_error"] is True
    assert result.sources == []
