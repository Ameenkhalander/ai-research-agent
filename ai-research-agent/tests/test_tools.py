import pytest

from research_agent.agent.sources import SourceRegistry
from research_agent.tools.calculator import calculate, safe_eval
from research_agent.tools.web import assert_public_url, html_to_text
from helpers import web_search_json


def test_calculator_basics():
    assert safe_eval("2 + 3 * 4") == 14
    assert safe_eval("(2 + 3) * 4") == 20
    assert safe_eval("2 ^ 10") == 1024
    assert safe_eval("sqrt(16) + abs(-3)") == 7
    assert safe_eval("mean([1, 2, 3, 4])") == 2.5
    assert calculate("12000 * 3 / 1000000 + 2500 * 15 / 1000000")["result"] == "0.0735"
    assert calculate("10 / 4")["result"] == "2.5"


def test_calculator_rejects_unsafe_input():
    for bad in ["__import__('os').system('echo hi')", "open('/etc/passwd')", "().__class__", "x + 1",
                "lambda: 1", "2 ** 100000", "9 ** 9 ** 9", "[1, 2]", "'a' * 3", "", "1 +"]:
        with pytest.raises(ValueError):
            safe_eval(bad)


def test_calculator_division_by_zero():
    with pytest.raises(ValueError):
        safe_eval("1 / 0")


def test_assert_public_url_blocks_local_targets():
    for url in ["http://127.0.0.1/", "http://169.254.169.254/latest/meta-data", "http://10.0.0.5/",
                "file:///etc/passwd", "ftp://example.com/x", "http://[::1]/"]:
        with pytest.raises(ValueError):
            assert_public_url(url)


def test_html_to_text_strips_scripts_and_keeps_title():
    html = ("<html><head><title>My Page</title><style>p{}</style></head><body><script>var x=1;</script>"
            "<h1>Heading</h1><p>Hello <b>world</b>.</p><ul><li>one</li><li>two</li></ul></body></html>")
    title, text = html_to_text(html)
    assert title == "My Page"
    assert "var x" not in text and "p{}" not in text
    assert "Heading" in text and "Hello world." in text and "one" in text


def test_source_registry_labels_and_dedupes():
    reg = SourceRegistry()
    out = reg.ingest_tool_output("web_search", web_search_json(
        ("A", "https://a.example/x", "alpha snippet"), ("B", "https://b.example/y", "beta snippet")))
    assert "[S1] A (https://a.example/x)" in out and "[S2] B" in out
    out2 = reg.ingest_tool_output("web_search", web_search_json(("A again", "https://a.example/x", "more alpha")))
    assert "[S1]" in out2 and len(reg) == 2
    assert "more alpha" in reg.get(1).content and "alpha snippet" in reg.get(1).content


def test_source_registry_documents_and_passthrough():
    reg = SourceRegistry()
    raw = '{"query": "q", "results": [{"document": "n.txt", "chunk_index": 3, "page": 2, "score": 0.9, "text": "chunk"}]}'
    out = reg.ingest_tool_output("search_documents", raw)
    assert "[S1] n.txt, p.2" in out
    src = reg.as_list()[0]
    assert src["kind"] == "document" and src["url"] is None and src["locator"] == "n.txt#chunk3"
    assert reg.ingest_tool_output("search_documents", '{"results": []}').startswith("No matching passages")
    assert reg.ingest_tool_output("summarize_text", "plain text") == "plain text"
    assert reg.ingest_tool_output("calculator", '{"expression": "1+1", "result": "2"}') == "Calculator: 1+1 = 2"
