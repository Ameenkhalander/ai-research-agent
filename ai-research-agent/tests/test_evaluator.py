import asyncio

from helpers import FakeLLM
from research_agent.evals.evaluator import (Evaluator, citation_validity, claim_sentences, hallucination_indicators,
                                            lexical_relevance, sentence_support)

SOURCES = [
    {"id": 1, "kind": "web", "title": "MCP spec", "locator": "https://spec.example/mcp", "url": "https://spec.example/mcp",
     "content": "The Model Context Protocol uses JSON-RPC 2.0 messages and supports stdio and HTTP transports."},
    {"id": 2, "kind": "document", "title": "notes.txt", "locator": "notes.txt#chunk0", "url": None,
     "content": "Servers expose tools, resources and prompts to clients."},
]
EVIDENCE = " ".join(s["content"] for s in SOURCES)
GOOD = ("The Model Context Protocol uses JSON-RPC 2.0 messages between clients and servers [S1]. "
        "Servers expose tools, resources and prompts to clients [S2]. It supports stdio and HTTP transports [S1].")
BAD = ("The Model Context Protocol was released in 2019 and handles 45,000 requests per second [S7]. "
       "It always guarantees perfect security, as described at https://made-up.example/spec. "
       "Quantum encryption keys rotate automatically every night across all deployments worldwide.")


def test_citation_validity_good_vs_bad():
    good, gflags = citation_validity(GOOD, SOURCES)
    bad, bflags = citation_validity(BAD, SOURCES)
    assert good.score > 0.95 and not gflags
    assert bad.score < 0.5 and any("S7" in f for f in bflags)
    assert bad.details["invalid_ids"] == [7]


def test_citation_not_applicable_without_sources():
    m, _ = citation_validity("2 + 2 is 4, which was computed with the calculator tool today.", [])
    assert m.applicable is False
    m2, flags = citation_validity("Claim [S1] without any sources existing.", [])
    assert m2.score == 0.0 and flags


def test_hallucination_indicators_detect_invented_content():
    ok, _ = hallucination_indicators(GOOD, EVIDENCE, SOURCES, "What is MCP?")
    bad, flags = hallucination_indicators(BAD, EVIDENCE, SOURCES, "What is MCP?")
    assert ok.score > 0.85
    assert bad.score < 0.5
    joined = " ".join(flags)
    assert "2019" in joined or "45000" in joined
    assert "made-up.example" in joined


def test_numbers_present_in_evidence_are_not_flagged():
    m, _ = hallucination_indicators("The protocol uses JSON-RPC 2.0 [S1].", EVIDENCE, SOURCES)
    assert m.details["unsupported_numbers"] == []


def test_relevance_and_support_helpers():
    assert lexical_relevance("What transports does MCP support?", GOOD).score > 0.5
    assert lexical_relevance("Explain photosynthesis in chloroplasts", GOOD).score < 0.2
    support, unsupported = sentence_support(BAD, EVIDENCE)
    assert support < 1 and unsupported
    assert claim_sentences("Short one. This sentence has more than eight words in it for sure.") == [
        "This sentence has more than eight words in it for sure."]


def test_evaluator_heuristic_only_ranks_good_above_bad():
    ev = Evaluator(llm=None)
    report = {"executive_summary": GOOD, "key_findings": [{"heading": "h", "content": GOOD}] * 3,
              "conclusion": "Done.", "limitations": ["x"]}
    good = asyncio.run(ev.evaluate(question="What transports does MCP support?", answer=GOOD, sources=SOURCES,
                                   evidence=EVIDENCE, report=report))
    bad = asyncio.run(ev.evaluate(question="What transports does MCP support?", answer=BAD, sources=SOURCES,
                                  evidence=EVIDENCE))
    assert good.overall > bad.overall + 0.2
    assert set(good.metrics) == {"answer_relevance", "factual_consistency", "citation_validity",
                                 "response_completeness", "hallucination_indicators"}
    assert good.judge_model is None and 0 <= good.overall <= 1


def test_evaluator_blends_llm_judge_scores():
    judge = {"answer_relevance": {"score": 5, "rationale": "on topic"},
             "factual_consistency": {"score": 4, "rationale": "mostly supported"},
             "response_completeness": {"score": 3, "rationale": "partial"},
             "hallucination_risk": {"score": 1, "rationale": "none"},
             "unsupported_claims": ["Claim X"]}
    llm = FakeLLM(structured=[judge])
    res = asyncio.run(Evaluator(llm).evaluate(question="What transports does MCP support?", answer=GOOD,
                                              sources=SOURCES, evidence=EVIDENCE))
    assert res.metrics["factual_consistency"].score == 0.75 and res.metrics["factual_consistency"].method == "llm_judge"
    assert res.metrics["answer_relevance"].method == "llm_judge+heuristic"
    assert any(f == "Judge: Claim X" for f in res.flags)
    assert res.judge_model == "fake-model"
    prompt = llm.calls[0][2]["prompt"]
    assert "<evidence>" in prompt and "JSON-RPC" in prompt


def test_evaluator_falls_back_when_judge_fails():
    class Broken(FakeLLM):
        async def structured(self, **kw):
            raise RuntimeError("API down")

    res = asyncio.run(Evaluator(Broken()).evaluate(question="What is MCP?", answer=GOOD, sources=SOURCES,
                                                   evidence=EVIDENCE))
    assert any("judge unavailable" in f.lower() for f in res.flags)
    assert res.judge_model is None and res.metrics["factual_consistency"].method == "heuristic"
