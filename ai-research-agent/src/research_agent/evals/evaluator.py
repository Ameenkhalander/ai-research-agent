"""Evaluation of generated research reports.

Five metrics, each scored 0-1 (higher is better):

* answer_relevance          - LLM judge (70%) + lexical coverage of the question's key terms (30%)
* factual_consistency       - LLM judge vs. the retrieved evidence (falls back to a sentence-support heuristic)
* citation_validity         - fully programmatic: do [S#] markers resolve, are claims cited, are URLs well-formed
* response_completeness     - LLM judge (70%) + structural heuristic (30%)
* hallucination_indicators  - programmatic indicators (unsupported numbers/URLs/sentences, absolutes) blended
                              with the judge's hallucination-risk rating. 1.0 = no indicators found.

When no LLM judge is available (or it fails) every metric falls back to its heuristic part and the result says so
in the ``method`` field. These are *indicators*, not proofs of correctness.
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from urllib.parse import urlparse

from research_agent.citations import extract_citation_ids, strip_citations

log = logging.getLogger(__name__)

WEIGHTS = {"answer_relevance": 0.20, "factual_consistency": 0.25, "citation_validity": 0.15,
           "response_completeness": 0.20, "hallucination_indicators": 0.20}

_STOP = frozenset(
    "the a an and or of to in on for with is are was were be been by as at from that this these those it its their "
    "his her not no but if then than which who whom what when where how why do does did can could should would may "
    "might will shall also into about over under between within across per via vs have has had having such more most "
    "other some any each both only very just so we you they them our your".split())
_WORD = re.compile(r"[a-z0-9][a-z0-9'\-]*")
_NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?")
_URL = re.compile(r"https?://[^\s)>\]]+")
_ABSOLUTES = re.compile(r"\b(always|never|guaranteed|undoubtedly|certainly|proven|impossible|100%)\b", re.I)


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "ly", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def keywords(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall((text or "").lower()) if w not in _STOP and len(w) > 2}


def claim_sentences(text: str, min_words: int = 8) -> list[str]:
    out = []
    for piece in re.split(r"(?<=[.!?])\s+|\n+", text or ""):
        piece = strip_citations(piece)
        piece = re.sub(r"^[#>*\-\s\d.)]+", "", piece).strip()
        if len(piece.split()) >= min_words:
            out.append(piece)
    return out


def _norm_num(token: str) -> str:
    token = token.replace(",", "").rstrip("%")
    try:
        return f"{float(token):g}"
    except ValueError:
        return token


def _numbers(text: str) -> set[str]:
    return {_norm_num(m) for m in _NUM.findall(strip_citations(text))}


@dataclass
class MetricScore:
    score: float
    method: str
    explanation: str = ""
    details: dict = field(default_factory=dict)
    applicable: bool = True


@dataclass
class EvaluationResult:
    metrics: dict[str, MetricScore]
    overall: float
    flags: list[str]
    judge_model: str | None = None

    def to_dict(self) -> dict:
        return {"overall": self.overall, "flags": self.flags, "judge_model": self.judge_model,
                "metrics": {k: asdict(v) for k, v in self.metrics.items()}}


# ---------------------------------------------------------------------------- heuristics
def lexical_relevance(question: str, answer: str) -> MetricScore:
    q, a = keywords(question), keywords(answer)
    if not q:
        return MetricScore(1.0, "heuristic", "No key terms in the question to compare.")
    covered = q & a
    return MetricScore(len(covered) / len(q), "heuristic",
                       f"{len(covered)}/{len(q)} key question terms appear in the answer.",
                       {"missing_terms": sorted(q - a)})


def citation_validity(answer: str, sources: list[dict]) -> tuple[MetricScore, list[str]]:
    flags: list[str] = []
    if not sources:
        cited = extract_citation_ids(answer)
        if cited:
            flags.append("Answer contains citation markers but no sources were collected.")
            return MetricScore(0.0, "programmatic", "Citations reference sources that do not exist.",
                               {"invalid_ids": sorted(set(cited))}), flags
        return MetricScore(1.0, "programmatic", "No sources were used, so citations are not applicable.",
                           applicable=False), flags
    valid_ids = {s["id"] for s in sources}
    cited = extract_citation_ids(answer)
    invalid = sorted({i for i in cited if i not in valid_ids})
    validity = (len(cited) - sum(1 for i in cited if i not in valid_ids)) / len(cited) if cited else 0.0
    # coverage is computed on raw sentences (citations intact)
    raw = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", answer or "")
           if len(strip_citations(p).split()) >= 8 and not p.lstrip().startswith("#")]
    with_cite = [p for p in raw if extract_citation_ids(p)]
    coverage = len(with_cite) / len(raw) if raw else 1.0
    web = [s for s in sources if s.get("kind") == "web"]
    url_ok = (sum(1 for s in web if urlparse(s.get("url") or "").scheme in ("http", "https")
                  and urlparse(s.get("url") or "").netloc) / len(web)) if web else 1.0
    score = 0.5 * validity + 0.35 * coverage + 0.15 * url_ok
    if invalid:
        flags.append(f"Invalid citation ids: {', '.join(f'S{i}' for i in invalid)}")
    if not cited:
        flags.append("The answer cites no sources even though sources were collected.")
    elif coverage < 0.6:
        flags.append(f"Only {coverage:.0%} of factual sentences carry a citation.")
    return MetricScore(
        round(score, 3), "programmatic",
        f"{len(cited) - sum(1 for i in cited if i not in valid_ids)}/{len(cited)} markers resolve to a source; "
        f"{coverage:.0%} of factual sentences are cited; {url_ok:.0%} of web URLs well-formed.",
        {"markers": len(cited), "invalid_ids": invalid, "citation_coverage": round(coverage, 3),
         "url_wellformed": round(url_ok, 3), "claim_sentences": len(raw)}), flags


def sentence_support(answer: str, evidence: str, threshold: float = 0.4) -> tuple[float, list[str]]:
    """Fraction of factual sentences whose key terms are mostly present in the evidence."""
    ev = keywords(evidence)
    sentences = claim_sentences(answer)
    if not sentences:
        return 1.0, []
    unsupported = []
    for s in sentences:
        k = keywords(s)
        if k and len(k & ev) / len(k) < threshold:
            unsupported.append(s)
    return 1 - len(unsupported) / len(sentences), unsupported


def hallucination_indicators(answer: str, evidence: str, sources: list[dict], question: str = "") -> tuple[MetricScore, list[str]]:
    flags: list[str] = []
    ev_numbers = _numbers(evidence) | _numbers(question)
    answer_numbers = {n for n in _numbers(answer)}
    meaningful = {n for n in answer_numbers if not (n.isdigit() and int(n) < 10)}
    unsupported_nums = sorted(n for n in meaningful if n not in ev_numbers)
    num_frac = len(unsupported_nums) / len(meaningful) if meaningful else 0.0

    known_urls = {(s.get("url") or "").rstrip("/.,") for s in sources if s.get("url")}
    urls = {u.rstrip("/.,;") for u in _URL.findall(answer or "")}
    unsupported_urls = sorted(u for u in urls if u not in known_urls and u not in evidence)

    support, unsupported_sents = sentence_support(answer, evidence)
    sent_frac = 1 - support
    absolutes = [m.group(0) for m in _ABSOLUTES.finditer(answer or "") if m.group(0).lower() not in evidence.lower()]

    risk = min(1.0, 0.4 * num_frac + 0.3 * sent_frac + 0.2 * (1.0 if unsupported_urls else 0.0)
               + 0.1 * min(1.0, len(absolutes) / 3))
    if unsupported_nums:
        flags.append("Numbers not found in the evidence: " + ", ".join(unsupported_nums[:8]))
    if unsupported_urls:
        flags.append("URLs not found in the sources: " + ", ".join(unsupported_urls[:3]))
    if sent_frac > 0.25:
        flags.append(f"{len(unsupported_sents)} sentence(s) have weak lexical support in the evidence.")
    if absolutes:
        flags.append("Absolute wording not present in the evidence: " + ", ".join(sorted(set(a.lower() for a in absolutes))))
    return MetricScore(
        round(1 - risk, 3), "programmatic",
        f"{len(unsupported_nums)} unsupported number(s), {len(unsupported_urls)} unsupported URL(s), "
        f"{len(unsupported_sents)} weakly supported sentence(s), {len(absolutes)} unsupported absolute(s).",
        {"unsupported_numbers": unsupported_nums, "unsupported_urls": unsupported_urls,
         "weakly_supported_sentences": unsupported_sents[:5], "absolutes": absolutes}), flags


def completeness_heuristic(question: str, answer: str, report: dict | None, style: str = "detailed") -> MetricScore:
    min_words = 60 if style == "concise" else 150
    words = len(answer.split())
    score, parts = 0.0, {}
    q, a = keywords(question), keywords(answer)
    parts["question_coverage"] = len(q & a) / len(q) if q else 1.0
    score += 0.35 * parts["question_coverage"]
    parts["length"] = min(1.0, words / min_words)
    score += 0.15 * parts["length"]
    if report:
        parts["has_summary"] = float(len((report.get("executive_summary") or "").split()) >= 15)
        parts["has_findings"] = min(1.0, len(report.get("key_findings") or []) / (1 if style == "concise" else 3))
        parts["has_conclusion"] = float(bool((report.get("conclusion") or "").strip()))
        parts["has_limitations"] = float(bool(report.get("limitations")))
        score += 0.15 * parts["has_summary"] + 0.2 * parts["has_findings"] + 0.1 * parts["has_conclusion"] \
            + 0.05 * parts["has_limitations"]
    else:
        score = score / 0.5  # only two components available
    return MetricScore(round(min(1.0, score), 3), "heuristic",
                       f"{words} words; question-term coverage {parts['question_coverage']:.0%}.",
                       {k: round(v, 3) for k, v in parts.items()})


# ---------------------------------------------------------------------------- LLM judge
def _score_obj(desc: str) -> dict:
    return {"type": "object", "description": desc, "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 5}, "rationale": {"type": "string"}},
        "required": ["score", "rationale"]}


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer_relevance": _score_obj("5 = directly and fully addresses the question; 1 = off-topic"),
        "factual_consistency": _score_obj("5 = every claim is supported by the evidence; 1 = many claims contradict "
                                          "or are absent from the evidence"),
        "response_completeness": _score_obj("5 = covers all aspects of the question; 1 = major aspects missing"),
        "hallucination_risk": _score_obj("1 = no invented content; 5 = severe invented facts, numbers or sources"),
        "unsupported_claims": {"type": "array", "items": {"type": "string"},
                               "description": "Claims in the answer not supported by the evidence (verbatim, short)"},
    },
    "required": ["answer_relevance", "factual_consistency", "response_completeness", "hallucination_risk",
                 "unsupported_claims"],
}

JUDGE_SYSTEM = """You are a strict, impartial evaluator of AI-generated research reports.
You receive a question, the report, and the EVIDENCE the agent actually retrieved. Judge the report ONLY against
the evidence: a claim is supported only if the evidence states or clearly implies it, regardless of whether it
happens to be true in general. Calculator outputs in the evidence count as supported. Do not reward length or
style. Be critical: use 5 only when it is clearly deserved. Citation labels like [S1] are not part of the content."""


def _norm5(score: int, invert: bool = False) -> float:
    s = max(1, min(5, int(score)))
    return (5 - s) / 4 if invert else (s - 1) / 4


class Evaluator:
    def __init__(self, llm=None, weights: dict[str, float] | None = None):
        self.llm = llm
        self.weights = weights or WEIGHTS

    async def _judge(self, question: str, answer: str, evidence: str) -> dict | None:
        if self.llm is None:
            return None
        prompt = (f"<question>\n{question}\n</question>\n\n<report>\n{answer[:12000]}\n</report>\n\n"
                  f"<evidence>\n{evidence[:30000] or '(no evidence was retrieved)'}\n</evidence>")
        try:
            return await self.llm.structured(
                system=JUDGE_SYSTEM, prompt=prompt, tool_name="submit_evaluation",
                description="Submit the evaluation scores", schema=JUDGE_SCHEMA, max_tokens=2000)
        except Exception as exc:
            log.warning("LLM judge failed, using heuristics only: %s", exc)
            self._judge_error = f"{type(exc).__name__}: {exc}"
            return None

    async def evaluate(self, *, question: str, answer: str, sources: list[dict], evidence: str,
                       report: dict | None = None, style: str = "detailed") -> EvaluationResult:
        self._judge_error = None
        flags: list[str] = []
        judge = await self._judge(question, answer, evidence)
        if self.llm is not None and judge is None:
            flags.append(f"LLM judge unavailable ({self._judge_error}); heuristic scores only.")

        # relevance
        lex = lexical_relevance(question, answer)
        if judge:
            j = judge["answer_relevance"]
            relevance = MetricScore(round(0.7 * _norm5(j["score"]) + 0.3 * lex.score, 3), "llm_judge+heuristic",
                                    j.get("rationale", ""), {"judge_score": j["score"], "lexical": round(lex.score, 3)})
        else:
            relevance = lex

        # factual consistency
        support, unsupported = sentence_support(answer, evidence)
        if judge:
            j = judge["factual_consistency"]
            factual = MetricScore(round(_norm5(j["score"]), 3), "llm_judge", j.get("rationale", ""),
                                  {"judge_score": j["score"], "lexical_support": round(support, 3)})
        else:
            factual = MetricScore(round(support, 3), "heuristic",
                                  f"{len(unsupported)} factual sentence(s) have weak lexical support in the evidence.",
                                  {"weakly_supported": unsupported[:5]})

        citation, cflags = citation_validity(answer, sources)
        flags += cflags

        comp_h = completeness_heuristic(question, answer, report, style)
        if judge:
            j = judge["response_completeness"]
            completeness = MetricScore(round(0.7 * _norm5(j["score"]) + 0.3 * comp_h.score, 3),
                                       "llm_judge+heuristic", j.get("rationale", ""),
                                       {"judge_score": j["score"], "heuristic": comp_h.score})
        else:
            completeness = comp_h

        hall_h, hflags = hallucination_indicators(answer, evidence, sources, question)
        flags += hflags
        if judge:
            j = judge["hallucination_risk"]
            hall = MetricScore(round(0.5 * _norm5(j["score"], invert=True) + 0.5 * hall_h.score, 3),
                               "llm_judge+programmatic", j.get("rationale", ""),
                               {**hall_h.details, "judge_risk_score": j["score"], "programmatic": hall_h.score})
            flags += [f"Judge: {c}" for c in (judge.get("unsupported_claims") or [])[:5]]
        else:
            hall = hall_h

        metrics = {"answer_relevance": relevance, "factual_consistency": factual, "citation_validity": citation,
                   "response_completeness": completeness, "hallucination_indicators": hall}
        applicable = {k: v for k, v in metrics.items() if v.applicable}
        total_w = sum(self.weights[k] for k in applicable)
        overall = sum(self.weights[k] * v.score for k, v in applicable.items()) / total_w if total_w else 0.0
        return EvaluationResult(metrics, round(overall, 3), flags,
                                getattr(self.llm, "model", None) if judge else None)
